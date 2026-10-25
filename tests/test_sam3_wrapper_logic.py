"""CPU tests for the logic of Sam3Segmenter against a stand-in `sam3` (no GPU, no weights).

The stand-in reproduces the real API's shapes, including the quirk that `predict_inst` squeezes dim 0
(one prompt -> [C, ...], N boxes -> [N, C, ...]). It does NOT test SAM 3 itself: see tests/gpu/.
"""

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from promptcorrect.data.geometry import rle_decode
from promptcorrect.types import Prompt

H, W = 120, 160


class FakeTransforms:
    def postprocess_masks(self, masks, orig_hw):
        return F.interpolate(masks.float(), orig_hw, mode="bilinear", align_corners=False)


class FakeModel:
    def __init__(self):
        self.inst_interactive_predictor = types.SimpleNamespace(_transforms=FakeTransforms())
        self.calls = []

    def predict_inst(self, state, point_coords=None, point_labels=None, box=None, mask_input=None,
                     multimask_output=True, return_logits=False):
        self.calls.append(dict(points=point_coords, labels=point_labels, box=box, mask_input=mask_input,
                               multimask=multimask_output))
        n = 1 if box is None else len(box)
        C = 3 if multimask_output else 1
        low = np.full((n, C, 288, 288), -5.0, np.float32)
        for i in range(n):
            if box is not None:
                x0, y0, x1, y1 = box[i]
            else:
                px, py = point_coords[0]
                x0, y0, x1, y1 = px - 10, py - 10, px + 10, py + 10
            low[i, :, int(y0 / H * 288):int(y1 / H * 288), int(x0 / W * 288):int(x1 / W * 288)] = 5.0
        scores = np.tile(np.linspace(0.5, 0.9, C), (n, 1)).astype(np.float32)
        masks = np.zeros((n, C, H, W), np.float32)
        if n == 1:  # the real predictor squeezes dim 0
            masks, scores, low = masks[0], scores[0], low[0]
        return masks, scores, low

    def forward_grounding(self, **kw):
        return {"presence_logit_dec": torch.tensor([2.0])}


class FakeProcessor:
    def __init__(self, model, device="cpu", confidence_threshold=0.5):
        self.model, self.confidence_threshold, self.find_stage = model, confidence_threshold, object()

    def set_image(self, image):
        return {"original_height": image.height, "original_width": image.width,
                "backbone_out": {"sam2_backbone_out": {}}}

    def set_text_prompt(self, prompt, state):
        state["backbone_out"]["language_features"] = "leak?"  # must not leak into the image state
        state["geometric_prompt"] = object()
        boxes = torch.tensor([[-3.0, 5.0, 60.0, 70.0], [10.0, 10.0, W + 9.0, H + 4.0], [20.0, 20.0, 40.0, 40.0]])
        scores = torch.tensor([0.6, 0.9, 0.7])
        masks = torch.zeros(3, 1, H, W, dtype=torch.bool)
        masks[:, :, 10:50, 10:60] = True
        keep = scores > self.confidence_threshold
        state.update(boxes=boxes[keep], scores=scores[keep], masks=masks[keep])
        return state


def _install(monkeypatch, model):
    mods = {n: types.ModuleType(n) for n in
            ["sam3", "sam3.model", "sam3.model.sam3_image_processor", "sam3.model_builder"]}
    mods["sam3.model.sam3_image_processor"].Sam3Processor = FakeProcessor
    mods["sam3.model_builder"].build_sam3_image_model = lambda **kw: model
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)


def _make(monkeypatch, **kw):
    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    model = FakeModel()
    _install(monkeypatch, model)
    s = Sam3Segmenter(ckpt_path="x.pt", device="cpu", image_dirs=[], **kw)
    s.set_image_array(np.zeros((H, W, 3), np.uint8))
    s.fake = model
    return s


@pytest.fixture
def seg(monkeypatch):
    return _make(monkeypatch)


def _inside(logits, cx, cy):  # logit at an image point (x, y) of a [288, 288] map
    return float(logits[int(cy / H * 288), int(cx / W * 288)])


def test_boxes_are_batched_in_one_call_and_independent(seg):
    outs = seg.predict([Prompt(box=[10, 10, 50, 50]), Prompt(box=[80, 20, 140, 90]), Prompt(box=[0, 0, 30, 30])])
    assert len(seg.fake.calls) == 1 and seg.fake.calls[0]["box"].shape == (3, 4)
    assert all(tuple(o.mask_logits.shape) == (288, 288) and o.presence_score is None for o in outs)
    assert _inside(outs[0].mask_logits, 30, 30) > 0 > _inside(outs[0].mask_logits, 110, 50)
    assert _inside(outs[1].mask_logits, 110, 50) > 0 > _inside(outs[1].mask_logits, 30, 30)  # rows not mixed up


def test_single_box_survives_the_squeeze(seg):
    out = seg.predict([Prompt(box=[10, 10, 50, 50])])[0]
    assert tuple(out.mask_logits.shape) == (288, 288) and isinstance(out.iou_score, float)


def test_mixed_prompts_keep_order_and_call_count(seg):
    ps = [Prompt(box=[10, 10, 50, 50]), Prompt(box=[80, 20, 140, 90], points=[(100.0, 40.0, 1)]),
          Prompt(box=[0, 0, 30, 30]), Prompt(points=[(40.0, 40.0, 1), (50.0, 50.0, 0)])]
    outs = seg.predict(ps)
    assert len(seg.fake.calls) == 3  # one batched call for the two box-only prompts + one each for the others
    assert _inside(outs[0].mask_logits, 30, 30) > 0 and _inside(outs[2].mask_logits, 15, 15) > 0
    assert _inside(outs[1].mask_logits, 110, 50) > 0 and _inside(outs[3].mask_logits, 40, 40) > 0
    pts_call = seg.fake.calls[-1]
    assert pts_call["box"] is None and pts_call["points"].shape == (2, 2)
    assert pts_call["labels"].dtype == np.int32 and pts_call["labels"].tolist() == [1, 0]


def test_chunking_for_many_boxes(monkeypatch):
    s = _make(monkeypatch, box_batch_size=4)
    outs = s.predict([Prompt(box=[10, 10, 50, 50])] * 10)
    assert len(outs) == 10 and len(s.fake.calls) == 3  # 4 + 4 + 2


def test_multimask_fields(seg):
    o = seg.predict([Prompt(box=[10, 10, 50, 50])], multimask=True)[0]
    assert tuple(o.alt_logits.shape) == (3, 288, 288) and tuple(o.alt_ious.shape) == (3,)
    assert o.iou_score == pytest.approx(float(o.alt_ious.max()))


def test_mask_input_is_passed_and_validated(seg):
    seg.predict([Prompt(box=[10, 10, 50, 50], mask_logits=torch.zeros(288, 288))])
    assert seg.fake.calls[-1]["mask_input"].shape == (1, 288, 288)
    with pytest.raises(ValueError, match="288"):
        seg.predict([Prompt(box=[10, 10, 50, 50], mask_logits=torch.zeros(256, 256))])


def test_invalid_prompt_and_missing_image(seg):
    with pytest.raises(ValueError):
        seg.predict([Prompt()])
    seg._state = None
    with pytest.raises(RuntimeError, match="set_image"):
        seg.predict([Prompt(box=[10, 10, 50, 50])])


def test_to_full_res(seg):
    out = seg.predict([Prompt(box=[10, 10, 50, 50])])[0]
    full = seg.to_full_res(out.mask_logits)
    assert tuple(full.shape) == (H, W)
    mask = (full > 0).numpy()
    assert mask[30, 30] and not mask[100, 140]
    assert seg.to_full_res(full) is full  # already full size: identity
    assert seg.full_res_mask(out).shape == (H, W)


def test_detect_text_sorted_clipped_and_not_leaking(seg):
    dets = seg.detect_text("person")
    assert [round(d[2], 2) for d in dets] == [0.9, 0.7, 0.6]
    for (x0, y0, x1, y1), rle, _ in dets:
        assert 0 <= x0 <= x1 <= W and 0 <= y0 <= y1 <= H
        assert rle_decode(rle).shape == (H, W)
    assert dets[0][0][2] == W and dets[0][0][3] == H and dets[2][0][0] == 0.0  # clipped from W+9, H+4 and -3
    assert seg.last_presence == pytest.approx(float(torch.tensor(2.0).sigmoid()))
    assert "language_features" not in seg._state["backbone_out"]
    assert seg.predict([Prompt(box=[10, 10, 50, 50])])  # image state still usable


def test_detect_text_threshold_override_is_restored(seg):
    assert len(seg.detect_text("person", threshold=0.8)) == 1
    assert seg.processor.confidence_threshold == 0.5
    assert len(seg.detect_text("person")) == 3


def test_track_presence_off(monkeypatch):
    s = _make(monkeypatch, track_presence=False)
    s.detect_text("person")
    assert s.last_presence is None


def test_factory_builds_sam3(monkeypatch):
    from promptcorrect.segmenters import make_segmenter

    _install(monkeypatch, FakeModel())
    assert make_segmenter("sam3", ckpt_path="x.pt", device="cpu", image_dirs=[]).backend == "sam3"
