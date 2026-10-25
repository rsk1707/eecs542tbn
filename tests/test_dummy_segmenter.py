import numpy as np
import pytest
from scipy import ndimage

from promptcorrect.data.geometry import rle_decode
from promptcorrect.segmenters import make_segmenter
from promptcorrect.types import Prompt


def _iou(a: np.ndarray, b: np.ndarray) -> float:  # test-local; the shared implementation is eval.metrics.iou (P3)
    return float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())


def _obj(seg, k=0):
    return seg._objs[k]


def _gt(seg, k=0):
    return rle_decode(_obj(seg, k).gt_mask)


def _run(seg, prompt):
    return _iou(seg.full_res_mask(seg.predict([prompt])[0]), _gt(seg))


def test_clean_box_is_good_but_not_perfect(seg):
    iou = _run(seg, Prompt(box=list(_obj(seg).gt_box)))
    assert 0.75 < iou < 1.0


def test_logits_are_low_res_and_upsample(seg):
    out = seg.predict([Prompt(box=list(_obj(seg).gt_box))])[0]
    H, W = seg._hw
    assert tuple(out.mask_logits.shape) == (256, 256)
    assert tuple(seg.to_full_res(out.mask_logits).shape) == (H, W)
    assert out.presence_score is None
    mask = seg.full_res_mask(out)
    assert abs(_iou(mask, _gt(seg)) - out.iou_score) < 0.03  # score is the true IoU, up to resampling


def test_truncated_box_hurts(seg):
    x0, y0, x1, y1 = _obj(seg).gt_box
    clean = _run(seg, Prompt(box=[x0, y0, x1, y1]))
    cut = _run(seg, Prompt(box=[x0, y0, x0 + 0.6 * (x1 - x0), y1]))
    assert cut < clean - 0.1


def test_positive_click_in_missing_region_helps(seg):
    x0, y0, x1, y1 = _obj(seg).gt_box
    box = [x0, y0, x0 + 0.6 * (x1 - x0), y1]
    base = seg.full_res_mask(seg.predict([Prompt(box=box)])[0])
    missing = _gt(seg) & ~base
    dist = ndimage.distance_transform_edt(missing)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    clicked = _run(seg, Prompt(box=box, points=[(float(x), float(y), 1)]))
    assert clicked > _iou(base, _gt(seg)) + 0.03


def test_negative_click_on_false_positive_helps(seg):
    box = list(_obj(seg).gt_box)
    base = seg.full_res_mask(seg.predict([Prompt(box=box)])[0])
    fp = base & ~_gt(seg)
    assert fp.any()
    dist = ndimage.distance_transform_edt(fp)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    clicked = _run(seg, Prompt(box=box, points=[(float(x), float(y), 0)]))
    assert clicked >= _iou(base, _gt(seg))


def test_wrong_way_click_hurts(seg):
    box = list(_obj(seg).gt_box)
    base_iou = _run(seg, Prompt(box=box))
    gt = _gt(seg)
    dist = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)  # negative click in the object centre
    assert _run(seg, Prompt(box=box, points=[(float(x), float(y), 0)])) < base_iou


def test_targets_second_object(seg):
    box = list(_obj(seg, 1).gt_box)
    out = seg.predict([Prompt(box=box)])[0]
    assert _iou(seg.full_res_mask(out), _gt(seg, 1)) > 0.75


def test_point_only_prompt(seg):
    gt = _gt(seg)
    y, x = np.unravel_index(np.argmax(ndimage.distance_transform_edt(gt)), gt.shape)
    out = seg.predict([Prompt(points=[(float(x), float(y), 1)])])[0]
    assert seg.full_res_mask(out).any()


def test_batch_and_multimask(seg):
    prompts = [Prompt(box=list(_obj(seg, k).gt_box)) for k in range(2)]
    outs = seg.predict(prompts, multimask=True)
    assert len(outs) == 2 and outs[0].alt_logits.shape[0] == 1


def test_detect_text(seg):
    dets = seg.detect_text(_obj(seg).category_name)
    assert len(dets) >= 1
    box, rle, score = dets[0]
    assert len(box) == 4 and "counts" in rle and 0 < score <= 1
    assert seg.detect_text("no such category") == []


def test_unknown_image(seg):
    with pytest.raises(KeyError):
        seg.set_image(999)


def test_factory(objects):
    assert make_segmenter("dummy", objects=objects).backend == "dummy"
    with pytest.raises(NotImplementedError):
        make_segmenter("sam21")  # SAM 2.1 wrapper is still a stub
    with pytest.raises(ValueError):
        make_segmenter("nope")
