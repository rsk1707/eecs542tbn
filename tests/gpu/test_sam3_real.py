"""Tests of Sam3Segmenter against the REAL SAM 3. Skipped unless PC_GPU_TESTS=1.

Run on a GPU node with the weights and COCO val in place (docs/environment.md):
    PC_GPU_TESTS=1 sbatch --account=eecs542f26s001_class --time=00:20:00 \
        slurm/run_gpu.sbatch -m pytest tests/gpu -q -s
Uses COCO val image 632 (a bed) and its GT box, as in the smoke test.
"""

import os

import numpy as np
import pytest

from promptcorrect.data.geometry import rle_decode
from promptcorrect.types import Prompt

pytestmark = pytest.mark.skipif(
    os.environ.get("PC_GPU_TESTS") != "1", reason="set PC_GPU_TESTS=1 on a GPU node with SAM 3 weights and COCO val")

IMAGE_ID = 632
BOX = [3.3, 266.9, 404.5, 475.1]


def _iou(a, b):
    return float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())


@pytest.fixture(scope="module")
def seg():
    from promptcorrect.segmenters.sam3 import Sam3Segmenter

    s = Sam3Segmenter()
    s.set_image(IMAGE_ID)
    return s


def test_box_prompt_shapes_and_sanity(seg):
    out = seg.predict([Prompt(box=BOX)])[0]
    assert tuple(out.mask_logits.shape) == (288, 288) and 0 < out.iou_score <= 1
    mask = seg.full_res_mask(out)
    x0, y0, x1, y1 = (int(v) for v in BOX)
    assert mask.sum() > 0 and mask[:y0 - 20].sum() == 0 and mask[:, :max(0, x0 - 20)].sum() == 0  # stays near the box


def test_batched_boxes_match_single_calls(seg):
    boxes = [BOX, [20.0, 280.0, 380.0, 470.0], [3.3, 266.9, 300.0, 475.1]]
    batched = seg.predict([Prompt(box=b) for b in boxes])
    for b, bo in zip(boxes, batched):
        single = seg.predict([Prompt(box=b)])[0]
        assert _iou(seg.full_res_mask(bo), seg.full_res_mask(single)) > 0.98


def test_to_full_res_matches_sam3s_own_masks(seg):
    box = np.asarray(BOX, np.float32)[None]
    full, _, low = seg.model.predict_inst(seg._state, point_coords=None, point_labels=None, box=box,
                                          multimask_output=False, return_logits=True)
    import torch

    ours = (seg.to_full_res(torch.from_numpy(np.asarray(low).reshape(288, 288))) > 0).numpy()
    theirs = np.asarray(full).reshape(seg._hw) > 0
    assert (ours == theirs).mean() > 0.999


def test_detect_text(seg):
    dets = seg.detect_text("bed")
    assert len(dets) >= 1
    H, W = seg._hw
    scores = [d[2] for d in dets]
    assert scores == sorted(scores, reverse=True)
    for (x0, y0, x1, y1), rle, _ in dets:
        assert 0 <= x0 <= x1 <= W and 0 <= y0 <= y1 <= H and rle_decode(rle).shape == (H, W)
    assert 0 < seg.last_presence <= 1


def test_text_call_does_not_break_prompts(seg):
    seg.detect_text("bed")
    out = seg.predict([Prompt(box=BOX)])[0]
    assert seg.full_res_mask(out).sum() > 0
