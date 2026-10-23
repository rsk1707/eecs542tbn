"""DummySegmenter: lets everyone develop and test on a laptop, no GPU, no weights, no images.

It cheats: it knows the GT masks of the objects it is constructed with. Behaviour (deterministic):

* Target object: the GT object whose box best overlaps `prompt.box`; with no box, the object
  containing the first positive point.
* Initial mask = (GT eroded) | (thin ring of false-positive pixels just outside the GT), clipped to the
  prompt box. So a truncated box loses GT pixels, a loose box keeps ring false positives.
* Clicks (applied in order, all of `prompt.points`): a positive click inside the GT adds a disk of GT
  pixels; a negative click outside the GT removes a disk of false-positive pixels. Wrong-way clicks
  hurt (positive outside the GT adds false positives, negative inside the GT removes GT pixels).
* Disk radius, erosion and ring width scale with sqrt(object area).
* `iou_score` is the true IoU (the dummy cheats); `presence_score` is None (tests the None path).
* Logits are computed at full resolution, then resized to `low_res` x `low_res` (aspect not
  preserved, like SAM 2). `to_full_res` resizes back, so coordinate-handling bugs show up in tests.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from promptcorrect.data.geometry import rle_decode, rle_encode
from promptcorrect.data.records import group_by_image
from promptcorrect.segmenters.base import Segmenter
from promptcorrect.types import ObjectRecord, Prompt, SegOutput, TextDetection


def _box_iou(a: list[float], b: list[float]) -> float:  # private; the shared severity code is P4's describe()
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class DummySegmenter(Segmenter):
    backend = "dummy"

    def __init__(
        self,
        objects: Iterable[ObjectRecord],
        low_res: int | None = 256,
        erode_frac: float = 0.04,
        ring_frac: float = 0.05,
        click_frac: float = 0.2,
    ):
        self._by_image = group_by_image(objects)
        self.low_res = low_res
        self.erode_frac, self.ring_frac, self.click_frac = erode_frac, ring_frac, click_frac
        self._objs: list[ObjectRecord] = []
        self._gt: list[np.ndarray] = []
        self._hw: tuple[int, int] = (0, 0)

    # -- interface -----------------------------------------------------------------------------
    def set_image(self, image_id: int) -> None:
        if image_id not in self._by_image:
            raise KeyError(f"DummySegmenter has no objects for image {image_id}")
        self._objs = self._by_image[image_id]
        self._hw = tuple(self._objs[0].gt_mask["size"])  # (H, W) from the RLE
        self._gt = [rle_decode(o.gt_mask) for o in self._objs]

    def predict(self, prompts: list[Prompt], multimask: bool = False) -> list[SegOutput]:
        outs = []
        for p in prompts:
            mask, gt = self._mask_for(p)
            inter = np.logical_and(mask, gt).sum()
            union = np.logical_or(mask, gt).sum()
            iou = float(inter / union) if union > 0 else 0.0
            logits = self._to_logits(mask)
            out = SegOutput(mask_logits=logits, iou_score=iou, presence_score=None)
            if multimask:
                out.alt_logits = logits[None].clone()
                out.alt_ious = torch.tensor([iou])
            outs.append(out)
        return outs

    def detect_text(self, phrase: str) -> list[TextDetection]:
        dets = []
        for obj in self._objs:
            if obj.category_name == phrase:
                out = self.predict([Prompt(box=list(obj.gt_box), origin="sam3_text")])[0]
                dets.append((list(obj.gt_box), rle_encode(self.full_res_mask(out)), 0.9))
        return dets

    def to_full_res(self, logits: torch.Tensor) -> torch.Tensor:
        if tuple(logits.shape) == self._hw:
            return logits
        return F.interpolate(logits[None, None], size=self._hw, mode="bilinear", align_corners=False)[0, 0]

    # -- internals -----------------------------------------------------------------------------
    def _target_index(self, p: Prompt) -> int:
        if p.box is not None:
            return int(np.argmax([_box_iou(p.box, o.gt_box) for o in self._objs]))
        for x, y, label in p.points:
            if label == 1:
                for i, g in enumerate(self._gt):
                    if g[self._clip_y(y), self._clip_x(x)]:
                        return i
        raise ValueError("Prompt has no box and no positive point inside any object")

    def _clip_x(self, x: float) -> int:
        return int(min(max(int(x), 0), self._hw[1] - 1))

    def _clip_y(self, y: float) -> int:
        return int(min(max(int(y), 0), self._hw[0] - 1))

    def _mask_for(self, p: Prompt) -> tuple[np.ndarray, np.ndarray]:
        gt = self._gt[self._target_index(p)]
        H, W = self._hw
        scale = float(np.sqrt(gt.sum()))
        e = max(1, round(self.erode_frac * scale))
        d = max(1, round(self.ring_frac * scale))
        r = max(3, round(self.click_frac * scale))

        eroded = ndimage.binary_erosion(gt, iterations=e)
        if not eroded.any():
            eroded = gt.copy()
        ring = ndimage.binary_dilation(gt, iterations=d) & ~gt
        mask = eroded | ring
        if p.box is not None:
            x0, y0, x1, y1 = p.box
            region = np.zeros((H, W), dtype=bool)
            region[max(0, int(np.floor(y0))):int(np.ceil(y1)), max(0, int(np.floor(x0))):int(np.ceil(x1))] = True
            mask = mask & region

        yy, xx = np.ogrid[:H, :W]
        for x, y, label in p.points:
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            inside = bool(gt[self._clip_y(y), self._clip_x(x)])
            if label == 1:
                mask = mask | (disk & (gt if inside else ~gt))
            else:
                mask = mask & ~(disk & (gt if inside else ~gt))
        return mask, gt

    def _to_logits(self, mask: np.ndarray) -> torch.Tensor:
        if not mask.any():
            sd = -np.full(mask.shape, 12.0)
        elif mask.all():
            sd = np.full(mask.shape, 12.0)
        else:
            sd = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
        logits = torch.from_numpy((4.0 * np.tanh(sd / 3.0)).astype(np.float32))  # sign matches the mask
        if self.low_res is not None:
            logits = F.interpolate(logits[None, None], size=(self.low_res, self.low_res),
                                   mode="bilinear", align_corners=False)[0, 0]
        return logits
