"""SAM 3 backend (docs/interfaces.md section 3). Owner: P1.

Built from the sprint smoke test (scripts/smoke_sam3.py, all checks passed on an A40) and SAM 3's code at
the pinned commit. Facts this wrapper relies on:

* Interactive path: `model.predict_inst(state, point_coords, point_labels, box, mask_input,
  multimask_output, return_logits)` after `Sam3Processor.set_image`. Boxes are xyxy pixels in the original
  image. Low-res logits are 288 x 288 (not 256), already clamped to [-32, 32].
* `predict_inst` squeezes dim 0, so one prompt returns (C, ...) and N boxes return (N, C, ...). We reshape
  explicitly instead of trusting the shape.
* Several boxes for the same image go through ONE call (rows are independent objects); points need one
  call per prompt.
* Full-res masks come from `_transforms.postprocess_masks`, which also fills holes of up to 256 low-res
  pixels before a bilinear resize. `to_full_res` reuses it so results match SAM 3's own masks exactly.
* The predicted IoU is overconfident (e.g. 0.93 predicted vs 0.58 real for an ambiguous point prompt) and
  multimask ordering is unreliable; "best of multimask by predicted IoU" is a weak baseline.
* Text path: `Sam3Processor.set_text_prompt` returns detections in no particular order, with boxes that can
  fall slightly outside the image, filtered by a confidence threshold (default 0.5). Its presence score is
  folded into `scores` and not stored, so `detect_text` reads the raw value with one extra grounding pass
  (`last_presence`). `predict` has no presence score (`SegOutput.presence_score` is None).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from promptcorrect.config import load_paths
from promptcorrect.data.geometry import rle_encode
from promptcorrect.segmenters.base import Segmenter
from promptcorrect.types import Prompt, SegOutput, TextDetection

LOW_RES = 288  # SAM 3's decoder output size (SAM 2.1 uses 256)


class Sam3Segmenter(Segmenter):
    backend = "sam3"

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        device: str | None = None,
        image_dirs: list[str | Path] | None = None,
        confidence_threshold: float = 0.5,
        track_presence: bool = True,
        box_batch_size: int = 32,
    ):
        """`ckpt_path` defaults to <ckpt_root>/sam3/sam3.pt; `image_dirs` default to COCO val2017 then
        train2017 under <data_root>/coco (configs/paths, PC_PATHS). `track_presence=False` skips the
        extra grounding pass in `detect_text` (about 2x faster text calls)."""
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        paths = load_paths()
        self.ckpt = Path(ckpt_path) if ckpt_path else paths.ckpt_root / "sam3" / "sam3.pt"
        self.image_dirs = [Path(d) for d in image_dirs] if image_dirs else [
            paths.data_root / "coco" / "val2017", paths.data_root / "coco" / "train2017"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.track_presence = track_presence
        self.box_batch_size = box_batch_size
        self.last_presence: float | None = None  # raw presence prob of the latest detect_text call

        self.model = build_sam3_image_model(
            checkpoint_path=str(self.ckpt), load_from_HF=False, enable_inst_interactivity=True, device=self.device)
        self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=confidence_threshold)
        self._state: dict | None = None
        self._hw: tuple[int, int] = (0, 0)

    # -- image ---------------------------------------------------------------------------------
    def set_image(self, image_id: int) -> None:
        """Load COCO image `image_id` from `image_dirs` and run the image encoder (no cache yet)."""
        name = f"{image_id:012d}.jpg"
        for d in self.image_dirs:
            if (d / name).exists():
                self.set_image_array(Image.open(d / name).convert("RGB"))
                return
        raise FileNotFoundError(f"{name} not found in {[str(d) for d in self.image_dirs]}")

    def set_image_array(self, image: Image.Image | np.ndarray) -> None:
        """Encode an in-memory RGB image (PIL or [H, W, 3] uint8)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self._state = self.processor.set_image(image)
        self._hw = (image.height, image.width)

    def _require_image(self) -> dict:
        if self._state is None:
            raise RuntimeError("call set_image() / set_image_array() first")
        return self._state

    # -- interactive prompts -------------------------------------------------------------------
    @staticmethod
    def _boxes_only(p: Prompt) -> bool:
        return p.box is not None and not p.points and p.mask_logits is None

    def _call(self, box: np.ndarray | None, points: np.ndarray | None, labels: np.ndarray | None,
              mask_input: np.ndarray | None, multimask: bool, n: int):
        """One predict_inst call; returns scores [n, C] and low-res logits [n, C, 288, 288] (numpy)."""
        with torch.inference_mode():
            _, scores, low = self.model.predict_inst(
                self._require_image(), point_coords=points, point_labels=labels, box=box,
                mask_input=mask_input, multimask_output=multimask, return_logits=True)
        scores = np.asarray(scores, dtype=np.float32).reshape(n, -1)
        low = np.asarray(low, dtype=np.float32).reshape(n, scores.shape[1], LOW_RES, LOW_RES)
        return scores, low

    @staticmethod
    def _output(scores: np.ndarray, low: np.ndarray, j: int, multimask: bool) -> SegOutput:
        best = int(np.argmax(scores[j])) if multimask else 0
        out = SegOutput(mask_logits=torch.from_numpy(low[j, best].copy()), iou_score=float(scores[j, best]),
                        presence_score=None)
        if multimask:
            out.alt_logits = torch.from_numpy(low[j].copy())
            out.alt_ious = torch.from_numpy(scores[j].copy())
        return out

    def predict(self, prompts: list[Prompt], multimask: bool = False) -> list[SegOutput]:
        self._require_image()
        for p in prompts:
            if p.box is None and not p.points:
                raise ValueError("Prompt needs a box or at least one point")
        outs: list[SegOutput | None] = [None] * len(prompts)

        # box-only prompts: batched, one decoder call per chunk (rows are independent objects)
        idx = [i for i, p in enumerate(prompts) if self._boxes_only(p)]
        for s in range(0, len(idx), self.box_batch_size):
            chunk = idx[s:s + self.box_batch_size]
            boxes = np.asarray([prompts[i].box for i in chunk], dtype=np.float32).reshape(-1, 4)
            scores, low = self._call(boxes, None, None, None, multimask, n=len(chunk))
            for j, i in enumerate(chunk):
                outs[i] = self._output(scores, low, j, multimask)

        # everything else (points, mask_input): one call per prompt
        for i, p in enumerate(prompts):
            if outs[i] is not None:
                continue
            box = None if p.box is None else np.asarray(p.box, dtype=np.float32).reshape(1, 4)
            pts = lab = None
            if p.points:
                pts = np.asarray([[x, y] for x, y, _ in p.points], dtype=np.float32)
                lab = np.asarray([int(l) for _, _, l in p.points], dtype=np.int32)
            mask_input = None
            if p.mask_logits is not None:
                mask_input = p.mask_logits.detach().float().cpu().numpy()
                if mask_input.size != LOW_RES * LOW_RES:
                    raise ValueError(f"Prompt.mask_logits must be {LOW_RES}x{LOW_RES}, got {tuple(mask_input.shape)}")
                mask_input = mask_input.reshape(1, LOW_RES, LOW_RES)
            scores, low = self._call(box, pts, lab, mask_input, multimask, n=1)
            outs[i] = self._output(scores, low, 0, multimask)
        return outs  # type: ignore[return-value]

    def to_full_res(self, logits: torch.Tensor) -> torch.Tensor:
        """Low-res logits [h, w] -> logits [H, W] at the original image size (CPU tensor), using SAM 3's own
        postprocessing (small-hole filling + bilinear resize) so it matches `predict_inst`'s masks."""
        H, W = self._hw
        if tuple(logits.shape) == (H, W):
            return logits
        transforms = self.model.inst_interactive_predictor._transforms
        x = logits.detach().float().to(self.device).reshape(1, 1, *logits.shape[-2:])
        with torch.inference_mode():
            return transforms.postprocess_masks(x, (H, W))[0, 0].detach().cpu()

    # -- text prompts --------------------------------------------------------------------------
    def detect_text(self, phrase: str, threshold: float | None = None) -> list[TextDetection]:
        """Detections for `phrase` on the current image: [(box xyxy clipped to the image, mask RLE, score)],
        sorted by score, highest first. `threshold` overrides the confidence threshold for this call.
        Sets `self.last_presence` to the raw presence probability (None if track_presence is off)."""
        base = self._require_image()
        st = dict(base)  # shallow copies: the text outputs must not leak into the image state
        st["backbone_out"] = dict(base["backbone_out"])
        old = self.processor.confidence_threshold
        if threshold is not None:
            self.processor.confidence_threshold = threshold
        try:
            st = self.processor.set_text_prompt(prompt=phrase, state=st)
            self.last_presence = self._presence(st) if self.track_presence else None
        finally:
            self.processor.confidence_threshold = old

        scores = st["scores"].detach().float().cpu().numpy()
        if len(scores) == 0:
            return []
        boxes = st["boxes"].detach().float().cpu().numpy()
        masks = st["masks"].detach().cpu().numpy().reshape(len(scores), *self._hw).astype(bool)
        H, W = self._hw
        dets: list[TextDetection] = []
        for k in np.argsort(-scores, kind="stable"):
            x0, y0, x1, y1 = boxes[k]
            box = [float(np.clip(x0, 0, W)), float(np.clip(y0, 0, H)), float(np.clip(x1, 0, W)), float(np.clip(y1, 0, H))]
            dets.append((box, rle_encode(masks[k]), float(scores[k])))
        return dets

    def _presence(self, st: dict) -> float:
        with torch.inference_mode():
            raw = self.model.forward_grounding(
                backbone_out=st["backbone_out"], find_input=self.processor.find_stage,
                geometric_prompt=st["geometric_prompt"], find_target=None)
        return float(raw["presence_logit_dec"].sigmoid().reshape(-1)[0])
