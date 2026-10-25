"""Shared contracts. Keep in sync with docs/interfaces.md (change both in the same PR).

Conventions: boxes are xyxy absolute pixels in the original image; masks are COCO RLE on disk
and low-res logits in memory (binary mask = logit > 0); object ids are strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep `import promptcorrect.types` torch-free
    from torch import Tensor

# One text detection: (box xyxy, mask as COCO RLE or None, score). LLMDet/Grounding DINO return mask=None.
TextDetection = tuple[list[float], "dict | None", float]


@dataclass
class ObjectRecord:  # owner: P2
    obj_id: str                # "coco:123456" / "lvis:98765"
    image_id: int              # COCO image id (LVIS shares it)
    source: str                # "coco" | "lvis"
    category_id: int
    category_name: str         # also the SAM 3 text prompt (underscores -> spaces; LVIS: first synonym)
    novelty: str               # "seen" | "unseen" (RQ3 split)
    gt_box: list[float]        # xyxy, pixels
    gt_mask: dict              # COCO RLE (counts is a str when serialized)
    area: float
    # stratification attributes, computed once by P2
    size_bin: str              # "small" | "medium" | "large" (COCO thresholds)
    thinness: float            # perimeter^2 / area
    n_neighbors: int           # same-class objects within one box-width


@dataclass
class Prompt:  # owner: P4
    box: list[float] | None = None                 # xyxy
    points: list[tuple[float, float, int]] = field(default_factory=list)  # (x, y, label) 1=pos, 0=neg
    mask_logits: "Tensor | None" = None            # previous low-res logits, optional
    origin: str = "gt"   # "gt" | "perturbed" | "sam3_text" | "llmdet" | "oracle" | "baseline:<name>" | "corrector"
    perturbation: dict | None = None               # {"type": "shift", "params": {...}}


@dataclass
class SegOutput:  # owner: P1
    mask_logits: "Tensor"                # native low-res logits, shape [h, w]
    iou_score: float
    presence_score: float | None = None  # None on SAM 2.1 (and possibly on the interactive head; see O6)
    # Only filled when predict(..., multimask=True): all candidate masks [K, h, w] and their scores [K] (O5).
    alt_logits: "Tensor | None" = None
    alt_ious: "Tensor | None" = None


@dataclass
class ResultRow:  # owner: P3; one row per (obj_id, method, step), saved as Parquet
    run_id: str
    obj_id: str
    image_id: int
    method: str
    step: int                            # 0 = initial mask before any correction
    iou: float
    boundary_iou: float
    prompt_origin: str
    pert_type: str | None = None
    pert_level: float | None = None
    box_iou_in: float | None = None
    coverage: float | None = None
    contamination: float | None = None
    # Proposed in docs/interfaces.md O4 (needed for RQ2 real-vs-synthetic comparison):
    delta_cx: float | None = None
    delta_cy: float | None = None
    delta_log_sw: float | None = None
    delta_log_sh: float | None = None
    novelty: str | None = None
    size_bin: str | None = None
    thinness: float | None = None
    n_decoder_calls: int = 0             # cumulative up to and including this step
    latency_ms: float = 0.0              # this step only, excluding the image encoder
