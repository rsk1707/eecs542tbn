"""Load corrector samples (samples.py) into memory as tensors. Owner: P5.

v0 datasets are small (tens of thousands of 128 x 128 crops, a few GB), so everything is held as one set
of tensors (on the GPU if it fits) and batches are gathered by index: no DataLoader workers needed.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from promptcorrect.corrector.model import build_inputs, error_targets
from promptcorrect.corrector.samples import ARRAYS


@dataclass
class SampleSet:
    rgb: Tensor      # uint8 [N, 3, S, S]
    logits: Tensor   # fp16 [N, S, S]
    gt: Tensor       # bool [N, S, S]
    valid: Tensor    # bool [N, S, S]
    box: Tensor      # bool [N, S, S]
    clicks: Tensor   # float32 [N, K, 3], label -1 = padding
    meta: list[dict]

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, idx) -> "SampleSet":
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return SampleSet(**{k: getattr(self, k)[idx_t.to(getattr(self, k).device)] for k in ARRAYS},
                         meta=[self.meta[i] for i in idx_t.tolist()])

    def to(self, device) -> "SampleSet":
        return SampleSet(**{k: getattr(self, k).to(device) for k in ARRAYS}, meta=self.meta)

    def batch(self, idx: Tensor, flip: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """-> (inputs [B,9,S,S], targets [B,2,S,S], valid [B,S,S]); `flip` bool [B] = horizontal flip."""
        rgb, logits, gt, valid, box, clicks = (getattr(self, k)[idx] for k in ARRAYS)
        if flip is not None and flip.any():
            S = rgb.shape[-1]
            f = flip.to(rgb.device)
            rgb = torch.where(f[:, None, None, None], rgb.flip(-1), rgb)
            logits, gt, valid, box = (torch.where(f[:, None, None], t.flip(-1), t) for t in (logits, gt, valid, box))
            clicks = clicks.clone()
            clicks[..., 0] = torch.where(f[:, None] & (clicks[..., 2] >= 0), S - clicks[..., 0], clicks[..., 0])
        return build_inputs(rgb, logits, box, valid, clicks), error_targets(logits, gt), valid


def load_samples(sample_dir: str | Path, max_samples: int | None = None) -> SampleSet:
    sample_dir = Path(sample_dir)
    meta = [json.loads(line) for line in (sample_dir / "meta.jsonl").read_text().splitlines() if line.strip()]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ARRAYS}
    for shard in sorted({m["shard"] for m in meta}):
        with np.load(sample_dir / f"shard_{shard:04d}.npz", allow_pickle=False) as z:
            for k in ARRAYS:
                parts[k].append(z[k])
    arrays = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    if len(meta) != len(arrays["rgb"]):
        raise ValueError(f"{sample_dir}: {len(meta)} meta rows but {len(arrays['rgb'])} samples")
    ds = SampleSet(**arrays, meta=meta)
    return ds.subset(range(max_samples)) if max_samples and max_samples < len(ds) else ds


def split_by_image(ds: SampleSet, val_frac: float = 0.1, salt: str = "p5v0") -> tuple[list[int], list[int]]:
    """Deterministic image-level holdout (no image contributes to both sides)."""
    train, val = [], []
    for i, m in enumerate(ds.meta):
        h = zlib.crc32(f"{salt}:{m['image_id']}".encode()) / 2**32
        (val if h < val_frac else train).append(i)
    return train, val
