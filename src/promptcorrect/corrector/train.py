"""Training loop and validation metrics for the error-map corrector. Owner: P5.

Validation reports, per error channel: pixel AUROC, IoU of the thresholded map with the true error map,
and for the click rule (clicks.py) the hit rate (click lands on a true error of the right kind) and the
abstain rate. Validation is an image-level holdout of the training images; the demo val images are kept
for the end-to-end rollout (rollout.py).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from promptcorrect.corrector.clicks import click_hits, propose_click
from promptcorrect.corrector.dataset import SampleSet
from promptcorrect.corrector.losses import error_map_loss
from promptcorrect.corrector.model import FN, FP, CorrectorNet, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    base: int = 32
    dice_weight: float = 1.0
    flip: bool = True
    amp: bool = True
    threshold: float = 0.5        # error-map threshold for IoU and clicks
    click_eval_samples: int = 2000  # per epoch; the final evaluation uses all val samples
    seed: int = 0


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (ties broken arbitrarily). NaN if only one class is present."""
    labels = labels.bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.float64, device=scores.device)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model: CorrectorNet, ds: SampleSet, cfg: TrainConfig, device, n_click: int | None = None,
             max_auroc_pixels: int = 2_000_000) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed)
    loss_sum, n_batches = 0.0, 0
    inter, union = np.zeros(2), np.zeros(2)
    pix_scores, pix_labels = [[], []], [[], []]
    hits, abstain, n_clicked = 0, 0, 0
    click_idx = set(torch.randperm(len(ds), generator=gen)[:n_click].tolist()) if n_click else set(range(len(ds)))
    keep_frac = min(1.0, max_auroc_pixels / max(1, len(ds) * ds.rgb.shape[-1] ** 2))
    for s in range(0, len(ds), cfg.batch_size * 2):
        idx = torch.arange(s, min(s + cfg.batch_size * 2, len(ds)))
        x, y, valid = ds.batch(idx.to(ds.rgb.device))
        x, y, valid = x.to(device), y.to(device), valid.to(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=cfg.amp and torch.device(device).type == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss, _ = error_map_loss(logits, y, valid, cfg.dice_weight)
        loss_sum, n_batches = loss_sum + float(loss), n_batches + 1
        probs = torch.sigmoid(logits)
        pred = probs > cfg.threshold
        v = valid[:, None]
        for c in (FP, FN):
            inter[c] += float((pred[:, c] & y[:, c].bool() & valid).sum())
            union[c] += float(((pred[:, c] | y[:, c].bool()) & valid).sum())
            sel = v[:, 0] & (torch.rand(valid.shape, generator=gen).to(device) < keep_frac)
            pix_scores[c].append(probs[:, c][sel].cpu())
            pix_labels[c].append(y[:, c][sel].bool().cpu())
        p_np, y_np = probs.cpu().numpy(), y.bool().cpu().numpy()
        for b, i in enumerate(idx.tolist()):
            if i not in click_idx:
                continue
            click = propose_click(p_np[b], cfg.threshold)
            hit = click_hits(click, y_np[b, FP], y_np[b, FN])
            if hit is None:
                abstain += 1
            else:
                n_clicked, hits = n_clicked + 1, hits + int(hit)
    out = {"val_loss": loss_sum / max(1, n_batches)}
    for c, name in ((FP, "fp"), (FN, "fn")):
        out[f"{name}_iou"] = inter[c] / union[c] if union[c] else float("nan")
        out[f"{name}_auroc"] = auroc(torch.cat(pix_scores[c]), torch.cat(pix_labels[c]))
    n_eval = n_clicked + abstain
    out.update(click_hit_rate=hits / n_clicked if n_clicked else float("nan"),
               abstain_rate=abstain / n_eval if n_eval else float("nan"), n_click_eval=n_eval)
    model.train()
    return out


def train(train_ds: SampleSet, val_ds: SampleSet, cfg: TrainConfig, out_dir: str | Path, device) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model = CorrectorNet(base=cfg.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(total, 1), pct_start=0.1)
    use_amp = cfg.amp and torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    gen = torch.Generator().manual_seed(cfg.seed)
    history, best, t0 = [], float("inf"), time.time()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val samples, {n_params / 1e6:.2f}M params, "
          f"{steps_per_epoch} steps/epoch", flush=True)
    log = (out_dir / "history.jsonl").open("w", encoding="utf-8")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_ds), generator=gen)
        run = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            flip = (torch.rand(len(idx), generator=gen) < 0.5) if cfg.flip else None
            x, y, valid = train_ds.batch(idx.to(train_ds.rgb.device), flip)
            x, y, valid = x.to(device, non_blocking=True), y.to(device), valid.to(device)
            with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
                logits = model(x)
            loss, parts = error_map_loss(logits.float(), y, valid, cfg.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run["loss"] += float(loss.detach()) * len(idx)
            run["bce"] += parts["bce"] * len(idx)
            run["dice"] += parts["dice"] * len(idx)
        row = {"epoch": epoch, **{f"train_{k}": v / len(train_ds) for k, v in run.items()},
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        row.update(evaluate(model, val_ds, cfg, device, n_click=cfg.click_eval_samples))
        history.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        save_checkpoint(model, out_dir / "last.pt", epoch=epoch, train_config=asdict(cfg))
        if row["val_loss"] < best:
            best = row["val_loss"]
            save_checkpoint(model, out_dir / "best.pt", epoch=epoch, train_config=asdict(cfg))
    log.close()
    return {"history": history, "best_val_loss": best, "n_params": n_params, "seconds": round(time.time() - t0, 1)}
