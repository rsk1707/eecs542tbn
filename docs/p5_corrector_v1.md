# Corrector v1 (P5)

**Status (2026-11-26):** code ready on `p5/corrector-v1`. CPU tests pass (`tests/p5/test_corrector_v1.py`, 7 tests;
full suite green). All three scripts were smoke-tested locally end to end with `DummySegmenter` standing in for SAM 3.
**Not yet run on GreatLakes / SAM 3: no v1 numbers exist yet.**
**Owner:** Sharvani (P5). Builds on v0 (`docs/p5_corrector_v0.md`) and P4's samples `p4_samples_synth_0`.

## What changes from v0 (docs/next_steps.md section 2, item 1)

| Requirement | v1 |
|---|---|
| (a) learned click-or-abstain | **Quality head** on the same U-Net: pooled bottleneck + decoder features + predicted error mass -> predicted IoU `q` of the current mask (BCE vs the true IoU). Click only if `q < tau`. |
| threshold on held-out *training* images | `p5_calibrate_v1.py` replays every held-out training sample (the train run's 10% image holdout) through SAM 3, places the corrector's click and **measures** the IoU change. It then picks (error threshold, `tau`) with the largest mean ΔIoU subject to **harm on good masks (start IoU >= 0.75) <= 5%**. `tau = 0` (never click) is always allowed. |
| (b) perturbed boxes | Trains on P4's `p4_samples_synth_0`: 13,005 objects x (clean GT box + 2 shift / scale / truncate boxes) = 39,015 samples from the 3,000 leakage-safe COCO train images. The crop and box channel follow the prompt box. |
| (c) logit feedback | `--feedback true` (default): each corrector click is sent with the previous low-res logits as `Prompt.mask_logits`. Calibration measures both settings. The oracle in the same eval uses the same setting. P4's rule baselines never feed back (their definition). |

Unchanged: inputs (9 channels, 128 x 128 crop, context 0.5), error-map head and loss, click rule (`clicks.py`), one
click per object (v1 is trained on step-0 states only, so `steps=1` is the in-distribution setting).

Code: `corrector/model.py` (`quality_head`), `losses.py` (`quality_loss`), `dataset.py` (per-sample `iou`),
`train.py` (quality loss and metrics), `policy.py` (decision), `calibrate.py`, `evaluate_v1.py`.
v0 checkpoints and `rollout.py` are untouched and still load and run.

## Evaluation (`p5_eval_v1.py`)

Demo val objects (2,937 in 487 COCO val images). Each object gets its clean GT box (`prompt_origin=gt`) plus 2 P4 perturbed boxes
(`perturbed`, drawn with `object_rng(seed, obj_id, "eval")`, independent of the training draws). Every method starts
from the **same step-0 mask** and gets the **same step budget** (1):

`b0_none`, `corrector_v1`, `corrector_v0` (with `--v0-train-run`: v0 settings), `b1_random_k`, `b2_multimask`,
`b3_box_refit`, `b3_unc_click` (P4's `baselines/rules.py`), `oracle_clicks` (P3's `eval/oracle_clicks.py`).

Rows are P3's `ResultRow`s with P4's `describe()` severity, written by P3's `write_results`, so P3's code can read
`results.parquet` directly for the shared table. The script also writes:

- `summary_final.csv`: per (source gt / perturbed / all) x (start < 0.75, >= 0.75, size bin) x method: mean IoU
  before/after, ΔIoU, % below 0.75 before/after, harm %, improved %, % clicked;
- `go_no_go.csv`: the proposed criterion: ΔIoU on starts < 0.75 beats the best rule baseline, **and** harm on
  starts >= 0.75 is < 5%;
- a row in `results/runs_index.csv` (commit it).

Detector boxes (Grounding DINO / SAM 3 text) are **not** in this eval: P3's `run_rq0` owns the matching and accepts
only its registered methods. Proposal for P3: let `run_rq0` take extra `{method: Rule}` callables. A corrector
`Rule` needs the current image, which `CorrectorPolicy.view` takes.

## Running on GreatLakes

One-time setup: clone the repo in `$HOME` and check out the branch.

```bash
ssh sharvani@greatlakes.arc-ts.umich.edu
cd ~/eecs542tbn            # or: git clone https://github.com/rsk1707/eecs542tbn.git ~/eecs542tbn && cd ~/eecs542tbn
git fetch && git checkout p5/corrector-v1 && git pull
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
ls $PC_ROOT/runs/p4_samples_synth_0/meta.jsonl $PC_ROOT/runs/demo_p5_train_0/best.pt   # inputs exist?
touch $PC_ROOT/runs/.write_test_sharvani && rm $PC_ROOT/runs/.write_test_sharvani      # can write runs/?
```

**1. Smoke test first** (about 15-30 min including the queue; small subsets, nothing goes to the index):

```bash
sbatch --account=eecs542f26s001_class --time=00:45:00 --mem=64G --job-name=p5v1_smoke \
    slurm/run_gpu.sbatch scripts/p5_run_v1.py --paths configs/paths/greatlakes.yaml --suffix smoke \
    --train-args "--max-samples 3000 --epochs 2" --calib-args "--max-images 20" \
    --eval-args "--max-images 10 --no-index --v0-train-run demo_p5_train_0"
squeue -u sharvani
tail -f p5v1_smoke-<jobid>.out
```

**2. Full run** (train about 30-60 min, calibrate about 10-20 min, eval about 1-2 h; estimates, not measured):

```bash
sbatch --account=eecs542f26s001_class --time=05:00:00 --mem=64G --job-name=p5v1 \
    slurm/run_gpu.sbatch scripts/p5_run_v1.py --paths configs/paths/greatlakes.yaml --suffix 0 \
    --eval-args "--v0-train-run demo_p5_train_0"
```

Run ids: `p5v1_train_0`, `p5v1_calib_0`, `p5v1_eval_0`. If the job dies (for example at the time limit), resubmit the same
command: finished steps are skipped. A half-finished train run is retrained, so delete `$PC_ROOT/runs/p5v1_train_0` first.
Post the job id and the GPU-hours in the team channel.

**What to look at**

1. `p5v1_train_0/final_val.json`: `q_mae`, `q_fail_auroc` (can the head tell failures apart?), error-map AUROCs.
2. `p5v1_calib_0/calibration.json`: the chosen `err_threshold` / `tau` per feedback setting, the held-out
   ΔIoU and harm there, and `no_gate_reference` (what clicking everything gives, as in v0).
   If `clicked_pct` is 0, the corrector cannot click without exceeding 5% harm: report that honestly.
3. `p5v1_eval_0/go_no_go.csv` and `summary_final.csv`.

**Variants** (one eval each, reusing the train run):

```bash
# no logit feedback (compare with the oracle without feedback)
sbatch --account=eecs542f26s001_class --time=03:00:00 --job-name=p5v1_eval_nofb slurm/run_gpu.sbatch \
    scripts/p5_eval_v1.py --paths configs/paths/greatlakes.yaml --run-id p5v1_eval_0_nofb \
    --train-run p5v1_train_0 --calib-run p5v1_calib_0 --feedback false
```

## Known limits

- Calibration and best-checkpoint selection use the same held-out training images (no val data is involved).
- The quality head predicts the current mask's IoU, not the gain of the click. A gain head trained from
  calibration-style rollouts on all training samples is the natural next step if the gate is too coarse.
- One click, one seed, 128 px crops (fine boundaries on large objects are lost).
