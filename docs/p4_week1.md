# P4 week 1: perturbations, severity, baselines, sensitivity sweep

Owner: Aman (P4). Status 2026-10-25: implemented and tested on the dummy segmenter (CPU). **Nothing has been run
on SAM 3 yet**; the commands below are ready for GreatLakes.

## What exists

| Module | What it does |
|---|---|
| `perturb/box.py` | `shift_box`, `scale_box`, `truncate_box` relative to the GT box, clipped to the image; `perturb(box, type, level, hw, rng)`; `sweep_prompts` (clean box + the grid); `sample_perturbation` (random training perturbations); `object_rng(seed, *keys)` (reproducible regardless of loop order) |
| `perturb/severity.py` | `describe(prompt, obj, neighbors)` -> `box_iou`, `coverage`, `contamination`, `delta` (the one shared implementation); `severity_columns` -> ResultRow names |
| `baselines/rules.py` | `b1_random_k`, `b2_multimask`, `b3_box_refit`, `b3_unc_click` and `rollout(segmenter, method, prompt, steps)` |
| `perturb/sweep.py` | `run_sweep` (sensitivity grid, optionally every method incl. `oracle_clicks`) and the summary tables |
| `viz/sensitivity_figures.py` | IoU vs level per type, IoU vs box IoU, baselines vs ideal clicks + harm on clean boxes |
| `scripts/run_sensitivity.py`, `scripts/make_sensitivity_figures.py` | thin CLIs |

## Definitions (also in `docs/interfaces.md` section 2)

- **Levels.** shift: displacement as a fraction of box size, `offset = level * (w cos a, h sin a)`, random direction `a`
  per object and level. scale: side-length factor around the centre. truncate: fraction cut off one random side.
  Default grid: shift 0.05-0.5 (6 levels), scale 0.5-2.0 (6), truncate 0.1-0.5 (5), plus the clean box = 18 prompts per object.
- **Severity** is measured by `describe`, not by `level`, whenever types (or synthetic vs. real boxes) are compared.
- **Baseline rules** never look at the GT. Assumptions worth discussing:
  - `b3_unc_click` labels a click positive if it lies inside the prompt box and negative outside (it trusts the box).
    Uncertain = `|logit| < 1` at full resolution; a region that already holds a click is not clicked again.
  - `b3_box_refit` uses the tight box of the current mask (`margin=0` by default). With a tight box it cannot undo
    truncation, since SAM's mask rarely leaves the box; `margin=0.1` is available if we want a stronger variant.
  - Clicks accumulate across steps; `Prompt.mask_logits` is not fed back (same as the RQ0 oracle).
- **Harm** = final IoU < step-0 IoU - 0.01 (P3's definition). Reported separately for clean GT boxes (RQ4-style) and perturbed ones.

## Commands (GreatLakes, from the repo root)

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
A=--account=eecs542f26s001_class; P=configs/paths/greatlakes.yaml

# 1. Sensitivity sweep, no correction. Talk-sized: 200 images (about 1,200 objects, about 35 min + startup).
#    --time-budget-min stops cleanly and SAVES what is done if the node is slow (keep it below --time).
sbatch $A --time=01:30:00 slurm/run_gpu.sbatch scripts/run_sensitivity.py --paths $P --run-id demo_sens_1     --max-images 200 --no-boundary --time-budget-min 70
#    The whole set (487 images, about 53k prompts, about 85 min + startup):
#    sbatch $A --time=02:30:00 slurm/run_gpu.sbatch scripts/run_sensitivity.py --paths $P --run-id demo_sens_full #        --no-boundary --time-budget-min 130

# 2. Baselines vs ideal clicks on a coarser grid (one decoder call per step, so fewer prompts)
sbatch $A --time=02:00:00 slurm/run_gpu.sbatch scripts/run_sensitivity.py --paths $P --run-id demo_baselines_1 \
    --methods all --steps 3 --levels shift=0.2,0.4 scale=0.67,1.5 truncate=0.3

# 3. Figures (CPU, seconds)
python scripts/make_sensitivity_figures.py --paths $P --run-id demo_sens_1
python scripts/make_sensitivity_figures.py --paths $P --run-id demo_baselines_1
```

Measured on an A40 node (job 64785358, 5 images, 40 objects, 720 prompts): 68.7 s of sweep, about 95 ms per prompt,
plus 5-10 minutes of cold start (imports and model load) that the script's own timer does not count. That puts the
full set at about 85 minutes, so the earlier `--time=01:00:00` for the full sweep would have timed out and, because
results are written at the end, saved nothing. `--time-budget-min` fixes the second part; the first is just sizing.
On a cluster core the per-prompt CPU work (mask IoU, severity, and Boundary IoU, which `--no-boundary` skips) is
likely the bottleneck rather than the GPU. The progress line printed every `--log-every` images (default 10) now
shows an ETA and the seconds spent in prediction, upsampling, metrics and rollouts, and `sweep_manifest.json`
records the same split (`seconds_by_phase`), `images_done` and `stopped_early`. Job 2 (baselines) has ~2.9k
objects x 6 prompts x 5 methods x 3 steps = ~260k decoder calls, about 25 min of decoder time plus CPU metrics;
use `--max-images` first, and `--time-budget-min` here too.

Local check without a GPU (needs an `objects_val.jsonl`; dummy IoUs are meaningless):
`python scripts/run_sensitivity.py --paths configs/paths/local.yaml --run-id local_test --backend dummy --max-images 5`.

## Changes made by P1 on 2026-10-25 (timing)

- `run_sweep` / `run_sensitivity.py`: `--time-budget-min`, `--no-boundary`, `--log-every`, progress with ETA, a
  per-phase time split, and the early-stop flags above.
- `by_box_iou` blanks cells with fewer than `MIN_BIN_N` (5) prompts, in the summary table and the figure
  (the 5-image check had a bin whose mean IoU of 0.011 came from one prompt).

## Not done yet (later weeks)

- Training-data variants (`train_data: synthetic | real | mixed | matched`): need P5's sample format and P2's mined
  real failures. `sample_perturbation` is the synthetic generator they will use.
- Distractor / wrong-instance / point perturbations (listed in `RunConfig.perturbations` examples): not implemented.
- Running the baselines on *real* detector failures (needs the RQ0 detections and `match`): the rules take any
  starting prompt, so this is a small runner change once P3's RQ0 rows are final.
