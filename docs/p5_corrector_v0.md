# Corrector v0 (P5, demo sprint item 5)

**Status:** code ready, CPU tests pass (`tests/p5/`), GPU smoke test passed (samples, train, eval on SAM 3).
**Owner:** Sharvani (P5). Code: `src/promptcorrect/corrector/`. Scripts: `scripts/p5_*.py`.

## What v0 is

A small U-Net (about 1.9M parameters) looks at an object-centric crop of the image, the frozen SAM 3 mask and
the prompt box, and predicts a 2-channel **error map**: false positive (mask but not GT) and false negative
(GT but not mask). One click is placed with the standard ideal-click convention, but applied to the
*predicted* error regions. A missed region gets a positive click and an extra region gets a negative one.
The click goes to the point farthest from the region's boundary. If no predicted error passes the threshold,
the corrector **abstains**, which is how it leaves good masks alone (RQ4). SAM 3 is never trained.

| Input channel (crop, S x S) | |
|---|---|
| 0-2 | RGB, ImageNet-normalized |
| 3 | `tanh(logit / 4)` from `segmenter.to_full_res` |
| 4 | `logit > 0` (the mask that is evaluated) |
| 5 | prompt box |
| 6 | valid (0 outside the image) |
| 7-8 | positive / negative click Gaussians (sigma 3 px); empty in v0, reserved for multi-step |

The crop is the prompt box grown by 50% of its size on every side, resized to 128 x 128 without keeping the
aspect ratio. Loss: BCE + soft Dice per (sample, channel), with pixels outside the image ignored.

## Only real code, no stand-ins

v0 uses only code that exists in the repo: SAM 3 (P1), records and the LVIS-leakage split (P2), IoU,
Boundary IoU and method names (P3). P4's perturbations and baselines and P3's ideal-click simulator do not exist
yet, so v0 does **not** use perturbed prompts, baselines or ideal clicks. Training data and evaluation both use
**clean GT boxes**: the corrector learns SAM 3's own mistakes. When those modules land, they plug in as follows:

| When this lands | Change |
|---|---|
| P4 `perturb/` | add perturbed prompts in `samples.py` and `rollout.py` (call P4's generator and `describe()`) |
| P4 `baselines/` | add `b1_random_k`, `b3_box_refit`, ... as methods in `rollout.py` |
| P3 ideal-click simulator | add `oracle_clicks` as the upper bound, and click states to the training data (then multi-step) |

## Sample format (for P1 / P4)

One sample = one frozen-segmenter state for one prompt. A run directory holds:

```
$RUNS_ROOT/<samples run>/
  shard_XXXX.npz      # np.savez_compressed, no pickles; arrays stacked over the shard's samples:
                      #   rgb uint8 [N,3,S,S], logits fp16 [N,S,S], gt bool [N,S,S], valid bool [N,S,S],
                      #   box bool [N,S,S], clicks float32 [N,4,3] = (u, v, label) in crop pixels, label -1 = empty
  meta.jsonl          # one row per sample: shard, row, sample, obj_id, image_id, category, size_bin, origin,
                      #   box, window (crop xyxy in image pixels), points (image coords), n_clicks,
                      #   iou (full-res, vs GT), pred_iou, crop_fp, crop_fn
  records.jsonl       # ObjectRecords of the images used (P2's builder)
  manifest.json       # settings, chosen image ids, LVIS-val exclusion counts, input hashes
```

Training images: 3,000 COCO train2017 images sampled **after removing LVIS-val images** (CLAUDE.md 4.5), at most
8 objects per image. Checkpoint selection uses 10% of those images held out, so the demo val images are only
seen in the final eval.

## Running on GreatLakes

From the repo root, after committing (manifests record the commit and a `dirty` flag). One job runs all three
steps, so there is one queue wait:

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
sbatch --account=eecs542f26s001_class --time=03:00:00 --mem=64G --job-name=p5_all \
    slurm/run_gpu.sbatch scripts/p5_run_all.py --paths configs/paths/greatlakes.yaml --suffix 0
```

The run ids are `demo_p5_{samples,train,eval}_0`. If the job dies, resubmit the same command: finished steps
are skipped. Each step can also be run on its own (`scripts/p5_make_samples.py`, `p5_train_corrector.py`,
`p5_eval_corrector.py`; see their docstrings). Logs: `p5_all-<jobid>.out` in the repo root.

If `objects_val.jsonl` on scratch is not readable (it was mode 600 on 2026-10-25), the eval rebuilds it with
P2's builder from the COCO val annotations and `splits_demo.json`. The rebuild was checked to be byte-identical
(sha256 86f4cb3d...), and the manifest records the source.

## Outputs

| Run | Files |
|---|---|
| `demo_p5_samples_0` | shards, `meta.jsonl`, `records.jsonl`, `manifest.json` |
| `demo_p5_train_0` | `best.pt`, `last.pt`, `history.jsonl`, `final_val.json` (FP/FN AUROC, error-map IoU, click hit rate, abstain rate, threshold sweep), `training_curves.png` |
| `demo_p5_eval_0` | `results.parquet` (one row per object x method x step), `summary_final.csv`, `summary_iou_by_step.csv`, `corrector_vs_none.png` |

`summary_final.csv` has one row per (group, method). The groups are all objects, objects starting below / above
IoU 0.75, and each size bin. For each it gives mean IoU before and after, ΔIoU, harm %, improved %, % of objects
clicked, % positive clicks, and success@0.5/0.75 before and after. Harm uses the shared definition
`iou_final < iou_step0 - 0.01`.

If `final_val.json` shows `abstain_rate` near 1, the threshold 0.5 is too strict. Rerun only the eval with a
threshold from `threshold_sweep`:

```bash
sbatch --account=eecs542f26s001_class --time=01:00:00 --job-name=p5_eval slurm/run_gpu.sbatch \
    scripts/p5_eval_corrector.py --paths configs/paths/greatlakes.yaml \
    --run-id demo_p5_eval_0_t03 --train-run demo_p5_train_0 --threshold 0.3
```

Known limits of v0: one correction step; 128 px crops lose fine boundary detail on large objects; one seed;
clean GT boxes only (the 20.9% of objects below IoU 0.75 in the RQ0 ceiling run are where it can help).
