# Corrector v0 (P5, demo sprint item 5)

**Status:** code ready, CPU tests pass on the dummy (`tests/p5/`). Not yet run on GPU.
**Owner:** Sharvani (P5). Code: `src/promptcorrect/corrector/`. Scripts: `scripts/p5_*.py`.

## What v0 is

A small U-Net (about 1.9M parameters) looks at an object-centric crop of the image, the frozen SAM 3 mask and
the prompt, and predicts a 2-channel **error map**: false positive (mask but not GT) and false negative
(GT but not mask). A click is then placed with the same rule as the ideal-click simulator, but on the
*predicted* error regions. A missed region gets a positive click and an extra region gets a negative one.
The click goes to the point farthest from the region's boundary. If no predicted error passes the threshold,
the corrector **abstains**, which is how it can leave good prompts alone (RQ4). SAM 3 is never trained.

| Input channel (crop, S x S) | |
|---|---|
| 0-2 | RGB, ImageNet-normalized |
| 3 | `tanh(logit / 4)` from `segmenter.to_full_res` |
| 4 | `logit > 0` (the mask that is evaluated) |
| 5 | prompt box |
| 6 | valid (0 outside the image) |
| 7-8 | positive / negative click Gaussians (sigma 3 px) |

The crop is the prompt box grown by 50% of its size on every side, resized to 128 x 128 without keeping the
aspect ratio. Loss: BCE + soft Dice per (sample, channel), with pixels outside the image ignored.

## Sample format (for P1 / P4)

One sample = one frozen-segmenter state for one prompt. A run directory holds:

```
$RUNS_ROOT/<samples run>/
  shard_XXXX.npz      # np.savez_compressed, no pickles; arrays stacked over the shard's samples:
                      #   rgb uint8 [N,3,S,S], logits fp16 [N,S,S], gt bool [N,S,S], valid bool [N,S,S],
                      #   box bool [N,S,S], clicks float32 [N,4,3] = (u, v, label) in crop pixels, label -1 = empty
  meta.jsonl          # one row per sample: shard, row, sample, obj_id, image_id, category, size_bin, origin,
                      #   pert_type, pert_level, perturbation, box, window (crop xyxy in image pixels), points
                      #   (image coords), n_clicks, box_iou_in, iou (full-res, vs GT), pred_iou, crop_fp, crop_fn
  records.jsonl       # ObjectRecords of the images used (P2's builder)
  manifest.json       # settings, chosen image ids, LVIS-val exclusion counts, input hashes, stand-in note
```

Per object, the prompts are the clean GT box plus 2 perturbed boxes (shift / scale / truncate, level uniform in
[0.05, 0.5]). With probability 0.5 a prompt also contributes a second sample after 1-2 ideal clicks, so the
click channels are not always empty. Training images are 1,500 COCO train2017 images sampled **after removing
LVIS-val images** (CLAUDE.md 4.5), with at most 8 objects per image.

## Temporary stand-ins (to replace with P4's code)

P4's perturbations, `describe()` and baselines were still stubs, so `corrector/standins.py` holds minimal
versions: the 3 perturbations, `box_iou`, and the `b1_random_k` / `b3_box_refit` rules. Every manifest records
this. **Before any number goes into the report, switch to P4's implementations and rerun.** For the demo
slide, label the baseline numbers "preliminary, stand-in baselines".

## Running on GreatLakes

Run from the repo root, after committing your changes (the manifests record the commit and a `dirty` flag):

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
P=configs/paths/greatlakes.yaml
A=--account=eecs542f26s001_class

# 0. smoke test (~10 GPU-min): tiny versions of all three steps, chained
j1=$(sbatch --parsable $A --time=00:30:00 --mem=64G --job-name=p5_smoke_samples slurm/run_gpu.sbatch \
     scripts/p5_make_samples.py --paths $P --run-id smoke_p5_samples --n-images 1500 --max-images 20)
j2=$(sbatch --parsable $A --time=00:20:00 --dependency=afterok:$j1 --job-name=p5_smoke_train slurm/run_gpu.sbatch \
     scripts/p5_train_corrector.py --paths $P --run-id smoke_p5_train --samples smoke_p5_samples --epochs 2)
sbatch $A --time=00:20:00 --dependency=afterok:$j2 --job-name=p5_smoke_eval slurm/run_gpu.sbatch \
     scripts/p5_eval_corrector.py --paths $P --run-id smoke_p5_eval --train-run smoke_p5_train --max-images 5

# 1-3. the real runs, chained
j1=$(sbatch --parsable $A --time=03:00:00 --mem=64G --job-name=p5_samples slurm/run_gpu.sbatch \
     scripts/p5_make_samples.py --paths $P --run-id demo_p5_samples_0 --n-images 1500)
j2=$(sbatch --parsable $A --time=01:30:00 --mem=64G --dependency=afterok:$j1 --job-name=p5_train slurm/run_gpu.sbatch \
     scripts/p5_train_corrector.py --paths $P --run-id demo_p5_train_0 --samples demo_p5_samples_0)
sbatch $A --time=03:00:00 --dependency=afterok:$j2 --job-name=p5_eval slurm/run_gpu.sbatch \
     scripts/p5_eval_corrector.py --paths $P --run-id demo_p5_eval_0 --train-run demo_p5_train_0
```

Logs go to `<job-name>-<jobid>.out` in the repo root. Delete the `smoke_p5_*` run directories after checking
them. Run ids are never reused: a script refuses to overwrite an existing run.

## Outputs

| Run | Files |
|---|---|
| `demo_p5_samples_0` | shards, `meta.jsonl`, `records.jsonl`, `manifest.json` |
| `demo_p5_train_0` | `best.pt`, `last.pt`, `history.jsonl`, `final_val.json` (FP/FN AUROC, error-map IoU, click hit rate, abstain rate), `training_curves.png` |
| `demo_p5_eval_0` | `results.parquet` (one row per object x prompt x method x step), `summary_*.csv`, `iou_by_step.png`, `gain_and_harm.png` |

The eval summary tables are: `final_{all,clean,perturbed,failures<0.75}` (mean IoU before and after, ΔIoU, harm %,
improved %, success@0.5/0.75, steps used), `iou_by_step_*` (last value carried forward after a method stops),
and `delta_by_type_level`. Harm uses the shared definition `iou_final < iou_step0 - 0.01`.

Known limits of v0: 128 px crops lose fine boundary detail on large objects; checkpoint selection uses
held-out *train* images, so the demo val images are never seen; one seed.
