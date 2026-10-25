# Demo sprint plan (proposal presentation, 6-7 min)

**Status:** draft v0.1, 2026-10-23. Budget: about 1 working day per person, spread over 2.5 days (roughly 3h + 3h + 2h each). Srikrishnan also does the setup tonight.
Companion docs: `CLAUDE.md` (project context), `docs/interfaces.md` (formats), `docs/container.md` (image build).

## Goal

Show **preliminary RQ0 numbers on real data**, plus a first look at the corrector if it works. Minimum viable demo = items 1-4 below. Item 5 is a stretch.

1. Failure rates of real prompts: SAM 3 text detections and detector boxes -> SAM (share of objects below IoU 0.5 / 0.75 / 0.9).
2. Sensitivity curves: IoU vs. box shift / scale / truncation severity.
3. Headroom: IoU after k ideal clicks and clicks-to-85/90 IoU (NoC) for the failures.
4. Baselines vs. the ideal-click upper bound: box refit, random clicks, uncertainty click.
5. Stretch: corrector v0 (error-map predictor), and whether clicking at its prediction beats the heuristics.

## Demo-only simplifications (so we finish in 2.5 days)

- **No feature cache.** The Segmenter runs the image encoder live (`cache=None`). The cache (`docs/interfaces.md` section 4, issues O7/O8) is built in week 2. Demo scale is small, so this is cheap.
- **Fixed demo subset:** 500 COCO val2017 images, seed 0, listed in `data/splits_demo.json` (P2). That is roughly 2-3k objects after filtering.
- **Detections only for categories present in the image's GT** (issue O9). No false-positive analysis.
- **Perturbations:** shift, scale, side truncation only. Distractor / wrong-instance are later.
- **Defaults adopted for open issues unless someone objects before Day 1 starts:** O1 (drop only if box w < 32 *and* h < 32), O2 (image-level split schema), O3 (`describe(prompt, obj, neighbors)`), O4 (add Δ columns to the results row). O6 gets verified by Srikrishnan's smoke test.
- **Method names (P3 registry):** `b0_none`, `b1_random_k`, `b3_box_refit`, `b3_unc_click`, `oracle_clicks`, `corrector_v0`.
- **Run IDs:** `demo_<experiment>_<n>`. Outputs go to `$RUNS_ROOT/<run_id>/`, never into git except summary figures/tables.
- **Dummy segmenter click response** (Srikrishnan builds it): the output is the GT mask eroded by a fixed amount; a positive click inside the GT adds a disk of GT pixels around it; a negative click outside the GT removes a disk of non-GT pixels. This lets P3 and P4 test click logic on a laptop.

## Timeline and gates

| When | What |
|---|---|
| **Day 0 (tonight)** | Srikrishnan: scaffold on `main`, data download job running, container build + upload started. Others: read `CLAUDE.md` and `docs/interfaces.md`, raise objections to the defaults above. |
| **Day 1** | Everyone builds their module against the dummy segmenter on a laptop. Merge to `main` by end of day. Srikrishnan delivers the SAM wrapper and Slurm templates. |
| **Day 2** | Integration. First real numbers on ~100 images by midday, then scale to 500. Generate v0 training samples, train v0. **Freeze experiments at the end of Day 2.** |
| **Day 3 (half day)** | Figures, slides, two timed rehearsals. No new experiments. |

**Gates**
- **G0 (end of tonight):** scaffold merged; `sbatch` download submitted; container build started.
- **G1 (Day 1 midday):** SAM 3 smoke test result (text + box + point prompts on one image, on SPGPU). If the interactive path doesn't work by Day 1 evening, **fall back to SAM 2.1** for the demo (no `detect_text`; text-path numbers dropped, detector-box numbers kept).
- **G2 (Day 1 end):** every module merged with passing CPU tests on the dummy. Srikrishnan runs one image through the whole chain.
- **G3 (Day 2 midday):** first real RQ0 numbers on ~100 images look sane (clean GT-box IoU is high; perturbed IoU drops as severity grows).

## Per-person plan

### Srikrishnan (P1 - segmenters & infra)
**Tonight (~3h)**
1. Scaffold the repo: `pyproject.toml`, `configs/paths/{local,greatlakes}.yaml`, `src/promptcorrect/` package, `types.py` from `docs/interfaces.md`, `Segmenter` base class and `DummySegmenter` (with the click response above), CPU test skeleton. Push to `main`.
2. GreatLakes: fill in `srikrish` in `CLAUDE.md` and `configs/paths/greatlakes.yaml`; create the shared dir and share it with teammates via `setfacl` (not group write); submit the download job on `standard`: COCO val2017 + annotations first, then train2017, LVIS v1 annotations, SAM 2.1 checkpoint, SAM 3 weights (your HF token), Grounding DINO.
3. Container: test `apptainer build --fakeroot` on GreatLakes; otherwise build in WSL and start the `rsync` upload overnight.

**Day 1 (~3h)**
- SAM 3 smoke test on SPGPU (G1). Record what works in `CLAUDE.md` section 12.
- `segmenters/sam3.py`: `set_image` (live encoder), `predict` with box and points, `detect_text`, `to_full_res`; round-trip unit test for box coordinates (issue O13). Measure time and memory.
- `slurm/run_gpu.sbatch` and `slurm/run_cpu.sbatch` templates; post one working example in the team chat so others can launch jobs.

**Day 2 (~3h)**: launch/babysit GPU jobs for P2, P3, P4; build and run the v0 sample-generation job with P5 and P4; fix wrapper issues fast.
**Day 3 (~2h)**: slides: problem, pipeline/compute, plan and risks.

### Shresth (P2 - data & detectors)
**Day 1 (~3h)** (annotations are small: download `annotations_trainval2017.zip` locally)
- `data/records.py`: `ObjectRecord` builder from COCO annotations: filtering, stratification fields (`size_bin`, `thinness`, `n_neighbors` per the doc), RLE as strings in JSONL. Tests on a few hand-checked annotations.
- Write `data/splits_demo.json` (500 val images, seed 0) and `data/objects_val.jsonl`. Draft the `splits.json` schema from O2.
- `detectors/`: Grounding DINO wrapper via `transformers` returning `(box, score)` per phrase. Time-box LLMDet to about 2h; if it fights the install, skip it for the demo.

**Day 2 (~3h)**: run the detector and SAM 3 text (`Segmenter.detect_text`) on the 500 images, GT-present categories only, writing `detections/<model>/val.jsonl`. Hand the files to P3.
**Day 3 (~2h)**: slides: datasets, subset, detectors; start the COCO<->LVIS category mapping if time remains.

### Divya (P3 - eval & oracle)
**Day 1 (~3.5h)** (all CPU, all on the dummy)
- `eval/metrics.py`: `iou`, `boundary_iou` on boolean masks at original resolution. Tests on hand-made masks.
- `eval/matching.py`: Hungarian `match(preds, gts)` on mask IoU; document its output (pairs with IoU, unmatched predictions, unmatched GTs); ignore predictions that overlap filtered objects (O1).
- `eval/oracle_clicks.py`: ideal-click simulator (click at the point farthest from the boundary in the largest error region; positive for false negative, negative for false positive); returns IoU per click and NoC@85/90. Test on the dummy.
- `eval/aggregate.py`: write the results Parquet (`ResultRow`) and compute failure rates, success@0.5/0.75/0.9, harm rate.

**Day 2 (~3h)**: run the oracle study on the real failures (with P1 for Slurm); produce the RQ0 table plus 3 figures: failure-rate bars, IoU vs. number of clicks, NoC histogram. Compare baselines vs. oracle with P4.
**Day 3 (~2h)**: slides: RQ0 results and headroom; assemble and edit the final deck.

### Aman (P4 - perturbations & baselines)
**Day 1 (~3.5h)** (all CPU, all on the dummy)
- `perturb/box.py`: shift, scale, side-truncation with the knobs/levels in `CLAUDE.md`/the description, parameterized relative to the GT box. Tests: `box_iou` decreases as level grows.
- `perturb/severity.py`: `describe(prompt, obj, neighbors)` returning `box_iou`, `coverage`, `contamination`, `delta` (NaN for box fields when there is no box).
- `baselines/`: `box_refit` (refit the box to the current mask and rerun), `random_clicks`, `uncertainty_click` (centre of the largest region where |logit| is small).

**Day 2 (~3h)**: sensitivity sweep script (object x factor x level -> ResultRow); run on GPU; plot IoU vs. severity per factor. Run the baselines on the real failures with P3. Help generate v0 training samples with perturbations.
**Day 3 (~2h)**: slides: sensitivity curves and baselines.

### Sharvani (P5 - corrector model)
**Day 1 (~3.5h)**
- Define the v0 sample format and write it in `docs/` (proposal: shards of samples with `obj_id`, the prompt as JSON and the low-res logits; the dataloader reads the RGB image and GT RLE from COCO). Post it to P1 and P4 by midday so they can generate data.
- `corrector/model.py`: small conv encoder-decoder; inputs RGB (downscaled), mask logits, box mask, positive/negative click heatmaps; output a 2-channel error map (false positive / false negative). `corrector/losses.py`: BCE + Dice. `corrector/dataset.py`, `corrector/train.py`. Verify it overfits one batch of fake tensors.
- Read the RITM/SimpleClick click simulation, since P3 and P5 use the same click convention.

**Day 2 (~3h)**: with P1/P4, generate samples from about 1.5k COCO train images with perturbed prompts; train v0 for 20-30 min on SPGPU; evaluate: error-map IoU/AUROC on perturbed val objects, then click at the predicted error peak, rerun SAM, and compare against `b3_box_refit` and `b1_random_k`. **If this doesn't work by Day 2 evening, show the architecture and training curves honestly.**
**Day 3 (~2h)**: slides: method/architecture and v0 result.

## Slide split (7 slides, about 1 min each)

| Slide | Owner |
|---|---|
| 1. Problem + failure example | Srikrishnan |
| 2. Approach / pipeline | Sharvani |
| 3. Data, subset, detectors | Shresth |
| 4. RQ0: failure rates + headroom | Divya |
| 5. Sensitivity curves + baselines | Aman |
| 6. Corrector v0 (or status) | Sharvani |
| 7. Plan, go/no-go, risks | Srikrishnan |

Each person presents their own slide(s). Divya assembles the deck.

## Rules for the sprint

- Work against the dummy first; never wait for someone else's code. If blocked, say so in the team chat within the hour.
- Small PRs, merged the same day. Pull `main` at the start of each session.
- Don't change `docs/interfaces.md` contracts without the owner and consumers agreeing.
- Rough GPU budget for the whole demo: a few GPU-hours (SPGPU). Log what you use.
- Report plainly what ran and what didn't. Do not put numbers in slides that didn't come from a saved results file.

## Contingencies

| If | Then |
|---|---|
| SAM 3 interactive path is broken at G1 | Use SAM 2.1 for the demo; drop text-path numbers |
| Container isn't ready on GreatLakes | Use a temporary conda env in `$HOME`/scratch for the demo; keep the container for week 2 |
| LLMDet won't install | Grounding DINO is the demo detector |
| Downloads are slow | Val + annotations are enough for items 1-4; train images are only needed for v0 |
| Corrector v0 doesn't beat baselines | Present the architecture, training curves and an honest status; the RQ0 results carry the demo |
