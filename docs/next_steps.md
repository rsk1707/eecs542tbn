# Next steps: final stretch (Nov 21 → freeze Dec 5)

**Written 2026-11-19 by P1 after merging `p3/eval` and `p2/lvis-data-preparation` into `main`.**
Agents: read `CLAUDE.md` first, then this file, then your own role section. If this file and `CLAUDE.md` disagree,
`CLAUDE.md` wins; raise the conflict in a PR comment instead of silently picking one.

## 0. Schedule

| Date | What |
|---|---|
| Nov 21 – 25 | **Sprint (the next 4-5 days).** Everyone pulls `main`, does the Section 3 tasks. Gate S at the end (Section 4). |
| Nov 26 – Dec 1 | v1 trained and evaluated; RQ2/RQ3 runs if the gate allows. |
| Dec 2 – 4 | Final runs, final tables and figures, report text. No new methods. |
| **Dec 5** | **Freeze.** Nothing new on `main` except fixes to figures/text. Every number in the poster comes from a run that is in `results/runs_index.csv`. |
| Dec 5 – 7 | Poster and a timed dry run. Poster session Dec 7. |

## 1. State of `main` (as of the merge, 2026-11-19)

- `main` now contains everything pushed so far: SAM 3 wrapper, data pipeline, Grounding DINO / SAM 3 text detections, RQ0 runner, ideal-click
  simulator, rule baselines, sensitivity sweep, corrector v0, P3's baselines-vs-ideal-clicks results (`results/demo_rq0_baselines_1/`) and P2's
  LVIS / leakage-safe preparation code (`docs/p2_rq3.md`).
- **Tests on the merged result, run on a Windows laptop (CPU torch):** 150 passed, 6 skipped, 2 failed. The 2 failures are in
  `tests/p2/test_splits_prepare.py` (`test_rooted_paths`, `test_new_files_get_normal_permissions`). They assert POSIX path and file-mode behaviour
  and are expected to pass on Linux. **They have not been run on GreatLakes yet**: whoever next logs in should run `pytest -q` there once and
  report the result in the team channel.
- **Not verified:** production LVIS preparation (needs the reviewed category file), scratch access after the break, and anything that needs a GPU
  since Oct 26. Scratch deletes files not accessed for about 60 days; the last accesses were around Oct 25-26, so everything is still there if
  touched before about Dec 20. Do a `ls`/`find -newer` pass on `$PC_ROOT` in the first session.

### What the results say so far (all on the 2,937 demo val objects; see `CLAUDE.md` section 12 for the full entries)

1. Failure is real: with a clean GT box SAM 3 gets mean IoU 0.817 and 20.9% of objects are below 0.75. Real prompts are worse: Grounding DINO
   boxes 0.529 (65.4% detected), SAM 3 text 0.711 (88.7% detected). For Grounding DINO the main loss is missed objects; for SAM 3 text it is mostly
   masks that are detected but wrong. Do not claim "missed detections dominate" for both.
2. No rule baseline helps: on the 614 failing GT-box masks, 5 steps of box refit / random / uncertainty click / multimask end at 0.58-0.605 vs a
   0.606 start, and they harm many masks (4-49% depending on the rule).
3. Ideal clicks help: 0.606 → 0.669 after 1 click, 0.708 after 5 (no logit feedback), 0.785 after 5 with logit feedback.
4. Corrector v0 (one click, error-map U-Net, no logit feedback): 0.606 → 0.627 on the failures, i.e. about **a third of one ideal click**
   (0.669). But it clicks on 100% of objects and harms 16.1% of the 2,323 objects that started at or above 0.75. **v0 has no working abstain
   decision; fixing that is the main job of v1.**

## 2. Target

**One table that answers RQ1 and RQ4**: rows = no correction, the four rule baselines, ideal clicks, corrector v0, corrector v1; columns = mean IoU
before/after, ΔIoU, % below 0.75 before/after, harm rate, % clicked, for (a) objects starting below 0.75 and (b) objects starting at or above 0.75,
on the same objects and the same prompt sources. It is produced by P3's code from `ResultRow`s, nothing else.

Priority order (cut from the bottom):

1. **MUST. Corrector v1** = v0 plus (a) a **learned click-or-abstain decision** (for example a head that predicts the IoU gain of the click, or the
   current mask's own IoU; click only above a threshold chosen on held-out *training* images), (b) training on **perturbed boxes** (synthetic
   errors, P4) and not only clean GT boxes, (c) the previous logits fed back into SAM 3 as `Prompt.mask_logits` (this was worth +0.045 for ideal
   clicks). Evaluate against the baselines with harm reported.
2. **SHOULD. RQ2**: v1 trained on synthetic only vs real mined failures vs mixed, tested on real detector failures.
3. **SHOULD. RQ3**: v1 on LVIS categories the corrector never saw vs seen ones (requires the reviewed category file).
4. **CUT FIRST** (do not start unless everything above is done): multi-step rollouts, box-adjustment head, SAM 2.1, DINOv3, ThinObject/DIS5K, SA-Co,
   SAM 3 Agent baseline, learned stopping policy.

**Go/no-go (proposed, the team must confirm in the channel by Nov 25):** v1 goes into the headline claim if, on objects starting below 0.75, its
mean ΔIoU after one click is larger than the best rule baseline's and its harm rate on objects starting at or above 0.75 is below 5%. Compare
against ideal clicks **without** logit feedback when v1 does not use feedback, and with feedback when it does; state which in the table. If v1
does not meet this by Dec 1, the report states that honestly (RQ1 answer: no, with the measured gap) and the poster leads with RQ0 + baselines +
where v0/v1 fall short. Never report a number that did not come from the shared eval code.

## 3. Work split

Each person stays inside their own directory (`CLAUDE.md` section 5). Changing a contract (`types.py`, `docs/interfaces.md`) needs agreement from
every consumer. Small PRs into `main`, one reviewer who skims it.

### P1 – Srikrishnan (segmenters, infra, GPU budget)
1. Nov 21: run `pytest -q` on GreatLakes against `main`, post the result. Check the shared venv, scratch ACLs for all five uniqnames and that the
   data and cache directories are still readable (see the `ls` note above).
2. Nov 21-22: help P2 submit the LVIS draft-review and prepare jobs if scratch access is a problem.
3. Provide GPU runs on request for P5 (v1 sample generation, evaluation) and P3 (adding v0/v1 to the table). Keep a running GPU-hour tally in the
   channel (class budget is limited; do not rebuild the venv, reuse it).
4. If time: the feature cache (`docs/interfaces.md` section 4) only if a profile shows SAM 3 encode time is the bottleneck; otherwise skip it.
5. Own the freeze: Dec 5 tag `freeze-2026-12-05` on `main` and list the run ids behind each poster number in `results/runs_index.csv`.

### P2 – Shresth (data and detectors)
1. Nov 21-23: complete the LVIS category review (`docs/p2_rq3.md`): 79 of 1,203 categories match by synset; the other 1,124 need a semantic
   decision (for example LVIS `bulldog` overlaps COCO `dog` and must not be called unseen). Agree the decisions with P3 and P5 in the channel and
   commit only the small reviewed JSON (versioned), not the large records.
2. Nov 23-25: submit the CPU prepare job (leakage-safe COCO train, COCO val, LVIS val). Confirm `objects_lvis_val.jsonl` exists with `lvis:` IDs and
   report the unseen-category counts per size bin.
3. After that: **mine real failures on COCO train** (detector or SAM 3 text mask with IoU about 0.3-0.85 vs GT), keeping the actual prompt (box /
   text) and the detector provenance for each object. Needs from P3: a way to persist the matched detector box per object (see
   `docs/p2_rq3.md`, "Remaining owner handoffs"). **Never mine from val or LVIS-val images.**
4. Add an image resolver for LVIS detector exports (LVIS uses `coco_url`, the detector loader expects `file_name`).

### P3 – Divya (eval, oracle, report)
1. Nov 21-23: add `corrector_v0` (already a method name) and `corrector_v1` to the shared eval so they enter the same `ResultRow` table as the
   baselines and the oracle. Do not fork the metric code; P5 calls your functions.
2. Nov 21-22: agree with P2 how to persist the matched detector box / prompt per object for failure mining, and make the RQ0 ignore-mask helper
   work for `lvis:` IDs (it hardcodes `coco:`).
3. Nov 24-26: produce the Section 2 table for v0 immediately (it exists), then v1 as soon as P5 delivers; report gain and harm on both objects
   starting below and at or above 0.75, and by size bin.
4. Write the go/no-go summary (RQ0) and the RQ4 harm paragraph. Keep editing the report as owner of the final document.

### P4 – Aman (perturbations and baselines)
1. Nov 21-23: **training-data variants for P5**: generate perturbed-box training samples in P5's sample format (`docs/p5_corrector_v0.md`, "Sample
   format"), using `sample_perturbation` and `describe()`. Start with `train_data: synthetic` over the 3,000 leakage-safe COCO train images
   (LVIS-val removed); each sample records the perturbation type and severity. Deliver a small shard to P5 by Nov 22 so v1 training can start.
2. Nov 24-27: the `real` / `mixed` / `matched` variants from P2's mined failures (RQ2). `matched` = same count and object set as `real` but with
   synthetic perturbations, so the comparison is fair.
3. Optional: run the baselines on perturbed boxes (not only clean and detector boxes) so the synthetic-to-real comparison has all methods.

### P5 – Sharvani (corrector)
This is the critical path.
1. Nov 21-22: v1 architecture change: add the click-or-abstain output and make `rollout.py` feed previous logits back. If the P4 samples are late,
   start with random shift / scale / truncate boxes generated by P4's functions on the fly; do not wait.
2. Nov 22-25: train v1 on synthetic perturbed boxes; choose the abstain threshold on held-out **training** images (never on the val objects).
3. Nov 25-27: evaluate through P3's eval code on the same 2,937 demo val objects and report gain + harm; then on detector boxes.
4. Nov 27 onward: train the RQ2 variants (synthetic / real / mixed / matched) and, if the category file exists, run RQ3.
5. Write down the v1 design and results in `docs/p5_corrector_v1.md`.

## 4. Gates

| Gate | When | Passes if |
|---|---|---|
| **S** (sprint exit) | Nov 25 | P3 tests green on GreatLakes; P4 synthetic samples delivered; v1 training has started; LVIS category review decided or explicitly postponed. |
| **G-v1** | Nov 28 | v1 evaluated on the demo val objects through P3's table; go/no-go in Section 2 answered. |
| **G-data** | Dec 1 | RQ2 variants evaluated; RQ3 evaluated or dropped. After this, no new experiments. |
| **Freeze** | Dec 5 | See Section 0. |

If a gate fails, say so in the channel the same day and cut from the bottom of the Section 2 list instead of stretching the schedule.

## 5. Working rules for the final stretch

- Every run: a config file + git commit hash + a seed, results in `$RUNS_ROOT/<run_id>/`, one row in `results/runs_index.csv`. Run ids:
  `<experiment>_<n>`.
- Compare methods like for like: same objects, same step-0 masks, same step budget, same logit-feedback setting (state it in the table).
- Validation objects are used only for the final evaluation. Anything chosen on data (threshold, checkpoint) is chosen on held-out *training* images.
- LVIS-val images never enter corrector training (they overlap COCO train).
- GPU jobs: `slurm/run_gpu.sbatch` on SPGPU with `--account=eecs542f26s001_class`. Post the job id and GPU-hours in the channel. CPU work
  (perturbation generation, LVIS preparation, metrics) goes on the standard partition.
- Open a PR for anything touching another role's code, or ask the owner. Add a line to `CLAUDE.md` section 12 for every finished run or decision.
- Windows teammates: two tests in `tests/p2/test_splits_prepare.py` assert POSIX path/mode behaviour and fail on Windows; they are expected to
  pass on Linux (to be confirmed on GreatLakes).
