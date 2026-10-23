# CLAUDE.md — Shared project context (EECS 542 / CSE542)

> **For every team member's AI agent: read this whole file before doing anything.**
> It is the single source of truth for project goals, scope, roles, interfaces and conventions.
> Five people are working in parallel with five separate agents; this file is how we avoid drift.
> If something here is wrong or outdated, fix it via a small PR (see "Keeping this file current").

Last updated: 2026-10-23 (project kickoff / demo sprint)

---

## 1. What we are building

**Title:** Learning to Correct Prompts for Frozen Segmentation Foundation Models

Promptable segmenters (SAM 3) turn a prompt (box, clicks, or text) into a mask. Output quality depends heavily on the prompt, and in automatic pipelines (detector -> segmenter) nobody fixes a bad prompt. We train a **small corrector network** that looks at the frozen segmenter's current mask, uncertainty and prompt, and proposes **corrective clicks / box adjustments** that are fed back to the frozen segmenter. **SAM 3 is never fine-tuned.**

Team: Divya Shekar, Shresth Tiwary, Sharvani Sinha, Srikrishnan Ravichandran, Aman Soni.

### Source-of-truth documents
| Document | Status |
|---|---|
| `EECS542_Project_Proposal.pdf` (what we submitted) | **Canonical.** RQ0–RQ4 below come from here. |
| `CV_Project_Description.pdf` (AI-written background) | Background/ideas only. Its H1–H6, stopping policy, SAM 3 Agent baseline etc. are **stretch**, not commitments. |
| `CV_Project_Plan.pdf` (AI-suggested split) | Superseded by sections 3–5 here (timeline is now 6 weeks, not 10). |

If these disagree with this file, **this file wins**.

## 2. Research questions (the thing we must answer)

- **RQ0 (preliminary):** How often do current prompts give poor masks (SAM 3 text detections; independent-detector boxes -> SAM 3)? How much correction is needed (clicks to reach 85/90 IoU with *ideal* clicks; IoU gap)? Separate prompt-fixable failures from model-limit failures. Includes a **go/no-go** check on headroom.
- **RQ1:** Does a learned corrector beat no correction, random extra clicks, and simple rules (e.g. refit box to current mask)?
- **RQ2:** Does training on synthetic prompt errors transfer to real detector failures?
- **RQ3:** Does it generalize to object categories the corrector never saw (COCO-train -> LVIS non-COCO categories)?
- **RQ4:** Does it leave good prompts alone (harm rate on clean prompts)?

Metrics: mask IoU, Boundary IoU, share of objects above IoU 0.75 (and 0.5/0.9), harm rate (% of objects made worse), ΔIoU vs. perturbation severity, #clicks / latency.

## 3. Scope

**In scope (commitments):** RQ0–RQ4 on COCO (train) / LVIS (unseen-category eval), SAM 3 as the frozen segmenter, one independent detector (LLMDet preferred, Grounding DINO as fallback), synthetic perturbations + mined real failures, baselines (no correction, random clicks, SAM multimask, box-refit / uncertainty-click heuristics), ideal-click oracle as upper bound.

**Stretch (only if ahead of schedule):** learned stopping policy, SAM 3 Agent baseline, SAM 2.1 transfer run, DINOv3 features ablation, ThinObject-5K / DIS5K, SA-Co/Gold, end-to-end gradients.

**Out of scope:** RefCOCO, interactive-segmentation benchmarks (GrabCut/DAVIS/SBD), video, fine-tuning any foundation model.

## 4. Key design decisions (do not silently change)

1. Foundation models (SAM 3, detector) are **frozen**. Only the corrector trains.
2. All code talks to the segmenter through the **`Segmenter` interface** (`src/promptcorrect/segmenters/base.py`). A `DummySegmenter` (GT mask, slightly eroded/dilated) lets everyone develop and unit-test **locally without a GPU**. SAM 3 and SAM 2.1 are drop-in implementations.
3. Primary evaluation unit is **one object**: a (possibly corrupted / detector-produced) box -> mask -> IoU vs GT. Text-prompt evaluation with Hungarian matching of predicted instances to GT is a secondary setting.
4. Perturbations are parameterized **relative to the GT box** (shift as fraction of box size, scale, truncation, etc.) and each has a severity measure, so synthetic and real errors are comparable.
5. **LVIS leakage:** some LVIS-val images are COCO-train2017 images. Exclude LVIS-val image IDs from corrector training or RQ3 is contaminated. "Unseen" means unseen *by the corrector*, not by SAM 3.
6. Real failures = masks with IoU roughly 0.3–0.85 vs GT from SAM 3 text / detector boxes.
7. Every number in the report must come from a `ResultRow` written by the shared eval code (section 7). No ad-hoc numbers from notebooks.

## 5. Roles

Roles are **self-selected**. Each role owns a directory under `src/promptcorrect/` and is the only one who changes its interfaces (others open an issue/PR comment). Everyone writes the report section for their own part; the P3 owner also edits the report.

### Role claim board (edit the Owner column to claim a role)

| Role | Owner | Owns | Leads | Est. effort* |
|---|---|---|---|---|
| **P1 — Segmenters & infra** | Srikrishnan | `container/`, GreatLakes setup, `segmenters/`, feature cache, `slurm/`, GPU budget tracking | supports everyone | 3–4 days |
| **P2 — Data & detectors** | Shresth | `data/`, `detectors/`, COCO/LVIS records, seen/unseen split, mined real failures | RQ3 | 2–3 days |
| **P3 — Eval & oracle** | Divya | `eval/`, metrics, matching, ideal-click simulator, result tables/figures, report editing | RQ0, RQ4 | 3–4 days |
| **P4 — Perturbations & baselines** | Aman | `perturb/`, `baselines/`, sensitivity sweeps, training-data variants and ablation launches | RQ2 | 3–4 days |
| **P5 — Corrector model** | Sharvani (and Srikrishnan to split any extra work) | `corrector/`: architecture, losses, training loop | RQ1 | 6–8 days |

\*Rough coding days over the 6 weeks, excluding report writing. P5 is the heaviest and the hardest to estimate.

**How to claim a role**
1. Read the role descriptions below and talk it over in the team chat.
2. Put your name in the Owner column in a small PR (or commit directly to `main`; this is the one exception). One person per role. Two people can share a role only if the team agrees, and it must be written here.
3. Update the Status & decisions log (section 12) with the date.
4. If you later want to swap, tell the team first, then edit the table.

**AI agents: never claim, change or assume a role for a user.** Only edit the Owner column when your user explicitly tells you which role they chose. If the role is still UNCLAIMED, ask the user.

### Role descriptions (to help you choose)

**P1 — Segmenters & infra** (systems / HPC / debugging comfort)
- Builds and maintains the Apptainer image, the GreatLakes shared directory and the Slurm templates.
- Writes the SAM 3 (and SAM 2.1) `Segmenter` wrappers and the feature cache, and measures time and memory.
- Everyone's GPU runs depend on this role, so it has the most handoffs. Hit your milestones early.
- Good fit if you like Linux, clusters, PyTorch internals and unblocking people.

**P2 — Data & detectors** (data wrangling, light model use)
- Sets up COCO and LVIS, builds object records and the seen/unseen category split (hand-check the COCO<->LVIS mapping and the LVIS-val image-leakage issue).
- Wraps the independent detector (LLMDet, or Grounding DINO if LLMDet won't install) and mines real SAM 3 / detector failures.
- Leads RQ3 (unseen categories). The lightest role in coding terms, with a few annoying data details.
- Good fit if you like datasets, annotation formats and careful bookkeeping.

**P3 — Eval & oracle** (metrics, analysis, writing)
- Writes metrics (IoU, Boundary IoU, success rates, harm rate), the prediction-to-GT matching, and the ideal-click simulator.
- Owns the results table format (`ResultRow`), the figures, and the final report editing. All numbers pass through this code.
- Leads RQ0 (including the go/no-go) and RQ4 (harm rate).
- Good fit if you like experimental design, careful measurement, plotting and writing.

**P4 — Perturbations & baselines** (geometry, experiment design)
- Builds the box/point perturbation generators with normalized severity measures, the sensitivity sweeps, and the rule-based baselines (random clicks, box refit, uncertainty click).
- Builds the training-data variants (synthetic / real / mixed / matched) and launches the ablation runs.
- Leads RQ2 (synthetic -> real transfer). Much of it runs on CPU against the dummy segmenter.
- Good fit if you like geometry, data generation and designing clean experiments.

**P5 — Corrector model** (deep learning / training)
- Designs the small corrector network (error map and click head first; box head later), the losses, the dataloader and the training loop, including multi-step rollouts.
- Leads RQ1 (learned vs. simple rules). Highest effort and highest risk; depends on cached features from P1 and training data from P4.
- Good fit if you want the core modeling work and are comfortable debugging training.

### Notes
- P5's job is the heaviest. P4 builds the data variants (`train_data: synthetic | real | mixed | matched` config flag) so P5 only owns model + training loop.
- P3 and P4 both feed one RQ0 write-up (P3 = ideal-click study, P4 = sensitivity sweeps).
- Nobody should block anyone: if you need something from another role, use the dummy segmenter / fake data to keep going and agree the interface in the PR.

## 6. Timeline (6 weeks, ~2 coding days per person per week)

Everyone codes on the **same two days each week** so handoffs happen within a session. End each session by submitting Slurm jobs; read results next session.

| Week | Focus |
|---|---|
| 1 | **Demo sprint (2.5 days):** scaffold + GreatLakes setup; RQ0 preliminary numbers on ~500–1000 COCO val images; sensitivity curves; baselines vs. oracle; corrector v0 (stretch); slides. |
| 2 | Full-val RQ0 + **go/no-go (hold firmly)**; mine real failures on a COCO-train subset; feature cache. |
| 3 | Corrector v1 on synthetic errors; all baselines done. |
| 4 | RQ2 data variants; RQ3 (unseen), RQ4 (harm rate); multi-step rollout if v1 works. |
| 5 | Freeze final runs; figures; start writing. |
| 6 | Write-up and presentation. **No new experiments.** |

### Demo-sprint deliverables (proposal presentation, 6–7 min)
1. Problem slide with a before/after failure example.
2. RQ0 failure rates (IoU < 0.5 / 0.75 / 0.9) for SAM 3 text and detector-box -> SAM.
3. Sensitivity curves (IoU vs. box shift / scale).
4. Headroom: oracle-click clicks-to-85/90 vs. box-refit and random-click baselines.
5. Corrector v0 preliminary result *if it works*; otherwise architecture + honest status.
6. Plan, go/no-go criterion, risks.

## 7. Shared interfaces (contracts)

Authoritative definitions live in `src/promptcorrect/types.py` once it exists; until then this table is authoritative. Changing a contract requires agreement from all roles that consume it.

| Interface | Owner | Shape |
|---|---|---|
| ObjectRecord | P2 | `image_id, ann_id, category, split (seen/unseen), gt_box, gt_mask (RLE)` |
| Prompt | P4 | `{box: [x0,y0,x1,y1] or None, pos_points: [...], neg_points: [...]}` |
| Segmenter call | P1 | `segment(image_id, prompt) -> mask_logits, iou_score, presence_score` (reuses cached image features) |
| Cache layout | P1 | `cache/features/{image_id}.pt` (fp16), `cache/detections/{model}/{image_id}.json` |
| ResultRow | P3 | `ann_id, method, step, iou, boundary_iou, latency_ms, perturbation_type, severity` |

Box format is always `xyxy` in absolute pixels. Masks are RLE (COCO format) on disk.

## 8. Compute: GreatLakes HPC

- Course account has (per RMP): **GPU 371 h, SPGPU 562 h, Standard 4,053 h, Largemem 1,318 h** (confirm units; ~930 GPU-hours total vs ~150 planned).
- **SPGPU (A40, 48 GB)** for SAM 3. `gpu` (V100) only for small tests. **Standard** for CPU-only work (eval, perturbation generation, metrics). Don't use Largemem.
- Never run heavy work on login nodes. Use `sbatch` / `srun`. Always pass `--account=<ACCOUNT>`. Log GPU-hours you use in the team channel.
- **Storage:** all shared data on `/scratch`, in a group-writable shared dir:
  ```
  /scratch/<ACCOUNT>_root/<ACCOUNT>/shared_data/eecs542tbn/
      data/        # COCO, LVIS (read-only after download)
      cache/       # features, detections
      ckpt/        # model weights
      containers/  # .sif files (versioned by date, `current.sif` symlink)
      runs/        # per-run outputs, checkpoints, logs
  ```
  Scratch purges files not accessed for ~60 days. Commit small result tables/figures to `results/`. Keep code/configs only in `$HOME`.
- **Environment = Apptainer container**, not a shared conda env. Build instructions: `docs/container.md`. Code, data and weights are bind-mounted, never baked into the image.
- Run pattern:
  ```
  apptainer exec --nv --bind /scratch/<ACCOUNT>_root/<ACCOUNT> \
    --env PYTHONPATH=$PWD/src $SIF python scripts/<script>.py --config configs/experiments/<x>.yaml
  ```
- HF access to SAM 3 is **already granted**. Put your own `HF_TOKEN` in your shell, never in the repo or image. Point `HF_HOME` to scratch.
- Workflow: code locally -> push to GitHub -> `git pull` on GreatLakes -> run via Slurm.

Placeholders `<ACCOUNT>`, `<uniqname>` to fill in here once confirmed: (TODO)

## 9. Repo layout (planned — the scaffold is being created)

```
configs/{paths,experiments}/   # paths/{local,greatlakes}.yaml: DATA_ROOT, CACHE_ROOT, CKPT_ROOT
container/                     # promptcorrect.def (Apptainer definition)
docs/                          # container.md, interfaces.md, greatlakes.md
src/promptcorrect/
    types.py                   # shared contracts
    data/  segmenters/  detectors/  perturb/  baselines/  eval/  corrector/  viz/
scripts/                       # thin CLI entrypoints only (no logic)
slurm/                         # *.sbatch
tests/                         # run on DummySegmenter, no GPU
results/                       # small CSV/PNG only
```

## 10. Conventions

- Branch per person/feature (`p3/oracle-clicks`), small PRs into `main`, at least one other person skims it. Don't push directly to `main` except trivial doc fixes.
- Stay inside your role's directory. Touching another role's code -> tell them / PR comment.
- No hardcoded paths: read `configs/paths/*.yaml`. No data, weights, caches or `.sif` files in git.
- Logic lives in `src/`; `scripts/` only parse args and call into `src/`.
- Windows users: line endings must be LF (`.gitattributes` enforces it); CRLF breaks `.sbatch` files on Linux.
- Fix random seeds and log them. Each experiment is a config file + git commit hash.
- Every module should have at least a basic test that runs on the dummy segmenter on a CPU.
- Python 3.12, PyTorch (see container). Type hints on public functions; keep comments sparse and useful.

## 11. Rules for AI agents working in this repo

1. Read this file first. Respect the roles in section 5 (and never claim one on a user's behalf) and the decisions in section 4.
2. Do not change shared interfaces (section 7) without the team's agreement; if a change seems necessary, say so and propose it rather than editing it silently.
3. Never fabricate results or claim something was run when it wasn't. Report failures plainly.
4. Mark anything you have not verified (for example SAM 3 API behavior) as an assumption, and add it to the open-questions list when it matters to others.
5. No secrets (HF tokens, SSH keys, uniqnames beyond what's needed) in commits.
6. Don't commit large files or run heavy jobs on login nodes.
7. Prefer small, reviewable changes. Don't refactor other people's modules.

## 12. Status & open questions (update continuously)

**Status (2026-10-23):** repo contains only the initial commit. Container recipe and docs drafted but **untested**. SAM 3 HF access granted. Data not yet downloaded.

**Known assumptions to verify:**
- SAM 3 interactive point/box API (`predict_inst`-style predictor) works as expected on A40.
- Per-image feature cache size (estimate ~10 MB fp16; unmeasured). Check against scratch quota.
- LLMDet's mmdet-based install may not fit the main container; Grounding DINO via `transformers` is the fallback.
- COCO <-> LVIS category mapping needs a hand check.
- SAM 3 text-path Hungarian matching design (secondary setting).

**Decisions log**
- 2026-10-23: Timeline cut from 10 to 6 weeks, ~2 coding days/week; scope reduced to RQ0–RQ4 (stretch list in section 3).
- 2026-10-23: Apptainer container instead of a shared conda env; built locally (WSL2) if rootless build on GreatLakes isn't available.
- 2026-10-23: Shared data lives on `/scratch` in a group-writable directory.
- 2026-10-23: Roles are self-selected via the claim board in section 5. All roles are currently UNCLAIMED. Srikrishnan is doing the initial GreatLakes/container setup and repo scaffold for the whole team regardless of which role they pick.

## 13. Keeping this file current

- Anyone who learns something that other people (or their agents) need — an API quirk, a path, a changed decision, a finished milestone — adds a line to section 12 (or edits the relevant section) in a small PR. Bump the "Last updated" date.
- Keep it short and factual. Long explanations go in `docs/`, linked from here.
- Role owners keep their own row and section up to date.
