# CLAUDE.md — Shared project context (EECS 542 / CSE542)

> **For every team member's AI agent: read this whole file before doing anything.**
> It is the single source of truth for project goals, scope, roles, interfaces and conventions.
> Five people are working in parallel with five separate agents; this file is how we avoid drift.
> If something here is wrong or outdated, fix it via a small PR (see "Keeping this file current").

Last updated: 2026-10-25 (P2 data pipeline and P3 eval contracts merged into main)

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
- Builds and maintains the shared Python environment, the GreatLakes shared directory and the Slurm templates.
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
Exact per-person plan, gates and contingencies: [`docs/demo_sprint.md`](docs/demo_sprint.md).

1. Problem slide with a before/after failure example.
2. RQ0 failure rates (IoU < 0.5 / 0.75 / 0.9) for SAM 3 text and detector-box -> SAM.
3. Sensitivity curves (IoU vs. box shift / scale).
4. Headroom: oracle-click clicks-to-85/90 vs. box-refit and random-click baselines.
5. Corrector v0 preliminary result *if it works*; otherwise architecture + honest status.
6. Plan, go/no-go criterion, risks.

## 7. Shared interfaces (contracts)

**Full spec: [`docs/interfaces.md`](docs/interfaces.md)** (read it before touching anything that crosses roles). Once `src/promptcorrect/types.py` exists, the code and that doc must agree. Changing a contract requires agreement from all roles that consume it.

| Interface | Owner | Summary |
|---|---|---|
| ObjectRecord | P2 | `obj_id ("coco:123"), image_id, source, category_id, category_name, novelty (seen/unseen), gt_box, gt_mask (RLE), area, size_bin, thinness, n_neighbors` |
| Prompt | P4 | `box (xyxy or None), points [(x, y, label)], mask_logits, origin, perturbation`; severity via `describe(prompt, obj)` |
| Segmenter | P1 | `set_image(image_id)`, `predict(prompts, multimask) -> [SegOutput(mask_logits, iou_score, presence_score)]`, `detect_text(phrase)`, `to_full_res(logits)` |
| Cache layout | P1 | `cache/manifest.json`, `cache/features/{backend}/{image_id}.safetensors` (fp16), `cache/detections/{model}/{split}.jsonl` |
| Results row | P3 | one row per `(obj_id, method, step)`: `iou, boundary_iou`, severity fields, `latency_ms`; Parquet |
| Run config | P2 | YAML: `run_id, git_commit, seed, backend, train_data, perturbations, max_steps, splits_file` |

Global conventions: boxes are `xyxy` absolute pixels in the original image; masks are COCO RLE on disk and low-res logits (`> 0`) in memory; object IDs are strings (`coco:…` / `lvis:…`); JSONL for records, Parquet for results, safetensors for features. One implementation each of `iou`/`boundary_iou`/`match` (P3), `describe` (P4) and `Segmenter` (P1). The interface doc also lists open issues (O1–O13) that need a team decision.

## 8. Compute: GreatLakes HPC

- Course account has (per RMP): **GPU 371 h, SPGPU 562 h, Standard 4,053 h, Largemem 1,318 h** (confirm units; ~930 GPU-hours total vs ~150 planned).
- **SPGPU (A40, 48 GB)** for SAM 3. `gpu` (V100) only for small tests. **Standard** for CPU-only work (eval, perturbation generation, metrics). Don't use Largemem.
- Never run heavy work on login nodes. Use `sbatch` / `srun`. Always pass `--account=eecs542f26s001_class`. Log GPU-hours you use in the team channel.
- **Storage:** all shared data on `/scratch`, in the owner's (Srikrishnan's) scratch folder, shared with the four teammates via per-user ACLs (`setfacl`, access + default ACLs for all five uniqnames). Do **not** use group permissions (`chmod g+...`): the account `eecs542f26s001_class` is class-wide, so the group is every student.
  ```
  /scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn/
      data/        # COCO, LVIS (read-only after download)
      cache/       # features, detections
      ckpt/        # model weights
      venv -> venv-<date>/   # shared Python 3.12 environment (slurm/setup_env.sbatch)
      src_deps/    # SAM 3 / SAM 2.1 checkouts used by the environment
      hf_cache/    # HF_HOME
      runs/        # per-run outputs, checkpoints, logs
  ```
  Scratch purges files not accessed for ~60 days. Commit small result tables/figures to `results/`. Keep code/configs only in `$HOME`.
- **Environment = one shared Python 3.12 venv on scratch** (`$PC_ROOT/venv`), built by `slurm/setup_env.sbatch`; details in `docs/environment.md`. GreatLakes cannot build containers (`--fakeroot` fails: no subuid mapping), so `container/` is an unused recipe. Dependency versions live in `pyproject.toml`.
- Run pattern (from the repo root):
  ```
  export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
  source $PC_ROOT/venv/bin/activate
  export PYTHONPATH=$PWD/src PC_PATHS=greatlakes
  python scripts/<script>.py --config configs/experiments/<x>.yaml
  ```
- HF access to SAM 3 is **already granted**. Put your own `HF_TOKEN` in your shell, never in the repo. Point `HF_HOME` to scratch.
- Workflow: code locally -> push to GitHub -> `git pull` on GreatLakes -> run via Slurm.

Slurm account: `eecs542f26s001_class`. Remaining placeholder: `srikrish` (Srikrishnan's uniqname; appears in `configs/paths/greatlakes.yaml` and the paths above) — to be filled in.

## 9. Repo layout

```
configs/{paths,experiments}/   # paths/{local,greatlakes}.yaml: data/cache/ckpt/runs roots (PC_PATHS env picks one)
container/                     # promptcorrect.def + check_env.py (def unused on GreatLakes; check_env.py is used)
docs/                          # interfaces.md, demo_sprint.md, environment.md (container.md: unused)
src/promptcorrect/
    types.py  config.py        # shared contracts; paths, RunConfig, seeding
    testing.py                 # make_synthetic_objects() for CPU tests (no COCO needed)
    data/                      # P2: geometry.py (RLE/box helpers), records.py (JSONL I/O; builder is a TODO)
    segmenters/                # P1: base.py (interface), dummy.py (done), sam3.py / sam21.py (TODO), make_segmenter()
    detectors/ perturb/ baselines/ eval/ corrector/ viz/   # stubs with owner + TODO signatures
scripts/                       # thin CLI entrypoints only (no logic); download_hf.py
slurm/                         # download_data.sbatch (more templates: P1, Day 1)
tests/                         # CPU tests on DummySegmenter, no GPU (pytest)
results/                       # small CSV/PNG only (not created yet)
```

Local dev: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]" && pytest` (see README).
`DummySegmenter(objects)` needs only `ObjectRecord`s; use `promptcorrect.testing.make_synthetic_objects()` for fake ones. Its click behaviour is documented in its module docstring. Its IoUs are meaningless scientifically; it exists to test logic.

## 10. Conventions

- Branch per person/feature (`p3/oracle-clicks`), small PRs into `main`, at least one other person skims it. Don't push directly to `main` except trivial doc fixes.
- Stay inside your role's directory. Touching another role's code -> tell them / PR comment.
- No hardcoded paths: read `configs/paths/*.yaml`. No data, weights, caches or `.sif` files in git.
- Logic lives in `src/`; `scripts/` only parse args and call into `src/`.
- Windows users: line endings must be LF (`.gitattributes` enforces it); CRLF breaks `.sbatch` files on Linux.
- Fix random seeds and log them. Each experiment is a config file + git commit hash.
- Every module should have at least a basic test that runs on the dummy segmenter on a CPU.
- Python 3.12, PyTorch (see docs/environment.md). Type hints on public functions; keep comments sparse and useful.

## 11. Rules for AI agents working in this repo

1. Read this file first. Respect the roles in section 5 (and never claim one on a user's behalf) and the decisions in section 4.
2. Do not change shared interfaces (section 7) without the team's agreement; if a change seems necessary, say so and propose it rather than editing it silently.
3. Never fabricate results or claim something was run when it wasn't. Report failures plainly.
4. Mark anything you have not verified (for example SAM 3 API behavior) as an assumption, and add it to the open-questions list when it matters to others.
5. No secrets (HF tokens, SSH keys, uniqnames beyond what's needed) in commits.
6. Don't commit large files or run heavy jobs on login nodes.
7. Prefer small, reviewable changes. Don't refactor other people's modules.

## 12. Status & open questions (update continuously)

**Status (2026-10-23):** repo scaffold written (types, config, data helpers, `Segmenter` interface + `DummySegmenter`, stubs for other roles; 22 CPU tests pass). Environment: `slurm/setup_env.sbatch` (shared venv) written but **not yet run on GreatLakes**; the container recipe is unusable there (no fakeroot). SAM 3 HF access granted. Data not yet downloaded (`slurm/download_data.sbatch` ready, not yet run). SAM 3 / SAM 2.1 wrappers, detectors, perturbations, metrics and corrector are TODO stubs.

**Known assumptions to verify:**
- SAM 3 interactive point/box API (`predict_inst`-style predictor) works as expected on A40.
- Per-image feature cache size (estimate ~10 MB fp16; unmeasured). Check against scratch quota.
- LLMDet's mmdet-based install may not fit the shared environment; Grounding DINO via `transformers` is the fallback.
- COCO <-> LVIS category mapping needs a hand check.
- SAM 3 text-path Hungarian matching design (secondary setting).

**Decisions log**
- 2026-10-25: P2 week 1 implementation prepared on `p2/week1-data-detectors`: COCO records, fixed demo split, leakage-safe split helper, Grounding DINO adapter and detection export. Official val annotations locally produced 500 images / 2,937 retained objects; CPU checks pass. GPU inference, LLMDet compatibility and full LVIS mapping remain unverified. O10 implementation proposals and reproducible commands: `docs/p2_week1.md`. Merged with the P1 scaffold on main (2026-10-25); `ObjectRecord` now has a single definition in `types.py`.
- 2026-10-23: Timeline cut from 10 to 6 weeks, ~2 coding days/week; scope reduced to RQ0–RQ4 (stretch list in section 3).
- 2026-10-23: Singularity container instead of a shared conda env; built locally (WSL2) if rootless build on GreatLakes isn't available.
- 2026-10-25: GreatLakes cannot build containers (`singularity build --fakeroot`: "no valid mapping entry found"). Decision: use ONE shared Python 3.12 venv at `$PC_ROOT/venv` built by `slurm/setup_env.sbatch` from `pyproject.toml` plus pinned torch/SAM versions (`docs/environment.md`). The container recipe stays in the repo, unused; a GitHub Actions image build is a possible later option.
- 2026-10-23: Shared data lives on `/scratch` in a group-writable directory.
- 2026-10-23: Roles are self-selected via the claim board in section 5 (now claimed; see the board). Srikrishnan is doing the initial GreatLakes/container setup and repo scaffold for the whole team.
- 2026-10-23: Shared formats written up in `docs/interfaces.md` (draft v0.1, from `CV_Project_Revised_Plan.pdf`). Open issues O1–O13 there need team decisions; the revised plan's 10-week schedule is ignored in favour of section 6 here.
- 2026-10-25: P3 (Divya) decided O11/O12 and fixed the `iou`/`boundary_iou`/`match` contracts (`docs/interfaces.md` v0.2). `eval/metrics.py` and `eval/methods.py` added with CPU tests. O1 settled; O4 waits on P4.

## 13. Keeping this file current

- Anyone who learns something that other people (or their agents) need — an API quirk, a path, a changed decision, a finished milestone — adds a line to section 12 (or edits the relevant section) in a small PR. Bump the "Last updated" date.
- Keep it short and factual. Long explanations go in `docs/`, linked from here.
- Role owners keep their own row and section up to date.
