# Python environment on GreatLakes

GreatLakes cannot build container images for us (`singularity build --fakeroot` fails with "no valid mapping entry found"), so we use **one shared Python 3.12 virtual environment on scratch**. It is built by [`slurm/setup_env.sbatch`](../slurm/setup_env.sbatch) from the same pins that were validated for the container recipe: torch 2.10.0 + torchvision 0.25.0 (CUDA 12.6 wheels), SAM 3 and SAM 2.1 at pinned commits, and every other package from [`pyproject.toml`](../pyproject.toml), installed in a single `pip` pass.

`pyproject.toml` is the single source of dependency versions. The script's own pins (torch, SAM commits) are at the top of the script.

## Using it (everyone)

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
source $PC_ROOT/venv/bin/activate
export PYTHONPATH=$PWD/src            # from the repo root; the project itself is not pip-installed
python scripts/<script>.py --config configs/experiments/<x>.yaml
```

In an sbatch script, put those three lines before the `python` command and request the GPU partition (`--partition=spgpu --gres=gpu:1`). Model weights and data are paths from `configs/paths/greatlakes.yaml`; set `PC_PATHS=greatlakes`.

Check the environment any time (expect all `OK`; `cuda available: True` only on a GPU node):

```bash
python container/check_env.py
```

## Building or rebuilding it (Srikrishnan only)

Run from the repo root on GreatLakes, as a batch job (not on the login node):

```bash
cd ~/eecs542tbn && git pull
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
sbatch --account=eecs542f26s001_class slurm/setup_env.sbatch
tail -f pc_env-<jobid>.out
```

What it does, in order: finds Python 3.12 (system, or a standalone one installed with `uv` under `$PC_ROOT/tools`) -> creates `$PC_ROOT/venv-<date>` -> `pip`, `wheel`, `setuptools<81` -> CUDA torch -> clones SAM 3 / SAM 2.1 into `$PC_ROOT/src_deps` at the pinned commits -> dependency lists from `pyproject.toml` -> one resolver pass -> `pip check` -> `check_env.py`.

- It only repoints the symlink `$PC_ROOT/venv` to the new directory if `check_env.py` shows no `FAIL`. Teammates never see a half-built environment, and the previous environment stays until you delete it.
- A same-day rebuild needs `FORCE=1 sbatch ...` (it deletes that day's directory first).
- To use a specific Python: `PYTHON=/path/to/python3.12 sbatch ...`.
- Roughly 4 cores for 20-40 minutes (about 3 GB of downloads), a few core-hours of the account budget.

After the first build, run `python container/check_env.py` on a GPU node (`srun --partition=spgpu --gres=gpu:1 ...`) and expect `cuda available: True`. Also run `nvidia-smi | head -4` and note the driver version (the CUDA 12.6 wheels need a recent enough driver).

## Running jobs (everyone)

Two generic templates run any script in the shared environment. Submit from the **repo root**:

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
# GPU (SPGPU, 1 GPU, 32 GB RAM, 2 h by default):
sbatch --account=eecs542f26s001_class slurm/run_gpu.sbatch scripts/<script>.py --config configs/experiments/<x>.yaml
# CPU only (standard): metrics, perturbation generation, plotting
sbatch --account=eecs542f26s001_class slurm/run_cpu.sbatch scripts/<script>.py ...
```

Override resources on the command line (`--time=04:00:00`, `--mem=64G`, `--job-name=rq0`). The log goes to `<job-name>-<jobid>.out` in the directory you submitted from (these are gitignored). Use the CPU template whenever a job does not need a GPU: SPGPU hours are the scarce budget.

## SAM 3 smoke test (sprint gate G1)

Needs the weights at `$PC_ROOT/ckpt/sam3/sam3.pt` (`scripts/download_hf.py`) and COCO val.

```bash
sbatch --account=eecs542f26s001_class --time=00:30:00 slurm/run_gpu.sbatch scripts/smoke_sam3.py --n-objects 3
tail -f pc_gpu-<jobid>.out
```

It prints one `[ OK ]` / `[FAIL]` line per check and writes `$PC_ROOT/runs/smoke_sam3/summary.json` and `overlay.png`. It tests, on real COCO objects: box prompts (is the box really xyxy pixels?), `return_logits`, point-only, box + point, `mask_input` refinement, a truncated box fixed by 4 ideal clicks, decoder latency, and the text path (box format, the number of detections at confidence 0.5 vs 0.1, and the raw presence score). **Send the summary to Srikrishnan / paste it into `CLAUDE.md` section 12**: it decides how `segmenters/sam3.py` is written.

What is already known from reading the SAM 3 code (pinned commit) and still needs the smoke test to confirm numerically:
- Interactive path: `model.predict_inst(state, point_coords, point_labels, box, mask_input, multimask_output, return_logits)` after `Sam3Processor.set_image`; needs `build_sam3_image_model(enable_inst_interactivity=True)`; masks come back at original size, low-res logits are fixed 256x256.
- Text path: `processor.set_text_prompt(prompt, state)` fills `state["masks"|"masks_logits"|"boxes"|"scores"]`; boxes are xyxy pixels; detections are filtered by a confidence threshold (default 0.5, which directly affects the RQ0 failure rates).
- **Presence score (open issue O6):** only the text path has one. `scores` already includes it as a factor and the raw value is not stored; `predict_inst` returns none. Getting the raw value needs a second `model.forward_grounding(...)` call.

## Day 2 pipeline: records and detections (P1 runs these; outputs feed P3)

All from the repo root on GreatLakes. `configs/paths/greatlakes.yaml` is the paths file for every script (key names are case-insensitive, so P2's `DATA_ROOT` style and our `data_root` style both work).

```bash
cd ~/eecs542tbn && git pull
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
A=--account=eecs542f26s001_class
P=configs/paths/greatlakes.yaml

# 1. Object records for the fixed 500-image demo subset (CPU, about a minute).
#    Copy the committed split first so everybody uses the same image list.
cp data/splits_demo.json $PC_ROOT/data/splits_demo.json
J1=$(sbatch --parsable $A slurm/run_cpu.sbatch scripts/prepare_p2_demo.py --paths $P)

# 2. Grounding DINO detections (independent detector; waits for step 1). Uses the base weights we downloaded.
J2=$(sbatch --parsable $A --dependency=afterok:$J1 --time=01:00:00 slurm/run_gpu.sbatch      scripts/export_p2_detections.py --paths $P --device cuda --seed 0      --model-id $PC_ROOT/ckpt/grounding-dino-base)

# 3. SAM 3 text detections (waits for step 1). The threshold changes the RQ0 numbers: agree on it first.
J3=$(sbatch --parsable $A --dependency=afterok:$J1 --time=01:00:00 slurm/run_gpu.sbatch      scripts/export_sam3_text.py --paths $P --threshold 0.5)
squeue -u $USER
```

Outputs (each next to a `*.manifest.json` that must exist for the run to count as complete):

| File | Producer |
|---|---|
| `$PC_ROOT/data/objects_val.jsonl` + `objects_val.manifest.json` | step 1 |
| `$PC_ROOT/cache/detections/grounding_dino/val.jsonl` | step 2 (`image_id, phrase, box, score`) |
| `$PC_ROOT/cache/detections/sam3_text/val.jsonl` | step 3 (adds `mask_rle`); the manifest also holds per-query SAM 3 presence |

The export scripts refuse to overwrite an existing output. To rerun, move the old files away first (or pass `--output` with a new name).

## RQ0 study (P1 runs it; P3 owns the analysis)

After the three Day 2 jobs finished:

```bash
source $PC_ROOT/venv/bin/activate && export PYTHONPATH=$PWD/src
python scripts/summarize_inputs.py --paths configs/paths/greatlakes.yaml     # counts only; checks both detection streams exist

# Ceiling only (GT box -> SAM 3, plus ideal clicks). Needs no detections and no match(); runs today.
sbatch $A --time=01:00:00 slurm/run_gpu.sbatch scripts/run_rq0.py --paths $P --run-id demo_rq0_ceiling --sources gt_box

# Full study: GT box, Grounding DINO boxes, SAM 3 text boxes, SAM 3 text masks (+ ideal clicks on the box sources).
# Needs P3's match() in eval/matching.py; the script says so and exits if it is still a stub.
sbatch $A --time=02:00:00 slurm/run_gpu.sbatch scripts/run_rq0.py --paths $P --run-id demo_rq0_1

python scripts/summarize_rq0.py --paths $P --run-id demo_rq0_1             # re-print the tables any time
```

Outputs in `$PC_ROOT/runs/<run-id>/`: `results.parquet` (one row per object, source, method, step; the long format of `docs/interfaces.md` section 5 plus `source`, `detected`, `det_score`, `pred_iou`, `n_gt_same_class`, `n_pred`, `n_fp`), `queries.parquet` (per image/phrase/source: detections, matches, false positives), `config.yaml`, `rq0_manifest.json`, and `summary_*.csv`. Per-object data is not committed (scratch only).

The four sources and what each answers (`source` column):

| `source` | Prompt | Question it answers |
|---|---|---|
| `gt_box` | the GT box | the ceiling: how good can a perfect prompt be |
| `grounding_dino_box` | independent detector boxes | failures of an automatic pipeline (RQ0 headline) |
| `sam3_text_box` | SAM 3's own detection boxes, re-run through the interactive head | failures caused by the box |
| `sam3_text_mask` | SAM 3's text masks as they are | failures of the detector's own masks |

A GT object nobody detects is **missed** (`detected=False`, IoU 0), counted as a failure in the "all" rates and kept out of the "detected" rates. `boundary_iou` is filled at step 0 only. Ideal clicks are the standard simulation: positive click in the largest false-negative region, negative click in the largest false-positive region, at the point farthest from that region's boundary. Expect IoU to be non-monotone in places (a click can hurt).

## Changing a dependency version

Edit `pyproject.toml` (or the pins at the top of the script for torch / SAM commits), push, `git pull` on GreatLakes, and rebuild as above. Tell the team when the symlink moves.

## Known limits

- **Not tested on GreatLakes yet.** The Python install steps were replayed on Windows with the CPU build of torch (see the "What was validated" section in [`container.md`](container.md)). The first real run is the real test. One expected difference: on Windows `import sam3` fails with `No module named 'triton'`; Linux CUDA torch installs `triton` itself, so it should work on GreatLakes. If `check_env.py` shows `FAIL sam3` there, read the error before changing anything.
- Less isolated than a container: the venv uses scratch paths and the cluster's system libraries. Scratch purges files not accessed for ~60 days; the environment can be rebuilt with one command.
- Teammates need read/execute access to `$PC_ROOT` (the `setfacl` step in `CLAUDE.md` section 8).
