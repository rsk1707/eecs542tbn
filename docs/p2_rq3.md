# P2 next handoff: leakage-safe COCO / LVIS records

## Status audited 2026-11-16

P2 week 1 is merged (PR #1). The current `main` also contains P1's shared
environment/SAM 3 wrapper, P3's RQ0 runner/results (PR #7), P4's sensitivity sweep,
and P5's corrector v0. The demo detection runs **have happened**: the saved
`demo_rq0_1` results include Grounding DINO and SAM 3 text. The week 1 notes saying
model inference is pending describe the earlier handoff, not current status.

The next P2 component is full COCO train/val and LVIS-val preparation for RQ3.
`scripts/prepare_p2_rq3.py` prepares shared `ObjectRecord` JSONL without changing
shared contracts or other roles' code. This component does not run a model,
mine training failures, or claim new evaluation results.

## Category review before production preparation

Draft a review using the [official LVIS COCO-to-synset mapping](https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json).
Use the actual LVIS v1 validation annotation file as the category source; a draft
records input hashes, and preparation rejects a draft from different annotations.
The script runs without model dependencies and refuses to overwrite files.

```bash
PYTHONPATH=src python scripts/prepare_p2_rq3.py --paths configs/paths/local.yaml draft-review \
  --coco-synsets lvis/coco_to_synset.json --output rq3_category_review.json
```

Every LVIS category has a row like this (illustrative only):

```json
{
  "lvis_category_id": 378,
  "name": "dog",
  "synset": "dog.n.01",
  "coco_category_ids": [18],
  "novelty": "seen",
  "reviewed": false,
  "review_notes": ""
}
```

Exact synset matches start as **unreviewed seen candidates**; other rows have
`novelty: null`. Reviewers set each row's novelty, COCO matches, `reviewed: true`,
and explanatory `review_notes`. Seen rows must list valid COCO IDs; unseen rows
must list none. Review all categories, including those absent after filtering,
and version the reviewed small mapping separately from the large artifacts.

A local metadata check using the official mapping and
[Detectron2's LVIS v1 category metadata](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis_v1_categories.py)
found 79 exact candidates among 1,203 LVIS categories for the 80 COCO categories.
This is a vocabulary sanity check, not a dataset or evaluation run. COCO's
`hot dog` synset `frank.n.02` is absent as an exact match in that LVIS metadata.
More importantly, subclasses such as LVIS `bulldog` can overlap COCO `dog` without
synset equality. **Do not label the 1,124 unmatched categories unseen automatically.**
Agree with P3/P5 on semantic overlap before finalizing the review. No final
category mapping is claimed by this PR.

## Prepare on Great Lakes (standard CPU partition)

No SSH is needed to review or test the code. Production preparation needs access
as `shresth` to the shared annotations and venv. The inputs are COCO
`instances_train2017.json`, `instances_val2017.json`, LVIS `lvis_v1_val.json`,
the official synset mapping, and the reviewed category decisions. The default
LVIS location is `data/lvis/lvis_v1_val.json`; override `--lvis-val` if the
download job placed it elsewhere. All relative artifact paths use `data_root`
from the supplied paths YAML.

From the repository root on Great Lakes, first run `draft-review` using the
CPU template, complete the review, then submit preparation:

```bash
export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn

sbatch --account=eecs542f26s001_class --job-name=p2_map \
  slurm/run_cpu.sbatch scripts/prepare_p2_rq3.py \
  --paths configs/paths/greatlakes.yaml draft-review \
  --coco-synsets lvis/coco_to_synset.json --output rq3_category_review.json

# After category review is complete; choose a fresh directory for each run.
sbatch --account=eecs542f26s001_class --job-name=p2_rq3 --mem=32G --time=04:00:00 \
  slurm/run_cpu.sbatch scripts/prepare_p2_rq3.py \
  --paths configs/paths/greatlakes.yaml prepare \
  --category-review rq3_category_review.json --output-dir rq3_full_s0 --seed 0
```

The resources above are an initial allocation, not measured runtime/memory.
For a smaller COCO training handoff add `--train-count 3000`; this samples images
after excluding **all** LVIS-val images. COCO val and LVIS val remain complete.
The sample includes images even if no objects survive filtering; inventory
distinguishes selected images from images with objects. Inspect training category
coverage in the manifest before describing a category as seen by a trained
corrector; a small subset can miss some COCO classes. There is no extra training
holdout or model-selection split in this component.

Output under `$PC_ROOT/data/rq3_full_s0/`:

| Artifact | Meaning |
|---|---|
| `splits.json` | Existing train/val/excluded-image schema; COCO train excludes every official LVIS-val image |
| `category_review.json` | Exact reviewed category decisions used by this run |
| `objects_train.jsonl` | Leakage-safe COCO train objects |
| `objects_val.jsonl` | Full COCO val objects |
| `objects_lvis_val.jsonl` | Full LVIS val objects with `lvis:` annotation IDs and original shared COCO image IDs |
| `manifest.json` | Completion marker, source/artifact SHA-256 hashes, git state, seed, dependency versions, inventories |

LVIS uses the same O1 filter, mask encoding, size bins, thinness and same-class
neighbor implementation as COCO. Text phrases use the **first synonym**, replacing
underscores with spaces. Thinness/neighbor definitions remain the week 1 O10
proposals; this PR does not declare team acceptance.

Do not consume a directory without `manifest.json`. Existing output directories
are refused. A failed run can leave an incomplete directory; inspect its log and
use a new output name. JSONL and manifests are written atomically per file. Keep
large records on scratch; do not commit generated records or annotation files.

## Remaining owner handoffs

- **P1 / Shresth:** check scratch ACLs and input locations, then submit the CPU
  job. No new venv or container is needed. Production preparation and SSH access
  have not been tested in this change.
- **P3 / P5 / Shresth:** review semantic category overlap. RQ3 evaluation must
  handle LVIS-val images from both COCO train2017 and val2017. Existing detector
  loading expects COCO `file_name`, whereas LVIS metadata uses `coco_url`; an
  image resolver is still needed for LVIS detector exports. P3's RQ0 ignore-mask
  helper also hardcodes `coco:` IDs; do not reuse it unchanged for LVIS detection
  matching. These are follow-up integration tasks for the respective owners.
- **P2 + P3/P4/P5:** next, mine real **training** failures (initial detected mask
  IoU about 0.3–0.85), retaining actual prompts/detection provenance for training.
  Existing demo RQ0 output is validation data, so it must not be used to train.
  P3's `ResultRow` does not retain the matched detector box; agree on how to
  persist that association rather than reconstructing it by order or changing
  another role's interface silently.

## Local validation

```bash
PYTHONPATH=src python -m pytest --confcutdir=tests/p2 tests/p2 -q
PYTHONPATH=src python scripts/prepare_p2_rq3.py --help
git diff --check
```

`--confcutdir` keeps P2's CPU-only suite independent of the repository-wide
DummySegmenter fixture (which imports torch). Tests cover LVIS record geometry,
first synonyms, explicit/stale category decisions, empty-image leakage,
deterministic subsampling, source/artifact hashes, CLI execution and incomplete
run handling. Tests use tiny synthetic annotations; full production annotation
preparation and model inference remain unrun here.
