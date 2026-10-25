# P2 week 1: data and detector handoff

This PR supplies the COCO demo data pipeline and the Grounding DINO demo fallback.
SAM 3 inference stays behind P1's `Segmenter`; matching, metrics and failure mining
use P3's evaluation code when available. No segmentation results are claimed here.

## Verified locally

On 2026-10-25, the official `instances_val2017.json` produced 500 images and
2,937 retained objects (683 small, 1,429 medium, 825 large), using seed 0.
These are dataset inventory counts, not evaluation results. The fixed IDs are in
`data/splits_demo.json`; provenance and artifact hashes are in
`data/objects_val.manifest.json`. Annotation and object JSONL files stay outside git.

CPU tests cover polygons, multipart masks, compressed/uncompressed RLE, filtering,
stratification, fixed splits, image leakage, atomic writes, detector adapter calls,
GT-present phrase export, the Segmenter text contract and run provenance.
Model inference and HPC integration have **not** been run. The tests use a fake
Segmenter because P1's DummySegmenter has not landed on main.

## Prepare records

Use Python 3.12. Install `requirements-p2.txt` in a virtual environment. Until P1's
packaging scaffold lands, set `PYTHONPATH=src` from the repository root.

Download the official annotation archive from
[COCO's S3 endpoint](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip).
Extract `annotations/instances_val2017.json` below `$DATA_ROOT/coco/`.
The image directory is `$DATA_ROOT/coco/val2017/` with filenames from the annotations.
Only records preparation needs annotations; detection also needs images and weights.

Create a local paths YAML (or use P1's paths config when available):

```yaml
DATA_ROOT: /absolute/path/to/data
CACHE_ROOT: /absolute/path/to/cache
```

```sh
PYTHONPATH=src python scripts/prepare_p2_demo.py --paths /path/to/paths.yaml
PYTHONPATH=src python -m pytest tests/p2 -q
```

The default outputs are `$DATA_ROOT/{splits_demo.json,objects_val.jsonl,
objects_val.manifest.json}`. Copy the committed split into `$DATA_ROOT` first to
validate and reuse the shared list. Preparation also regenerates the identical
list from sorted annotation IDs with Python's `random.Random(0).sample(..., 500)`.
An existing split must match; existing records and manifests are never overwritten.
Paths inside the paths YAML resolve relative to that YAML; artifact arguments
resolve relative to their named data/cache root.

The `ObjectRecord` dataclass lives in P2's `data/records.py`, with the fields from
`docs/interfaces.md` unchanged. P1 can re-export it from `types.py` when creating
the shared scaffold, so consumers use one definition. This integration is pending.
`configs/experiments/p2_demo.yaml` supplies shared run-config defaults; the P2 CLIs
use explicit arguments and save resolved data/detection provenance alongside outputs.
Loading shared experiment configs remains part of P1's scaffold.

## Export detections

Install `torch`, `torchvision`, and `transformers>=4.51,<5` in the model environment
(container dependencies remain P1-owned). The wrapper follows the
[official Transformers Grounding DINO API](https://huggingface.co/docs/transformers/model_doc/grounding-dino).
It runs a frozen model in eval mode with gradients disabled, one phrase at a time,
and returns `(xyxy_box, score)` in original-image pixels.

Run on a scheduled GPU compute node, using P1's Slurm/container setup:

```sh
PYTHONPATH=src python scripts/export_p2_detections.py \
  --paths /path/to/paths.yaml --device cuda --seed 0
```

An optional `--revision <checkpoint-commit>` pins the downloaded model. The export
manifest records its resolved revision, thresholds, seed, package versions,
source/output hashes, git commit and dirty flag. Keep the manifest with the JSONL.
Both files must exist for a completed handoff. No shared cache manifest is changed.

Outputs are `$CACHE_ROOT/detections/grounding_dino/{val.jsonl,val.manifest.json}`.
Every line has exactly `image_id`, `phrase`, `box`, `score`. Only unique phrases
from **retained** GT objects are queried. The manifest lists queries, including
queries yielding zero detections, so empty output is distinguishable from a
missing run. Repeated categories are queried once per image.

For SAM 3 text, use the same exporter after P1 supplies a configured Segmenter:

```python
from promptcorrect.data.records import read_records
from promptcorrect.detectors.export import export_detections

# segmenter is a P1-created instance with image loading configured.
export_detections(
    read_records(objects_path), sam_text_output_path, segmenter=segmenter,
)
```

This calls `set_image(image_id)` once per image and `detect_text(phrase)` once per
unique category, writing the same fields plus JSON-safe `mask_rle`. Retain the
P1 backend/checkpoint metadata alongside this output. It does not implement SAM
loading, feature caching or coordinate transforms. Hand both detector streams,
the fixed split, original annotations and object records to P3. P3 needs the
original annotations to handle filtered/crowd instances when matching.

## Defaults and proposals for team review

- O1/O2/O9 follow the demo sprint defaults: drop crowds and boxes whose width
  **and** height are below 32; split at image level; query GT-present categories.
- O10 proposal: thinness uses the squared length of all pixel-edge contours,
  including holes and image-border edges, divided by decoded mask area. This is
  explicitly a pixel-grid perimeter, not a smoothed or marching-squares contour.
  `area` and `size_bin` use annotation area and the COCO thresholds 32²/96².
- O10 proposal: neighbors count retained same-class objects at Euclidean center
  distance <= the reference object's width. Counts can be asymmetric. Review
  these two definitions with P3/P4 before using their stratification in reports.
- The schema for full splits is `{train: [ids], val: [ids],
  lvis_val_excluded_from_train: [ids]}`. `training_split` removes every LVIS-val
  image from COCO train; the exclusion list records the actual train intersection.
  Demo splits have no training IDs and do not require LVIS annotations.

## Remaining work and slide 3 outline

Week 1 GPU handoff requires COCO images, the model environment, P1's SAM 3 wrapper
and scheduled compute. Grounding DINO is implemented as the demo fallback; LLMDet
installation was not attempted and its compatibility remains unverified.
Real detections, evaluation numbers and examples must be filled in after those runs.
Full COCO-train/LVIS preparation, hand-checked category mapping and mined real
training failures remain subsequent work; no unseen-category labels are guessed.

Slide 3 can show: COCO val2017 -> seed-0 fixed 500-image subset -> retained object
records -> GT-present phrases -> Grounding DINO boxes / SAM 3 text instances ->
P3 evaluation. Label Grounding DINO as the demo fallback and model runs as pending
until executed. Explain that later RQ3 uses LVIS non-COCO categories and removes
LVIS-val image IDs from corrector training.
