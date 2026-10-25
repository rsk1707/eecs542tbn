# Shared interfaces and formats

**Status:** Draft v0.1 (2026-10-23). Transcribed from `CV_Project_Revised_Plan.pdf` (AI-suggested, not yet signed off by the team).
**Purpose:** one place that defines every format two roles must agree on, so nobody writes converters or hits coordinate bugs.
**Overrides:** this doc is the detailed contract behind `CLAUDE.md` section 7. Once `src/promptcorrect/types.py` exists, the code and this doc must say the same thing; if they differ, fix the one that is wrong in the same PR.

**How to read this doc**
- Plain text is taken from the revised plan.
- **[Proposed]** is an addition or clarification I made while transcribing. It needs team agreement.
- **[Open]** marks an ambiguity or gap. All of them are collected in section 9 with a recommended resolution.

**Changing a contract:** the owner proposes the change in a PR touching this file; every role that consumes it approves; bump the changelog (section 10). Nobody edits a contract silently.

Guiding rule: **reuse COCO and SAM conventions wherever they exist.**

---

## 0. Global conventions (decide once)

| Thing | Convention | Why |
|---|---|---|
| Coordinates | Absolute pixels in the **original image**; boxes are `xyxy` floats | SAM-family APIs take `xyxy`. COCO stores `xywh`, so convert once when loading and never again |
| Masks on disk | COCO RLE via `pycocotools`, at original resolution | Standard and compact; COCO/LVIS ground truth already uses it |
| Masks in memory | Low-resolution logits from the decoder; binary mask = `logit > 0` | Keeps the soft values the corrector needs |
| Object ID | `obj_id` is a string: `"coco:123456"` / `"lvis:98765"` | COCO and LVIS annotation IDs can collide. `image_id` stays the shared COCO integer |
| File formats | JSONL for records and detections, Parquet for results, safetensors for features | Readable with pandas or one standard library; safetensors avoids pickles |
| Object filtering | Drop `iscrowd=1` and objects under 32x32 px, **applied once, in the object records** | Nobody filters differently later |

**[Proposed]** clarifications:
- Box coordinates are continuous pixel edges (`x2 = x + w` from COCO `xywh`), not pixel indices. Points `(x, y)` use the same coordinate system.
- IoU is always computed at original resolution: `to_full_res(logits) > 0` against the GT mask.
- Persistent artifacts (detections JSONL, results Parquet) never contain tensors or logits.
- Relative paths in configs are relative to the roots in `configs/paths/*.yaml` (`DATA_ROOT`, `CACHE_ROOT`, `CKPT_ROOT`, `RUNS_ROOT`).

---

## 1. Object record (owner: P2, Shresth)

Stored in `data/objects_{train,val}.jsonl`, one line per object.

```python
@dataclass
class ObjectRecord:
    obj_id: str           # "coco:123456"
    image_id: int         # COCO image id (LVIS shares it)
    source: str           # "coco" | "lvis"
    category_id: int
    category_name: str    # also used as the text prompt for SAM 3
    novelty: str          # "seen" | "unseen" (RQ3 split)
    gt_box: list[float]   # xyxy, pixels
    gt_mask: dict         # COCO RLE
    area: float
    # stratification attributes, computed once by P2
    size_bin: str         # "small" | "medium" | "large" (COCO thresholds)
    thinness: float       # perimeter^2 / area
    n_neighbors: int      # same-class objects within one box-width
```

- **Why the stratification fields live here:** P4's sensitivity analysis and P3's result breakdowns both need size, thinness and crowdedness. Computing them once gives both identical values.
- **Text phrase for SAM 3** = `category_name` with underscores replaced by spaces. For LVIS, use the first synonym only.
- **[Proposed]** `gt_mask` RLE `counts` must be stored as a `str` in JSONL (pycocotools returns `bytes`; decode before dumping, re-encode on load).
- **[Open]** O1 (size filter vs. `size_bin`), O2 (image-level splits), O10 (definitions of `thinness` and `n_neighbors`).

---

## 2. Prompt (owner: P4, Aman)

```python
@dataclass
class Prompt:
    box: list[float] | None = None                 # xyxy
    points: list[tuple[float, float, int]] = field(default_factory=list)
                                                   # (x, y, label) 1=pos, 0=neg
    mask_logits: "Tensor | None" = None            # previous low-res logits, optional
    origin: str = "gt"      # "gt" | "perturbed" | "sam3_text" | "llmdet" |
                            # "oracle" | "baseline:<name>" | "corrector"
    perturbation: dict | None = None               # {"type": "shift", "params": {...}}
```

- **One `points` list with labels**, not separate positive/negative lists. It matches SAM's `point_coords` / `point_labels` input directly, and click order is preserved, which matters for the ideal-click simulator.
- **Severity is not stored on the prompt.** It is computed by one shared function owned by P4:

```python
describe(prompt, obj) -> {box_iou, coverage, contamination, delta}
```
  - `box_iou`: IoU of the prompt box with the GT box.
  - `coverage`: fraction of the GT mask inside the box (truncation).
  - `contamination`: fraction of the box covered by other objects' masks (distractor inclusion).
  - `delta`: `(dcx/w, dcy/h, log s_w, log s_h)` relative to the GT box.
  - RQ2 depends on synthetic and real errors being measured by the same code, hence one function.
- **[Open]** O3: `describe` needs the other objects in the image to compute `contamination`; its signature takes only `(prompt, obj)`. Point-only prompts have no box, so the box fields must be defined (NaN/None).

---

## 3. Segmenter (owner: P1, Srikrishnan)

```python
@dataclass
class SegOutput:
    mask_logits: Tensor            # decoder's native low-res logits
    iou_score: float
    presence_score: float | None   # None on the SAM 2.1 backend

class Segmenter:
    def __init__(self, backend: str = "sam3"): ...        # "sam3" | "sam21"
    def set_image(self, image_id: int) -> None: ...       # loads cached features, no encoder run
    def predict(self, prompts: list[Prompt], multimask: bool = False) -> list[SegOutput]: ...
    def detect_text(self, phrase: str) -> list[tuple[list[float], dict, float]]: ...
                                                          # (box, mask_rle, score)
    def to_full_res(self, logits: Tensor) -> Tensor: ...  # upsample to original image size
```

- **Batch over prompts for one image.** The ideal-click simulator and the perturbation sweeps both need many prompts per image. Each `predict` call is only a decoder run.
- **`detect_text` lives here**, not in P2's code, because it is a SAM 3 call. LLMDet gets its own small wrapper returning the same `(box, score)` shape (no mask).
- Run under `torch.no_grad()` by default. Gradients through the prompt encoder are for the stretch goal only; add a flag when someone needs it.
- Fallback: on SAM 2.1 `presence_score` is `None`, so consumers must handle `None`.
- **[Proposed]** A `DummySegmenter` (GT mask, slightly eroded/dilated) implements this same interface so everyone can develop and test on a CPU. Its logits are synthesized so `to_full_res(...) > 0` gives the dummy mask.
- **[Proposed]** Nobody outside `segmenters/` resizes, pads or rescales logits or coordinates. Mapping to and from the model's input resolution lives only in the wrapper and `to_full_res`.
- **[Implemented in the scaffold, tentative]** `Segmenter` is an abstract base class (`segmenters/base.py`); backends are built with `make_segmenter(backend, **kwargs)` (`"dummy" | "sam3" | "sam21"`) instead of `Segmenter(backend=...)`. `SegOutput.mask_logits` has shape `[h, w]`; `SegOutput` also has optional `alt_logits` / `alt_ious` (the suggested fix for O5). `Segmenter.full_res_mask(output)` is a convenience returning a boolean `[H, W]` mask.
- **[Open]** O5 (multimask return shape), O6 (where `presence_score` comes from), O7 (cache scope and cache-miss behaviour), O13 (internal box conventions).

---

## 4. Cache layout (owner: P1, Srikrishnan)

```
cache/
  manifest.json                                    # backend, checkpoint name + hash, input res, dtype, date
  features/{backend}/{image_id}.safetensors        # fp16
  detections/sam3_text/{split}.jsonl               # {image_id, phrase, box, mask_rle, score}
  detections/llmdet/{split}.jsonl                  # {image_id, phrase, box, score}
```

- **One file per image** to start. Switch to shards of about 500 images only if the cluster complains about file counts.
- **`manifest.json` is the one non-optional extra.** Everyone checks it on load, so a cache built with the wrong checkpoint or resolution fails loudly instead of silently giving wrong numbers.
- **Detections: one JSONL per model per split**, not one per image, because P2 and P3 read them all at once.
- `backend` is `sam3` or `sam21`. `split` is `train` or `val`.
- **[Proposed]** The safetensors tensor key names and shapes are documented in the manifest by P1, so consumers don't guess.
- **[Open]** O8 (manifest per backend, concurrent writes, atomic writes), O9 (which phrases get run per image).

Example detection lines:
```json
{"image_id": 397133, "phrase": "person", "box": [12.0, 34.5, 200.2, 410.0], "mask_rle": {"size": [427, 640], "counts": "..."}, "score": 0.91}
{"image_id": 397133, "phrase": "person", "box": [12.0, 34.5, 200.2, 410.0], "score": 0.88}
```

---

## 5. Results row (owner: P3, Divya)

Long format: **one row per (object, method, step)**, saved as a Parquet file per run (see O11 for the location).

| Column | Notes |
|---|---|
| `run_id`, `obj_id`, `image_id`, `method`, `step` | `step` 0 = initial mask before any correction |
| `iou`, `boundary_iou` | one shared `metrics.py`; nobody re-implements IoU |
| `prompt_origin`, `pert_type`, `pert_level` | from the Prompt |
| `box_iou_in`, `coverage`, `contamination` | from `describe()`, so results can be binned by severity |
| `novelty`, `size_bin`, `thinness` | copied from the object record so grouping needs no join |
| `n_decoder_calls`, `latency_ms` | cost columns |

- **Store only raw values.** Success@0.75, harm rate and transfer ratio are computed when analyzing results, from these columns.
- **Harm** is defined once: `iou_final < iou_step0 - 0.01`.
- **Shared results sheet:** one row per *run* only: `run_id`, config path, git commit, headline numbers. Per-object data stays in the Parquet files.
- **[Proposed]** dtypes and semantics: `run_id`/`obj_id`/`method`/`prompt_origin`/`pert_type`/`novelty`/`size_bin` string; `image_id`/`step`/`n_decoder_calls` int; the rest float; `pert_type`/`pert_level` null for non-perturbed prompts; `n_decoder_calls` cumulative up to and including this step; `latency_ms` for this step only, excluding the image encoder; `iou_final` = IoU at the method's last step for that object.
- **[Open]** O4 (Δ-vector columns needed for RQ2), O11 (where files live), O12 (method names).

---

## 6. Run config (defaults owned by P2)

```yaml
run_id: rq2_mixed_s0
git_commit: auto
seed: 0
backend: sam3
train_data: mixed          # synthetic | real | mixed | matched
perturbations: [shift, scale, truncate, distractor, wrong_instance, point]
max_steps: 3
splits_file: data/splits.json   # fixed once, never regenerated
```

- Save the config next to each run's Parquet file.
- Plain YAML loaded into a dataclass. No config framework.
- **[Proposed]** `git_commit: auto` is resolved to the real commit hash (plus a dirty flag) at launch, and the resolved config is what gets saved. Unknown keys raise an error.
- **[Open]** O2: the content/schema of `splits.json` is not defined yet.

---

## 7. Functions that must have exactly one implementation

| Function | Owner |
|---|---|
| `iou()`, `boundary_iou()` | P3 |
| `describe(prompt, obj)` (severity) | P4 |
| `match(preds, gts)` (Hungarian on mask IoU) | P3 |
| `Segmenter` | P1 |

If two people write their own IoU or severity function, numbers in different report sections won't agree.

**[Proposed]** additions to this list: COCO `xywh` -> `xyxy` and RLE encode/decode helpers (P2, in `data/`); YAML config loader and seed setter (P1). **[Open]** `match` output contract and `boundary_iou` dilation ratio are for P3 to fix and write down here.

---

## 8. Ownership summary

| Interface | Owner | Main consumers |
|---|---|---|
| ObjectRecord, run-config defaults, splits | P2 (Shresth) | everyone |
| Prompt, `describe()` | P4 (Aman) | P3, P5 |
| Segmenter, cache layout | P1 (Srikrishnan) | everyone, P5 most |
| Results row, `iou`, `match` | P3 (Divya) | everyone |
| Corrector inputs/outputs | P5 (Sharvani) | P4 (training variants) |

---

## 9. Open issues found while transcribing (need a decision)

Recommended resolutions are suggestions, not decisions. **Before coding starts:** O1–O4, O6–O8. The rest can be settled this week.

| # | Issue | Why it matters | Recommended resolution | Owner |
|---|---|---|---|---|
| O1 | "Drop objects under 32x32 px" vs. `size_bin` using COCO thresholds (small = area < 32^2) | If the filter is on area, the `small` bin is empty; if on box size it's ambiguous. Text-prompt matching also sees predictions for the dropped objects | Define as `box_w < 32 and box_h < 32`. Keeps thin, long objects. `match` must ignore predictions that overlap dropped/crowd objects | P2, P3 |
| O2 | `splits.json` content is undefined, and LVIS-val images overlap COCO-train2017 images | Without image-level splits, RQ3 is contaminated by seen images | Image-level split: `{"train": [ids], "val": [ids], "lvis_val_excluded_from_train": [ids]}`. Both COCO and LVIS objects for an image go to that image's split | P2 |
| O3 | `describe(prompt, obj)` can't compute `contamination` without the other objects; point-only prompts have no box | Wrong or missing severity values | Signature `describe(prompt, obj, neighbors)`; box-based fields are `NaN` when `box is None` | P4 |
| O4 | Results row has no Δ-vector columns | RQ2 compares real vs. synthetic error distributions (Wasserstein/MMD over Δ) and does the same-object study | Add `delta_cx, delta_cy, delta_log_sw, delta_log_sh` (and `n_points`) | P3, P4 |
| O5 | `predict(multimask=True)` returns `list[SegOutput]` with one mask per prompt | Multimask yields K masks per prompt | Keep the best mask in `SegOutput`; add optional `alt_logits`, `alt_ious` filled only when `multimask=True` | P1 |
| O6 | `presence_score` probably exists only on SAM 3's text/detector path, not on the interactive head | The corrector input list assumes it for every step | Verify. If so, `predict()` returns `None` and presence comes from `detect_text` per image/phrase; update the corrector's input spec | P1, P5 |
| O7 | Cache scope: do the cached features serve both the interactive head and `detect_text`? What happens on a cache miss? | `set_image` promises "no encoder run" | Document what is cached per backend; on a miss, raise by default with an opt-in flag to run the encoder live | P1 |
| O8 | Single `manifest.json` for two backends; array jobs may race; killed jobs leave partial files | Silent wrong numbers or corrupt reads | Manifest keyed by backend, written once at job start; write each file to a temp name then rename | P1 |
| O9 | Which phrases run on each image for detections | Determines whether false positives exist in the study | Run only the categories present in the image's GT objects (false-positive analysis out of scope); record this in the manifest | P2, P3 |
| O10 | `thinness` and `n_neighbors` aren't precisely defined | Two people would compute different values | `thinness` = (total mask contour length)^2 / mask area at original resolution. `n_neighbors` = other same-class objects in the image whose box centre is within one box-width of this box's centre | P2 |
| O11 | Plan says `results/{run_id}.parquet`; `CLAUDE.md` says only small files go in git `results/` | Per-object data on scratch would otherwise be committed or lost | Write to `$RUNS_ROOT/{run_id}/{results.parquet, config.yaml}` on scratch. Commit only summary tables/figures and a `results/runs_index.csv` (or a shared sheet) | P3 |
| O12 | `method` names and `match()` output contract aren't fixed | Group-by breaks if names drift | P3 keeps a registry of stable method names (`b0_none`, `b1_random_k`, ...) and documents `match` output (matched pairs with IoU, plus unmatched predictions and GTs) | P3 |
| O13 | SAM 3's processor and interactive predictor may use different box conventions (for example normalized `cxcywh` vs. `xyxy` pixels), and some SAM variants resize without keeping aspect | Coordinate bugs that are hard to spot | Hide all of it inside the wrapper; add a unit test that round-trips a box through `predict` + `to_full_res` on a known image. Verify against the SAM 3 code | P1 |

Notes outside the formats, for the extra datasets: images from ThinObject/DIS/SA-Co have no COCO integer `image_id`; if they are added later (stretch), `image_id` needs a namespace such as `"dis:0123"`.

---

## 10. Changelog

- 2026-10-23: v0.1 draft transcribed from `CV_Project_Revised_Plan.pdf`; open issues O1–O13 collected.
- 2026-10-23: scaffold implements these in `src/promptcorrect/types.py`: `ResultRow` includes the proposed Δ columns (O4); `Segmenter` is an ABC with a `make_segmenter` factory and `SegOutput.alt_*` fields (O5). Review and either accept or revise.
