"""SAM 3 smoke test (sprint gate G1). Run on a GPU node; see docs/environment.md.

Answers, with numbers, the questions the Segmenter wrapper depends on:
  * Do box / point / box+point prompts work through `model.predict_inst`, and is a box really xyxy pixels?
  * What shapes and dtypes come back (full-res masks, 256x256 low-res logits, scores)?
  * Does the predicted IoU score track the real IoU? Do ideal clicks improve a bad (truncated) box?
  * Does the text path work, what box format / keys does it return, and is there a presence score (O6)?
  * How long does one image encode and one decoder call take, and how much GPU memory?

Each section runs independently: a failure in one prints its traceback and the script continues.
Writes $RUNS_ROOT/smoke_sam3/summary.json and overlay.png (paths come from configs/paths/greatlakes.yaml).

    python scripts/smoke_sam3.py                 # uses PC_PATHS (set PC_PATHS=greatlakes on the cluster)
    python scripts/smoke_sam3.py --n-objects 5 --ckpt /path/to/sam3.pt
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage


def oracle_click(pred: np.ndarray, gt: np.ndarray):
    """Ideal click: in the larger error region (false negative -> positive click, false positive ->
    negative click), the point farthest from that region's boundary. None if there is no error."""
    fn, fp = gt & ~pred, pred & ~gt
    if not fn.any() and not fp.any():
        return None
    region, label = (fn, 1) if fn.sum() >= fp.sum() else (fp, 0)
    comp, n = ndimage.label(region)
    sizes = ndimage.sum(region, comp, range(1, n + 1))
    biggest = comp == (1 + int(np.argmax(sizes)))
    dist = ndimage.distance_transform_edt(np.pad(biggest, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y), label


def pick_objects(ann_file: Path, n: int):
    """First n non-crowd, medium/large COCO val objects from distinct images."""
    from pycocotools.coco import COCO

    coco = COCO(str(ann_file))
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    picked, seen = [], set()
    for img_id in sorted(coco.getImgIds()):
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            if 20000 <= ann["area"] <= 200000 and img_id not in seen:
                img = coco.loadImgs(img_id)[0]
                same = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=ann["category_id"], iscrowd=False))]
                picked.append(dict(img=img, ann=ann, cat=cats[ann["category_id"]], n_same_class=len(same),
                                   gt=coco.annToMask(ann).astype(bool)))
                seen.add(img_id)
                break
        if len(picked) == n:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="path to sam3.pt (default: <ckpt_root>/sam3/sam3.pt)")
    ap.add_argument("--n-objects", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="allow running without a GPU (very slow)")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from promptcorrect.config import load_paths
    from promptcorrect.data.geometry import xywh_to_xyxy
    from promptcorrect.eval.metrics import iou

    paths = load_paths()
    out_dir = paths.runs_root / "smoke_sam3"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt) if args.ckpt else paths.ckpt_root / "sam3" / "sam3.pt"
    ann_file = paths.data_root / "coco" / "annotations" / "instances_val2017.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.cpu:
        raise SystemExit("No GPU visible. Run on a GPU node (see docs/environment.md) or pass --cpu.")

    report: dict = {"device": device, "torch": torch.__version__, "ckpt": str(ckpt)}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def section(name, fn):
        t = time.time()
        try:
            out = fn() or {}
            report[name] = {"ok": True, "seconds": round(time.time() - t, 3), **out}
            print(f"[ OK ] {name}: {out}", flush=True)
            return out
        except Exception as e:  # keep going: report everything we can in one run
            report[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            print(f"[FAIL] {name}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            return None

    def binary(m):  # predict_inst masks may come back as bool, 0/1 floats or logits
        return np.asarray(m).squeeze() > 0.5

    # ---- inputs -------------------------------------------------------------------------------
    objs = pick_objects(ann_file, args.n_objects)
    report["objects"] = [dict(image_id=o["img"]["id"], category=o["cat"], area=o["ann"]["area"],
                              n_same_class=o["n_same_class"]) for o in objs]
    print("objects:", report["objects"], flush=True)
    if not objs:
        raise SystemExit(f"no objects found in {ann_file}")

    # ---- model --------------------------------------------------------------------------------
    holder: dict = {}

    def load_model():
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t = time.time()
        model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False,
                                       enable_inst_interactivity=True, device=device)
        sync()
        holder["model"], holder["processor"] = model, Sam3Processor(model, device=device)
        out = {"load_seconds": round(time.time() - t, 2)}
        if device == "cuda":
            out["gpu_mem_gb_after_load"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        return out

    if section("load_model", load_model) is None:
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
        raise SystemExit("model failed to load; see the traceback above")
    model, processor = holder["model"], holder["processor"]

    def predict(state, point_coords=None, point_labels=None, box=None, mask_input=None, multimask=False,
                return_logits=False):
        with torch.inference_mode():
            return model.predict_inst(
                state,
                point_coords=None if point_coords is None else np.asarray(point_coords, np.float32),
                point_labels=None if point_labels is None else np.asarray(point_labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32)[None, :],
                mask_input=mask_input, multimask_output=multimask, return_logits=return_logits)

    # ---- per-object tests ---------------------------------------------------------------------
    first_overlay = {}
    for k, o in enumerate(objs):
        tag = f"obj{k}_{o['cat'].replace(' ', '_')}"
        gt = o["gt"]
        box = xywh_to_xyxy(o["ann"]["bbox"])
        image = Image.open(paths.data_root / "coco" / "val2017" / o["img"]["file_name"]).convert("RGB")
        state_holder = {}

        def encode():
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sync(); t = time.time()
            state_holder["state"] = processor.set_image(image)
            sync()
            out = {"encode_seconds": round(time.time() - t, 3), "image_size_hw": list(gt.shape)}
            if device == "cuda":
                out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            return out

        if section(f"{tag}/set_image", encode) is None:
            continue
        state = state_holder["state"]

        def box_prompt():
            masks, scores, low = predict(state, box=box)
            m = binary(masks)
            first_overlay.setdefault("box_mask", (image, m, box))
            return {"masks_shape": list(np.asarray(masks).shape), "masks_dtype": str(np.asarray(masks).dtype),
                    "scores_shape": list(np.asarray(scores).shape), "low_res_shape": list(np.asarray(low).shape),
                    "pred_iou_score": [round(float(s), 3) for s in np.atleast_1d(scores)],
                    "true_iou": round(iou(m, gt), 3),
                    "box_convention_ok": bool(iou(m, gt) > 0.5)}

        section(f"{tag}/box_prompt(xyxy pixels)", box_prompt)

        def box_logits():
            masks, scores, low = predict(state, box=box, return_logits=True)
            m_logit = np.asarray(masks).squeeze() > 0
            m_bin = binary(predict(state, box=box)[0])
            return {"logits_dtype": str(np.asarray(masks).dtype), "logit_min_max": [round(float(np.min(masks)), 2), round(float(np.max(masks)), 2)],
                    "logits>0 equals binary mask": bool((m_logit == m_bin).all()),
                    "low_res_logit_range": [round(float(np.min(low)), 2), round(float(np.max(low)), 2)]}

        section(f"{tag}/return_logits", box_logits)

        def point_only():
            dist = ndimage.distance_transform_edt(gt)
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            out = {}
            for mm in (False, True):
                masks, scores, _ = predict(state, point_coords=[[x, y]], point_labels=[1], multimask=mm)
                ms = np.asarray(masks).reshape(-1, *gt.shape)
                out[f"multimask={mm}"] = {"n_masks": int(ms.shape[0]), "true_iou": [round(iou(m > 0.5, gt), 3) for m in ms],
                                          "pred_iou_score": [round(float(s), 3) for s in np.atleast_1d(scores)]}
            return out

        section(f"{tag}/point_only", point_only)

        def box_plus_point_and_refine():
            dist = ndimage.distance_transform_edt(gt)
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            masks, scores, low = predict(state, point_coords=[[x, y]], point_labels=[1], box=box)
            first = iou(binary(masks), gt)
            masks2, _, _ = predict(state, point_coords=[[x, y]], point_labels=[1], box=box,
                                   mask_input=np.asarray(low)[:1])
            return {"box+point_iou": round(first, 3), "with_mask_input_iou": round(iou(binary(masks2), gt), 3)}

        section(f"{tag}/box_plus_point+mask_input", box_plus_point_and_refine)

        def click_loop():
            x0, y0, x1, y1 = box
            bad = [x0, y0, x0 + 0.6 * (x1 - x0), y1]  # truncated box: loses the right 40%
            pts, ious = [], []
            for step in range(4):
                masks, _, _ = predict(state, box=bad,
                                      point_coords=[p[:2] for p in pts] or None,
                                      point_labels=[p[2] for p in pts] or None)
                m = binary(masks)
                ious.append(round(iou(m, gt), 3))
                click = oracle_click(m, gt)
                if click is None:
                    break
                pts.append(click)
            return {"iou_after_k_ideal_clicks (k=0..)": ious, "clicks_xyl": [tuple(round(v, 1) for v in p) for p in pts]}

        section(f"{tag}/truncated_box+ideal_clicks", click_loop)

        def latency():
            rng = np.random.default_rng(0)
            n = 20
            sync(); t = time.time()
            for _ in range(n):
                jitter = rng.uniform(-0.05, 0.05, 4) * np.array([box[2] - box[0], box[3] - box[1]] * 2)
                predict(state, box=np.array(box) + jitter)
            sync()
            return {"decoder_ms_per_call": round((time.time() - t) / n * 1000, 1), "calls": n}

        if k == 0:
            section(f"{tag}/decoder_latency", latency)

        def text_prompt():
            st = processor.set_image(image)  # fresh state: set_text_prompt mutates it
            sync(); t = time.time()
            out = processor.set_text_prompt(prompt=o["cat"], state=st)
            sync()
            st = out if isinstance(out, dict) else st
            boxes = st.get("boxes")
            masks = st.get("masks")
            scores = st.get("scores")
            n = 0 if scores is None else int(len(scores))
            best = None
            if n:
                ious = [iou(np.asarray(m.cpu() if hasattr(m, "cpu") else m).squeeze() > 0.5, gt) for m in masks]
                best = round(max(ious), 3)
            # O6: the processor folds the presence score into `scores` and does not store it; call the
            # model's grounding forward once more to read the raw value and see what else is returned.
            with torch.inference_mode():
                raw = model.forward_grounding(backbone_out=st["backbone_out"], find_input=processor.find_stage,
                                              geometric_prompt=st["geometric_prompt"], find_target=None)
            presence = None
            if "presence_logit_dec" in raw:
                presence = round(float(raw["presence_logit_dec"].sigmoid().reshape(-1)[0]), 4)
            low_state = processor.set_confidence_threshold(0.1, st)
            res = {"forward_grounding_keys": sorted(raw.keys()), "presence_prob(raw, per prompt)": presence,
                   "NOTE": "stored `scores` = per-query prob x presence prob; `predict_inst` returns no presence",
"prompt": o["cat"], "gt_instances_of_class_in_image": o["n_same_class"], "n_detections@0.5": n,
                   "best_iou_vs_this_gt_object": best,
                   "scores": [round(float(s), 3) for s in (scores.cpu() if hasattr(scores, "cpu") else scores)][:8] if n else [],
                   "boxes_dtype/shape": None if boxes is None else [str(boxes.dtype), list(boxes.shape)],
                   "first_box": None if not n else [round(float(v), 1) for v in boxes[0].cpu().tolist()],
                   "gt_box_xyxy": [round(v, 1) for v in box],
                   "text_seconds": round(time.time() - t, 3),
                   "state_keys": sorted(st.keys()) if isinstance(st, dict) else str(type(st)),
                   "presence_like_keys": [kk for kk in (st.keys() if isinstance(st, dict) else []) if "presence" in kk.lower()]}
            if low_state is not None and isinstance(low_state, dict) and low_state.get("scores") is not None:
                res["n_detections@0.1"] = int(len(low_state["scores"]))
            processor.set_confidence_threshold(0.5, st)
            return res

        section(f"{tag}/text_prompt", text_prompt)

    # ---- artifacts ----------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "box_mask" in first_overlay:
            image, m, box = first_overlay["box_mask"]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(image)
            ov = np.zeros((*m.shape, 4)); ov[m] = (0.1, 0.5, 1.0, 0.55)
            ax.imshow(ov)
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="lime", lw=2))
            ax.axis("off")
            fig.savefig(out_dir / "overlay.png", dpi=110, bbox_inches="tight")
    except Exception as e:
        print("overlay skipped:", e)

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    n_ok = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok") is True)
    n_fail = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok") is False)
    print(f"\n{n_ok} sections OK, {n_fail} failed. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
