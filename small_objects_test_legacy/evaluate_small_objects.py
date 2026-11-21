#!/usr/bin/env python3
"""Strict test-only, inference-only evaluation. Run --help or read README.md."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evaluation_data import audit_dataset, existing_path, inventory, object_hash, sha256_file, write_json
from scale_metrics import DEFINITION, EVALUATOR, GROUPS, IOU_THRESHOLDS, evaluate, validate_predictions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_SCHEMA = "paper_original_xyxy_predictions_v1"
LABELS = {"all": "All objects", "small": "Small objects", "non-small": "Non-small objects"}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "configs/data.yaml")
    p.add_argument("--weights", nargs="+", help="One or more local .pt paths or quoted glob patterns; default: recursive small_objects_test_legacy/*.pt")
    p.add_argument("--list-weights", action="store_true", help="List/checkpoint-hash available weights without importing PyTorch")
    p.add_argument("--audit-only", action="store_true", help="Audit every test image/label; do not import or run the detector")
    p.add_argument("--check-only", action="store_true", help="Audit data and load selected checkpoints; no forward pass or AP results")
    p.add_argument("--cache-in", nargs="+", type=Path, help="Recompute custom metrics from compatible complete prediction caches; no model/GPU import")
    p.add_argument("--save-cache", action="store_true", help="Save original-image predictions and generating configuration")
    p.add_argument("--output", type=Path, default=HERE / "evaluation_runs", help="Parent directory; a fresh UTC subdirectory is always created")
    p.add_argument("--device", default="0", help="Single visible GPU index, cuda:index, or explicit cpu; never silently fall back")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--stride", type=int, default=32, help="Common preprocessing stride; must be divisible by every model stride")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--ap-conf", type=float, default=0.001, help="NMS prediction collection uses score > this value")
    p.add_argument("--recall-conf", type=float, default=0.25, help="Recall uses score >= this value, separately from AP")
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300, help="Maximum post-NMS detections per image across all classes")
    p.add_argument("--max-nms", type=int, default=30000, help="Native pre-NMS candidate cap")
    p.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True,
                   help="Match YOLO.model.val() rectangular batching; --no-rect reproduces the first evaluator release")
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--single-label", action="store_true", help="Use best class per proposal instead of native val multi-label NMS")
    p.add_argument("--no-fuse", action="store_true", help="Disable in-memory inference Conv/BN fusion")
    p.add_argument("--invalid-labels", choices=("error", "skip"), default="error")
    p.add_argument("--missing-labels", choices=("error", "empty"), default="error")
    p.add_argument("--boundary-tolerance", type=float, default=1e-6,
                   help="Normalized corner roundoff tolerance; accepted GT values are retained unchanged; use 0 for strict bounds")
    return p


def select_weights(args):
    paths = []
    if args.weights:
        for token in args.weights:
            matched = glob.glob(str(Path(token).expanduser()), recursive=True)
            if not matched:
                raise FileNotFoundError(f"No weight matches {token}")
            paths.extend(existing_path(p) for p in sorted(matched))
    else:
        paths = sorted(HERE.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt weights under {HERE}; provide --weights")
    if any(p.suffix.lower() != ".pt" or not p.is_file() for p in paths):
        raise ValueError("All weights must be existing .pt files")
    if len(set(paths)) != len(paths):
        raise ValueError("The same checkpoint was specified more than once")
    return paths


def inference_config(args):
    return {"imgsz": args.imgsz, "batch": args.batch, "workers": args.workers, "stride": args.stride,
            "precision": args.precision, "ap_conf": args.ap_conf, "nms_iou": args.nms_iou,
            "max_det": args.max_det, "max_nms": args.max_nms, "rect": args.rect,
            "agnostic_nms": args.agnostic_nms, "multi_label": not args.single_label, "fuse": not args.no_fuse,
            "augment": False, "max_wh": 7680, "nms_timeout": "disabled",
            "preprocess": "native YOLODataset validation resize/LetterBox/Format; RGB float /255",
            "prediction_coordinates": "stored original image, xyxy, separate x/y inverse gains, clipped",
            "gt_coordinates": "stored original image, xyxy, unmodified original normalized labels in float64"}


def percent(value):
    return "N/A" if value is None else value * 100


def percentage_difference(first, second):
    if first is None or second is None:
        return "N/A"
    return (first - second) * 100


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, run):
    long_rows, wide_rows, class_rows, native_rows = [], [], [], []
    for model in run["models"]:
        if "groups" not in model:
            continue
        wide = {"Model": model["model"]}
        for group in GROUPS:
            g = model["groups"][group]
            long_rows.append({"Model": model["model"], "Group": LABELS[group],
                              "Ground-truth objects": g["ground_truth_objects"], "Evaluated classes": g["evaluated_classes"],
                              "Evaluated class names": "|".join(g["evaluated_class_names"]),
                              "mAP50 (%)": percent(g["map50_0_1"]), "mAP50–95 (%)": percent(g["map50_95_0_1"]),
                              "Recall@0.5 (%)": percent(g["recall_at_0_5_0_1"]), "TP": g["tp"], "FN": g["fn"], "FP": g["fp"],
                              "mAP50 (0–1)": g["map50_0_1"] if g["map50_0_1"] is not None else "N/A",
                              "mAP50–95 (0–1)": g["map50_95_0_1"] if g["map50_95_0_1"] is not None else "N/A",
                              "Recall confidence": run["recall_conf"], "Recall averaging": "micro: sum TP / sum GT",
                              "Evaluator": EVALUATOR})
            prefix = {"all": "", "small": "Small-object ", "non-small": "Non-small-object "}[group]
            wide[prefix + "mAP50 (%)"] = percent(g["map50_0_1"])
            wide[prefix + "mAP50–95 (%)"] = percent(g["map50_95_0_1"])
            for c in g["per_class"]:
                class_rows.append({"Model": model["model"], "Group": LABELS[group], "Class ID": c["class_id"],
                                   "Class": c["class_name"], "Ground-truth objects": c["ground_truth_objects"],
                                   "Included in mAP": c["included_in_map"], "Evaluated classes": g["evaluated_classes"],
                                   "AP50 (%)": percent(c["ap50_0_1"]), "AP50–95 (%)": percent(c["ap50_95_0_1"]),
                                   "Recall@0.5 (%)": percent(c["recall_at_0_5_0_1"]), "TP": c["tp"], "FN": c["fn"], "FP": c["fp"],
                                   "Ignored at Recall confidence": c["ignored_at_recall_conf"]})
        wide["Evaluator"] = EVALUATOR
        wide_rows.append(wide)
        native, all_group = model["native_reference"], model["groups"]["all"]
        native_rows.append({"Model": model["model"], "Native mAP50 (%)": percent(native["map50_0_1"]),
                            "Native mAP50–95 (%)": percent(native["map50_95_0_1"]),
                            "Custom All mAP50 (%)": percent(all_group["map50_0_1"]),
                            "Custom All mAP50–95 (%)": percent(all_group["map50_95_0_1"]),
                            "Custom minus native AP50 (percentage points)": percentage_difference(all_group["map50_0_1"], native["map50_0_1"]),
                            "Custom minus native AP50–95 (percentage points)": percentage_difference(all_group["map50_95_0_1"], native["map50_95_0_1"]),
                            "Native status": native["status"], "Native evaluator": native["evaluator"],
                            "Rect": run["inference_config"]["rect"]})
    for name, rows in (("metrics_long.csv", long_rows), ("paper_wide.csv", wide_rows),
                       ("per_class.csv", class_rows), ("native_comparison.csv", native_rows)):
        save_csv(output / name, rows)
    write_json(output / "results.json", run)


def load_cache(path, audit, recall_conf):
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"Unsupported prediction-cache schema: {path}")
    if cache["dataset_fingerprint"] != audit["dataset_fingerprint"] or cache["names"] != audit["names"]:
        raise ValueError("Cache differs from current full test manifest, dimensions, labels, classes or annotation policy")
    if recall_conf <= cache["inference_config"]["ap_conf"]:
        raise ValueError("Recall threshold must exceed cache AP collection threshold (collection uses >)")
    if object_hash(cache["predictions"]) != cache["predictions_sha256"]:
        raise ValueError("Prediction-cache content hash mismatch")
    validate_predictions(audit["records"], cache["predictions"], audit["names"],
                         cache["inference_config"]["max_det"], cache["inference_config"]["ap_conf"])
    return cache


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.imgsz, args.batch, args.stride, args.max_det, args.max_nms) <= 0 or args.workers < 0:
        raise ValueError("Sizes/batch/detection caps must be positive and workers nonnegative")
    if args.imgsz % args.stride:
        raise ValueError("imgsz must be an exact multiple of shared stride; no silent rounding")
    if (not 0 <= args.ap_conf < 1 or not 0 < args.recall_conf <= 1 or not 0 < args.nms_iou <= 1
            or (not args.cache_in and args.ap_conf >= args.recall_conf)):
        raise ValueError("Require 0 <= ap-conf < recall-conf <= 1 and 0 < nms-iou <= 1")
    if not 0 <= args.boundary_tolerance <= 1e-4:
        raise ValueError("boundary-tolerance must be between 0 and 1e-4 normalized units")
    if sum((args.audit_only, args.check_only, bool(args.cache_in), args.list_weights)) > 1:
        raise ValueError("Choose just one of audit-only/check-only/cache-in/list-weights")
    if args.cache_in:
        if args.weights or args.save_cache:
            raise ValueError("cache-in cannot be combined with weights/save-cache")
        inference_flags = {"--device", "--imgsz", "--batch", "--workers", "--stride", "--precision", "--ap-conf",
                           "--nms-iou", "--max-det", "--max-nms", "--rect", "--no-rect", "--agnostic-nms", "--single-label", "--no-fuse"}
        tokens = sys.argv[1:] if argv is None else argv
        if any(t.split("=", 1)[0] in inference_flags for t in tokens):
            raise ValueError("Cache recomputation uses its generating inference configuration; only Recall threshold may change")
    weights = [] if args.cache_in or args.audit_only else inventory(select_weights(args))
    for weight in weights:
        print(f"WEIGHT {weight['path']} ({weight['bytes']} bytes) SHA256={weight['sha256']}", flush=True)
    if args.list_weights:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output.expanduser().resolve() / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    run = {"schema": "paper_size_evaluation_run_v1", "status": "running", "started_utc": stamp,
           "protocol_revision": "2026-10-11-rect-default-matches-model-val",
           "command": sys.argv if argv is None else argv, "project_root": str(ROOT), "split": "test",
           "evaluator": EVALUATOR, "metric_units_json": "0_to_1; null means N/A", "metric_units_csv": "explicit in column names",
           "small_definition": DEFINITION, "groups": list(GROUPS), "iou_thresholds": IOU_THRESHOLDS.tolist(),
           "ap_interpolation": "precision envelope sampled at recall 0:0.01:1, arithmetic mean of 101 samples",
           "map_averaging": "macro over classes with valid GT in the current group; absent classes excluded and N/A",
           "recall_conf": args.recall_conf, "recall_averaging": "micro: total TP / total GT at IoU 0.5, score >= recall_conf",
           "matching": "per-image/class score-greedy 1:1; valid GT before ignored GT; unmatched predictions filtered by their original size",
           "weights": weights, "models": []}
    run["evaluation_software"] = {"python": sys.version, **{p: version(p) for p in ("numpy", "Pillow", "PyYAML")}}
    run["evaluation_source_sha256"] = {name: sha256_file(HERE / name) for name in
                                        ("evaluate_small_objects.py", "evaluation_data.py", "scale_metrics.py", "project_inference.py")}
    try:
        audit = audit_dataset(args.data, args.invalid_labels, args.missing_labels, args.boundary_tolerance)
        write_json(output / "dataset_audit.json", audit)
        (output / "test_images.txt").write_text("\n".join(r["image_path"] for r in audit["records"]) + "\n", encoding="utf-8")
        run["dataset_fingerprint"], run["names"] = audit["dataset_fingerprint"], audit["names"]
        run["data_yaml"], run["annotation_policy"] = audit["data_yaml"], audit["annotation_policy"]
        print(f"TEST: {audit['image_count']} images, {audit['ground_truth_objects']} valid GT; "
              f"invalid rows={audit['counts'].get('invalid_label_rows', 0)}, "
              f"boundary roundoff retained={len(audit['boundary_rounding_rows'])}, fatal issues={audit['fatal_issue_count']}", flush=True)
        if audit["fatal_issue_count"]:
            raise ValueError(f"Dataset audit failed; see {output / 'dataset_audit.json'}. No images/labels were changed.")
        if args.audit_only:
            run["status"] = "dataset_audit_passed_no_inference"
            export_results(output, run)
            return 0
        config = inference_config(args)
        if not args.cache_in:
            # Select the local checkout before importing Ultralytics.
            sys.path.insert(0, str(ROOT))
            sys.dont_write_bytecode = True
            # Pre-create the settings directory to keep writes inside this run.
            (output / "runtime_config/Ultralytics").mkdir(parents=True)
            (output / "matplotlib_config").mkdir()
            os.environ["YOLO_AUTOINSTALL"] = "false"
            os.environ["YOLO_OFFLINE"] = "true"
            os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime_config")
            os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_config")
            os.environ["WANDB_MODE"] = "disabled"
            import project_inference as backend
            device, runtime = backend.environment(ROOT, args.device, args.precision, output)
            run["runtime"], run["inference_config"] = runtime, config
            print(f"Shared inference protocol: {json.dumps(config, ensure_ascii=False)}", flush=True)
            print("paper_wide.csv: custom COCO-style AP for ALL groups. "
                  "Compare previous model.val() results with the Native columns in native_comparison.csv; "
                  f"current rect={config['rect']}.", flush=True)
        cache_signature = None
        for index, task in enumerate(args.cache_in or weights, 1):
            model_output = output / f"model_{index:02d}"
            model_output.mkdir()
            if args.cache_in:
                cache = load_cache(existing_path(task), audit, args.recall_conf)
                signature = object_hash({"config": cache["inference_config"], "runtime": {
                    k: cache["runtime"].get(k) for k in ("source_fingerprint", "torch", "torchvision", "numpy", "opencv",
                                                       "cuda_build", "device", "precision", "gpu")}})
                if cache_signature is not None and signature != cache_signature:
                    raise ValueError("Batch caches were generated with different source/software/device/inference protocols")
                cache_signature = signature
                predictions, native, metadata = cache["predictions"], cache["native_reference"], cache["model_metadata"]
                native["status"] = "loaded_from_generating_run_not_rerun"
                weight, config, runtime = cache["weight"], cache["inference_config"], cache["runtime"]
                run["inference_config"], run["generating_runtime"] = config, runtime
            else:
                weight = task
                predictions, native, metadata = backend.infer(Path(weight["path"]), audit, config, device,
                                                                model_output, ROOT, args.check_only)
                if sha256_file(weight["path"]) != weight["sha256"]:
                    raise RuntimeError("Checkpoint changed during evaluation")
            model_id = weight["name"]
            if any(m["model"] == model_id for m in run["models"]):
                model_id += "_" + weight["sha256"][:10] + f"_{index}"
            item = {"model": model_id, "weight": weight, "model_metadata": metadata}
            if args.check_only:
                item["status"] = "checkpoint_loaded_no_forward_no_metrics"
            else:
                if not args.cache_in:  # load_cache already validates cached predictions.
                    validate_predictions(audit["records"], predictions, audit["names"], config["max_det"], config["ap_conf"])
                item.update({"status": "metrics_computed", "native_reference": native,
                             "groups": evaluate(audit["records"], predictions, audit["names"], args.recall_conf)})
                if args.save_cache:
                    cache = {"schema": CACHE_SCHEMA, "generated_utc": stamp, "weight": weight,
                             "dataset_fingerprint": audit["dataset_fingerprint"], "names": audit["names"],
                             "data_yaml": audit["data_yaml"], "data_yaml_sha256": audit["data_yaml_sha256"],
                             "annotation_policy": audit["annotation_policy"], "small_definition": DEFINITION,
                             "test_manifest": [{k: r[k] for k in ("id", "width", "height", "image_sha256", "label_sha256")}
                                               for r in audit["records"]],
                             "inference_config": config, "runtime": runtime, "model_metadata": metadata,
                             "native_reference": native, "predictions": predictions, "predictions_sha256": object_hash(predictions)}
                    write_json(model_output / "predictions_original_xyxy.json", cache)
                if args.cache_in:
                    item["cache_source"] = str(task.resolve())
            write_json(model_output / "model_results.json", item)
            run["models"].append(item)
            export_results(output, run)  # preserve finished models if a later model fails
        run["status"] = "checkpoint_preflight_passed_no_inference" if args.check_only else "completed"
        run["completed_utc"] = datetime.now(timezone.utc).isoformat()
        export_results(output, run)
        print(f"{run['status']}: {output}", flush=True)
        return 0
    except Exception as exc:
        run.update({"status": "failed", "error": str(exc)})
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        export_results(output, run)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
