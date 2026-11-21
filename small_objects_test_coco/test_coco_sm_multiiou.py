"""COCO bounding-box AP for small/medium objects at IoU .50, .75 and .50:.95."""

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import sys
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCAL_SOURCE = HERE.parent
DEFAULT_WEIGHTS = (LOCAL_SOURCE / "experiments/BSQ-YOLO/weights/best.pt",)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
METRIC_SPECS = (
    ("AP50_S", "small", 0.50), ("AP50_M", "medium", 0.50),
    ("AP75_S", "small", 0.75), ("AP75_M", "medium", 0.75),
    ("AP50_95_S", "small", None), ("AP50_95_M", "medium", None),
)
METRIC_KEYS = tuple(spec[0] for spec in METRIC_SPECS)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_names(names):
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Class IDs must be consecutive integers starting at zero")
        names = [names[k] for k in range(len(names))]
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("data.yaml must define nonempty class names")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate class names")
    return list(names)


def build_test_annotations(dataset, names):
    """Read all test labels; convert normalized YOLO boxes to original-pixel COCO boxes."""
    from PIL import Image

    image_root, label_root = dataset / "test/images", dataset / "test/labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Missing test/images or test/labels under {dataset}")
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("The test split contains no supported images")
    gt = {"info": {"description": "Custom test split; original-image bounding boxes"},
          "licenses": [], "images": [], "annotations": [],
          "categories": [{"id": k + 1, "name": name} for k, name in enumerate(names)]}
    image_ids, label_paths, clipped_boxes = {}, set(), []
    for image_id, image_path in enumerate(images, 1):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        if label_path in label_paths:
            raise ValueError(f"Two images share the same label file: {label_path}")
        label_paths.add(label_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing annotation (not silently treated as background): {label_path}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Ambiguous EXIF orientation; check image/label alignment: {image_path}")
            width, height = image.size
        image_ids[str(image_path.resolve())] = image_id
        gt["images"].append({"id": image_id, "file_name": relative.as_posix(), "width": width, "height": height})
        seen = set()
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            values = tuple(float(x) for x in line.split())
            location = f"{label_path}:{line_no}"
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError(f"Expected five finite YOLO bbox values: {location}")
            cls, cx, cy, bw, bh = values
            if cls != int(cls) or not 0 <= cls < len(names):
                raise ValueError(f"Invalid class ID: {location}")
            if not all(0 <= x <= 1 for x in (cx, cy, bw, bh)) or min(bw, bh) <= 0:
                raise ValueError(f"Invalid normalized bbox: {location}")
            if values in seen:
                raise ValueError(f"Duplicate annotation: {location}")
            seen.add(values)
            corners = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if min(corners) < -1e-6 or max(corners) > 1 + 1e-6:
                raise ValueError(f"BBox extends outside image beyond rounding tolerance: {location}")
            x1, y1 = max(0.0, corners[0] * width), max(0.0, corners[1] * height)
            x2, y2 = min(float(width), corners[2] * width), min(float(height), corners[3] * height)
            if min(corners) < 0 or max(corners) > 1:
                clipped_boxes.append({"label": str(label_path), "line": line_no, "normalized_bbox": values})
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                raise ValueError(f"Zero-area bbox after boundary clipping: {location}")
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": image_id,
                "category_id": int(cls) + 1, "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
    if not gt["annotations"]:
        raise ValueError("The test split has no valid ground truth")
    stats = {"images": len(images), "instances": len(gt["annotations"]),
             "small_gt_COCO_inclusive": sum(0 <= a["area"] <= 32**2 for a in gt["annotations"]),
             "medium_gt_COCO_inclusive": sum(32**2 <= a["area"] <= 96**2 for a in gt["annotations"]),
             "boundary_area_1024": sum(a["area"] == 32**2 for a in gt["annotations"]),
             "boundary_area_9216": sum(a["area"] == 96**2 for a in gt["annotations"]),
             "class_counts": {c["name"]: sum(a["category_id"] == c["id"] for a in gt["annotations"])
                              for c in gt["categories"]},
             "clipped_rounding_boxes": clipped_boxes}
    return gt, image_ids, stats


def run_coco_evaluator(ground_truth, predictions, *, small_medium_only=True):
    """Use official COCO matching/accumulation; do not implement a replacement AP algorithm.

    The full-range option is ONLY for independent synthetic checks of extraction.
    The real model evaluation always uses small_medium_only=True.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = {im["id"] for im in ground_truth["images"]}
    category_ids = {cat["id"] for cat in ground_truth["categories"]}
    for pred in predictions:
        if pred["image_id"] not in image_ids or pred["category_id"] not in category_ids:
            raise ValueError(f"Prediction has unknown image/category ID: {pred}")
        bbox = pred["bbox"]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or min(bbox[2:]) < 0:
            raise ValueError(f"Invalid prediction bbox: {pred}")
        if not math.isfinite(pred["score"]) or not 0 <= pred["score"] <= 1:
            raise ValueError(f"Invalid prediction score: {pred}")
    gt = COCO()
    gt.dataset = deepcopy(ground_truth)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = {"info": {}, "images": deepcopy(gt.dataset["images"]),
                      "categories": deepcopy(gt.dataset["categories"]), "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    params = evaluator.params
    if not (np.allclose(params.iouThrs, np.linspace(.5, .95, 10))
            and np.allclose(params.recThrs, np.linspace(0, 1, 101))
            and params.maxDets == [1, 10, 100] and params.useCats == 1):
        raise RuntimeError("Unexpected COCO defaults; inspect pycocotools before comparing results")
    params.imgIds = sorted(image_ids)
    params.catIds = sorted(category_ids)
    if small_medium_only:
        indices = [params.areaRngLbl.index(name) for name in ("small", "medium")]
        params.areaRng = [params.areaRng[index] for index in indices]
        params.areaRngLbl = ["small", "medium"]
    evaluator.evaluate()
    evaluator.accumulate()
    # COCOeval.summarize() assumes all four area ranges. Read the official
    # precision tensor instead, using exactly its valid-entry averaging rule.
    return evaluator


def extract_six_metrics(evaluator, category_id=None):
    """precision shape: [IoU, Recall, Category, Area, maxDets]. -1 is absent GT."""
    import numpy as np

    params = evaluator.params
    precision = evaluator.eval["precision"]
    maxdet_index = params.maxDets.index(100)
    category_indices = list(range(len(params.catIds))) if category_id is None else [params.catIds.index(category_id)]
    metrics = {}
    for key, area_name, threshold in METRIC_SPECS:
        area_index = params.areaRngLbl.index(area_name)
        iou_indices = np.arange(len(params.iouThrs)) if threshold is None else np.flatnonzero(
            np.isclose(params.iouThrs, threshold, rtol=0, atol=1e-9))
        if threshold is not None and len(iou_indices) != 1:
            raise RuntimeError(f"Missing or ambiguous COCO IoU threshold: {threshold}")
        values = precision[:, :, :, area_index, maxdet_index]
        values = values[iou_indices][:, :, category_indices]
        valid = values[values > -1]
        metrics[key] = float(valid.mean()) if valid.size else None
    return metrics


def coco_protocol(evaluator, predictions_path):
    """COCOeval.evaluate() normalizes IDs into NumPy integers; export Python ints."""
    return {
        "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "recall_thresholds": evaluator.params.recThrs.tolist(),
        "maxDets": [int(v) for v in evaluator.params.maxDets],
        "area_ranges": [[float(v) for v in bounds] for bounds in evaluator.params.areaRng],
        "area_labels": list(evaluator.params.areaRngLbl),
        "category_ids": [int(v) for v in evaluator.params.catIds],
        "image_ids": [int(v) for v in evaluator.params.imgIds],
        "area_definition": "original-image bbox area after clipping negligible label rounding overflow",
        "predictions_sha256": file_hash(predictions_path),
    }


def collect_predictions(weight, yaml_path, run_dir, image_ids, gt_count, names, settings):
    """Use the local validator's resize/letterbox, inference, NMS and inverse scaling."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import LOGGER, nms

    captured = []

    class PredictionCollector(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            if ordered_names(model.names) != names:
                raise ValueError(f"Class mismatch: model={model.names}, dataset={names}")
            actual = [str(Path(p).resolve()) for p in self.dataloader.dataset.im_files]
            if len(actual) != len(image_ids) or set(actual) != set(image_ids):
                raise ValueError("Validation loader changed the test image set")
            actual_gt = sum(len(label["cls"]) for label in self.dataloader.dataset.labels)
            if actual_gt != gt_count:
                raise ValueError(f"Validation loader changed annotations: {actual_gt} vs {gt_count}")
            self.processed = Counter()

        def get_desc(self):
            return "Collecting predictions for COCO Small/Medium AP"

        def postprocess(self, preds):
            outputs = nms.non_max_suppression(
                preds, self.args.conf, self.args.iou, nc=0, multi_label=True,
                agnostic=False, max_det=self.args.max_det, end2end=self.end2end,
                rotated=False, max_time_img=math.inf,
            )
            return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
                    for x in outputs]

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                path = str(Path(pbatch["im_file"]).resolve())
                self.processed[path] += 1
                self.seen += 1
                predn = self.scale_preds(pred, pbatch)
                for b, score, cls in zip(predn["bboxes"].tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
                    self.jdict.append({"image_id": image_ids[path], "category_id": int(cls) + 1,
                        "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": score})

        def get_stats(self):
            return {}  # No YOLO AP is substituted for the requested COCO metrics.

        def finalize_metrics(self):
            self.metrics.speed = self.speed
            self.metrics.save_dir = self.save_dir

        def print_results(self):
            LOGGER.info(f"Collected {len(self.jdict)} boxes from {self.seen} test images; COCO evaluation follows.")

        def eval_json(self, stats):
            return stats

    def factory(**kwargs):
        validator = PredictionCollector(**kwargs)
        captured.append(validator)
        return validator

    model = None
    try:
        model = YOLO(str(weight), task="detect")
        if ordered_names(model.names) != names:
            raise ValueError(f"Class mismatch: model={model.names}, dataset={names}")
        model.val(validator=factory, data=str(yaml_path), project=str(run_dir),
                  name=weight.stem, exist_ok=False, **settings)
        validator = captured[-1]
        expected = Counter({path: 1 for path in image_ids})
        if validator.processed != expected:
            raise ValueError("Incomplete or duplicated test image processing")
        predictions = validator.jdict
        model_dir = Path(validator.save_dir)
        write_json(model_dir / "predictions.json", predictions)
        write_json(model_dir / "inference_audit.json", {
            "images_processed": validator.seen, "prediction_boxes": len(predictions),
            "all_test_images_processed_exactly_once": True, "speed_ms_per_image": validator.speed,
        })
        return predictions, model_dir
    finally:
        captured.clear()
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_outputs(run_dir, rows, dataset_stats):
    fields = ["model", *METRIC_KEYS, "status", "error"]
    with (run_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = ["# COCO Small/Medium AP at multiple IoU thresholds", "",
        "All numbers in this table are percentages; CSV/JSON use values in [0, 1].",
        "", "| Model | IoU=.50 AP_S | IoU=.50 AP_M | IoU=.75 AP_S | IoU=.75 AP_M | IoU=.50:.95 AP_S | IoU=.50:.95 AP_M |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    latex = [r"\begin{table*}[!t]", r"\centering",
        r"\caption{Scale-wise detection performance under multiple IoU thresholds (COCO bbox evaluation).}",
        r"\label{tab:coco_sm_multiiou}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Method & \multicolumn{2}{c}{IoU=0.50} & \multicolumn{2}{c}{IoU=0.75} & \multicolumn{2}{c}{IoU=0.50--0.95} \\",
        r" & AP$_{S}$ & AP$_{M}$ & AP$_{S}$ & AP$_{M}$ & AP$_{S}$ & AP$_{M}$ \\", r"\midrule"]
    for row in rows:
        if row["status"] != "OK":
            report += ["", f"{row['model']}: ERROR; {row['error']}"]
            continue
        values = ["N/A" if row[key] is None else f"{row[key] * 100:.2f}" for key in METRIC_KEYS]
        report.append("| " + " | ".join([row["model"], *values]) + " |")
        name = row["model"].replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        latex.append(" & ".join([name, *values]) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    report += ["", "IoU .50:.95 means the 10 thresholds .50, .55, ..., .95, not only the two endpoints.",
        "Area = ground-truth bbox width * height in original-image pixels (not letterbox/input pixels).",
        "Official COCO ranges: Small [0, 1024], Medium [1024, 9216] pixels squared; inclusive boundaries.",
        "IoU matching, ignored detections, 101 recall samples, and category averaging are from pycocotools.",
        "AP uses maxDets=100, with the default maxDets list [1, 10, 100].",
        "Categories with no GT in the evaluated area range are excluded, not scored as zero. N/A means no valid GT.",
        "ALL test images and ALL annotations are retained for COCO ignore matching, including other object sizes.",
        "The same predictions are reused for all six metrics. NMS IoU remains 0.7; it is not the evaluation IoU.",
        "No AP_L metric is computed or reported. No previous run's predictions or scores are loaded.",
        "", "Dataset statistics:", "```json", json.dumps(dataset_stats, ensure_ascii=False, indent=2), "```"]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (run_dir / "table_coco_sm.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=LOCAL_SOURCE / "configs/data.yaml")
    parser.add_argument("--device", default="0", help="CUDA device 0 by default; use cpu explicitly for CPU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weights", type=Path, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--output", type=Path, default=HERE / "coco_sm_results")
    parser.add_argument("--check-only", action="store_true", help="Audit data/dependencies without loading models or writing scores")
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.workers < 0:
        parser.error("batch/imgsz must be positive and workers must be nonnegative")
    weights = [p.expanduser().resolve() for p in args.weights]
    if len({p.stem for p in weights}) != len(weights):
        parser.error("Weight filename stems must be unique")
    if "," in args.device:
        parser.error("Use a single GPU for this evaluation")
    data_yaml = args.data.expanduser().resolve()
    for path in [LOCAL_SOURCE / "ultralytics/__init__.py", data_yaml, *weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    import yaml
    import numpy as np
    from pycocotools.cocoeval import COCOeval

    original = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    dataset = Path(original.get("path") or data_yaml.parent).expanduser()
    if not dataset.is_absolute():
        dataset = data_yaml.parent / dataset
    dataset = dataset.resolve()
    if original.get("test") != "test/images":
        raise ValueError("This evaluator expects test: test/images and test/labels under the dataset root")
    names = ordered_names(original["names"])
    gt, image_ids, dataset_stats = build_test_annotations(dataset, names)
    if args.check_only:
        print(f"Interpreter: {sys.executable}")
        print(json.dumps(dataset_stats, ensure_ascii=True, indent=2))
        print("Data/dependencies OK. No model loaded, no inference, no AP scores written.")
        return 0

    sys.path.insert(0, str(LOCAL_SOURCE))
    import torch
    import ultralytics
    from ultralytics.utils import RANK
    from ultralytics.utils.torch_utils import init_seeds

    if RANK != -1:
        raise RuntimeError("Run directly with Python; distributed execution is not supported")
    if Path(ultralytics.__file__).resolve().parent != (LOCAL_SOURCE / "ultralytics").resolve():
        raise RuntimeError(f"Unexpected Ultralytics package: {ultralytics.__file__}")
    # Same FP32 policy for both models; do not switch precision after an OOM.
    init_seeds(0, deterministic=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timestamp = datetime.now(timezone(timedelta(hours=8)))
    run_dir = args.output.expanduser().resolve() / timestamp.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "test_annotations_coco.json", gt)
    write_json(run_dir / "dataset_statistics.json", dataset_stats)
    config = {"path": dataset.as_posix(), "names": names, "nc": len(names),
              **{split: (dataset / f"{split}/images").as_posix() for split in ("train", "val", "test")}}
    yaml_path = run_dir / "test_data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    settings = dict(split="test", imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
        conf=0.001, iou=0.7, max_det=300, half=False, augment=False, rect=True, plots=True,
        save_json=True, save_txt=False, single_cls=False, agnostic_nms=False, verbose=False,
        seed=0, deterministic=True)
    weight_records = [{"path": str(path), "sha256": file_hash(path)} for path in weights]
    write_json(run_dir / "run_config.json", {
        "created_at": timestamp.isoformat(), "interpreter": sys.executable,
        "torch": torch.__version__, "numpy": np.__version__,
        "pycocotools": importlib.metadata.version("pycocotools"),
        "ultralytics": ultralytics.__version__, "ultralytics_source": ultralytics.__file__,
        "script_sha256": file_hash(__file__), "cocoeval_source_sha256": file_hash(sys.modules[COCOeval.__module__].__file__),
        "dataset": config, "ground_truth_sha256": file_hash(run_dir / "test_annotations_coco.json"),
        "weights": weight_records, "inference_settings": settings, "nms_time_limit": "disabled",
        "tf32": False, "cudnn_benchmark": False,
    })
    print(f"Interpreter: {sys.executable}\nOutput: {run_dir}", flush=True)
    rows = []
    for weight, record in zip(weights, weight_records):
        row = {"model": weight.stem, **dict.fromkeys(METRIC_KEYS), "status": "ERROR", "error": ""}
        try:
            print(f"\nEvaluating {weight.name}", flush=True)
            predictions, model_dir = collect_predictions(weight, yaml_path, run_dir, image_ids,
                                                         len(gt["annotations"]), names, settings)
            if file_hash(weight) != record["sha256"]:
                raise RuntimeError("The checkpoint changed during evaluation")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                evaluator = run_coco_evaluator(gt, predictions)
            (model_dir / "coco_evaluation_log.txt").write_text(log.getvalue(), encoding="utf-8")
            row.update(extract_six_metrics(evaluator))
            row["status"] = "OK"
            write_json(model_dir / "coco_sm_metrics.json", {key: row[key] for key in METRIC_KEYS})
            write_json(model_dir / "coco_protocol.json", coco_protocol(evaluator, model_dir / "predictions.json"))
            class_rows = [{"class": cat["name"], **extract_six_metrics(evaluator, cat["id"])} for cat in gt["categories"]]
            with (model_dir / "coco_sm_per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["class", *METRIC_KEYS])
                writer.writeheader()
                writer.writerows(class_rows)
            np.savez_compressed(model_dir / "coco_precision.npz", precision=evaluator.eval["precision"],
                iou_thresholds=evaluator.params.iouThrs, recall_thresholds=evaluator.params.recThrs,
                category_ids=evaluator.params.catIds, area_labels=evaluator.params.areaRngLbl,
                maxDets=evaluator.params.maxDets)
            for key in METRIC_KEYS:
                print(f"{key}: " + ("N/A (no valid GT)" if row[key] is None else f"{row[key] * 100:.4f}%"))
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = str(exc)
            traceback.print_exc()
            (run_dir / f"{weight.stem}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            rows.append(row)
            save_outputs(run_dir, rows, dataset_stats)
    print(f"\nReport: {run_dir / 'report.md'}", flush=True)
    return int(any(row["status"] != "OK" for row in rows))


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
