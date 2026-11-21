"""Project validation transforms, NMS and metrics with a read-only dataset.

Set sys.path and YOLO_* environment variables before importing this module.
"""
from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # also makes native NMS consistently use torchvision.ops.nms
import ultralytics
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import SETTINGS_FILE, nms
from ultralytics.utils.metrics import compute_ap

from evaluation_data import class_names, object_hash, sha256_file
from scale_metrics import restore_boxes


class ReadOnlyDataset(YOLODataset):
    """Keep native validation transforms; bypass label caches and .npy images."""

    def __init__(self, records, **kwargs):
        self.audit_records = records
        self.by_path = {r["image_path"]: r for r in records}
        super().__init__(img_path=[r["image_path"] for r in records], **kwargs)

    def get_img_files(self, img_path):
        return list(img_path)

    def get_labels(self):
        return [{"im_file": r["image_path"], "shape": (r["height"], r["width"]),
                 "cls": np.asarray([a["class_id"] for a in r["annotations"]], dtype=np.float32).reshape(-1, 1),
                 "bboxes": np.asarray([a["xywh_normalized"] for a in r["annotations"]], dtype=np.float32).reshape(-1, 4),
                 "segments": [], "keypoints": None, "normalized": True, "bbox_format": "xywh"}
                for r in self.audit_records]

    def load_image(self, i, rect_mode=True):
        path = self.im_files[i]
        # Match BaseDataset resizing while bypassing its .npy cache.
        data = np.fromfile(path, dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        record = self.by_path[path]
        if im is None or im.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Image decode/shape changed after audit: {path}")
        h0, w0 = im.shape[:2]
        if rect_mode:
            ratio = self.imgsz / max(h0, w0)
            if ratio != 1:
                w, h = min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif (h0, w0) != (self.imgsz, self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]


def environment(project_root, device_arg, precision, output):
    expected = (project_root / "ultralytics" / "__init__.py").resolve()
    actual = Path(ultralytics.__file__).resolve()
    print(f"ultralytics.__file__: {actual}\nultralytics.__version__: {ultralytics.__version__}", flush=True)
    if actual != expected:
        raise RuntimeError(f"Wrong Ultralytics import! Expected {expected}, got {actual}")
    if not Path(SETTINGS_FILE).resolve().is_relative_to(output):
        raise RuntimeError(f"Runtime settings escaped the dedicated output directory: {SETTINGS_FILE}")
    available = torch.cuda.is_available()
    print(f"PyTorch {torch.__version__}; CUDA build={torch.version.cuda}; CUDA available={available}", flush=True)
    if device_arg == "cpu":
        if precision != "fp32":
            raise ValueError("CPU evaluation requires --precision fp32")
        device = torch.device("cpu")
        print("CPU explicitly selected; this is not an automatic fallback.", flush=True)
    else:
        index = device_arg.removeprefix("cuda:")
        if not index.isdigit():
            raise ValueError("--device must be cpu, a single GPU index, or cuda:index")
        if not available:
            raise RuntimeError("No CUDA GPU is available. Request a GPU, or explicitly select --device cpu")
        if int(index) >= torch.cuda.device_count():
            raise ValueError(f"GPU index {index} is outside visible device count {torch.cuda.device_count()}")
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = {}
    for path in sorted((project_root / "ultralytics").rglob("*.py")):
        if not {"datasets", "runs", "__pycache__"}.intersection(path.relative_to(project_root).parts):
            source[path.relative_to(project_root).as_posix()] = sha256_file(path)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source[path.relative_to(project_root).as_posix()] = sha256_file(path)
    source["ultralytics/cfg/default.yaml"] = sha256_file(project_root / "ultralytics/cfg/default.yaml")
    info = {"python": sys.version, "platform": platform.platform(), "ultralytics_file": str(actual),
            "ultralytics_version": ultralytics.__version__, "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": available, "device": str(device),
            "ultralytics_settings_file": str(SETTINGS_FILE),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": precision,
            "tf32": False, "seed": 0, "cudnn_deterministic": True,
            "source_fingerprint": object_hash(source), "source_sha256": source}
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        info["gpu"] = {"name": prop.name, "total_memory_bytes": prop.total_memory,
                       "compute_capability": [prop.major, prop.minor]}
    print(f"Device: {info.get('gpu', {}).get('name', str(device))}; precision={precision}", flush=True)
    return device, info


def clean_metadata(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): clean_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_metadata(v) for v in value]
    return str(value)


def native_ap_from_stats(native):
    """Compute native AP from validator statistics without the F1 threshold search."""
    stats = {k: np.concatenate(v, 0) for k, v in native.metrics.stats.items()}
    order = np.argsort(-stats["conf"])  # exactly the native default (including ties)
    tp, cls = stats["tp"][order], stats["pred_cls"][order]
    ids, counts = np.unique(stats["target_cls"], return_counts=True)
    aps = np.zeros((len(ids), 10), dtype=float)
    for ci, (class_id, count) in enumerate(zip(ids, counts)):
        selected = tp[cls == class_id]
        if not len(selected):
            continue
        cumulative_tp = selected.cumsum(0)
        cumulative_fp = (1 - selected).cumsum(0)
        recall = cumulative_tp / (count + 1e-16)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-16)
        for ti in range(10):
            aps[ci, ti] = compute_ap(recall[:, ti], precision[:, ti])[0]
    return ids.astype(int).tolist(), aps


def load_model(weight, names, device, config, project_root):
    model, ckpt = load_checkpoint(str(weight), device=torch.device("cpu"), fuse=False)
    if getattr(model, "task", None) != "detect":
        raise ValueError("Only horizontal bounding-box detection checkpoints are supported")
    if class_names(model.names) != names:
        raise ValueError(f"Checkpoint class order/names differ from data.yaml: {model.names} versus {names}")
    if getattr(model, "end2end", False):
        raise ValueError("End-to-end/NMS-free checkpoints cannot share this NMS protocol")
    stride = int(model.stride.max())
    if config["stride"] % stride:
        raise ValueError(f"Model stride {stride} does not divide shared --stride {config['stride']}")
    modules = sorted({f"{type(m).__module__}.{type(m).__name__}" for m in model.modules()})
    for name, mod in list(sys.modules.items()):
        if name == "ultralytics" or name.startswith("ultralytics."):
            file = getattr(mod, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(project_root / "ultralytics"):
                raise RuntimeError(f"Mixed Ultralytics source detected: {name}: {file}")
    metadata = {"names": names, "task": model.task, "maximum_stride": stride, "modules": modules,
                "checkpoint": {k: clean_metadata(ckpt.get(k)) for k in
                               ("epoch", "date", "version", "train_args", "train_metrics", "best_fitness")},
                "paper_result_identity": "not independently established; compare checkpoint hash, train_args, test manifest and paper run"}
    # Fuse in memory; keep the checkpoint unchanged.
    backend = AutoBackend(model=model, device=device, fp16=config["precision"] == "fp16",
                          fuse=config["fuse"], verbose=False)
    backend.eval()
    backend.requires_grad_(False)
    if bool(backend.fp16) != (config["precision"] == "fp16") or backend.device != device:
        raise RuntimeError("Backend silently changed device or precision")
    del ckpt, model
    return backend, metadata


def infer(weight, audit, config, device, output, project_root, check_only=False):
    backend, metadata = load_model(weight, audit["names"], device, config, project_root)
    if check_only:
        del backend
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None, None, metadata
    names = dict(enumerate(audit["names"]))
    args = {"task": "detect", "mode": "val", "split": "test", "imgsz": config["imgsz"],
            "batch": config["batch"], "workers": config["workers"], "half": config["precision"] == "fp16",
            "rect": config["rect"], "conf": config["ap_conf"], "iou": config["nms_iou"],
            "max_det": config["max_det"], "agnostic_nms": config["agnostic_nms"],
            "plots": False, "save_json": False, "save_txt": False, "augment": False, "single_cls": False}
    native = DetectionValidator(save_dir=output / "native_reference", args=args)
    native.device, native.training, native.stride = device, False, config["stride"]
    native.data = {"names": names, "nc": len(names), "channels": 3, "test": audit["resolved_test"]}
    native.init_metrics(backend)
    dataset = ReadOnlyDataset(audit["records"], data=native.data, task="detect", imgsz=config["imgsz"],
                              batch_size=config["batch"], augment=False, hyp=native.args, rect=config["rect"],
                              stride=config["stride"], pad=0.5, cache=False, single_cls=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch"], shuffle=False,
                                         num_workers=config["workers"], collate_fn=dataset.collate_fn,
                                         pin_memory=device.type == "cuda", drop_last=False)
    predictions, cap_images, zero_area, shapes = {}, [], 0, set()
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = native.preprocess(batch)
            shapes.add(tuple(batch["img"].shape[1:]))
            raw = backend(batch["img"], augment=False)
            # Disable NMS timeouts so every image is processed.
            detections = nms.non_max_suppression(
                raw, conf_thres=config["ap_conf"], iou_thres=config["nms_iou"],
                multi_label=config["multi_label"], agnostic=config["agnostic_nms"],
                max_det=config["max_det"], max_nms=config["max_nms"], max_time_img=float("inf"),
                max_wh=7680, nc=0, end2end=False, rotated=False)
            native_predictions = [{"bboxes": p[:, :4], "conf": p[:, 4], "cls": p[:, 5], "extra": p[:, 6:]}
                                  for p in detections]
            native.update_metrics(native_predictions, batch)
            for i, detection in enumerate(detections):
                record = dataset.by_path[batch["im_file"][i]]
                if record["id"] in predictions:
                    raise RuntimeError("Dataloader repeated a test image")
                array = detection.detach().float().cpu().numpy()
                boxes = restore_boxes(array[:, :4], batch["ratio_pad"][i], record["width"], record["height"])
                if len(array) == config["max_det"]:
                    cap_images.append(record["id"])
                zero_area += int(np.any(boxes[:, 2:] == boxes[:, :2], axis=1).sum())
                predictions[record["id"]] = {"boxes_xyxy": boxes.tolist(), "scores": array[:, 4].tolist(),
                                             "classes": array[:, 5].astype(int).tolist()}
            print(f"{Path(weight).name}: batch {batch_index + 1}/{len(loader)}, "
                  f"images {len(predictions)}/{len(dataset)}", flush=True)
    indices, native_aps = native_ap_from_stats(native)
    native_result = {"evaluator": "project_DetectionValidator_same_predictions",
                     "status": "computed_in_this_inference_run", "metric_units": "0_to_1",
                     "map50_0_1": float(native_aps[:, 0].mean()) if indices else None,
                     "map50_95_0_1": float(native_aps.mean()) if indices else None,
                     "evaluated_class_ids": indices,
                     "per_class": [{"class_id": c, "class_name": names[c],
                                    "ap50_0_1": float(native_aps[i, 0]),
                                    "ap50_95_0_1": float(native_aps[i].mean())} for i, c in enumerate(indices)],
                     "scope": "native update_metrics + compute_ap on validation-coordinate tensors; no second inference and no F1 threshold search",
                     "difference_notes": ["native IoU-priority matching versus custom confidence-greedy matching",
                                          "native interpolated trapezoid versus custom 101-point envelope mean",
                                          "native float32 validation coordinates versus original-image float64 custom coordinates",
                                          "native unstable score ties versus stable manifest-order custom ties"],
                     "native_auto_F1_recall_omitted": True}
    metadata.update({"input_tensor_shapes_CHW": [list(s) for s in sorted(shapes)],
                     "image_inference_order": list(predictions), "images_at_max_det": cap_images,
                     "zero_area_predictions_after_clipping_retained": zero_area,
                     "elapsed_seconds_including_io_and_native_metrics": time.monotonic() - started,
                     "inference_grad_enabled": False, "model_training": backend.training})
    del backend, native, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, native_result, metadata
