"""Read-only test-split/YOLO-label audit and reproducibility manifests."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickletools
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from scale_metrics import annotation_small

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def existing_path(path):
    # Separate existence checking also works in restricted Windows environments
    # where strict realpath needs permissions beyond ordinary file reading.
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required path is unavailable: {path}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def class_names(value):
    if isinstance(value, dict):
        converted = {int(k): str(v) for k, v in value.items()}
        if sorted(converted) != list(range(len(converted))):
            raise ValueError("Class IDs in names must be consecutive integers from zero")
        value = [converted[i] for i in range(len(converted))]
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError("data.yaml must specify nonempty names, not just nc")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate class names are ambiguous")
    return value


def resolve_test(data_yaml):
    data_yaml = existing_path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or not cfg.get("test"):
        raise ValueError("data.yaml has no usable test split; train/val fallback is forbidden")
    names = class_names(cfg.get("names"))
    if "nc" in cfg and int(cfg["nc"]) != len(names):
        raise ValueError("nc and names disagree")
    if cfg.get("channels", 3) != 3:
        raise ValueError("Only three-channel horizontal-box detection is supported")
    root = Path(cfg.get("path") or data_yaml.parent).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = existing_path(root)
    specs = cfg["test"] if isinstance(cfg["test"], list) else [cfg["test"]]
    files, resolved = [], []
    for spec in specs:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("test entries must be nonempty local paths")
        path = Path(spec).expanduser()
        path = existing_path(path if path.is_absolute() else root / path)
        resolved.append(str(path))
        if path.is_dir():
            images = sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        elif path.suffix.lower() == ".txt":
            images = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip()).expanduser()
                item = existing_path(item if item.is_absolute() else path.parent / item)
                if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(f"Unsupported test-list image: {item}")
                images.append(item)
        else:
            raise ValueError(f"test must point to an image directory or image-list .txt: {path}")
        if not images:
            raise ValueError(f"test path contains no supported images: {path}")
        files.extend(images)
    if len(set(files)) != len(files):
        raise ValueError("Repeated test image paths; refusing to silently deduplicate or double count")
    # Canonical manifest order also defines equal-confidence tie order across images.
    return data_yaml, cfg, root, names, sorted(files), resolved


def label_path(image):
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Cannot map labels: image path has no 'images' component: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")  # preserves dots in Roboflow basenames


def audit_dataset(data_yaml, invalid_labels="error", missing_labels="error", boundary_tolerance=1e-6):
    data_yaml, cfg, root, names, files, resolved = resolve_test(data_yaml)
    records, issues, rounding = [], [], []
    counts = Counter()
    for index, path in enumerate(files):
        label = label_path(path)
        try:
            with Image.open(path) as im:
                width, height = im.size
                orientation = int(im.getexif().get(274, 1))
                im.load()  # fully decode, without native JPEG repair/write behavior
            if orientation != 1:
                raise ValueError(f"EXIF orientation={orientation}; stored-pixel/label orientation needs confirmation")
        except Exception as exc:
            issues.append({"image": str(path), "reason": "invalid_image", "detail": str(exc), "fatal": True})
            continue  # the entire run will fail, never evaluate a reduced image set
        try:
            ident = path.relative_to(root).as_posix()
        except ValueError:
            ident = os.path.relpath(path, root).replace(os.sep, "/")
        annotations, seen = [], set()
        if not label.is_file():
            counts["missing_label_files"] += 1
            issues.append({"image": str(path), "label": str(label), "reason": "missing_label",
                           "fatal": missing_labels == "error"})
            lines = []
        else:
            lines = label.read_text(encoding="utf-8-sig").splitlines()
            if not any(line.strip() for line in lines):
                counts["empty_label_files"] += 1
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            counts["nonempty_label_rows"] += 1
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("expected exactly 5 YOLO detection fields (no segmentation/OBB)")
                values = tuple(map(float, fields))
                c, x, y, w, h = values
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("nonfinite value")
                if not c.is_integer() or not 0 <= c < len(names):
                    raise ValueError("class ID outside data.yaml names")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError("invalid normalized center/size")
                bounds = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                excess = max(0.0, -bounds[0], -bounds[1], bounds[2] - 1, bounds[3] - 1)
                if excess > boundary_tolerance:
                    raise ValueError(f"box exceeds image by {excess:.12g} normalized units")
                if values in seen:
                    raise ValueError("duplicate class/box row; later duplicate is invalid")
                seen.add(values)
                if excess:
                    rounding.append({"image_id": ident, "label": str(label), "line": line_number,
                                     "normalized_excess": excess, "raw": line, "action": "retained_unclipped"})
                annotations.append({"class_id": int(c), "xywh_normalized": [x, y, w, h],
                                    "box_xyxy": [bounds[0] * width, bounds[1] * height,
                                                 bounds[2] * width, bounds[3] * height], "line": line_number})
            except ValueError as exc:
                counts["invalid_label_rows"] += 1
                issues.append({"image_id": ident, "label": str(label), "line": line_number, "raw": line,
                               "reason": "invalid_label", "detail": str(exc), "fatal": invalid_labels == "error"})
        records.append({"id": ident, "image_path": str(path), "label_path": str(label),
                        "width": width, "height": height, "image_sha256": sha256_file(path),
                        "label_sha256": sha256_file(label) if label.is_file() else None,
                        "annotations": annotations})
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(files)} test images", flush=True)
    per_class = []
    for c, name in enumerate(names):
        total, small = 0, 0
        for record in records:
            objects = [a for a in record["annotations"] if a["class_id"] == c]
            total += len(objects)
            small += sum(annotation_small(a, record["width"], record["height"]) for a in objects)
        per_class.append({"class_id": c, "class_name": name, "all": total, "small": small, "non-small": total - small})
    policy = {"invalid_labels": invalid_labels, "missing_labels": missing_labels,
              "boundary_tolerance_normalized": boundary_tolerance,
              "boundary_policy": "retain original values within tolerance; no clipping/rewriting of GT",
              "invalid_skip_policy": "skip invalid rows only; keep every image; later duplicate row is invalid"}
    # Absolute paths excluded so identical uploaded content can reuse a cache on Linux.
    fingerprint = object_hash({"names": names, "policy": policy, "records": [
        {k: r[k] for k in ("id", "width", "height", "image_sha256", "label_sha256", "annotations")}
        for r in records]})
    return {"data_yaml": str(data_yaml), "data_yaml_sha256": sha256_file(data_yaml),
            "data_yaml_contents": cfg, "dataset_root": str(root), "resolved_test": resolved,
            "path_rule": "path relative to YAML directory; test relative to dataset root; list entries relative to list directory",
            "split": "test", "names": names, "expected_image_count": len(files), "image_count": len(records),
            "ground_truth_objects": sum(x["all"] for x in per_class), "per_class_counts": per_class,
            "counts": dict(counts), "annotation_policy": policy, "issues": issues,
            "boundary_rounding_rows": rounding, "fatal_issue_count": sum(i["fatal"] for i in issues),
            "dataset_fingerprint": fingerprint, "records": records}


def inventory(paths):
    result = []
    for path in paths:
        path = existing_path(path)
        info = {"path": str(path), "name": path.stem, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        try:
            with zipfile.ZipFile(path) as archive:
                entry = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
                info["serialized_globals"] = sorted({arg for op, arg, _ in pickletools.genops(archive.read(entry))
                                                     if op.name == "GLOBAL"})
        except (zipfile.BadZipFile, StopIteration, ValueError) as exc:
            info["static_metadata_note"] = str(exc)
        result.append(info)
    return result
