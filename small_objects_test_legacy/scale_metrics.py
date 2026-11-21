"""Paper-specific size-group AP with confidence-ordered, one-to-one matching.

Boxes use original-image float64 xyxy coordinates. This is not COCO AP_S.
"""
from __future__ import annotations

import numpy as np

GROUPS = ("all", "small", "non-small")
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
DEFINITION = "((width < 32) AND (height < 32)) OR (width * height / (image_width * image_height) < 0.001)"
EVALUATOR = "paper_size_coco_style_101point_v1"


def is_small(boxes, width, height):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    wh = boxes[:, 2:] - boxes[:, :2]
    return small_dimensions(wh[:, 0], wh[:, 1], width, height)


def small_dimensions(box_width, box_height, image_width, image_height):
    w, h = np.asarray(box_width), np.asarray(box_height)
    return ((w < 32) & (h < 32)) | (w * h / (image_width * image_height) < 0.001)


def annotation_small(annotation, width, height):
    if "xywh_normalized" in annotation:
        # Use label side lengths to avoid rounding errors at exactly 32 pixels.
        w, h = annotation["xywh_normalized"][2:]
        return bool(small_dimensions(w * width, h * height, width, height))
    return bool(is_small([annotation["box_xyxy"]], width, height)[0])


def membership(boxes, width, height, group):
    if group == "all":
        return np.ones(len(boxes), dtype=bool)
    small = is_small(boxes, width, height)
    if group == "small":
        return small
    if group == "non-small":
        return ~small
    raise ValueError(f"Unknown group: {group}")


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(2)
    union = (np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)[:, None] +
             np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)[None, :] - inter)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def restore_boxes(boxes, ratio_pad, width, height):
    """Undo resize/padding using separate x/y gains, then clip predictions.

    ratio_pad = ((resized_h/original_h, resized_w/original_w), (left, top)).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    (gain_y, gain_x), (left, top) = ratio_pad
    if gain_x <= 0 or gain_y <= 0:
        raise ValueError("Invalid resize gain")
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / gain_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / gain_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def match_image_class(gt_boxes, pred_boxes, scores, width, height, group, thresholds=IOU_THRESHOLDS, gt_small=None):
    """Return score-sorted TP/FP/ignore matrices [detections, IoUs] and valid GT count.

    Classes and images must already be separated by the caller. All GT is kept.
    Equal scores preserve input order; equal IoUs choose the later GT within the
    valid/ignored partition, as in COCOeval. Ignored GT can be matched only once.
    """
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    pred, scores = pred[order], scores[order]
    gt_valid = membership(gt, width, height, group)
    if gt_small is not None and group != "all":
        gt_valid = np.asarray(gt_small, dtype=bool)
        if gt_valid.shape != (len(gt),):
            raise ValueError("GT size flags do not match GT boxes")
        if group == "non-small":
            gt_valid = ~gt_valid
    gt_order = np.argsort(~gt_valid, kind="stable")
    gt, gt_valid = gt[gt_order], gt_valid[gt_order]
    pred_valid = membership(pred, width, height, group)
    ious = box_iou(pred, gt)
    tp = np.zeros((len(pred), len(thresholds)), dtype=bool)
    ignored = np.zeros_like(tp)
    for ti, threshold in enumerate(thresholds):
        used = np.zeros(len(gt), dtype=bool)
        for di in range(len(pred)):
            best, best_iou = -1, float(threshold)
            for gi in range(len(gt)):
                if used[gi]:
                    continue
                if best >= 0 and gt_valid[best] and not gt_valid[gi]:
                    break  # a valid match has priority, even at lower IoU
                if ious[di, gi] >= best_iou:
                    best, best_iou = gi, ious[di, gi]
            if best >= 0:
                used[best] = True
                tp[di, ti] = gt_valid[best]
                ignored[di, ti] = not gt_valid[best]
            else:
                ignored[di, ti] = not pred_valid[di]
    return {"scores": scores, "tp": tp, "fp": ~tp & ~ignored,
            "ignored": ignored, "gt": int(gt_valid.sum())}


def ap_101(tp, fp, gt_count):
    """Mean precision envelope at recall 0:0.01:1 (COCO sampling, no trapezoid)."""
    if not gt_count:
        return None
    tp, fp = np.cumsum(tp, dtype=np.float64), np.cumsum(fp, dtype=np.float64)
    if not len(tp):
        return 0.0
    recall = tp / gt_count
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indexes = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    sampled = np.zeros(101, dtype=np.float64)
    valid = indexes < len(envelope)
    sampled[valid] = envelope[indexes[valid]]
    return float(sampled.mean())


def validate_predictions(records, predictions, names, max_det, ap_conf):
    expected = {r["id"] for r in records}
    if set(predictions) != expected:
        raise ValueError("Prediction cache must contain every test image exactly once, including empty images")
    for record in records:
        p = predictions[record["id"]]
        boxes = np.asarray(p["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(p["scores"], dtype=np.float64)
        classes = np.asarray(p["classes"], dtype=np.float64)
        if scores.shape != (len(boxes),) or classes.shape != (len(boxes),) or len(boxes) > max_det:
            raise ValueError(f"Invalid prediction lengths/max_det: {record['id']}")
        if not all(np.isfinite(a).all() for a in (boxes, scores, classes)):
            raise ValueError(f"Nonfinite predictions: {record['id']}")
        if np.any((classes != np.floor(classes)) | (classes < 0) | (classes >= len(names))):
            raise ValueError(f"Invalid predicted class: {record['id']}")
        if np.any((scores <= ap_conf) | (scores > 1)):
            raise ValueError(f"Scores do not match the recorded AP collection threshold: {record['id']}")
        # Zero-area boxes after clipping remain as detections (IoU = 0), not silently dropped.
        if np.any(boxes[:, 2:] < boxes[:, :2]) or np.any(boxes < 0):
            raise ValueError(f"Invalid prediction boxes: {record['id']}")
        if np.any(boxes[:, [0, 2]] > record["width"]) or np.any(boxes[:, [1, 3]] > record["height"]):
            raise ValueError(f"Predictions are not clipped original-image coordinates: {record['id']}")


def _evaluate_class(records, predictions, class_id, name, group, recall_conf):
    matches = []
    for record in records:  # Retain images with no GT in this group/class.
        class_gt = [a for a in record["annotations"] if a["class_id"] == class_id]
        prediction = predictions[record["id"]]
        class_mask = np.asarray(prediction["classes"], dtype=int) == class_id
        matches.append(match_image_class(
            [a["box_xyxy"] for a in class_gt],
            np.asarray(prediction["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)[class_mask],
            np.asarray(prediction["scores"], dtype=np.float64)[class_mask],
            record["width"], record["height"], group,
            gt_small=[annotation_small(a, record["width"], record["height"]) for a in class_gt]))
    gt = sum(match["gt"] for match in matches)
    scores = np.concatenate([match["scores"] for match in matches])
    order = np.argsort(-scores, kind="stable")
    tp, fp, ignored = (np.concatenate([match[key] for match in matches])[order]
                       for key in ("tp", "fp", "ignored"))
    # Each IoU threshold has its own ignore mask.
    aps = [ap_101(tp[~ignored[:, t], t], fp[~ignored[:, t], t], gt) for t in range(10)]
    selected = scores[order] >= recall_conf
    true_positives = int(tp[selected, 0].sum())
    false_positives = int(fp[selected, 0].sum())
    return {"class_id": class_id, "class_name": name, "ground_truth_objects": gt,
            "included_in_map": bool(gt), "ap50_0_1": aps[0],
            "ap50_95_0_1": float(np.mean(aps)) if gt else None,
            "ap_by_iou_0_1": aps, "recall_at_0_5_0_1": true_positives / gt if gt else None,
            "tp": true_positives, "fn": gt - true_positives, "fp": false_positives,
            "ignored_at_recall_conf": int(ignored[selected, 0].sum())}


def evaluate(records, predictions, names, recall_conf):
    result = {}
    for group in GROUPS:
        rows = [_evaluate_class(records, predictions, class_id, name, group, recall_conf)
                for class_id, name in enumerate(names)]
        valid = [r for r in rows if r["included_in_map"]]
        total = sum(r["ground_truth_objects"] for r in rows)
        tp = sum(r["tp"] for r in rows)
        result[group] = {"ground_truth_objects": total, "evaluated_classes": len(valid),
                         "evaluated_class_ids": [r["class_id"] for r in valid],
                         "evaluated_class_names": [r["class_name"] for r in valid],
                         "map50_0_1": float(np.mean([r["ap50_0_1"] for r in valid])) if valid else None,
                         "map50_95_0_1": float(np.mean([r["ap50_95_0_1"] for r in valid])) if valid else None,
                         "recall_at_0_5_0_1": tp / total if total else None, "tp": tp, "fn": total - tp,
                         "fp": sum(r["fp"] for r in rows), "per_class": rows}
    return result
