"""Hand-checkable regression cases. These are synthetic, never paper results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation_data import audit_dataset, object_hash, resolve_test, write_json
from evaluate_small_objects import CACHE_SCHEMA, load_cache, main, parser
from scale_metrics import ap_101, evaluate, is_small, match_image_class, restore_boxes, validate_predictions


def record(ident, boxes=(), classes=None, width=640, height=640):
    classes = [0] * len(boxes) if classes is None else classes
    return {"id": ident, "width": width, "height": height,
            "annotations": [{"box_xyxy": b, "class_id": c} for b, c in zip(boxes, classes)]}


def pred(boxes=(), scores=None, classes=None):
    return {"boxes_xyxy": list(boxes), "scores": [0.9] * len(boxes) if scores is None else scores,
            "classes": [0] * len(boxes) if classes is None else classes}


class MatchingTests(unittest.TestCase):
    def test_rect_cli_default_and_explicit_legacy_mode(self):
        self.assertTrue(parser().parse_args([]).rect)
        self.assertTrue(parser().parse_args(["--rect"]).rect)
        self.assertFalse(parser().parse_args(["--no-rect"]).rect)

    def test_strict_size_boundaries_and_actual_image_size(self):
        boxes = [[0, 0, 31.999, 31.999], [0, 0, 32, 32], [0, 0, 10, 100], [0, 0, 10, 99.99]]
        # 1000x1000: area 1000 is exactly the 0.001 boundary and is NOT small.
        self.assertEqual(is_small(boxes, 1000, 1000).tolist(), [True, False, False, True])
        self.assertEqual(is_small([[0, 0, 40, 40]], 640, 640).tolist(), [False])
        self.assertEqual(is_small([[0, 0, 40, 40]], 1920, 1080).tolist(), [True])

    def test_gt_32_pixel_boundary_uses_original_label_side_lengths(self):
        r = record("x", [[506.99999999999, 506.99999999999, 538.99999999998, 538.99999999998]])
        r["annotations"][0]["xywh_normalized"] = [.8171875, .8171875, .05, .05]
        result = evaluate([r], {"x": pred()}, ["a"], .25)
        self.assertEqual(result["small"]["ground_truth_objects"], 0)
        self.assertEqual(result["non-small"]["ground_truth_objects"], 1)

    def test_mixed_sizes_duplicate_detection(self):
        small, big = [10, 10, 30, 30], [100, 100, 200, 200]
        result = match_image_class([small, big], [small, big, small], [.9, .8, .7], 640, 640, "small")
        self.assertEqual(result["tp"][:, 0].tolist(), [True, False, False])
        self.assertEqual(result["ignored"][:, 0].tolist(), [False, True, False])
        self.assertEqual(result["fp"][:, 0].tolist(), [False, False, True])
        self.assertEqual(result["gt"], 1)

    def test_matched_prediction_uses_gt_group_in_both_directions(self):
        small, big = [0, 0, 31, 31], [0, 0, 33, 33]
        m = match_image_class([small], [big], [.9], 640, 640, "small", [.5, .95])
        self.assertEqual(m["tp"].tolist(), [[True, False]])
        self.assertEqual(m["ignored"].tolist(), [[False, True]])
        m = match_image_class([big], [small], [.9], 640, 640, "small", [.5, .95])
        self.assertEqual(m["ignored"].tolist(), [[True, False]])
        self.assertEqual(m["fp"].tolist(), [[False, True]])
        m = match_image_class([big], [small], [.9], 640, 640, "non-small", [.5])
        self.assertTrue(m["tp"][0, 0])

    def test_valid_gt_has_priority_over_higher_iou_ignored_gt(self):
        small, big = [0, 0, 31, 31], [0, 0, 33, 33]
        m = match_image_class([big, small], [big], [.9], 640, 640, "small", [.5])
        self.assertTrue(m["tp"][0, 0])  # IoU 961/1089 beats ignored IoU=1 by priority
        self.assertFalse(m["ignored"][0, 0])

    def test_ignored_gt_is_one_to_one_not_crowd(self):
        m = match_image_class([[0, 0, 33, 33]], [[0, 0, 31, 31]] * 2, [.9, .8], 640, 640, "small", [.5])
        self.assertEqual(m["ignored"][:, 0].tolist(), [True, False])
        self.assertEqual(m["fp"][:, 0].tolist(), [False, True])

    def test_confidence_order_wins_over_iou_order(self):
        m = match_image_class([[0, 0, 20, 20]], [[1, 0, 21, 20], [0, 0, 20, 20]], [.9, .8], 640, 640, "all", [.5, .95])
        self.assertEqual(m["tp"].tolist(), [[True, False], [False, True]])

    def test_wrong_class_is_fp_and_gt_is_fn(self):
        r = [record("x", [[0, 0, 20, 20]], [0])]
        result = evaluate(r, {"x": pred([[0, 0, 20, 20]], classes=[1])}, ["a", "b"], .25)["small"]
        self.assertEqual((result["tp"], result["fn"], result["fp"]), (0, 1, 1))
        self.assertEqual(result["map50_0_1"], 0)
        self.assertIsNone(result["per_class"][1]["ap50_0_1"])
        self.assertEqual(result["evaluated_class_ids"], [0])

    def test_no_small_gt_image_false_positive_changes_ap(self):
        s, b = [0, 0, 20, 20], [100, 100, 200, 200]
        r = [record("no_small", [b]), record("small", [s])]
        p = {"no_small": pred([s], [.95]), "small": pred([s], [.9])}
        result = evaluate(r, p, ["a"], .25)["small"]
        self.assertAlmostEqual(result["map50_0_1"], .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_empty_image_and_unmatched_outside_group(self):
        s, b = [0, 0, 20, 20], [100, 100, 200, 200]
        m = match_image_class([], [s, b], [.9, .8], 640, 640, "small", [.5])
        self.assertEqual(m["fp"][:, 0].tolist(), [True, False])
        self.assertEqual(m["ignored"][:, 0].tolist(), [False, True])

    def test_macro_map_and_micro_recall_and_na(self):
        boxes = [[0, 0, 20, 20], [30, 0, 50, 20], [60, 0, 80, 20], [90, 0, 110, 20]]
        r = [record("x", boxes, [0, 1, 1, 1])]
        p = {"x": pred([boxes[0]], [.9], [0])}
        result = evaluate(r, p, ["a", "b", "absent"], .25)["all"]
        self.assertAlmostEqual(result["map50_0_1"], .5)  # macro: (1 + 0)/2
        self.assertEqual(result["recall_at_0_5_0_1"], .25)  # micro: 1/4
        self.assertEqual(result["evaluated_classes"], 2)
        self.assertIsNone(result["per_class"][2]["ap50_95_0_1"])

    def test_recall_threshold_independent_of_ap(self):
        r = [record("x", [[0, 0, 20, 20]])]
        p = {"x": pred([[0, 0, 20, 20]], [.25])}
        lo, hi = (evaluate(r, p, ["a"], c)["small"] for c in [.25, .3])
        self.assertEqual(lo["map50_0_1"], hi["map50_0_1"])
        self.assertEqual((lo["tp"], hi["tp"], hi["fn"]), (1, 0, 1))

    def test_101_point_sampling_has_known_values(self):
        self.assertAlmostEqual(ap_101([1], [0], 1), 1)
        self.assertAlmostEqual(ap_101([1], [0], 2), 51 / 101)
        self.assertAlmostEqual(ap_101([0, 1], [1, 0], 1), .5)
        self.assertEqual(ap_101([], [], 1), 0)
        self.assertIsNone(ap_101([], [], 0))

    def test_stable_equal_confidence_ties(self):
        s = [0, 0, 20, 20]
        r = [record("first"), record("second", [s])]
        p = {"first": pred([s], [.9]), "second": pred([s], [.9])}
        self.assertEqual(evaluate(r, p, ["a"], .25)["small"]["map50_0_1"], .5)

    def test_empty_group_is_na_and_no_predictions_is_zero_for_present_class(self):
        r = [record("x", [[0, 0, 100, 100]])]
        result = evaluate(r, {"x": pred()}, ["a"], .25)
        self.assertIsNone(result["small"]["map50_0_1"])
        self.assertEqual(result["non-small"]["map50_0_1"], 0)
        self.assertEqual(result["non-small"]["fn"], 1)

    def test_coordinate_restore_handles_distinct_gains_and_clipping(self):
        # 1001x501 resized to 640x321, then top-padded 159 pixels.
        original = np.asarray([[10, 20, 41, 51]], dtype=float)
        scaled = original * [640 / 1001, 321 / 501, 640 / 1001, 321 / 501] + [0, 159, 0, 159]
        restored = restore_boxes(scaled, ((321 / 501, 640 / 1001), (0, 159)), 1001, 501)
        np.testing.assert_allclose(restored, original, atol=1e-10)
        clipped = restore_boxes([[-2, -2, 100, 100]], ((1, 1), (0, 0)), 80, 90)
        self.assertEqual(clipped.tolist(), [[0, 0, 80, 90]])


class DataAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_parent = Path(__file__).resolve().parent / "local_checks/unit_tmp"
        self.temp_parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_parent)
        self.root = Path(self.temp.name)
        (self.root / "test/images").mkdir(parents=True)
        (self.root / "test/labels").mkdir()
        Image.new("RGB", (100, 80)).save(self.root / "test/images/a.rf.123.png")
        self.label = self.root / "test/labels/a.rf.123.txt"
        self.label.write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
        self.yaml = self.root / "data.yaml"
        self.yaml.write_text("test: test/images\nnames: [trash]\n", encoding="utf-8")

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.temp_parent.resolve())
        self.temp.cleanup()

    def test_no_test_split_or_missing_path_never_falls_back(self):
        self.yaml.write_text("train: test/images\nval: test/images\nnames: [trash]\n")
        with self.assertRaises(ValueError):
            resolve_test(self.yaml)
        self.yaml.write_text("test: missing/images\nval: test/images\nnames: [trash]\n")
        with self.assertRaises(FileNotFoundError):
            resolve_test(self.yaml)

    def test_invalid_rows_reported_and_skip_keeps_all_images(self):
        self.label.write_text("0 .5 .5 .2 .25\n0 .5 .5 .2 .25\n0 nan .5 .2 .2\n1 .5 .5 .2 .2\n0 .5 .5 0 .2\n")
        audit = audit_dataset(self.yaml)
        self.assertEqual(audit["fatal_issue_count"], 4)
        self.assertEqual(audit["counts"]["invalid_label_rows"], 4)
        audit = audit_dataset(self.yaml, invalid_labels="skip")
        self.assertEqual(audit["fatal_issue_count"], 0)
        self.assertEqual(audit["image_count"], 1)
        self.assertEqual(audit["ground_truth_objects"], 1)

    def test_missing_and_empty_labels_are_distinct(self):
        self.label.unlink()
        self.assertEqual(audit_dataset(self.yaml)["fatal_issue_count"], 1)
        self.assertEqual(audit_dataset(self.yaml, missing_labels="empty")["fatal_issue_count"], 0)
        self.label.write_text("")
        self.assertEqual(audit_dataset(self.yaml)["counts"]["empty_label_files"], 1)

    def test_boundary_roundoff_is_retained_and_reported(self):
        self.label.write_text("0 .979688 .5 .040625 .1\n")
        before = self.label.read_bytes()
        audit = audit_dataset(self.yaml)
        self.assertEqual(len(audit["boundary_rounding_rows"]), 1)
        self.assertGreater(audit["records"][0]["annotations"][0]["box_xyxy"][2], 100)
        self.assertEqual(audit_dataset(self.yaml, boundary_tolerance=0)["fatal_issue_count"], 1)
        self.assertEqual(self.label.read_bytes(), before)

    def test_full_cache_manifest_and_hash_are_required(self):
        audit = audit_dataset(self.yaml)
        predictions = {audit["records"][0]["id"]: pred()}
        cache = {"schema": CACHE_SCHEMA, "dataset_fingerprint": audit["dataset_fingerprint"], "names": ["trash"],
                 "inference_config": {"ap_conf": .001, "max_det": 300}, "predictions": predictions,
                 "predictions_sha256": object_hash(predictions)}
        path = self.root / "cache.json"
        write_json(path, cache)
        self.assertEqual(load_cache(path, audit, .25)["predictions"], predictions)
        with self.assertRaises(ValueError):
            load_cache(path, audit, .001)
        self.label.write_text("0 .6 .5 .2 .2\n")
        with self.assertRaises(ValueError):
            load_cache(path, audit_dataset(self.yaml), .25)
        cache["predictions"] = {}
        write_json(path, cache)
        with self.assertRaises(ValueError):
            load_cache(path, audit, .25)

    def test_bad_prediction_records_fail(self):
        r = [record("x")]
        for p in ({}, {"x": pred([[0, 0, 20, 20]], [float("nan")])},
                  {"x": pred([[0, 0, 20, 20]], [.9], [2])},
                  {"x": pred([[0, 0, 1000, 20]], [.9])}):
            with self.assertRaises(ValueError):
                validate_predictions(r, p, ["a"], 300, .001)

    def test_cache_recall_uses_generating_ap_threshold_not_cli_default(self):
        # Reach cache file resolution at Recall=.0005; the default CLI AP=.001
        # must not reject it, because a cache may have been collected at .0001.
        with self.assertRaises(FileNotFoundError):
            main(["--data", str(self.yaml), "--cache-in", str(self.root / "missing_cache.json"),
                  "--recall-conf", ".0005", "--output", str(self.root / "out")])

    def test_zero_area_prediction_is_retained(self):
        r = [record("x", [[0, 0, 20, 20]])]
        p = {"x": pred([[0, 0, 0, 20]], [.9])}
        validate_predictions(r, p, ["a"], 300, .001)
        result = evaluate(r, p, ["a"], .25)["small"]
        self.assertEqual((result["fp"], result["fn"]), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
