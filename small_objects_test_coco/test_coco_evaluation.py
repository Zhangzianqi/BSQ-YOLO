"""Synthetic checks of official COCO size filtering and metric extraction."""
import contextlib
import io
import unittest

import numpy as np

from test_coco_sm_multiiou import extract_six_metrics, run_coco_evaluator


def ground_truth():
    return {
        "info": {}, "images": [{"id": 1, "width": 640, "height": 640}],
        "categories": [{"id": 1, "name": "litter"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 33, 33], "area": 1089, "iscrowd": 0},
        ],
    }


def evaluate(predictions, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_coco_evaluator(ground_truth(), predictions, **kwargs)


class CocoEvaluationTests(unittest.TestCase):
    def test_match_to_medium_gt_is_ignored_for_small_ap(self):
        prediction = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 31, 31], "score": .9}
        evaluator = evaluate([prediction])
        small = next(e for e in evaluator.evalImgs if e is not None and e["aRng"] == [0, 1024])
        np.testing.assert_array_equal(small["dtMatches"][:1], [[2]])
        self.assertTrue(small["dtIgnore"][0, 0])
        # At IoU=.95 the same small detection is unmatched and counts as FP.
        self.assertEqual(small["dtMatches"][-1, 0], 0)
        self.assertFalse(small["dtIgnore"][-1, 0])

    def test_unmatched_detection_uses_its_own_area(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [300, 300, 20, 20], "score": .9},
            {"image_id": 1, "category_id": 1, "bbox": [400, 400, 100, 100], "score": .8},
        ]
        evaluator = evaluate(predictions)
        small = next(e for e in evaluator.evalImgs if e is not None and e["aRng"] == [0, 1024])
        np.testing.assert_array_equal(small["dtIgnore"][0], [False, True])

    def test_six_metrics_equal_full_range_official_evaluation(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": .9}
            for a in ground_truth()["annotations"]
        ]
        restricted = extract_six_metrics(evaluate(predictions))
        full = extract_six_metrics(evaluate(predictions, small_medium_only=False))
        self.assertEqual(restricted, full)
        for value in restricted.values():
            self.assertAlmostEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
