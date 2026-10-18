"""
tests/test_eval.py
==================
Tests for the evaluation harness (eval/run_eval.py).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.run_eval import (
    edit_distance,
    character_error_rate,
    compute_metrics,
    compute_split_metrics,
    LabelRow,
    PredictionRow,
)


class TestEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_distance("ABC", "ABC"), 0)

    def test_empty(self):
        self.assertEqual(edit_distance("", ""), 0)
        self.assertEqual(edit_distance("ABC", ""), 3)
        self.assertEqual(edit_distance("", "ABC"), 3)

    def test_substitution(self):
        self.assertEqual(edit_distance("ABC", "AXC"), 1)

    def test_insertion(self):
        self.assertEqual(edit_distance("AC", "ABC"), 1)

    def test_deletion(self):
        self.assertEqual(edit_distance("ABC", "AC"), 1)

    def test_mixed(self):
        self.assertEqual(edit_distance("TN09AB1234", "TN09AB1235"), 1)

    def test_completely_different(self):
        self.assertEqual(edit_distance("AAAA", "BBBB"), 4)


class TestCER(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(character_error_rate("TN09AB1234", "TN09AB1234"), 0.0)

    def test_one_error(self):
        # 1 substitution out of 10 characters
        self.assertAlmostEqual(character_error_rate("TN09AB1235", "TN09AB1234"), 0.1)

    def test_empty_true(self):
        self.assertAlmostEqual(character_error_rate("ABC", ""), 1.0)
        self.assertAlmostEqual(character_error_rate("", ""), 0.0)


class TestComputeMetrics(unittest.TestCase):
    def test_empty(self):
        m = compute_metrics([], {})
        self.assertEqual(m.n_labels, 0)
        self.assertEqual(m.vehicle_recall, 0.0)

    def test_perfect_predictions(self):
        labels = [
            LabelRow("t1", "TN09AB1234", "car", "phone", "day"),
            LabelRow("t2", "KA51CD5678", "truck", "phone", "day"),
        ]
        preds = {
            "t1": PredictionRow("t1", "TN09AB1234", 0.95, True, "car",
                                vehicle_detected=True, plate_detected=True),
            "t2": PredictionRow("t2", "KA51CD5678", 0.90, True, "truck",
                                vehicle_detected=True, plate_detected=True),
        }
        m = compute_metrics(labels, preds, emit_threshold=0.5)
        self.assertAlmostEqual(m.vehicle_recall, 1.0)
        self.assertAlmostEqual(m.plate_recall_per_vehicle, 1.0)
        self.assertAlmostEqual(m.plate_exact_match, 1.0)
        self.assertAlmostEqual(m.character_error_rate, 0.0)
        self.assertAlmostEqual(m.type_accuracy, 1.0)
        self.assertAlmostEqual(m.precision_at_threshold, 1.0)

    def test_partial_detection(self):
        labels = [
            LabelRow("t1", "TN09AB1234", "car", "phone", "day"),
            LabelRow("t2", "KA51CD5678", "truck", "phone", "day"),
        ]
        preds = {
            "t1": PredictionRow("t1", "TN09AB1234", 0.95, True, "car",
                                vehicle_detected=True, plate_detected=True),
            # t2 not detected at all
        }
        m = compute_metrics(labels, preds, emit_threshold=0.5)
        self.assertAlmostEqual(m.vehicle_recall, 0.5)
        self.assertEqual(m.n_vehicles_detected, 1)

    def test_wrong_plate(self):
        labels = [
            LabelRow("t1", "TN09AB1234", "car", "phone", "day"),
        ]
        preds = {
            "t1": PredictionRow("t1", "TN09AB1235", 0.95, True, "car",
                                vehicle_detected=True, plate_detected=True),
        }
        m = compute_metrics(labels, preds, emit_threshold=0.5)
        self.assertAlmostEqual(m.plate_exact_match, 0.0)
        self.assertGreater(m.character_error_rate, 0.0)


class TestSplitMetrics(unittest.TestCase):
    def test_split_by_source(self):
        labels = [
            LabelRow("t1", "TN09AB1234", "car", "phone", "day"),
            LabelRow("t2", "KA51CD5678", "truck", "cctv", "night"),
        ]
        preds = {
            "t1": PredictionRow("t1", "TN09AB1234", 0.95, True, "car",
                                vehicle_detected=True, plate_detected=True),
            "t2": PredictionRow("t2", "KA51CD5678", 0.90, True, "truck",
                                vehicle_detected=True, plate_detected=True),
        }
        results = compute_split_metrics(labels, preds)
        self.assertIn("all", results)
        self.assertIn("source:phone", results)
        self.assertIn("source:cctv", results)
        self.assertIn("condition:day", results)
        self.assertIn("condition:night", results)
        self.assertEqual(results["source:phone"].n_labels, 1)
        self.assertEqual(results["source:cctv"].n_labels, 1)


if __name__ == "__main__":
    unittest.main()
