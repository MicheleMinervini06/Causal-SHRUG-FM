"""Small synthetic gate fits only: missing support, leakage guards and exact ties."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from scripts import segmentation_gate as gate


def fixture(root):
    provenance = {"cache_fingerprint": "cache", "encoder_sha256": "encoder", "reference_support": "joint",
                  "manifest_sha256": "manifest", "inspection_sha256": "inspection"}
    densities = []
    for split, count in (("validation", 30), ("test", 10), ("bolivia_holdout", 4)):
        directory = root / split
        directory.mkdir()
        rows = []
        for i in range(count):
            row = {"patch_id": f"{split}_{i}", "event_id": f"event_{i // 5}" if split != "bolivia_holdout" else "Bolivia",
                   "original_split": split, "reference_fraction": .1, "predicted_fraction": .11,
                   "F1": .3 if i % 2 else .8, "n_valid_pixels": 100,
                   "average_entropy": .05 if i % 2 else .4, "mutual_information": .03 if i % 2 else .01,
                   "roi_empty": False, "threshold": .5, "roi_threshold": .05, "model_seeds": "0;1;2"}
            if i == 0:
                row.update(reference_fraction=0., predicted_fraction=0., F1=None,
                           roi_empty=True, average_entropy=None, mutual_information=None)
            if i == 1:
                row.update(F1=0., predicted_fraction=0., roi_empty=True, average_entropy=None, mutual_information=None)
            rows.append(row)
            density = (i % 7 + 1) * 1e-9
            densities.append({**{k: row[k] for k in ("patch_id", "event_id", "original_split")},
                              "pretraining_density": density, "pretraining_log_density": float(np.log(density))})
        gate.cache.write_index(directory / "predictions.csv", rows)
        gate.cache.write_json(directory / "prediction_config.json", {
            "limit": None, "threshold": .5, "roi_threshold": .05, "data_provenance": provenance,
            "members": [{"model_seed": i, "sha256": f"member-{i}"} for i in range(3)]})
        gate.cache.write_json(directory / "prediction_summary.json", {
            "complete": True, "limited": False, "patches": count, "requested_patches": count,
            "counts_by_split": {split: count}})
    directory = root / "density"
    directory.mkdir()
    gate.cache.write_index(directory / "spatial_density.csv", densities)
    gate.cache.write_json(directory / "density_config.json", {
        "data_provenance": provenance, "intended_encoder_sha256": "encoder",
        "pretraining": {"sha256": "pretraining"}, "kde": {"bandwidth_km": 100}})
    gate.cache.write_json(directory / "density_summary.json", {
        "complete": True, "limited": False, "patches": len(densities),
        "features_sha256": gate.cache.digest(directory / "spatial_density.csv")})
    return root / "validation", root / "test", directory


class GateMathTests(unittest.TestCase):
    def test_exact_ties_match_all_within_group_permutations(self):
        scores, labels = np.array([0., 0., 1., 1.]), np.array([1, 0, 1, 0])
        permutations = [np.array(a + b) for a in itertools.permutations([0, 1]) for b in itertools.permutations([2, 3])]
        brute = np.mean([np.cumsum(labels[p]) / np.arange(1, 5) for p in permutations], axis=0)
        np.testing.assert_allclose(gate.risk_curve(scores, labels), brute)
        np.testing.assert_allclose(gate.risk_curve(scores[::-1], labels[::-1]), brute)
        np.testing.assert_allclose(gate.risk_curve(np.zeros(4), labels), labels.mean())
        for x, y in (([np.nan], [1]), ([], []), ([1], [2])):
            with self.assertRaises(ValueError):
                gate.risk_curve(x, y)

    def test_feature_matrix_excludes_labels_and_rescales_tiny_density(self):
        rows = [{"average_entropy": 0., "mutual_information": 0., "pretraining_density": (i + 1) * 1e-9,
                 "reference_fraction": i / 10, "predicted_fraction": .1, "F1": i / 10} for i in range(8)]
        matrix = gate.feature_matrix(rows)
        changed = deepcopy(rows)
        for row in changed:
            row.update(reference_fraction=.99, predicted_fraction=.98, F1=.97, event_id="other")
        np.testing.assert_array_equal(matrix, gate.feature_matrix(changed))
        self.assertEqual(matrix.shape, (8, 3))
        model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(matrix, [0] * 4 + [1] * 4)
        self.assertEqual(model.tree_.feature[0], 2)
        np.testing.assert_array_equal(gate.failure_scores(model, matrix), [0] * 4 + [1] * 4)

    def test_native_missing_routing_and_single_class_fold_probabilities(self):
        values = np.array([[np.nan, .1, .2], [np.nan, .1, .2], [.2, .1, .2], [.2, .1, .2]])
        model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(values, [1, 1, 0, 0])
        np.testing.assert_array_equal(gate.failure_scores(model, values), [1, 1, 0, 0])
        for label in (0, 1):
            model.fit(values, np.full(4, label))
            np.testing.assert_array_equal(gate.failure_scores(model, values), np.full(4, label))


class GateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.validation, self.test, self.density = fixture(self.root)

    def test_missing_labels_preserved_and_missing_UT_failure_not_dropped(self):
        rows, _, _ = gate.load_inputs(self.validation, self.density, "validation")
        self.assertEqual(len(rows), 30)
        self.assertIsNone(rows[0]["failure"])
        self.assertEqual(rows[1]["failure"], 1)
        self.assertTrue(np.isnan(rows[1]["average_entropy"]))
        self.assertTrue(np.isnan(gate.feature_matrix(rows)[1, 0]))
        # The boundary is strict: F1 == .6 is a success.
        path = self.validation / "predictions.csv"
        source = gate.cache.read_rows(path, ("patch_id",))
        source[2]["F1"] = .6
        gate.cache.write_index(path, source)
        self.assertEqual(gate.load_inputs(self.validation, self.density, "validation")[0][2]["failure"], 0)

    def test_provenance_changed_ids_and_corrupt_density_are_rejected(self):
        path = self.density / "density_config.json"
        config = gate.read_json(path)
        config["data_provenance"]["reference_support"] = "label"
        gate.cache.write_json(path, config)
        with self.assertRaisesRegex(ValueError, "support identities"):
            gate.load_inputs(self.validation, self.density, "validation")
        config["data_provenance"]["reference_support"] = "joint"
        gate.cache.write_json(path, config)
        with (self.density / "spatial_density.csv").open("a") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(ValueError, "density export"):
            gate.load_inputs(self.validation, self.density, "validation")

    def test_optional_embedding_features_are_joined_but_never_tree_inputs(self):
        directory = self.root / "embedding"
        directory.mkdir()
        densities = gate.cache.read_rows(self.density / "spatial_density.csv", ("patch_id",))
        embeddings = [{**{k: r[k] for k in ("patch_id", "event_id", "original_split")},
                       "normalized_distance": 1. + i / 100, "NCDD": 200. - i} for i, r in enumerate(densities)]
        gate.cache.write_index(directory / "embedding_uncertainty.csv", embeddings)
        gate.cache.write_json(directory / "embedding_config.json", {"downstream_metadata": {"cache_fingerprint": "cache"}})
        gate.cache.write_json(directory / "embedding_summary.json", {
            "n_patches": len(embeddings), "scores_sha256": gate.cache.digest(directory / "embedding_uncertainty.csv")})
        rows, _, _ = gate.load_inputs(self.validation, self.density, "validation", directory)
        self.assertEqual(rows[0]["normalized_distance"], 1.)
        self.assertEqual(gate.feature_matrix(rows).shape[1], 3)
        gate.cache.write_json(directory / "embedding_config.json", {"downstream_metadata": {"cache_fingerprint": "wrong"}})
        with self.assertRaisesRegex(ValueError, "Embedding and prediction caches differ"):
            gate.load_inputs(self.validation, self.density, "validation", directory)

    def test_fit_score_and_held_out_evaluation_are_separate_and_grouped(self):
        rows, protocol, sources = gate.load_inputs(self.validation, self.density, "validation")
        output = self.root / "gate"
        configuration = gate.train(rows, protocol, sources, output, folds=3, min_samples_leaf=2)
        held_ids = [key for fold in configuration["folds"] for key in fold["held_patch_ids"]]
        labelled_ids = {r["patch_id"] for r in rows if r["failure"] is not None}
        self.assertEqual(set(held_ids), labelled_ids)
        self.assertEqual(len(held_ids), len(labelled_ids))
        self.assertIn(rows[1]["patch_id"], configuration["fit_patch_ids"])
        self.assertNotIn(rows[0]["patch_id"], configuration["fit_patch_ids"])
        for fold in configuration["folds"]:
            self.assertFalse(set(fold["fit_events"]) & set(fold["held_events"]))
        exported = gate.cache.read_rows(output / "gate_scores.csv", ("patch_id", "gate_score", "oof_gate_score"))
        self.assertEqual(len(exported), len(rows))
        self.assertEqual(exported[0]["oof_gate_score"], "")
        self.assertTrue(all(np.isfinite(float(r["gate_score"])) for r in exported))
        summary = gate.read_json(output / "evaluation_summary.json")
        self.assertEqual(summary["n_labelled"], 29)
        self.assertEqual(summary["cohorts"]["common_baselines"]["n"], 28)
        test_rows, test_protocol, test_sources = gate.load_inputs(self.test, self.density, "test")
        with patch.object(DecisionTreeClassifier, "fit", side_effect=AssertionError("Never refit on held-out data")):
            evaluation = gate.evaluate(test_rows, test_protocol, test_sources, output, self.root / "evaluation")
        self.assertEqual(evaluation["n_input_patches"], 10)
        self.assertEqual(evaluation["n_labelled"], 9)
        self.assertEqual(evaluation["overlapping_fit_event_ids"], ["event_0", "event_1"])
        self.assertTrue((self.root / "evaluation/risk_coverage.png").is_file())
        wrong = deepcopy(test_protocol)
        wrong["roi_threshold"] = .1
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            gate.evaluate(test_rows, wrong, test_sources, output, self.root / "bad")
        bad_rows = deepcopy(test_rows)
        bad_rows[0]["patch_id"] = configuration["fit_patch_ids"][0]
        with self.assertRaisesRegex(ValueError, "overlaps fitted"):
            gate.evaluate(bad_rows, test_protocol, test_sources, output, self.root / "bad")
        with self.assertRaisesRegex(ValueError, "Fit only on validation"):
            gate.train(rows + test_rows, protocol, sources, self.root / "mixed", folds=3)
        with self.assertRaisesRegex(ValueError, "validation-only"):
            gate.load_inputs(self.test, self.density, "validation")


if __name__ == "__main__":
    unittest.main()
