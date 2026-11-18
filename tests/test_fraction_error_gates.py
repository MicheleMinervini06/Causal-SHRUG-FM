"""E03 phase-1 checks: identical cohort/folds, continuous tie losses and no test fit."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from scripts import segmentation_gate as gate
from scripts import train_fraction_error_gates as fraction
from tests.test_segmentation_gate import fixture


class RetainedErrorTests(unittest.TestCase):
    def test_continuous_losses_and_squared_losses_match_exact_tie_permutations(self):
        scores, error = np.array([0., 0., 0., 1.]), np.array([.01, .03, .08, .2])
        permutations = [np.array(list(p)+[3]) for p in itertools.permutations([0, 1, 2])]
        for loss in (error, error**2):
            expected = np.mean([np.cumsum(loss[p])/np.arange(1, 5) for p in permutations], axis=0)
            np.testing.assert_allclose(fraction.retained_mean(scores, loss), expected)
            np.testing.assert_allclose(fraction.retained_mean(scores[::-1], loss[::-1]), expected)
        for x, y in (([], []), ([np.nan], [.1]), ([1], [-.1]), ([1], [np.inf]), ([1, 2], [.1])):
            with self.assertRaises(ValueError):
                fraction.retained_mean(x, y)


class FractionGateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        validation, cls.test, cls.density = fixture(cls.root)
        raw = gate.cache.read_rows(validation / "predictions.csv", ("patch_id",))
        for i, row in enumerate(raw):
            if i >= 2:
                row["predicted_fraction"] = .11 if i % 2 == 0 else .14
        # Exactly delta is a negative target. A zero-reference false positive has F1=0.
        raw[2].update(reference_fraction=0., predicted_fraction=.02, F1=0.)
        gate.cache.write_index(validation / "predictions.csv", raw)
        cls.rows, cls.protocol, cls.sources = gate.load_inputs(validation, cls.density, "validation")
        cls.baseline_dir, cls.output = cls.root / "f1", cls.root / "e03"
        with redirect_stdout(io.StringIO()):
            cls.baseline = gate.train(cls.rows, cls.protocol, cls.sources, cls.baseline_dir, folds=3)
            cls.baseline_hashes = {p.name: gate.cache.digest(p) for p in cls.baseline_dir.iterdir() if p.is_file()}
            cls.result = fraction.train(cls.rows, cls.protocol, cls.sources, cls.baseline_dir, cls.output)

    def test_same_fit_ids_and_identical_group_folds_for_both_targets(self):
        self.assertEqual((self.result["n_validation"], self.result["n_fit"], self.result["n_excluded_from_fit_and_OOF"]), (30, 29, 1))
        self.assertEqual(self.result["n_missing_UT_in_fit"], 1)
        expected_positive = sum(abs(r["predicted_fraction"]-r["reference_fraction"]) > .02
                                for r in self.rows if r["failure"] is not None)
        self.assertEqual(self.result["fraction_error_class_counts"][1], expected_positive)
        for kind in fraction.KINDS:
            config = gate.read_json(self.output / kind / "gate_config.json")
            self.assertEqual(config["fit_patch_ids"], self.baseline["fit_patch_ids"])
            held = []
            for original, new in zip(self.baseline["folds"], config["folds"]):
                for key in ("fit_patch_ids", "held_patch_ids", "fit_events", "held_events"):
                    self.assertEqual(new[key], original[key])
                self.assertFalse(set(new["fit_events"]) & set(new["held_events"]))
                held.extend(new["held_patch_ids"])
            self.assertEqual(len(held), len(set(held)))
            self.assertEqual(set(held), set(self.baseline["fit_patch_ids"]))
            self.assertEqual(config["schema_version"], 2)
            self.assertEqual(config["primary"], kind == "regression")
            self.assertEqual(config["delta"], .02 if kind == "classification" else None)

    def test_excluded_patch_preserved_but_not_used_for_OOF_and_nan_row_retained(self):
        for kind in fraction.KINDS:
            exported = gate.cache.read_rows(self.output / kind / "gate_scores.csv", ("patch_id", "oof_gate_score"))
            self.assertEqual(len(exported), 30)
            self.assertEqual(exported[0]["oof_gate_score"], "")
            self.assertEqual(exported[2]["fraction_error_failure"], "0")
            self.assertEqual(exported[2]["f1_failure"], "1")
            self.assertNotIn("failure", exported[2])
            self.assertTrue(all(np.isfinite(float(r["oof_gate_score"])) for r in exported[1:]))
            self.assertTrue(all(np.isfinite(float(r["gate_score"])) for r in exported))
            with (self.output / kind / "gate.pkl").open("rb") as stream:
                model = pickle.load(stream)
            matrix = gate.feature_matrix(self.rows)
            self.assertEqual(model.n_features_in_, 3)
            self.assertTrue(np.isnan(matrix[1, 0]))
            self.assertTrue(np.isfinite(model.predict(matrix)).all())
            structure = gate.read_json(self.output / kind / "tree_structure.json")
            self.assertTrue(all(isinstance(r["missing_go_to_left"], bool) for r in structure))
            leaves = [r for r in structure if r["left"] == -1]
            self.assertGreaterEqual(min(r["n_fit_samples"] for r in leaves), 5)
        self.assertEqual(self.baseline_hashes, {p.name: gate.cache.digest(p) for p in self.baseline_dir.iterdir() if p.is_file()})

    def test_comparison_uses_common_OOF_cohort_and_correct_fraction_units(self):
        metrics = self.result["metrics"]
        self.assertEqual(set(metrics), {"regression", "classification", "f1_gate", "random_expected", "oracle_diagnostic"})
        error = np.array([abs(r["predicted_fraction"]-r["reference_fraction"]) for r in self.rows if r["failure"] is not None])
        for values in metrics.values():
            full = values["at_coverage"][0]
            self.assertEqual(full["retained"], 29)
            self.assertAlmostEqual(full["MAE"], error.mean())
            self.assertAlmostEqual(full["RMSE"], np.sqrt(np.mean(error**2)))
        for values in metrics["random_expected"]["at_coverage"]:
            self.assertAlmostEqual(values["MAE"], error.mean())
        self.assertEqual([r["retained"] for r in metrics["regression"]["at_coverage"]], [29, 21, 14])

    def test_validation_only_and_existing_output_guards(self):
        test_rows, _, _ = gate.load_inputs(self.test, self.density, "test")
        with self.assertRaisesRegex(ValueError, "validation only"):
            fraction.train(self.rows+test_rows, self.protocol, self.sources, self.baseline_dir, self.root / "mixed")
        with self.assertRaisesRegex(ValueError, "empty E03"):
            fraction.train(self.rows, self.protocol, self.sources, self.baseline_dir, self.output)
        wrong = deepcopy(self.protocol)
        wrong["roi_threshold"] = .4
        with self.assertRaisesRegex(ValueError, "protocol differ"):
            fraction.train(self.rows, wrong, self.sources, self.baseline_dir, self.root / "wrong-protocol")
        with self.assertRaisesRegex(ValueError, "checksums differ"):
            fraction.train(self.rows, self.protocol, {"predictions.csv": "wrong"}, self.baseline_dir, self.root / "wrong-source")
        changed = deepcopy(self.rows)
        changed[1]["failure"] = None
        with self.assertRaisesRegex(ValueError, "fitting cohort"):
            fraction.train(changed, self.protocol, self.sources, self.baseline_dir, self.root / "wrong-cohort")

    def test_event_leakage_and_missing_OOF_scores_are_rejected(self):
        for problem in ("events", "OOF", "version"):
            with self.subTest(problem=problem):
                directory = self.root / ("bad-"+problem)
                directory.mkdir()
                config = deepcopy(self.baseline)
                raw = gate.cache.read_rows(self.baseline_dir / "gate_scores.csv", ("patch_id",))
                if problem == "events":
                    config["folds"][0]["held_events"] = config["folds"][0]["fit_events"]
                elif problem == "version":
                    config["sklearn_version"] = "different-version"
                else:
                    raw[1]["oof_gate_score"] = ""
                gate.cache.write_json(directory / "gate_config.json", config)
                gate.cache.write_index(directory / "gate_scores.csv", raw)
                with self.assertRaises(ValueError):
                    fraction.train(self.rows, self.protocol, self.sources, directory, self.root / ("rejected-"+problem))


if __name__ == "__main__":
    unittest.main()
