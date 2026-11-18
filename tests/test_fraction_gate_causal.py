"""E03 frozen scoring, E02 replay, matched selection and Monte Carlo contrasts."""

from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
from shutil import copytree
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiments import fraction_gate_causal as e03
from experiments import fraction_gate_bias as e04
from experiments import semisynthetic_confounder as e02
from tests.test_semisynthetic_confounder import fixture


class PhaseTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        e02_dir = cls.root / "e02"
        scores_path = cls.root / "gate_scores.csv"
        with redirect_stdout(io.StringIO()):
            fixture(cls.root)
            raw = e02.read_rows(scores_path)
            for i, row in enumerate(raw):
                row.update(average_entropy=.4 if i else "", mutual_information=.01 if i else "",
                           pretraining_density=(i + 1) * 1e-9, roi_empty=not i, event_id="test_event",
                           normalized_distance=i / 10, NCDD=-i)
            e02.write_csv(scores_path, raw[::-1])  # Gate scores and E02 patch table have different orders.
            e02.run(Namespace(output=e02_dir, repeats=4, seed=42, a0=0., gate_evaluation_dir=cls.root,
                              inspection=cls.root / "inspection.csv", gate_config=cls.root / "gate_config.json"))
        gates = cls.root / "gates"
        summary = {"model_sha256": {}, "delta": .02}
        train_x = [[.1, .03, .1], [.4, .01, .8], [.3, .02, 1.1], [.2, .04, 2.0]]
        for kind, tree, labels in (("regression", DecisionTreeRegressor(random_state=0), [.1, .2, .3, .1]),
                                   ("classification", DecisionTreeClassifier(random_state=0), [0, 0, 1, 1])):
            directory = gates / kind
            directory.mkdir(parents=True)
            path = directory / "gate.pkl"
            path.write_bytes(pickle.dumps(tree.fit(train_x, labels)))
            summary["model_sha256"][kind] = e02.digest(path)
            e02.write_json(directory / "gate_config.json", {
                "schema_version": 2, "kind": kind, "fit_split": "validation", "fit_patch_ids": ["validation_1"],
                "delta": .02 if kind == "classification" else None,
                "protocol": {"ensemble": "fixture"}, "parameters": tree.get_params(),
                "f1_baseline_model_sha256": "frozen-model", "model_sha256": e02.digest(path),
                "features": list(e03.gate.FEATURES), "density_scale": e03.gate.DENSITY_SCALE,
                "sklearn_version": e03.sklearn.__version__})
        e02.write_json(gates / "training_summary.json", summary)
        for attr, path in (("E02_MANIFEST", e02_dir / "run_summary.json"),
                           ("PHASE1_MANIFEST", gates / "training_summary.json")):
            override = patch.object(e03, attr, e02.digest(path))
            override.start()
            cls.addClassCleanup(override.stop)
        cls.args = Namespace(e02_dir=e02_dir, test_scores=scores_path, gates_dir=gates,
                             output=cls.root / "phase2", repeats=4)
        with redirect_stdout(io.StringIO()), patch.object(DecisionTreeRegressor, "fit", side_effect=AssertionError), \
                patch.object(DecisionTreeClassifier, "fit", side_effect=AssertionError):
            e03.run(cls.args)

    def test_full_run_replays_original_estimates_and_draws(self):
        output = self.args.output
        done = e03.gate.read_json(output / "run_summary.json")
        self.assertEqual((done["full_run"], done["e02_control_groups_verified"], done["regressions_attempted"]),
                         (True, 30, 4 * 2 * 9 * 3))
        for f, sha in done["output_sha256"].items():
            self.assertEqual(e02.digest(output / f), sha)
        for filename in ("estimates.csv", "selections.csv", "summary.csv"):
            new = e02.read_rows(output / filename)
            controls = [r for r in new if r["selection"] in ("full", "random", "gate")]
            self.assertEqual(controls, e02.read_rows(self.args.e02_dir / filename))
        patches = e02.read_rows(output / "patches.csv")
        self.assertEqual(len(patches), 20)
        self.assertEqual(patches[0]["average_entropy"], "")
        self.assertTrue(all(0 <= float(r[col]) <= 1 for _, col in e03.EXTRA_GATES for r in patches))
        self.assertAlmostEqual(float(patches[0]["pretraining_density"]), 1e-9)
        with self.assertRaisesRegex(ValueError, "new output"):
            e03.run(self.args)

    def test_shared_ties_and_nested_retention(self):
        patches = e02.read_rows(self.args.output / "patches.csv")
        for row in patches:
            row.update(regression_score=row["gate_score"], classification_score=row["gate_score"])
        subsets = {(name, coverage): ids for name, coverage, ids in
                   e02.selections(patches, 42, 2, extra_gates=e03.EXTRA_GATES)}
        for c in e02.COVERAGES:
            self.assertEqual(subsets["regression", c], subsets["gate", c])
            self.assertEqual(subsets["classification", c], subsets["gate", c])
        self.assertEqual(subsets["regression", .5], subsets["regression", .75][:10])

    def test_bias_diagnostic_replays_e03_and_preserves_stratum_quotas(self):
        patches = e02.read_rows(self.args.output / "patches.csv")
        for row in patches:
            row["regression_score"] = float(row["regression_score"])
        summary, _ = e04.diagnose(patches, self.args.repeats, seed=42, a0=0.)
        original = {(float(r["a1"]), r["selection"], float(r["target_coverage"])): float(r["bias"])
                    for r in e02.read_rows(self.args.output / "summary.csv") if r["estimator"] == "proxy"}
        for r in summary:
            if r["selection"] in ("regression", "random"):
                self.assertAlmostEqual(r["bias"], original[r["a1"], r["selection"], r["coverage"]])
        # One saturated cell, one partially sampled cell and one with no quota.
        strata = ["A", "A", "B", "B", "B", "C"]
        self.assertEqual(e04.matched_sample([0, 1, 2], [5, 4, 2, 1, 0, 3], strata), [4, 1, 0])

    def test_paired_metrics_use_successful_pairs(self):
        methods = {"regression": [3., 1., None], "random": [1., 2., 9.], "classification": [1.] * 3, "gate": [None] * 3}
        rows = [{"a1": 1., "target_coverage": .5, "selection": name, "repetition": i, "tau_proxy": v}
                for name, values in methods.items() for i, v in enumerate(values)]
        contrasts = {(r["selection"], r["comparator"]): r for r in e03.paired_contrasts(rows[::-1])}
        compared = contrasts["regression", "random"]
        self.assertEqual(compared["n_successful_pairs"], 2)
        self.assertEqual(compared["delta_bias"], .5)
        self.assertAlmostEqual(compared["mcse_delta_bias"], 1.5)
        self.assertAlmostEqual(compared["delta_rmse"], np.sqrt(2) - np.sqrt(.5))
        self.assertAlmostEqual(compared["mcse_delta_rmse"], 3 / (2 * np.sqrt(2)))
        self.assertIsNone(contrasts["gate", "random"]["delta_bias"])
        self.assertIsNone(contrasts["regression", "classification"]["mcse_delta_rmse"])

    def test_selection_diagnostics_use_retained_errors_and_event_denominators(self):
        patches = [{"patch_id": str(i), "event_id": str(i // 2), "reference_fraction": 0., "predicted_fraction": i / 10}
                   for i in range(4)]
        selections = [{"repetition": rep, "target_coverage": c, "selection": kind,
                       "patch_ids": json.dumps([str(i) for i in order[:int(c * 4)]])}
                      for rep in range(2) for c in e02.COVERAGES
                      for kind, order in (("gate", range(4)), ("random", range(3, -1, -1)),
                                          ("regression", range(4)), ("classification", range(4)))]
        comp, events, overlaps = e03.selection_diagnostics(patches, selections, .02)
        row = next(r for r in comp if r["selection"] == "regression" and r["target_coverage"] == .5)
        self.assertAlmostEqual(row["fraction_MAE"], .05)
        self.assertAlmostEqual(row["fraction_RMSE"], np.sqrt(.005))
        self.assertEqual(row["fraction_error_failure_rate"], .5)
        self.assertEqual([r["event_retention"] for r in events if r["selection"] == "gate"
                          and r["target_coverage"] == .5], [1., 0.])
        self.assertEqual(next(r["mean_shared_patches"] for r in overlaps if r["selection"] == "gate"
                              and r["comparator"] == "random" and r["target_coverage"] == .5), 0.)

    def test_incompatible_files_and_changed_controls_fail(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            e03.load_test(self.args.e02_dir, self.args.output / "patches.csv")
        with patch.object(e03, "PHASE1_MANIFEST", "unreviewed"), self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            e03.score_gates(self.args.gates_dir, [], None, {})
        original = e02.read_rows(self.args.e02_dir / "summary.csv")
        with self.assertRaisesRegex(ValueError, "summary groups"):
            e03.check_baselines([], original)
        current = [{k: v if k in ("estimator", "selection") else float(v) for k, v in row.items()}
                   for row in original]
        current[0]["bias"] += .01
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            e03.check_baselines(current, original)

    def test_replaced_gate_is_rejected_before_deserialization(self):
        gates = copytree(self.args.gates_dir, self.root / "changed-gates")
        (gates / "regression/gate.pkl").write_bytes(b"different model")
        patches, values, baseline, _ = e03.load_test(self.args.e02_dir, self.args.test_scores)
        with patch.object(pickle, "load", side_effect=AssertionError), self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            e03.score_gates(gates, patches, values, baseline)

    def test_embedding_comparison_replays_e03_and_uses_frozen_ue_scores(self):
        ue = self.root / "ue"
        ue.mkdir()
        reg = e03.gate.read_json(self.args.gates_dir / "regression/gate_config.json")
        model = DecisionTreeRegressor(random_state=0).fit(
            [[.4, .01, .1, 0., 0.], [.4, .01, .1, 2., -20.]], [.1, .3])
        path = ue / "gate.pkl"
        path.write_bytes(pickle.dumps(model))
        sha = e02.digest(path)
        e02.write_json(ue / "training_summary.json", {"model_sha256": sha})
        e02.write_json(ue / "gate_config.json", {**reg, "schema_version": 3, "kind": "regression_with_UE",
                       "features": [*e03.gate.FEATURES, *e03.UE], "target": "absolute_fraction_error",
                       "model_sha256": sha, "three_feature_model_sha256": reg["model_sha256"]})
        args = Namespace(**{**vars(self.args), "ue_dir": ue, "e03_dir": self.args.output,
                            "output": self.root / "extended"})
        with patch.object(e03, "UE_MANIFEST", e02.digest(ue / "training_summary.json")), \
             patch.object(e03, "E03_MANIFEST", e02.digest(args.e03_dir / "run_summary.json")), \
             patch.object(DecisionTreeRegressor, "fit", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            e03.run(args, with_ue=True)
        summary = e03.gate.read_json(args.output / "run_summary.json")
        self.assertEqual(summary["e03_control_groups_verified"], 54)
        contrasts = e02.read_rows(args.output / "paired_comparisons.csv")
        self.assertEqual(len(contrasts), 4)
        self.assertTrue(all((r["selection"], r["comparator"]) == ("regression_with_UE", "regression") for r in contrasts))
        patches = e02.read_rows(args.output / "patches.csv")
        self.assertEqual([float(r["NCDD"]) for r in patches], list(range(0, -20, -1)))
        self.assertEqual({round(float(r["regression_with_UE_score"]), 6) for r in patches}, {.1, .3})
        for name in ("estimates.csv", "selections.csv"):
            controls = [r for r in e02.read_rows(args.output / name) if r["selection"] != "regression_with_UE"]
            self.assertEqual(controls, e02.read_rows(args.e03_dir / name))


if __name__ == "__main__":
    unittest.main()
