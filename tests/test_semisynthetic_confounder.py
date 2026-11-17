"""Check scientific pairing, exact-size selection and failure accounting for E02."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import semisynthetic_confounder as e02


def fixture(directory, n=20):
    """A held-out cohort with undefined F1, a tied gate and an exact proxy."""
    patches = [{"patch_id": f"test_{i:03d}", "event_id": "test_event", "original_split": "test",
                "model_seeds": "0;1;2", "n_valid_pixels": 100, "reference_fraction": i / (n - 1),
                "predicted_fraction": i / (n - 1), "gate_score": .5, "score_kind": "held_out", "F1": "" if i == 0 else "1"}
               for i in range(n)]
    inspection = [{"patch_id": r["patch_id"], "event_id": r["event_id"], "original_split": "test", "usable": "True",
                   "reference_valid_pixels": 100, "reference_water_fraction": r["reference_fraction"]} for r in patches]
    with contextlib.redirect_stdout(io.StringIO()):
        e02.write_csv(directory / "gate_scores.csv", patches)
        e02.write_csv(directory / "inspection.csv", inspection)
    config = {"schema_version": 1, "model_sha256": "frozen-model", "fit_patch_ids": ["validation_1"],
              "fit_events": ["test_event"], "protocol": {"ensemble": [[0, "a"], [1, "b"], [2, "c"]],
              "data_provenance": {"inspection_sha256": e02.digest(directory / "inspection.csv")}}}
    e02.write_json(directory / "gate_config.json", config)
    e02.write_json(directory / "evaluation_summary.json", {"split": "test", "gate_model_sha256": "frozen-model", "n_input_patches": n})
    return e02.load_patches(directory, directory / "gate_config.json", directory / "inspection.csv")[0]


class CohortTests(unittest.TestCase):
    def test_complete_cohort_and_shared_reference_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            patches = fixture(directory)
            self.assertEqual(len(patches), 20)  # F1 is not an eligibility condition.
            shifted = [{**r, "predicted_fraction": r["reference_fraction"] + .1} for r in patches]
            c, proxy, scaling = e02.standardize(shifted)
            self.assertAlmostEqual(sum(c), 0)
            self.assertAlmostEqual(sum(v * v for v in c) / len(c), 1)
            for x, z in zip(c, proxy):
                self.assertAlmostEqual(z - x, .1 / scaling["scale"])
            score_path = directory / "gate_scores.csv"
            changed = e02.read_rows(score_path)
            changed[0]["score_kind"] = "validation_in_sample"
            with contextlib.redirect_stdout(io.StringIO()):
                e02.write_csv(score_path, changed)
            with self.assertRaisesRegex(ValueError, "held-out"):
                e02.load_patches(directory, directory / "gate_config.json", directory / "inspection.csv")

    def test_duplicate_or_missing_patch_and_gate_mismatch_rejected(self):
        for problem in ("duplicate", "missing", "model", "support"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                fixture(directory)
                patched = e02.read_rows(directory / "gate_scores.csv")
                if problem == "duplicate":
                    patched[-1] = patched[0]
                elif problem == "missing":
                    patched.pop()
                elif problem == "support":
                    patched[0]["n_valid_pixels"] = 99
                else:
                    e02.write_json(directory / "evaluation_summary.json", {"split": "test", "gate_model_sha256": "other", "n_input_patches": 20})
                with contextlib.redirect_stdout(io.StringIO()):
                    e02.write_csv(directory / "gate_scores.csv", patched)
                with self.assertRaises(ValueError):
                    e02.load_patches(directory, directory / "gate_config.json", directory / "inspection.csv")


class SimulationTests(unittest.TestCase):
    def test_nested_ties_keep_exact_counts_and_prefer_lower_scores(self):
        patches = [{"gate_score": i // 20} for i in range(88)]
        original = [r.copy() for r in patches]
        first = e02.selections(patches, 42, 0)
        self.assertEqual(first, e02.selections(patches, 42, 0))
        self.assertNotEqual(first, e02.selections(patches, 42, 1))
        groups = {(name, c): ids for name, c, ids in first}
        for name in ("gate", "random"):
            self.assertEqual(len(set(groups[(name, .5)])), 44)
            self.assertEqual(len(set(groups[(name, .75)])), 66)
            self.assertLessEqual(set(groups[(name, .5)]), set(groups[(name, .75)]))
        for coverage in e02.COVERAGES:
            kept = set(groups[("gate", coverage)])
            self.assertLessEqual(max(patches[i]["gate_score"] for i in kept),
                                 min(patches[i]["gate_score"] for i in range(88) if i not in kept))
        self.assertEqual(patches, original)

    def test_DGP_pairing_and_randomized_control(self):
        c, uniforms, noise = [-2, -.5, .5, 2], [.3, .6, .4, .8], [.1, -.2, .3, -.4]
        t0, y0, p0 = e02.simulate(c, uniforms, noise, 0, 0)
        t1, y1, _ = e02.simulate(c, uniforms, noise, 0, 1)
        self.assertEqual(p0, [.5] * 4)
        self.assertEqual(t0, [1, 0, 1, 0])
        self.assertNotEqual(t0, t1)
        for y, t in ((y0, t0), (y1, t1)):
            for outcome, treatment, confounder, epsilon in zip(y, t, c, noise):
                self.assertAlmostEqual(outcome - treatment - confounder, epsilon)

    def test_exact_proxy_matches_oracle_and_draws_reproduce(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            patches = fixture(Path(tmp), n=40)
            estimates, selected = e02.run_simulation(patches, 8, 42, 0)
            self.assertEqual((estimates, selected), e02.run_simulation(patches, 8, 42, 0))
        self.assertEqual(len(estimates), 8 * 2 * 5)
        self.assertEqual(len(selected), 8 * 5)
        for row in estimates:
            self.assertEqual(row["status_oracle"], "ok")
            self.assertEqual(row["tau_oracle"], row["tau_proxy"])
        for repetition in range(8):
            full = [r for r in estimates if r["repetition"] == repetition and r["selection"] == "full"]
            subsets = [r for r in selected if r["repetition"] == repetition]
            self.assertEqual(len(full), 2)  # Full cohort is not duplicated per coverage.
            self.assertEqual({r["n"] for r in subsets}, {40, 30, 20})

    def test_each_method_uses_same_full_treatment_outcome_and_scaling(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            patches = fixture(Path(tmp), n=40)
            with patch.object(e02, "estimate", wraps=e02.estimate) as fitted:
                e02.run_simulation(patches, 1, 42, 0)
        for offset in (0, 5):
            y, t, c, z = fitted.call_args_list[offset].args
            full = {x: (outcome, treatment, proxy) for x, outcome, treatment, proxy in zip(c, y, t, z)}
            for call in fitted.call_args_list[offset + 1:offset + 5]:
                sy, st, sc, sz = call.args
                self.assertEqual([(outcome, treatment, proxy) for outcome, treatment, proxy in zip(sy, st, sz)],
                                 [full[x] for x in sc])

    def test_degenerate_fits_are_recorded_without_dropping_attempts(self):
        failed = e02.estimate([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3], [0, 1, 2, 3])
        valid = e02.estimate([0, 2, 2, 4], [0, 1, 0, 1], [0, 1, 2, 3], [0, 1, 2, 3])
        base = {"a0": 0, "a1": 1, "selection": "gate", "target_coverage": .5, "actual_coverage": .5, "n": 4}
        summaries = e02.summarize([{**base, **failed}, {**base, **valid}])
        for row in summaries:
            self.assertEqual((row["n_attempted"], row["n_success"], row["n_failed"]), (2, 1, 1))
            self.assertIsNone(row["mcse_bias"])
        for row in e02.summarize([{**base, **failed}]):
            self.assertIsNone(row["bias"])
            self.assertIsNone(row["rmse"])

    def test_bias_rmse_and_mcse_do_not_conflate_errors(self):
        base = {"a0": 0, "a1": 1, "selection": "full", "target_coverage": 1., "actual_coverage": 1., "n": 4,
                **{f"status_{e}": "ok" for e in e02.ESTIMATORS}}
        rows = [{**base, **{f"tau_{e}": v for e in e02.ESTIMATORS}} for v in (0, 2)]
        for summary in e02.summarize(rows):
            self.assertEqual(summary["bias"], 0)
            self.assertEqual(summary["rmse"], 1)
            self.assertAlmostEqual(summary["mcse_bias"], 1)

    def test_end_to_end_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            fixture(directory)
            args = argparse.Namespace(gate_evaluation_dir=directory, gate_config=directory / "gate_config.json",
                                      inspection=directory / "inspection.csv", output=directory / "output", repeats=2, seed=42, a0=0.)
            e02.run(args)
            summary = json.loads((args.output / "run_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["regressions_attempted"], 2 * 2 * 5 * 3)
            self.assertEqual(len(e02.read_rows(args.output / "summary.csv")), 30)
            for name, expected_hash in summary["output_sha256"].items():
                self.assertEqual(e02.digest(args.output / name), expected_hash)
            with self.assertRaisesRegex(ValueError, "empty output"):
                e02.run(args)


if __name__ == "__main__":
    unittest.main()
