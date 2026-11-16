"""Analytical uncertainty checks and a tiny-raster integration check only."""

from contextlib import redirect_stdout
import csv
import io
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_segmentation_head as head_tests

if head_tests.HAS_TORCH:
    import rasterio
    import torch
    from scripts import predict_segmentation_ensemble as ensemble
    from scripts import segmentation_common as common
    from scripts import train_segmentation_head as training
    from signal_task.models.uq_model_components.ensemble_uq_model import EnsembleUQModel


@unittest.skipUnless(head_tests.HAS_TORCH, "requires uv's encoder extra")
class UncertaintyTests(unittest.TestCase):
    def test_uncertain_agreement_and_confident_disagreement(self):
        maps = ensemble.ensemble_maps([[[.5, .01]], [[.5, .99]]])
        np.testing.assert_allclose(maps["probability"], [[.5, .5]])
        np.testing.assert_allclose(maps["predictive_entropy"], math.log(2))
        self.assertAlmostEqual(float(maps["average_entropy"][0, 0]), math.log(2), places=6)
        self.assertEqual(maps["mutual_information"][0, 0], 0)
        low_entropy = -.01 * math.log(.01) - .99 * math.log(.99)
        self.assertAlmostEqual(float(maps["average_entropy"][0, 1]), low_entropy, places=6)
        self.assertAlmostEqual(float(maps["mutual_information"][0, 1]), math.log(2) - low_entropy, places=6)

    def test_endpoints_decomposition_order_and_original_implementation(self):
        p = torch.tensor([[[0., 1., .02, .5]], [[0., 1., .98, .5]], [[0., 1., .4, .8]]], dtype=torch.float64)
        maps = ensemble.ensemble_maps(p)
        reversed_maps = ensemble.ensemble_maps(p.flip(0))

        class ProbabilityMember(torch.nn.Module):
            def __init__(self, probability):
                super().__init__()
                self.probability = probability

            def forward(self, x):
                return torch.stack((1 - self.probability, self.probability)).unsqueeze(0)

        original = EnsembleUQModel([ProbabilityMember(value) for value in p])
        x = torch.zeros(1)
        np.testing.assert_allclose(maps["predictive_entropy"], original.entropy(x)[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(maps["mutual_information"], original.mutual_information(x)[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(maps["average_entropy"] + maps["mutual_information"], maps["predictive_entropy"], atol=1e-7)
        for name in ensemble.SCORES:
            self.assertTrue(np.isfinite(maps[name]).all())
            self.assertTrue((maps[name] >= 0).all())
            self.assertTrue((maps[name] <= math.log(2) + 1e-7).all())
            np.testing.assert_allclose(maps[name], reversed_maps[name], atol=1e-7)
            np.testing.assert_array_equal(maps[name][0, :2], [0, 0])
        for invalid in ([[[.5]]], [[[float("nan")]], [[.5]]], [[[1.1]], [[.5]]]):
            with self.assertRaisesRegex(ValueError, "probability maps"):
                ensemble.ensemble_maps(invalid)

    def test_roi_uses_inclusive_mean_probability_and_valid_support(self):
        maps = ensemble.ensemble_maps([[[.05, .04, .99]], [[.05, .04, .01]]])
        valid = np.array([[True, True, False]])
        scores = ensemble.patch_uncertainty(maps, valid, .05)
        self.assertEqual(scores["n_roi_pixels"], 1)
        self.assertEqual(scores["roi_fraction"], .5)
        self.assertFalse(scores["roi_empty"])
        self.assertEqual(scores["mutual_information"], 0)
        self.assertAlmostEqual(scores["average_entropy"], -.05 * math.log(.05) - .95 * math.log(.95), places=6)
        # A segmentation threshold of 0.5 would select none of these valid pixels.
        self.assertFalse((valid & (maps["probability"] >= .5)).any())
        empty = ensemble.patch_uncertainty(maps, valid, .9)
        self.assertTrue(empty["roi_empty"])
        self.assertEqual(empty["n_roi_pixels"], 0)
        for name in ensemble.SCORES:
            self.assertIsNone(empty[name])


@unittest.skipUnless(head_tests.HAS_TORCH, "requires uv's encoder extra")
class EnsembleWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing four-split, variable-grid synthetic raster fixture.
        head_tests.DataAndWorkflowTests.setUp(self)

    def make_heads(self):
        paths = []
        for seed in (0, 1):
            output = self.root / f"head-{seed}"
            args = training.parser().parse_args([
                "--manifest", str(self.manifest), "--inspection", str(self.audit),
                "--feature-dir", str(self.feature_dir), "--output", str(output),
                "--epochs", "1", "--patience", "1", "--seed", str(seed), "--threads", "1"])
            training.run_training(args)
            paths.append(output / "best_head.pt")
        return paths

    def prediction_args(self, paths, output):
        return ensemble.parser().parse_args([
            "--checkpoints", *(str(path) for path in paths),
            "--manifest", str(self.manifest), "--inspection", str(self.audit),
            "--feature-dir", str(self.feature_dir), "--output", str(output), "--threads", "1"])

    def test_tiny_workflow_preserves_grids_support_ids_and_validation_default(self):
        with patch.object(common, "build_decoder", side_effect=lambda: torch.nn.Conv2d(384, 2, 1)), \
                redirect_stdout(io.StringIO()):
            paths = self.make_heads()
            output = self.root / "predictions"
            args = self.prediction_args(paths, output)
            original_prepare = common.FeatureCatalog.prepare
            touched = []

            def prepare(catalog, record):
                touched.append(record[0]["original_split"])
                return original_prepare(catalog, record)

            with patch.object(common.FeatureCatalog, "prepare", prepare):
                summary = ensemble.run_prediction(args)
            self.assertEqual(touched, ["validation"])
            self.assertEqual(summary["model_seeds"], [0, 1])
            self.assertFalse(summary["limited"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                ensemble.run_prediction(args)
            args.output = self.root / "all-splits"
            args.splits = list(common.SPLITS)
            summary = ensemble.run_prediction(args)
            empty_args = self.prediction_args(paths, self.root / "empty-roi")
            empty_args.splits = list(common.SPLITS)
            empty_args.roi_threshold, empty_args.limit = 1., 1
            empty_summary = ensemble.run_prediction(empty_args)
        self.assertTrue(empty_summary["limited"])
        self.assertEqual(empty_summary["empty_roi_patches"], 1)
        with (empty_args.output / "predictions.csv").open(newline="") as stream:
            empty_row = next(csv.DictReader(stream))
        self.assertEqual(empty_row["roi_empty"], "True")
        self.assertEqual(empty_row["n_roi_pixels"], "0")
        for name in ensemble.SCORES:
            self.assertEqual(empty_row[name], "")
        self.assertEqual(summary["patches"], 4)
        with (args.output / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(rows[0]), set(ensemble.FIELDS))
        self.assertEqual(rows[-1]["event_id"], "Bolivia")
        catalog = common.FeatureCatalog(self.manifest, self.audit, self.feature_dir)
        for row, record in zip(rows, catalog.select(common.SPLITS)):
            sample = catalog.prepare(record)
            valid = sample["target"].numpy() != -1
            maps = {}
            for name in ("probability", *ensemble.SCORES):
                with rasterio.open(args.output / row[f"{name}_path"]) as source:
                    maps[name] = source.read(1)
                    self.assertEqual(source.shape, tuple(sample["target"].shape))
                    self.assertEqual(source.crs, sample["geometry"]["crs"])
                    self.assertEqual(source.transform, sample["geometry"]["transform"])
                    self.assertTrue(np.isnan(maps[name][~valid]).all())
                    self.assertEqual(source.tags()["model_seeds"], "0;1")
            roi = valid & (maps["probability"] >= .05)
            self.assertEqual(int(row["n_roi_pixels"]), int(roi.sum()))
            self.assertEqual(int(row["n_valid_pixels"]), int(valid.sum()))
            for name in ensemble.SCORES:
                if roi.any():
                    self.assertAlmostEqual(float(row[name]), float(maps[name][roi].mean(dtype=np.float64)))
                else:
                    self.assertEqual(row[name], "")
            with rasterio.open(args.output / row["mask_path"]) as source:
                mask = source.read(1)
                np.testing.assert_array_equal(mask[~valid], -1)
                np.testing.assert_array_equal(mask[valid], maps["probability"][valid] >= .5)
                self.assertAlmostEqual(float(row["predicted_fraction"]), float(mask[valid].mean()))
            self.assertEqual(row["model_seeds"], "0;1")
        config = json.loads((args.output / "prediction_config.json").read_text())
        self.assertEqual(config["threshold"], .5)
        self.assertEqual(config["roi_threshold"], .05)
        self.assertEqual(config["uncertainty"]["units"], "nats; natural logarithm")
        self.assertEqual([member["epoch"] for member in config["members"]], [1, 1])

    def test_incompatible_or_duplicate_members_fail_before_prediction(self):
        with patch.object(common, "build_decoder", side_effect=lambda: torch.nn.Conv2d(384, 2, 1)), \
                redirect_stdout(io.StringIO()):
            paths = self.make_heads()
            catalog = common.FeatureCatalog(self.manifest, self.audit, self.feature_dir)
            with self.assertRaisesRegex(ValueError, "at least two"):
                ensemble.load_members(paths[:1], catalog)
            with self.assertRaisesRegex(ValueError, "distinct"):
                ensemble.load_members([paths[0], paths[0]], catalog)
            checkpoint = torch.load(paths[1], weights_only=True)
            checkpoint["config"]["batch_size"] += 1
            common.save_checkpoint(paths[1], checkpoint)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                ensemble.load_members(paths, catalog)
            checkpoint["data_provenance"]["cache_fingerprint"] = "different"
            common.save_checkpoint(paths[1], checkpoint)
            with self.assertRaisesRegex(ValueError, "does not match"):
                ensemble.load_members(paths, catalog)

    def test_each_members_logits_are_resized_before_softmax_and_averaging(self):
        class GridHead(torch.nn.Module):
            def __init__(self, right_logit):
                super().__init__()
                self.register_buffer("logits", torch.tensor([[[[0., 0.]], [[0., right_logit]]]]))

            def forward(self, x):
                return self.logits

        models = [GridHead(4.), GridHead(-2.)]
        members = [{"model_seed": seed, "sha256": str(seed)} for seed in (0, 1)]
        output = self.root / "resize-check"
        args = self.prediction_args([self.root / "member0", self.root / "member1"], output)
        with patch.object(ensemble, "load_members", return_value=(models, members, {})), \
                redirect_stdout(io.StringIO()):
            ensemble.run_prediction(args)
        with (output / "predictions.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        with rasterio.open(output / row["probability_path"]) as source:
            # The native five-column grid's middle pixel interpolates logits to
            # 2 and -1. Softmax first, or averaging logits first, gives a different result.
            expected = ((1 / (1 + math.exp(-2))) + (1 / (1 + math.exp(1)))) / 2
            self.assertAlmostEqual(float(source.read(1)[2, 2]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
