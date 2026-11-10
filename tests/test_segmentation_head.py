"""Implementation checks using tiny rasters/heads; no dataset model is trained."""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from experiments import inspect_segmentation_data as inspection
    from scripts import cache_encoder_features as cache
    from scripts import segmentation_common as common
    from scripts import train_segmentation_head as training
    from scripts import predict_segmentation as prediction


@unittest.skipUnless(HAS_TORCH, "requires uv's encoder extra")
class NumericalTests(unittest.TestCase):
    def setUp(self):
        common.configure_runtime(17, "cpu", 2)

    def test_original_decoder_shape_and_no_architecture_simplification(self):
        model = common.build_decoder().eval()
        self.assertEqual(model.__class__.__module__, "signal_task.models.model_components.decoder")
        self.assertEqual(model.depths, [2, 2, 8, 2])
        self.assertEqual(model.dims, [160, 320, 640, 1280])
        with torch.inference_mode():
            logits = model(torch.zeros(1, 384, 14, 14))
        self.assertEqual(tuple(logits.shape), (1, 2, 224, 224))
        self.assertTrue(torch.isfinite(logits).all().item())
        # Also verify the reported scale, rather than substituting a tiny test head.
        self.assertGreater(sum(p.numel() for p in model.parameters()), 40_000_000)
        self.assertLess(sum(p.numel() for p in model.parameters()), 43_000_000)

    def test_counts_ignore_invalid_pixels_and_empty_scores_are_undefined(self):
        target = torch.tensor([[1, 0, -1], [1, 0, -1]])
        counts = common.water_counts(torch.ones_like(target, dtype=torch.bool), target)
        self.assertEqual(counts, dict(tp=2, fp=2, fn=0, n_valid_pixels=4))
        self.assertAlmostEqual(common.water_metrics(counts)["F1"], 2 / 3)
        self.assertEqual(common.water_metrics(counts)["IoU"], .5)
        self.assertEqual(common.water_metrics(dict(tp=0, fp=0, fn=0)), dict(F1=None, IoU=None))

    def test_native_resize_changes_logits_only_and_rejects_nonfinite_output(self):
        labels = torch.tensor([[0, 1, -1], [-1, 0, 1]])
        before = labels.clone()
        logits = common.native_logits(torch.zeros(1, 2, 1, 1), labels.shape)
        self.assertEqual(logits.shape, (1, 2, 2, 3))
        self.assertTrue(torch.equal(labels, before))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            common.native_logits(torch.full((1, 2, 1, 1), float("nan")), labels.shape)

    def test_cross_entropy_has_no_gradient_on_invalid_targets(self):
        class PixelHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pixels = torch.nn.Parameter(torch.zeros(1, 2, 2, 2))

            def forward(self, features):
                return self.pixels.expand(len(features), -1, -1, -1)

        model = PixelHead()
        sample = {"features": torch.zeros(384, 14, 14), "target": torch.tensor([[1, -1], [0, -1]])}
        loader = torch.utils.data.DataLoader([sample], collate_fn=common.collate_samples)
        metrics = common.run_epoch(model, loader, "cpu", torch.optim.SGD(model.parameters(), lr=.1))
        self.assertAlmostEqual(metrics["loss"], np.log(2), places=6)
        self.assertEqual(metrics["n_valid_pixels"], 2)
        self.assertTrue(torch.equal(model.pixels.grad[0, :, :, 1], torch.zeros(2, 2)))
        self.assertGreater(model.pixels.grad[0, :, :, 0].abs().sum().item(), 0)

    def test_fraction_error_uses_matching_support_and_fixed_inclusive_threshold(self):
        sample = {"row": dict(patch_id="Example_1", event_id="Example", original_split="test"),
                  "target": torch.tensor([[1, 0, -1], [1, 0, -1]], dtype=torch.int8)}
        probability = np.array([[.5, .8, 1], [.4, .1, 1]], dtype=np.float32)
        row, exported, mask, counts = prediction.prediction_result(sample, probability, 7, .5, "p.tif", "m.tif")
        self.assertEqual(row["reference_fraction"], .5)
        self.assertEqual(row["predicted_fraction"], .5)
        self.assertEqual(row["signed_fraction_error"], 0)
        self.assertEqual(row["F1"], .5)
        self.assertAlmostEqual(row["IoU"], 1 / 3)
        self.assertEqual(counts["n_valid_pixels"], 4)
        self.assertTrue(np.isnan(exported[:, 2]).all())
        np.testing.assert_array_equal(mask[:, 2], [-1, -1])
        self.assertEqual(mask[0, 0], 1)
        sample["target"].fill_(-1)
        with self.assertRaisesRegex(ValueError, "No valid"):
            prediction.prediction_result(sample, probability, 7, .5, "p", "m")

    def test_best_checkpoint_uses_validation_and_early_stops(self):
        model = torch.nn.Conv2d(384, 2, 1)
        config = dict(learning_rate=5e-5, weight_decay=5e-6, warmup_epochs=0,
                      epochs=4, device="cpu", max_batches=None, patience=1, debug=False, seed=7)
        values = [dict(loss=.1, F1=.99, IoU=.99), dict(loss=.4, F1=.8, IoU=.7),
                  dict(loss=.01, F1=1., IoU=1.), dict(loss=.2, F1=.6, IoU=.5)]
        train_loader, validation_loader = object(), object()
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "run_epoch", side_effect=values) as run, redirect_stdout(io.StringIO()):
            summary = training.fit(model, train_loader, validation_loader, directory, config, {"fixture": True})
            checkpoint = torch.load(Path(directory) / "best_head.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(summary["best_validation_IoU"], .7)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertIs(run.call_args_list[0].args[1], train_loader)
            self.assertIs(run.call_args_list[1].args[1], validation_loader)
            self.assertEqual(run.call_args_list[1].args, (model, validation_loader, "cpu"))


@unittest.skipUnless(HAS_TORCH, "requires uv's encoder extra")
class DataAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.audit = self.root / "audit/patches.csv"
        self.feature_dir = self.root / "features"
        self.data.mkdir()
        self.audit.parent.mkdir()
        self.feature_dir.mkdir()
        self.manifest = self.data / "manifest.csv"
        rows = []
        for index, split in enumerate(common.SPLITS):
            event = "Bolivia" if split == "bolivia_holdout" else "Example"
            key = f"{event}_{index}"
            row = dict(patch_id=key, event_id=event, original_split=split,
                       image_path=f"images/{key}_S2Hand.tif", label_path=f"labels/{key}_LabelHand.tif")
            rows.append(row)
            height, width = index + 3, 5
            image = np.full((13, height, width), 1500, dtype=np.int16)
            image[:, 0, 0] = 0
            label = np.zeros((1, height, width), dtype=np.int16)
            label[0, 0, 0] = label[0, 1, 1] = label[0, 1, 2] = 1
            label[0, -1, -1] = -1
            for field, values, nodata in (("image_path", image, 0), ("label_path", label, None)):
                path = self.data / row[field]
                path.parent.mkdir(exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                                   count=len(values), dtype="int16", nodata=nodata,
                                   crs="EPSG:4326", transform=from_origin(-65, -14 - index, .01, .01)) as source:
                    source.write(values)
                    if len(values) == 13:
                        source.descriptions = cache.SENTINEL2_BANDS
        common.write_csv(self.manifest, rows)
        audited = [inspection.inspect_pair(row, expected_size=None) for row in inspection.read_manifest(self.manifest)]
        self.assertTrue(all(row["usable"] for row in audited))
        inspection.save_csv(self.audit, audited)
        cache.write_json(self.audit.parent / "inspection_summary.json", {
            "inspection_schema_version": 2, "manifest_sha256": cache.digest(self.manifest), "reference_support": "joint"})
        selected, _, _ = cache.selected_rows(self.manifest, self.audit)
        contract = {"encoder": {"sha256": cache.SHA256}, "spatial_shape": [384, 14, 14], "preprocessing": "fixture"}
        fingerprint = cache.fingerprint(contract)
        cache.write_json(self.feature_dir / "cache_config.json", dict(contract=contract, fingerprint=fingerprint))
        feature_rows = []
        for index, (row, qc) in enumerate(selected):
            identity, label_hash = cache.source_identity(row, qc, self.data)
            feature_path = f"patches/{row['patch_id']}.npz"
            path = self.feature_dir / feature_path
            cache.write_cache(path, np.full((384, 14, 14), .1 * (index + 1), dtype=np.float32),
                              np.zeros(384, dtype=np.float32), identity, fingerprint, contract)
            feature_rows.append({**row, "feature_path": feature_path, "image_sha256": identity["image_sha256"],
                                 "label_sha256": label_hash, "cache_fingerprint": fingerprint,
                                 "cache_file_sha256": cache.digest(path), "encoder_sha256": cache.SHA256})
        common.write_csv(self.feature_dir / "feature_manifest.csv", feature_rows)
        cache.write_json(self.feature_dir / "cache_summary.json", {
            "complete_usable_manifest": True, "cache_fingerprint": fingerprint,
            "manifest_sha256": cache.digest(self.manifest), "inspection_sha256": cache.digest(self.audit),
            "inspection_reference_support": "joint"})

    def catalog(self):
        return common.FeatureCatalog(self.manifest, self.audit, self.feature_dir)

    def training_args(self, output):
        return training.parser().parse_args(["--manifest", str(self.manifest), "--inspection", str(self.audit),
                                            "--feature-dir", str(self.feature_dir), "--output", str(output),
                                            "--debug", "--seed", "9", "--threads", "1"])

    def test_dataset_retains_native_support_and_variable_mask_grids(self):
        catalog = self.catalog()
        a, b = (catalog.prepare(record) for record in catalog.select(("train", "validation")))
        self.assertEqual(a["target"].shape, (3, 5))
        self.assertEqual(b["target"].shape, (4, 5))
        self.assertEqual(a["target"][0, 0].item(), -1)  # image-invalid, label-water
        self.assertEqual(a["target"][-1, -1].item(), -1)  # label-invalid
        self.assertEqual(int((a["target"] == 1).sum()), 2)
        features, samples = common.collate_samples([a, b])
        self.assertEqual(features.shape, (2, 384, 14, 14))
        self.assertEqual(samples[1]["target"].shape, (4, 5))

    def test_changed_label_or_validity_sidecar_is_rejected(self):
        catalog = self.catalog()
        record = catalog.select(("train",))[0]
        path = self.data / record[0]["label_path"]
        with rasterio.open(path, "r+") as source:
            source.write_mask(np.ones(source.shape, dtype=np.uint8) * 255)
        with self.assertRaisesRegex(ValueError, "changed since inspection"):
            catalog.prepare(record)

    def test_corrupt_features_and_changed_root_contract_are_rejected(self):
        catalog = self.catalog()
        record = catalog.select(("train",))[0]
        path = self.feature_dir / record[0]["feature_path"]
        path.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "changed since caching"):
            catalog.prepare(record)
        config = json.loads((self.feature_dir / "cache_config.json").read_text())
        config["contract"]["encoder"]["sha256"] = "different"
        cache.write_json(self.feature_dir / "cache_config.json", config)
        with self.assertRaisesRegex(ValueError, "cache contract"):
            self.catalog()

    def test_tiny_workflow_is_split_safe_and_exports_native_georeferenced_outputs(self):
        output = self.root / "training"
        original_prepare = common.FeatureCatalog.prepare
        touched = []

        def prepare(catalog, record):
            touched.append(record[0]["original_split"])
            return original_prepare(catalog, record)

        # Only replace the head for this integration check. The actual CoreDecoder
        # is independently tested above; no real dataset model is fitted here.
        with patch.object(common, "build_decoder", side_effect=lambda: torch.nn.Conv2d(384, 2, 1)), \
                patch.object(common.FeatureCatalog, "prepare", prepare), redirect_stdout(io.StringIO()):
            training.run_training(self.training_args(output))
            self.assertEqual(touched, ["train", "validation"])
            checkpoint = torch.load(output / "best_head.pt", weights_only=True)
            self.assertEqual(checkpoint["model_seed"], 9)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(set(checkpoint["config"]["selected_patches"]), {"train", "validation"})
            with self.assertRaisesRegex(ValueError, "not empty"):
                training.run_training(self.training_args(output))
            catalog = self.catalog()
            with patch.object(catalog, "provenance", {**catalog.provenance, "reference_support": "label"}), \
                    self.assertRaisesRegex(ValueError, "does not match"):
                common.load_head(output / "best_head.pt", catalog)
            prediction_output = self.root / "prediction"
            args = prediction.parser().parse_args([
                "--checkpoint", str(output / "best_head.pt"), "--manifest", str(self.manifest),
                "--inspection", str(self.audit), "--feature-dir", str(self.feature_dir),
                "--output", str(prediction_output), "--threads", "1"])
            with self.assertRaisesRegex(ValueError, "Debug checkpoint"):
                prediction.run_prediction(args)
            args.allow_debug_head = True
            summary = prediction.run_prediction(args)
        self.assertEqual(summary["patches"], 4)
        with (prediction_output / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(rows[0]), set(prediction.FIELDS))
        self.assertEqual({row["original_split"] for row in rows}, set(common.SPLITS))
        self.assertEqual(rows[-1]["event_id"], "Bolivia")
        for row in rows:
            original = next(record for record in self.catalog().records if record[0]["patch_id"] == row["patch_id"])
            sample = self.catalog().prepare(original)
            valid = sample["target"].numpy() != -1
            with rasterio.open(prediction_output / row["probability_path"]) as probability, \
                    rasterio.open(prediction_output / row["mask_path"]) as mask:
                self.assertEqual(probability.shape, tuple(sample["target"].shape))
                self.assertEqual(probability.crs, sample["geometry"]["crs"])
                self.assertEqual(probability.transform, sample["geometry"]["transform"])
                self.assertEqual(int((mask.read(1) != -1).sum()), int(row["n_valid_pixels"]))
                self.assertTrue(np.isnan(probability.read(1)[~valid]).all())
                np.testing.assert_array_equal(mask.read(1)[valid], (probability.read(1)[valid] >= .5).astype(np.int16))
                self.assertAlmostEqual(float(row["predicted_fraction"]), float((mask.read(1)[valid] == 1).mean()))


if __name__ == "__main__":
    unittest.main()
