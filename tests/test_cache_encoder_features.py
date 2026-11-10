"""Validate token layout, preprocessing, provenance and cache reuse/invalidation."""

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

from experiments import inspect_segmentation_data as inspection
from scripts import cache_encoder_features as cache


HAS_ENCODER = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "patch.npz"
        self.identity = {"patch_id": "Example_1", "image_sha256": "source-hash"}
        self.contract = {"preprocessing": "test", "encoder": "test"}
        self.fingerprint = cache.fingerprint(self.contract)
        self.spatial = np.ones((384, 14, 14), dtype=np.float32)
        self.cls = np.zeros(384, dtype=np.float32)
        cache.write_cache(self.path, self.spatial, self.cls, self.identity, self.fingerprint, self.contract)

    def test_roundtrip_preserves_spatial_and_separate_cls_arrays_without_pickle(self):
        spatial, cls = cache.read_cache(self.path, self.identity, self.fingerprint)
        np.testing.assert_array_equal(spatial, self.spatial)
        np.testing.assert_array_equal(cls, self.cls)
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
        self.assertEqual(metadata["contract"], self.contract)
        self.assertFalse(self.path.with_name(self.path.name + ".part").exists())

    def test_changed_input_or_config_are_incompatible(self):
        with self.assertRaises(cache.IncompatibleCacheError):
            cache.read_cache(self.path, {**self.identity, "image_sha256": "changed"}, self.fingerprint)
        with self.assertRaises(cache.IncompatibleCacheError):
            cache.read_cache(self.path, self.identity, "different-contract")

    def test_finite_payload_corruption_and_invalid_metadata_are_detected(self):
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = archive["metadata"].copy()
        self.spatial[0, 0, 0] = 99
        cache.save_npz(self.path, spatial_features=self.spatial, cls_embedding=self.cls, metadata=metadata)
        with self.assertRaisesRegex(cache.IncompatibleCacheError, "corrupt spatial_features"):
            cache.read_cache(self.path, self.identity, self.fingerprint)
        cache.save_npz(self.path, spatial_features=self.spatial, cls_embedding=self.cls, metadata=np.asarray("[]"))
        with self.assertRaises(cache.IncompatibleCacheError):
            cache.read_cache(self.path, self.identity, self.fingerprint)

    def test_wrong_shapes_and_nonfinite_values_are_rejected(self):
        for spatial in (np.zeros((384,), dtype=np.float32), np.full((384, 14, 14), np.nan, dtype=np.float32)):
            cache.write_cache(self.path, spatial, self.cls, self.identity, self.fingerprint, self.contract)
            with self.assertRaises(cache.IncompatibleCacheError):
                cache.read_cache(self.path, self.identity, self.fingerprint)


@unittest.skipUnless(HAS_ENCODER, "requires uv's encoder extra")
class EncoderMathTests(unittest.TestCase):
    def test_spatial_layout_matches_shrug_decoder_and_cls_is_not_spatial(self):
        import torch
        tokens = torch.arange(2 * 197 * 384, dtype=torch.float32).reshape(2, 197, 384)
        spatial, cls = cache.tokens_to_arrays(tokens)
        self.assertEqual(spatial.shape, (2, 384, 14, 14))
        for batch, channel, row, column in ((0, 0, 0, 0), (0, 17, 3, 9), (1, 383, 13, 13)):
            self.assertEqual(spatial[batch, channel, row, column], tokens[batch, 1 + row * 14 + column, channel])
        np.testing.assert_array_equal(cls, tokens[:, 0, :].numpy())
        with self.assertRaises(ValueError):
            cache.tokens_to_arrays(tokens[:, :196])

    def test_full_patch_normalization_resize_and_nodata_are_deterministic(self):
        import torch
        raw = np.broadcast_to(np.arange(13, dtype=np.float32)[:, None, None] + 1000, (13, 2, 2)).copy()
        valid = np.ones(raw.shape, dtype=bool)
        valid[:, 0, 0] = False
        config = cache.preprocessing_config(cache.ROOT / "src/config/dataset.toml")
        first = cache.preprocess_image(raw, valid, config)
        second = cache.preprocess_image(raw, valid, config)
        self.assertEqual(first.shape, (13, 224, 224))
        self.assertTrue(torch.equal(first, second))
        np.testing.assert_allclose(first[:, 0, 0].numpy(), -np.array(config["mean"]) / np.array(config["std"]), rtol=1e-6)
        np.testing.assert_allclose(first[:, -1, -1].numpy(),
                                   (raw[:, -1, -1] - np.array(config["mean"])) / np.array(config["std"]), rtol=1e-6)
        self.assertIsNone(config["crop"])
        self.assertEqual(config["augmentation"], "none; canonical full-patch image")


@unittest.skipUnless(HAS_ENCODER, "requires uv's encoder extra")
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.manifest = self.data / "manifest.csv"
        self.audit = self.root / "audit" / "patches.csv"
        self.audit.parent.mkdir()
        self.checkpoint = self.root / "encoder.pth"
        self.checkpoint.write_bytes(b"mock-checkpoint")
        self.output = self.root / "features"
        self.rows = []
        for index, split in enumerate(("train", "test")):
            key = f"Example_{index}"
            row = dict(patch_id=key, event_id="Example", original_split=split,
                       image_path=f"images/{key}_S2Hand.tif", label_path=f"labels/{key}_LabelHand.tif")
            self.rows.append(row)
            for field, count, dtype, nodata, value in (("image_path", 13, "float32", 0, 1500),
                                                      ("label_path", 1, "int16", None, index)):
                path = self.data / row[field]
                path.parent.mkdir(exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=count,
                                   dtype=dtype, nodata=nodata, crs="EPSG:4326",
                                   transform=from_origin(-65, -14, .01, .01)) as source:
                    source.write(np.full((count, 2, 2), value, dtype=dtype))
                    if count == 13:
                        source.descriptions = cache.SENTINEL2_BANDS
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=cache.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.refresh_audit()

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward_features(model, inputs):
                assert not model.training
                assert not torch.is_grad_enabled()
                assert torch.is_inference_mode_enabled()
                assert not any(parameter.requires_grad for parameter in model.parameters())
                return inputs.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 197, 384).clone()

        self.mock_loader = patch.object(cache, "load_encoder", side_effect=lambda *a, **kw: TinyEncoder()).start()
        self.addCleanup(patch.stopall)
        patch.object(cache, "valid_checkpoint", return_value=True).start()

    def refresh_audit(self):
        records = [inspection.inspect_pair(row, expected_size=None)
                   for row in inspection.read_manifest(self.manifest)]
        inspection.save_csv(self.audit, records)
        cache.write_json(self.audit.parent / "inspection_summary.json", {
            "inspection_schema_version": 2, "manifest_sha256": cache.digest(self.manifest),
            "reference_support": "joint",
        })

    def run_cache(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return cache.run_cache(self.manifest, self.audit, self.checkpoint, self.output,
                                   cache.ROOT / "src/config/dataset.toml", threads=1, **kwargs)

    def test_first_run_then_reuse_without_an_encoder_forward(self):
        first = self.run_cache(batch_size=1)
        self.assertEqual(first["counts"]["new"], 2)
        paths = sorted((self.output / "patches").glob("*.npz"))
        hashes = [cache.digest(path) for path in paths]
        self.mock_loader.reset_mock()
        second = self.run_cache(batch_size=2)
        self.assertEqual(second["counts"]["reused"], 2)
        self.mock_loader.assert_not_called()
        self.assertEqual(hashes, [cache.digest(path) for path in paths])
        with np.load(self.output / "cls_embeddings.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["cls_embeddings"].shape, (2, 384))
            self.assertEqual(archive["patch_ids"].tolist(), ["Example_0", "Example_1"])

    def test_changed_preprocessing_is_rejected_without_overwriting(self):
        self.run_cache()
        before = cache.digest(self.output / "cache_config.json")
        original = cache.preprocessing_config(cache.ROOT / "src/config/dataset.toml")
        changed = {**original, "resize": {"size": [224, 224], "mode": "bilinear"}}
        with patch.object(cache, "preprocessing_config", return_value=changed), \
                self.assertRaises(cache.IncompatibleCacheError):
            self.run_cache()
        self.assertEqual(before, cache.digest(self.output / "cache_config.json"))

    def test_source_mutation_requires_reinspection_then_explicit_rebuild(self):
        self.run_cache()
        path = self.data / self.rows[0]["image_path"]
        with rasterio.open(path, "r+") as source:
            values = source.read()
            values[:, 0, 0] += 1
            source.write(values)
        with self.assertRaisesRegex(ValueError, "changed since inspection"):
            self.run_cache()
        self.refresh_audit()
        with self.assertRaises(cache.IncompatibleCacheError):
            self.run_cache()
        result = self.run_cache(rebuild_incompatible=True)
        self.assertEqual(result["counts"], {"new": 0, "reused": 1, "rebuild": 1})

    def test_updated_labels_after_reinspection_do_not_recompute_image_features(self):
        self.run_cache()
        with rasterio.open(self.data / self.rows[0]["label_path"], "r+") as source:
            source.write(np.ones((1, 2, 2), dtype=np.int16))
        self.refresh_audit()
        self.mock_loader.reset_mock()
        self.assertEqual(self.run_cache()["counts"]["reused"], 2)
        self.mock_loader.assert_not_called()

    def test_partial_runs_do_not_replace_the_full_index(self):
        self.run_cache(limit=1)
        self.assertTrue((self.output / "feature_manifest.limit-1.csv").exists())
        self.assertFalse((self.output / "feature_manifest.csv").exists())
        self.run_cache()
        before = cache.digest(self.output / "feature_manifest.csv")
        self.run_cache(limit=1)
        self.assertEqual(before, cache.digest(self.output / "feature_manifest.csv"))

    def test_dry_run_writes_nothing(self):
        result = self.run_cache(dry_run=True)
        self.assertEqual(result["new"], 2)
        self.assertFalse(self.output.exists())
        self.mock_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
