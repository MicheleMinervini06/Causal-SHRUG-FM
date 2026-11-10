"""Check encoder acquisition and input validation without pretrained weights."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts import get_ssl4eo_encoder as encoder


class CheckpointTests(unittest.TestCase):
    def test_verified_file_is_reused_and_corrupt_file_is_replaced(self):
        payload = b"tiny-checkpoint-test-fixture"
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(encoder, "EXPECTED_BYTES", len(payload)), \
                patch.object(encoder, "SHA256", hashlib.sha256(payload).hexdigest()), \
                redirect_stdout(io.StringIO()):
            path = Path(temporary) / "encoder.pth"
            path.write_bytes(payload)
            with patch.object(encoder.urllib.request, "urlopen") as request:
                encoder.download(path)
                request.assert_not_called()
            path.write_bytes(b"x" * len(payload))
            with patch.object(encoder.urllib.request, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(payload)):
                encoder.download(path)
            self.assertTrue(encoder.valid_checkpoint(path))
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_failed_verification_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.pth"
            path.write_bytes(b"old")
            with patch.object(encoder.urllib.request, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(b"bad")) as request, \
                    patch.object(encoder.time, "sleep"), self.assertRaises(ValueError):
                encoder.download(path)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_loader_checks_integrity_before_importing_or_loading(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(encoder, "encoder_dependencies") as runtime:
            with self.assertRaisesRegex(ValueError, "Local checkpoint"):
                encoder.load_encoder(Path(temporary) / "missing.pth")
            runtime.assert_not_called()

    def test_loader_uses_strict_weights_only_loading_and_freezes_encoder(self):
        torch, timm, model = Mock(), Mock(), Mock()
        state = {"encoder.weight": object()}
        torch.load.return_value = state
        timm.create_model.return_value = model
        model.to.return_value = model
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        path = Path("verified-fixture.pth")
        with patch.object(encoder, "valid_checkpoint", return_value=True), \
                patch.object(encoder, "encoder_dependencies", return_value=(torch, timm)), \
                redirect_stdout(io.StringIO()):
            self.assertIs(encoder.load_encoder(path), model)
        timm.create_model.assert_called_once_with(
            "vit_small_patch16_224", pretrained=False, in_chans=13, num_classes=0
        )
        torch.load.assert_called_once_with(path, map_location="cpu", weights_only=True)
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once()


class ImageTests(unittest.TestCase):
    def write_image(self, path, data, descriptions=True):
        with rasterio.open(
            path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
            count=data.shape[0], dtype="float32", nodata=0, crs="EPSG:4326",
            transform=from_origin(-65, -14, 0.001, 0.001),
        ) as source:
            source.write(data)
            if descriptions and data.shape[0] == 13:
                source.descriptions = encoder.SENTINEL2_BANDS

    def test_real_raster_reader_reports_band_order_and_nodata_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.tif"
            data = np.full((13, 8, 8), 1500, dtype=np.float32)
            data[:, :, 0] = 0
            self.write_image(path, data)
            raw, info = encoder.read_image(path)
            np.testing.assert_array_equal(raw, data)
            self.assertEqual(info["raster_shape"], [13, 8, 8])
            self.assertEqual(info["band_order_check"], "verified from descriptions")
            self.assertEqual(info["valid_pixel_fraction"], 7 / 8)
            self.assertEqual(info["image_sha256"], encoder.digest(path))

    def test_wrong_band_count_and_nonfinite_or_empty_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.tif"
            samples = [
                np.ones((3, 8, 8), dtype=np.float32),
                np.full((13, 8, 8), np.nan, dtype=np.float32),
                np.zeros((13, 8, 8), dtype=np.float32),
            ]
            for data in samples:
                self.write_image(path, data)
                with self.assertRaises(ValueError):
                    encoder.read_image(path)

    def test_reordered_bands_are_rejected_and_absent_descriptions_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.tif"
            self.write_image(path, np.ones((13, 8, 8), dtype=np.float32), descriptions=False)
            _, info = encoder.read_image(path)
            self.assertIn("assumed", info["band_order_check"])
            with rasterio.open(path, "r+") as source:
                source.descriptions = tuple(reversed(encoder.SENTINEL2_BANDS))
            with self.assertRaisesRegex(ValueError, "band order"):
                encoder.read_image(path)


class CommandTests(unittest.TestCase):
    def test_dry_run_never_accesses_network_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "models"
            with patch("sys.argv", ["get_ssl4eo_encoder.py", "--output", str(output), "--dry-run"]), \
                    patch.object(encoder.urllib.request, "urlopen") as request, \
                    patch.object(encoder, "encoder_dependencies") as runtime, \
                    redirect_stdout(io.StringIO()):
                encoder.main()
            self.assertFalse(output.exists())
            request.assert_not_called()
            runtime.assert_not_called()

    def test_verify_only_is_read_only(self):
        payload = b"tiny-checkpoint-test-fixture"
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(encoder, "EXPECTED_BYTES", len(payload)), \
                patch.object(encoder, "SHA256", hashlib.sha256(payload).hexdigest()):
            output = Path(temporary)
            (output / encoder.FILENAME).write_bytes(payload)
            with patch("sys.argv", ["get_ssl4eo_encoder.py", "--output", str(output), "--verify-only"]), \
                    patch.object(encoder.urllib.request, "urlopen") as request, \
                    redirect_stdout(io.StringIO()):
                encoder.main()
            self.assertEqual([p.name for p in output.iterdir()], [encoder.FILENAME])
            request.assert_not_called()

    def test_missing_runtime_fails_before_downloading(self):
        with patch("sys.argv", ["get_ssl4eo_encoder.py", "--check"]), \
                patch.object(encoder, "encoder_dependencies", side_effect=RuntimeError("missing extra")), \
                patch.object(encoder, "download") as download:
            with self.assertRaisesRegex(RuntimeError, "missing extra"):
                encoder.main()
            download.assert_not_called()

    def test_metadata_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata" / "checkpoint.json"
            encoder.write_json(path, encoder.checkpoint_metadata())
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["revision"], encoder.REVISION)
            self.assertEqual(metadata["sha256"], encoder.SHA256)
            self.assertEqual(metadata["bands"], 13)
            self.assertFalse(path.with_name(path.name + ".part").exists())


if __name__ == "__main__":
    unittest.main()
