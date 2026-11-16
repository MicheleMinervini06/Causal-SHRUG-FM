"""Spherical KDE and tiny georeferenced-input checks; no GPU/model inference."""

from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments import inspect_segmentation_data as inspection
from scripts import compute_spatial_density as density
from tests import test_inspect_segmentation_data as inspection_tests


class DensityMathTests(unittest.TestCase):
    def test_normalization_integrates_to_one_on_sphere(self):
        for bandwidth in (100., density.EARTH_RADIUS_KM, 50000.):
            theta = np.linspace(0, min(math.pi, 12 * bandwidth / density.EARTH_RADIUS_KM), 20001)
            values = np.exp(-.5 * (theta * density.EARTH_RADIUS_KM / bandwidth) ** 2
                            - density.gaussian_log_normalizer(bandwidth))
            mass = 2 * math.pi * density.EARTH_RADIUS_KM ** 2 * np.trapezoid(values * np.sin(theta), theta)
            self.assertAlmostEqual(mass, 1., places=6)
        # Very broad kernels approach uniform surface density, not planar density.
        log_density = density.kde_log_density([[0, 0]], [[0, 0], [180, 0]], 1e9)
        np.testing.assert_allclose(np.exp(log_density), 1 / (4 * math.pi * density.EARTH_RADIUS_KM ** 2), rtol=1e-9)

    def test_equal_weight_kde_matches_known_kernel_values(self):
        bandwidth = 100.
        result = density.kde_log_density([[0, 0], [1, 0]], [[0, 0], [.5, 0], [10, 0]], bandwidth)
        distance = math.pi / 180 * density.EARTH_RADIUS_KM
        expected = math.log((1 + math.exp(-.5 * (distance / bandwidth) ** 2)) / 2)
        expected -= density.gaussian_log_normalizer(bandwidth)
        self.assertAlmostEqual(result[0], expected, places=12)
        self.assertGreater(result[0], result[2])
        # Repeating every location equally must not multiply normalized density.
        repeated = density.kde_log_density([[0, 0], [1, 0]] * 4, [[0, 0], [.5, 0], [10, 0]], bandwidth)
        np.testing.assert_allclose(result, repeated, atol=1e-12)

    def test_dateline_and_polar_geometry_are_spherical(self):
        dateline = density.kde_log_density([[180, 0]], [[179.9, 0], [-179.9, 0]], 50.)
        equator = density.kde_log_density([[0, 0]], [[.1, 0], [-.1, 0]], 50.)
        np.testing.assert_allclose(dateline, equator, atol=1e-12)
        pole = density.kde_log_density([[0, 89]], [[180, 89]], 100.)
        equator = density.kde_log_density([[0, 0]], [[2, 0]], 100.)
        np.testing.assert_allclose(pole, equator, atol=1e-11)

    def test_log_density_survives_tiny_raw_density_and_invalid_inputs_fail(self):
        log_density = density.kde_log_density([[0, 0]], [[180, 0]], 1.)
        self.assertTrue(np.isfinite(log_density).all())
        self.assertEqual(np.exp(log_density)[0], 0.)
        for coordinates in ([], [[float("nan"), 0]], [[0, 91]], [[181, 0]], [[0]]):
            with self.assertRaisesRegex(ValueError, "coordinates"):
                density.checked_coordinates(coordinates)
        for bandwidth in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Bandwidth"):
                density.kde_log_density([[0, 0]], [[1, 0]], bandwidth)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "coordinates.csv"

    def read_fixture(self, text):
        self.path.write_bytes(text.encode())
        with patch.object(density, "SOURCE_SHA256", density.cache.digest(self.path)):
            return density.read_pretraining(self.path)

    def test_official_headerless_format_blank_lines_and_unique_ids(self):
        coordinates, metadata = self.read_fixture("0,-65,-14\r\r\n1,10,20\r\r\n")
        np.testing.assert_array_equal(coordinates, [[-65, -14], [10, 20]])
        self.assertEqual(metadata["locations"], 2)
        for text in ("id,longitude,latitude\n0,0,0\n", "0,0,0\n0,1,1\n", "0,1,100\n"):
            with self.assertRaises(ValueError):
                self.read_fixture(text)

    def test_corrupt_source_and_bad_download_are_rejected_without_partial_file(self):
        self.path.write_text("wrong file")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            density.read_pretraining(self.path)
        destination = self.root / "download.csv"
        with patch.object(density.urllib.request, "urlopen") as urlopen, \
                redirect_stdout(io.StringIO()):
            urlopen.return_value.__enter__.return_value.read.side_effect = [b"<html>not CSV</html>", b""]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                density.read_pretraining(destination, download=True)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name(destination.name + ".part").exists())

    def test_projected_full_patch_midpoint_and_stale_geometry(self):
        path = self.root / "image.tif"
        affine = from_origin(-1000, 1000, 250, 500)  # 8x4 footprint centered on (0,0).
        with rasterio.open(path, "w", driver="GTiff", height=4, width=8, count=1,
                           dtype="uint8", crs="EPSG:3857", transform=affine) as image:
            image.write(np.ones((1, 4, 8), dtype=np.uint8))
        row = {"patch_id": "Test_1", "image_path": "image.tif"}
        audit = {"image_crs": "EPSG:3857", "image_width": "8", "image_height": "4",
                 "image_transform": json.dumps(list(affine)[:6])}
        np.testing.assert_allclose(density.patch_center(row, audit, self.root), [0, 0], atol=1e-12)
        audit["image_width"] = "4"
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            density.patch_center(row, audit, self.root)


class WorkflowTests(unittest.TestCase):
    pair = inspection_tests.InspectionTests.pair
    manifest = inspection_tests.InspectionTests.manifest

    def test_preserves_ids_excludes_unusable_and_records_provenance(self):
        inspection_tests.InspectionTests.setUp(self)
        for index, split in enumerate(density.SPLITS):
            self.pair(key=f"{'Bolivia' if split == 'bolivia_holdout' else 'Test'}_{index}", split=split)
        self.pair(key="Test_bad", labels=[[-1, -1], [-1, -1]])
        manifest = self.manifest()
        audit = self.root / "audit/patches.csv"
        audit.parent.mkdir()
        inspected = [inspection.inspect_pair(row, expected_size=None) for row in inspection.read_manifest(manifest)]
        inspection.save_csv(audit, inspected)
        density.cache.write_json(audit.parent / "inspection_summary.json", {
            "manifest_sha256": density.cache.digest(manifest), "inspection_schema_version": 2,
            "reference_support": "joint"})
        coordinates = self.root / "pretraining.csv"
        coordinates.write_text("0,-65,-14\n1,10,20\n")
        args = density.parser().parse_args([
            "--manifest", str(manifest), "--inspection", str(audit),
            "--pretraining-coordinates", str(coordinates), "--output", str(self.root / "output")])
        with patch.object(density, "SOURCE_SHA256", density.cache.digest(coordinates)), \
                redirect_stdout(io.StringIO()):
            summary = density.run(args)
            with self.assertRaisesRegex(ValueError, "not empty"):
                density.run(args)
            args.output, args.limit, args.splits = self.root / "limited", 1, ["validation"]
            limited = density.run(args)
        self.assertEqual(summary["patches"], 4)
        self.assertEqual(summary["counts_by_split"], dict.fromkeys(density.SPLITS, 1))
        self.assertEqual(summary["skipped_unusable_patches"][0]["patch_id"], "Test_bad")
        self.assertEqual(limited["counts_by_split"], {"validation": 1})
        with (self.root / "output/spatial_density.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(rows[0]), set(density.FIELDS))
        self.assertEqual(rows[-1]["event_id"], "Bolivia")
        for row in rows:
            self.assertAlmostEqual(float(row["longitude"]), -64.99)
            self.assertAlmostEqual(float(row["latitude"]), -14.01)
            self.assertAlmostEqual(math.log(float(row["pretraining_density"])), float(row["pretraining_log_density"]))
        config = json.loads((self.root / "output/density_config.json").read_text())
        self.assertEqual(config["kde"]["bandwidth_km"], 100.)
        self.assertEqual(config["pretraining"]["sha256"], density.cache.digest(coordinates))
        self.assertEqual(config["data_provenance"]["manifest_sha256"], density.cache.digest(manifest))


if __name__ == "__main__":
    unittest.main()
