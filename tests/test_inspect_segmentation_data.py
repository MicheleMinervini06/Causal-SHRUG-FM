"""Scientific data-contract checks on tiny synthetic GeoTIFF pairs."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from experiments import inspect_segmentation_data as inspection


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        self.root.mkdir()
        self.rows = []

    def pair(self, key="Test_1", labels=None, image=None, split="train", shift=0,
             label_nodata=None, descriptions=True):
        labels = np.asarray(labels if labels is not None else [[1, 0], [1, -1]], dtype=np.int16)
        image = np.asarray(image if image is not None else np.ones((13, *labels.shape)), dtype=np.float32)
        row = dict(patch_id=key, event_id=key.rsplit("_", 1)[0], original_split=split,
                   image_path=f"images/{key}_S2Hand.tif", label_path=f"labels/{key}_LabelHand.tif")
        for kind, array, nodata in (("image", image, 0), ("label", labels[np.newaxis], label_nodata)):
            path = self.root / row[f"{kind}_path"]
            path.parent.mkdir(exist_ok=True)
            with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1],
                               count=array.shape[0], dtype=array.dtype, crs="EPSG:4326", nodata=nodata,
                               transform=from_origin(-65 + (shift if kind == "label" else 0), -14, .01, .01)) as src:
                src.write(array)
                if kind == "image" and descriptions and array.shape[0] == 13:
                    src.descriptions = inspection.BANDS
        self.rows.append(row)
        return row

    def manifest(self):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=inspection.MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def inspect(self, **kwargs):
        source = inspection.read_manifest(self.manifest())[0]
        return inspection.inspect_pair(source, expected_size=None, **kwargs)

    def test_joint_and_label_fractions_have_explicit_different_denominators(self):
        image = np.ones((13, 2, 2), dtype=np.float32)
        image[:, 0, 0] = 0
        self.pair(image=image)
        row = self.inspect()
        self.assertTrue(row["usable"])
        self.assertEqual(row["label_valid_pixels"], 3)
        self.assertEqual(row["joint_valid_pixels"], 2)
        self.assertEqual(row["label_valid_without_image_pixels"], 1)
        self.assertAlmostEqual(row["label_water_fraction"], 2 / 3)
        self.assertEqual(row["reference_water_fraction"], .5)
        self.assertEqual(row["reference_valid_pixels"], 2)
        self.assertAlmostEqual(self.inspect(support="label")["reference_water_fraction"], 2 / 3)

    def test_empty_label_support_stays_undefined(self):
        self.pair(labels=[[-1, -1], [-1, -1]])
        row = self.inspect()
        self.assertFalse(row["usable"])
        self.assertIn("no_valid_label_pixels", row["errors"])
        self.assertIsNone(row["reference_water_fraction"])
        self.assertIsNone(row["label_water_fraction"])

    def test_no_valid_image_is_unusable_even_with_valid_labels(self):
        self.pair(image=np.zeros((13, 2, 2), dtype=np.float32))
        row = self.inspect()
        self.assertIn("no_valid_image_pixels", row["errors"])
        self.assertFalse(row["usable"])
        self.assertIsNone(row["reference_water_fraction"])
        self.assertIsNotNone(row["label_water_fraction"])

    def test_all_land_and_all_water_are_retained(self):
        for value in (0, 1):
            self.rows.clear()
            self.pair(labels=np.full((2, 2), value))
            row = self.inspect()
            self.assertTrue(row["usable"])
            self.assertEqual(row["reference_water_fraction"], value)

    def test_unknown_labels_are_flagged_without_a_reference_fraction(self):
        self.pair(labels=[[0, 1], [-1, 2]])
        row = self.inspect()
        self.assertIn("unexpected_label_values", row["errors"])
        self.assertIsNone(row["reference_water_fraction"])

    def test_declared_label_nodata_cannot_silently_remove_a_valid_class(self):
        self.pair(label_nodata=0)
        row = self.inspect()
        self.assertIn("label_nodata_conflicts_with_valid_class", row["errors"])
        self.assertFalse(row["usable"])
        self.assertIsNotNone(row["label_value_counts"])

    def test_tiny_grid_roundoff_is_accepted_but_subpixel_shift_is_rejected(self):
        self.pair(shift=1e-12)
        self.assertTrue(self.inspect()["usable"])
        self.rows.clear()
        self.pair(shift=.001)
        row = self.inspect()
        self.assertIn("grid_misalignment", row["errors"])
        self.assertIsNone(row["reference_water_fraction"])

    def test_missing_files_and_wrong_band_count_are_reported(self):
        row = self.pair(image=np.ones((12, 2, 2), dtype=np.float32))
        self.assertIn("unexpected_image_band_count", self.inspect()["errors"])
        (self.root / row["label_path"]).unlink()
        self.assertIn("missing_label_file", self.inspect()["errors"])

    def test_unreadable_raster_is_flagged_instead_of_aborting(self):
        row = self.pair()
        (self.root / row["image_path"]).write_bytes(b"not a GeoTIFF")
        result = self.inspect()
        self.assertIn("raster_read_error", result["errors"])
        self.assertFalse(result["usable"])
        self.assertIsNone(result["reference_water_fraction"])

    def test_shape_mismatch_and_wrong_band_order_are_unusable(self):
        self.pair(image=np.ones((13, 3, 3), dtype=np.float32))
        self.assertIn("dimension_mismatch", self.inspect()["errors"])
        self.rows.clear()
        row = self.pair()
        with rasterio.open(self.root / row["image_path"], "r+") as image:
            image.descriptions = tuple(reversed(inspection.BANDS))
        self.assertIn("unexpected_band_order", self.inspect()["errors"])

    def test_gdal_masked_nonfinite_image_pixels_are_excluded_from_joint_support(self):
        image = np.ones((13, 2, 2), dtype=np.float32)
        image[0, 0, 0] = np.nan
        self.pair(image=image)
        row = self.inspect()
        self.assertTrue(row["usable"])
        self.assertEqual(row["image_nonfinite_pixels"], 1)
        self.assertEqual(row["reference_water_fraction"], .5)
        self.assertIn("nonfinite_image_pixels_excluded", row["warnings"])

    def test_duplicate_ids_are_flagged_on_every_occurrence(self):
        row = self.pair()
        self.rows.append(dict(row, original_split="test"))
        records = inspection.read_manifest(self.manifest())
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIn("duplicate_patch_id", record["errors"])
            self.assertIn("duplicate_image_path", record["errors"])

    def test_paths_cannot_escape_the_dataset_and_official_split_is_checked(self):
        row = self.pair()
        checked = self.inspect(assignments={"Test_1": "test"})
        self.assertIn("official_split_mismatch", checked["errors"])
        row["image_path"] = "../elsewhere.tif"
        record = inspection.read_manifest(self.manifest())[0]
        self.assertIn("unsafe_image_path", record["errors"])
        self.assertNotIn("_image_path", record)

    def test_patch_mean_and_pooled_fraction_are_distinct(self):
        rows = [dict(usable=True, event_id="Test", reference_valid_pixels=1,
                     reference_water_pixels=1, reference_water_fraction=1),
                dict(usable=True, event_id="Test", reference_valid_pixels=3,
                     reference_water_pixels=0, reference_water_fraction=0)]
        summary = inspection.summarize(rows)[0]
        self.assertEqual(summary["water_fraction_mean"], .5)
        self.assertEqual(summary["pooled_water_fraction"], .25)

    def test_examples_cover_original_splits_and_water_fraction_extremes(self):
        rows = [dict(patch_id=f"Patch_{i}", usable=True, original_split=split,
                     reference_water_fraction=value)
                for i, (split, value) in enumerate([
                    ("train", 0), ("train", .02), ("train", .99),
                    ("validation", .04), ("validation", .05), ("validation", .06),
                    ("test", .07), ("test", .08), ("test", .09),
                    ("bolivia_holdout", .1), ("bolivia_holdout", .11), ("bolivia_holdout", .12),
                ])]
        examples = inspection.select_examples(rows, 6)
        self.assertEqual(len(examples), 6)
        self.assertEqual({row["original_split"] for row in examples}, set(inspection.SPLIT_FILES.values()))
        self.assertIn(0, [row["reference_water_fraction"] for row in examples])
        self.assertIn(.99, [row["reference_water_fraction"] for row in examples])

    def test_full_report_includes_bad_patches_groups_figures_and_split_audit(self):
        self.pair("Test_1", labels=np.zeros((2, 2), dtype=np.int16))
        self.pair("Test_2", labels=np.ones((2, 2), dtype=np.int16), split="test")
        self.pair("Other_3", labels=np.full((2, 2), -1, dtype=np.int16), split="validation")
        splits = self.root / "splits"
        splits.mkdir()
        for filename, split in inspection.SPLIT_FILES.items():
            with (splits / filename).open("w", newline="") as stream:
                writer = csv.writer(stream)
                for row in self.rows:
                    if row["original_split"] == split:
                        writer.writerow([row["patch_id"] + "_S1Hand.tif", row["patch_id"] + "_LabelHand.tif"])
        output = Path(self.temporary.name) / "results"
        with redirect_stdout(io.StringIO()):
            report = inspection.run_inspection(self.manifest(), output, expected_size=None, examples=2)
        self.assertEqual(report["overall"]["patch_count"], 3)
        self.assertEqual(report["overall"]["usable_count"], 2)
        self.assertEqual(report["overall"]["water_fraction_mean"], .5)
        self.assertTrue(report["official_splits_cross_checked"])
        self.assertEqual(report["events_shared_between_splits"]["Test"], ["test", "train"])
        self.assertEqual(report["cross_split_bbox_intersections"], 1)
        for filename in ("patches.csv", "by_event.csv", "by_split.csv", "by_event_split.csv",
                         "inspection_summary.json", "report.md", "rgb_mask_examples.png",
                         "water_fraction_distributions.png", "cross_split_bbox_intersections.csv"):
            self.assertTrue((output / filename).is_file(), filename)
        with (output / "patches.csv").open(newline="") as stream:
            patches = list(csv.DictReader(stream))
        self.assertEqual(len(patches), 3)
        self.assertEqual(patches[2]["reference_water_fraction"], "")
        self.assertEqual(json.loads((output / "inspection_summary.json").read_text())["reference_support"], "joint")


if __name__ == "__main__":
    unittest.main()
