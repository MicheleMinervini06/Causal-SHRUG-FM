"""Downloader checks using tiny local payloads and mocked HTTP responses."""

import base64
import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from scripts import download_sen1floods11 as downloader


def object_info(name, payload):
    return {
        "name": name, "generation": "123", "size": str(len(payload)),
        "md5Hash": base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(),
    }


class DownloadTests(unittest.TestCase):
    def test_existing_file_requires_matching_size_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.tif"
            payload = b"valid-raster-fixture"
            item = object_info("image.tif", payload)
            path.write_bytes(payload)
            with patch.object(downloader, "open_url") as request:
                self.assertEqual(downloader.fetch_object(item, path), "reused")
                request.assert_not_called()
            path.write_bytes(b"x" * len(payload))
            with patch.object(downloader, "open_url", side_effect=lambda _: io.BytesIO(payload)):
                self.assertEqual(downloader.fetch_object(item, path), "downloaded")
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_failed_checksum_preserves_existing_file_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.tif"
            path.write_bytes(b"old")
            item = object_info("image.tif", b"expected")
            with patch.object(downloader, "open_url", side_effect=lambda _: io.BytesIO(b"bad")), \
                    patch.object(downloader.time, "sleep"), self.assertRaises(ValueError):
                downloader.fetch_object(item, path)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_pagination_keeps_all_pages(self):
        responses = [
            {"items": [{"name": "first"}], "nextPageToken": "next page"},
            {"items": [{"name": "second"}]},
        ]
        with patch.object(downloader, "open_url", side_effect=[
            io.BytesIO(json.dumps(page).encode()) for page in responses
        ]) as request:
            items = downloader.list_objects("prefix/")
        self.assertEqual([item["name"] for item in items], ["first", "second"])
        query = parse_qs(urlparse(request.call_args_list[1].args[0]).query)
        self.assertEqual(query["pageToken"], ["next page"])

    def test_duplicate_patch_ids_are_rejected(self):
        item = object_info("prefix/Bolivia_1_S2Hand.tif", b"image")
        with self.assertRaisesRegex(ValueError, "Duplicate patch"):
            downloader.index_patches([item, item], "_S2Hand.tif")


class SplitTests(unittest.TestCase):
    def contents(self):
        return {
            filename: f"Country_{i}_S1Hand.tif,Country_{i}_LabelHand.tif\n"
            for i, filename in enumerate(downloader.SPLIT_NAMES)
        }

    def test_official_s1_pairs_match_s2_patch_ids_and_preserve_holdout(self):
        assignments = downloader.parse_split_map(self.contents())
        self.assertEqual(assignments["Country_0"], "train")
        self.assertEqual(assignments["Country_3"], "bolivia_holdout")
        self.assertEqual(len(assignments), 4)

    def test_overlapping_splits_are_rejected(self):
        contents = self.contents()
        filenames = list(contents)
        contents[filenames[1]] = contents[filenames[0]]
        with self.assertRaisesRegex(ValueError, "overlapping split"):
            downloader.parse_split_map(contents)

    def test_mismatched_image_and_label_are_rejected(self):
        contents = self.contents()
        contents["flood_train_data.csv"] = "A_1_S1Hand.tif,A_2_LabelHand.tif\n"
        with self.assertRaisesRegex(ValueError, "Mismatched split pair"):
            downloader.parse_split_map(contents)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.payloads = {downloader.EVENT_METADATA: b'{"type":"FeatureCollection","features":[]}'}
        self.inventory = {
            downloader.HAND + "/S2Hand/": [],
            downloader.HAND + "/LabelHand/": [],
            downloader.SPLITS: [],
        }
        for filename, key in zip(downloader.SPLIT_NAMES, ["Ghana_1", "India_2", "Spain_3", "Bolivia_4"]):
            split_payload = f"{key}_S1Hand.tif,{key}_LabelHand.tif\n".encode()
            self.add(downloader.SPLITS, filename, split_payload)
            self.add(downloader.HAND + "/S2Hand/", f"{key}_S2Hand.tif", b"image:" + key.encode())
            self.add(downloader.HAND + "/LabelHand/", f"{key}_LabelHand.tif", b"label:" + key.encode())

    def add(self, prefix, name, payload):
        item = object_info(prefix + name, payload)
        self.inventory[prefix].append(item)
        self.payloads[downloader.object_url(item)] = payload

    def run_main(self, *args):
        with patch("sys.argv", ["download_sen1floods11.py", *map(str, args)]), \
                patch.object(downloader, "list_objects", side_effect=lambda prefix: self.inventory[prefix]), \
                patch.object(downloader, "open_url", side_effect=lambda url: io.BytesIO(self.payloads[url])) as request, \
                redirect_stdout(io.StringIO()):
            downloader.main()
        return request

    def test_dry_run_validates_splits_without_writing_or_requesting_rasters(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            request = self.run_main("--output", output, "--dry-run")
            self.assertFalse(output.exists())
            self.assertEqual(len(request.call_args_list), 4)
            self.assertTrue(all(".csv?" in call.args[0] for call in request.call_args_list))

    def test_download_and_rerun_produce_verified_manifests_and_reuse_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            self.run_main("--output", output, "--limit", "2", "--workers", "1")
            with (output / "manifest.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["patch_id"] for row in rows], ["Bolivia_4", "Ghana_1"])
            self.assertEqual(rows[0]["original_split"], "bolivia_holdout")
            for row in rows:
                self.assertTrue((output / row["image_path"]).is_file())
                self.assertTrue((output / row["label_path"]).is_file())
            audit = json.loads((output / "download_manifest.json").read_text())
            self.assertEqual(audit["available_pairs"], 4)
            self.assertEqual(audit["selected_pairs"], 2)
            self.assertFalse(audit["complete_labelled_sentinel2_subset"])
            self.assertEqual(len(audit["files"]), 8)  # Four rasters and four split CSVs.
            request = self.run_main("--output", output, "--limit", "2", "--workers", "1")
            self.assertEqual([call.args[0] for call in request.call_args_list], [downloader.EVENT_METADATA])


if __name__ == "__main__":
    unittest.main()
