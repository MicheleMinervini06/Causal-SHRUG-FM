"""Analytic scores, archive integrity and a tiny synthetic reference pipeline."""

import hashlib
import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import cache_encoder_features as cache
from scripts import cache_pretraining_embeddings as reference
from scripts import compute_embedding_uncertainty as ue
from scripts import download_ssl4eo_reference as download


class ScoreTests(unittest.TestCase):
    def test_known_euclidean_radius_normalization_and_ncdd(self):
        centroids = np.array([[0, 0], [6, 0]], dtype=np.float32)
        rows = ue.score([[0, 0], [2, 0], [3, 0], [6, 0]], centroids, [2, 2])
        np.testing.assert_allclose(rows, [[0, 0, 0, 3], [0, 2, 1, 1], [0, 3, 1.5, 0], [1, 0, 0, 3]])
        # Assignment is before normalization: nearest normalized cluster can differ.
        row = ue.score([[2, 0]], centroids, [1, 10])[0]
        np.testing.assert_allclose(row, [0, 2, 2, -1.6])
        for radii in ([0, 1], [-1, 1], [float("nan"), 1], [1]):
            with self.assertRaises(ValueError):
                ue.score([[2, 0]], centroids, radii)

    def test_fit_uses_mean_euclidean_radius_and_is_reproducible(self):
        rng = np.random.default_rng(9)
        vectors = np.concatenate([rng.normal(-4, .2, (80, 3)), rng.normal(4, .2, (80, 3))]).astype(np.float32)
        centroids, radii, counts = ue.fit(vectors, clusters=2, seed=7, iterations=10, restarts=2, threads=1)
        other = ue.fit(vectors, clusters=2, seed=7, iterations=10, restarts=2, threads=1)
        np.testing.assert_array_equal(centroids, other[0])
        raw = ue.distances(vectors, centroids)
        assigned = raw.argmin(axis=1)
        for j in range(2):
            self.assertEqual(counts[j], (assigned == j).sum())
            self.assertAlmostEqual(radii[j], raw[assigned == j, j].mean(), places=10)
        with self.assertRaises(ValueError):
            ue.fit(vectors[:10], clusters=2)

    def test_end_to_end_synthetic_reference_and_incompatibility_guards(self):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(100, 384)).astype(np.float32)
        contract = {"encoder": {"sha256": "checkpoint"}, "preprocessing": {"name": "fixed"}}
        metadata = {"contract": contract, "reference_complete": True, "payload_sha256": cache.array_hash(vectors)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache.save_npz(root / "reference.npz", cls_embeddings=vectors, patch_ids=np.arange(100).astype(str),
                           metadata=np.asarray(json.dumps(metadata)))
            cache.save_npz(root / "downstream.npz", cls_embeddings=vectors[:2], patch_ids=np.asarray(["a", "b"]),
                           event_ids=np.asarray(["event_a", "event_b"]), original_splits=np.asarray(["train", "validation"]),
                           metadata=np.asarray(json.dumps({"contract": contract})))
            ue.run(root / "reference.npz", root / "downstream.npz", root / "result", clusters=2,
                   threads=1, iterations=5, restarts=1)
            rows = cache.read_rows(root / "result/embedding_uncertainty.csv", ("patch_id", "event_id", "original_split", "NCDD"))
            self.assertEqual([r["patch_id"] for r in rows], ["a", "b"])
            self.assertEqual(rows[1]["event_id"], "event_b")
            metadata["reference_complete"] = False
            cache.save_npz(root / "reference.npz", cls_embeddings=vectors, patch_ids=np.arange(100).astype(str),
                           metadata=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "Debug/incomplete"):
                ue.run(root / "reference.npz", root / "downstream.npz", root / "debug-result", clusters=2)
            metadata["reference_complete"] = True
            metadata["contract"] = {**contract, "preprocessing": {"name": "different"}}
            cache.save_npz(root / "reference.npz", cls_embeddings=vectors, patch_ids=np.arange(100).astype(str),
                           metadata=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                ue.run(root / "reference.npz", root / "downstream.npz", root / "bad-result", clusters=2)


class DownloadTests(unittest.TestCase):
    def test_download_completes_subset_without_draining_archive_and_reuses_it(self):
        buffer = io.BytesIO()
        payload = np.random.default_rng(2).bytes(65536)
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for location in ("0000001", "0000002", "0000003"):
                for season in ("seasonA", "seasonB"):
                    for band in cache.SENTINEL2_BANDS:
                        info = tarfile.TarInfo(f"s2c/{location}/{season}/{band}.tif")
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
        compressed = gzip.compress(buffer.getvalue())
        parts = [{"path": "partaa", "size": len(compressed), "sha256": hashlib.sha256(compressed).hexdigest()}]

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes 0-{len(compressed) - 1}/{len(compressed)}"}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(download, "source_parts", return_value=parts), patch.object(
                    download.urllib.request, "urlopen", return_value=Response(compressed)):
                summary = download.download(output, locations=1)
            self.assertTrue(summary["reference_complete"])
            self.assertEqual(summary["n_locations"], 1)
            self.assertEqual(summary["n_images"], 2)
            self.assertEqual(summary["selected_location_ids_in_archive_order"], ["0000001"])
            self.assertEqual(summary["season_counts"], {"0000001": 2})
            self.assertLess(summary["transferred_bytes"], len(compressed))
            self.assertEqual(summary["fully_verified_archive_parts"], 0)
            self.assertEqual(summary["downloaded_prefix_sha256"], hashlib.sha256(compressed[:summary["transferred_bytes"]]).hexdigest())
            with patch.object(download, "source_parts", side_effect=AssertionError("Must not download again")):
                self.assertEqual(download.download(output, locations=1), summary)

    def test_parts_concatenate_check_hashes_and_resume_after_disconnect(self):
        blobs = [b"abc", b"defghi"]
        parts = [{"path": str(i), "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for i, b in enumerate(blobs)]
        requests = []

        class Response(io.BytesIO):
            status = 206

        def open_request(request, timeout):
            start = int(request.headers["Range"].split("=")[1][:-1])
            part = int(request.full_url.split("?")[0].rsplit("/", 1)[1])
            requests.append((part, start))
            response = Response(blobs[part][start:])
            response.headers = {"Content-Range": f"bytes {start}-{len(blobs[part]) - 1}/{len(blobs[part])}"}
            if len(requests) == 1:
                original_read = response.read

                def disconnect(size):
                    if response.tell():
                        raise OSError("connection lost after a successful chunk")
                    return original_read(size)

                response.read = disconnect
            return response

        with patch.object(download.urllib.request, "urlopen", side_effect=open_request), patch.object(download.time, "sleep"):
            with download.ArchiveStream(parts) as stream:
                chunks = []
                while data := stream.read(2):
                    chunks.append(data)
                self.assertEqual(b"".join(chunks), b"abcdefghi")
                self.assertEqual(stream.index, 2)
                self.assertEqual(requests[:2], [(0, 0), (0, 2)])
            parts[0]["sha256"] = "bad"
            with download.ArchiveStream(parts) as stream, self.assertRaisesRegex(ValueError, "SHA-256"):
                stream.read(3)

    def test_only_prefix_images_and_explicit_band_names_are_retained(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for location in ("0000001", "0000002"):
                for band in cache.SENTINEL2_BANDS:
                    info = tarfile.TarInfo(f"s2c/{location}/20200101T000000/{band}.tif")
                    info.size = 3
                    archive.addfile(info, io.BytesIO(b"tif"))
            unsafe = tarfile.TarInfo("s2c/../../escape/B1.tif")
            unsafe.size = 3
            archive.addfile(unsafe, io.BytesIO(b"bad"))
        buffer.seek(0)
        with tempfile.TemporaryDirectory() as temporary, tarfile.open(fileobj=buffer, mode="r|") as archive:
            rows, incomplete = download.extract_images(archive, Path(temporary), 1)
            self.assertEqual(len(rows), 1)
            self.assertFalse(incomplete)
            self.assertEqual(rows[0]["location_id"], "0000001")
            with np.load(Path(temporary) / rows[0]["image_path"]) as image:
                self.assertEqual(set(image.files), set(cache.SENTINEL2_BANDS))

    def test_prefix_keeps_every_season_before_stopping_at_next_location(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for location, season in (("0000001", "seasonA"), ("0000001", "seasonB"), ("0000002", "seasonC")):
                for band in cache.SENTINEL2_BANDS:
                    info = tarfile.TarInfo(f"s2c/{location}/{season}/{band}.tif")
                    info.size = 3
                    archive.addfile(info, io.BytesIO(b"tif"))
        buffer.seek(0)
        with tempfile.TemporaryDirectory() as temporary, tarfile.open(fileobj=buffer, mode="r|") as archive:
            with patch.object(archive, "extractfile", wraps=archive.extractfile) as extracted:
                rows, incomplete = download.extract_images(archive, Path(temporary), 1)
            self.assertEqual(extracted.call_count, 26)
            self.assertEqual({r["season_id"] for r in rows}, {"seasonA", "seasonB"})
            self.assertEqual({r["location_id"] for r in rows}, {"0000001"})
            self.assertFalse(incomplete)

    def test_noncontiguous_locations_and_incomplete_images_fail(self):
        cases = [(["0000001", "0000002", "0000001"], False, "Noncontiguous"),
                 (["0000001", "0000002"], True, "Incomplete")]
        for groups, omit_band, message in cases:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for index, location in enumerate(groups):
                    for band in cache.SENTINEL2_BANDS:
                        if omit_band and index == 0 and band == "B12":
                            continue
                        info = tarfile.TarInfo(f"s2c/{location}/season{index}/{band}.tif")
                        info.size = 3
                        archive.addfile(info, io.BytesIO(b"tif"))
            buffer.seek(0)
            with tempfile.TemporaryDirectory() as temporary, tarfile.open(fileobj=buffer, mode="r|") as archive:
                with self.assertRaisesRegex(ValueError, message):
                    download.extract_images(archive, Path(temporary), 3)


if __name__ == "__main__":
    unittest.main()
