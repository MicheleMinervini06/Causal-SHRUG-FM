"""Retain the first 5,000 location blocks from the original SSL4EO-S12 archive.

Stop when the next location begins; keep all seasons in each selected block.
This is an archive-prefix sample, not a random global pretraining sample.
--debug keeps just the first complete image and cannot become a fitted reference.
"""

import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request

import numpy as np

if __package__:
    from . import cache_encoder_features as cache
else:
    import cache_encoder_features as cache

ROOT = cache.ROOT
REPOSITORY = "wangyi111/SSL4EO-S12"
REVISION = "3f5ddad68ba2ea29d019b0cef6cf292ff8af0d62"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
DEFAULT_OUTPUT = ROOT / "data/ssl4eo-s12/reference-prefix-n5000"
MEMBER = re.compile(r"s2c/(\d{7})/([A-Za-z0-9_]+)/((?:B[0-9]+|B8A)\.tif)")


def source_parts():
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/s2_l1c"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    parts = [{"path": e["path"], "size": e["size"], "sha256": e["lfs"]["oid"]}
             for e in entries if re.fullmatch(r"s2_l1c/s2_l1c.tar.gz.part[a-z]{2}", e["path"])]
    parts.sort(key=lambda p: p["path"])
    if len(parts) != 12 or any(p["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) for p in parts):
        raise ValueError("Unexpected pinned archive layout or missing LFS checksums.")
    return parts


class ArchiveStream(io.RawIOBase):
    """Concatenate HTTP parts; retry at the exact byte offset and hash each part."""

    def __init__(self, parts):
        self.parts, self.index, self.offset = parts, 0, 0
        self.response = None
        self.sha = hashlib.sha256()
        self.prefix_sha = hashlib.sha256()
        self.total = 0
        self.last_report = time.monotonic()

    def readable(self):
        return True

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Archive must be read in bounded chunks.")
        if not size or self.index == len(self.parts):
            return b""
        part = self.parts[self.index]
        size = min(size, part["size"] - self.offset)
        for attempt in range(4):
            try:
                if self.response is None:
                    # Offset query avoids a cached redirect signed for another range.
                    url = f"{BASE_URL}/{part['path']}?download=true&offset={self.offset}"
                    request = urllib.request.Request(url, headers={"Range": f"bytes={self.offset}-"})
                    self.response = urllib.request.urlopen(request, timeout=120)
                    content_range = self.response.headers.get("Content-Range", "")
                    expected = f"bytes {self.offset}-{part['size'] - 1}/{part['size']}"
                    if self.response.status != 206 or content_range != expected:
                        raise ValueError(f"Server did not honor archive range: {content_range}")
                data = self.response.read(size)
                if not data:
                    raise OSError("Premature HTTP EOF")
                break
            except (OSError, http.client.HTTPException):
                if self.response is not None:
                    self.response.close()
                self.response = None
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        self.sha.update(data)
        self.prefix_sha.update(data)
        self.offset += len(data)
        self.total += len(data)
        if time.monotonic() - self.last_report >= 30:
            print(f"Transferred {self.total / 1e9:.2f} GB; part {self.index + 1}/12", flush=True)
            self.last_report = time.monotonic()
        if self.offset == part["size"]:
            if self.sha.hexdigest() != part["sha256"]:
                raise ValueError(f"Archive SHA-256 mismatch: {part['path']}")
            print(f"Verified archive part {self.index + 1}/12", flush=True)
            self.response.close()
            self.response = None
            self.index += 1
            self.offset, self.sha = 0, hashlib.sha256()
        return data


def extract_images(archive, output, locations, *, debug=False):
    """Stop before reading bands from location N+1; require contiguous blocks."""
    pending, rows, seen, selected = {}, [], set(), set()
    current = None
    bands = set(cache.SENTINEL2_BANDS)
    for member in archive:
        archive.members.clear()  # Do not retain millions of TarInfo objects in a streaming scan.
        match = MEMBER.fullmatch(member.name.removeprefix("./"))
        if not match or not member.isfile():
            continue
        location, season, filename = match.groups()
        band = filename[:-4]
        if band not in bands:
            continue
        if location != current:
            if location in selected:
                raise ValueError("Noncontiguous location blocks; cannot safely retain all seasons from a prefix.")
            if pending:
                raise ValueError(f"Incomplete images in location {current}: {sorted(pending)}")
            if len(selected) == locations:
                return rows, {}
            selected.add(location)
            current = location
        key = f"{location}_{season}"
        if (key, band) in seen or member.size > 5_000_000:
            raise ValueError(f"Duplicate or oversized band: {member.name}")
        seen.add((key, band))
        with archive.extractfile(member) as stream:
            payload = stream.read()
        if len(payload) != member.size:
            raise ValueError(f"Truncated band: {member.name}")
        pending.setdefault(key, {})[band] = np.frombuffer(payload, dtype=np.uint8)
        if len(pending) > 128:
            raise ValueError("Too many incomplete images; unexpected archive ordering.")
        if set(pending[key]) == bands:
            path = output / "images" / f"{key}.npz"
            cache.save_npz(path, **pending.pop(key))
            rows.append({"image_id": key, "location_id": location, "season_id": season,
                         "image_path": path.relative_to(output).as_posix(), "sha256": cache.digest(path)})
            if len(rows) % 100 == 0:
                print(f"Retained {len(rows)} complete reference images", flush=True)
            if debug:
                return rows, {}
    return rows, {key: sorted(bands - set(values)) for key, values in pending.items()}


def download(output, locations=5000, *, debug=False, dry_run=False):
    output = Path(output)
    if locations < 1:
        raise ValueError("Locations must be positive.")
    summary_path = output / "download_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (summary["locations_requested"], summary.get("selection"), summary["debug"]) != (locations, "archive_prefix", debug):
            raise ValueError("Existing reference uses different sampling settings; choose another output.")
        if cache.digest(output / "reference_manifest.csv") != summary["manifest_sha256"]:
            raise ValueError("Reference manifest changed.")
        if cache.digest(output / "download_config.json") != summary["config_sha256"]:
            raise ValueError("Reference download configuration changed.")
        for row in cache.read_rows(output / "reference_manifest.csv", ("patch_id", "image_path", "sha256")):
            if cache.digest(cache.relative_path(output, row["image_path"])) != row["sha256"]:
                raise ValueError(f"Changed reference image: {row['patch_id']}")
        print(f"Reused verified reference: {output}", flush=True)
        return summary
    parts = source_parts()
    total = sum(p["size"] for p in parts)
    print(f"Retain the first {locations} location blocks/all seasons, then stop. Debug={debug}. "
          f"Full archive size {total / 1e9:.2f} GB; only the prefix is transferred.", flush=True)
    if dry_run:
        return {"source_parts": parts, "full_archive_bytes": total, "selection": "archive_prefix"}
    output.mkdir(parents=True, exist_ok=True)
    plan = {"repository": REPOSITORY, "revision": REVISION, "source_parts": parts,
            "locations_requested": locations, "debug": debug, "selection": "archive_prefix",
            "selection_rule": "first complete image (debug)" if debug else "first N contiguous location blocks in archive order; all seasons in each block",
            "limitation": "not a random global sample; geographic/embedding representativeness unverified"}
    plan_path = output / "download_config.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Incompatible interrupted download; choose another output directory.")
    cache.write_json(plan_path, plan)
    print("Streaming archive; interrupted runs rescan from the beginning.", flush=True)
    with closing(ArchiveStream(parts)) as source, gzip.GzipFile(fileobj=source) as uncompressed:
        with tarfile.open(fileobj=uncompressed, mode="r|", bufsize=1024 * 1024) as archive:
            rows, incomplete = extract_images(archive, output, locations, debug=debug)
        transferred = source.total
        prefix_sha256, verified_parts = source.prefix_sha.hexdigest(), source.index
    if not rows or incomplete:
        raise ValueError(f"No images or incomplete selected images: {incomplete}")
    selected_ids = list(dict.fromkeys(r["location_id"] for r in rows))
    if not debug and len(selected_ids) != locations:
        raise ValueError(f"Expected {locations} location blocks, found {len(selected_ids)}.")
    # Keep patch_id for reuse of the project's checked CSV reader.
    rows = [{"patch_id": r.pop("image_id"), **r} for r in sorted(rows, key=lambda r: r["image_id"])]
    cache.write_index(output / "reference_manifest.csv", rows)
    present = {r["location_id"] for r in rows}
    summary = {"locations_requested": locations, "selection": "archive_prefix", "debug": debug,
               "reference_complete": not debug, "n_images": len(rows), "n_locations": len(present),
               "selected_location_ids_in_archive_order": selected_ids,
               "season_counts": dict(Counter(r["location_id"] for r in rows)),
               "fully_verified_archive_parts": verified_parts, "downloaded_prefix_sha256": prefix_sha256,
               "integrity_scope": "pinned HTTPS source; full parts checked against LFS SHA-256; partial prefix hash recorded locally; gzip end CRC not checked after early stop",
               "transferred_bytes": transferred, "manifest_sha256": cache.digest(output / "reference_manifest.csv"),
               "config_sha256": cache.digest(plan_path), "implementation_sha256": cache.digest(Path(__file__))}
    cache.write_json(summary_path, summary)
    print(f"Reference download complete: {len(rows)} images, {len(present)} locations; {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locations", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.output, args.locations, debug=args.debug, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
