#!/usr/bin/env python3
"""Download the complete Sen1Floods11 v1.1 hand-labelled Sentinel-2 subset.

Run from the project: uv run scripts/download_sen1floods11.py
Uses the Python standard library; no cloud credentials are required.
Downloads S2Hand and LabelHand, not the Sentinel-1 or weakly labelled subsets.
Existing TIFFs are reused only when their size and published MD5 match.
Use --dry-run to inspect the inventory and splits without downloading rasters
or writing files. --limit N selects the first N patch IDs in sorted order.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

BUCKET = "sen1floods11"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
HAND = "v1.1/data/flood_events/HandLabeled"
SPLITS = "v1.1/splits/flood_handlabeled/"
EVENT_METADATA = (
    "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/"
    "master/Sen1Floods11_Metadata.geojson"
)
SPLIT_NAMES = {
    "flood_train_data.csv": "train",
    "flood_valid_data.csv": "validation",
    "flood_test_data.csv": "test",
    "flood_bolivia_data.csv": "bolivia_holdout",
}


def open_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SHRUG-FM-Sen1Floods11-downloader/1.0"})
    return urllib.request.urlopen(request, timeout=60)


def read_url(url):
    """Read small inventories/metadata with retries; rasters are streamed separately."""
    for attempt in range(3):
        try:
            with open_url(url) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def list_objects(prefix):
    objects, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": 1000,
                 "fields": "items(name,size,md5Hash,generation),nextPageToken"}
        if token:
            query["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
        page = json.loads(read_url(url + urllib.parse.urlencode(query)))
        objects.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return objects


def object_url(item):
    path = urllib.parse.quote(item["name"], safe="/")
    query = urllib.parse.urlencode({"generation": item["generation"]})
    return f"https://storage.googleapis.com/{BUCKET}/{path}?{query}"


def matches(path, item):
    if not path.is_file() or path.stat().st_size != int(item["size"]):
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode() == item["md5Hash"]


def fetch_object(item, path):
    if "md5Hash" not in item:
        raise ValueError(f"Published MD5 is missing for {item['name']}")
    if matches(path, item):
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with open_url(object_url(item)) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if not matches(partial, item):
                raise ValueError(f"Size or MD5 mismatch: {item['name']}")
            partial.replace(path)
            return "downloaded"
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def patch_id(filename):
    name = Path(filename).name
    for suffix in ("_S1Hand.tif", "_S2Hand.tif", "_LabelHand.tif"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError(f"Unrecognized patch filename: {filename}")


def index_patches(objects, suffix):
    """Reject duplicate patch IDs instead of silently replacing inventory entries."""
    indexed = {}
    for item in objects:
        if not item["name"].endswith(suffix):
            continue
        key = patch_id(item["name"])
        if key in indexed:
            raise ValueError(f"Duplicate patch in public inventory: {key}")
        for field in ("generation", "size", "md5Hash"):
            if field not in item:
                raise ValueError(f"Missing {field} for {item['name']}")
        indexed[key] = item
    return indexed


def parse_split_map(contents):
    """Read official S1/label filenames, matching them to the same S2 patch IDs."""
    assignments = {}
    for filename, split in SPLIT_NAMES.items():
        with io.StringIO(contents[filename].lstrip("\ufeff"), newline="") as stream:
            for row in csv.reader(stream):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) != 2:
                    raise ValueError(f"Expected image and label columns in {filename}: {row}")
                image, label = (patch_id(cell.strip()) for cell in row)
                if image != label:
                    raise ValueError(f"Mismatched split pair in {filename}: {row}")
                if image in assignments:
                    raise ValueError(f"Duplicate/overlapping split assignment: {image}")
                assignments[image] = split
    return assignments


def load_splits(split_items, directory):
    """Validate small split files in memory, without writing during a dry run."""
    payloads = {}
    for filename in SPLIT_NAMES:
        if filename not in split_items:
            raise ValueError(f"Required official split is missing: {filename}")
        item = split_items[filename]
        if "md5Hash" not in item:
            raise ValueError(f"Published MD5 is missing for {item['name']}")
        local = directory / filename
        if matches(local, item):
            payload = local.read_bytes()
        else:
            payload = read_url(object_url(item))
        digest = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()
        if len(payload) != int(item["size"]) or digest != item["md5Hash"]:
            raise ValueError(f"Size or MD5 mismatch: {item['name']}")
        payloads[filename] = payload
    assignments = parse_split_map({name: data.decode("utf-8-sig") for name, data in payloads.items()})
    return assignments, payloads


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination (default: this project's data/sen1floods11).")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4, metavar="1-16")
    parser.add_argument("--limit", type=int, help="Download only the first N pairs for a smoke test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read public inventories/small split files; write nothing and download no rasters.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    out = args.output.resolve()

    print("Listing public data objects...", flush=True)
    images = index_patches(list_objects(HAND + "/S2Hand/"), "_S2Hand.tif")
    labels = index_patches(list_objects(HAND + "/LabelHand/"), "_LabelHand.tif")
    if not images or images.keys() != labels.keys():
        raise ValueError("Image and label inventories are empty or do not match")
    split_items = {Path(o["name"]).name: o for o in list_objects(SPLITS)}
    assignments, split_payloads = load_splits(split_items, out / "splits")
    if set(images) != set(assignments):
        raise ValueError("Public image inventory does not match the official split inventory")

    selected = sorted(images)[:args.limit] if args.limit else sorted(images)
    jobs = []
    rows = []
    for key in selected:
        image_path = Path("images") / Path(images[key]["name"]).name
        label_path = Path("labels") / Path(labels[key]["name"]).name
        jobs.extend([(images[key], out / image_path), (labels[key], out / label_path)])
        rows.append({"patch_id": key, "event_id": key.rsplit("_", 1)[0],
                     "original_split": assignments[key], "image_path": image_path.as_posix(),
                     "label_path": label_path.as_posix()})
    size = sum(int(item["size"]) for item, _ in jobs)
    print(f"Destination: {out}", flush=True)
    print(f"Selected {len(selected)} pairs from {len(images)} available; {size / 1024**3:.2f} GiB before reusing existing files.", flush=True)
    selected_splits = {name: sum(assignments[key] == name for key in selected)
                       for name in SPLIT_NAMES.values()}
    print(f"Selected pairs by official split: {selected_splits}", flush=True)
    if args.dry_run:
        print("DRY RUN: inventories and split assignments validated; no rasters downloaded or files written.")
        return

    out.mkdir(parents=True, exist_ok=True)
    for filename, payload in split_payloads.items():
        path = out / "splits" / filename
        if not matches(path, split_items[filename]):
            write_atomic(path, payload)
    counts = {"downloaded": 0, "reused": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(fetch_object, item, path): path for item, path in jobs}
        for done, task in enumerate(as_completed(tasks), 1):
            try:
                counts[task.result()] += 1
            except Exception as error:
                failures.append(f"{tasks[task].name}: {error}")
            if done % 25 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} files checked; downloaded={counts['downloaded']}, reused={counts['reused']}, failed={len(failures)}", flush=True)
    if failures:
        raise RuntimeError("Some downloads failed; rerun the same command to retry.\n" + "\n".join(failures))

    metadata = read_url(EVENT_METADATA)
    if json.loads(metadata).get("type") != "FeatureCollection":
        raise ValueError("Unexpected event metadata format")
    metadata_path = out / "metadata" / "Sen1Floods11_Metadata.geojson"
    write_atomic(metadata_path, metadata)

    manifest = out / "manifest.csv"
    temporary = manifest.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    audit = {
        "dataset": "Sen1Floods11", "version": "v1.1", "bucket": BUCKET,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_pairs": len(images), "selected_pairs": len(selected),
        "subset": "HandLabeled/S2Hand + HandLabeled/LabelHand",
        "selection_limit": args.limit, "selected_split_counts": selected_splits,
        "complete_labelled_sentinel2_subset": len(selected) == len(images),
        "files": [{"local_path": path.relative_to(out).as_posix(),
                   "object": item["name"], "generation": item["generation"],
                   "size": int(item["size"]), "md5": item["md5Hash"]}
                  for item, path in jobs] + [
                      {"local_path": "splits/" + name, "object": split_items[name]["name"],
                       "generation": split_items[name]["generation"],
                       "size": int(split_items[name]["size"]), "md5": split_items[name]["md5Hash"]}
                      for name in SPLIT_NAMES],
        "event_metadata": {"url": EVENT_METADATA, "sha256": hashlib.sha256(metadata).hexdigest()},
    }
    temporary = out / "download_manifest.json.part"
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out / "download_manifest.json")
    print(f"DONE: {len(rows)} verified image-mask pairs.\nManifest: {manifest}", flush=True)
    print("Next: validate raster alignment, pixel values and valid support with Rasterio.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
