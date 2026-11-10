"""Cache frozen SSL4EO-S12 MoCo spatial features and CLS embeddings.

Run with: uv run --extra encoder scripts/cache_encoder_features.py
No weights are downloaded. A current, fingerprinted segmentation audit is required.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tomllib
import zipfile

import numpy as np
import rasterio

if __package__:
    from .get_ssl4eo_encoder import (
        DEFAULT_OUTPUT as MODEL_DIR, FILENAME, REVISION, SHA256, SENTINEL2_BANDS,
        digest, encoder_dependencies, load_encoder, valid_checkpoint, write_json,
    )
else:
    from get_ssl4eo_encoder import (
        DEFAULT_OUTPUT as MODEL_DIR, FILENAME, REVISION, SHA256, SENTINEL2_BANDS,
        digest, encoder_dependencies, load_encoder, valid_checkpoint, write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("patch_id", "event_id", "original_split", "image_path", "label_path")
SCHEMA_VERSION = 1


class IncompatibleCacheError(ValueError):
    pass


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def preprocessing_config(settings: Path) -> dict:
    """Use the fixed all-band statistics from the repository's flood setup."""
    config = tomllib.loads(settings.read_text(encoding="utf-8-sig"))
    statistics = np.asarray(config["flood"]["train"]["normalization"], dtype=np.float32)
    if statistics.shape != (13, 2) or not np.isfinite(statistics).all() or (statistics[:, 1] <= 0).any():
        raise ValueError("Flood normalization must contain 13 finite mean/std pairs with positive SDs.")
    return {
        "name": "shrug_flood_full_patch_v1", "bands": list(SENTINEL2_BANDS),
        "input_units": "stored Sentinel-2 L1C values (reflectance scaled by 10000)",
        "normalization": "(raw - mean) / std in float32; fixed external statistics",
        "mean": statistics[:, 0].tolist(), "std": statistics[:, 1].tolist(),
        "normalization_source": "src/config/dataset.toml:flood.train.normalization",
        "invalid_pixels": "GDAL-invalid or nonfinite values filled with raw 0, independently per band, before normalization",
        "resize": {"size": [224, 224], "mode": "nearest-exact"},
        "crop": None, "clip": None, "divide_by_10000": False,
        "augmentation": "none; canonical full-patch image",
        "target_geometry": "native label grid retained; this script does not resize labels",
    }


def preprocess_image(raw: np.ndarray, band_valid: np.ndarray, config: dict):
    """Deterministic full-patch transform shared by caching and later inference."""
    torch, _ = encoder_dependencies()
    if raw.ndim != 3 or raw.shape[0] != 13 or band_valid.shape != raw.shape:
        raise ValueError("Expected a 13-band image and equally shaped per-band validity masks.")
    valid = band_valid.astype(bool) & np.isfinite(raw)
    if not valid.all(axis=0).any():
        raise ValueError("No jointly valid pixels across the image bands.")
    values = np.where(valid, raw, 0).astype(np.float32)
    mean = np.asarray(config["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(config["std"], dtype=np.float32)[:, None, None]
    tensor = torch.from_numpy(np.ascontiguousarray((values - mean) / std)).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=config["resize"]["size"],
                                            mode=config["resize"]["mode"])
    if tensor.shape != (1, 13, 224, 224) or not torch.isfinite(tensor).all().item():
        raise ValueError("Preprocessing did not produce finite [1,13,224,224] input.")
    return tensor.squeeze(0)


def tokens_to_arrays(tokens):
    """Match SSL4EO.forward/reshape: discard CLS, permute channels, reshape rows."""
    torch, _ = encoder_dependencies()
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (197, 384):
        raise ValueError(f"Expected [B,197,384] encoder tokens; got {tuple(tokens.shape)}.")
    if not torch.isfinite(tokens).all().item():
        raise ValueError("Encoder produced nonfinite tokens.")
    spatial = tokens[:, 1:, :].permute(0, 2, 1).reshape(-1, 384, 14, 14)
    cls = tokens[:, 0, :]
    return (spatial.detach().cpu().float().numpy().copy(),
            cls.detach().cpu().float().numpy().copy())


def read_rows(path: Path, fields) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {fields}")
        rows = list(reader)
    ids = [row["patch_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key or "") for key in ids):
        raise ValueError(f"Duplicate, empty, or unsafe patch IDs in {path}.")
    if not rows:
        raise ValueError(f"Empty input table: {path}")
    return rows


def relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    result = (root / path).resolve()
    if path.is_absolute() or not result.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path must be relative and inside its root: {value}")
    return result


def selected_rows(manifest: Path, inspection: Path):
    rows = read_rows(manifest, FIELDS)
    audited = read_rows(inspection, (*FIELDS, "usable", "image_files_sha256", "label_files_sha256"))
    summary_path = inspection.parent / "inspection_summary.json"
    if not summary_path.is_file():
        raise ValueError("Inspection summary is missing; rerun inspect_segmentation_data.py.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("manifest_sha256") != digest(manifest) or summary.get("inspection_schema_version", 0) < 2:
        raise ValueError("Inspection is stale or lacks source hashes; rerun inspect_segmentation_data.py.")
    lookup = {row["patch_id"]: row for row in audited}
    if set(lookup) != {row["patch_id"] for row in rows}:
        raise ValueError("Manifest and inspection patch IDs differ; rerun inspection.")
    usable, skipped = [], []
    for row in rows:
        audit = lookup[row["patch_id"]]
        if any(audit[field] != row[field] for field in FIELDS):
            raise ValueError(f"Inspection identity mismatch: {row['patch_id']}")
        if audit["usable"] not in ("True", "False"):
            raise ValueError(f"Invalid usability value: {row['patch_id']}")
        if audit["usable"] == "True":
            usable.append((row, audit))
        else:
            skipped.append({**row, "inspection_errors": audit.get("errors", "")})
    return usable, skipped, summary


def file_hashes(dataset) -> dict:
    parent = Path(dataset.name).resolve().parent
    return {Path(name).resolve().relative_to(parent).as_posix(): digest(Path(name))
            for name in dataset.files}


def source_identity(row, audit, root):
    """Reject changed imagery, sidecars or labels until the audit is refreshed."""
    for kind, field in (("image", "image_path"), ("label", "label_path")):
        path = relative_path(root, row[field])
        if not path.is_file():
            raise ValueError(f"Missing {kind} for {row['patch_id']}: {path}")
        with rasterio.open(path) as source:
            current = file_hashes(source)
            if current != json.loads(audit[f"{kind}_files_sha256"]):
                raise ValueError(f"{kind} changed since inspection: {row['patch_id']}; rerun inspection.")
            if kind == "image":
                if source.count != 13 or tuple(source.descriptions) != SENTINEL2_BANDS:
                    raise ValueError(f"Unverified 13-band order for {row['patch_id']}.")
                identity = {key: row[key] for key in ("patch_id", "event_id", "original_split", "image_path")}
                identity.update(image_sha256=current[path.name], image_files_sha256=current,
                                native_shape=list(source.shape), crs=str(source.crs),
                                transform=list(source.transform)[:6], bounds=list(source.bounds))
            else:
                label_sha256 = current[path.name]
    return identity, label_sha256


def cache_contract(preprocessing, device):
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "timm", "numpy", "rasterio")}
    return {
        "schema_version": SCHEMA_VERSION,
        "encoder": {"architecture": "vit_small_patch16_224", "revision": REVISION, "sha256": SHA256},
        "preprocessing": preprocessing, "versions": versions, "device": device,
        "precision": "float32; inference_mode; highest matmul precision; no autocast or TF32",
        "spatial_shape": [384, 14, 14], "cls_shape": [384], "dtype": "float32",
        "spatial_order": "tokens[:,1:,:].permute(0,2,1).reshape(B,384,14,14)",
        "implementation_sha256": digest(Path(__file__)),
        "loader_sha256": digest(Path(__file__).with_name("get_ssl4eo_encoder.py")),
    }


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_cache(path: Path, identity: dict, cache_fingerprint: str):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"spatial_features", "cls_embedding", "metadata"}:
                raise ValueError("unexpected archive fields")
            metadata = json.loads(archive["metadata"].item())
            if not isinstance(metadata, dict) or not isinstance(metadata.get("payload_sha256"), dict):
                raise ValueError("invalid metadata object")
            if (metadata.get("schema_version") != SCHEMA_VERSION
                    or metadata.get("cache_fingerprint") != cache_fingerprint
                    or fingerprint(metadata.get("contract")) != cache_fingerprint
                    or metadata.get("identity") != identity):
                raise ValueError("encoder/preprocessing/runtime/source identity mismatch")
            spatial, cls = archive["spatial_features"], archive["cls_embedding"]
            for name, array, shape in (("spatial_features", spatial, (384, 14, 14)),
                                       ("cls_embedding", cls, (384,))):
                if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"invalid {name} shape, dtype or values")
                if metadata.get("payload_sha256", {}).get(name) != array_hash(array):
                    raise ValueError(f"corrupt {name} payload")
            return spatial, cls
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise IncompatibleCacheError(f"Incompatible/corrupt cache {path.name}: {error}") from error


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_cache(path, spatial, cls, identity, cache_fingerprint, contract):
    metadata = {"schema_version": SCHEMA_VERSION, "cache_fingerprint": cache_fingerprint,
                "contract": contract,
                "identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "payload_sha256": {"spatial_features": array_hash(spatial), "cls_embedding": array_hash(cls)}}
    save_npz(path, spatial_features=spatial, cls_embedding=cls,
             metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def write_index(path, rows):
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_cache(manifest, inspection, checkpoint, output, settings, *, batch_size=4,
              threads=4, device="cpu", limit=None, dry_run=False, rebuild_incompatible=False):
    if batch_size < 1 or threads < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size, threads, and any limit must be positive.")
    manifest, inspection, checkpoint, output, settings = map(Path, (manifest, inspection, checkpoint, output, settings))
    manifest, inspection, checkpoint, output = (path.resolve() for path in (manifest, inspection, checkpoint, output))
    if not valid_checkpoint(checkpoint):
        raise ValueError("Verified pretrained checkpoint not found; run scripts/get_ssl4eo_encoder.py first.")
    torch, _ = encoder_dependencies()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    preprocessing = preprocessing_config(settings)
    contract = cache_contract(preprocessing, device)
    cache_fingerprint = fingerprint(contract)
    config_path = output / "cache_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != cache_fingerprint or existing.get("contract") != contract:
            raise IncompatibleCacheError("Cache configuration differs. Choose a new --output directory; existing files are untouched.")
    selected, skipped, audit_summary = selected_rows(manifest, inspection)
    available = len(selected)
    selected = selected[:limit] if limit else selected
    if not selected:
        raise ValueError("No usable audited patches to cache.")
    plans, incompatible = [], []
    for row, audit in selected:
        identity, label_sha256 = source_identity(row, audit, manifest.parent)
        path = output / "patches" / f"{row['patch_id']}.npz"
        cls, status = None, "new"
        if path.exists():
            try:
                _, cls = read_cache(path, identity, cache_fingerprint)
                status = "reused"
            except IncompatibleCacheError as error:
                incompatible.append(str(error))
                status = "rebuild"
        plans.append(dict(row=row, identity=identity, path=path, label_sha256=label_sha256,
                          cls=cls, status=status))
    counts = {status: sum(plan["status"] == status for plan in plans) for status in ("new", "reused", "rebuild")}
    print(f"Selected {len(plans)}/{available} usable patches; skipped {len(skipped)} unusable audit entries.")
    print(f"Cache fingerprint: {cache_fingerprint}\nPlan: {counts}", flush=True)
    if incompatible:
        print("\n".join(incompatible), flush=True)
        if not rebuild_incompatible:
            raise IncompatibleCacheError("Use --rebuild-incompatible to replace these derived files, or choose a new --output.")
    if dry_run:
        print("DRY RUN: source/cache checks complete; no encoder execution or files written.")
        return counts
    output.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        write_json(config_path, {"fingerprint": cache_fingerprint, "contract": contract})
    pending = [plan for plan in plans if plan["status"] != "reused"]
    if pending:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("highest")
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model = load_encoder(checkpoint, device=device)
        model.eval().requires_grad_(False)
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                inputs = []
                for plan in batch:
                    with rasterio.open(relative_path(manifest.parent, plan["row"]["image_path"])) as source:
                        if file_hashes(source) != plan["identity"]["image_files_sha256"]:
                            raise ValueError(f"Image changed during caching: {plan['row']['patch_id']}")
                        inputs.append(preprocess_image(source.read(), source.read_masks() > 0, preprocessing))
                spatial, cls = tokens_to_arrays(model.forward_features(torch.stack(inputs).to(device)))
                for index, plan in enumerate(batch):
                    write_cache(plan["path"], spatial[index], cls[index], plan["identity"], cache_fingerprint, contract)
                    plan["cls"] = cls[index]
                print(f"Encoded {min(start + len(batch), len(pending))}/{len(pending)} pending patches", flush=True)
    suffix = f".limit-{limit}" if limit is not None else ""
    index_rows = [{**plan["row"], "feature_path": plan["path"].relative_to(output).as_posix(),
                   "image_sha256": plan["identity"]["image_sha256"], "label_sha256": plan["label_sha256"],
                   "cache_fingerprint": cache_fingerprint, "cache_file_sha256": digest(plan["path"]),
                   "encoder_sha256": SHA256} for plan in plans]
    write_index(output / f"feature_manifest{suffix}.csv", index_rows)
    save_npz(output / f"cls_embeddings{suffix}.npz",
             patch_ids=np.asarray([plan["row"]["patch_id"] for plan in plans]),
             event_ids=np.asarray([plan["row"]["event_id"] for plan in plans]),
             original_splits=np.asarray([plan["row"]["original_split"] for plan in plans]),
             cls_embeddings=np.stack([plan["cls"] for plan in plans]),
             metadata=np.asarray(json.dumps({"cache_fingerprint": cache_fingerprint, "contract": contract}, sort_keys=True)))
    summary = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "selected_patches": len(plans), "available_usable_patches": available, "counts": counts,
               "complete_usable_manifest": len(plans) == available, "limit": limit, "skipped_patches": skipped,
               "cache_fingerprint": cache_fingerprint, "manifest_sha256": digest(manifest),
               "inspection_sha256": digest(inspection), "inspection_reference_support": audit_summary["reference_support"],
               "normalization_config_sha256": digest(settings), "batch_size": batch_size, "threads": threads,
               "index": f"feature_manifest{suffix}.csv", "cls_archive": f"cls_embeddings{suffix}.npz"}
    write_json(output / f"cache_summary{suffix}.json", summary)
    print(f"Cache complete: {output / summary['index']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/sen1floods11/manifest.csv")
    parser.add_argument("--inspection", type=Path, default=ROOT / "results/segmentation_data/patches.csv")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / FILENAME)
    parser.add_argument("--output", type=Path, default=ROOT / "data/sen1floods11/encoder_features/ssl4eo-s12-moco")
    parser.add_argument("--settings", type=Path, default=ROOT / "src/config/dataset.toml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-incompatible", action="store_true")
    args = parser.parse_args()
    try:
        run_cache(args.manifest, args.inspection, args.checkpoint, args.output, args.settings,
                  batch_size=args.batch_size, threads=args.threads, device=args.device, limit=args.limit,
                  dry_run=args.dry_run, rebuild_incompatible=args.rebuild_incompatible)
    except (OSError, ValueError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
