"""Shared native-grid data, decoder and metric contracts for downstream scripts."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

if __package__:
    from . import cache_encoder_features as cache
else:
    import cache_encoder_features as cache

ROOT = Path(__file__).resolve().parents[1]
# The original project is not packaged by uv. Import its unchanged components,
# without importing the legacy training entrypoint or its unrelated dependencies.
for source in (ROOT, ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.inspect_segmentation_data import grid_displacement, valid_support
from signal_task.models.model_components.decoder import CoreDecoder

DEFAULT_MANIFEST = ROOT / "data/sen1floods11/manifest.csv"
DEFAULT_INSPECTION = ROOT / "results/segmentation_data/patches.csv"
DEFAULT_CACHE = ROOT / "data/sen1floods11/encoder_features/ssl4eo-s12-moco"
SPLITS = ("train", "validation", "test", "bolivia_holdout")
IGNORE_INDEX = -1
ARCHITECTURE = {
    "embedding_dim": 384, "output_dim": 2, "depths": [2, 2, 8, 2],
    "dims": [160, 320, 640, 1280], "activation": "relu", "norm": "batch",
    "padding": "same", "dropout": 0.0,
}


def decoder_sources():
    directory = ROOT / "src/signal_task/models/model_components"
    # Git can translate LF/CRLF between Anunna and Windows checkouts. Hash the
    # source text with canonical LF newlines, so this alone cannot reject a head.
    return {name: hashlib.sha256((directory / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("decoder.py", "training_utils.py")}


def build_decoder():
    return CoreDecoder(**ARCHITECTURE)


def configure_runtime(seed, device, threads):
    if not 0 <= seed < 2**32 or threads < 1:
        raise ValueError("Seed must be in [0, 2**32); threads must be positive.")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable; CUDA needs a CUDA-enabled PyTorch installation.")
    # Required by deterministic CUDA matmul; set before CUDA work starts.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def runtime_versions():
    return {name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "rasterio", "pyyaml")}


class FeatureCatalog:
    """Bind the complete feature index to the current manifest and hashed audit.

    Only requested split samples are opened/prepared. Training never loads test
    or Bolivia features/labels. Source hashes include GDAL validity sidecars.
    """

    def __init__(self, manifest=DEFAULT_MANIFEST, inspection=DEFAULT_INSPECTION,
                 feature_dir=DEFAULT_CACHE):
        self.manifest, self.inspection, self.feature_dir = (
            Path(path).resolve() for path in (manifest, inspection, feature_dir))
        usable, self.skipped, audit = cache.selected_rows(self.manifest, self.inspection)
        self.support = audit["reference_support"]
        if self.support not in ("joint", "label"):
            raise ValueError("Unknown inspection support.")
        config = json.loads((self.feature_dir / "cache_config.json").read_text(encoding="utf-8"))
        self.contract, self.fingerprint = config["contract"], config["fingerprint"]
        if (cache.fingerprint(self.contract) != self.fingerprint
                or self.contract.get("encoder", {}).get("sha256") != cache.SHA256
                or self.contract.get("spatial_shape") != [384, 14, 14]):
            raise ValueError("Invalid/incompatible encoder cache contract.")
        summary = json.loads((self.feature_dir / "cache_summary.json").read_text(encoding="utf-8"))
        if (summary.get("cache_fingerprint") != self.fingerprint
                or not summary.get("complete_usable_manifest")
                or summary.get("manifest_sha256") != cache.digest(self.manifest)
                or summary.get("inspection_sha256") != cache.digest(self.inspection)
                or summary.get("inspection_reference_support") != self.support):
            raise ValueError("Incomplete/stale feature index; refresh the audit/cache first.")
        index = self.feature_dir / "feature_manifest.csv"
        fields = (*cache.FIELDS, "feature_path", "cache_fingerprint", "cache_file_sha256",
                  "image_sha256", "label_sha256", "encoder_sha256")
        rows = cache.read_rows(index, fields)
        lookup = {row["patch_id"]: (row, qc) for row, qc in usable}
        if set(lookup) != {row["patch_id"] for row in rows}:
            raise ValueError("Feature index must cover exactly the complete usable audited cohort.")
        paths = set()
        self.records = []
        for row in rows:
            original, qc = lookup[row["patch_id"]]
            if (any(row[key] != original[key] for key in cache.FIELDS)
                    or row["original_split"] not in SPLITS
                    or row["cache_fingerprint"] != self.fingerprint
                    or row["encoder_sha256"] != cache.SHA256
                    or any(row[f"{kind}_sha256"] != qc[f"{kind}_sha256"] for kind in ("image", "label"))):
                raise ValueError(f"Feature/audit identity mismatch: {row['patch_id']}")
            path = cache.relative_path(self.feature_dir, row["feature_path"])
            if path in paths:
                raise ValueError("Multiple patches reference one feature file.")
            paths.add(path)
            self.records.append((row, qc))
        self.provenance = {
            "cache_fingerprint": self.fingerprint, "cache_contract": self.contract,
            "encoder_sha256": cache.SHA256, "reference_support": self.support,
            "manifest_sha256": cache.digest(self.manifest),
            "inspection_sha256": cache.digest(self.inspection),
            "feature_manifest_sha256": cache.digest(index),
        }

    def select(self, splits, limit_per_split=None):
        if not splits or len(set(splits)) != len(splits) or not set(splits).issubset(SPLITS):
            raise ValueError("Select at least one known split.")
        if limit_per_split is not None and limit_per_split < 1:
            raise ValueError("Per-split limit must be positive.")
        selected = []
        for split in splits:
            subset = [(row, qc) for row, qc in self.records if row["original_split"] == split]
            if not subset:
                raise ValueError(f"No usable cached patches for {split}.")
            selected.extend(subset[:limit_per_split] if limit_per_split is not None else subset)
        return selected

    def prepare(self, record):
        row, qc = record
        root = self.manifest.parent
        identity, label_hash = cache.source_identity(row, qc, root)
        path = cache.relative_path(self.feature_dir, row["feature_path"])
        if label_hash != row["label_sha256"] or cache.digest(path) != row["cache_file_sha256"]:
            raise ValueError(f"Label/feature changed since caching: {row['patch_id']}")
        spatial, _ = cache.read_cache(path, identity, self.fingerprint)
        with rasterio.open(cache.relative_path(root, row["image_path"])) as image, \
                rasterio.open(cache.relative_path(root, row["label_path"])) as label:
            if (image.shape != label.shape or image.crs != label.crs
                    or label.count != 1 or label.nodata in (0, 1)
                    or grid_displacement(image, label) > 1e-4):
                raise ValueError(f"Image/label grid or NoData mismatch: {row['patch_id']}")
            values = label.read(1)
            _, label_valid, joint = valid_support(image.read(), image.read_masks(), values, label.read_masks(1))
            valid = joint if self.support == "joint" else label_valid
            n_valid = int(valid.sum())
            fraction = float(np.count_nonzero((values == 1) & valid) / n_valid) if n_valid else None
            if (not n_valid or n_valid != int(qc["reference_valid_pixels"])
                    or not np.isclose(fraction, float(qc["reference_water_fraction"]), atol=1e-12, rtol=0)):
                raise ValueError(f"Reference support differs from inspection: {row['patch_id']}")
            # Preserve native labels/support. int8 storage keeps preloaded CPU data small.
            target = np.full(values.shape, IGNORE_INDEX, dtype=np.int8)
            target[valid] = values[valid].astype(np.int8)
            geometry = {"height": label.height, "width": label.width,
                        "crs": label.crs, "transform": label.transform}
        return {"row": row, "features": torch.from_numpy(spatial),
                "target": torch.from_numpy(target), "geometry": geometry}


class PreparedDataset(torch.utils.data.Dataset):
    """Read and verify selected sources once, retaining features and int8 targets."""

    def __init__(self, catalog, records):
        self.samples = []
        for index, record in enumerate(records):
            self.samples.append(catalog.prepare(record))
            if (index + 1) % 50 == 0:
                print(f"Verified/prepared {index + 1}/{len(records)} patches", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_samples(samples):
    # Labels can have different native dimensions; never resize/pad them to batch.
    return torch.stack([sample["features"] for sample in samples]), samples


def native_logits(logits, shape):
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Decoder must return [B,2,H,W] logits.")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite decoder logits.")
    return F.interpolate(logits, size=shape, mode="bilinear", align_corners=False)


def water_counts(predicted, target):
    valid = target != IGNORE_INDEX
    water = target == 1
    return {"tp": int((predicted & water & valid).sum().item()),
            "fp": int((predicted & ~water & valid).sum().item()),
            "fn": int((~predicted & water & valid).sum().item()),
            "n_valid_pixels": int(valid.sum().item())}


def water_metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    # Undefined empty-water scores are null/blank, rather than rewarding dry patches.
    return {"F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else None}


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals = dict(tp=0, fp=0, fn=0, n_valid_pixels=0)
    loss_sum, patches, batches = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_index, (features, samples) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(features.to(device))
            batch_loss, batch_valid = None, 0
            for index, sample in enumerate(samples):
                target = sample["target"].to(device=device, dtype=torch.long)
                logits = native_logits(outputs[index:index + 1], target.shape)
                loss = F.cross_entropy(logits, target.unsqueeze(0), ignore_index=IGNORE_INDEX, reduction="sum")
                predicted = logits.detach().softmax(dim=1)[0, 1] >= 0.5
                counts = water_counts(predicted, target)
                batch_valid += counts["n_valid_pixels"]
                batch_loss = loss if batch_loss is None else batch_loss + loss
                for key in totals:
                    totals[key] += counts[key]
            if not batch_valid or not torch.isfinite(batch_loss).item():
                raise ValueError("Batch has no valid pixels or has nonfinite loss.")
            if training:
                (batch_loss / batch_valid).backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise ValueError("Nonfinite decoder gradients.")
                optimizer.step()
            loss_sum += float(batch_loss.detach().item())
            patches += len(samples)
            batches += 1
    if not batches or not totals["n_valid_pixels"]:
        raise ValueError("Epoch contains no valid batches.")
    return {"loss": loss_sum / totals["n_valid_pixels"], **water_metrics(totals),
            **totals, "patches": patches, "batches": batches}


def write_csv(path, rows, fields=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_head(path, catalog, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("decoder_sources") != decoder_sources()):
        raise ValueError("Incompatible head architecture/source version.")
    if checkpoint.get("data_provenance") != catalog.provenance:
        raise ValueError("Head does not match this encoder/preprocessing/feature index/reference support.")
    model = build_decoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model parameters/buffers.")
    return model.to(device).eval().requires_grad_(False), checkpoint
