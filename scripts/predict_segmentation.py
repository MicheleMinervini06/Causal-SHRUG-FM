"""Export native-grid water probabilities, masks and matched-support fractions."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np
import rasterio
import torch

if __package__:
    from . import segmentation_common as common
else:
    import segmentation_common as common


FIELDS = ("patch_id", "event_id", "original_split", "model_seed", "reference_fraction",
          "predicted_fraction", "signed_fraction_error", "absolute_fraction_error",
          "n_valid_pixels", "F1", "IoU", "probability_path", "mask_path", "threshold")


def prediction_result(sample, probability, model_seed, threshold, probability_path, mask_path):
    """All fractions and water scores use exactly the target's native valid support."""
    target = sample["target"].cpu()
    probability = np.asarray(probability, dtype=np.float32)
    if (probability.shape != tuple(target.shape) or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Expected finite native-grid foreground probabilities in [0,1].")
    valid = target.numpy() != common.IGNORE_INDEX
    n_valid = int(valid.sum())
    if not n_valid:
        raise ValueError(f"No valid reference pixels: {sample['row']['patch_id']}")
    predicted = torch.from_numpy(probability >= threshold)
    counts = common.water_counts(predicted, target)
    reference = int(((target == 1) & (target != common.IGNORE_INDEX)).sum().item()) / n_valid
    fraction = int(np.count_nonzero(predicted.numpy() & valid)) / n_valid
    error = fraction - reference
    row = {key: sample["row"][key] for key in ("patch_id", "event_id", "original_split")}
    row.update(model_seed=model_seed, reference_fraction=reference, predicted_fraction=fraction,
               signed_fraction_error=error, absolute_fraction_error=abs(error),
               n_valid_pixels=n_valid, **common.water_metrics(counts),
               probability_path=probability_path, mask_path=mask_path, threshold=threshold)
    # Export evaluation support explicitly: NaN probabilities / -1 labels elsewhere.
    exported_probability = np.where(valid, probability, np.nan).astype(np.float32)
    mask = np.where(valid, predicted.numpy(), common.IGNORE_INDEX).astype(np.int16)
    return row, exported_probability, mask, counts


def write_raster(path, values, geometry, nodata, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with rasterio.open(temporary, "w", driver="GTiff", count=1, dtype=str(values.dtype),
                           nodata=nodata, compress="deflate", **geometry) as destination:
            destination.write(values, 1)
            destination.update_tags(**{key: str(value) for key, value in tags.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_prediction(args):
    if (not math.isfinite(args.threshold) or not 0 < args.threshold < 1
            or args.batch_size < 1 or (args.limit is not None and args.limit < 1)):
        raise ValueError("Threshold must be in (0,1); batch size and any limit must be positive.")
    catalog = common.FeatureCatalog(args.manifest, args.inspection, args.feature_dir)
    # Load/check on CPU before configuring the requested device. Seed comes from the head.
    model, checkpoint = common.load_head(args.checkpoint, catalog)
    seed = checkpoint["model_seed"]
    common.configure_runtime(seed, args.device, args.threads)
    model.to(args.device)
    if checkpoint["config"]["debug"] and not args.allow_debug_head:
        raise ValueError("Debug checkpoint: pass --allow-debug-head only for implementation checks.")
    records = catalog.select(args.splits)
    if args.limit is not None:
        records = records[:args.limit]
    output = (args.output or common.ROOT / "results/segmentation_predictions" /
              f"seed-{seed}{'-debug' if checkpoint['config']['debug'] else ''}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; select a new --output directory: {output}")
    checkpoint_hash = common.cache.digest(args.checkpoint)
    provenance = {"head_sha256": checkpoint_hash, "model_seed": seed, "head_epoch": checkpoint["epoch"],
                  "architecture": checkpoint["architecture"], "decoder_sources": checkpoint["decoder_sources"],
                  "data_provenance": catalog.provenance, "threshold": args.threshold,
                  "checkpoint_validation_threshold": checkpoint["threshold"],
                  "debug_head": checkpoint["config"]["debug"], "splits": args.splits,
                  "limit": args.limit, "device": args.device, "versions": common.runtime_versions(),
                  "resize_logits": {"mode": "bilinear", "align_corners": False},
                  "probability": "softmax(native-grid two-class logits)[water=1]",
                  "export_support": "same as reference; NaN probability / -1 mask outside valid support",
                  "empty_water_metrics": "undefined (JSON null / blank CSV) when TP+FP+FN=0",
                  "evaluation_note": "Train predictions are in-sample; validation selected the head. Main splits share events.",
                  "implementation_sha256": {
                      "predict_segmentation.py": common.cache.digest(Path(__file__)),
                      "segmentation_common.py": common.cache.digest(Path(common.__file__)),
                  }}
    output.mkdir(parents=True, exist_ok=True)
    common.cache.write_json(output / "prediction_config.json", provenance)
    rows = []
    totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, n_valid_pixels=0))
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            samples = [catalog.prepare(record) for record in records[start:start + args.batch_size]]
            outputs = model(torch.stack([sample["features"] for sample in samples]).to(args.device))
            for index, sample in enumerate(samples):
                logits = common.native_logits(outputs[index:index + 1], sample["target"].shape)
                probability = logits.softmax(dim=1)[0, 1].cpu().numpy()
                key = sample["row"]["patch_id"]
                probability_path, mask_path = f"probabilities/{key}.tif", f"masks/{key}.tif"
                row, probability, mask, counts = prediction_result(
                    sample, probability, seed, args.threshold, probability_path, mask_path)
                tags = {"patch_id": key, "event_id": row["event_id"], "original_split": row["original_split"],
                        "model_seed": seed, "head_sha256": checkpoint_hash,
                        "cache_fingerprint": catalog.fingerprint, "encoder_sha256": common.cache.SHA256,
                        "threshold": args.threshold, "reference_support": catalog.support,
                        "water_class": 1, "non_water_class": 0}
                write_raster(output / probability_path, probability, sample["geometry"], np.nan, tags)
                write_raster(output / mask_path, mask, sample["geometry"], common.IGNORE_INDEX, tags)
                rows.append(row)
                for name, count in counts.items():
                    totals[row["original_split"]][name] += count
            print(f"Predicted {len(rows)}/{len(records)} patches", flush=True)
    common.write_csv(output / "predictions.csv", rows, FIELDS)
    summary = {"complete": True, "patches": len(rows), "counts_by_split": dict(Counter(row["original_split"] for row in rows)),
               "split_metrics": {split: {**counts, **common.water_metrics(counts)} for split, counts in totals.items()},
               "skipped_unusable_patches": catalog.skipped,
               "predictions": "predictions.csv", "paths_relative_to": "prediction output directory"}
    common.cache.write_json(output / "prediction_summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True, help="best_head.pt from train_segmentation_head.py")
    result.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    result.add_argument("--inspection", type=Path, default=common.DEFAULT_INSPECTION)
    result.add_argument("--feature-dir", type=Path, default=common.DEFAULT_CACHE)
    result.add_argument("--output", type=Path, help="New/empty output directory (default results/segmentation_predictions/seed-N).")
    result.add_argument("--splits", choices=common.SPLITS, nargs="+", default=list(common.SPLITS))
    result.add_argument("--threshold", type=float, default=0.5, help="Fixed for the entire run; initially 0.5, never tuned here.")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--limit", type=int, help="Limit total patches for an implementation check.")
    result.add_argument("--allow-debug-head", action="store_true")
    return result


def main():
    try:
        run_prediction(parser().parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Prediction failed: {error}") from error


if __name__ == "__main__":
    main()
