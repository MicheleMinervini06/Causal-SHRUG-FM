"""Predict from frozen-feature heads and compute SHRUG-FM task uncertainty.

Average member entropy and mutual information follow SHRUG-FM equations 7-9.
Use native-grid member probabilities, then average scores over valid pixels
with mean water probability >= --roi-threshold (0.05). The segmentation/fraction
threshold is separate (0.5). Empty ROIs have undefined scores, not zero.
"""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import segmentation_common as common
    from . import predict_segmentation as prediction
else:
    import segmentation_common as common
    import predict_segmentation as prediction


SCORES = ("average_entropy", "mutual_information", "predictive_entropy")
# The seed and checkpoint epoch may differ; the training protocol must match.
MATCHING_CONFIG = ("epochs", "batch_size", "learning_rate", "weight_decay", "patience",
                   "warmup_epochs", "max_batches", "loss", "optimizer", "selection_metric",
                   "threshold", "resize_logits", "augmentation", "versions",
                   "implementation_sha256", "selected_patches")
FIELDS = tuple(field for field in prediction.FIELDS if field != "model_seed") + (
    "model_seeds", "n_members", "roi_threshold", "n_roi_pixels", "roi_fraction", "roi_empty",
    *SCORES, *(f"{name}_path" for name in SCORES))


def ensemble_maps(water_probabilities):
    """Binary entropy in nats; input [members,H,W], output native float32 maps.

    This is the original EnsembleUQModel entropy/MI calculation, expressed once
    on saved-forward probabilities to expose average entropy without repeating
    model inference. Binary class probabilities are [1-p_water, p_water].
    """
    probabilities = torch.as_tensor(water_probabilities, dtype=torch.float64, device="cpu")
    if (probabilities.ndim != 3 or probabilities.shape[0] < 2
            or not torch.isfinite(probabilities).all().item()
            or (probabilities < 0).any().item() or (probabilities > 1).any().item()):
        raise ValueError("Expected at least two finite [H,W] member probability maps in [0,1].")

    def entropy(p):
        # xlogy defines 0*log(0)=0, including exactly confident probabilities.
        return -(torch.xlogy(p, p) + torch.xlogy(1 - p, 1 - p))

    mean = probabilities.mean(dim=0)
    average_entropy = entropy(probabilities).mean(dim=0)
    predictive_entropy = entropy(mean)
    values = {"probability": mean, "average_entropy": average_entropy,
              "mutual_information": (predictive_entropy - average_entropy).clamp_min(0),
              "predictive_entropy": predictive_entropy}
    return {name: value.numpy().astype(np.float32) for name, value in values.items()}


def patch_uncertainty(maps, valid, threshold):
    """Aggregate over predicted event support, never reference water labels."""
    roi = valid & (maps["probability"] >= threshold)
    count = int(roi.sum())
    return {"n_roi_pixels": count, "roi_fraction": count / int(valid.sum()),
            "roi_empty": count == 0,
            **{name: float(maps[name][roi].mean(dtype=np.float64)) if count else None
               for name in SCORES}}


def load_members(paths, catalog):
    """Use the existing strict loader; require distinct, compatible members."""
    if len(paths) < 2:
        raise ValueError("Supply at least two independently trained checkpoints.")
    models, members, baseline = [], [], None
    seeds, hashes = set(), set()
    for path in paths:
        model, checkpoint = common.load_head(path, catalog)
        config = checkpoint["config"]
        if config["debug"]:
            raise ValueError("Debug checkpoints cannot be used for ensemble evaluation.")
        seed, digest = checkpoint["model_seed"], common.cache.digest(path)
        if seed in seeds or digest in hashes:
            raise ValueError("Ensemble members must have distinct model seeds and checkpoint hashes.")
        protocol = {key: config[key] for key in MATCHING_CONFIG}
        if baseline is None:
            baseline = protocol
        differences = [key for key in MATCHING_CONFIG if protocol[key] != baseline[key]]
        if differences:
            raise ValueError("Ensemble training configurations differ in: " + ", ".join(differences))
        seeds.add(seed)
        hashes.add(digest)
        models.append(model)
        members.append({"checkpoint": str(Path(path).resolve()), "sha256": digest,
                        "model_seed": seed, "epoch": checkpoint["epoch"],
                        "validation": checkpoint["validation"]})
    return models, members, baseline


def run_prediction(args):
    if (not math.isfinite(args.threshold) or not 0 < args.threshold < 1
            or not math.isfinite(args.roi_threshold) or not 0 <= args.roi_threshold <= 1
            or (args.limit is not None and args.limit < 1)):
        raise ValueError("Segmentation threshold must be in (0,1), ROI threshold in [0,1], and limit positive.")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty; select a new --output directory: {output}")
    catalog = common.FeatureCatalog(args.manifest, args.inspection, args.feature_dir)
    models, members, protocol = load_members(args.checkpoints, catalog)
    runtime = common.configure_runtime(members[0]["model_seed"], args.device, args.threads)
    models = [model.to(args.device) for model in models]
    records = catalog.select(args.splits)
    requested = len(records)
    if args.limit is not None:
        records = records[:args.limit]
    seeds = ";".join(str(member["model_seed"]) for member in members)
    provenance = {
        "schema_version": 1, "members": members, "training_protocol": protocol,
        "data_provenance": catalog.provenance, "architecture": common.ARCHITECTURE,
        "decoder_sources": common.decoder_sources(), "splits": args.splits, "limit": args.limit,
        "threshold": args.threshold, "roi_threshold": args.roi_threshold,
        "device": args.device, "versions": common.runtime_versions(), "reproducibility": runtime,
        "probability": "mean(member softmax(bilinearly resized native-grid logits)[water=1])",
        "resize_logits": {"mode": "bilinear", "align_corners": False},
        "uncertainty": {"average_entropy": "mean_m H(p_m)",
                        "predictive_entropy": "H(mean_m p_m)",
                        "mutual_information": "H(mean_m p_m) - mean_m H(p_m)",
                        "units": "nats; natural logarithm",
                        "precision": "float64 calculation; float32 exported maps",
                        "roi": "reference-valid pixels with ensemble water probability >= roi_threshold",
                        "empty_roi": "undefined scores (null/blank CSV), roi_empty=true, n_roi_pixels=0",
                        "map_support": "all reference-valid pixels; NaN elsewhere"},
        "reference": "https://arxiv.org/html/2511.10370v2",
        "evaluation_note": "Train predictions are in-sample; validation selected heads. Main splits share events. No selector is fitted.",
        "implementation_sha256": {path.name: common.cache.digest(path) for path in
                                  (Path(__file__), Path(prediction.__file__), Path(common.__file__))},
    }
    output.mkdir(parents=True, exist_ok=True)
    common.cache.write_json(output / "prediction_config.json", provenance)
    rows = []
    totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, n_valid_pixels=0))
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            sample = catalog.prepare(record)
            features = sample["features"].unsqueeze(0).to(args.device)
            # One forward per member/patch. Labels retain their native grid.
            probabilities = [common.native_logits(model(features), sample["target"].shape)
                             .softmax(dim=1)[0, 1].cpu() for model in models]
            maps = ensemble_maps(torch.stack(probabilities))
            key = sample["row"]["patch_id"]
            probability_path, mask_path = f"probabilities/{key}.tif", f"masks/{key}.tif"
            row, probability, mask, counts = prediction.prediction_result(
                sample, maps["probability"], None, args.threshold, probability_path, mask_path)
            row.pop("model_seed")
            valid = sample["target"].numpy() != common.IGNORE_INDEX
            row.update(model_seeds=seeds, n_members=len(members), roi_threshold=args.roi_threshold,
                       **patch_uncertainty(maps, valid, args.roi_threshold))
            tags = {**{name: row[name] for name in ("patch_id", "event_id", "original_split")},
                    "model_seeds": seeds, "n_members": len(members),
                    "checkpoint_sha256": ";".join(member["sha256"] for member in members),
                    "cache_fingerprint": catalog.fingerprint, "reference_support": catalog.support,
                    "threshold": args.threshold, "roi_threshold": args.roi_threshold,
                    "water_class": 1, "non_water_class": 0}
            prediction.write_raster(output / probability_path, probability, sample["geometry"], np.nan, tags)
            prediction.write_raster(output / mask_path, mask, sample["geometry"], common.IGNORE_INDEX, tags)
            for name in SCORES:
                path = f"uncertainty/{name}/{key}.tif"
                row[f"{name}_path"] = path
                values = np.where(valid, maps[name], np.nan).astype(np.float32)
                prediction.write_raster(output / path, values, sample["geometry"], np.nan,
                                        {**tags, "metric": name, "units": "nats"})
            rows.append(row)
            for name, count in counts.items():
                totals[row["original_split"]][name] += count
            print(f"Ensemble {index}/{len(records)}: {key} | ROI={row['n_roi_pixels']}/{row['n_valid_pixels']} | "
                  f"AE={row['average_entropy']} | MI={row['mutual_information']}", flush=True)
    common.write_csv(output / "predictions.csv", rows, FIELDS)
    summary = {"complete": True, "patches": len(rows), "requested_patches": requested,
               "limited": len(rows) < requested, "model_seeds": [member["model_seed"] for member in members],
               "counts_by_split": dict(Counter(row["original_split"] for row in rows)),
               "empty_roi_patches": sum(row["roi_empty"] for row in rows),
               "split_metrics": {split: {**counts, **common.water_metrics(counts)} for split, counts in totals.items()},
               "skipped_unusable_patches": catalog.skipped,
               "predictions": "predictions.csv", "paths_relative_to": "prediction output directory"}
    common.cache.write_json(output / "prediction_summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="Distinct compatible best_head.pt files.")
    result.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    result.add_argument("--inspection", type=Path, default=common.DEFAULT_INSPECTION)
    result.add_argument("--feature-dir", type=Path, default=common.DEFAULT_CACHE)
    result.add_argument("--output", type=Path, required=True, help="New/empty output directory.")
    result.add_argument("--splits", choices=common.SPLITS, nargs="+", default=["validation"])
    result.add_argument("--threshold", type=float, default=0.5, help="Segmentation and water-fraction threshold.")
    result.add_argument("--roi-threshold", type=float, default=0.05, help="Uncertainty averaging region, independent of segmentation threshold.")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--limit", type=int, help="Limit total selected patches for an implementation check.")
    return result


def main():
    try:
        run_prediction(parser().parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Ensemble prediction failed: {error}") from error


if __name__ == "__main__":
    main()
