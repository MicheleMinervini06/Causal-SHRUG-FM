"""Train the original SHRUG-FM CoreDecoder on verified frozen spatial features.

No encoder is loaded or updated. See --help; --debug is a small plumbing check,
not a model intended for evaluation.
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import torch

if __package__:
    from . import segmentation_common as common
else:
    import segmentation_common as common


def fit(model, train_loader, validation_loader, output, config, provenance):
    """Fit only train batches; select by aggregate validation water IoU.

    Exposed separately so the loop can be checked with a tiny synthetic head.
    """
    output = Path(output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    # Linear warmup then constant LR, as in SHRUG-FM. Explicitly use a 0.1
    # starting factor rather than the legacy code's unusually small lr/100 factor.
    warmup = config["warmup_epochs"]
    history, best_key, best_epoch, stale = [], None, None, 0
    for epoch in range(1, config["epochs"] + 1):
        factor = 0.1 + 0.9 * min((epoch - 1) / warmup, 1.0) if warmup else 1.0
        learning_rate = config["learning_rate"] * factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train = common.run_epoch(model, train_loader, config["device"], optimizer,
                                 max_batches=config["max_batches"])
        validation = common.run_epoch(model, validation_loader, config["device"],
                                      max_batches=config["max_batches"])
        if validation["IoU"] is None:
            raise ValueError("Validation water IoU is undefined; choose a validation cohort containing water.")
        row = {"epoch": epoch, "learning_rate": learning_rate}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"validation_{key}": value for key, value in validation.items()})
        history.append(row)
        common.write_csv(output / "history.csv", history)
        # Lower validation loss breaks exact IoU ties. Never consult test/Bolivia.
        key = (validation["IoU"], -validation["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            checkpoint = {
                "schema_version": 1, "architecture": common.ARCHITECTURE,
                "decoder_sources": common.decoder_sources(),
                "model_state_dict": {name: value.detach().cpu().clone()
                                     for name, value in model.state_dict().items()},
                "model_seed": config["seed"], "epoch": epoch, "validation": validation,
                "selection_metric": "validation_micro_water_IoU; tie: lower validation CE",
                "threshold": 0.5, "config": config, "data_provenance": provenance,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            common.save_checkpoint(output / "best_head.pt", checkpoint)
        else:
            stale += 1
        print(f"Epoch {epoch}/{config['epochs']}: train CE={train['loss']:.5f}, "
              f"validation CE={validation['loss']:.5f}, water IoU={validation['IoU']:.5f}, "
              f"F1={validation['F1']:.5f}; best epoch={best_epoch}", flush=True)
        if stale >= config["patience"]:
            break
    summary = {"best_epoch": best_epoch, "best_validation_IoU": best_key[0],
               "epochs_completed": len(history), "early_stopped": len(history) < config["epochs"],
               "debug": config["debug"], "checkpoint": "best_head.pt"}
    common.cache.write_json(output / "training_summary.json", summary)
    return summary


def run_training(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.warmup_epochs < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("Invalid epochs, batch size, patience, warmup, learning rate or weight decay.")
    common.configure_runtime(args.seed, args.device, args.threads)
    output = (args.output or common.ROOT / "models/segmentation" /
              f"core_decoder_seed-{args.seed}{'-debug' if args.debug else ''}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; select a new --output directory: {output}")
    catalog = common.FeatureCatalog(args.manifest, args.inspection, args.feature_dir)
    limit = 2 if args.debug else None
    records = {split: catalog.select((split,), limit_per_split=limit)
               for split in ("train", "validation")}
    # Full source/cache verification happens before any optimization/output files.
    datasets = {split: common.PreparedDataset(catalog, subset) for split, subset in records.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=split == "train", generator=generator,
                                          num_workers=0, collate_fn=common.collate_samples)
        for split, dataset in datasets.items()
    }
    model = common.build_decoder().to(args.device)
    config = {
        "seed": args.seed, "device": args.device, "threads": args.threads,
        "epochs": 1 if args.debug else args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "patience": args.patience, "warmup_epochs": args.warmup_epochs,
        "max_batches": 1 if args.debug else None, "debug": args.debug,
        "loss": "valid-pixel mean cross_entropy; ignore_index=-1; unweighted classes",
        "optimizer": "AdamW; linear warmup 0.1 to 1, then constant learning rate",
        "selection_metric": "validation micro water IoU; tie: lower validation CE",
        "threshold": 0.5, "resize_logits": {"mode": "bilinear", "align_corners": False},
        "augmentation": "none; canonical cached image features",
        "empty_water_metrics": "undefined (JSON null / blank CSV) when TP+FP+FN=0",
        "versions": common.runtime_versions(), "architecture": common.ARCHITECTURE,
        "decoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoder_sources": common.decoder_sources(),
        "implementation_sha256": {
            "train_segmentation_head.py": common.cache.digest(Path(__file__)),
            "segmentation_common.py": common.cache.digest(Path(common.__file__)),
        },
        "paths": {"manifest": str(catalog.manifest), "inspection": str(catalog.inspection),
                  "feature_dir": str(catalog.feature_dir)},
        "selected_patches": {split: [row["patch_id"] for row, _ in subset]
                             for split, subset in records.items()},
        "selected_events": {split: sorted({row["event_id"] for row, _ in subset})
                            for split, subset in records.items()},
        "evaluation_note": "Original main splits share events; test/Bolivia never used for fitting or selection.",
    }
    output.mkdir(parents=True, exist_ok=True)
    common.cache.write_json(output / "config.json", {**config, "data_provenance": catalog.provenance})
    print(f"Training original CoreDecoder ({config['decoder_parameters']:,} parameters); "
          f"train={len(datasets['train'])}, validation={len(datasets['validation'])}, "
          f"debug={args.debug}. No encoder execution.", flush=True)
    return fit(model, loaders["train"], loaders["validation"], output, config, catalog.provenance)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    result.add_argument("--inspection", type=Path, default=common.DEFAULT_INSPECTION)
    result.add_argument("--feature-dir", type=Path, default=common.DEFAULT_CACHE)
    result.add_argument("--output", type=Path, help="New/empty directory (default models/segmentation/core_decoder_seed-N).")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--learning-rate", type=float, default=5e-5)
    result.add_argument("--weight-decay", type=float, default=5e-6)
    result.add_argument("--warmup-epochs", type=int, default=3)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--debug", action="store_true", help="One epoch, first two train/validation patches, at most one batch each; original architecture.")
    return result


def main():
    args = parser().parse_args()
    try:
        run_training(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Training failed: {error}") from error


if __name__ == "__main__":
    main()
