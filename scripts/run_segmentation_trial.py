"""Run one controlled decoder trial after a disposable real-batch training check.

The Slurm array maps task IDs 0/1/2 to reference, larger batch and lower LR.
All trials use fresh initialization, 50 epochs and patience 50. Seed defaults
to 0; additional ensemble members can set --seed and --reference-config.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import torch

if __package__:
    from . import segmentation_common as common
    from . import train_segmentation_head as training
else:
    import segmentation_common as common
    import train_segmentation_head as training


TRIALS = (
    {"name": "reference_b1_lr5e-5", "batch_size": 1, "learning_rate": 5e-5},
    {"name": "batch8_lr5e-5", "batch_size": 8, "learning_rate": 5e-5},
    {"name": "lower_lr_b1_lr2e-5", "batch_size": 1, "learning_rate": 2e-5},
)


def code_identity():
    def git(*args):
        result = subprocess.run(["git", "-C", str(common.ROOT), *args],
                                capture_output=True, check=True)
        return result.stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain", "--untracked-files=normal").decode().strip()
        diff_hash = hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, status, diff_hash = None, None, None
    paths = (Path(__file__), Path(common.__file__), Path(training.__file__),
             common.ROOT / "scripts/hpc/train_segmentation_trials.sbatch",
             common.ROOT / "scripts/hpc/train_segmentation_ensemble.sbatch")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None,
            "git_status": status, "git_diff_sha256": diff_hash,
            "source_sha256_lf": {path.relative_to(common.ROOT).as_posix():
                                  hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                                  for path in paths},
            "decoder_sources": common.decoder_sources()}


def check_training_batch(catalog, profile, *, device, threads, seed=0):
    """Exercise actual features, native masks, decoder, gradients AND AdamW state.

    The model/optimizer are discarded. run_training resets the seed and creates a
    separate fresh model afterward; no preflight update enters the experiment.
    """
    runtime = common.configure_runtime(seed, device, threads)
    records = catalog.select(("train",), limit_per_split=profile["batch_size"])
    if len(records) != profile["batch_size"]:
        raise ValueError("Not enough train patches to check the requested physical batch size.")
    dataset = common.PreparedDataset(catalog, records)
    loader = torch.utils.data.DataLoader(dataset, batch_size=profile["batch_size"],
                                         shuffle=False, num_workers=0,
                                         collate_fn=common.collate_samples)
    model = common.build_decoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile["learning_rate"], weight_decay=5e-6)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = common.run_epoch(model, loader, device, optimizer, max_batches=1,
                               log_every=1, progress_label="Disposable batch preflight")
    if device == "cuda":
        torch.cuda.synchronize()
    return {"passed": True, "runtime": runtime, "batch_size": profile["batch_size"],
            "patch_ids": [row["patch_id"] for row, _ in records],
            "native_shapes": [list(sample["target"].shape) for sample in dataset.samples],
            "metrics": metrics,
            "cuda_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "cuda_peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None,
            "note": "Disposable model/optimizer; subsequent training resets seed and initializes fresh weights."}


def training_arguments(args, profile, output):
    return training.parser().parse_args([
        "--manifest", str(args.manifest), "--inspection", str(args.inspection),
        "--feature-dir", str(args.feature_dir), "--output", str(output),
        "--device", args.device, "--threads", str(args.threads),
        "--epochs", "50", "--patience", "50", "--seed", str(args.seed),
        "--batch-size", str(profile["batch_size"]),
        "--learning-rate", str(profile["learning_rate"]),
        "--weight-decay", "5e-6", "--warmup-epochs", "3", "--log-every", "1",
    ])


def verify_reference(path, train_args, catalog):
    """Check comparability with an existing member before any optimizer step.

    The seed and output paths may differ. Exact source, runtime, data and cohort
    checks ensure that the unchanged trainer has the same scientific settings.
    Source hashes use LF to match the reference configuration written on Linux.
    """
    path = Path(path).resolve()
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("seed") == train_args.seed:
        raise ValueError("An additional ensemble member must use a different seed from the reference.")
    expected = {key: getattr(train_args, key) for key in
                ("device", "threads", "epochs", "batch_size", "log_every", "learning_rate",
                 "weight_decay", "patience", "warmup_epochs", "debug")}
    expected.update(
        max_batches=None, architecture=common.ARCHITECTURE,
        decoder_sources=common.decoder_sources(), versions=common.runtime_versions(),
        implementation_sha256={
            path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
            for path in (Path(training.__file__), Path(common.__file__))},
        data_provenance=catalog.provenance,
        selected_patches={split: [row["patch_id"] for row, _ in catalog.select((split,))]
                          for split in ("train", "validation")},
    )
    mismatches = [key for key, value in expected.items() if reference.get(key) != value]
    if mismatches:
        raise ValueError("Reference configuration differs in: " + ", ".join(mismatches)
                         + ". Keep the same training code, environment, data and settings.")
    return {"path": str(path), "sha256": common.cache.digest(path),
            "seed": reference["seed"], "matching_fields": list(expected), "passed": True}


def run_trial(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_group):
        raise ValueError("Run group must be a safe directory name containing letters, digits, '_' or '-'.")
    if args.threads < 1:
        raise ValueError("Threads must be positive.")
    profile = dict(TRIALS[args.trial_id])
    # Preserve historical seed-0 paths; different seeds always get separate directories.
    member_name = profile["name"] + (f"_seed-{args.seed}" if args.seed else "")
    output = Path(args.output_root).resolve() / args.run_group / member_name
    metadata_dir = Path(args.metadata_root).resolve() / args.run_group / member_name
    for path in (output, metadata_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Trial directory already contains results; choose a new run group: {path}")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "trial.json"
    train_args = training_arguments(args, profile, output)
    metadata = {"schema_version": 1, "trial_id": args.trial_id, "profile": profile,
                "seed": args.seed, "run_group": args.run_group,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "preflight", "code": code_identity(),
                "slurm": {key: os.environ.get(key) for key in
                          ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")},
                "training_arguments": {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(train_args).items()},
                "optimizer_steps_note": "With 250 train patches: batch 1 gives 250 updates/epoch; batch 8 gives 32, including a final batch of 2.",
                "evaluation_note": "Validation selects best head; test/Bolivia are not used by these trials."}
    common.cache.write_json(metadata_path, metadata)
    try:
        print(f"Trial {args.trial_id}: {profile['name']} | batch={profile['batch_size']} | "
              f"lr={profile['learning_rate']} | seed={args.seed} | epochs=50 | patience=50", flush=True)
        print(f"Model output: {output}\nTrial metadata: {metadata_path}", flush=True)
        catalog = common.FeatureCatalog(args.manifest, args.inspection, args.feature_dir)
        metadata["data_provenance"] = catalog.provenance
        if args.reference_config is not None:
            metadata["reference"] = verify_reference(args.reference_config, train_args, catalog)
            print("Reference member configuration, data and training code match.", flush=True)
        metadata["preflight"] = check_training_batch(catalog, profile, device=args.device,
                                                     threads=args.threads, seed=args.seed)
        # Ensure no temporary optimizer/model allocation survives into training.
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        metadata["status"] = "training"
        common.cache.write_json(metadata_path, metadata)
        print("Physical-batch forward/backward/AdamW check passed; starting fresh training.", flush=True)
        metadata["summary"] = training.run_training(train_args)
        metadata["status"] = "complete"
        common.cache.write_json(metadata_path, metadata)
        return metadata
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        common.cache.write_json(metadata_path, metadata)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trial-id", type=int, choices=range(len(TRIALS)), required=True)
    result.add_argument("--run-group", required=True, help="Fresh experiment group, normally the Slurm array job ID.")
    result.add_argument("--seed", type=int, default=0, help="Decoder initialization and training shuffle seed.")
    result.add_argument("--reference-config", type=Path,
                        help="Existing member's config.json; require matching protocol, data and runtime, with a different seed.")
    result.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    result.add_argument("--inspection", type=Path, default=common.DEFAULT_INSPECTION)
    result.add_argument("--feature-dir", type=Path, default=common.DEFAULT_CACHE)
    result.add_argument("--output-root", type=Path, default=common.ROOT / "models/segmentation/config_comparison")
    result.add_argument("--metadata-root", type=Path, default=common.ROOT / "results/segmentation_config_comparison")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--threads", type=int, default=4)
    return result


def main():
    try:
        run_trial(parser().parse_args())
    except (OSError, ValueError, KeyError, RuntimeError, IndexError) as error:
        raise SystemExit(f"Trial failed: {error}") from error


if __name__ == "__main__":
    main()
