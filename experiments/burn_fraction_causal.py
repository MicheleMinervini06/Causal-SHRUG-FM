"""E07, burn-scar replication of E05: fit on validation, simulate on test, collect."""

import argparse
from ast import literal_eval
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments import fraction_calibration as calibration
from experiments import fraction_calibration_causal as causal
from experiments import semisynthetic_confounder as e02


INPUTS = Path("results/burn_scars/export-audit/inputs")
CALIBRATION = Path("results/causal_e07/burn_calibration_seed42")
METRICS = "burnscars_test_metrics.csv"
DENSITY = "burn_scars_density.csv"
HASHES = {METRICS: "7a17ad89f677d6320d34c69ac161f5c23f01b16ebe58e1d896f7883ba7fc567b",
          DENSITY: "5f5027cc43eb675a345eaa9de13a65115ca01e80ed51b85e0f0c16df3b372f77"}
EXPERIMENT = "E07 burn scars, E05 replication"
SUPPORT = "Exported burned pixel counts divided by all 512 x 512 pixels; conditional on burn-containing scenes."


def load_split(directory, split):
    """Use the metrics split authority and the assumed within-split density order."""
    def indexed(name):
        rows = [r for r in e02.read_rows(causal.verified(directory / name, HASHES[name])) if r["Set"] == split]
        by_index = {int(r["Sample ID"]): r for r in rows}
        if len(by_index) != len(rows) or set(by_index) != set(range({"val": 54, "test": 263}[split])):
            raise ValueError(f"Unexpected {split} scene IDs in {name}.")
        return by_index

    metrics, density = indexed(METRICS), indexed(DENSITY)
    patches, scores = [], []
    for i in range(len(metrics)):
        row = metrics[i]
        c, proxy = (float(row[k]) / 512**2 for k in ("GT_Area", "Pred_Area"))
        if not 0 < c < 1 or not 0 <= proxy <= 1:
            raise ValueError(f"Invalid burned fraction at {split}/{i}.")
        total, mi, variance = (float(row[f"{k}_0.05_scaled"]) for k in ("Entropy", "Mutual Information", "Variance"))
        spatial = float(density[i]["density"])
        if not np.isfinite([total, mi, variance, spatial]).all() or not 0 <= mi <= total or spatial <= 0:
            raise ValueError(f"Invalid reliability score at {split}/{i}.")
        empty = total == mi == variance == 0
        scores.append([np.nan if empty else total - mi, np.nan if empty else mi, spatial])
        patches.append({"patch_id": f"burn-{split}-{i:03d}", "original_split": split, "location": row["Location"],
                        "reference_fraction": c, "predicted_fraction": proxy})
    return patches, np.array([r["predicted_fraction"] for r in patches]), np.asarray(scores)


def geographic_folds(patches):
    """Five contiguous-region approximations based solely on validation scene centers."""
    lon, lat = np.radians([literal_eval(r["location"]) for r in patches]).T
    xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    return cut_tree(linkage(xyz, method="ward"), n_clusters=5).ravel()


def fit(args):
    patches, proxy, scores = load_split(args.inputs, "val")
    reference = np.array([r["reference_fraction"] for r in patches])
    folds = geographic_folds(patches)
    config = {
        "stage": EXPERIMENT + ", validation only", "support": SUPPORT,
        "model": "Positive beta with logit mean and log precision. No presence component.",
        "posterior": "Joint coefficient draw shared across all scenes; calibration fit only on validation.",
        "intervals": "Equal-tailed positive beta posterior predictive intervals.",
        "folds": "Five Ward clusters of validation unit-sphere scene centers; independent fires unverified.",
        "empty_roi": "All three exported task summaries zero at 0.05: use proxy-only distribution.",
        "score_definition": "Average member entropy = ensemble entropy minus MI at the same 0.05 ROI.",
        "density_alignment": "Assumed by original split and sample index; no scene ID in density export.",
        "n_fallback": int((~np.isfinite(scores).all(axis=1)).sum()),
        "raw_proxy_MAE": float(np.mean(abs(reference-proxy))), "raw_proxy_RMSE": float(np.mean((reference-proxy)**2)**.5),
        "adapter_sha256": e02.digest(Path(__file__)),
    }
    calibration.fit_validation((patches, reference, proxy, scores, folds), args.output, 42,
                               source_hashes=HASHES, burn_protocol=config)


def simulate(args):
    summary_path = args.calibration / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["stage"] != EXPERIMENT + ", validation only" or summary["source_sha256"] != HASHES
            or summary["n_patches"] != 54 or summary["seed"] != 42):
        raise ValueError("Expected frozen E07 burn-scar validation posterior.")
    posterior_hash = summary["output_sha256"]["posterior.npz"]
    posterior = causal.verified(args.calibration / "posterior.npz", posterior_hash)
    patches, proxy, scores = load_split(args.inputs, "test")
    config = causal.joint_protocol(args.nodes, hurdle=False)
    config.update(experiment=EXPERIMENT, simulation="263 fixed burn test scenes; E02 draws, true tau=1, a0=0, a1=0/1",
                  controls="OLS, model-based Student-t(260) intervals", support=SUPPORT, source_sha256=HASHES,
                  n_patches=len(patches), n_fallback=int((~np.isfinite(scores).all(axis=1)).sum()),
                  calibration_summary_sha256=e02.digest(summary_path), posterior_sha256=posterior_hash)
    config["implementation"][Path(__file__).name] = e02.digest(Path(__file__))
    causal.simulate_cohort(args, patches, proxy, scores, posterior, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--inputs", type=Path, default=INPUTS)
    train.add_argument("--output", type=Path, default=CALIBRATION)
    sim = commands.add_parser("simulate")
    sim.add_argument("--inputs", type=Path, default=INPUTS)
    sim.add_argument("--calibration", type=Path, default=CALIBRATION)
    sim.add_argument("--nodes", type=int, default=16)
    sim.add_argument("--integration-seed", type=int, default=105)
    sim.add_argument("--start", type=int, default=0)
    sim.add_argument("--repeats", type=int, default=causal.TOTAL)
    sim.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("collect")
    combine.add_argument("directory", type=Path)
    combine.add_argument("--repeats", type=int, default=causal.TOTAL)
    args = parser.parse_args()
    if args.command == "collect":
        causal.collect(args.directory, args.directory / "combined", args.repeats, experiment=EXPERIMENT)
    elif args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")
    elif args.command == "fit":
        fit(args)
    elif args.start < 0 or not 1 <= args.repeats <= causal.TOTAL - args.start or args.nodes < 2:
        parser.error("Invalid repetition range or quadrature order.")
    else:
        simulate(args)


if __name__ == "__main__":
    main()
