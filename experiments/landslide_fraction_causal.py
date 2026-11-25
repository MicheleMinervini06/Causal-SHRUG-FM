"""E07, landslide replication of E05 using the SHRUG-FM evaluation exports."""

import argparse
from ast import literal_eval
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments import fraction_calibration as calibration
from experiments import fraction_calibration_causal as causal
from experiments import semisynthetic_confounder as e02


ROOT = Path("results/causal_e07/landslides")
METRICS = "landslide_test_metrics.csv"
DENSITY = "landslides_density.csv"
HASHES = {
    METRICS: "82dfc08c9b2240ec0a3c08e2d05fa2a61bce51c6dc76a5549ac56956539cce8d",
    DENSITY: "a1974d9a279e87969563a643919e8af318aa17c2a9221834baad5378f0183b0b",
}
EXPERIMENT = "E07 landslides, E05 replication"
STAGE = EXPERIMENT + ", validation only"
SUPPORT = ("Exported positive pixel counts divided by the full 64 x 64 reference-mask grid. "
           "The exports do not provide invalid-pixel counts. Cohort conditional on landslide-containing patches.")
SIZE = {"val": 41, "test": 68}
RECOVERY_BATCHES = 18


def load_split(inputs, split):
    def indexed(name):
        rows = [r for r in e02.read_rows(causal.verified(inputs / name, HASHES[name])) if r["Set"] == split]
        by_id = {int(r["Sample ID"]): r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != set(range(SIZE[split])):
            raise ValueError(f"Unexpected {split} sample IDs in {name}.")
        return by_id

    metrics, density = indexed(METRICS), indexed(DENSITY)
    patches, scores, boxes = [], [], []
    for i in range(SIZE[split]):
        row = metrics[i]
        reference, proxy = [int(float(row[k])) for k in ("GT_Area", "Pred_Area")]
        if not (0 < reference < 64**2 and 0 <= proxy <= 64**2):
            raise ValueError(f"Unexpected landslide pixel counts at {split}/{i}.")
        total, mi, variance = (float(row[f"{k}_0.05_scaled"]) for k in ("Entropy", "Mutual Information", "Variance"))
        spatial = float(density[i]["density"])
        if not np.isfinite([total, mi, variance, spatial]).all() or not (0 <= mi <= total and variance >= 0 and spatial > 0):
            raise ValueError(f"Invalid reliability score at {split}/{i}.")
        all_zero = total == mi == variance == 0
        scores.append([np.nan if all_zero else total - mi, np.nan if all_zero else mi, spatial])
        bbox = np.asarray(literal_eval(row["Location"]), dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or not np.allclose(bbox[2:] - bbox[:2], 640, rtol=0):
            raise ValueError(f"Unexpected projected bounding box at {split}/{i}.")
        boxes.append(bbox)
        patches.append({"patch_id": f"landslide-{split}-{i:03d}", "original_split": split,
                        "location": row["Location"], "density_id": density[i]["id"],
                        "reference_fraction": reference / 64**2, "predicted_fraction": proxy / 64**2})
    return patches, np.asarray([r["predicted_fraction"] for r in patches]), np.asarray(scores), np.asarray(boxes)


def fit(args):
    patches, proxy, scores, boxes = load_split(args.inputs, "val")
    reference = np.asarray([r["reference_fraction"] for r in patches])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    folds = cut_tree(linkage(centers, method="ward"), n_clusters=5).ravel()
    config = {
        "stage": STAGE, "support": SUPPORT,
        "model": "Positive beta with logit mean and log precision, conditional on presence.",
        "posterior": "Joint calibration draws estimated on validation, shared across patches.",
        "intervals": "Equal-tailed beta posterior predictive intervals.",
        "folds": "Five Ward clusters of exported projected box centers. CRS and event IDs unavailable.",
        "score_definition": "Average member entropy = ensemble entropy minus MI on the 0.05 probability region.",
        "density_alignment": "Join on Set and Sample ID. Density export has no coordinates for independent verification.",
        "empty_roi": "All three task summaries zero at 0.05 triggers the proxy-only distribution.",
        "n_fallback": int((~np.isfinite(scores).all(axis=1)).sum()),
        "raw_proxy_MAE": float(np.mean(abs(reference-proxy))),
        "raw_proxy_RMSE": float(np.sqrt(np.mean((reference-proxy)**2))),
        "adapter_sha256": e02.digest(Path(__file__)),
    }
    calibration.fit_validation((patches, reference, proxy, scores, folds), args.output, 42,
                               source_hashes=HASHES, burn_protocol=config)


def recovery_protocol(path, current):
    meta = json.loads(path.read_text(encoding="utf-8"))
    causal.verified(path.with_name("estimates.csv"), meta["estimates_sha256"])
    archived = meta["protocol"]
    expected = {**current, "git_commit": archived["git_commit"],
                "implementation": dict(current["implementation"])}
    for name in (Path(__file__).name, Path(causal.__file__).name, Path(causal.inference.__file__).name):
        previous = subprocess.check_output(["git", "show", f"{archived['git_commit']}:experiments/{name}"],
                                           cwd=e02.ROOT)
        expected["implementation"][name] = sha256(previous).hexdigest()
    if expected != archived or "recovery" in meta:
        raise ValueError("Recovery requires a compatible, completed shard from the original landslide array.")
    return archived


def simulate(args):
    summary_path = args.calibration / "calibration_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["stage"] != STAGE or summary["source_sha256"] != HASHES or summary["seed"] != 42
            or summary["n_patches"] != SIZE["val"] or summary["support"] != SUPPORT):
        raise ValueError("Expected the frozen E07 landslide validation posterior.")
    posterior_hash = summary["output_sha256"]["posterior.npz"]
    posterior = causal.verified(args.calibration / "posterior.npz", posterior_hash)
    with np.load(posterior, allow_pickle=False) as bank:
        if not bank.get("positive_only", False):
            raise ValueError("Expected a positive beta calibration.")
    patches, proxy, scores, _ = load_split(args.inputs, "test")
    config = causal.joint_protocol(args.nodes, hurdle=False)
    config.update(experiment=EXPERIMENT, simulation="68 fixed landslide test patches; E02 draws, true tau=1, a0=0, a1=0/1",
                  controls=f"OLS, model-based Student-t({len(patches)-3}) intervals", support=SUPPORT,
                  source_sha256=HASHES, n_patches=len(patches), n_fallback=int((~np.isfinite(scores).all(axis=1)).sum()),
                  calibration_summary_sha256=e02.digest(summary_path), posterior_sha256=posterior_hash)
    config["implementation"][Path(__file__).name] = e02.digest(Path(__file__))
    if args.resume_from:
        config["integration_seed"] = args.integration_seed
        original = recovery_protocol(args.resume_from, config)
        causal.simulate_cohort(args, patches, proxy, scores, posterior, original,
                               max_batches=RECOVERY_BATCHES,
                               recovery={"execution_protocol": config, "max_batches": RECOVERY_BATCHES})
    else:
        causal.simulate_cohort(args, patches, proxy, scores, posterior, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    sim = commands.add_parser("simulate")
    for command in (train, sim):
        command.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    train.add_argument("--output", type=Path, default=ROOT / "landslide_calibration_seed42")
    sim.add_argument("--output", type=Path, required=True)
    sim.add_argument("--calibration", type=Path, default=ROOT / "landslide_calibration_seed42")
    sim.add_argument("--nodes", type=int, default=16)
    sim.add_argument("--integration-seed", type=int, default=105)
    sim.add_argument("--start", type=int, default=0)
    sim.add_argument("--repeats", type=int, default=causal.TOTAL)
    sim.add_argument("--resume-from", type=Path, metavar="RUN_SUMMARY",
                     help="Allow extra sampling batches and verify compatibility with an original shard.")
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
