"""E07 single-scene verification using a shared, read-only burn calibration grid."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from experiments import burn_fraction_causal as burn
from experiments import single_patch_causal as replay
from experiments import verification_scores as analysis


EXPERIMENT = "E07 burn scars, single-patch verification"
POSTERIOR = "b6b94e5ad63e2ecadfb34fc036f720d3f358a99703c1b6fced43f554b15a4a39"
SUMMARY = "84fa8b44e4c1d31f590020901716285c6abf2d0dac4762872442adee15733080"
EMBEDDINGS = "burnscars_ncdd_k64.csv"
EMBEDDING_HASH = "2faf500f22ab1955a61a518acc5486939fcaf405b37f856e0cf70d331d329136"


def cohort(inputs):
    patches, proxy, scores = burn.load_split(inputs, "test")
    embeddings = burn.e02.read_rows(burn.causal.verified(inputs / EMBEDDINGS, EMBEDDING_HASH))
    by_id = {f"burn-test-{int(r['index']):03d}": r for r in embeddings if r["split"] == "test"}
    if set(by_id) != {p["patch_id"] for p in patches} or sum(r["split"] == "test" for r in embeddings) != len(patches):
        raise ValueError("Expected one embedding score row per burn test scene.")
    for patch, (entropy, mi, density) in zip(patches, scores, strict=True):
        embedding = by_id[patch["patch_id"]]
        patch.update(event_id="", average_entropy=float(entropy) if np.isfinite(entropy) else None,
                     mutual_information=float(mi) if np.isfinite(mi) else None, spatial_rarity=-np.log(density),
                     normalized_distance=analysis.number(embedding["normalized_distance"]),
                     negative_NCDD=-analysis.number(embedding["ncdd"]))
    return patches, proxy, scores


def protocol(patches):
    config = burn.causal.joint_protocol(replay.NODES, hurdle=False)
    config.update(source_sha256={**burn.HASHES, EMBEDDINGS: EMBEDDING_HASH}, posterior_sha256=POSTERIOR,
                  baseline_summary_sha256=SUMMARY, support=burn.SUPPORT, integration_seed=105)
    for module in (burn, replay, analysis):
        config["implementation"][Path(module.__file__).name] = burn.e02.digest(Path(module.__file__))
    config["implementation"][Path(__file__).name] = burn.e02.digest(Path(__file__))
    return replay.verification_protocol(config, patches, EXPERIMENT, strengths=(1.,))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepare(cache, posterior, patches, proxy, scores, config):
    reference = np.array([p["reference_fraction"] for p in patches])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as tmp, np.load(posterior, allow_pickle=False) as model:
        stage = Path(tmp) / "cache"
        stage.mkdir()
        hashes, bank_ess = {}, []
        for method in replay.METHODS:
            started = monotonic()
            (grid, mass, mean, evidence), ess = replay.prepare_method(model, proxy, scores, reference, method)
            # Positive beta uses identical quadrature weights for every draw and scene.
            if mass.strides[:2] != (0, 0):
                raise ValueError("Expected a burn calibration without a zero atom.")
            for name, array in (("grid", grid), ("weights", mass[0, 0]), ("mean", mean), ("evidence", evidence)):
                path = stage / f"{method}-{name}.npy"
                np.save(path, array, allow_pickle=False)
                hashes[path.name] = digest(path)
            bank_ess.append(ess)
            print(f"Prepared {method} in {monotonic()-started:.1f}s", flush=True)
        burn.e02.write_json(stage / "manifest.json", {"protocol": config, "sha256": hashes, "bank_ess": bank_ess})
        stage.rename(cache)


def load_cache(cache, config):
    meta = json.loads((cache / "manifest.json").read_text())
    if config != meta["protocol"]:
        raise ValueError("Cache does not match this cohort, code version or integration settings.")
    grids = {}
    for method in replay.METHODS:
        arrays = []
        for name in ("grid", "weights", "mean", "evidence"):
            path = cache / f"{method}-{name}.npy"
            if digest(path) != meta["sha256"][path.name]:
                raise ValueError(f"Damaged verification cache: {path.name}")
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        grid, weights, mean, evidence = arrays
        grids[method] = grid, np.broadcast_to(weights, grid.shape), mean, evidence
    return grids, meta["bank_ess"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "collect"))
    parser.add_argument("root", type=Path, help="Verification run directory (shared cache, shards and report).")
    parser.add_argument("--inputs", type=Path, default=Path("results/causal_e07/inputs"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=burn.causal.TOTAL)
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.repeats <= burn.causal.TOTAL - args.start:
        parser.error("Invalid repetition range.")
    patches, proxy, scores = cohort(args.inputs)
    config = protocol(patches)
    cache = args.root / "cache"
    if args.command == "prepare":
        posterior = burn.causal.verified(args.inputs / "posterior.npz", POSTERIOR)
        burn.causal.verified(args.inputs / "summary.csv", SUMMARY)
        if cache.exists():
            load_cache(cache, config)
            print(f"Reusing verified cache: {cache}", flush=True)
        else:
            prepare(cache, posterior, patches, proxy, scores, config)
    elif args.command == "run":
        started = monotonic()
        grids, ess = load_cache(cache, config)
        config["cache_manifest_sha256"] = digest(cache / "manifest.json")
        print(f"Loaded verified grids in {monotonic()-started:.1f}s", flush=True)
        args.output = args.root / f"shard-{args.start}"
        replay.run_repetitions(args, patches, grids, ess, config)
    else:
        config["cache_manifest_sha256"] = digest(cache / "manifest.json")
        combined = args.root / "combined"
        if not combined.exists():
            replay.collect(args.root, combined, args.start, args.repeats, args.inputs / "summary.csv",
                           summary_hash=SUMMARY, expected_protocol=config)
        meta = json.loads((combined / "run_summary.json").read_text())
        if (meta["start"], meta["repeats"]) != (args.start, args.repeats):
            raise ValueError("Existing collection covers different repetitions.")
        analysis.write_report(combined, args.root / "score-analysis", {p["patch_id"]: p for p in patches},
                              EXPERIMENT, [cache / "manifest.json", Path(__file__)], config)


if __name__ == "__main__":
    main()
