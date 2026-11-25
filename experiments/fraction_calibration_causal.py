"""E05 phase 2. Run paired repetitions or collect the completed Anunna array."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numba
import scipy

from experiments import fraction_calibration as calibration
from experiments import fraction_causal_posterior as inference
from experiments import semisynthetic_confounder as e02
from experiments.fraction_gate_causal import verified


POSTERIOR = "390bc86712fa5a6fe7ba8807af289652970348e2ec71ca6368607ea4affcf4d6"
TEST_SCORES = "55e8d1facc8a00927b761743083a6e16003238145284b0886729e14f900aa10d"
E02_SUMMARY = "f778d10b8fdb58674fb13080d79868af4a8786d06d86884839eb72d183d6d59c"
METHODS = ("oracle", "raw_proxy", *calibration.MODELS)
TOTAL = 5000


def summarize(rows):
    summary, pairs = [], []
    for strength in e02.ASSIGNMENT_STRENGTHS:
        by_method = {}
        for method in METHODS:
            group = sorted((r for r in rows if float(r["a1"]) == strength and r["method"] == method),
                           key=lambda r: int(r["repetition"]))
            tau, lo, hi = (np.array([float(r[key]) for r in group]) for key in ("estimate", "lower95", "upper95"))
            error = tau - e02.TRUE_EFFECT
            metrics = {"bias": error, "rmse": error**2,
                       "coverage95": ((lo <= e02.TRUE_EFFECT) & (e02.TRUE_EFFECT <= hi)).astype(float),
                       "width95": hi - lo}
            means = {k: float(np.mean(v)) for k, v in metrics.items()}
            means["rmse"] = np.sqrt(means["rmse"])
            summary.append({"a1": strength, "method": method, "n_repeats": len(tau), **means})
            by_method[method] = metrics
        for name, left in by_method["with_scores"].items():
            right = by_method["proxy_only"][name]
            delta, influence = (left - right).mean(), left - right
            if name == "rmse":
                a, b = np.sqrt(left.mean()), np.sqrt(right.mean())
                delta, influence = a-b, left/(2*a) - right/(2*b)
            se = influence.std(ddof=1) / np.sqrt(len(left)) if len(left) > 1 else None
            pairs.append({"a1": strength, "metric": name, "n_pairs": len(left), "with_minus_without_scores": delta,
                          "mcse": se, "mc95_low": None if se is None else delta-1.96*se,
                          "mc95_high": None if se is None else delta+1.96*se})
    return summary, pairs


def protocol(nodes):
    return {**joint_protocol(nodes, hurdle=True), "experiment": "E05 phase 2", "posterior_sha256": POSTERIOR,
            "test_scores_sha256": TEST_SCORES, "controls": "OLS, model-based Student-t(85) intervals",
            "simulation": "E02 full 88 patches, shared uniforms/noise, true tau=1, a0=0, a1=0/1"}


def joint_protocol(nodes, *, hurdle):
    atom = " plus exact zero mass" if hurdle else ""
    return {"seed": 42, "nodes": nodes, "implementation": {p.name: e02.digest(p) for p in
            (Path(__file__), Path(inference.__file__), Path(inference.fk.__file__), Path(calibration.__file__),
             Path(e02.__file__), Path(e02.__file__).with_name("synthetic_confounder.py"))},
            "git_commit": e02.git_commit(), "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__,
            "inference": "latent fraction centered/scaled using only observed test proxy mean/SD; "
                         "Gaussian outcome intercept, tau, slope ~ independent N(0,5^2); log sigma ~ N(0,1); "
                         "randomized assignment known p=0.5; confounded logit intercept and slope ~ N(0,2.5^2). "
                         "Uniform shared row of frozen validation posterior; conditioned on observed T,Y. "
                         f"Gauss-Legendre integration over positive beta quantiles{atom}; "
                         "posterior mean and equal-tailed 95% intervals from the accepted batch "
                         "(ESS >= 1000, max normalized weight < 0.02). Failed batches adapt both proposals.",
            "contrast": "with_scores minus proxy_only, paired normal MC intervals conditional on these patches"}


def publish(output, rows, config):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "complete"
        stage.mkdir()
        summary, pairs = summarize(rows)
        for name, table in (("estimates.csv", rows), ("summary.csv", summary), ("paired_comparisons.csv", pairs)):
            e02.write_csv(stage / name, table)
        e02.write_json(stage / "run_summary.json", {**config, "estimates_sha256": e02.digest(stage / "estimates.csv")})
        stage.rename(output)


def cohort(scores_path):
    scores_path = verified(scores_path, TEST_SCORES)
    patches = sorted(e02.read_rows(scores_path), key=lambda r: r["patch_id"])
    for r in patches:
        r["reference_fraction"] = float(r["reference_fraction"])
        r["predicted_fraction"] = float(r["predicted_fraction"])
    proxy = np.array([r["predicted_fraction"] for r in patches])
    scores = np.array([[calibration.number(r[k], missing=True) for k in calibration.FEATURES] for r in patches])
    return patches, proxy, scores


def run(args):
    patches, proxy, scores = cohort(args.test_scores)
    simulate_cohort(args, patches, proxy, scores, verified(args.posterior, POSTERIOR), protocol(args.nodes))


def simulate_cohort(args, patches, proxy, scores, posterior_path, config, *,
                    max_batches=inference.STANDARD_BATCHES, recovery=None):
    # Only the simulator and oracle below receive the standardized reference.
    reference, _, _ = e02.standardize(patches)
    with np.load(posterior_path, allow_pickle=False) as frozen:
        grids = {}
        for name in calibration.MODELS:
            print(f"Preparing {name} calibration grid for {len(proxy)} patches", flush=True)
            grid, mass = inference.latent_grid(frozen, proxy, scores, name, args.nodes)
            mean_c = np.einsum("ijk,ijk->j", grid, mass) / len(grid)
            grids[name] = grid, mass, mean_c
    rows = []
    for rep in range(args.start, args.start + args.repeats):
        uniforms, noise = e02.simulation_draws(patches, 42, rep)
        for strength in e02.ASSIGNMENT_STRENGTHS:
            t, y, _ = e02.simulate(reference, uniforms, noise, 0., strength)
            t, y = np.array(t), np.array(y)
            for method in METHODS:
                if method in grids:
                    try:
                        result = inference.bayes(t, y, *grids[method], randomized=strength == 0,
                                                 seed=[args.integration_seed, rep, int(strength)],
                                                 max_batches=max_batches)
                    except ValueError as error:
                        raise ValueError(f"Repetition {rep}, a1={strength}, {method}: {error}") from error
                else:
                    result = inference.control(t, y, reference if method == "oracle" else proxy)
                rows.append({"repetition": rep, "a1": strength, "method": method, "ess": None,
                             "proposals": None, "retained_proposals": None, "max_weight": None, **result})
        if (rep-args.start+1) % 10 == 0:
            print(f"Completed repetition {rep+1}", flush=True)
    config = {**config, "integration_seed": args.integration_seed}
    manifest = {"start": args.start, "repeats": args.repeats, "protocol": config}
    if recovery is not None:
        manifest["recovery"] = recovery
    publish(args.output, rows, manifest)


def check_e02(summary, archive):
    old = {(float(r["a1"]), r["estimator"]): r for r in e02.read_rows(verified(archive, E02_SUMMARY))
           if r["selection"] == "full"}
    for r in summary:
        if r["method"] in ("raw_proxy", "oracle"):
            baseline = old[r["a1"], "proxy" if r["method"] == "raw_proxy" else "oracle"]
            if any(abs(r[k] - float(baseline[k])) > 1e-12 for k in ("bias", "rmse")):
                raise ValueError("E02 full-cohort bias/RMSE replay differs.")


def collect(directory, output, repeats, e02_summary=Path("results/causal_e02/test_seed42/summary.csv"), *,
            experiment="E05 phase 2"):
    rows, configs, recovered = [], [], []
    recovery = None
    for path in sorted(directory.glob("shard-*/run_summary.json")):
        meta = json.loads(path.read_text())
        table = e02.read_rows(verified(path.with_name("estimates.csv"), meta["estimates_sha256"]))
        configs.append(meta["protocol"])
        rows.extend(table)
        if "recovery" in meta:
            if recovery is not None and meta["recovery"] != recovery:
                raise ValueError("Incompatible recovery settings across shards.")
            recovery = meta["recovery"]
            recovered.append(path.parent.name)
    keys = {(int(r["repetition"]), float(r["a1"]), r["method"]) for r in rows}
    expected = {(i, a, m) for i in range(repeats) for a in e02.ASSIGNMENT_STRENGTHS for m in METHODS}
    if len(keys) != len(rows) or keys != expected or not configs or any(c != configs[0] for c in configs):
        raise ValueError("Incomplete, duplicated, or incompatible shards; no combined report written.")
    rows.sort(key=lambda r: (int(r["repetition"]), float(r["a1"]), METHODS.index(r["method"])))
    if configs[0].get("experiment", "E05 phase 2") != experiment:
        raise ValueError(f"Expected {experiment} shards.")
    verify_e02 = repeats == TOTAL and experiment == "E05 phase 2"
    if verify_e02:
        check_e02(summarize(rows)[0], e02_summary)
    manifest = {"start": 0, "repeats": repeats, "shards": len(configs), "protocol": configs[0],
                "full_e02_controls_verified": verify_e02}
    if recovery is not None:
        manifest.update(recovery=recovery, recovered_shards=recovered,
                        fits_needing_extra_batches=sum(int(r["proposals"] or 0) > 262144 for r in rows))
    publish(output, rows, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=Path("results/causal_e05/phase1_seed42/posterior.npz"))
    parser.add_argument("--test-scores", type=Path, default=Path("results/reliability_gate/test_seed0-pred-job-44220183/gate_scores.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=TOTAL)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--integration-seed", type=int, default=105)
    parser.add_argument("--collect", type=Path, help="Combine complete shard outputs covering repetitions 0..repeats-1.")
    args = parser.parse_args()
    if args.output.exists() or args.start < 0 or not 1 <= args.repeats <= TOTAL - args.start or args.nodes < 2:
        parser.error("Choose a new output, a valid E02 repeat range and at least two quadrature nodes.")
    if args.collect:
        collect(args.collect, args.output, args.repeats)
    else:
        run(args)
