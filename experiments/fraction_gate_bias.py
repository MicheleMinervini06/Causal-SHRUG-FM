"""E04. Diagnose the three-feature E03 regressor's bias on the fixed test patches."""

import argparse
from collections import Counter, defaultdict
import json
from math import floor, isclose, sqrt
from pathlib import Path
from statistics import fmean, stdev

from experiments import semisynthetic_confounder as e02
from experiments.fraction_gate_causal import E03_MANIFEST, verified


def matched_sample(gate_ids, random_order, strata):
    """Draw uniformly without replacement within each gate-defined stratum quota."""
    remaining = Counter(strata[i] for i in gate_ids)
    selected = []
    for i in random_order:
        key = strata[i]
        if remaining[key] > 0:
            selected.append(i)
            remaining[key] -= 1
    return selected


def diagnose(patches, repeats, seed, a0):
    c = [float(r["reference_standardized"]) for r in patches]
    proxy = [float(r["proxy_standardized"]) for r in patches]
    water = [float(r["reference_fraction"]) for r in patches]
    abs_error = [abs(float(r["predicted_fraction"]) - w) for r, w in zip(patches, water)]
    events = [r["event_id"] for r in patches]
    # Three fixed, interpretable bands: dry, positive up to 10%, above 10%.
    cells = [(event, 0 if w == 0 else 1 if w <= .10 else 2) for event, w in zip(events, water)]
    observed, contrasts = defaultdict(list), defaultdict(list)
    for rep in range(repeats):
        orders = dict(e02.selection_orders(patches, seed, rep, (("regression", "regression_score"),)))
        uniforms, noise = e02.simulation_draws(patches, seed, rep)
        scenarios = {a1: e02.simulate(c, uniforms, noise, a0, a1)[:2] for a1 in e02.ASSIGNMENT_STRENGTHS}
        for coverage in e02.COVERAGES:
            k = floor(coverage * len(patches))
            retained = orders["regression"][:k]
            random_order = orders["random"]
            samples = {"regression": retained, "random": random_order[:k],
                       "event_matched": matched_sample(retained, random_order, events),
                       "event_water_matched": matched_sample(retained, random_order, cells)}
            for a1, (treatment, outcome) in scenarios.items():
                measurements = {}
                for method, ids in samples.items():
                    t, z, y, ref = ([v[i] for i in ids] for v in (treatment, proxy, outcome, c))
                    # Linearity in the outcome: bias realization = residual confounding + outcome noise.
                    try:
                        bias_draw = e02.treatment_coefficient(y, t, z) - e02.TRUE_EFFECT
                        residual = e02.treatment_coefficient(ref, t, z)
                        noise_term = bias_draw - residual
                    except ValueError:  # Same treatment-design degeneracies as E02.
                        bias_draw = residual = noise_term = None
                    metrics = dict(bias=bias_draw, residual_confounding=residual, noise=noise_term,
                                   fraction_MAE=fmean(abs_error[i] for i in ids), reference_mean=fmean(water[i] for i in ids),
                                   overlap_with_gate=len(set(ids).intersection(retained)) / k)
                    measurements[method] = metrics
                    observed[a1, coverage, method].append(metrics)
                for method in ("random", "event_matched", "event_water_matched"):
                    difference = {}
                    for name in ("bias", "residual_confounding", "noise", "fraction_MAE"):
                        left, right = measurements["regression"][name], measurements[method][name]
                        difference[name] = None if left is None or right is None else left - right
                    contrasts[a1, coverage, method].append(difference)
        if (rep + 1) % 500 == 0:
            print(f"Completed {rep + 1}/{repeats}", flush=True)
    summaries, comparisons = [], []
    for (a1, coverage, method), group in observed.items():
        means = {}
        for key in group[0]:
            values = [r[key] for r in group if r[key] is not None]
            means[key] = fmean(values) if values else None
        errors = [r["bias"] for r in group if r["bias"] is not None]
        summaries.append({"a1": a1, "coverage": coverage, "selection": method, "n": floor(coverage * len(patches)),
                          "n_fits": len(errors), **means,
                          "causal_RMSE": sqrt(fmean(x * x for x in errors)) if errors else None})
    for (a1, coverage, method), group in contrasts.items():
        for metric in group[0]:
            draws = [r[metric] for r in group if r[metric] is not None]
            mean = fmean(draws) if draws else None
            margin = 1.96 * stdev(draws) / sqrt(len(draws)) if len(draws) > 1 else None
            comparisons.append({"a1": a1, "coverage": coverage, "comparator": method, "metric": metric,
                                "n_pairs": len(draws), "gate_minus_comparator": mean,
                                "mc95_low": mean - margin if margin is not None else None,
                                "mc95_high": mean + margin if margin is not None else None})
    return summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e03-dir", type=Path, default=Path("results/causal_e03/test_seed42"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    archive = json.loads(verified(args.e03_dir / "run_summary.json", E03_MANIFEST).read_text())
    hashes = archive["output_sha256"]
    if not 2 <= args.repeats <= archive["repeats"]:
        parser.error(f"Choose 2 to {archive['repeats']} paired repetitions.")
    output = args.output or Path(f"results/causal_e04/test_seed42-r{args.repeats}")
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    patches = e02.read_rows(verified(args.e03_dir / "patches.csv", hashes["patches.csv"]))
    for patch in patches:
        patch["regression_score"] = float(patch["regression_score"])
    baseline = e02.read_rows(verified(args.e03_dir / "summary.csv", hashes["summary.csv"]))
    # The pinned E03 archive uses seed 42 and a0=0 at both assignment strengths.
    summaries, contrasts = diagnose(patches, args.repeats, seed=42, a0=0.)
    full = args.repeats == archive["repeats"]
    if full:
        old = {(float(r["a1"]), float(r["target_coverage"]), r["selection"]): r for r in baseline if r["estimator"] == "proxy"}
        for row in summaries:
            if row["selection"] in ("regression", "random"):
                ref = old[row["a1"], row["coverage"], row["selection"]]
                if row["n_fits"] != int(ref["n_success"]):
                    raise ValueError("E03 reproduction failed: number of successful fits")
                for name, field in (("bias", "bias"), ("causal_RMSE", "rmse")):
                    if not isclose(row[name], float(ref[field]), abs_tol=1e-12, rel_tol=0):
                        raise ValueError(f"E03 reproduction failed: {row['selection']} {name}")
    output.mkdir(parents=True)
    e02.write_csv(output / "summary.csv", summaries)
    e02.write_csv(output / "paired_comparisons.csv", contrasts)
    e02.write_json(output / "run_summary.json", {
        "complete": True, "repeats": args.repeats, "seed": 42, "n_patches": len(patches),
        "e03_controls_verified": full, "e03_manifest_sha256": E03_MANIFEST, "git_commit": e02.git_commit(),
        "implementation_sha256": e02.digest(Path(__file__)), "gate": "E03 three-feature fraction-error regressor",
        "water_bands": ["C = 0", "0 < C <= 0.10", "C > 0.10"],
        "matching": "Random order from E03 restricted to the gate quotas each repetition; gate patches remain eligible.",
        "intervals": "paired Monte Carlo intervals conditional on the 88 fixed patches"})
    print(f"Saved E04 results to {output}", flush=True)


if __name__ == "__main__":
    main()
