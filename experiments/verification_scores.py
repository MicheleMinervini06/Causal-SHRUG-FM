"""E06 step 2. Compare held-out SHRUG-FM scores with single-patch verification impacts."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from experiments import fraction_calibration_causal as e05
from experiments import semisynthetic_confounder as e02
from scripts.segmentation_gate import number


# Same directions as the held-out segmentation baselines. Higher means more uncertain.
SCORES = {"average_entropy": ("average_entropy", 1), "mutual_information": ("mutual_information", 1),
          "spatial_rarity": ("pretraining_log_density", -1), "normalized_distance": ("normalized_distance", 1),
          "negative_NCDD": ("NCDD", -1)}
ENDPOINTS = ("rmse_reduction", "mean_absolute_effect_shift", "absolute_bias_reduction",
             "interval_width_reduction", "coverage95_change", "absolute_fraction_error")


def scored_cohort(path):
    patches, _, _ = e05.cohort(path)
    scored = {}
    for patch in patches:
        signals = {}
        for name, (source, sign) in SCORES.items():
            raw = number(patch[source], missing=True)
            signals[name] = sign*raw if np.isfinite(raw) else None
        scored[patch["patch_id"]] = {**{k: patch[k] for k in ("patch_id", "event_id", "original_split",
                                     "reference_fraction", "predicted_fraction")}, **signals}
    return scored


def image_cases(scored):
    """Choose examples from score ranks alone, before reading the causal impacts."""
    cases = []
    for name in SCORES:
        ordered = sorted((p for p in scored.values() if p[name] is not None),
                         key=lambda p: (p[name], p["patch_id"]))
        for side, q in (("lower", .25), ("higher", .75)):
            if ordered:
                patch = ordered[int(np.floor((len(ordered)-1)*q + .5))]
                cases.append({"score": name, "side": side, "quantile": q, "patch_id": patch["patch_id"],
                              "event_id": patch["event_id"], "score_value": patch[name]})
    return cases


def join_impacts(scored, impacts, meta):
    protocol = meta["protocol"]
    repeats = meta["repeats"]
    if (protocol["experiment"] != "E06 single-patch verification" or protocol["test_scores_sha256"] != e05.TEST_SCORES
            or protocol["posterior_sha256"] != e05.POSTERIOR or repeats < 1
            or (repeats == e05.TOTAL and (meta["start"] != 0 or meta["max_e05_replay_difference"] is None
                                          or not 0 <= meta["max_e05_replay_difference"] <= 1e-8))):
        raise ValueError("Unexpected E06 cohort, calibration or baseline replay.")
    expected = {(p, s, m) for p in scored for s in e02.ASSIGNMENT_STRENGTHS for m in e05.calibration.MODELS}
    rows = []
    for r in impacts:
        key = r["patch_id"], number(r["a1"]), r["method"]
        if key not in expected:
            raise ValueError(f"Unexpected or duplicate impact: {key}")
        expected.remove(key)
        p = scored[key[0]]
        n = int(r["n_resolved"])
        if (any(r[k] != p[k] for k in ("event_id", "original_split"))
                or any(number(r[k]) != p[k] for k in ("reference_fraction", "predicted_fraction"))
                or int(r["n_requested"]) != repeats or not 0 <= n <= repeats
                or r["complete"] != str(n == repeats)):
            raise ValueError(f"Patch metadata or denominator mismatch: {key}")
        row = {**r, **p, "a1": key[1], "n_resolved": n, "complete": n == repeats,
               "absolute_fraction_error": abs(p["predicted_fraction"]-p["reference_fraction"])}
        row.update({k: None for k in ENDPOINTS[:-1]})
        if n:
            row.update(rmse_reduction=-number(r["rmse_change"]),
                       mean_absolute_effect_shift=number(r["mean_absolute_effect_shift"]),
                       absolute_bias_reduction=abs(number(r["bias_before"]))-abs(number(r["bias_after"])),
                       interval_width_reduction=-number(r["width95_change"]), coverage95_change=number(r["coverage95_change"]))
        rows.append(row)
    if (expected or sum(repeats-r["n_resolved"] for r in rows) != meta["n_unresolved"]
            or sum(not r["complete"] for r in rows) != meta["n_incomplete_patch_groups"]):
        raise ValueError("Missing impacts or collector totals do not match.")
    return rows


def correlation(x, y):
    if len(x) >= 2 and np.ptp(x) and np.ptp(y):
        return float(np.corrcoef(x, y)[0, 1])
    return None


def compare(x, y, events):
    """Equal-size halves, averaging all orderings of ties at the boundary."""
    k = len(x)//2
    stats = {"n_each_half": k, "low_mean": None, "high_mean": None, "high_minus_low": None,
             "spearman": None, "within_event_rank_correlation": None, "n_informative_events": 0}
    if not k:
        return stats
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    low = (np.clip(k-(end-counts), 0, counts)/counts)[inverse]
    high = (np.clip(end-(len(x)-k), 0, counts)/counts)[inverse]
    lo, hi = low @ y/k, high @ y/k
    stats.update(low_mean=lo, high_mean=hi, high_minus_low=hi-lo, spearman=correlation(rankdata(x), rankdata(y)))
    # Rank within each informative event, then center to remove event means.
    # Percentile ranks put events of different sizes on a comparable scale.
    xs, ys = [], []
    for event in np.unique(events):
        mask = events == event
        if np.ptp(x[mask]) and np.ptp(y[mask]):
            for values, target in ((x, xs), (y, ys)):
                ranks = rankdata(values[mask])/mask.sum()
                target.extend(ranks-ranks.mean())
            stats["n_informative_events"] += 1
    stats["within_event_rank_correlation"] = correlation(xs, ys)
    return stats


def associations(rows):
    table = []
    for strength in e02.ASSIGNMENT_STRENGTHS:
        for method in e05.calibration.MODELS:
            group = [r for r in rows if r["a1"] == strength and r["method"] == method]
            for name in SCORES:
                defined = [r for r in group if r[name] is not None]
                for population in ("complete", "any_resolved"):
                    used = [r for r in defined if r["complete"] or (population == "any_resolved" and r["n_resolved"])]
                    x = np.array([r[name] for r in used])
                    events = np.array([r["event_id"] for r in used])
                    for endpoint in ENDPOINTS:
                        y = np.array([r[endpoint] for r in used])
                        table.append({"a1": strength, "method": method, "score": name, "endpoint": endpoint,
                                      "population": population, "n_patches": len(used), "n_events": len(set(events)),
                                      "missing_score": len(group)-len(defined), "excluded_resolution": len(defined)-len(used),
                                      "min_resolved": min((r["n_resolved"] for r in used), default=None),
                                      **compare(x, y, events)})
    return table


def plot_primary(rows, repeats, path):
    fig, axes = plt.subplots(2, len(SCORES), figsize=(15, 6), sharex=True, sharey="row", layout="constrained")
    for col, name in enumerate(SCORES):
        used = [r for r in rows if r["a1"] == 1 and r["method"] == "proxy_only" and r["complete"] and r[name] is not None]
        ranks = rankdata([r[name] for r in used])
        axes[0, col].set_title(name.replace("_", " ") + f"\n(n={len(used)})")
        for ax, endpoint in zip(axes[:, col], ENDPOINTS[:2]):
            if used:
                ax.scatter((ranks-.5)/len(used), [r[endpoint] for r in used], s=14, alpha=.65)
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set_xlim(0, 1)
    axes[0, 0].set_ylabel("RMSE before − after")
    axes[1, 0].set_ylabel("Mean |effect estimate shift|")
    fig.supxlabel("Patch score percentile (higher = more uncertain)")
    title = "Single-patch verification · confounded assignment · proxy-only calibration"
    fig.suptitle(title + (f"\nImplementation check only ({repeats} repetitions)" if repeats != e05.TOTAL else ""))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(combined, test_scores, output):
    if output.exists():
        raise ValueError(f"Choose a new report directory: {output}")
    scored = scored_cohort(test_scores)
    cases = image_cases(scored)
    manifest, impacts = combined/"run_summary.json", combined/"patch_impacts.csv"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    rows = join_impacts(scored, e02.read_rows(impacts), meta)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"report"
        stage.mkdir()
        for name, table in (("patch_scores_and_impacts", rows), ("score_associations", associations(rows)), ("image_cases", cases)):
            e02.write_csv(stage/(name+".csv"), table)
        plot_primary(rows, meta["repeats"], stage/"score_impacts.png")
        e02.write_json(stage/"analysis_summary.json", {
            "experiment": "E06 score analysis", "repeats": meta["repeats"],
            "full_run": meta["repeats"] == e05.TOTAL, "n_unresolved": meta["n_unresolved"],
            "primary": "proxy_only, a1=1, complete patches; RMSE reduction and mean absolute effect shift",
            "directions": SCORES, "patches": len(scored), "uncertainty_unit": "patch",
            "simulation_commit": meta["protocol"]["git_commit"], "analysis_commit": e02.git_commit(),
            "sha256": {p.name: e02.digest(p) for p in (manifest, impacts, test_scores, Path(__file__))}})
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True, type=Path)
    parser.add_argument("--test-scores", type=Path, default=Path("results/reliability_gate/test_seed0-pred-job-44220183/gate_scores.csv"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.combined, args.test_scores, args.output)
