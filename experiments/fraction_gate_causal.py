"""E03 phase 2. Score frozen validation gates and repeat the paired E02 test experiment.

Run from the project root with python -m experiments.fraction_gate_causal.
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, groupby
import json
from math import isclose, sqrt
from pathlib import Path
import pickle
from statistics import fmean
import sys

import numpy as np
import sklearn

from experiments import semisynthetic_confounder as e02
from scripts import segmentation_gate as gate
from scripts.validate_embedding_gate import UE

# Reviewed Anunna manifests bind the original cohort, feature export and model bytes.
E02_MANIFEST = "e42f0ff4ab7a4e3413bf1b4905a23b4bd12b964ce30eda42f52dd86b014c4b0f"
PHASE1_MANIFEST = "0984c949573ce667939bc07a7436791ca286dceebd72f65d4da02e1f35816c73"
UE_MANIFEST = "94ffebc8e6655952ed9a16892b2d13a766771f807a82399ccaae6046cba6fb3f"
E03_MANIFEST = "a8665b30d767f9d09c40e09b8c76afbbcd567e7b12f8ff0f6dfd2a5448f31f12"
EXTRA_GATES = (("regression", "regression_score"), ("classification", "classification_score"))
COMPARISONS = tuple(combinations(("regression", "classification", "gate", "random"), 2))


def verified(path, expected):
    if e02.digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def load_test(e02_dir, test_scores, *, with_ue=False):
    baseline = gate.read_json(verified(e02_dir / "run_summary.json", E02_MANIFEST))
    outputs = baseline["output_sha256"]
    config = gate.read_json(verified(e02_dir / "config.json", outputs["config.json"]))
    patches = e02.read_rows(verified(e02_dir / "patches.csv", outputs["patches.csv"]))
    sources = {Path(p).name: v for p, v in config["source_sha256"].items()}
    raw = e02.read_rows(verified(test_scores, sources["gate_scores.csv"]))
    by_id = {r["patch_id"]: r for r in raw}
    ids = [r["patch_id"] for r in patches]
    if (ids != sorted(by_id) or len(raw) != len(by_id) or len(ids) != config["n_patches"]
            or any(r["original_split"] != "test" or r["score_kind"] != "held_out" for r in raw)):
        raise ValueError("Expected the complete, ordered E02 held-out test cohort.")
    features = []
    for row in patches:
        for k in ("reference_fraction", "predicted_fraction", "reference_standardized", "proxy_standardized", "gate_score"):
            row[k] = float(row[k])
        source = by_id[row["patch_id"]]
        features.append({k: gate.number(source[k], missing=k in gate.FEATURES[:2]) for k in gate.FEATURES})
        row.update({k: source[k] for k in (*gate.FEATURES, "roi_empty")})
        if with_ue:
            row.update({k: gate.number(source[k]) for k in UE})
    return patches, gate.feature_matrix(features), config, outputs


def score_gates(directory, patches, values, baseline):
    summary = gate.read_json(verified(directory / "training_summary.json", PHASE1_MANIFEST))
    ids = {r["patch_id"] for r in patches}
    inputs = {}
    for kind, column in EXTRA_GATES:
        cfg_path = directory / kind / "gate_config.json"
        cfg = gate.read_json(cfg_path)
        model_hash = summary["model_sha256"][kind]
        if (cfg["schema_version"] != 2 or cfg["kind"] != kind or cfg["fit_split"] != "validation"
                or cfg["model_sha256"] != model_hash or cfg["f1_baseline_model_sha256"] != baseline["gate_model_sha256"]
                or cfg["delta"] != (summary["delta"] if kind == "classification" else None)
                or cfg["features"] != list(gate.FEATURES) or cfg["density_scale"] != gate.DENSITY_SCALE
                or cfg["sklearn_version"] != sklearn.__version__ or ids.intersection(cfg["fit_patch_ids"])):
            raise ValueError(f"Incompatible frozen {kind} gate, features or fit cohort.")
        # Deserialize only the two models whose bytes were recorded in the reviewed summary.
        with verified(directory / kind / "gate.pkl", model_hash).open("rb") as stream:
            model = pickle.load(stream)
        scores = model.predict(values) if kind == "regression" else gate.failure_scores(model, values)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid {kind} scores on held-out patches.")
        for row, score in zip(patches, scores, strict=True):
            row[column] = float(score)
        inputs[kind] = {"model_sha256": model_hash, "config_sha256": e02.digest(cfg_path)}
    return inputs, summary["delta"]


def score_embedding_gate(directory, patches, values, regression_config):
    summary = gate.read_json(verified(directory / "training_summary.json", UE_MANIFEST))
    cfg_path = directory / "gate_config.json"
    cfg = gate.read_json(cfg_path)
    model_hash = summary["model_sha256"]
    if (cfg["schema_version"] != 3 or cfg["kind"] != "regression_with_UE" or cfg["fit_split"] != "validation"
            or cfg["model_sha256"] != model_hash or cfg["three_feature_model_sha256"] != regression_config["model_sha256"]
            or cfg["features"] != [*gate.FEATURES, *UE] or cfg["density_scale"] != gate.DENSITY_SCALE
            or cfg["target"] != "absolute_fraction_error" or cfg["sklearn_version"] != sklearn.__version__
            or any(cfg[k] != regression_config[k] for k in ("parameters", "fit_patch_ids", "protocol"))):
        raise ValueError("Incompatible five-feature gate or three-feature comparator.")
    with verified(directory / "gate.pkl", model_hash).open("rb") as stream:
        model = pickle.load(stream)
    extended = np.column_stack((values, [[r[k] for k in UE] for r in patches]))
    scores = model.predict(extended)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid five-feature predictions on held-out patches.")
    for row, score in zip(patches, scores, strict=True):
        row["regression_with_UE_score"] = float(score)
    return {"model_sha256": model_hash, "config_sha256": e02.digest(cfg_path)}


def check_baselines(summaries, previous):
    """Check all control methods represented in the archived reference summary."""
    def key(row):
        return float(row["a1"]), row["selection"], float(row["target_coverage"]), row["estimator"]

    control_methods = {r["selection"] for r in previous}
    current = {key(r): r for r in summaries if r["selection"] in control_methods}
    if set(current) != {key(r) for r in previous} or len(previous) != len(current):
        raise ValueError("Archived control summary groups differ.")
    for old in previous:
        new = current[key(old)]
        for field, value in old.items():
            if field in ("selection", "estimator"):
                continue
            if value == "" and new[field] is None:
                continue
            if new[field] is None or value == "" or not isclose(float(value), new[field], rel_tol=1e-12, abs_tol=1e-13):
                raise ValueError(f"Archived baseline differs for {key(old)}: {field}")


def paired_contrasts(estimates, comparisons=COMPARISONS):
    """Matched proxy comparisons on successful repetition pairs; MC uncertainty is conditional on patches."""
    groups = defaultdict(lambda: defaultdict(dict))
    for r in estimates:
        if r["selection"] != "full":
            groups[r["a1"], r["target_coverage"]][r["selection"]][r["repetition"]] = r["tau_proxy"]
    rows = []
    for (a1, coverage), methods in groups.items():
        for left, right in comparisons:
            x, y = methods[left], methods[right]
            ids = [i for i in x if x[i] is not None and y[i] is not None]
            row = {"a1": a1, "target_coverage": coverage, "selection": left, "comparator": right,
                   "n_attempted_pairs": len(x), "n_successful_pairs": len(ids), "delta_absolute_bias": None}
            for metric in ("bias", "rmse"):
                row.update({f"delta_{metric}": None, f"mcse_delta_{metric}": None,
                            f"mc95_{metric}_low": None, f"mc95_{metric}_high": None})
            if ids:
                a, b = (np.array([v[i] - e02.TRUE_EFFECT for i in ids]) for v in (x, y))
                ra, rb = (np.sqrt(np.mean(v**2)) for v in (a, b))
                row["delta_absolute_bias"] = float(abs(a.mean()) - abs(b.mean()))
                # Paired first-order influence for a difference of root mean squared errors.
                rmse_influence = a**2 / (2 * ra) - b**2 / (2 * rb) if min(ra, rb) > 0 else None
                for metric, delta, influence in (("bias", (a - b).mean(), a - b), ("rmse", ra - rb, rmse_influence)):
                    row[f"delta_{metric}"] = float(delta)
                    if len(ids) >= 2 and influence is not None:
                        se = float(influence.std(ddof=1) / sqrt(len(ids)))
                        row.update({f"mcse_delta_{metric}": se, f"mc95_{metric}_low": delta - 1.96 * se,
                                    f"mc95_{metric}_high": delta + 1.96 * se})
            rows.append(row)
    return rows


def selection_diagnostics(patches, selections, delta, comparisons=COMPARISONS):
    """Pool retained patch errors across repetitions and keep event-specific denominators."""
    by_id = {r["patch_id"]: r for r in patches}
    available = Counter(r["event_id"] for r in patches)
    included = defaultdict(Counter)
    shared = defaultdict(list)
    repeats = 0
    for _, rep in groupby(selections, key=lambda r: r["repetition"]):
        repeats += 1
        masks = {}
        for r in rep:
            name, coverage = r["selection"], r["target_coverage"]
            ids = set(json.loads(r["patch_ids"]))
            included[name, coverage].update(ids)
            masks[name, coverage] = ids
        for coverage in e02.COVERAGES:
            for a, b in comparisons:
                shared[coverage, a, b].append(len(masks[a, coverage] & masks[b, coverage]))
    composition, events = [], []
    for (name, coverage), counts in included.items():
        total = counts.total()
        # Stable patch order also makes floating-point sums reproducible across Python processes.
        ids = sorted(counts)
        weights = np.array([counts[i] / total for i in ids])
        reference = np.array([by_id[i]["reference_fraction"] for i in ids])
        errors = np.array([by_id[i]["predicted_fraction"] for i in ids]) - reference
        by_event = Counter()
        for i, n in counts.items():
            by_event[by_id[i]["event_id"]] += n
        composition.append({"selection": name, "target_coverage": coverage, "retained": total // repeats,
                            "fraction_MAE": weights @ abs(errors), "fraction_RMSE": sqrt(weights @ errors**2),
                            "fraction_error_failure_rate": weights @ (abs(errors) > delta),
                            "signed_fraction_error": weights @ errors, "reference_mean": weights @ reference})
        for event, n in sorted(available.items()):
            events.append({"selection": name, "target_coverage": coverage, "event_id": event, "available": n,
                           "mean_retained": by_event[event] / repeats, "event_retention": by_event[event] / (repeats * n)})
    overlap = [{"target_coverage": c, "selection": a, "comparator": b, "mean_shared_patches": fmean(values),
                "mean_fraction_of_retained_shared": fmean(values) / int(c * len(patches))}
               for (c, a, b), values in shared.items()]
    return composition, events, overlap


def treatment_overlap(estimates):
    groups = defaultdict(list)
    for row in estimates:
        groups[row["a1"], row["selection"], row["target_coverage"]].append(row)
    return [{"a1": a1, "selection": name, "target_coverage": coverage,
             "min_treated": min(r["n_treated"] for r in rows), "min_control": min(r["n_control"] for r in rows),
             "propensity_min": min(r["propensity_min"] for r in rows),
             "propensity_max": max(r["propensity_max"] for r in rows),
             "mean_n_propensity_outside_005_095": fmean(r["n_propensity_outside_005_095"] for r in rows)}
            for (a1, name, coverage), rows in groups.items()]


def run(args, *, with_ue=False):
    if args.output.exists():
        raise ValueError("Choose a new output directory.")
    print("Checking the archived E02 cohort and both frozen Anunna gates.", flush=True)
    patches, values, baseline, hashes = load_test(args.e02_dir, args.test_scores, with_ue=with_ue)
    if not 1 <= args.repeats <= baseline["repeats"]:
        raise ValueError(f"Choose 1 to {baseline['repeats']} paired repetitions.")
    models, delta = score_gates(args.gates_dir, patches, values, baseline)
    original = e02.read_rows(verified(args.e02_dir / "summary.csv", hashes["summary.csv"]))
    gates, comparisons = EXTRA_GATES, COMPARISONS
    extension = {}
    if with_ue:
        reg = gate.read_json(args.gates_dir / "regression/gate_config.json")
        models["regression_with_UE"] = score_embedding_gate(args.ue_dir, patches, values, reg)
        gates = (*EXTRA_GATES, ("regression_with_UE", "regression_with_UE_score"))
        comparisons = (("regression_with_UE", "regression"),)
        reference = gate.read_json(verified(args.e03_dir / "run_summary.json", E03_MANIFEST))
        previous_e03 = e02.read_rows(verified(args.e03_dir / "summary.csv", reference["output_sha256"]["summary.csv"]))
        extension = {"comparison": "five minus three features, exploratory on previously inspected test",
                     "ue_training_summary_sha256": UE_MANIFEST, "e03_run_summary_sha256": E03_MANIFEST}
    args.output.mkdir(parents=True)
    config = {**baseline, **extension, "experiment": "E03", "stage": "held_out_causal", "repeats": args.repeats,
              "e02_git_commit": baseline["git_commit"], "e02_implementation_sha256": baseline["implementation_sha256"],
              "git_commit": e02.git_commit(), "python_version": sys.version,
              "implementation_sha256": {p.name: e02.digest(p) for p in
                                        (Path(__file__), Path(e02.__file__), e02.ROOT / "experiments/synthetic_confounder.py",
                                         Path(gate.__file__))},
              "e02_run_summary_sha256": E02_MANIFEST, "phase1_training_summary_sha256": PHASE1_MANIFEST,
              "e02_patches_sha256": hashes["patches.csv"], "frozen_gates": models, "sklearn_version": sklearn.__version__,
              "primary": "regression_with_UE" if with_ue else "regression",
              "secondary": "regression" if with_ue else "classification", "delta": delta, "gate": "original F1 gate",
              "ties": "same random priority per patch across gates and coverages in each repetition",
              "intervals": "paired normal 95% Monte Carlo intervals, conditional on successful pairs and fixed patches; delta method for RMSE"}
    e02.write_json(args.output / "config.json", config)
    print(f"E03: scored {len(patches)} held-out patches with both frozen gates. Starting paired simulation.", flush=True)
    estimates, selected = e02.run_simulation(patches, args.repeats, baseline["seed"], baseline["a0"], extra_gates=gates)
    summaries = e02.summarize(estimates)
    full_run = args.repeats == baseline["repeats"]
    if full_run:
        check_baselines(summaries, original)
        if with_ue:
            check_baselines(summaries, previous_e03)
    composition, events, shared = selection_diagnostics(patches, selected, delta, comparisons)
    tables = {"patches.csv": patches, "estimates.csv": estimates, "selections.csv": selected, "summary.csv": summaries,
              "paired_comparisons.csv": paired_contrasts(estimates, comparisons), "selection_summary.csv": composition,
              "event_composition.csv": events, "selection_overlap.csv": shared,
              "treatment_overlap.csv": treatment_overlap(estimates)}
    for filename, rows in tables.items():
        e02.write_csv(args.output / filename, rows)
    e02.write_json(args.output / "run_summary.json", {
        "complete": True, "full_run": full_run, "e02_control_groups_verified": len(original) if full_run else 0,
        **({"e03_control_groups_verified": len(previous_e03) if full_run else 0} if with_ue else {}),
        "n_patches": len(patches), "repeats": args.repeats, "regressions_attempted": len(estimates) * len(e02.ESTIMATORS),
        "regressions_failed": sum(r["n_failed"] for r in summaries),
        "output_sha256": {name: e02.digest(args.output / name) for name in ("config.json", *tables)}})
    print("Confounded scenario, proxy adjustment: selection, coverage, bias, RMSE", flush=True)
    for row in summaries:
        if row["a1"] == 1 and row["estimator"] == "proxy" and row["n_success"]:
            print(f"{row['selection']:>14} {row['target_coverage']:>5.0%} {row['bias']:+.5f} {row['rmse']:.5f}", flush=True)
    print(f"E03 phase 2 complete. Results: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e02-dir", type=Path, default=Path("results/causal_e02/test_seed42"))
    parser.add_argument("--test-scores", type=Path, default=Path("results/reliability_gate/test_seed0-pred-job-44220183/gate_scores.csv"))
    parser.add_argument("--gates-dir", type=Path, default=Path("models/reliability_gate/e03_fraction_error_seed0"))
    parser.add_argument("--with-ue", action="store_true", help="Compare the frozen five- and three-feature regressors.")
    parser.add_argument("--ue-dir", type=Path, default=Path("models/reliability_gate/e03_regression_with_ue_seed0"))
    parser.add_argument("--e03-dir", type=Path, default=Path("results/causal_e03/test_seed42"))
    parser.add_argument("--repeats", type=int, default=5000, help="Use a short prefix for an implementation check.")
    parser.add_argument("--output", type=Path, help="New directory for outputs. Short checks use a separate default.")
    args = parser.parse_args()
    if args.output is None:
        name = "test_seed42" if args.repeats == 5000 else f"check_seed42-r{args.repeats}"
        args.output = Path("results/causal_e03") / (f"ue_{name}" if args.with_ue else name)
    try:
        run(args, with_ue=args.with_ue)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"E03 phase 2 failed: {error}\n")


if __name__ == "__main__":
    main()
