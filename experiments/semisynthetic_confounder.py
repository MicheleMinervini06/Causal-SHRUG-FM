"""E02: fixed real water-fraction errors and gate scores, simulated T and Y.

Reuse E01's intercept-inclusive OLS. No encoder, decoder or gate is fitted.
All usable test patches participate, including those with undefined mask F1.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import floor, isclose, isfinite, sqrt
from pathlib import Path
from random import Random
from statistics import fmean, pstdev, stdev
import subprocess
import sys

if __package__:
    from .synthetic_confounder import TRUE_EFFECT, sigmoid, treatment_coefficient, write_csv
else:
    from synthetic_confounder import TRUE_EFFECT, sigmoid, treatment_coefficient, write_csv

ROOT = Path(__file__).resolve().parents[1]
COVERAGES = (.75, .5)
ASSIGNMENT_STRENGTHS = (0., 1.)
ESTIMATORS = ("unadjusted", "oracle", "proxy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_patches(evaluation_dir, gate_config_path, inspection_path):
    """Require the entire audited test cohort and the reviewed frozen gate identity."""
    paths = [evaluation_dir / "gate_scores.csv", evaluation_dir / "evaluation_summary.json",
             gate_config_path, inspection_path]
    patches = read_rows(paths[0])
    evaluation, gate = [json.loads(p.read_text(encoding="utf-8")) for p in paths[1:3]]
    audited = read_rows(inspection_path)
    expected = {r["patch_id"]: r for r in audited if r["original_split"] == "test" and r["usable"] == "True"}
    if (evaluation["split"] != "test" or evaluation["gate_model_sha256"] != gate["model_sha256"]
            or gate["schema_version"] != 1
            or digest(inspection_path) != gate["protocol"]["data_provenance"]["inspection_sha256"]):
        raise ValueError("Test split, frozen gate identity or inspection checksum mismatch.")
    ids = [r["patch_id"] for r in patches]
    if (len(ids) != len(set(ids)) or set(ids) != set(expected)
            or len(ids) != evaluation["n_input_patches"] or set(ids) & set(gate["fit_patch_ids"])):
        raise ValueError("Expected every usable test patch exactly once, without gate-fit overlap.")
    members = sorted(int(m[0]) for m in gate["protocol"]["ensemble"])
    for row in patches:
        original = expected[row["patch_id"]]
        if (row["original_split"] != "test" or row["score_kind"] != "held_out"
                or row["event_id"] != original["event_id"]
                or sorted(int(s) for s in row["model_seeds"].split(";")) != members):
            raise ValueError(f"Mismatched held-out patch identity: {row['patch_id']}")
        for key in ("reference_fraction", "predicted_fraction", "gate_score"):
            row[key] = float(row[key])
            if not isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key}: {row['patch_id']}")
        row["n_valid_pixels"] = int(row["n_valid_pixels"])
        if (row["n_valid_pixels"] < 1 or row["n_valid_pixels"] != int(original["reference_valid_pixels"])
                or not isclose(row["reference_fraction"], float(original["reference_water_fraction"]), rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError(f"Reference fraction or valid support mismatch: {row['patch_id']}")
    patches.sort(key=lambda r: r["patch_id"])
    c, proxy, scaling = standardize(patches)
    aligned = [{**{k: r[k] for k in ("patch_id", "event_id", "original_split", "model_seeds",
                                    "n_valid_pixels", "reference_fraction", "predicted_fraction", "gate_score")},
                "reference_standardized": x, "proxy_standardized": z} for r, x, z in zip(patches, c, proxy)]
    return aligned, {"source_sha256": {str(p.resolve()): digest(p) for p in paths},
                     "gate_model_sha256": gate["model_sha256"], "standardization": scaling,
                     "overlapping_gate_fit_events": sorted({r["event_id"] for r in patches} & set(gate["fit_events"]))}


def standardize(patches):
    values = [r["reference_fraction"] for r in patches]
    if len(values) < 6 or floor(min(COVERAGES) * len(values)) < 3:
        raise ValueError("Too few patches for the declared coverages.")
    center, scale = fmean(values), pstdev(values)
    if scale == 0:
        raise ValueError("Reference confounder has no variation.")
    return ([(x - center) / scale for x in values],
            [(r["predicted_fraction"] - center) / scale for r in patches],
            {"center": center, "scale": scale, "ddof": 0, "fit_on": "all usable test reference fractions",
             "policy": "same fixed center/scale for reference, proxy, scenarios and subsets"})


def selections(patches, seed, repetition):
    """Nested exact-size selections, independent of simulated treatment/outcome."""
    n = len(patches)
    tie_rng = Random(f"E02:{seed}:{repetition}:gate_ties")
    ties = [tie_rng.random() for _ in patches]
    gate_order = sorted(range(n), key=lambda i: (patches[i]["gate_score"], ties[i]))
    random_order = Random(f"E02:{seed}:{repetition}:random_selection").sample(range(n), n)
    return [("full", 1., list(range(n)))] + [
        (name, coverage, order[:floor(coverage * n)]) for coverage in COVERAGES
        for name, order in (("gate", gate_order), ("random", random_order))]


def simulate(confounder, uniforms, noise, a0, a1):
    probabilities = [sigmoid(a0 + a1 * c) for c in confounder]
    treatment = [int(u < p) for u, p in zip(uniforms, probabilities)]
    outcome = [TRUE_EFFECT * t + c + e for t, c, e in zip(treatment, confounder, noise)]
    return treatment, outcome, probabilities


def estimate(outcome, treatment, confounder, proxy):
    result = {}
    for name, adjustment in zip(ESTIMATORS, (None, confounder, proxy)):
        try:
            value = treatment_coefficient(outcome, treatment, adjustment)
            if not isfinite(value):
                raise ValueError("Non-finite treatment coefficient.")
            result[f"tau_{name}"], result[f"status_{name}"] = value, "ok"
        except ValueError as error:
            result[f"tau_{name}"], result[f"status_{name}"] = None, str(error)
    return result


def run_simulation(patches, repeats, seed, a0):
    """Re-randomize T/epsilon and independent rejection/tie draws, not the patches."""
    if repeats < 1 or not isfinite(a0):
        raise ValueError("Repeats must be positive and a0 finite.")
    confounder = [r["reference_standardized"] for r in patches]
    proxy = [r["proxy_standardized"] for r in patches]
    estimates, selected_rows = [], []
    for repetition in range(repeats):
        noise_rng = Random(f"E02:{seed}:{repetition}:outcome_noise")
        assignment_rng = Random(f"E02:{seed}:{repetition}:treatment_uniforms")
        noise = [noise_rng.gauss(0, 1) for _ in patches]
        uniforms = [assignment_rng.random() for _ in patches]
        subsets = selections(patches, seed, repetition)
        for name, coverage, indices in subsets:
            subset = [patches[i] for i in indices]
            selected_rows.append({"repetition": repetition, "selection": name, "target_coverage": coverage,
                                  "n": len(indices), "actual_coverage": len(indices) / len(patches),
                                  "patch_ids": json.dumps([r["patch_id"] for r in subset]),
                                  "event_counts": json.dumps(dict(sorted(Counter(r["event_id"] for r in subset).items()))),
                                  "reference_mean": fmean(r["reference_fraction"] for r in subset),
                                  "reference_sd": pstdev(r["reference_fraction"] for r in subset),
                                  "proxy_MAE": fmean(abs(r["predicted_fraction"] - r["reference_fraction"]) for r in subset)})
        for a1 in ASSIGNMENT_STRENGTHS:
            treatment, outcome, probabilities = simulate(confounder, uniforms, noise, a0, a1)
            for name, coverage, indices in subsets:
                c, z, t, y, p = [[values[i] for i in indices] for values in (confounder, proxy, treatment, outcome, probabilities)]
                row = {"repetition": repetition, "a0": a0, "a1": a1, "selection": name,
                       "target_coverage": coverage, "actual_coverage": len(indices) / len(patches), "n": len(indices),
                       "n_treated": sum(t), "n_control": len(t) - sum(t),
                       "propensity_min": min(p), "propensity_max": max(p),
                       "n_propensity_outside_005_095": sum(v < .05 or v > .95 for v in p)}
                for group, value in (("treated", 1), ("control", 0)):
                    observed = [x for x, assigned in zip(c, t) if assigned == value]
                    row[f"{group}_C_min"] = min(observed) if observed else None
                    row[f"{group}_C_max"] = max(observed) if observed else None
                estimates.append({**row, **estimate(y, t, c, z)})
        if (repetition + 1) % max(1, repeats // 10) == 0:
            print(f"Completed {repetition + 1}/{repeats} repetitions", flush=True)
    return estimates, selected_rows


def summarize(estimates):
    groups = defaultdict(list)
    for row in estimates:
        groups[(row["a0"], row["a1"], row["selection"], row["target_coverage"], row["n"])].append(row)
    summaries = []
    for (a0, a1, selection, coverage, n), group in groups.items():
        for name in ESTIMATORS:
            values = [r[f"tau_{name}"] for r in group if r[f"status_{name}"] == "ok"]
            bias = fmean(v - TRUE_EFFECT for v in values) if values else None
            summaries.append({"a0": a0, "a1": a1, "selection": selection, "target_coverage": coverage,
                              "actual_coverage": group[0]["actual_coverage"], "n": n, "estimator": name, "true_effect": TRUE_EFFECT,
                              "n_attempted": len(group), "n_success": len(values), "n_failed": len(group) - len(values),
                              "mean_effect": fmean(values) if values else None, "bias": bias,
                              "absolute_bias": abs(bias) if bias is not None else None,
                              "rmse": sqrt(fmean((v - TRUE_EFFECT) ** 2 for v in values)) if values else None,
                              "sd_effect": stdev(values) if len(values) > 1 else None,
                              "mcse_bias": stdev(values) / sqrt(len(values)) if len(values) > 1 else None})
    return summaries


def run(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Use a new/empty output directory.")
    if args.repeats < 1 or not isfinite(args.a0):
        raise ValueError("Repeats must be positive and a0 finite.")
    patches, provenance = load_patches(args.gate_evaluation_dir, args.gate_config, args.inspection)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    configuration = {**provenance, "git_commit": commit, "python_version": sys.version,
                     "implementation_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "experiments/synthetic_confounder.py")},
                     "n_patches": len(patches), "repeats": args.repeats, "seed": args.seed, "a0": args.a0,
                     "a1": ASSIGNMENT_STRENGTHS, "true_effect": TRUE_EFFECT, "outcome_confounder_coefficient": 1,
                     "outcome_noise_sd": 1, "rejection_coverages": COVERAGES,
                     "rng": "Python Random seeded by E02:{seed}:{repetition}:{stream}; streams: outcome_noise, treatment_uniforms, gate_ties, random_selection",
                     "pairing": "same uniforms/noise across a1; same full T/Y and selections across methods/coverages; nested independent random and gate-tie ordering",
                     "cohort": "every usable test patch; F1 and U_T missingness never filter the causal cohort",
                     "estimator": "E01 OLS with intercept; one reference or proxy adjustment variable",
                     "failure_policy": "retain every attempted regression/status; summary moments conditional on successful estimation, with failure counts",
                     "interpretation": "conditional on fixed patches/predictions/gate; no image resampling, model fitting or new-event generalization; MCSE is simulation precision"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", configuration)
    write_csv(args.output / "patches.csv", patches)
    print(f"E02: {len(patches)} fixed test patches; n retained: full={len(patches)}, gate/random="
          + ", ".join(str(floor(c * len(patches))) for c in COVERAGES), flush=True)
    estimates, selected = run_simulation(patches, args.repeats, args.seed, args.a0)
    summaries = summarize(estimates)
    write_csv(args.output / "estimates.csv", estimates)
    write_csv(args.output / "selections.csv", selected)
    write_csv(args.output / "summary.csv", summaries)
    failures = Counter(r[f"status_{name}"] for r in estimates for name in ESTIMATORS if r[f"status_{name}"] != "ok")
    write_json(args.output / "run_summary.json", {"complete": True, "repeats": args.repeats, "n_patches": len(patches),
               "regressions_attempted": len(estimates) * len(ESTIMATORS), "failures_by_reason": dict(failures),
               "output_sha256": {name: digest(args.output / name) for name in ("config.json", "patches.csv", "estimates.csv", "selections.csv", "summary.csv")}})
    print("a1 selection coverage estimator bias RMSE MCSE(bias) failures", flush=True)
    for row in summaries:
        print(f"{row['a1']:g} {row['selection']} {row['target_coverage']:g} {row['estimator']} "
              + " ".join(f"{row[k]:.5f}" if row[k] is not None else "NA" for k in ("bias", "rmse", "mcse_bias"))
              + f" {row['n_failed']}", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-evaluation-dir", type=Path, required=True, help="Held-out test gate export directory.")
    parser.add_argument("--gate-config", type=Path, default=ROOT / "models/reliability_gate/three_features_seed0/gate_config.json")
    parser.add_argument("--inspection", type=Path, default=ROOT / "results/segmentation_data/patches.csv")
    parser.add_argument("--output", type=Path, required=True, help="New/empty directory.")
    parser.add_argument("--repeats", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--a0", type=float, default=0.)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"E02 failed: {error}\n")


if __name__ == "__main__":
    main()
