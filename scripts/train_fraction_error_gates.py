"""E03 phase 1: train fraction-error gates on the frozen F1 gate's validation cohort.

Reuse its features, parameters, fitting IDs and event-grouped folds. This entry
point never reads test predictions or runs a causal simulation.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

if __package__:
    from . import segmentation_gate as gate
else:
    import segmentation_gate as gate

DELTA = .02  # Two percentage points; fixed before E03 test evaluation.
KINDS = ("regression", "classification")
DISPLAY_FEATURES = [*gate.FEATURES[:2], "density x 1e8 (km^-2)"]


def validation_design(rows, protocol, sources, f1_dir):
    """Check the pinned baseline and recover exactly its original fit/held IDs."""
    if not rows or any(r["original_split"] != "validation" for r in rows):
        raise ValueError("E03 phase 1 accepts validation only.")
    paths = [Path(f1_dir) / name for name in ("gate_config.json", "gate_scores.csv")]
    baseline = gate.read_json(paths[0])
    expected_parameters = {"max_depth": 3, "min_samples_leaf": 5, "criterion": "gini",
                           "splitter": "best", "random_state": 0, "class_weight": None}
    if (baseline["schema_version"] != 1 or baseline["features"] != list(gate.FEATURES)
            or baseline["density_scale"] != gate.DENSITY_SCALE or baseline["failure_threshold"] != .6
            or baseline["fit_split"] != "validation" or baseline["parameters"] != expected_parameters
            or baseline["sklearn_version"] != sklearn.__version__
            or gate.cache.fingerprint(baseline["protocol"]) != gate.cache.fingerprint(protocol)):
        raise ValueError("F1 baseline parameters, features or data/encoder/KDE protocol differ.")
    old_sources = {Path(path).name: value for path, value in baseline["source_sha256"].items()}
    if any(old_sources.get(Path(path).name) != value for path, value in sources.items()):
        raise ValueError("Validation input checksums differ from the frozen F1 baseline.")
    lookup = {r["patch_id"]: i for i, r in enumerate(rows)}
    fit_ids = baseline["fit_patch_ids"]
    fit_set = set(fit_ids)
    if (len(lookup) != len(rows) or len(fit_set) != len(fit_ids)
            or fit_set != {r["patch_id"] for r in rows if r["failure"] is not None}):
        raise ValueError("Expected exactly the original F1-labelled fitting cohort.")
    fit = np.array([lookup[key] for key in fit_ids])
    previous = gate.cache.read_rows(paths[1], ("patch_id", "original_split", "score_kind", "oof_gate_score"))
    if len(previous) != len(rows) or {r["patch_id"] for r in previous} != set(lookup):
        raise ValueError("F1 score export does not cover the same validation patches.")
    f1_oof = np.full(len(rows), np.nan)
    for old in previous:
        if old["original_split"] != "validation" or old["score_kind"] != "validation_in_sample":
            raise ValueError("Expected the frozen validation F1 score export.")
        if old["patch_id"] in fit_set:
            f1_oof[lookup[old["patch_id"]]] = gate.number(old["oof_gate_score"])
    if not np.isfinite(f1_oof[fit]).all() or not ((f1_oof[fit] >= 0) & (f1_oof[fit] <= 1)).all():
        raise ValueError("F1 OOF scores must be finite probabilities on the fitting cohort.")
    folds, held_once = [], Counter()
    for saved in baseline["folds"]:
        train_ids, held_ids = saved["fit_patch_ids"], saved["held_patch_ids"]
        if (len(set(train_ids)) != len(train_ids) or len(set(held_ids)) != len(held_ids)
                or set(train_ids) & set(held_ids) or set(train_ids) | set(held_ids) != fit_set):
            raise ValueError("Invalid F1 fold patch membership.")
        train, held = [np.array([lookup[key] for key in keys]) for keys in (train_ids, held_ids)]
        train_events, held_events = [{rows[i]["event_id"] for i in indices} for indices in (train, held)]
        if (not len(train) or not len(held) or train_events & held_events
                or train_events != set(saved["fit_events"]) or held_events != set(saved["held_events"])):
            raise ValueError("Invalid F1 fold event separation.")
        folds.append((train, held))
        held_once.update(held_ids)
    if len(folds) < 2 or set(held_once) != fit_set or set(held_once.values()) != {1}:
        raise ValueError("Each original fitting patch must be held out exactly once.")
    return baseline, fit, folds, f1_oof, {str(p.resolve()): gate.cache.digest(p) for p in paths}


def retained_mean(scores, losses):
    """Exact expected retained loss for uniform ordering inside score ties."""
    scores, losses = np.asarray(scores), np.asarray(losses)
    if (scores.ndim != 1 or losses.shape != scores.shape or not len(scores)
            or not np.isfinite(scores).all() or not np.isfinite(losses).all() or (losses < 0).any()):
        raise ValueError("Expected equal nonempty finite scores/losses and nonnegative losses.")
    order = np.argsort(scores, kind="stable")
    values, losses = scores[order], losses[order]
    _, starts, counts = np.unique(values, return_index=True, return_counts=True)
    cumulative = np.empty(len(losses))
    previous = 0.
    for start, count in zip(starts, counts):
        total = losses[start:start + count].sum()
        cumulative[start:start + count] = previous + np.arange(1, count + 1) * total / count
        previous += total
    return cumulative / np.arange(1, len(losses) + 1)


def comparison(rows, fit, rankings, output):
    error = np.array([rows[i]["absolute_fraction_error"] for i in fit])
    rankings = {**rankings, "random_expected": np.zeros(len(error)), "oracle_diagnostic": error}
    exported, metrics = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = {"regression": "#2563eb", "regression_with_UE": "#922e87", "classification": "#c2410c", "f1_gate": "#15803d",
              "random_expected": "#64748b", "oracle_diagnostic": "black"}
    for name, scores in rankings.items():
        mae = retained_mean(scores, error)
        mse = retained_mean(scores, error ** 2)
        risk = gate.risk_curve(scores, (error > DELTA).astype(int))
        n = len(error)
        metrics[name] = {"mean_retained_MAE_over_coverages": float(mae.mean()),
                         "at_coverage": [{"target_coverage": cov, "retained": k, "actual_coverage": k / n,
                                          "MAE": float(mae[k-1]), "RMSE": float(np.sqrt(mse[k-1])),
                                          "fraction_error_failure_rate": float(risk[k-1])}
                                         for cov in (1., .75, .5) for k in [max(1, math.floor(cov * n))]]}
        exported.extend({"method": name, "n_evaluable": n, "retained": k, "coverage": k/n,
                         "MAE": float(mae[k-1]), "RMSE": float(np.sqrt(mse[k-1])),
                         "fraction_error_failure_rate": float(risk[k-1])} for k in range(1, n+1))
        style = "--" if name == "random_expected" else ":" if name == "oracle_diagnostic" else "-"
        for ax, values in zip(axes, (mae, np.sqrt(mse))):
            ax.plot(np.arange(1, n+1)/n, values*100, label=name, color=colors[name], ls=style)
    for ax, label in zip(axes, ("Retained fraction MAE (percentage points)", "Retained fraction RMSE (percentage points)")):
        ax.set(xlabel="Retained fraction of original fitting cohort", ylabel=label)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"E03 phase 1: event-grouped OOF, same {len(fit)} validation patches\nExact expectation over ties; RMSE = sqrt(expected retained squared error)")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_error_coverage.{extension}", dpi=170)
    plt.close(fig)
    gate.cache.write_index(output / "validation_error_coverage.csv", exported)
    return metrics


def save_tree(model, output, classification, feature_names=DISPLAY_FEATURES):
    (output / "tree_rules.txt").write_text(export_text(model, feature_names=feature_names, decimals=6)
                                          + "\nExact NaN routes are recorded in tree_structure.json.\n", encoding="utf-8")
    tree = model.tree_
    structure = [{"node": i, "feature": feature_names[tree.feature[i]] if tree.feature[i] >= 0 else None,
                  "threshold": float(tree.threshold[i]) if np.isfinite(tree.threshold[i]) else "inf",
                  "left": int(tree.children_left[i]), "right": int(tree.children_right[i]),
                  "missing_go_to_left": bool(tree.missing_go_to_left[i]), "n_fit_samples": int(tree.n_node_samples[i]),
                  "value": tree.value[i].ravel().tolist()} for i in range(tree.node_count)]
    gate.cache.write_json(output / "tree_structure.json", structure)
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_tree(model, feature_names=feature_names, class_names=["error <= 0.02", "error > 0.02"] if classification else None,
              filled=True, ax=ax, fontsize=8)
    fig.savefig(output / "tree.png", dpi=170)
    plt.close(fig)


def train(rows, protocol, sources, f1_dir, output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new/empty E03 output directory.")
    baseline, fit, folds, f1_oof, baseline_sources = validation_design(rows, protocol, sources, f1_dir)
    fit_set = set(fit)
    enriched = []
    for i, source in enumerate(rows):
        row = dict(source)
        row["f1_failure"], row["f1_label_status"] = row.pop("failure"), row.pop("label_status")
        row["absolute_fraction_error"] = abs(row["predicted_fraction"]-row["reference_fraction"])
        row["fraction_error_failure"] = int(row["absolute_fraction_error"] > DELTA)
        row["in_fit_cohort"] = i in fit_set
        enriched.append(row)
    error = np.array([r["absolute_fraction_error"] for r in enriched])
    labels = np.array([r["fraction_error_failure"] for r in enriched])
    if not np.isfinite(error).all() or (error > 1).any() or len(np.unique(labels[fit])) != 2:
        raise ValueError("Need finite valid fraction errors and both threshold-error classes in the fitting cohort.")
    values = gate.feature_matrix(enriched)
    output.mkdir(parents=True, exist_ok=True)
    ranks, scores_by_kind, model_hashes = {}, {}, {}
    for kind in KINDS:
        classification = kind == "classification"
        parameters = dict(baseline["parameters"])
        if not classification:
            parameters.pop("class_weight")
            parameters["criterion"] = "squared_error"
        model_class = DecisionTreeClassifier if classification else DecisionTreeRegressor
        target = labels if classification else error
        oof, fold_records = np.full(len(rows), np.nan), []
        for fold, (train_ids, held_ids) in enumerate(folds):
            model = model_class(**parameters).fit(values[train_ids], target[train_ids])
            oof[held_ids] = gate.failure_scores(model, values[held_ids]) if classification else model.predict(values[held_ids])
            fold_records.append({"fold": fold, "fit_patch_ids": [rows[i]["patch_id"] for i in train_ids],
                                 "held_patch_ids": [rows[i]["patch_id"] for i in held_ids],
                                 "fit_events": sorted({rows[i]["event_id"] for i in train_ids}),
                                 "held_events": sorted({rows[i]["event_id"] for i in held_ids}),
                                 "fit_fraction_error_class_counts": dict(Counter(map(int, labels[train_ids])))})
            print(f"{kind}: OOF fold {fold+1}/{len(folds)}; fit={len(train_ids)}, held={len(held_ids)}", flush=True)
        if not np.isfinite(oof[fit]).all():
            raise ValueError("Incomplete OOF scores on the original fitting cohort.")
        model = model_class(**parameters).fit(values[fit], target[fit])
        scores = gate.failure_scores(model, values) if classification else model.predict(values)
        directory = output / kind
        directory.mkdir()
        with (directory / "gate.pkl").open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        config = {"schema_version": 2, "experiment": "E03", "stage": "validation_only", "kind": kind,
                  "primary": not classification, "features": list(gate.FEATURES), "density_scale": gate.DENSITY_SCALE,
                  "target": "absolute_fraction_error > delta" if classification else "absolute_fraction_error",
                  "delta": DELTA if classification else None, "parameters": parameters, "protocol": protocol,
                  "fit_split": "validation", "fit_patch_ids": baseline["fit_patch_ids"], "folds": fold_records,
                  "fit_events": baseline["fit_events"], "f1_baseline_model_sha256": baseline["model_sha256"],
                  "source_sha256": {**sources, **baseline_sources}, "model_sha256": gate.cache.digest(directory / "gate.pkl"),
                  "implementation_sha256": gate.cache.digest(Path(__file__)),
                  "reused_gate_implementation_sha256": gate.cache.digest(Path(gate.__file__)),
                  "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                  "score_meaning": "uncalibrated P(absolute_fraction_error > delta)" if classification else "predicted mean absolute fraction error",
                  "missing_features": "native NaN routing; no imputation or row deletion",
                  "excluded_rows": "same original F1 fitting cohort; excluded undefined-F1 validation rows receive final scores only",
                  "selection_policy": "keep lowest score; exact tie expectation for validation; fixed 75/50% retention for later causal phase"}
        gate.cache.write_json(directory / "gate_config.json", config)
        gate.write_scores(directory / "gate_scores.csv", enriched, scores, "validation_in_sample", oof=oof)
        save_tree(model, directory, classification)
        ranks[kind], scores_by_kind[kind], model_hashes[kind] = oof[fit], oof, config["model_sha256"]
    ranks["f1_gate"] = f1_oof[fit]
    comparison_rows = [{**r, "f1_oof_score": float(f1_oof[i]) if np.isfinite(f1_oof[i]) else None,
                        **{kind+"_oof_score": float(scores_by_kind[kind][i]) if np.isfinite(scores_by_kind[kind][i]) else None
                           for kind in KINDS}} for i, r in enumerate(enriched)]
    gate.cache.write_index(output / "validation_scores.csv", [{k: None if isinstance(v, float) and not math.isfinite(v) else v
                                                             for k, v in r.items()} for r in comparison_rows])
    metrics = comparison(enriched, fit, ranks, output)
    summary = {"complete": True, "experiment": "E03", "stage": "validation_only", "n_validation": len(rows),
               "n_fit": len(fit), "n_excluded_from_fit_and_OOF": len(rows)-len(fit),
               "n_missing_UT_in_fit": int(np.isnan(values[fit, 0]).sum()),
               "delta": DELTA, "fraction_error_class_counts": dict(Counter(map(int, labels[fit]))),
               "primary": "regression", "secondary": "classification", "same_f1_cohort_and_event_folds": True,
               "f1_baseline_model_sha256": baseline["model_sha256"], "model_sha256": model_hashes,
               "source_sha256": {**sources, **baseline_sources}, "metrics": metrics,
               "tie_policy": "uniform ordering inside equal-score groups, exact expected retained loss",
               "RMSE_convention": "sqrt(expected retained squared fraction error), not expected within-subset RMSE",
               "limitations": "OOF development on validation previously used for decoder selection; no independent test, new-event or causal conclusion",
               "next_stage": "review both frozen gates before separate exploratory test/causal evaluation; no automatic phase 2"}
    gate.cache.write_json(output / "training_summary.json", summary)
    print(f"E03 phase 1 complete: {len(fit)}/{len(rows)} validation patches fitted; outputs: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True, help="Original complete validation prediction export.")
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--f1-gate-dir", type=Path, default=Path("models/reliability_gate/three_features_seed0"))
    parser.add_argument("--output", type=Path, required=True, help="New/empty E03 model directory.")
    args = parser.parse_args()
    try:
        rows, protocol, sources = gate.load_inputs(args.predictions_dir, args.density_dir, "validation")
        train(rows, protocol, sources, args.f1_gate_dir, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"E03 phase 1 failed: {error}\n")


if __name__ == "__main__":
    main()
