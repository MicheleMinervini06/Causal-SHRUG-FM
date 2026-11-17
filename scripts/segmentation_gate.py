"""Train a three-feature segmentation-failure tree, or evaluate its frozen scores.

Fit on validation only; event-grouped out-of-fold scores are development checks.
Evaluate explicitly on test or Bolivia. No encoder, masks or causal estimators
are loaded. Undefined F1 stays unlabelled; missing U_T stays NaN for tree routing.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

if __package__:
    from . import cache_encoder_features as cache
else:
    import cache_encoder_features as cache

FEATURES = ("average_entropy", "mutual_information", "pretraining_density")
DENSITY_SCALE = 1e8  # Our km^-2 values would otherwise fall below CART's 1e-7 feature tolerance.
FAILURE_THRESHOLD = .6
MAX_DEPTH = 3
COLORS = {"gate": "#2563eb", "average_entropy": "#15803d", "mutual_information": "#9333ea",
          "spatial_rarity": "#dc2626", "normalized_distance": "#b45309", "negative_NCDD": "#0891b2",
          "random_expected": "#64748b", "oracle_diagnostic": "black"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, *, missing=False):
    if missing and value in (None, ""):
        return float("nan")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Numeric inputs must be finite; use an empty field for missing scores.")
    return result


def load_inputs(predictions_dir, density_dir, split, embedding_dir=None):
    """Join by verified IDs and match preprocessing/support/ensemble provenance."""
    predictions_dir, density_dir = Path(predictions_dir), Path(density_dir)
    paths = [predictions_dir / n for n in ("predictions.csv", "prediction_config.json", "prediction_summary.json")]
    paths += [density_dir / n for n in ("spatial_density.csv", "density_config.json", "density_summary.json")]
    config, summary = read_json(paths[1]), read_json(paths[2])
    density_config, density_summary = read_json(paths[4]), read_json(paths[5])
    predictions = cache.read_rows(paths[0], ("patch_id", "event_id", "original_split", "reference_fraction", "predicted_fraction",
                                            "F1", "n_valid_pixels", "average_entropy", "mutual_information", "roi_empty",
                                            "threshold", "roi_threshold", "model_seeds"))
    if (not summary["complete"] or summary["limited"] or config["limit"] is not None
            or len(predictions) != summary["patches"] or len(predictions) != summary["requested_patches"]
            or dict(Counter(r["original_split"] for r in predictions)) != summary["counts_by_split"]):
        raise ValueError("Predictions must be complete, un-limited exports with consistent split counts.")
    if split == "validation" and any(r["original_split"] != split for r in predictions):
        raise ValueError("Gate fitting accepts a validation-only prediction export.")
    if config["threshold"] != .5 or config["roi_threshold"] != .05:
        raise ValueError("This baseline requires segmentation threshold 0.5 and U_T ROI threshold 0.05.")
    provenance = config["data_provenance"]
    if any(provenance[k] != density_config["data_provenance"][k]
           for k in ("manifest_sha256", "inspection_sha256", "reference_support")):
        raise ValueError("Density and prediction dataset/audit/valid-support identities differ.")
    if (density_config["intended_encoder_sha256"] != provenance["encoder_sha256"]
            or not density_summary["complete"] or density_summary["limited"]
            or cache.digest(paths[3]) != density_summary["features_sha256"]):
        raise ValueError("Invalid/incomplete density export or encoder mismatch.")
    densities = cache.read_rows(paths[3], ("patch_id", "event_id", "original_split", "pretraining_density", "pretraining_log_density"))
    density_lookup = {r["patch_id"]: r for r in densities}
    if len(densities) != density_summary["patches"]:
        raise ValueError("Density row count disagrees with its summary.")
    embeddings = None
    if embedding_dir is not None:
        embedding_dir = Path(embedding_dir)
        extras = [embedding_dir / n for n in ("embedding_uncertainty.csv", "embedding_config.json", "embedding_summary.json")]
        embedding_config, embedding_summary = read_json(extras[1]), read_json(extras[2])
        if (embedding_config["downstream_metadata"]["cache_fingerprint"] != provenance["cache_fingerprint"]
                or cache.digest(extras[0]) != embedding_summary["scores_sha256"]):
            raise ValueError("Embedding and prediction caches differ or embedding CSV checksum failed.")
        embedding_rows = cache.read_rows(extras[0], ("patch_id", "event_id", "original_split", "normalized_distance", "NCDD"))
        if len(embedding_rows) != embedding_summary["n_patches"]:
            raise ValueError("Embedding row count disagrees with its summary.")
        embeddings = {r["patch_id"]: r for r in embedding_rows}
        paths += extras
    members = sorted((int(m["model_seed"]), m["sha256"]) for m in config["members"])
    if len(members) < 2 or len({s for s, _ in members}) != len(members) or len({h for _, h in members}) != len(members):
        raise ValueError("Expected distinct segmentation ensemble members.")
    rows = []
    for source in predictions:
        if source["original_split"] != split:
            continue
        key = source["patch_id"]
        row = {k: source[k] for k in ("patch_id", "event_id", "original_split")}
        for k in ("reference_fraction", "predicted_fraction"):
            row[k] = number(source[k])
            if not 0 <= row[k] <= 1:
                raise ValueError(f"Invalid fraction: {key}")
        row["n_valid_pixels"] = int(source["n_valid_pixels"])
        row["model_seeds"] = source["model_seeds"]
        if row["n_valid_pixels"] < 1 or source["roi_empty"] not in ("True", "False"):
            raise ValueError(f"Invalid valid support/ROI flag: {key}")
        if (number(source["threshold"]) != .5 or number(source["roi_threshold"]) != .05
                or sorted(int(s) for s in source["model_seeds"].split(";")) != [s for s, _ in members]):
            raise ValueError(f"Row/ensemble threshold or member identity mismatch: {key}")
        row["roi_empty"] = source["roi_empty"] == "True"
        for k in FEATURES[:2]:
            row[k] = number(source[k], missing=True)
            if np.isnan(row[k]) != row["roi_empty"] or (np.isfinite(row[k]) and row[k] < 0):
                raise ValueError(f"U_T missingness must agree with the ROI flag: {key}")
        row["F1"] = number(source["F1"], missing=True)
        if np.isnan(row["F1"]):
            if row["reference_fraction"] != 0 or row["predicted_fraction"] != 0:
                raise ValueError(f"Undefined F1 on nonempty foreground masks: {key}")
            row["failure"], row["label_status"] = None, "undefined_F1_both_masks_empty"
        else:
            if not 0 <= row["F1"] <= 1:
                raise ValueError(f"Invalid F1: {key}")
            row["failure"], row["label_status"] = int(row["F1"] < FAILURE_THRESHOLD), "defined"
        for lookup, fields in ((density_lookup, ("pretraining_density", "pretraining_log_density")),
                               (embeddings, ("normalized_distance", "NCDD"))):
            if lookup is None:
                continue
            extra = lookup.get(key)
            if extra is None or any(extra[k] != row[k] for k in ("event_id", "original_split")):
                raise ValueError(f"Missing/mismatched joined feature identity: {key}")
            row.update({k: number(extra[k]) for k in fields})
        if (row["pretraining_density"] <= 0
                or not math.isclose(math.log(row["pretraining_density"]), row["pretraining_log_density"], abs_tol=1e-10, rel_tol=1e-12)):
            raise ValueError(f"Invalid density/log-density pair: {key}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows for requested split {split}.")
    protocol = {"data_provenance": provenance, "ensemble": members, "threshold": config["threshold"],
                "roi_threshold": config["roi_threshold"], "density_kde": density_config["kde"],
                "density_reference_sha256": density_config["pretraining"]["sha256"]}
    return rows, protocol, {str(p.resolve()): cache.digest(p) for p in paths}


def feature_matrix(rows):
    # Labels, C, C_hat, event IDs and splits never enter the model's feature matrix.
    values = np.asarray([[r[k] for k in FEATURES] for r in rows], dtype=np.float64)
    values[:, 2] *= DENSITY_SCALE
    return values


def failure_scores(tree, values):
    probabilities = tree.predict_proba(values)
    positive = np.flatnonzero(tree.classes_ == 1)
    return probabilities[:, positive[0]] if len(positive) else np.zeros(len(values))


def risk_curve(scores, failure):
    """Exact expected risk for uniform ordering within equal-score groups."""
    scores, failure = np.asarray(scores, dtype=float), np.asarray(failure)
    if (scores.ndim != 1 or failure.shape != scores.shape or not len(scores)
            or not np.isfinite(scores).all() or not np.isin(failure, [0, 1]).all()):
        raise ValueError("Risk curves require matching nonempty finite scores and binary labels.")
    order = np.argsort(scores, kind="stable")
    values, labels = scores[order], failure[order]
    _, starts, counts = np.unique(values, return_index=True, return_counts=True)
    expected = np.empty(len(labels), dtype=float)
    previous = 0.
    for start, count in zip(starts, counts):
        failures = labels[start:start + count].sum()
        expected[start:start + count] = previous + np.arange(1, count + 1) * failures / count
        previous += failures
    return expected / np.arange(1, len(labels) + 1)


def write_scores(path, rows, scores, kind, oof=None):
    saved = []
    for i, row in enumerate(rows):
        clean = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
        saved.append({**clean, "gate_score": float(scores[i]), "score_kind": kind,
                      "oof_gate_score": None if oof is None or not np.isfinite(oof[i]) else float(oof[i])})
    cache.write_index(path, saved)


def evaluate_curves(rows, scores, output, title):
    labels = np.asarray([r["failure"] if r["failure"] is not None else np.nan for r in rows])
    labelled = np.isfinite(labels)
    if not labelled.any():
        raise ValueError("No defined F1 labels for risk evaluation.")
    methods = {"gate": np.asarray(scores),
               "average_entropy": np.array([r["average_entropy"] for r in rows]),
               "mutual_information": np.array([r["mutual_information"] for r in rows]),
               "spatial_rarity": -np.array([r["pretraining_log_density"] for r in rows])}
    if "normalized_distance" in rows[0]:
        methods.update(normalized_distance=np.array([r["normalized_distance"] for r in rows]),
                       negative_NCDD=-np.array([r["NCDD"] for r in rows]))
    common = labelled & np.logical_and.reduce([np.isfinite(v) for v in methods.values()])
    cohorts = {"all_labelled": (labelled, {"gate": methods["gate"]}), "common_baselines": (common, methods)}
    exported, result = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, (name, (mask, compared)) in zip(axes, cohorts.items()):
        n = int(mask.sum())
        if n == 0:
            result[name] = {"n": 0, "status": "no_common_complete_scores"}
            ax.set_title(f"{name}: no eligible rows")
            continue
        y, coverage = labels[mask].astype(int), np.arange(1, n + 1) / n
        curves = {k: risk_curve(v[mask], y) for k, v in compared.items()}
        curves["random_expected"] = np.full(n, y.mean())
        curves["oracle_diagnostic"] = risk_curve(y, y)
        metrics = {}
        for method, curve in curves.items():
            metrics[method] = {"AURC": float(curve.mean()),
                               "risk_at_coverage": [{"target_coverage": c, "retained": k, "actual_coverage": k / n,
                                                     "risk": float(curve[k - 1])}
                                                    for c in (.5, .75, .9) for k in [max(1, int(np.floor(c * n)))]]}
            exported.extend({"cohort": name, "method": method, "n_evaluable": n, "retained": k,
                             "coverage": k / n, "risk": float(value)} for k, value in enumerate(curve, 1))
            ax.plot(coverage, curve, label=method.replace("_", " "), color=COLORS[method], lw=2 if method == "gate" else 1.2,
                    ls="--" if method == "random_expected" else ":" if method == "oracle_diagnostic" else "-")
        result[name] = {"n": n, "n_failures": int(y.sum()), "failure_rate": float(y.mean()), "methods": metrics,
                        "patch_ids": [r["patch_id"] for r, keep in zip(rows, mask) if keep]}
        ax.set(xlabel="Retained fraction of evaluable patches", ylabel="Failure risk (F1 < 0.60)", ylim=(-.02, 1.02),
               title=f"{name.replace('_', ' ')}: n={n}")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title + "\nCoverage is conditional on defined F1; ties averaged analytically", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"risk_coverage.{extension}", dpi=170)
    plt.close(fig)
    cache.write_index(output / "risk_coverage.csv", exported)
    return {"n_input_patches": len(rows), "n_labelled": int(labelled.sum()), "n_undefined_F1": int((~labelled).sum()),
            "n_missing_UT": sum(r["roi_empty"] for r in rows), "cohorts": result,
            "AURC_convention": "mean risk at k/n, k=1..n (right-endpoint discrete integral)",
            "tie_policy": "uniform retention within equal scores; exact expectation, equivalent to infinite random permutations",
            "random_policy": "uniform random retention in expectation on each identical cohort; no confidence interval implied"}


def train(rows, protocol, sources, output, *, seed=0, folds=5, min_samples_leaf=5):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty gate output directory.")
    if any(r["original_split"] != "validation" for r in rows):
        raise ValueError("Fit only on validation; training/test/Bolivia labels are forbidden.")
    if not 0 <= seed < 2**32 or min_samples_leaf < 1:
        raise ValueError("Invalid seed or minimum leaf size.")
    labelled = np.array([r["failure"] is not None for r in rows])
    indices = np.flatnonzero(labelled)
    y = np.array([rows[i]["failure"] for i in indices])
    groups = np.array([rows[i]["event_id"] for i in indices])
    if len(np.unique(y)) != 2 or not 2 <= folds <= len(set(groups)):
        raise ValueError("Need both failure classes and at least n_splits labelled event groups.")
    values = feature_matrix(rows)
    parameters = {"max_depth": MAX_DEPTH, "min_samples_leaf": min_samples_leaf, "criterion": "gini",
                  "splitter": "best", "random_state": seed, "class_weight": None}
    oof, fold_records = np.full(len(rows), np.nan), []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(values[indices], y, groups)):
        fit_ids, held_ids = indices[fit_indices], indices[held_indices]
        model = DecisionTreeClassifier(**parameters).fit(values[fit_ids], y[fit_indices])
        oof[held_ids] = failure_scores(model, values[held_ids])
        fit_events, held_events = sorted(set(groups[fit_indices])), sorted(set(groups[held_indices]))
        assert not set(fit_events) & set(held_events)
        fold_records.append({"fold": fold, "fit_events": fit_events, "held_events": held_events,
                             "fit_patch_ids": [rows[i]["patch_id"] for i in fit_ids],
                             "held_patch_ids": [rows[i]["patch_id"] for i in held_ids],
                             "fit_class_counts": dict(Counter(map(int, y[fit_indices])))})
        print(f"Gate OOF fold {fold + 1}/{folds}: fit={len(fit_ids)}, held={len(held_ids)}", flush=True)
    assert np.isfinite(oof[labelled]).all()
    model = DecisionTreeClassifier(**parameters).fit(values[indices], y)
    scores = failure_scores(model, values)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gate.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    configuration = {"schema_version": 1, "features": list(FEATURES), "density_scale": DENSITY_SCALE,
                     "failure_threshold": FAILURE_THRESHOLD, "parameters": parameters, "folds": fold_records,
                     "fit_split": "validation", "fit_patch_ids": [rows[i]["patch_id"] for i in indices],
                     "fit_events": sorted(set(groups)), "protocol": protocol, "source_sha256": sources,
                     "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                     "model_sha256": cache.digest(output / "gate.pkl"), "implementation_sha256": cache.digest(Path(__file__)),
                     "missing_features": "native NaN routing in DecisionTreeClassifier; no zero fill or row deletion",
                     "undefined_F1": "both masks empty: exclude from fit/risk denominators, preserve row and frozen gate score",
                     "score_meaning": "leaf failure frequency; uncalibrated ranking score",
                     "selection": "fixed three-feature baseline; no RFE, hyperparameter search or test-based tuning",
                     "differences_from_SHRUG_FM": "Sen1Floods11; validation-only fit; grouped OOF; minimum leaf size; partial U_I; native NaN handling; exact expected tie curves"}
    cache.write_json(output / "gate_config.json", configuration)
    write_scores(output / "gate_scores.csv", rows, scores, "validation_in_sample", oof=oof)
    cache.write_json(output / "evaluation_summary.json", evaluate_curves(rows, oof, output, "Validation event-grouped OOF (development only)"))
    display_names = [*FEATURES[:2], "density x 1e8 (km^-2)"]
    rules = export_text(model, feature_names=display_names, decimals=6)
    (output / "tree_rules.txt").write_text(rules + "\nNaN routing is specified per internal node in tree_structure.json.\n", encoding="utf-8")
    structure = []
    for node in range(model.tree_.node_count):
        feature = int(model.tree_.feature[node])
        structure.append({"node": node, "feature": display_names[feature] if feature >= 0 else None,
                          "threshold": float(model.tree_.threshold[node]) if math.isfinite(model.tree_.threshold[node]) else "inf",
                          "left": int(model.tree_.children_left[node]), "right": int(model.tree_.children_right[node]),
                          "missing_go_to_left": bool(model.tree_.missing_go_to_left[node]),
                          "n_fit_samples": int(model.tree_.n_node_samples[node])})
    cache.write_json(output / "tree_structure.json", structure)
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_tree(model, feature_names=display_names, class_names=["success", "failure"], filled=True, ax=ax, fontsize=8)
    fig.savefig(output / "tree.png", dpi=170)
    plt.close(fig)
    print(f"Gate fitted on {len(indices)} labelled validation patches; scored all {len(rows)}. Saved: {output}", flush=True)
    return configuration


def evaluate(rows, protocol, sources, gate_dir, output):
    gate_dir, output = Path(gate_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty evaluation directory.")
    config = read_json(gate_dir / "gate_config.json")
    if (config["schema_version"] != 1 or config["features"] != list(FEATURES)
            or config["density_scale"] != DENSITY_SCALE or config["failure_threshold"] != FAILURE_THRESHOLD):
        raise ValueError("Incompatible gate feature/schema contract.")
    # JSON stores member pairs as lists, so compare canonical JSON, not tuple types.
    if cache.fingerprint(protocol) != cache.fingerprint(config["protocol"]):
        raise ValueError("Gate and evaluation ensemble/dataset/support/KDE protocols differ.")
    if config["sklearn_version"] != sklearn.__version__ or cache.digest(gate_dir / "gate.pkl") != config["model_sha256"]:
        raise ValueError("Gate checksum or scikit-learn version mismatch.")
    splits = {r["original_split"] for r in rows}
    if len(splits) != 1 or not splits.issubset({"test", "bolivia_holdout"}):
        raise ValueError("Evaluate one held-out split at a time: test or bolivia_holdout.")
    if {r["patch_id"] for r in rows} & set(config["fit_patch_ids"]):
        raise ValueError("Evaluation overlaps fitted gate patch IDs.")
    with (gate_dir / "gate.pkl").open("rb") as stream:
        model = pickle.load(stream)
    scores = failure_scores(model, feature_matrix(rows))
    output.mkdir(parents=True, exist_ok=True)
    write_scores(output / "gate_scores.csv", rows, scores, "held_out")
    result = evaluate_curves(rows, scores, output, "Held-out " + next(iter(splits)))
    result.update(split=next(iter(splits)), source_sha256=sources, gate_model_sha256=config["model_sha256"],
                  overlapping_fit_event_ids=sorted(set(r["event_id"] for r in rows) & set(config["fit_events"])),
                  evaluation_note="Original main splits share events; held-out patches do not establish unseen-event generalization.")
    cache.write_json(output / "evaluation_summary.json", result)
    print(f"Held-out gate evaluation complete: {len(rows)} scored patches; {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--predictions-dir", type=Path, required=True)
        command.add_argument("--density-dir", type=Path, required=True)
        command.add_argument("--embedding-dir", type=Path, help="Optional U_E join/baselines; never used as tree inputs.")
        command.add_argument("--output", type=Path, required=True)
        if name == "train":
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--folds", type=int, default=5)
            command.add_argument("--min-samples-leaf", type=int, default=5)
        else:
            command.add_argument("--gate-dir", type=Path, required=True)
            command.add_argument("--split", choices=("test", "bolivia_holdout"), default="test")
    args = parser.parse_args()
    print(f"Gate {args.command}: reading exported patch tables; CPU only.", flush=True)
    split = "validation" if args.command == "train" else args.split
    rows, protocol, sources = load_inputs(args.predictions_dir, args.density_dir, split, args.embedding_dir)
    if args.command == "train":
        train(rows, protocol, sources, args.output, seed=args.seed, folds=args.folds, min_samples_leaf=args.min_samples_leaf)
    else:
        evaluate(rows, protocol, sources, args.gate_dir, args.output)


if __name__ == "__main__":
    main()
