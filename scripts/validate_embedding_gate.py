"""E03 validation: compare frozen three-feature OOF scores with a five-feature regressor."""

import argparse
from pathlib import Path
import pickle

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

if __package__:
    from . import train_fraction_error_gates as phase1
else:
    import train_fraction_error_gates as phase1

gate = phase1.gate
UE = ("normalized_distance", "NCDD")


def validate(rows, protocol, sources, f1_dir, prior_dir, output):
    if output.exists():
        raise ValueError("Use a new U_E validation output directory.")
    baseline, fit, folds, _, f1_sources = phase1.validation_design(rows, protocol, sources, f1_dir)
    prior_path = prior_dir / "training_summary.json"
    prior = gate.read_json(prior_path)
    if (not prior["complete"] or prior["stage"] != "validation_only" or prior["primary"] != "regression"
            or prior["f1_baseline_model_sha256"] != baseline["model_sha256"]):
        raise ValueError("Expected the completed original E03 validation comparison.")
    previous_sources = {Path(k).name: v for k, v in prior["source_sha256"].items()}
    if any(previous_sources.get(Path(k).name) != v for k, v in f1_sources.items()):
        raise ValueError("F1 cohort or folds differ from the E03 validation inputs.")
    scores_path = prior_dir / "validation_scores.csv"
    archived = {r["patch_id"]: r for r in gate.cache.read_rows(scores_path, ("regression_oof_score", "in_fit_cohort"))}
    if set(archived) != {r["patch_id"] for r in rows}:
        raise ValueError("E03 validation patch IDs differ.")
    base = gate.feature_matrix(rows)
    extended = np.column_stack((base, [[gate.number(r[k]) for k in UE] for r in rows]))
    errors = np.array([abs(r["predicted_fraction"] - r["reference_fraction"]) for r in rows])
    old = np.full(len(rows), np.nan)
    new = old.copy()
    fit_set = set(fit)
    for i, r in enumerate(rows):
        saved = archived[r["patch_id"]]
        if (saved["original_split"] != "validation" or saved["event_id"] != r["event_id"]
                or saved["in_fit_cohort"] != str(i in fit_set)
                or float(saved["absolute_fraction_error"]) != errors[i]):
            raise ValueError("E03 validation identity, target or cohort mismatch.")
        old[i] = gate.number(saved["regression_oof_score"], missing=True)
    if not np.array_equal(np.isfinite(old), np.isin(np.arange(len(rows)), fit)):
        raise ValueError("Archived OOF scores must cover exactly the original fitting cohort.")
    params = {k: v for k, v in baseline["parameters"].items() if k != "class_weight"}
    params["criterion"] = "squared_error"
    model = DecisionTreeRegressor(**params)
    for train_ids, held in folds:
        model.fit(base[train_ids], errors[train_ids])
        if not np.allclose(model.predict(base[held]), old[held], rtol=1e-12, atol=1e-14):
            raise ValueError("Three-feature fold predictions differ from the archived OOF scores.")
        model.fit(extended[train_ids], errors[train_ids])
        new[held] = model.predict(extended[held])
    model.fit(extended[fit], errors[fit])
    output.mkdir(parents=True)
    model_path = output / "gate.pkl"
    model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    phase1.save_tree(model, output, False, [*phase1.DISPLAY_FEATURES, *UE])
    enriched = [{**r, "absolute_fraction_error": error} for r, error in zip(rows, errors)]
    metrics = phase1.comparison(enriched, fit, {"regression": old[fit], "regression_with_UE": new[fit]}, output)
    final_scores = model.predict(extended)
    gate.cache.write_index(output / "validation_scores.csv", [
        {**archived[r["patch_id"]], **{k: r[k] for k in UE},
         "regression_with_UE_oof_score": new[i] if i in fit_set else None,
         "regression_with_UE_final_score": final_scores[i]} for i, r in enumerate(rows)])
    model_hash = gate.cache.digest(model_path)
    gate.cache.write_json(output / "gate_config.json", {
        "schema_version": 3, "experiment": "E03", "kind": "regression_with_UE", "target": "absolute_fraction_error",
        "features": [*gate.FEATURES, *UE], "density_scale": gate.DENSITY_SCALE, "protocol": protocol,
        "parameters": params, "fit_split": "validation", "fit_patch_ids": baseline["fit_patch_ids"],
        "folds": baseline["folds"], "model_sha256": model_hash, "sklearn_version": sklearn.__version__,
        "source_sha256": {**sources, **f1_sources, **{str(p.resolve()): gate.cache.digest(p) for p in (prior_path, scores_path)}},
        "implementation_sha256": gate.cache.digest(Path(__file__)), "three_feature_model_sha256": prior["model_sha256"]["regression"]})
    summary = {"complete": True, "stage": "validation_only", "model_sha256": model_hash,
               "n_validation": len(rows), "n_fit": len(fit), "n_folds": len(folds), "baseline_OOF_verified": True,
               "metrics": metrics, "tie_policy": prior["tie_policy"], "RMSE_convention": prior["RMSE_convention"]}
    gate.cache.write_json(output / "training_summary.json", summary)
    print(f"U_E validation complete: {len(fit)} fitting patches, {len(folds)} folds. Frozen model: {model_path}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/segmentation_uncertainty/array-44157060/job-44161703"))
    parser.add_argument("--density-dir", type=Path, default=Path("results/input_uncertainty/spatial_density/h-100km-job-44166500"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("results/embedding_uncertainty/prefix-n5000-job-44198535"))
    parser.add_argument("--f1-dir", type=Path, default=Path("models/reliability_gate/three_features_seed0"))
    parser.add_argument("--prior-dir", type=Path, default=Path("models/reliability_gate/e03_fraction_error_seed0"))
    parser.add_argument("--output", type=Path, default=Path("models/reliability_gate/e03_regression_with_ue_seed0"))
    args = parser.parse_args()
    try:
        rows, protocol, sources = gate.load_inputs(args.predictions_dir, args.density_dir, "validation", args.embedding_dir)
        validate(rows, protocol, sources, args.f1_dir, args.prior_dir, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"U_E validation failed: {error}\n")


if __name__ == "__main__":
    main()
