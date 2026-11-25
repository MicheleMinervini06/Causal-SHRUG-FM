"""E05 phase 1: Bayesian calibration of validation surface-water fractions.

Run with uv run python -m experiments.fraction_calibration. The frozen archive
stores joint coefficient draws (one row per draw, shared across patches).
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, logsumexp
from scipy.stats import multivariate_t

from experiments import semisynthetic_confounder as e02
from scripts.segmentation_gate import FEATURES, number


INPUT_HASHES = {
    "gate_config.json": "79dc4f3a0cbcb61ac12afc5070c2966eeda63fbce09e25140685cb9eaf08a99b",
    "gate_scores.csv": "a53e24a4cb327eb53470e43623ded59dd97e01cf820a5d3075b8f6dd91bda5bc",
}
MODELS = ("proxy_only", "with_scores")
N_DRAWS = 8192


def load_validation(directory):
    for name, sha in INPUT_HASHES.items():
        if e02.digest(directory / name) != sha:
            raise ValueError(f"Expected the reviewed validation export: {name}")
    rows = e02.read_rows(directory / "gate_scores.csv")
    folds = json.loads((directory / "gate_config.json").read_text())["folds"]
    event_fold = {}
    for i, fold in enumerate(folds):
        for event in fold["held_events"]:
            if event in event_fold or event in fold["fit_events"]:
                raise ValueError(f"Invalid event partition: {event}")
            event_fold[event] = i
    if any(r["original_split"] != "validation" or int(r["n_valid_pixels"]) <= 0 for r in rows):
        raise ValueError("Positive valid support and validation-only inputs required.")
    reference = np.array([number(r["reference_fraction"]) for r in rows])
    proxy = np.array([number(r["predicted_fraction"]) for r in rows])
    scores = np.array([[number(r[k], missing=True) for k in FEATURES] for r in rows])
    if np.any((reference < 0) | (reference >= 1)) or np.any((proxy < 0) | (proxy > 1)):
        raise ValueError("Expected C in [0,1) and predicted fractions in [0,1].")
    return rows, reference, proxy, scores, np.array([event_fold[r["event_id"]] for r in rows])


def design(proxy, scores, center, scale):
    """Apply training-fold scaling; zero predictions are valid inputs."""
    values = np.column_stack((np.log1p(100 * proxy), scores))
    return np.column_stack((np.ones(len(proxy)), (values - center) / scale))


def beta_shapes(parameters, x):
    mean_coef, precision_coef = np.split(parameters, 2, axis=-1)
    eta = mean_coef @ x.T
    precision = np.exp(precision_coef @ x.T)
    return expit(eta) * precision, expit(-eta) * precision


def posterior(x, y, component, rng):
    """Correct a mode-based multivariate Student proposal by importance weighting."""
    d = x.shape[1]
    if component == "presence":
        loc = np.zeros(d)
        sd = np.array([2., 1.5])
    else:
        loc = np.zeros(2 * d)
        loc[0], loc[d] = -2., np.log(20.)
        sd = np.r_[2., 1.5, np.full(d - 2, .5), 1.5, 1., np.full(d - 2, .5)]
        log_y, log_one_minus_y = np.log(y), np.log1p(-y)

    def log_target(theta):
        prior = -.5 * np.sum(((theta - loc) / sd) ** 2, axis=-1)
        if component == "presence":
            signed = (theta @ x.T) * (1 - 2 * y)
            return prior - np.logaddexp(0, signed).sum(axis=-1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            a, b = beta_shapes(theta, x)
            ll = (a - 1) * log_y + (b - 1) * log_one_minus_y - betaln(a, b)
            target = prior + ll.sum(axis=-1)
        return np.where(np.isfinite(target), target, -np.inf)

    def gradient(theta):
        prior = (theta - loc) / sd**2
        if component == "presence":
            return prior + (expit(x @ theta) - y) @ x
        a, b = beta_shapes(theta, x)
        phi = a + b
        da = log_y - digamma(a) + digamma(phi)
        db = log_one_minus_y - digamma(b) + digamma(phi)
        return prior - np.r_[((a * b / phi) * (da - db)) @ x, (a * da + b * db) @ x]

    # Compute curvature at the actual mode; BFGS's last update can be inaccurate.
    fit = minimize(lambda t: -log_target(t), loc, jac=gradient, method="BFGS", options={"gtol": 1e-4})
    if not fit.success:
        raise ValueError(f"{component} posterior mode failed: {fit.message}")
    step = 1e-4
    hessian = np.array([(gradient(fit.x + v) - gradient(fit.x - v)) / (2 * step)
                        for v in step * np.eye(len(loc))])
    proposal = multivariate_t(loc=fit.x, shape=np.linalg.inv((hessian + hessian.T) / 2), df=7)
    candidates = proposal.rvs(size=2 * N_DRAWS, random_state=rng)
    log_w = np.concatenate([log_target(chunk) for chunk in np.array_split(candidates, 16)])
    log_w -= proposal.logpdf(candidates)
    weights = np.exp(log_w - logsumexp(log_w))
    ess = float(1 / (weights @ weights))
    if not np.isfinite(ess) or ess < 500:
        raise ValueError(f"Insufficient {component} importance-sampling effective sample size: {ess}")
    return candidates[rng.choice(len(weights), N_DRAWS, p=weights)], {
        "importance_ESS": ess, "maximum_weight": float(weights.max()), "n_fit": len(y)}


def fit_models(reference, proxy, scores, seed, *, positive_only=False):
    raw = np.column_stack((np.log1p(100 * proxy), scores))
    center, scale = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("A calibration predictor has no usable training-fold variation.")
    x = design(proxy, scores, center, scale)
    wet = reference > 0
    if positive_only and not np.all(wet & (reference < 1)):
        raise ValueError("Positive-only beta calibration requires reference fractions strictly inside (0,1).")
    scored_wet = wet & np.isfinite(x).all(axis=1)
    components = (("presence", x[:, :2], wet.astype(float)),
                  ("proxy_only", x[wet, :2], reference[wet]),
                  ("with_scores", x[scored_wet], reference[scored_wet]))
    model, diagnostics = {"center": center, "scale": scale}, {}
    for i, (name, values, y) in enumerate(components):
        if name == "presence" and positive_only:
            model["positive_only"] = np.array(True)
            continue
        rng = np.random.default_rng([seed, i])
        model[name], diagnostics[name] = posterior(values, y, name, rng)
    return model, diagnostics


def predictive_components(model, proxy, scores, method):
    """One patch: aligned posterior draws of Pr(C>0), alpha, beta, for Phase 2."""
    x = design(np.array([proxy]), np.asarray(scores).reshape(1, -1), model["center"], model["scale"])[0]
    present = (np.ones(len(model["proxy_only"])) if model.get("positive_only", False)
               else expit(model["presence"] @ x[:2]))
    if method not in MODELS:
        raise ValueError(f"Unknown calibration model: {method}")
    if method == "proxy_only" or not np.isfinite(x).all():
        return present, *beta_shapes(model["proxy_only"], x[:2])
    return present, *beta_shapes(model["with_scores"], x)


def predict(model, proxy, scores, y, method, seed):
    present, a, b = predictive_components(model, proxy, scores, method)
    rng = np.random.default_rng(seed)
    samples = np.where(rng.random(len(a)) < present, rng.beta(a, b), 0.)
    lower, upper = np.quantile(samples, [.025, .975])
    if y == 0:
        log_score = np.log1p(-present.mean())
    else:
        log_density = np.log(present) + (a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - betaln(a, b)
        log_score = logsumexp(log_density) - np.log(len(a))
    return {"probability_positive": float(present.mean()), "predicted_mean": float(np.mean(present * a / (a + b))),
            "lower95": float(lower), "upper95": float(upper), "log_predictive_score": float(log_score)}


def metrics(rows):
    c = np.array([r["reference_fraction"] for r in rows])
    error = np.array([r["predicted_mean"] for r in rows]) - c
    lo, hi = (np.array([r[k] for r in rows]) for k in ("lower95", "upper95"))
    return {"n": len(c), "MAE": float(np.mean(abs(error))), "RMSE": float(np.sqrt(np.mean(error**2))),
            "coverage95": float(np.mean((lo <= c) & (c <= hi))), "mean_width95": float(np.mean(hi - lo)),
            "mean_log_predictive_score": float(np.mean([r["log_predictive_score"] for r in rows])),
            "presence_brier": float(np.mean((np.array([r["probability_positive"] for r in rows]) - (c > 0))**2))}


def run(directory, output, seed):
    data = load_validation(directory)
    return fit_validation(data, output, seed, source_hashes=INPUT_HASHES)


def fit_validation(data, output, seed, *, source_hashes, burn_protocol=None):
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output}")
    rows, c, proxy, scores, folds = data
    positive_only = burn_protocol is not None
    records, diagnostics = [], {}
    for fold in np.unique(folds):
        train = folds != fold
        model, diag = fit_models(c[train], proxy[train], scores[train], seed + int(fold), positive_only=positive_only)
        diagnostics[f"fold_{fold}"] = diag
        for i in np.flatnonzero(~train):
            for method in MODELS:
                records.append({"patch_id": rows[i]["patch_id"], "event_id": rows[i].get("event_id"), "fold": int(fold),
                                "method": method, "reference_fraction": c[i], "predicted_fraction": proxy[i],
                                "missing_score": not np.isfinite(scores[i]).all(),
                                **predict(model, proxy[i], scores[i], c[i], method, [seed, 100, int(i)])})
        print(f"Completed validation fold {fold + 1}: {sum(~train)} held-out patches", flush=True)
    summaries = {}
    for method in MODELS:
        held = [r for r in records if r["method"] == method]
        summaries[method] = {"all": metrics(held),
                             "zero_proxy": metrics([r for r in held if r["predicted_fraction"] == 0])}
    final, diagnostics["full_validation"] = fit_models(c, proxy, scores, seed + len(np.unique(folds)), positive_only=positive_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "calibration"
        stage.mkdir()
        np.savez_compressed(stage / "posterior.npz", **final)
        e02.write_csv(stage / "validation_oof.csv", records)
        e02.write_json(stage / "calibration_summary.json", {
            "stage": "E05 phase 1, validation only", "schema_version": 1, "n_patches": len(c), "seed": seed,
            "n_draws": N_DRAWS, "source_sha256": source_hashes, "implementation_sha256": e02.digest(Path(__file__)),
            "git_commit": e02.git_commit(), "numpy": np.__version__, "scipy": scipy.__version__,
            "features": ["log1p(100 * predicted_fraction)", *FEATURES], "scaling": "training mean and population SD",
            "model": "logit presence; positive beta with logit mean, log precision. Independent Gaussian priors.",
            "priors": {"presence_sd": [2., 1.5], "mean_intercept_location": -2., "precision_intercept_location": float(np.log(20.)),
                       "other_locations": 0., "mean_sd_intercept_proxy_each_score": [2., 1.5, .5],
                       "log_precision_sd_intercept_proxy_each_score": [1.5, 1., .5]},
            "posterior": "Resampled importance draws. Share each coefficient row across patches; presence shared across methods.",
            "missing": "Any missing score uses the full proxy-only predictive distribution.",
            "intervals": "equal-tailed 95% posterior predictive, including the point mass at zero",
            "diagnostics": diagnostics, "oof_metrics": summaries,
            **(burn_protocol or {}),
            "output_sha256": {f: e02.digest(stage / f) for f in ("posterior.npz", "validation_oof.csv")}})
        stage.rename(output)
    print(f"Frozen calibration and held-out predictions: {output}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=Path("models/reliability_gate/three_features_seed0"))
    parser.add_argument("--output", type=Path, default=Path("results/causal_e05/phase1_seed42"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.validation_dir, args.output, args.seed)
