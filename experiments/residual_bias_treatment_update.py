"""E09 extension: propagate E05 calibration through observed treatment into proxy OLS bias."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from numba import njit
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiments import burn_fraction_causal as burn
from experiments import landslide_fraction_causal as landslides
from experiments import residual_bias_warning as warning

e05, e02 = warning.e05, warning.e02
POSTERIOR = Path("results/causal_e05/phase1_seed42/posterior.npz")
RISK_SEED = 9109
CONSTANT_SEED = 19042
REPLICATION_PROTOCOL = Path(__file__).with_name("e09_treatment_replication_protocol.md")
THRESHOLD_PROTOCOL = Path(__file__).with_name("e09_threshold_protocol.md")
COMMON_THRESHOLDS = (.025, .05, .10, .20, .50)
E07_POSTERIORS = {
    "burns": "b6b94e5ad63e2ecadfb34fc036f720d3f358a99703c1b6fced43f554b15a4a39",
    "landslides": "26e90f68fe5659faeb83966ec8e5537d2f9332493ee0d7f7fe002c690b5cde28",
}


def load(dataset, split):
    if dataset == "floods":
        rows, _, _, hashes = warning.load(dataset, split)
        proxy = np.array([row["predicted_fraction"] for row in rows])
        scores = np.array([[e05.calibration.number(row[key], missing=True)
                            for key in e05.calibration.FEATURES] for row in rows])
        return rows, proxy, scores, hashes
    if dataset == "burns":
        directory, adapter = burn.INPUTS, burn
    else:
        directory, adapter = landslides.ROOT/"inputs", landslides
    rows, proxy, scores, *_ = adapter.load_split(directory, split)
    return rows, proxy, scores, {str(directory/name): sha for name, sha in adapter.HASHES.items()}


def verified_posterior(dataset, n_validation):
    if dataset == "floods":
        return e05.verified(POSTERIOR, e05.POSTERIOR), {str(POSTERIOR): e05.POSTERIOR}
    if dataset == "burns":
        path, stage, hashes = burn.CALIBRATION, burn.EXPERIMENT + ", validation only", burn.HASHES
    else:
        path = landslides.ROOT/"landslide_calibration_seed42"
        stage, hashes = landslides.STAGE, landslides.HASHES
    summary_path = path/"calibration_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_sha = E07_POSTERIORS[dataset]
    if (summary["stage"] != stage or summary["source_sha256"] != hashes or summary["seed"] != 42
            or summary["n_patches"] != n_validation or summary["output_sha256"]["posterior.npz"] != expected_sha):
        raise ValueError(f"Expected the frozen E07 validation calibration for {dataset}.")
    posterior = e05.verified(path/"posterior.npz", expected_sha)
    return posterior, {str(posterior): expected_sha, str(summary_path): e02.digest(summary_path)}


def conditional_nodes(grid, mass, intercept, slope):
    """Precompute each patch's treatment evidence, conditional mean and node CDF."""
    n_draws, n_patches, n_nodes = grid.shape
    counts = 2 if slope else 1
    means = np.empty((counts, n_draws, n_patches))
    evidence = np.empty_like(means)
    cdf = np.empty((counts, n_patches, n_draws, n_nodes))
    for patch in range(n_patches):
        nodes = grid[:, patch]
        for treatment in range(counts):
            sign = 2*treatment - 1
            weighted = mass[:, patch] * expit(sign*(intercept + slope*nodes))
            total = weighted.sum(axis=1)
            evidence[treatment, :, patch] = np.log(total)
            normalized = weighted/total[:, None]
            means[treatment, :, patch] = np.sum(normalized*nodes, axis=1)
            np.cumsum(normalized, axis=1, out=cdf[treatment, patch])
            cdf[treatment, patch, :, -1] = 1.
    return evidence, means, cdf


@njit
def threshold_counts(grid, cdf, treatment, weights, rows, uniforms, thresholds):
    counts = np.zeros((2, len(thresholds)), dtype=np.int64)
    for draw, row in enumerate(rows):
        bias = 0.
        for patch in range(len(weights)):
            index = np.searchsorted(cdf[treatment[patch], patch, row], uniforms[draw, patch], side="right")
            bias += weights[patch] * grid[row, patch, index]
        half = int(draw >= len(rows)//2)
        for j in range(len(thresholds)):
            counts[half, j] += abs(bias) > thresholds[j]
    return counts


def predict(grid, mass, treatments, weights, assignment, risk_draws, thresholds):
    evidence, means, cdf = conditional_nodes(grid, mass, *assignment)
    uniform_rows = len(evidence) == 1
    mean_bias = np.empty(len(treatments))
    risk_counts = np.empty((len(treatments), 2, len(thresholds)), dtype=int)
    ess = np.full(len(treatments), len(grid), dtype=float)
    if uniform_rows:
        mean_bias[:] = weights @ means[0].mean(axis=0)
        expected = (grid*mass).sum(axis=-1)
        np.testing.assert_allclose(means[0], expected, rtol=0., atol=1e-12)
    else:
        log_difference = evidence[1] - evidence[0]
        mean_difference = means[1] - means[0]
        log_control = evidence[0].sum(axis=1)
    for start in range(0, len(treatments), 64):
        stop = min(start+64, len(treatments))
        if not uniform_rows:
            treatment = treatments[start:stop]
            weight = weights[start:stop]
            row_prob = softmax(log_control + treatment @ log_difference.T, axis=1)
            conditional_bias = weight @ means[0].T + (weight*treatment) @ mean_difference.T
            mean_bias[start:stop] = np.sum(row_prob*conditional_bias, axis=1)
            ess[start:stop] = 1/np.sum(row_prob**2, axis=1)
        for rep in range(start, stop):
            rng = np.random.default_rng([RISK_SEED, rep])
            row_uniform = rng.random(risk_draws)
            if uniform_rows:
                rows = (len(grid)*row_uniform).astype(int)
                observed = np.zeros(grid.shape[1], dtype=int)
            else:
                row_cdf = np.cumsum(row_prob[rep-start])
                row_cdf[-1] = 1.
                rows = np.searchsorted(row_cdf, row_uniform, side="right")
                observed = treatments[rep].astype(int)
            risk_counts[rep] = threshold_counts(grid, cdf, observed, weights[rep], rows,
                rng.random((risk_draws, grid.shape[1])), thresholds)
    return mean_bias, risk_counts, ess


def synthetic_check():
    """At the same proxy, treatment tilts only the uncertain patch's two-point prior."""
    grid = np.array([[[.4, .4], [.2, .6]]])
    mass = np.full_like(grid, .5)
    evidence, means, cdf = conditional_nodes(grid, mass, -.4, 1.)
    p = expit(.2)
    np.testing.assert_allclose(np.exp(evidence), .5, atol=1e-14)
    np.testing.assert_allclose(means[:, 0], [[.4, .6-.4*p], [.4, .2+.4*p]], atol=1e-14)
    np.testing.assert_allclose(cdf[:, 1, 0, 0], [p, 1-p], atol=1e-14)
    _, randomized_mean, randomized_cdf = conditional_nodes(grid, mass, 0., 0.)
    np.testing.assert_allclose(randomized_mean, .4, atol=1e-14)
    np.testing.assert_allclose(randomized_cdf[..., 0], .5, atol=1e-14)
    np.testing.assert_allclose(randomized_cdf[..., 1], 1., atol=1e-14)
    uniforms = np.tile([.5, .5], (4, 1))
    counts = threshold_counts(grid, cdf, np.array([0, 1]), np.array([-1., 1.]),
                              np.zeros(4, dtype=int), uniforms, np.array([.05, .30]))
    np.testing.assert_array_equal(counts, [[2, 0], [2, 0]])


def treatment_draws(patches, standardized, n, seed, strength):
    uniforms = np.array([e02.simulation_draws(patches, seed, rep)[0] for rep in range(n)])
    return (uniforms < expit(strength*standardized)).astype(float)


def validation_biases(patches, proxy, reference):
    absolute_bias = {}
    for strength in (0, 1):
        treatment = treatment_draws(patches, reference, e05.TOTAL, CONSTANT_SEED, strength)
        absolute_bias[strength] = abs(warning.partial_weights(treatment, proxy) @ reference)
    return absolute_bias


def brier(probability, event, draws):
    # Remove the finite-draw upward bias in squared probability error.
    return (probability-event)**2 - probability*(1-probability)/(draws-1)


def paired_interval(difference):
    mean = difference.mean()
    radius = 1.96*difference.std(ddof=1)/np.sqrt(len(difference))
    return mean, mean-radius, mean+radius


def threshold_summary(predictions, scenarios, true_bias, thresholds, names, constants, draws):
    records, arrays = [], {}
    order = np.argsort(thresholds)
    for scenario, (strength, _, _) in scenarios.items():
        bias = true_bias[strength]
        event = abs(bias[:, None]) > thresholds
        arrays[f"{scenario}_true_bias"] = bias
        constant_loss = (constants[strength]-event)**2
        base_counts = predictions[scenario]["proxy_only"][1]
        base_loss = brier(base_counts.sum(axis=1)/draws, event, draws)
        for method, (_, counts, _) in predictions[scenario].items():
            risk = counts.sum(axis=1)/draws
            half = counts[:, 0]/(draws//2)
            if np.any(np.diff(risk[:, order], axis=1) > 0):
                raise ValueError("Bias exceedance risk must decrease with the threshold.")
            arrays[f"{scenario}_{method}_risk"] = risk
            loss = brier(risk, event, draws)
            for j, (name, cutoff) in enumerate(zip(names, thresholds, strict=True)):
                cases = int(event[:, j].sum())
                controls = len(bias)-cases
                delta, lo, hi = paired_interval(loss[:, j]-base_loss[:, j])
                constant_delta, constant_lo, constant_hi = paired_interval(loss[:, j]-constant_loss[:, j])
                records.append(dict(scenario=scenario, method=method, threshold_id=name, threshold=cutoff,
                    above=cases, at_or_below=controls, auc=roc_auc_score(event[:, j], risk[:, j])
                    if cases and controls else None,
                    mean_risk=risk[:, j].mean(), brier=loss[:, j].mean(),
                    brier_half=brier(half[:, j], event[:, j], draws//2).mean(),
                    delta_brier=delta, delta_brier_lo=lo, delta_brier_hi=hi, constant=constants[strength][j],
                    constant_brier=constant_loss[:, j].mean(), delta_brier_constant=constant_delta,
                    delta_brier_constant_lo=constant_lo, delta_brier_constant_hi=constant_hi))
    return records, arrays


def run(args):
    synthetic_check()
    validation, validation_proxy, _, val_hash = load(args.dataset, "val")
    vref, _, vscale = e02.standardize(validation)
    vref = np.asarray(vref)
    validation_bias = validation_biases(validation, validation_proxy, vref)
    if args.threshold_sensitivity:
        thresholds = np.r_[COMMON_THRESHOLDS, np.quantile(validation_bias[1], [.5, .75])]
        names = [*(f"fixed_{cutoff:g}" for cutoff in COMMON_THRESHOLDS), "validation_q50", "validation_q75"]
    else:
        thresholds = np.array([warning.THRESHOLD])
        names = ["fixed_0.05"]
    default_threshold = names.index("fixed_0.05")
    constants = {strength: (bias[:, None] > thresholds).mean(axis=0) for strength, bias in validation_bias.items()}
    vt = treatment_draws(validation, vref, 1, 9042, 1)[0]
    fit = LogisticRegression(C=1.).fit(vref[:, None], vt)
    posterior, posterior_hashes = verified_posterior(args.dataset, len(validation))
    patches, proxy, task_scores, test_hash = load(args.dataset, "test")
    reference, _, scale = e02.standardize(patches)
    scenarios = {
        "known": (1, scale, (0., 1.)),
        "validation_fit": (1, vscale, (float(fit.intercept_[0]), float(fit.coef_[0, 0]))),
        "randomized": (0, scale, (0., 0.)),
    }
    treatment = {a: treatment_draws(patches, np.asarray(reference), args.repeats, 42, a) for a in (0, 1)}
    weights = {a: warning.partial_weights(treatment[a], proxy) for a in treatment}
    true_bias = {a: w @ reference for a, w in weights.items()}
    predictions, times = {name: {} for name in scenarios}, {}
    with np.load(posterior, allow_pickle=False) as frozen:
        if bool(frozen.get("positive_only", False)) != (args.dataset != "floods"):
            raise ValueError("Calibration support differs from the cohort protocol.")
        for method in e05.calibration.MODELS:
            start = monotonic()
            grid, mass = e05.inference.latent_grid(frozen, proxy, task_scores, method, args.nodes)
            grid *= proxy.std()
            grid += proxy.mean()
            times[f"grid_{method}"] = monotonic()-start
            print(f"{method}: prepared {args.nodes}-node grid in {times[f'grid_{method}']:.1f}s", flush=True)
            for name, (strength, scaling, assignment) in scenarios.items():
                start = monotonic()
                normalized = (grid-scaling["center"])/scaling["scale"]
                predictions[name][method] = predict(normalized, mass, treatment[strength],
                                                     weights[strength], assignment, args.risk_draws, thresholds)
                del normalized
                times[f"{name}_{method}"] = monotonic()-start
                print(f"{name} {method}: {args.repeats} assignments in {monotonic()-start:.1f}s", flush=True)
            del grid, mass

    rows, repetitions = [], []
    for scenario, (strength, _, _) in scenarios.items():
        truth = true_bias[strength]
        event = abs(truth) > warning.THRESHOLD
        constant = constants[strength][default_threshold]
        constant_loss = (constant-event)**2
        base, base_counts, _ = predictions[scenario]["proxy_only"]
        base_loss = brier(base_counts[:, :, default_threshold].sum(axis=1)/args.risk_draws, event, args.risk_draws)
        for method, (mean, counts, ess) in predictions[scenario].items():
            counts = counts[:, :, default_threshold]
            risk = counts.sum(axis=1)/args.risk_draws
            half = counts[:, 0]/(args.risk_draws//2)
            loss = brier(risk, event, args.risk_draws)
            delta, lo, hi = paired_interval(loss-base_loss)
            delta_constant, constant_lo, constant_hi = paired_interval(loss-constant_loss)
            metrics = warning.summary(mean, truth, base, 1.)
            del metrics["known_scale_mae"]
            rows.append(dict(scenario=scenario, method=method, **metrics, brier=loss.mean(),
                delta_brier=delta, delta_brier_lo=lo, delta_brier_hi=hi,
                brier_half=brier(half, event, args.risk_draws//2).mean(),
                validation_constant=constant, constant_brier=constant_loss.mean(),
                delta_brier_constant=delta_constant, delta_brier_constant_lo=constant_lo,
                delta_brier_constant_hi=constant_hi,
                min_row_ess=ess.min(), median_row_ess=np.median(ess)))
            repetitions.extend(dict(scenario=scenario, method=method, repetition=i,
                true_bias=b, predicted_bias=m, risk=p, risk_half=h, row_ess=e)
                for i, (b, m, p, h, e) in enumerate(zip(truth, mean, risk, half, ess, strict=True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp)/"complete"
        stage.mkdir()
        e02.write_csv(stage/"summary.csv", rows)
        e02.write_csv(stage/"repetitions.csv", repetitions)
        if args.threshold_sensitivity:
            threshold_rows, arrays = threshold_summary(predictions, scenarios, true_bias,
                                                      thresholds, names, constants, args.risk_draws)
            e02.write_csv(stage/"threshold_summary.csv", threshold_rows)
            np.savez_compressed(stage/"threshold_risks.npz", thresholds=thresholds, threshold_ids=names, **arrays)
        e02.write_json(stage/"config.json", dict(dataset=args.dataset, inputs={**val_hash, **test_hash, **posterior_hashes},
            repeats=args.repeats, nodes=args.nodes, risk_draws=args.risk_draws, risk_seed=RISK_SEED,
            constant_seed=CONSTANT_SEED, constant_repeats=e05.TOTAL,
            validation_constant={k: v[default_threshold] for k, v in constants.items()},
            threshold_grid=dict(zip(names, thresholds, strict=True)),
            threshold=warning.THRESHOLD, scenarios=scenarios, seconds=times, commit=e02.git_commit(),
            implementation_sha256={p.name: e02.digest(p) for p in (Path(__file__),
                Path(__file__).with_name("e09_treatment_update_protocol.md"), REPLICATION_PROTOCOL, Path(warning.__file__),
                THRESHOLD_PROTOCOL, Path(e05.inference.__file__), Path(e05.calibration.__file__), Path(e02.__file__),
                Path(burn.__file__), Path(landslides.__file__))}))
        stage.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("floods", "burns", "landslides"), default="floods")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--risk-draws", type=int, default=512)
    parser.add_argument("--threshold-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        name = "treatment_update" if args.dataset == "floods" else f"treatment_update_{args.dataset}"
        if args.threshold_sensitivity:
            name = f"threshold_sensitivity_{args.dataset}"
        args.output = Path("results/causal_e09")/name
    if args.output.exists() or not 2 <= args.repeats <= 5000 or args.nodes < 2:
        parser.error("Use a new output directory, 2–5000 repeats and at least two nodes.")
    if args.risk_draws < 4 or args.risk_draws % 2:
        parser.error("Risk draws must be an even integer of at least four.")
    with threadpool_limits(limits=1):
        run(args)
