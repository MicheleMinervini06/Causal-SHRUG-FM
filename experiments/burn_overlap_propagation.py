"""Propagate the conditional burned-fraction distribution through the fixed E08 rules."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import betaln, betainc, betaincc, expit, logit, logsumexp, ndtri
from scipy.stats import qmc, rankdata
from threadpoolctl import threadpool_limits

from experiments import burn_overlap_sensitivity as sensitivity

pilot = sensitivity.pilot
burn = pilot.burn
cal = burn.calibration
PROTOCOL = Path(__file__).with_name("e08_propagation_protocol.md")


def forward_likelihood(proxy, reference, large_prob):
    """Rows are observed proxies, columns are candidate latent patches."""
    error = proxy[:, None] - reference
    size = abs(error)
    opposite = reference - error
    direction = .5 + .5 * ((opposite < 0) | (opposite > 1))
    return direction * (((size <= .02) / .02) * (1 - large_prob)
                        + (((size >= .10) & (size <= .25)) / .15) * large_prob)


def known_risks(proxy, c, group, large_prob, weak):
    likelihood = forward_likelihood(proxy, c, large_prob)
    risks = []
    for weights in (likelihood, likelihood * (group[:, None] == group)):
        total = weights.sum(axis=1)
        if np.any(total <= 0):
            raise ValueError("Synthetic draw outside forward-model support.")
        risks.append(weights @ weak / total[:, None])
    return np.stack(risks, axis=-1)


def learned(proxy, c, density_feature, folds, cuts, *, importance=False):
    features = np.column_stack((np.log1p(100 * proxy), density_feature))
    risk = np.empty((len(c), len(cuts), 2))
    log_score = np.empty((len(c), 2))
    for fold in np.unique(folds):
        train = folds != fold
        z = (features - features[train].mean(axis=0)) / features[train].std(axis=0)
        design = np.column_stack((np.ones(len(c)), z))
        for j, method in enumerate(cal.MODELS):
            x = design[:, :2+j]
            if importance:
                params, _ = cal.posterior(x[train], c[train], method, np.random.default_rng([808, int(fold), j]))
            else:
                center, cov, _ = cal.posterior_mode(x[train], c[train], method)
                normals = ndtri(qmc.Sobol(len(center), scramble=True, seed=808).random_base2(7))
                params = center + normals @ np.linalg.cholesky(cov).T
            a, b = cal.beta_shapes(params, x[~train])
            actual = c[~train]
            log_score[~train, j] = logsumexp(
                (a-1)*np.log(actual) + (b-1)*np.log1p(-actual) - betaln(a, b), axis=0) - np.log(len(a))
            for r, (lo, hi) in enumerate(cuts):
                risk[~train, r, j] = (betainc(a, b, lo) + betaincc(a, b, hi)).mean(axis=0)
    if not np.isfinite(log_score).all() or not np.all((risk >= 0) & (risk <= 1)):
        raise ValueError("Invalid posterior predictive summary.")
    return risk, log_score


def run(output):
    patches, actual, scores = burn.load_split(burn.INPUTS, "test")
    c = np.array([p["reference_fraction"] for p in patches])
    density = scores[:, 2]
    ranks = (rankdata(density)-1) / (len(c)-1)
    group = np.searchsorted(np.quantile(density, [.25, .5, .75]), density, side="left")
    prob_large = .95 - .90*ranks
    folds = burn.geographic_folds(patches)
    z, _, scale = burn.e02.standardize(patches)
    z = np.array(z)
    prob = np.column_stack((expit(z), expit(4*(z-np.median(z)))))
    weak = pilot.overlap(prob) < pilot.THRESHOLD
    edges = logit([pilot.THRESHOLD, 1-pilot.THRESHOLD])
    cuts = np.clip(scale["center"] + scale["scale"] * np.array([edges, np.median(z)+edges/4]), 0, 1)
    np.testing.assert_array_equal(weak, (c[:, None] < cuts[:, 0]) | (c[:, None] > cuts[:, 1]))
    records, predictions = [], []
    masks = {"all": np.ones(len(c), bool), **{str(i): folds == i for i in np.unique(folds)}}

    def evaluate(proxy, arm, rep):
        base, score = learned(proxy, c, group, folds, cuts)
        models = {"beta_quarter": (base, score)}
        if arm == "real":
            models["beta_log_density"] = learned(proxy, c, np.log(density), folds, cuts)
            for name, feature in (("quarter", group), ("log_density", np.log(density))):
                models[f"importance_{name}"] = learned(proxy, c, feature, folds, cuts, importance=True)
        else:
            large = prob_large if arm == "positive" else prob_large.mean()
            models["known_quarter"] = (known_risks(proxy, c, group, large, weak), None)
        for r, rule in enumerate(sensitivity.RULES):
            adequate = pilot.overlap(pilot.apparent(proxy, prob[:, r], folds)) >= pilot.THRESHOLD
            for model, (risk, log_score) in models.items():
                loss = (risk[:, r] - weak[:, r, None])**2
                for fold, mask in masks.items():
                    keep = mask & adequate
                    records.append(dict(arm=arm, rep=rep, rule=rule, model=model, fold=fold,
                        n=int(mask.sum()), adequate=int(keep.sum()), hidden=int((keep & weak[:, r]).sum()),
                        proxy_brier=sensitivity.mean(loss[keep, 0]), density_brier=sensitivity.mean(loss[keep, 1]),
                        change=sensitivity.mean(loss[keep, 1]-loss[keep, 0]),
                        proxy_log_score=sensitivity.mean(log_score[mask, 0]) if log_score is not None else None,
                        density_log_score=sensitivity.mean(log_score[mask, 1]) if log_score is not None else None))
                if arm == "real":
                    for i, patch in enumerate(patches):
                        predictions.append(dict(patch_id=patch["patch_id"], fold=int(folds[i]), rule=rule,
                            model=model, density_quarter=int(group[i])+1, adequate=int(adequate[i]),
                            weak=int(weak[i, r]), proxy_risk=risk[i, r, 0], density_risk=risk[i, r, 1]))

    evaluate(actual, "real", 0)
    rng = np.random.default_rng(sensitivity.SEED)
    for rep in range(sensitivity.DRAWS):
        for arm, proxy in sensitivity.controls(c, density, rng):
            evaluate(proxy, arm, rep)
        if (rep+1) % 10 == 0:
            print(f"Completed {rep+1}/{sensitivity.DRAWS} paired controls", flush=True)
    pooled = [r for r in records if r["fold"] == "all"]
    summary = []
    for rule in sensitivity.RULES:
        for model in ("beta_quarter", "known_quarter"):
            pos, neg = (np.array([r["change"] for r in pooled if r["arm"] == arm
                                  and r["rule"] == rule and r["model"] == model]) for arm in ("positive", "negative"))
            summary.append(dict(rule=rule, model=model, draws=len(pos), positive_mean=pos.mean(),
                negative_mean=neg.mean(), paired_change=(pos-neg).mean(),
                paired_mcse=(pos-neg).std(ddof=1)/np.sqrt(len(pos)), positive_improves=int((pos<0).sum()),
                negative_improves=int((neg<0).sum()), paired_wins=int((pos<neg).sum())))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/"complete"
        stage.mkdir()
        for name, rows in (("fold_results", records), ("real_predictions", predictions), ("controls", summary)):
            burn.e02.write_csv(stage/f"{name}.csv", rows)
        burn.e02.write_json(stage/"config.json", dict(source_sha256=burn.HASHES, seed=sensitivity.SEED,
            draws=sensitivity.DRAWS, neighbors=pilot.NEIGHBORS, threshold=pilot.THRESHOLD,
            weak_fraction_boundaries=cuts.tolist(), implementation_sha256={p.name: burn.e02.digest(p) for p in
                (Path(__file__), Path(cal.__file__), Path(sensitivity.__file__), Path(pilot.__file__),
                 Path(burn.__file__), PROTOCOL)}))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=sensitivity.BASE.with_name("burn_overlap_propagation"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    with threadpool_limits(limits=1):
        run(args.output)
