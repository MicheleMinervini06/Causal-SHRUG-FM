"""Local E08 controls and sensitivity analysis. See e08_sensitivity_protocol.md."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from experiments import burn_overlap as pilot


BASE = Path("results/causal_e08/burn_overlap")
KS = (5, 15, 30)
RULES = ("e06", "steep")
SEED, DRAWS = 808, 100


def controls(reference, density, rng):
    ranks = (rankdata(density) - 1) / (len(density) - 1)
    large = rng.random(len(density)) < .95 - .90 * ranks
    size = np.where(large, rng.uniform(.10, .25, len(ranks)), rng.uniform(0, .02, len(ranks)))
    sign = rng.choice([-1, 1], len(ranks))
    for arm, magnitude in (("positive", size), ("negative", rng.permutation(size))):
        proposed = reference + sign * magnitude
        proxy = np.where((proposed < 0) | (proposed > 1), reference - sign * magnitude, proposed)
        assert np.allclose(np.abs(proxy-reference), magnitude) and np.all((proxy >= 0) & (proxy <= 1))
        yield arm, proxy


def predict_error(proxy, density, error, folds):
    loss = np.empty((len(proxy), 2))
    x = np.log1p(100 * proxy[:, None])
    for fold in np.unique(folds):
        train = folds != fold
        basis = SplineTransformer(n_knots=4, degree=2, include_bias=False).fit(x[train]).transform(x)
        for j, features in enumerate((basis, np.column_stack((basis, np.log(density))))):
            scaled = StandardScaler().fit(features[train]).transform(features)
            fit = Ridge(alpha=1.).fit(scaled[train], error[train])
            loss[~train, j] = abs(fit.predict(scaled[~train]) - error[~train])
    return loss


def mean(x):
    return float(np.mean(x)) if x.size else None


def run(output):
    burn = pilot.burn
    patches, actual, scores = burn.load_split(burn.INPUTS, "test")
    c = np.array([p["reference_fraction"] for p in patches])
    density = scores[:, 2]
    folds = burn.geographic_folds(patches)
    low = density <= np.quantile(density, .25)
    z, _, scaling = burn.e02.standardize(patches)
    z = np.asarray(z)
    zp = (actual - scaling["center"]) / scaling["scale"]
    probability = (expit(z), expit(4 * (z-np.median(z))))
    direct = (expit(zp), expit(4 * (zp-np.median(z))))
    baseline = json.loads((BASE / "summary.json").read_text())
    masks = {"all": np.ones(len(c), bool), **{str(i): folds == i for i in np.unique(folds)}}
    errors, comparisons, mechanisms, injected = [], [], [], []
    residual = actual-c
    loss = predict_error(actual, density, abs(residual), folds)
    for fold, mask in masks.items():
        errors.append(dict(fold=fold, n=int(mask.sum()), proxy_mae=mean(loss[mask, 0]),
                           density_mae=mean(loss[mask, 1]), difference=mean(loss[mask, 1]-loss[mask, 0]),
                           low_fraction_mae=mean(abs(residual[mask & low])),
                           other_fraction_mae=mean(abs(residual[mask & ~low]))))

    def evaluate(proxy, arm, rep):
        for rule, p, dp in zip(RULES, probability, direct, strict=True):
            weak = pilot.overlap(p) < pilot.THRESHOLD
            for k in KS:
                app = pilot.overlap(pilot.apparent(proxy, p, folds, k))
                eligible = app >= pilot.THRESHOLD
                risk = pilot.diagnostic_risks(proxy, density, p, folds, eligible, k)
                squared = (risk - weak[:, None])**2
                hidden = weak & eligible
                if arm == "real":
                    direct_hidden = weak & (pilot.overlap(dp) >= pilot.THRESHOLD)
                    oracle_hidden = weak & (pilot.overlap(pilot.apparent(c, p, folds, k)) >= pilot.THRESHOLD)
                for fold, mask in masks.items():
                    selected = mask & eligible
                    h = mask & hidden
                    row = dict(arm=arm, rep=rep, rule=rule, k=k, fold=fold, n=int(mask.sum()),
                               weak=int((weak & mask).sum()), eligible=int(selected.sum()), hidden=int(h.sum()),
                               brier_proxy=mean(squared[selected, 0]), brier_density=mean(squared[selected, 1]),
                               difference=mean(squared[selected, 1]-squared[selected, 0]))
                    comparisons.append(row)
                    if arm == "real":
                        visible = mask & weak & ~hidden
                        mechanisms.append(dict(rule=rule, k=k, fold=fold, hidden=int(h.sum()), visible=int(visible.sum()),
                            hidden_mae=mean(abs(residual[h])), visible_mae=mean(abs(residual[visible])),
                            hidden_signed_error=mean(residual[h]), visible_signed_error=mean(residual[visible]),
                            direct_crossings=int((mask & direct_hidden).sum()),
                            direct_and_hidden=int((h & direct_hidden).sum()), oracle_hidden=int((mask & oracle_hidden).sum()),
                            introduced_by_proxy=int((h & ~oracle_hidden).sum()),
                            removed_by_proxy=int((mask & oracle_hidden & ~hidden).sum())))
                        if fold == "all" and k == 15:
                            orig = baseline["rules"][rule]
                            assert row["hidden"] == orig["hidden_among_weak"]["hidden"]
                            np.testing.assert_allclose([row["brier_proxy"], row["brier_density"]],
                                                       [orig["brier_proxy"], orig["brier_proxy_density"]], atol=1e-12, rtol=0)

    evaluate(actual, "real", 0)
    rng = np.random.default_rng(SEED)
    for rep in range(DRAWS):
        for arm, proxy in controls(c, density, rng):
            error = abs(proxy-c)
            injected.append(dict(rep=rep, arm=arm, mae=mean(error), low_mae=mean(error[low]),
                                 other_mae=mean(error[~low])))
            evaluate(proxy, arm, rep)
        assert np.isclose(injected[-1]["mae"], injected[-2]["mae"])
        if (rep+1) % 20 == 0:
            print(f"Completed {rep+1}/{DRAWS} paired draws", flush=True)

    frequencies = []
    for rule in RULES:
        for k in KS:
            group = [r for r in comparisons if r["rule"] == rule and r["k"] == k and r["fold"] == "all"]
            pos, neg = (np.array([r["difference"] for r in group if r["arm"] == arm])
                        for arm in ("positive", "negative"))
            wins = np.mean(pos < neg)
            frequencies.append(dict(rule=rule, k=k, draws=DRAWS, positive_wins=int((pos < neg).sum()),
                win_mcse=float(np.sqrt(wins*(1-wins)/DRAWS)), positive_improves=int((pos < 0).sum()),
                negative_improves=int((neg < 0).sum()), positive_mean=mean(pos), negative_mean=mean(neg)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/"complete"
        stage.mkdir()
        for name, records in (("fold_results", comparisons), ("control_frequencies", frequencies),
                              ("control_errors", injected), ("real_error_prediction", errors),
                              ("real_crossings", mechanisms)):
            burn.e02.write_csv(stage/f"{name}.csv", records)
        burn.e02.write_json(stage/"config.json", {"seed": SEED, "draws": DRAWS, "neighbors": KS,
            "threshold": pilot.THRESHOLD, "source_sha256": burn.HASHES,
            "implementation_sha256": {path.name: burn.e02.digest(path) for path in
                (Path(__file__), Path(pilot.__file__), Path(__file__).with_name("e08_sensitivity_protocol.md"))},
            "baseline_summary_sha256": burn.e02.digest(BASE/"summary.json")})
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE.with_name("burn_overlap_sensitivity"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    with threadpool_limits(limits=1):
        run(args.output)
