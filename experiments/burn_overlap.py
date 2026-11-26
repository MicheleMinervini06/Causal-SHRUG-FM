"""E08. Diagnose weak treatment overlap concealed by the burn-fraction proxy."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from experiments import burn_fraction_causal as burn


THRESHOLD = 0.10
NEIGHBORS = 15


def overlap(probability):
    return np.minimum(probability, 1 - probability)


def apparent(proxy, probability, folds, k=NEIGHBORS):
    """Cross-fit using nearest fractions, including all ties at the kth distance."""
    prediction = np.empty_like(probability)
    for fold in np.unique(folds):
        train = folds != fold
        if train.sum() < k:
            raise ValueError("Too few training patches to estimate apparent overlap.")
        distance = abs(proxy[~train, None] - proxy[train])
        radius = np.partition(distance, k - 1, axis=1)[:, k - 1]
        neighbors = distance <= radius[:, None]
        prediction[~train] = neighbors @ probability[train] / neighbors.sum(axis=1)
    return prediction


def diagnostic_risks(proxy, density, probability, folds, eligible, k=NEIGHBORS):
    """Fit both diagnostics on apparent-adequate training patches, selected by inner cross-fitting."""
    output = np.full((len(proxy), 2), np.nan)
    weak = overlap(probability) < THRESHOLD
    for fold in np.unique(folds):
        outer = folds != fold
        test = ~outer & eligible
        if not test.any():
            continue
        train = outer.copy()
        train[outer] = overlap(apparent(proxy[outer], probability[outer], folds[outer], k)) >= THRESHOLD
        target = weak[train]
        if not len(target):
            raise ValueError(f"No eligible training patches for fold {fold}.")
        if np.unique(target).size == 1:
            output[test] = (target.sum() + 0.5) / (len(target) + 1)
            continue
        x = np.log1p(100 * proxy[:, None])
        spline = SplineTransformer(n_knots=4, degree=2, include_bias=False).fit(x[train])
        basis = spline.transform(x)
        for column, features in enumerate((basis, np.column_stack((basis, np.log(density))))):
            scaler = StandardScaler().fit(features[train])
            model = LogisticRegression(C=1., max_iter=1000).fit(scaler.transform(features[train]), target)
            output[test, column] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    return output


def analyze(proxy, density, folds, probability, low):
    p_proxy = apparent(proxy, probability, folds)
    ref, app = overlap(probability), overlap(p_proxy)
    eligible = app >= THRESHOLD
    weak = ref < THRESHOLD
    hidden = weak & eligible
    risks = diagnostic_risks(proxy, density, probability, folds, eligible)

    def counts(mask):
        n = int(mask.sum())
        k = int(hidden[mask].sum())
        return {"hidden": k, "denominator": n, "fraction": k/n if n else None}

    losses = (risks[eligible] - weak[eligible, None])**2
    base, augmented = losses.mean(axis=0)
    summary = {"hidden_among_weak": counts(weak), "hidden_among_apparent_adequate": counts(eligible),
               "low_density": counts(eligible & low), "other_density": counts(eligible & ~low),
               "brier_proxy": float(base), "brier_proxy_density": float(augmented),
               "brier_difference": float(augmented - base),
               "folds": []}
    for fold in np.unique(folds):
        member = folds[eligible] == fold
        difference = np.diff(losses[member], axis=1)
        summary["folds"].append({"fold": int(fold), **counts(eligible & (folds == fold)),
                                 "brier_difference": float(difference.mean()) if member.any() else None})
    columns = {"e_ref": probability, "e_proxy": p_proxy, "o_ref": ref, "o_proxy": app,
               "delta": app-ref, "hidden": hidden.astype(int),
               "risk_proxy": risks[:, 0], "risk_proxy_density": risks[:, 1]}
    return columns, summary


def poster(path, results, density):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharex=True, sharey=True, layout="constrained")
    for ax, (_, (cols, summary)), title in zip(
            axes, results.items(), ("E06 assignment", "Steep threshold assignment"), strict=True):
        ax.fill_between([THRESHOLD, .5], 0, THRESHOLD, color="#f2e6df", zorder=0)
        ax.plot([0, .5], [0, .5], color=".65", linewidth=.8, zorder=0)
        ax.axhline(THRESHOLD, linestyle="--", linewidth=.9, color=".38")
        ax.axvline(THRESHOLD, linestyle="--", linewidth=.9, color=".38")
        points = ax.scatter(cols["o_proxy"], cols["o_ref"], c=np.log10(density), cmap="viridis",
                            s=25, alpha=.88, linewidths=.3, edgecolors="white")
        count = summary["hidden_among_weak"]
        ax.set(title=f"{title}\n{count['hidden']} of {count['denominator']} weak patches hidden",
               xlabel="Apparent overlap", xlim=(-.01, .51), ylim=(-.01, .51), aspect="equal")
    axes[0].set_ylabel("Reference overlap")
    fig.colorbar(points, ax=axes, shrink=.8, label="Pretraining density (log₁₀)")
    for suffix in ("pdf", "png"):
        fig.savefig(path / f"overlap_density.{suffix}", dpi=300)
    plt.close(fig)


def run(inputs, output):
    patches, proxy, scores = burn.load_split(inputs, "test")
    z, _, scaling = burn.e02.standardize(patches)
    z = np.asarray(z)
    density = scores[:, 2]
    folds = burn.geographic_folds(patches)
    cutoff = float(np.quantile(density, .25))
    low = density <= cutoff
    rules = {"e06": expit(z), "steep": expit(4 * (z - np.median(z)))}
    results = {name: analyze(proxy, density, folds, p, low) for name, p in rules.items()}
    rows = []
    for i, patch in enumerate(patches):
        row = {**patch, "density": density[i], "low_density": int(low[i]), "fold": int(folds[i])}
        for name, (columns, _) in results.items():
            for key, values in columns.items():
                row[f"{name}_{key}"] = values[i].item() if np.isfinite(values[i]) else None
        rows.append(row)
    report = {"experiment": "E08 Burn Scars overlap pilot", "n": len(rows), "threshold": THRESHOLD,
              "neighbors": NEIGHBORS, "low_density_cutoff": cutoff, "scaling": scaling,
              "assignment": ["expit(z)", "expit(4 * (z - cohort median z))"],
              "diagnostic": "Geographic five-fold outer evaluation; inner geographic cross-fit for eligibility. "
                            "Quadratic spline of log1p(100 * proxy), 4 uniform knots, standardized L2 logistic "
                            "models with C=1, optionally adding log density. Single-class fits use Jeffreys smoothing.",
              "density_join": "Assumed alignment on original split and sample index.",
              "source_sha256": burn.HASHES, "script_sha256": burn.e02.digest(Path(__file__)),
              "rules": {name: summary for name, (_, summary) in results.items()}}
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "pilot"
        stage.mkdir()
        burn.e02.write_csv(stage / "patches.csv", rows)
        burn.e02.write_json(stage / "summary.json", report)
        poster(stage, results, density)
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=burn.INPUTS)
    parser.add_argument("--output", type=Path, default=Path("results/causal_e08/burn_overlap"))
    args = parser.parse_args()
    run(args.inputs, args.output)
