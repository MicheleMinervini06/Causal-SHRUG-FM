"""E09: predict the conditional bias of whole-cohort proxy-adjusted OLS."""

import argparse
from ast import literal_eval
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist
from scipy.special import expit
from threadpoolctl import threadpool_limits

from experiments import burn_single_patch as burn
from experiments import landslide_fraction_causal as land
from experiments import verification_scores as scores
from experiments import fraction_calibration_causal as e05

e02 = e05.e02
METHODS = ("zero", "proxy_only", *scores.SCORES)
FLOOD = Path("results/segmentation_runs")
LAND_EMBED = Path("data/landslides/landslides_ncdd_k64.csv")
LAND_SHA = "e10b382ba9bb808d557c6a0a8352773141a358cc39e8aa401934e73f60fe37f8"
THRESHOLD = .05


def load(dataset, split):
    test = split == "test"
    if dataset == "floods":
        folder = "gate-test-pred-job-44220183-analysis" if test else "gate-job-44217005-analysis"
        path = FLOOD/folder/"inputs/gate_scores.csv"
        sha = e05.TEST_SCORES if test else e05.calibration.INPUT_HASHES["gate_scores.csv"]
        rows = sorted(e02.read_rows(e05.verified(path, sha)), key=lambda r: r["patch_id"])
        signals = np.array([[sign*scores.number(r[key], missing=True) for key, sign in scores.SCORES.values()]
                            for r in rows])
        for r in rows:
            for key in ("reference_fraction", "predicted_fraction"):
                r[key] = float(r[key])
        return rows, signals, np.array([r["event_id"] for r in rows]), {str(path): sha}
    if dataset == "burns":
        directory = burn.burn.INPUTS
        rows, _, task = burn.burn.load_split(directory, split)
        lon, lat = np.radians([literal_eval(r["location"]) for r in rows]).T
        coords = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        path, sha, hashes = directory/burn.EMBEDDINGS, burn.EMBEDDING_HASH, burn.burn.HASHES
    else:
        directory = land.ROOT/"inputs"
        rows, _, task, boxes = land.load_split(directory, split)
        coords = (boxes[:, :2]+boxes[:, 2:])/2
        path, sha, hashes = LAND_EMBED, LAND_SHA, land.HASHES
    embed = sorted((r for r in e02.read_rows(e05.verified(path, sha)) if r["split"] == split),
                   key=lambda r: int(r["index"]))
    if [int(r["index"]) for r in embed] != list(range(len(rows))):
        raise ValueError("Embedding indices differ from the split's patch indices.")
    signals = np.column_stack((task[:, :2], -np.log(task[:, 2]),
                               [float(r["normalized_distance"]) for r in embed], [-float(r["ncdd"]) for r in embed]))
    return rows, signals, coords, {**{str(directory/k): v for k, v in hashes.items()}, str(path): sha}


def partial_weights(treatment, proxy):
    """Each row maps a patch vector to its adjusted treatment coefficient."""
    residual = treatment - treatment.mean(axis=1, keepdims=True)
    centered = proxy - proxy.mean()
    residual -= np.outer(residual @ centered, centered)/(centered @ centered)
    ss = np.sum(residual**2, axis=1)
    if np.any(ss < 1e-10):
        raise ValueError("Rank-deficient test assignment.")
    return residual / ss[:, None]


def predict_error(train_x, test_x, train_t, test_t, error):
    """Fit the same ridge for each assignment, sharing the fixed-feature algebra."""
    features = []
    for train, test in zip(train_x.T, test_x.T, strict=True):
        observed = train[np.isfinite(train)]
        if len(observed) and observed.std() > 1e-12:
            features.append([np.nan_to_num((v-observed.mean())/observed.std(), nan=0.) for v in (train, test)])
    x = np.column_stack([np.ones(len(train_x)), *(pair[0] for pair in features)])
    z = np.column_stack([np.ones(len(test_x)), *(pair[1] for pair in features)])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    project = np.linalg.solve(x.T @ x + penalty, x.T)
    fixed = project @ error
    t_projection = train_t @ project.T
    remainder = train_t - t_projection @ x.T
    t_coef = (train_t @ (error - x @ fixed))/(1 + np.sum(train_t*remainder, axis=1))
    return z @ fixed + t_coef[:, None] * (test_t - t_projection @ z.T)


def summary(pred, truth, base, scale_ratio):
    error = pred - truth
    difference = abs(error)-abs(base-truth)
    mc = 1.96*difference.std(ddof=1)/np.sqrt(len(truth))
    flag, case = abs(pred) > THRESHOLD, abs(truth) > THRESHOLD
    tp, tn = int((case & flag).sum()), int((~case & ~flag).sum())
    return dict(n=len(truth), mean_true_bias=truth.mean(), mean_warning=pred.mean(), mae=abs(error).mean(),
                rmse=np.mean(error**2)**.5, delta_mae=difference.mean(), mc_lo=difference.mean()-mc,
                mc_hi=difference.mean()+mc, known_scale_mae=abs(pred*scale_ratio-truth).mean(),
                cases=int(case.sum()), alerts=int(flag.sum()), true_positive=tp, true_negative=tn,
                sensitivity=tp/case.sum() if case.any() else None,
                specificity=tn/(~case).sum() if (~case).any() else None, ppv=tp/flag.sum() if flag.any() else None)


def run(output):
    report, partitions, provenance = [], [], {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"run"
        stage.mkdir()
        for dataset in ("floods", "burns", "landslides"):
            val, vs, vg, vh = load(dataset, "val")
            test, ts, tg, th = load(dataset, "test")
            if dataset != "floods":
                test_groups = cut_tree(linkage(tg, method="ward"), n_clusters=5).ravel()
                vg, tg = test_groups[cdist(vg, tg).argmin(axis=1)], test_groups
            for rows, regions in ((val, vg), (test, tg)):
                partitions.extend(dict(dataset=dataset, split=r["original_split"], patch_id=r["patch_id"],
                                       region=g) for r, g in zip(rows, regions, strict=True))
            if {r["patch_id"] for r in test} & {r["patch_id"] for r in val}:
                raise ValueError("Calibration and evaluation IDs overlap.")
            def arrays(rows, seed):
                ref, _, units = e02.standardize(rows)
                proxy = np.array([r["predicted_fraction"] for r in rows])
                uniforms = np.array([e02.simulation_draws(rows, seed, i)[0] for i in range(e05.TOTAL)])
                return np.array(ref), units["scale"], proxy, uniforms
            vc, val_sd, vx, vu = arrays(val, 9042)
            tc, test_sd, tx, tu = arrays(test, 42)
            residual = (np.array([r["reference_fraction"] for r in val])-vx)/val_sd
            provenance[dataset] = dict(inputs={**vh, **th}, validation_sd=val_sd, test_truth_sd=test_sd,
                missing_test_scores=dict(zip(scores.SCORES, np.isnan(ts).sum(axis=0).tolist())),
                missing_validation_scores=dict(zip(scores.SCORES, np.isnan(vs).sum(axis=0).tolist())))
            for strength in (0, 1):
                vt = (vu < expit(strength*vc)).astype(float)
                tt = (tu < expit(strength*tc)).astype(float)
                weights = partial_weights(tt, tx)
                truth = weights @ tc
                for i in (0, e05.TOTAL-1):
                    noisy_free = e02.treatment_coefficient(tt[i]+tc, tt[i], tx)
                    np.testing.assert_allclose(truth[i], noisy_free-1, atol=1e-12)
                estimates = {"zero": np.zeros_like(truth)}
                for j, method in enumerate(METHODS[1:]):
                    features = (vx[:, None], tx[:, None]) if j == 0 else (
                        np.column_stack((vx, vs[:, j-1])), np.column_stack((tx, ts[:, j-1])))
                    warning = np.zeros_like(truth)
                    for group in np.unique(tg):
                        tr, te = vg != group, tg == group
                        patch_errors = predict_error(features[0][tr], features[1][te], vt[:, tr], tt[:, te], residual[tr])
                        warning += np.sum(weights[:, te]*patch_errors, axis=1)
                    estimates[method] = warning
                for method, pred in estimates.items():
                    report.append(dict(dataset=dataset, a1=strength, method=method,
                        **summary(pred, truth, estimates["proxy_only"], val_sd/test_sd)))
                np.savez_compressed(stage/f"{dataset}_a1-{strength}.npz", repetition=np.arange(e05.TOTAL),
                                    true_bias=truth, **estimates)
        e02.write_csv(stage/"summary.csv", report)
        e02.write_csv(stage/"partitions.csv", partitions)
        e02.write_json(stage/"config.json", dict(cohorts=provenance, threshold=THRESHOLD, repetitions=e05.TOTAL,
            validation_seed=9042, test_seed=42, git_parent=e02.git_commit(), sha256={p.name:e02.digest(p) for p in
                (Path(__file__), Path(__file__).with_name("e09_protocol.md"), Path(e02.__file__), Path(scores.__file__),
                 Path(burn.burn.__file__), Path(land.__file__))}))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/causal_e09/seed42"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    with threadpool_limits(limits=1):
        run(args.output)
