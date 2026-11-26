"""E06: compare uncertainty-guided and random verification at fixed budgets."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from experiments import single_patch_causal as e06
from experiments import verification_scores as verification

e05, e02 = verification.e05, verification.e02
TEST_SCORES = Path("results/segmentation_runs/gate-test-pred-job-44220183-analysis/inputs/gate_scores.csv")
PROTOCOL = Path(__file__).with_name("e06_fixed_budget_protocol.md")
BUDGETS = (5, 10)
SELECTION_SEED = 60642
SIMULATION_SEED = 42
POLICIES = ("random", *verification.SCORES)


def adjusted_effect(treatment, outcome, fraction):
    """Return the treatment coefficient and model-based 95% interval for each assignment."""
    centered = [x - x.mean(axis=-1, keepdims=True) for x in (treatment, outcome, fraction)]
    t, y, c = centered
    covariate_ss = np.sum(c*c, axis=-1, keepdims=True)
    if np.any(covariate_ss <= 0):
        raise ValueError("Adjustment fraction must vary in every fit.")
    t -= c * (np.sum(t*c, axis=-1, keepdims=True)/covariate_ss)
    y -= c * (np.sum(y*c, axis=-1, keepdims=True)/covariate_ss)
    treatment_ss = np.sum(t*t, axis=-1)
    if np.any(treatment_ss < 1e-10):
        raise ValueError("Treatment effect is rank deficient.")
    estimate = np.sum(t*y, axis=-1)/treatment_ss
    df = t.shape[-1]-3
    residual = y - estimate[:, None]*t
    variance = np.sum(residual**2, axis=-1)/df
    radius = student_t.ppf(.975, df)*np.sqrt(variance/treatment_ss)
    return np.column_stack((estimate, estimate-radius, estimate+radius))


def run(args):
    scored = verification.scored_cohort(args.test_scores)
    ids = list(scored)
    patches = list(scored.values())
    if len(ids) != 88:
        raise ValueError("Expected the complete 88-patch E06 cohort.")
    reference = np.array([p["reference_fraction"] for p in patches])
    raw_proxy = np.array([p["predicted_fraction"] for p in patches])
    standard_reference, _, scaling = e02.standardize(patches)
    eligible = [i for i, p in enumerate(patches) if all(p[k] is not None for k in verification.SCORES)]
    if len(eligible) < max(BUDGETS):
        raise ValueError("Too few commonly scored patches for the declared budgets.")
    rankings = {key: sorted(eligible, key=lambda i: (-patches[i][key], ids[i]))
                for key in verification.SCORES}
    rng = np.random.default_rng(SELECTION_SEED)
    random_order = np.array([rng.permutation(eligible)[:max(BUDGETS)] for _ in range(args.repeats)])
    draws = [e02.simulation_draws(patches, SIMULATION_SEED, rep) for rep in range(args.repeats)]
    variants = {}
    for budget in BUDGETS:
        random_fractions = np.broadcast_to(raw_proxy, (args.repeats, len(ids))).copy()
        reveal = random_order[:, :budget]
        random_fractions[np.arange(args.repeats)[:, None], reveal] = reference[reveal]
        variants[budget, "random"] = random_fractions
        for key, order in rankings.items():
            mixed = raw_proxy.copy()
            mixed[order[:budget]] = reference[order[:budget]]
            variants[budget, key] = mixed
    variants[0, "no_verification"], variants[len(ids), "all_reference"] = raw_proxy, reference

    arrays = {"patch_id": np.asarray(ids), "random_reveals": random_order}
    summaries, comparisons = [], []
    for strength in e02.ASSIGNMENT_STRENGTHS:
        t, y, _ = zip(*(e02.simulate(standard_reference, *draw, 0., strength) for draw in draws), strict=True)
        treatment, outcome = np.asarray(t, dtype=float), np.asarray(y)
        fitted = {key: adjusted_effect(treatment, outcome, fraction) for key, fraction in variants.items()}
        for (budget, policy), estimates in fitted.items():
            arrays[f"a1_{strength:g}_{budget}_{policy}"] = estimates
            chosen = variants[budget, policy]
            for rep in (0, args.repeats-1):
                adjustment = chosen[rep] if chosen.ndim == 2 else chosen
                expected = e05.inference.control(treatment[rep], outcome[rep], adjustment)
                np.testing.assert_allclose(estimates[rep], list(expected.values()), atol=1e-12, rtol=0)
            base = fitted[budget, "random"] if budget in BUDGETS else estimates
            values = list(e06.metrics(np.column_stack((base, estimates))))
            summaries.append(dict(a1=strength, budget=budget, policy=policy, n=args.repeats,
                                  **{name: after for name, _, after, _, _ in values}))
            if budget in BUDGETS and policy != "random":
                for name, before, after, delta, influence in values:
                    se = e06.mcse(influence)
                    comparisons.append(dict(a1=strength, budget=budget, policy=policy, metric=name,
                        n_pairs=args.repeats, random=before, targeted=after, delta=delta, mcse=se,
                        mc95_low=delta-1.96*se if se is not None else None,
                        mc95_high=delta+1.96*se if se is not None else None))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp)/"complete"
        stage.mkdir()
        e02.write_csv(stage/"summary.csv", summaries)
        e02.write_csv(stage/"paired_comparisons.csv", comparisons)
        e02.write_csv(stage/"rankings.csv", [
            dict(policy=policy, rank=rank, patch_id=ids[i], event_id=patches[i]["event_id"],
                 score=patches[i][policy])
            for policy, order in rankings.items() for rank, i in enumerate(order, 1)])
        np.savez_compressed(stage/"repetitions.npz", **arrays)
        e02.write_json(stage/"config.json", dict(
            n_patches=len(ids), eligible=[ids[i] for i in eligible], repeats=args.repeats,
            seed=SIMULATION_SEED, selection_seed=SELECTION_SEED, budgets=BUDGETS, policies=POLICIES,
            scaling=scaling, test_scores_sha256=e05.TEST_SCORES, git_commit=e02.git_commit(),
            estimates_columns=["estimate", "lower95", "upper95"],
            implementation_sha256={p.name: e02.digest(p) for p in (Path(__file__), PROTOCOL,
                Path(verification.__file__), Path(e05.__file__), Path(e06.__file__), Path(e02.__file__),
                Path(e05.inference.__file__))}))
        stage.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-scores", type=Path, default=TEST_SCORES)
    parser.add_argument("--output", type=Path, default=Path("results/causal_e06/fixed_budget_seed42"))
    parser.add_argument("--repeats", type=int, default=e05.TOTAL)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.repeats <= e05.TOTAL:
        parser.error(f"Use a new output directory and 2–{e05.TOTAL} repetitions.")
    with threadpool_limits(limits=1):
        run(args)
