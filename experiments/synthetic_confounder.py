"""A dependency-free benchmark of confounder measurement error.

C ~ N(0, 1), T ~ Bernoulli(sigmoid(a*C)), Y = T + C + epsilon,
C_hat = C + sigma*Z, with independent standard-normal epsilon and Z.
Every regression includes an intercept. No foundation model is involved.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from math import exp, fsum, isfinite, sqrt
from pathlib import Path
from random import Random
from statistics import fmean, stdev
from typing import Sequence


TRUE_EFFECT = 1.0
ASSIGNMENT_STRENGTHS = (1.0, 0.0)
NOISE_LEVELS = (0.0, 0.5, 1.0)


@dataclass
class Population:
    confounder: list[float]
    treatment: list[float]
    outcome: list[float]
    measurement_noise: list[float]


@dataclass
class Result:
    repetition: int
    seed: int
    n: int
    a: float
    sigma: float
    tau_unadjusted: float
    tau_oracle: float
    tau_proxy: float
    true_effect: float = TRUE_EFFECT


@dataclass
class Summary:
    a: float
    sigma: float
    estimator: str
    mean_effect: float
    bias: float
    rmse: float
    repetitions: int
    mcse_bias: float | None


@dataclass
class Diagnostics:
    a: float
    seed: int
    n: int
    n_treated: int
    n_control: int
    variable_stats: dict[str, tuple[float, float]]
    zero_noise_max_difference: float
    zero_noise_effect_difference: float


def sigmoid(value: float) -> float:
    """Evaluate the logistic function without overflowing exp()."""
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def generate_population(n: int, a: float, seed: int) -> Population:
    """Generate independent noise draws; reuse seeds to pair scenarios.

    Drawing all latent variables before assignment makes C, epsilon, Z and
    treatment uniforms identical across a=0 and a=1 for the same seed.
    """
    rng = Random(seed)
    confounder = [rng.gauss(0.0, 1.0) for _ in range(n)]
    outcome_noise = [rng.gauss(0.0, 1.0) for _ in range(n)]
    measurement_noise = [rng.gauss(0.0, 1.0) for _ in range(n)]
    uniforms = [rng.random() for _ in range(n)]
    treatment = [float(u < sigmoid(a * c)) for c, u in zip(confounder, uniforms)]
    outcome = [
        TRUE_EFFECT * t + c + epsilon
        for t, c, epsilon in zip(treatment, confounder, outcome_noise)
    ]
    return Population(confounder, treatment, outcome, measurement_noise)


def treatment_coefficient(
    outcome: Sequence[float],
    treatment: Sequence[float],
    confounder: Sequence[float] | None = None,
) -> float:
    """Return the OLS coefficient of T in Y ~ 1 + T [+ confounder].

    Centering includes the intercept. For adjustment, residualize T against
    the centered confounder, then use the partial-regression identity.
    This is ordinary OLS, specialized to at most one adjustment variable.
    """
    n = len(outcome)
    if n < 3 or len(treatment) != n:
        raise ValueError("Outcome and treatment need equal lengths of at least three.")
    if confounder is not None and len(confounder) != n:
        raise ValueError("Confounder must have the same length as outcome.")

    y_mean, t_mean = fmean(outcome), fmean(treatment)
    y_centered = [y - y_mean for y in outcome]
    t_centered = [t - t_mean for t in treatment]
    treatment_ss = fsum(t * t for t in t_centered)
    if treatment_ss == 0:
        raise ValueError("Treatment has no variation; increase --n or change --seed.")

    t_residual = t_centered
    if confounder is not None:
        c_mean = fmean(confounder)
        c_centered = [c - c_mean for c in confounder]
        confounder_ss = fsum(c * c for c in c_centered)
        if confounder_ss == 0:
            raise ValueError("Confounder has no variation.")
        slope = fsum(t * c for t, c in zip(t_centered, c_centered)) / confounder_ss
        t_residual = [t - slope * c for t, c in zip(t_centered, c_centered)]

    residual_ss = fsum(t * t for t in t_residual)
    if residual_ss <= 1e-12 * treatment_ss:
        raise ValueError("Treatment is collinear with the adjustment variable.")
    return fsum(t * y for t, y in zip(t_residual, y_centered)) / residual_ss


def run_benchmark(n: int, repeats: int, seed: int) -> list[Result]:
    """Fit all three regressions on paired populations and noise levels."""
    if n < 3 or repeats < 1:
        raise ValueError("n must be at least three and repeats must be positive.")
    results = []
    for repetition in range(repeats):
        population_seed = seed + repetition
        for a in ASSIGNMENT_STRENGTHS:
            population = generate_population(n, a, population_seed)
            y, t, c = population.outcome, population.treatment, population.confounder
            unadjusted = treatment_coefficient(y, t)
            oracle = treatment_coefficient(y, t, c)
            for sigma in NOISE_LEVELS:
                proxy = [
                    value + sigma * noise
                    for value, noise in zip(c, population.measurement_noise)
                ]
                results.append(Result(
                    repetition, population_seed, n, a, sigma,
                    unadjusted, oracle, treatment_coefficient(y, t, proxy),
                ))
    return results


def diagnose_population(population: Population, a: float, seed: int) -> Diagnostics:
    """Check one population; report moments without requiring exact normality."""
    c, t, y, z = (
        population.confounder, population.treatment,
        population.outcome, population.measurement_noise,
    )
    n = len(c)
    if n < 3 or any(len(values) != n for values in (t, y, z)):
        raise ValueError("Diagnostic variables need equal lengths of at least three.")
    if not all(isfinite(value) for values in (c, t, y, z) for value in values):
        raise ValueError("Diagnostic variables contain non-finite values.")
    if not all(value in (0, 1) for value in t):
        raise ValueError("Treatment must be binary.")
    n_treated = int(sum(t))
    if n_treated in (0, n):
        raise ValueError("Diagnostics require both treated and control observations.")

    variables = {
        "C": c,
        "T": t,
        "Y": y,
        "epsilon": [outcome - TRUE_EFFECT * treatment - confounder
                    for outcome, treatment, confounder in zip(y, t, c)],
        "Z": z,
    }
    for sigma in NOISE_LEVELS:
        variables[f"C_hat (sigma={sigma:g})"] = [
            value + sigma * noise for value, noise in zip(c, z)
        ]
    exact_proxy = variables["C_hat (sigma=0)"]
    max_difference = max(abs(proxy - true) for proxy, true in zip(exact_proxy, c))
    effect_difference = abs(
        treatment_coefficient(y, t, exact_proxy) - treatment_coefficient(y, t, c)
    )
    if max_difference != 0 or effect_difference > 1e-12:
        raise ValueError("Zero-noise proxy and oracle checks failed.")
    return Diagnostics(
        a, seed, n, n_treated, n - n_treated,
        {name: (fmean(values), stdev(values)) for name, values in variables.items()},
        max_difference, effect_difference,
    )


def print_diagnostics(diagnostics: list[Diagnostics]) -> None:
    print("Initial diagnostics: first repetition of each treatment setting")
    for diagnostic in diagnostics:
        print(
            f"a={diagnostic.a:g}, seed={diagnostic.seed}, n={diagnostic.n}: "
            f"treated={diagnostic.n_treated}, controls={diagnostic.n_control}"
        )
        print(f"  {'Variable':<20} {'Mean':>10} {'Sample SD':>10}")
        for name, (mean, sd) in diagnostic.variable_stats.items():
            print(f"  {name:<20} {mean:10.4f} {sd:10.4f}")
        print("  PASS: equal lengths, finite values, binary T, both treatment groups.")
        print(
            "  PASS: sigma=0 matches C exactly and matches the oracle coefficient "
            f"(max measurement difference={diagnostic.zero_noise_max_difference:g}, "
            f"coefficient difference={diagnostic.zero_noise_effect_difference:g})."
        )
    print()


def summarize_results(results: list[Result]) -> list[Summary]:
    """Summarize bias, RMSE, and Monte Carlo SE of mean bias per estimator."""
    summaries = []
    for a, sigma in dict.fromkeys((r.a, r.sigma) for r in results):
        group = [r for r in results if r.a == a and r.sigma == sigma]
        for estimator, field in (
            ("No adjustment", "tau_unadjusted"),
            ("Oracle", "tau_oracle"),
            ("Proxy", "tau_proxy"),
        ):
            estimates = [getattr(r, field) for r in group]
            errors = [estimate - r.true_effect for estimate, r in zip(estimates, group)]
            summaries.append(Summary(
                a, sigma, estimator, fmean(estimates), fmean(errors),
                sqrt(fmean(error * error for error in errors)),
                len(estimates),
                stdev(estimates) / sqrt(len(estimates)) if len(estimates) > 1 else None,
            ))
    return summaries


def print_summary(results: list[Result], n: int, repeats: int, seed: int) -> None:
    print(f"Synthetic confounder benchmark | true treatment effect = {TRUE_EFFECT:g}")
    print(f"n = {n:,} per population | repetitions = {repeats} | starting seed = {seed}")
    print("All regressions include an intercept; summaries are across repetitions.")
    print()
    print(f"{'a':>3} {'sigma':>6} {'Estimator':<15} {'Mean effect':>12} {'Bias':>10} {'RMSE':>10} {'MCSE(bias)':>12}")
    for summary in summarize_results(results):
        mcse = f"{summary.mcse_bias:.4f}" if summary.mcse_bias is not None else "N/A"
        print(
            f"{summary.a:3.0f} {summary.sigma:6.1f} {summary.estimator:<15} "
            f"{summary.mean_effect:12.4f} {summary.bias:10.4f} {summary.rmse:10.4f} {mcse:>12}"
        )
    print()
    print("Bias = mean(estimate - true effect); RMSE = sqrt(mean((estimate - true effect)^2)).")
    print("With one repetition, these are signed error and absolute error, respectively.")
    print("MCSE(bias) = sample SD(estimates) / sqrt(repetitions); N/A for one repetition.")
    print("MCSE measures simulation precision, not an individual treatment-effect standard error.")
    print("For a=0, treatment is randomized: all three methods should approach 1.")
    print("For sigma=0, proxy and oracle adjustments are identical.")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write estimates or summaries with full floating-point precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2_000, help="Units per population (default: 2000).")
    parser.add_argument("--repeats", type=int, default=200, help="Independent populations (default: 200).")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed (default: 42).")
    parser.add_argument("--csv", type=Path, help="Optional output path for every repetition's coefficients.")
    parser.add_argument("--summary-csv", type=Path, help="Optional output path for bias, RMSE and Monte Carlo SE summaries.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Check the first population in each treatment setting without running the simulation.")
    args = parser.parse_args()
    if args.n < 3 or args.repeats < 1:
        parser.error("--n must be at least 3 and --repeats must be at least 1")
    if args.diagnostics_only and (args.csv or args.summary_csv):
        parser.error("--diagnostics-only cannot be combined with CSV output options")
    if args.csv and args.summary_csv and args.csv.resolve() == args.summary_csv.resolve():
        parser.error("--csv and --summary-csv must use different paths")
    try:
        diagnostics = [
            diagnose_population(generate_population(args.n, a, args.seed), a, args.seed)
            for a in ASSIGNMENT_STRENGTHS
        ]
        print_diagnostics(diagnostics)
        if args.diagnostics_only:
            return
        results = run_benchmark(args.n, args.repeats, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print_summary(results, args.n, args.repeats, args.seed)
    if args.csv:
        write_csv(args.csv, [asdict(result) for result in results])
    if args.summary_csv:
        write_csv(args.summary_csv, [asdict(summary) for summary in summarize_results(results)])


if __name__ == "__main__":
    main()
