"""Numerical and scientific checks for the synthetic causal benchmark."""

import unittest
from dataclasses import replace
from statistics import fmean

from experiments.synthetic_confounder import (
    Result,
    diagnose_population,
    generate_population,
    run_benchmark,
    sigmoid,
    summarize_results,
    treatment_coefficient,
)


class RegressionTests(unittest.TestCase):
    def test_known_coefficients_with_intercept(self):
        treatment = [0, 1, 0, 1, 0, 1]
        confounder = [-2, -1, 0, 0, 1, 2]
        outcome = [7 + 1.5 * t + 2 * c for t, c in zip(treatment, confounder)]
        self.assertAlmostEqual(treatment_coefficient(outcome, treatment, confounder), 1.5)
        # For binary T, unadjusted OLS equals the difference in group means.
        difference = fmean(y for y, t in zip(outcome, treatment) if t == 1) - fmean(
            y for y, t in zip(outcome, treatment) if t == 0
        )
        self.assertAlmostEqual(treatment_coefficient(outcome, treatment), difference)
        shifted = [100 + 3 * c for c in confounder]
        self.assertAlmostEqual(treatment_coefficient(outcome, treatment, shifted), 1.5)

    def test_degenerate_regressions_are_rejected(self):
        with self.assertRaises(ValueError):
            treatment_coefficient([1, 2, 3], [0, 0, 0])
        with self.assertRaises(ValueError):
            treatment_coefficient([1, 2, 3], [0, 1, 0], [0, 1, 0])
        with self.assertRaises(ValueError):
            treatment_coefficient([1, 2, 3], [0, 1, 0], [1, 2])

    def test_assignment_function(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1000), 1.0)
        self.assertEqual(sigmoid(-1000), 0.0)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_benchmark(n=2_000, repeats=40, seed=42)

    def test_pairing_and_reproducibility(self):
        confounded = generate_population(100, a=1, seed=17)
        randomized = generate_population(100, a=0, seed=17)
        self.assertEqual(confounded, generate_population(100, a=1, seed=17))
        self.assertEqual(confounded.confounder, randomized.confounder)
        self.assertEqual(confounded.measurement_noise, randomized.measurement_noise)

    def test_no_measurement_error_matches_oracle_exactly(self):
        for result in self.results:
            if result.sigma == 0:
                self.assertEqual(result.tau_proxy, result.tau_oracle)

    def test_confounded_population_has_residual_proxy_bias(self):
        groups = {
            sigma: [r for r in self.results if r.a == 1 and r.sigma == sigma]
            for sigma in (0, 0.5, 1)
        }
        self.assertGreater(fmean(r.tau_unadjusted for r in groups[0]), 1.7)
        oracle = next(row for row in summarize_results(groups[0]) if row.estimator == "Oracle")
        # Four MCSEs is a regression-test tolerance, not proof of unbiasedness.
        self.assertLessEqual(abs(oracle.bias), 4 * oracle.mcse_bias)
        proxy_means = [fmean(r.tau_proxy for r in groups[s]) for s in (0, 0.5, 1)]
        self.assertGreater(proxy_means[1], proxy_means[0] + 0.1)
        self.assertGreater(proxy_means[2], proxy_means[1] + 0.1)

    def test_randomized_control_recovers_effect(self):
        randomized = [r for r in self.results if r.a == 0]
        for summary in summarize_results(randomized):
            self.assertLessEqual(abs(summary.bias), 4 * summary.mcse_bias)


class SummaryTests(unittest.TestCase):
    def test_bias_and_rmse_for_each_estimator(self):
        results = [
            Result(0, 42, 100, 1, 0.5, 0, 1, 2),
            Result(1, 43, 100, 1, 0.5, 2, 1, 4),
        ]
        summaries = {row.estimator: row for row in summarize_results(results)}
        self.assertEqual(set(summaries), {"No adjustment", "Oracle", "Proxy"})
        # Opposite errors cancel in bias, but remain in RMSE.
        self.assertEqual(summaries["No adjustment"].bias, 0)
        self.assertEqual(summaries["No adjustment"].rmse, 1)
        self.assertEqual(summaries["Oracle"].bias, 0)
        self.assertEqual(summaries["Oracle"].rmse, 0)
        self.assertEqual(summaries["Proxy"].mean_effect, 3)
        self.assertEqual(summaries["Proxy"].bias, 2)
        self.assertAlmostEqual(summaries["Proxy"].rmse, 5 ** 0.5)
        self.assertEqual(summaries["No adjustment"].repetitions, 2)
        self.assertAlmostEqual(summaries["No adjustment"].mcse_bias, 1)
        self.assertEqual(summaries["Oracle"].mcse_bias, 0)
        self.assertAlmostEqual(summaries["Proxy"].mcse_bias, 1)

    def test_scenarios_are_summarized_separately(self):
        results = [
            Result(0, 42, 100, 1, 0, 1, 1, 1),
            Result(0, 42, 100, 1, 1, 1, 1, 3),
            Result(0, 42, 100, 0, 1, 1, 1, 0),
        ]
        summaries = summarize_results(results)
        self.assertEqual(len(summaries), 9)
        proxies = {(row.a, row.sigma): row for row in summaries if row.estimator == "Proxy"}
        self.assertEqual(proxies[(1, 0)].bias, 0)
        self.assertEqual(proxies[(1, 1)].bias, 2)
        self.assertEqual(proxies[(0, 1)].bias, -1)
        self.assertEqual(proxies[(0, 1)].rmse, 1)
        self.assertIsNone(proxies[(0, 1)].mcse_bias)


class DiagnosticTests(unittest.TestCase):
    def test_counts_variables_and_zero_noise_agreement(self):
        for a in (0, 1):
            population = generate_population(100, a=a, seed=42)
            diagnostic = diagnose_population(population, a=a, seed=42)
            self.assertEqual(diagnostic.n_treated + diagnostic.n_control, 100)
            self.assertGreater(diagnostic.n_treated, 0)
            self.assertGreater(diagnostic.n_control, 0)
            self.assertEqual(diagnostic.n_treated, sum(population.treatment))
            self.assertEqual(set(diagnostic.variable_stats), {
                "C", "T", "Y", "epsilon", "Z",
                "C_hat (sigma=0)", "C_hat (sigma=0.5)", "C_hat (sigma=1)",
            })
            self.assertEqual(diagnostic.variable_stats["C"],
                             diagnostic.variable_stats["C_hat (sigma=0)"])
            self.assertEqual(diagnostic.zero_noise_max_difference, 0)
            self.assertEqual(diagnostic.zero_noise_effect_difference, 0)

    def test_invalid_data_are_rejected(self):
        population = generate_population(100, a=1, seed=42)
        invalid = [
            replace(population, outcome=[float("nan")] + population.outcome[1:]),
            replace(population, treatment=[2] + population.treatment[1:]),
            replace(population, treatment=[1] * 100),
            replace(population, measurement_noise=population.measurement_noise[:-1]),
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                diagnose_population(data, a=1, seed=42)


if __name__ == "__main__":
    unittest.main()
