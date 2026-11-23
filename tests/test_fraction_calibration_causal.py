from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from experiments import fraction_calibration_causal as phase2
from experiments import fraction_causal_posterior as inference


class PhaseTwoTests(unittest.TestCase):
    def test_failed_batch_adapts_both_proposals_then_uses_fresh_draws(self):
        rng = np.random.default_rng(43)
        c = rng.normal(size=32)
        t = rng.binomial(1, .5, len(c))
        y = t + c + rng.normal(size=len(c))
        # A point-mass calibration makes the oracle a check on the fitted effect.
        grid = np.tile(c[None, :, None], (4, 1, 1))
        masses = np.ones_like(grid)
        initial = inference.proposal_distribution(t, y, c, True)
        loc = initial.loc.copy()
        loc[1] += 1.0
        shifted = inference.multivariate_t(loc=loc, shape=initial.shape, df=7)
        with patch.object(inference, "proposal_distribution", return_value=shifted), \
                patch.object(inference, "row_probabilities", wraps=inference.row_probabilities) as row_updates:
            result = inference.bayes(t, y, grid, masses, c, True, 42)
        self.assertGreater(row_updates.call_count, 1)
        self.assertFalse(np.array_equal(row_updates.call_args_list[0].args[0], row_updates.call_args.args[0]))
        self.assertGreater(result["proposals"], result["retained_proposals"])
        self.assertTrue(1000 <= result["ess"] <= result["retained_proposals"])
        self.assertLess(result["max_weight"], .02)
        self.assertAlmostEqual(result["estimate"], inference.control(t, y, c)["estimate"], delta=.08)

    def test_quadrature_retains_zero_mass_and_shared_missing_score_fallback(self):
        model = {"center": np.zeros(4), "scale": np.ones(4), "presence": np.zeros((2, 2)),
                 "proxy_only": np.zeros((2, 4)), "with_scores": np.zeros((2, 10))}
        model["with_scores"][:, 0] = [-1., 1.]
        proxy = np.array([.1, .6])
        scores = np.array([[0., 1., 0.], [np.nan, np.nan, 1.]])
        grid, masses = inference.latent_grid(model, proxy, scores, "with_scores", 32)
        fractions = grid * proxy.std() + proxy.mean()
        np.testing.assert_allclose(fractions[:, :, 0], 0, atol=1e-15)
        np.testing.assert_allclose(masses[:, :, 0], .5)
        np.testing.assert_allclose(masses.sum(axis=2), 1)
        means = (masses * fractions).sum(axis=2)
        np.testing.assert_allclose(means[:, 0], .5 * expit([-1., 1.]), atol=.0001)
        np.testing.assert_allclose(means[:, 1], .25, atol=.0001)

    def test_joint_likelihood_integrates_patches_before_shared_calibration(self):
        t = np.array([1, 0])
        y = np.array([.8, -.5])
        grid = np.array([[[0., .5], [0., .9]], [[0., .8], [0., .6]]])
        mass = np.array([[[.6, .4], [.2, .8]], [[.1, .9], [.7, .3]]])
        params = np.array([[.2, .7, .9, .3, -.7, .8]] * 2)
        for randomized in (False, True):
            result = inference.log_likelihood(params, t, y, grid, mass, randomized)
            for draw in range(2):
                product = 1.
                for i in range(2):
                    c = grid[draw, i]
                    kernel = np.exp(-.5*((y[i]-.2-.7*t[i]-.9*c)/np.exp(.3))**2) / np.exp(.3)
                    if not randomized:
                        kernel *= expit((2*t[i]-1)*(-.7+.8*c))
                    product *= mass[draw, i] @ kernel
                self.assertAlmostEqual(result[draw], np.log(product))

    def test_paired_summary_and_incomplete_shard(self):
        rows = [{"repetition": i, "method": m, "a1": a, "estimate": 1.1 if m == "with_scores" else 1.2,
                 "lower95": .9 if m == "with_scores" else 1.1, "upper95": 1.4}
                for i in range(2) for a in (0., 1.) for m in phase2.METHODS]
        summary, pairs = phase2.summarize(rows)
        self.assertEqual(len(summary), 8)
        diffs = {r["metric"]: r["with_minus_without_scores"] for r in pairs}
        for name, expected in (("bias", -.1), ("rmse", -.1), ("coverage95", 1.), ("width95", .2)):
            self.assertAlmostEqual(diffs[name], expected)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2.publish(root / "shard-0", rows, {"start": 0, "repeats": 2, "protocol": {}})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                phase2.collect(root, root / "combined", 3)
            self.assertFalse((root / "combined").exists())
            phase2.collect(root, root / "combined", 2)
            self.assertTrue((root / "combined" / "summary.csv").exists())
