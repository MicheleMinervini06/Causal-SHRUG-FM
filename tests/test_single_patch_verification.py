import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit
from scipy.stats import beta, multivariate_t

from experiments import single_patch_verification as verification
from experiments import burn_single_patch as burn
from experiments.single_patch_causal import metrics


class VerificationTests(unittest.TestCase):
    def test_burn_cache_matches_direct_updates_and_rejects_damage(self):
        rng = np.random.default_rng(42)
        proxy = np.array([0., .2, .6, .05])
        reference = np.array([.03, .3, .5, .04])
        scores = rng.normal(size=(4, 3))
        scores[0, :2] = np.nan
        model = {"center": np.zeros(4), "scale": np.ones(4), "positive_only": True,
                 "proxy_only": rng.normal(0, .1, (6, 4)), "with_scores": rng.normal(0, .1, (6, 10))}
        with TemporaryDirectory() as tmp:
            cache, posterior = Path(tmp)/"cache", Path(tmp)/"posterior.npz"
            np.savez(posterior, **model)
            patches = [{"reference_fraction": value} for value in reference]
            burn.prepare(cache, posterior, patches, proxy, scores, {"nodes": 16})
            loaded, ess = burn.load_cache(cache, {"nodes": 16})
            for i, method in enumerate(burn.replay.METHODS):
                direct, expected_ess = burn.replay.prepare_method(model, proxy, scores, reference, method)
                np.testing.assert_array_equal(ess[i], expected_ess)
                for old, new in zip(direct, loaded[method], strict=True):
                    np.testing.assert_array_equal(old, new)
                    self.assertFalse(new.flags.writeable)
                t = np.array([0, 1, 1, 0])
                proposal = multivariate_t(loc=np.zeros(6), shape=.1*np.eye(6), df=7)
                def update(arrays):
                    grid, mass, _, evidence = arrays
                    return verification.batch(proposal, np.full(6, 1/6), grid, mass, evidence,
                                              (reference-proxy.mean())/proxy.std(), t, t+reference,
                                              False, np.arange(4), 256, np.random.default_rng(19))
                for old, new in zip(update(direct), update(loaded[method]), strict=True):
                    np.testing.assert_array_equal(old, new)
            del loaded
            with self.assertRaisesRegex(ValueError, "does not match"):
                burn.load_cache(cache, {"nodes": 32})
            (cache/"proxy_only-grid.npy").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Damaged"):
                burn.load_cache(cache, {"nodes": 16})

    def test_reference_likelihood_uses_zero_atom_and_positive_density(self):
        model = {"center": np.zeros(4), "scale": np.ones(4), "presence": np.array([[0., 0.], [1., 0.]]),
                 "proxy_only": np.array([[0., 0., 0., 0.], [.2, 0., 0., 0.]])}
        proxy, scores, ref = np.array([.1, .3]), np.zeros((2, 3)), np.array([0., .42])
        evidence = np.exp(verification.reference_evidence(model, proxy, scores, ref, "proxy_only"))
        p, a, b = verification.calibration.predictive_components(model, proxy[1], scores[1], "proxy_only")
        np.testing.assert_allclose(evidence[:, 0], 1-p)
        np.testing.assert_allclose(evidence[:, 1], p*beta.pdf(.42, a, b))

    def test_update_equals_direct_product_leaving_other_patch_latent(self):
        t, y, z = np.array([0, 1]), np.array([-.3, 1.1]), np.array([-.1, .9])
        grid = np.array([[[0., .2], [0., .4]], [[0., .8], [0., .7]]])
        mass = np.array([[[.3, .7], [.2, .8]], [[.1, .9], [.6, .4]]])
        f, row_p = np.array([[.3, .12], [.1, .81]]), np.array([.25, .75])
        for randomized in (True, False):
            prior_sd = np.array([5., 5., 5., 1.] + ([] if randomized else [2.5, 2.5]))
            prop = multivariate_t(loc=np.zeros(len(prior_sd)), shape=.1*np.eye(len(prior_sd)), df=7)
            result, _, _ = verification.batch(prop, row_p, grid, mass, np.log(f), z, t, y, randomized,
                                              np.arange(2), 400, np.random.default_rng(85))
            rng = np.random.default_rng(85)
            params = prop.rvs(size=400, random_state=rng)
            rows = rng.choice(2, size=400, p=row_p)
            weights = np.empty((400, 3))
            for j, (v, r) in enumerate(zip(params, rows)):
                likelihood, known = [], []
                for i in range(2):
                    c = np.append(grid[r, i], z[i])
                    kernel = np.exp(-.5*((y[i]-v[0]-v[1]*t[i]-v[2]*c)/np.exp(v[3]))**2-v[3])
                    if not randomized:
                        kernel *= expit((2*t[i]-1)*(v[4]+v[5]*c))
                    likelihood.append(mass[r, i] @ kernel[:-1])
                    known.append(f[r, i]*kernel[-1])
                w = np.exp(-.5*np.sum((v/prior_sd)**2)-prop.logpdf(v))/(2*row_p[r])
                weights[j] = w*np.array([np.prod(likelihood), known[0]*likelihood[1], likelihood[0]*known[1]])
            np.testing.assert_allclose(result[:, 0], np.average(np.tile(params[:, 1, None], (1, 3)),
                                                                axis=0, weights=weights), atol=1e-12)

    def test_paired_metrics_and_empty_group(self):
        rows = np.array([[1., .5, 1.5, 1.2, .4, 1.6], [2., 1.2, 2.8, 1.4, .8, 2.]])
        results = {k: d for k, _, _, d, _ in metrics(rows)}
        self.assertAlmostEqual(results["bias"], -.2)
        self.assertAlmostEqual(results["rmse"], np.sqrt(.1)-np.sqrt(.5))
        self.assertAlmostEqual(results["coverage95"], .5)
        self.assertTrue(all(d is None for _, _, _, d, _ in metrics(rows[:0])))
