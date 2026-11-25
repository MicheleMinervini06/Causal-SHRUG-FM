import unittest

import numpy as np
from scipy.special import logsumexp

from experiments import fast_kernels as fk


class CompiledLikelihoodTests(unittest.TestCase):
    def test_zero_residual_when_sigma_rounds_to_zero(self):
        params = np.array([[0., 1., 1., -750.]])
        grid = np.array([[[0., 1.]]])
        masses = np.array([[[.25, .75]]])
        result = fk.patch_log_kernels(params, np.array([1]), np.array([1.]), grid, masses, True)
        np.testing.assert_allclose(result, [[np.log(.25)]], rtol=0., atol=1e-12)

    def test_indexed_and_aligned_match_log_space_quadrature(self):
        rng = np.random.default_rng(76)
        grid = rng.uniform(.001, .999, size=(7, 9, 16))
        rows = np.array([5, 1, 5, 0, 6, 2])
        params = rng.normal(size=(len(rows), 6))
        params[:, 3] = 0.
        params[1, :3] = (48., 0., 0.)  # all Gaussian factors round to zero
        params[3, 4:] = (-720., 0.)     # assignment factor in the denormal range
        t = np.arange(9) % 2
        y = np.linspace(-.5, .5, 9)
        weights = np.polynomial.legendre.leggauss(16)[1] / 2
        positive = np.broadcast_to(weights, grid.shape)
        hurdle = rng.dirichlet(np.ones(16), size=grid.shape[:2])
        hurdle[:, 0, 0] = 0.
        hurdle[5, 1] = 0.  # explicit all-zero mass case
        for masses in (positive, hurdle):
            if masses is hurdle:
                grid[:, :, 0] = 0.
            for randomized in (True, False):
                with self.subTest(broadcast=masses is positive, randomized=randomized):
                    c = grid[rows]
                    v = params[:, :, None, None]
                    log_kernel = -.5 * ((y[:, None]-v[:, 0]-t[:, None]*v[:, 1]-v[:, 2]*c)/np.exp(v[:, 3]))**2
                    if not randomized:
                        eta = (2*t[:, None]-1) * (v[:, 4]+v[:, 5]*c)
                        log_kernel -= np.logaddexp(0., -eta)
                    with np.errstate(divide="ignore"):
                        reference = logsumexp(np.log(masses[rows]) + log_kernel, axis=2)
                    actual = fk.patch_log_kernels_indexed(params, t, y, grid, masses, rows, randomized)
                    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(actual, fk.patch_log_kernels(params, t, y, c, masses[rows], randomized))
                    reduced = fk.log_likelihood_indexed(params, t, y, grid, masses, rows, randomized)
                    np.testing.assert_allclose(reduced, reference.sum(axis=1)-len(t)*params[:, 3],
                                               rtol=2e-15, atol=1e-12)
