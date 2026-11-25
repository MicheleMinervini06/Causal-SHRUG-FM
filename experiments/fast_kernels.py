"""Indexed CPU quadrature for the E05/E06 joint likelihood (float64, no fastmath)."""

import math

import numpy as np
from numba import njit, prange


@njit
def _log_patch(v, t, y, nodes, masses, sigma, randomized):
    residual = y - v[0] - v[1] * t
    sign = 2.0 * t - 1.0
    total = 0.0
    for k in range(len(nodes)):
        c = nodes[k]
        z = np.divide(residual - v[2] * c, sigma)
        likelihood = math.exp(-0.5 * z**2)
        if not randomized:
            eta = sign * (v[4] + v[5] * c)
            likelihood /= 1.0 + math.exp(-eta)
        total += masses[k] * likelihood
    if total >= 1e-290:
        return math.log(total)

    # Tiny weighted sums can lose the Gaussian or assignment factor to underflow.
    log_total = -np.inf
    for k in range(len(nodes)):
        if masses[k] > 0:
            c = nodes[k]
            delta = residual - v[2] * c
            if sigma == 0.0:
                z = math.exp(math.log(abs(delta)) - v[3]) if delta else 0.0
            else:
                z = delta / sigma
            term = math.log(masses[k]) - (0.5 * z) * z
            if not randomized:
                eta = sign * (v[4] + v[5] * c)
                term += min(eta, 0.0) - math.log1p(math.exp(-abs(eta)))
            log_total = np.logaddexp(log_total, term)
    return log_total


@njit(parallel=True, cache=True)
def _indexed(params, t, y, grid, masses, rows, randomized, reduce_patches):
    result = np.empty((len(rows), 1 if reduce_patches else len(t)))
    for s in prange(len(rows)):
        row = rows[s]
        v = params[s]
        sigma = math.exp(v[3])
        total = 0.0
        for i in range(len(t)):
            value = _log_patch(v, t[i], y[i], grid[row, i], masses[row, i], sigma, randomized)
            if reduce_patches:
                total += value
            else:
                result[s, i] = value
        if reduce_patches:
            result[s, 0] = total - len(t) * v[3]
    return result


def _check_inputs(params, t, y, grid, masses, rows, randomized):
    # Validate at the Python boundary before accessing unchecked native array indices.
    if (grid.ndim != 3 or grid.shape != masses.shape or t.ndim != 1 or t.shape != y.shape
            or grid.shape[1] != len(t) or params.ndim != 2 or params.shape[1] < (4 if randomized else 6)
            or rows.shape != (len(params),) or rows.dtype.kind not in "iu"
            or (len(rows) and (rows.min() < 0 or rows.max() >= len(grid)))):
        raise ValueError("Incompatible likelihood arrays or calibration row indices.")


def patch_log_kernels_indexed(params, t, y, grid, masses, rows, randomized):
    _check_inputs(params, t, y, grid, masses, rows, randomized)
    return _indexed(params, t, y, grid, masses, rows, randomized, False)


def patch_log_kernels(params, t, y, grid, masses, randomized):
    return patch_log_kernels_indexed(params, t, y, grid, masses, np.arange(len(params)), randomized)


def log_likelihood_indexed(params, t, y, grid, masses, rows, randomized):
    _check_inputs(params, t, y, grid, masses, rows, randomized)
    return _indexed(params, t, y, grid, masses, rows, randomized, True)[:, 0]
