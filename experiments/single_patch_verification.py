"""E06: condition the E05 posterior on one measured fraction at a time."""

import numpy as np
from scipy.special import betaln, log_expit, logsumexp

from experiments import fast_kernels as fk
from experiments import fraction_calibration as calibration
from experiments import fraction_causal_posterior as inference


# Absolute tolerances between two independent integration estimates of the CHANGE.
# Units are treatment-effect units (the true effect is 1), also for interval width.
MEAN_TOL, WIDTH_TOL = .005, .02
BATCH_SIZES = tuple(4096 * 2**level for level in range(8))
CHUNK = 8192


def reference_evidence(model, proxy, scores, reference, method):
    """Log mass at zero or log density at a positive reference, per shared calibration row."""
    columns = []
    for x, s, c in zip(proxy, scores, reference, strict=True):
        p, a, b = calibration.predictive_components(model, x, s, method)
        if c == 0:
            column = np.log1p(-p)
        elif 0 < c < 1:
            column = np.log(p) + (a-1)*np.log(c) + (b-1)*np.log1p(-c) - betaln(a, b)
        else:
            raise ValueError("Reference outside the support of the frozen hurdle beta model.")
        columns.append(column)
    log_f = np.column_stack(columns)
    if not np.isfinite(logsumexp(log_f, axis=0)).all():
        raise ValueError("Reference has no support in the calibration bank.")
    return log_f


def batch(proposal, row_p, grid, masses, log_f, z, t, y, randomized, ids, size, rng):
    """Evaluate no reveal and each separate reveal on the same fresh importance proposals."""
    params = proposal.rvs(size=size, random_state=rng)
    indices = rng.choice(len(grid), size=size, p=row_p)
    prior_sd = [5., 5., 5., 1.] + ([] if randomized else [2.5, 2.5])
    log_base = -.5 * np.sum((params / prior_sd)**2, axis=1) - proposal.logpdf(params)
    log_base -= np.log(len(grid) * row_p[indices])
    log_base -= len(t) * params[:, 3]
    weights = np.empty((size, 1 + len(ids)))
    for start in range(0, size, CHUNK):
        sl = slice(start, start + CHUNK)
        rows, v = indices[sl], params[sl, :, None]
        marginal = fk.patch_log_kernels_indexed(params[sl], t, y, grid, masses, rows, randomized)
        base = log_base[sl] + marginal.sum(axis=1)
        exact = -.5*((y[ids] - v[:, 0] - v[:, 1]*t[ids] - v[:, 2]*z[ids]) / np.exp(v[:, 3]))**2
        if not randomized:
            exact += log_expit((2*t[ids]-1)*(v[:, 4] + v[:, 5]*z[ids]))
        weights[sl, 0] = base
        # Replace ONE integrated factor by f_theta(C_i) * L(T_i,Y_i | C_i).
        weights[sl, 1:] = base[:, None] + exact - marginal[:, ids] + log_f[rows[:, None], ids]
    order = np.argsort(params[:, 1])
    tau = params[order, 1]
    weights -= logsumexp(weights, axis=0)
    weights = np.exp(weights[order])
    estimates = weights.T @ tau
    ess, peaks = 1 / np.einsum("ij,ij->j", weights, weights), weights.max(axis=0)
    limits = np.array([np.interp([.025, .975], np.cumsum(w)-.5*w, tau) for w in weights.T])
    return np.column_stack((estimates, limits)), ess, peaks


def verify(t, y, grid, masses, mean_c, log_f, z, randomized, rep):
    seed = [105, rep, int(not randomized)]
    baseline, proposal, base_p = inference.bayes(t, y, grid, masses, mean_c, randomized, seed, return_proposal=True)
    pending = np.arange(len(t))
    results = {}
    for level, size in enumerate(BATCH_SIZES):
        # Target difficult reference rows only on retries; importance correction
        # keeps all before/after targets identical regardless of proposal choice.
        row_p = base_p
        if level:
            target = np.log(base_p[:, None]) + log_f[:, pending]
            tilted = np.exp(target - logsumexp(target, axis=0))
            row_p = .5 * base_p + .5 * tilted.mean(axis=1)
        runs = [batch(proposal, row_p, grid, masses, log_f, z, t, y, randomized, pending, size,
                      np.random.default_rng([*seed, 606, level, stream])) for stream in (0, 1)]
        estimates = np.stack([r[0] for r in runs])
        change = estimates[:, 1:] - estimates[:, :1]
        mean_gap = abs(change[0, :, 0] - change[1, :, 0])
        width_change = change[:, :, 2] - change[:, :, 1]
        width_gap = abs(width_change[0] - width_change[1])
        minimum = np.minimum(runs[0][1], runs[1][1])
        maximum = np.maximum(runs[0][2], runs[1][2])
        ess, peak = np.minimum(minimum[0], minimum[1:]), np.maximum(maximum[0], maximum[1:])
        passed = (ess >= 1000) & (peak < .02) & (mean_gap <= MEAN_TOL) & (width_gap <= WIDTH_TOL)
        mean = estimates.mean(axis=0)
        for k, patch in enumerate(pending):
            if passed[k] or level == len(BATCH_SIZES)-1:
                results[patch] = {"resolved": int(passed[k]), "batches_per_stream": level+1,
                                  "final_draws_per_stream": size, "min_ess": ess[k], "max_weight": peak[k],
                                  "mean_shift_seed_gap": mean_gap[k], "width_shift_seed_gap": width_gap[k],
                                  **dict(zip(("before", "before_lo", "before_hi"), mean[0])),
                                  **dict(zip(("after", "after_lo", "after_hi"), mean[k+1]))}
        pending = pending[~passed]
        if not len(pending):
            break
    return baseline, results
