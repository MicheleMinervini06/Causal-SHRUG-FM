"""E05 joint likelihood for treatment, outcome and the frozen calibration posterior."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaincinv, expit
from scipy.stats import multivariate_t, t as student_t

from experiments.fraction_calibration import predictive_components


def latent_grid(model, proxy, scores, method, nodes):
    """Integrate the beta part on its probability scale, retaining the exact atom at zero."""
    q, w = np.polynomial.legendre.leggauss(nodes)
    center, scale = proxy.mean(), proxy.std()
    hurdle = not model.get("positive_only", False)
    draws = len(model["proxy_only"])
    grid = np.empty((draws, len(proxy), nodes + int(hurdle)))
    masses = np.empty_like(grid) if hurdle else np.broadcast_to(w / 2, grid.shape)
    for i, (value, features) in enumerate(zip(proxy, scores, strict=True)):
        present, a, b = predictive_components(model, value, features, method)
        grid[:, i, int(hurdle):] = betaincinv(a[:, None], b[:, None], (q + 1) / 2)
        if hurdle:
            grid[:, i, 0] = 0.
            masses[:, i, 0] = 1 - present
            masses[:, i, 1:] = present[:, None] * (w / 2)
    grid -= center
    grid /= scale
    if not np.isfinite(grid).all() or not np.isfinite(masses).all():
        raise ValueError("Non-finite calibration quadrature.")
    return grid, masses


def ols(t, y, c):
    x = np.column_stack((np.ones(len(t)), t, c))
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 3:
        raise ValueError("Nonidentifiable treatment regression.")
    variance = np.sum((y - x @ beta)**2) / (len(t) - 3)
    cov = variance * np.linalg.inv(x.T @ x)
    return beta, cov, variance


def control(t, y, c):
    beta, cov, _ = ols(t, y, c)
    radius = student_t.ppf(.975, len(t) - 3) * np.sqrt(cov[1, 1])
    return {"estimate": beta[1], "lower95": beta[1] - radius, "upper95": beta[1] + radius}


def proposal_distribution(t, y, mean_c, randomized):
    # This plug-in fit only chooses where to propose; the joint likelihood sets all weights.
    beta, cov, variance = ols(t, y, mean_c)
    loc = [*beta, .5 * np.log(variance)]
    shape = np.zeros((4 if randomized else 6,) * 2)
    shape[:3, :3] = cov
    shape[3, 3] = .5 / (len(t) - 3)
    if not randomized:
        design = np.column_stack((np.ones(len(t)), mean_c))

        def logistic(v):
            eta = design @ v
            return (np.sum(np.logaddexp(0, eta) - t * eta) + .5 * (v @ v) / 2.5**2,
                    design.T @ (expit(eta) - t) + v / 2.5**2)

        fit = minimize(logistic, np.zeros(2), jac=True, method="BFGS")
        if not fit.success:
            raise ValueError(f"Assignment proposal failed: {fit.message}")
        p = expit(design @ fit.x)
        shape[4:, 4:] = np.linalg.inv(design.T @ ((p * (1-p))[:, None] * design) + np.eye(2) / 2.5**2)
        loc.extend(fit.x)
    return multivariate_t(loc=loc, shape=shape * 1.2, df=7)


def patch_log_kernels(params, t, y, grid, masses, randomized):
    """Integrated patch kernels without the shared Gaussian normalization."""
    v = params[:, :, None, None]
    sigma = np.exp(v[:, 3])
    resid = (y[:, None] - v[:, 0] - v[:, 1] * t[:, None] - v[:, 2] * grid) / sigma
    likelihood = np.exp(-.5 * resid**2)
    if not randomized:
        likelihood *= expit((2*t[:, None] - 1) * (v[:, 4] + v[:, 5] * grid))
    # Constant Gaussian and randomized-treatment normalizers cancel in the weights.
    with np.errstate(divide="ignore"):
        return np.log((likelihood * masses).sum(axis=2))


def log_likelihood(params, t, y, grid, masses, randomized):
    """Integrate each latent C first, then multiply patch likelihoods within a shared draw."""
    return patch_log_kernels(params, t, y, grid, masses, randomized).sum(axis=1) - len(t) * params[:, 3]


def row_probabilities(center, t, y, grid, masses, randomized):
    # Tilt the row proposal toward calibrations supported by this T,Y. A uniform
    # component preserves support; the weight below restores the frozen row prior.
    log_row = np.empty(len(grid))
    for start in range(0, len(grid), 128):
        sl = slice(start, start + 128)
        params = np.broadcast_to(center, (len(grid[sl]), len(center)))
        log_row[sl] = log_likelihood(params, t, y, grid[sl], masses[sl], randomized)
    row_p = np.exp(.7 * (log_row - log_row.max()))
    return .8 * row_p / row_p.sum() + .2 / len(grid)


def bayes(t, y, grid, masses, mean_c, randomized, seed, *, return_proposal=False):
    initial = proposal_distribution(t, y, mean_c, randomized)
    proposal = initial
    rng = np.random.default_rng(seed)
    prior_sd = np.array([5., 5., 5., 1.] + ([] if randomized else [2.5, 2.5]))
    attempted = 0
    for step in range(8):
        size = 2048 if step == 0 else 2048 * 2**(step-1)
        attempted += size
        row_p = row_probabilities(proposal.loc, t, y, grid, masses, randomized)
        params = proposal.rvs(size=size, random_state=rng)
        indices = rng.choice(len(grid), size=size, p=row_p)
        log_w = -.5 * np.sum((params / prior_sd)**2, axis=1) - proposal.logpdf(params)
        log_w -= np.log(len(grid) * row_p[indices])
        for start in range(0, size, 128):
            sl = slice(start, start + 128)
            log_w[sl] += log_likelihood(params[sl], t, y, grid[indices[sl]], masses[indices[sl]], randomized)
        w = np.exp(log_w - np.max(log_w))
        w /= w.sum()
        ess = 1 / (w @ w)
        if ess >= 1000 and w.max() < .02:
            order = np.argsort(params[:, 1])
            tau, weights = params[order, 1], w[order]
            lo, hi = np.interp([.025, .975], np.cumsum(weights) - .5 * weights, tau)
            result = {"estimate": tau @ weights, "lower95": lo, "upper95": hi, "ess": ess,
                      "proposals": attempted, "retained_proposals": size, "max_weight": w.max()}
            if return_proposal:
                return result, proposal, row_p
            return result
        # Failed batches train the next proposal only. Keep a covariance floor
        # from the plug-in fit so a single dominant draw cannot collapse it.
        center = np.average(params, axis=0, weights=w)
        cov = np.cov(params, rowvar=False, aweights=w, ddof=0)
        blend = min(.95, ess / (ess + 30))
        proposal = multivariate_t(loc=blend*center + (1-blend)*proposal.loc,
                                  shape=blend*cov + (1-blend)*initial.shape, df=7)
    raise ValueError(f"Importance sampling unresolved after {attempted} proposals "
                     f"(last batch ESS={ess:.1f}, maximum weight={w.max():.4g}).")
