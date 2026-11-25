"""FP64 CUDA quadrature for the E07 positive-beta, confounded benchmark only."""

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _patches(params, rows, grid, masses, treatment, outcome, result,
             N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    draw = tl.program_id(1)
    patch = tl.program_id(0) * B + tl.arange(0, B)
    node = tl.arange(0, K)
    row = tl.load(rows + draw)
    offset = (row * N + patch[:, None]) * K + node[None, :]
    c = tl.load(grid + offset, mask=patch[:, None] < N, other=0.)
    mass = tl.load(masses + node)
    t = tl.load(treatment + patch, mask=patch < N, other=0.)
    y = tl.load(outcome + patch, mask=patch < N, other=0.)
    v = params + draw * 6
    residual = y - tl.load(v) - tl.load(v + 1) * t
    delta = residual[:, None] - tl.load(v + 2) * c
    log_sigma = tl.load(v + 3)
    sigma = libdevice.exp(log_sigma)
    z = tl.div_rn(delta, sigma)
    eta = (2. * t - 1.)[:, None] * (tl.load(v + 4) + tl.load(v + 5) * c)
    prob = tl.div_rn(libdevice.exp(-0.5 * z * z), 1. + libdevice.exp(-eta))
    total = tl.sum(mass[None, :] * prob, axis=1)
    value = libdevice.log(total)
    if tl.min(tl.where(total >= 1e-290, 1, 0), axis=0) == 0:
        if sigma == 0.:
            z = tl.where(delta == 0., 0., libdevice.exp(libdevice.log(tl.abs(delta)) - log_sigma))
        term = (libdevice.log(mass)[None, :] - 0.5 * z * z
                + tl.minimum(eta, 0.) - libdevice.log1p(libdevice.exp(-tl.abs(eta))))
        peak = tl.max(term, axis=1)
        shift = tl.where(peak == -float("inf"), 0., peak)
        stable = shift + libdevice.log(tl.sum(libdevice.exp(term - shift[:, None]), axis=1))
        value = tl.where(total >= 1e-290, value, stable)
    tl.store(result + draw * N + patch, value, mask=patch < N)


class BurnGPU:
    def __init__(self, grids):
        self.banks = {}
        for grid, masses, _, _ in grids.values():
            if (grid.dtype != np.float64 or grid.shape[-1] != 16 or not grid.flags.c_contiguous
                    or masses.shape != grid.shape or masses.strides[:2] != (0, 0)):
                raise ValueError("Expected a contiguous E07 grid and shared 16-node quadrature weights.")
            self.banks[id(grid)] = (torch.tensor(grid, device="cuda"),
                                    torch.tensor(masses[0, 0], dtype=torch.float64, device="cuda"))

    def _evaluate(self, params, t, y, grid, rows, randomized, reduce):
        if randomized or params.shape != (len(rows), 6):
            raise ValueError("The benchmark supports only confounded E07 proposals.")
        bank, mass = self.banks[id(grid)]
        t_gpu, y_gpu = [torch.tensor(x, dtype=torch.float64, device="cuda") for x in (t, y)]
        shape = (len(rows),) if reduce else (len(rows), len(t))
        result = np.empty(shape, dtype=np.float64)
        for start in range(0, len(rows), 8192):
            sl = slice(start, start + 8192)
            v = torch.tensor(np.ascontiguousarray(params[sl]), device="cuda")
            index = torch.tensor(rows[sl], dtype=torch.int64, device="cuda")
            out = torch.empty((len(v), len(t)), device="cuda", dtype=torch.float64)
            _patches[(triton.cdiv(len(t), 16), len(v))](
                v, index, bank, mass, t_gpu, y_gpu, out, len(t), 16, 16, enable_fp_fusion=False)
            if reduce:
                out = out.sum(dim=1) - len(t) * v[:, 3]
            result[sl] = out.cpu().numpy()
        return result

    def patch_log_kernels_indexed(self, params, t, y, grid, masses, rows, randomized):
        return self._evaluate(params, t, y, grid, rows, randomized, False)

    def log_likelihood_indexed(self, params, t, y, grid, masses, rows, randomized):
        return self._evaluate(params, t, y, grid, rows, randomized, True)
