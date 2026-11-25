"""Replay two completed E07 CPU repetitions on one GPU, outside the production run."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter
from unittest.mock import patch

import numpy as np

from experiments import burn_single_patch as burn
from experiments import fast_kernels as cpu
from experiments import single_patch_causal as replay


REPS = (4, 5)


def compare(reference, candidate, rep):
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        if int(a["repetition"]) != rep or int(b["repetition"]) != rep:
            raise ValueError("Mismatched repetition IDs.")
        differences = {}
        passed = True
        for key, fields in (("canonical", replay.CANONICAL), ("values", replay.FIELDS)):
            left, right = a[key], b[key]
            expected = (1, 2, len(fields)) if key == "canonical" else (1, 2, 263, len(fields))
            if left.shape != expected or right.shape != expected:
                raise ValueError(f"Unexpected {key} shape for repetition {rep}.")
            for i, field in enumerate(fields):
                delta = np.abs(left[..., i] - right[..., i])
                maximum = float(delta.max()) if np.isfinite(delta).all() else None
                differences[field] = maximum
                tol = 1e-6 if field == "min_ess" else 1e-10
                if field in ("resolved", "final_draws_per_stream"):
                    tol = 0.  # The final draw count uniquely identifies the retry level.
                passed &= maximum is not None and maximum <= tol
    return {"equivalent": passed, "max_abs_difference": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "inputs", "cpu_log", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    from experiments.burn_gpu_quadrature import BurnGPU, torch, triton

    if not torch.cuda.is_available():
        parser.error("This benchmark requires an allocated CUDA GPU.")
    cache, shard = args.root / "cache", args.root / "shard-0"
    archived = json.loads((cache / "manifest.json").read_text())["protocol"]
    patches, _, _ = burn.cohort(args.inputs)
    current = burn.protocol(patches)
    if {**current, "git_commit": archived["git_commit"]} != archived:
        raise ValueError("CPU source hashes, cohort, versions or statistical protocol differ.")
    cpu_protocol = {**archived, "cache_manifest_sha256": burn.digest(cache / "manifest.json")}
    if json.loads((shard / "config.json").read_text())["protocol"] != cpu_protocol:
        raise ValueError("CPU shard does not match the archived calibration grids.")
    log = args.cpu_log.read_text()
    cpu_times = {}
    for rep in REPS:
        matches = re.findall(rf"^Saved repetition {rep} in ([0-9.]+)s$", log, re.MULTILINE)
        if len(matches) != 1 or not (shard / f"rep-{rep:04d}.npz").is_file():
            raise ValueError(f"Missing or ambiguous CPU reference for repetition {rep}.")
        cpu_times[rep] = float(matches[0])

    args.output.mkdir(parents=True, exist_ok=False)
    grids, ess = burn.load_cache(cache, archived)
    started = perf_counter()
    gpu = BurnGPU(grids)
    # Exercise indexed and summed paths, a partial final tile and the log-space fallback.
    grid, masses, _, _ = grids[replay.METHODS[0]]
    params = np.broadcast_to([0., 1., .3, 0., .2, 1.], (3, 6))
    rows = np.array([0, len(grid)-1, 0])
    t = np.arange(grid.shape[1]) % 2
    y = t.astype(float)
    y[0] = 45.
    for name in ("patch_log_kernels_indexed", "log_likelihood_indexed"):
        inputs = (params, t, y, grid, masses, rows, False)
        np.testing.assert_allclose(getattr(gpu, name)(*inputs), getattr(cpu, name)(*inputs),
                                   rtol=0, atol=1e-10)
    setup = perf_counter() - started
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "triton": triton.__version__, "cpu_commit": archived["git_commit"],
              "benchmark_commit": current["git_commit"], "cpu_manifest_sha256": cpu_protocol["cache_manifest_sha256"],
              "bank_upload_and_preflight_seconds": setup, "repetitions": {}}
    print(f"{report['gpu']} FP64 preflight passed in {setup:.1f}s", flush=True)
    # Scope the substitution to this process. Production replay code and its hashes stay identical.
    with (patch.object(cpu, "patch_log_kernels_indexed", gpu.patch_log_kernels_indexed),
          patch.object(cpu, "log_likelihood_indexed", gpu.log_likelihood_indexed)):
        for rep in REPS:
            replay_args = argparse.Namespace(start=rep, repeats=1, output=args.output / f"gpu-{rep}")
            started = perf_counter()
            replay.run_repetitions(replay_args, patches, grids, ess,
                                   {**cpu_protocol, "benchmark_gpu_commit": current["git_commit"]})
            elapsed = perf_counter() - started
            name = f"rep-{rep:04d}.npz"
            row = compare(shard / name, replay_args.output / name, rep)
            row.update(cpu_seconds=cpu_times[rep], gpu_seconds=elapsed, speedup=cpu_times[rep]/elapsed)
            report["repetitions"][rep] = row
            replay.e02.write_json(args.output / "benchmark.json", report)
            print(f"rep={rep} CPU={cpu_times[rep]:.1f}s GPU={elapsed:.1f}s "
                  f"speedup={row['speedup']:.2f}x equivalent={row['equivalent']}", flush=True)
    timings = report["repetitions"].values()
    report["single_worker_speedup"] = sum(cpu_times.values()) / sum(r["gpu_seconds"] for r in timings)
    report["equivalent"] = all(r["equivalent"] for r in timings)
    replay.e02.write_json(args.output / "benchmark.json", report)
    if not report["equivalent"]:
        raise SystemExit("GPU replay differed from CPU. See benchmark.json for the affected fields.")


if __name__ == "__main__":
    main()
