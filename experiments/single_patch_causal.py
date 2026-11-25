"""E06. Verify one reference fraction at a time on the complete E05 cohort."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic

import numpy as np
from scipy.special import logsumexp

from experiments import fraction_calibration_causal as e05
from experiments import single_patch_verification as verification
from experiments import semisynthetic_confounder as e02


METHODS = e05.calibration.MODELS
STRENGTHS = e02.ASSIGNMENT_STRENGTHS
FIELDS = ("before", "before_lo", "before_hi", "after", "after_lo", "after_hi", "resolved",
          "min_ess", "max_weight", "mean_shift_seed_gap", "width_shift_seed_gap", "final_draws_per_stream")
CANONICAL = ("estimate", "lower95", "upper95")
NODES = 16
E05_SUMMARY = "cec46e7b859e5310cbc1405938915716a52879987162812a175af37e127559e6"


def protocol(patches):
    config = e05.protocol(NODES)
    config.pop("controls")
    return verification_protocol(config, patches, "E06 single-patch verification")


def verification_protocol(config, patches, experiment, *, strengths=STRENGTHS):
    config.update(experiment=experiment, methods=METHODS, strengths=strengths,
                  contrast=f"reveal one manual fraction minus no reveal, keeping the other {len(patches)-1} latent",
                  integration_seed=105, independent_streams=2, sizes=verification.BATCH_SIZES,
                  mean_shift_tolerance=verification.MEAN_TOL, width_shift_tolerance=verification.WIDTH_TOL,
                  fields=FIELDS, canonical_fields=CANONICAL,
                  patches=[{k: r[k] for k in ("patch_id", "event_id", "original_split",
                                    "reference_fraction", "predicted_fraction")} for r in patches])
    for path in (Path(__file__), Path(verification.__file__)):
        config["implementation"][path.name] = e02.digest(path)
    return json.loads(json.dumps(config))


def run(args):
    patches, proxy, scores = e05.cohort(args.test_scores)
    posterior = e05.verified(args.posterior, e05.POSTERIOR)
    reference = np.array([r["reference_fraction"] for r in patches])
    grids, bank_ess = {}, []
    with np.load(posterior, allow_pickle=False) as frozen:
        for method in METHODS:
            print(f"Preparing {method} on {len(proxy)} patches", flush=True)
            grids[method], ess = prepare_method(frozen, proxy, scores, reference, method)
            bank_ess.append(ess)
    run_repetitions(args, patches, grids, bank_ess, protocol(patches))


def prepare_method(frozen, proxy, scores, reference, method):
    grid, mass = e05.inference.latent_grid(frozen, proxy, scores, method, NODES)
    evidence = verification.reference_evidence(frozen, proxy, scores, reference, method)
    ess = np.exp(2*logsumexp(evidence, axis=0) - logsumexp(2*evidence, axis=0)).tolist()
    mean_c = np.einsum("ijk,ijk->j", grid, mass)/len(grid)
    return (grid, mass, mean_c, evidence), ess


def run_repetitions(args, patches, grids, bank_ess, run_protocol):
    strengths = run_protocol["strengths"]
    simulated_c, _, _ = e02.standardize(patches)
    proxy = np.array([r["predicted_fraction"] for r in patches])
    revealed_c = (np.array([r["reference_fraction"] for r in patches]) - proxy.mean()) / proxy.std()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"start": args.start, "repeats": args.repeats, "protocol": run_protocol}
    manifest = args.output / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Existing shard uses a different configuration or code version.")
    elif any(args.output.iterdir()):
        raise ValueError("Unrecognized nonempty shard directory.")
    else:
        e02.write_json(manifest, config)
    checksums = {}
    for rep in range(args.start, args.start + args.repeats):
        path = args.output / f"rep-{rep:04d}.npz"
        if not path.exists():
            started = monotonic()
            uniforms, noise = e02.simulation_draws(patches, 42, rep)
            values = np.empty((len(strengths), len(METHODS), len(patches), len(FIELDS)))
            canonical = np.empty((*values.shape[:2], len(CANONICAL)))
            for s, strength in enumerate(strengths):
                t, y, _ = e02.simulate(simulated_c, uniforms, noise, 0., strength)
                for m, method in enumerate(METHODS):
                    base, updates = verification.verify(np.asarray(t), np.asarray(y), *grids[method],
                                                        revealed_c, strength == 0, rep)
                    canonical[s, m] = [base[k] for k in CANONICAL]
                    values[s, m] = [[updates[i][k] for k in FIELDS] for i in range(len(patches))]
                    print(f"rep={rep} a1={strength:g} {method}: "
                          f"{sum(r['resolved'] for r in updates.values())}/{len(patches)} resolved", flush=True)
            with NamedTemporaryFile(dir=args.output, suffix=".tmp", delete=False) as tmp:
                np.savez_compressed(tmp, values=values, canonical=canonical, repetition=rep)
            Path(tmp.name).replace(path)
            print(f"Saved repetition {rep} in {monotonic()-started:.1f}s", flush=True)
        checksums[path.name] = e02.digest(path)
    e02.write_json(args.output / "run_summary.json", {**config, "sha256": checksums,
                                                      "calibration_only_ess": np.asarray(bank_ess).tolist()})


def metrics(rows):
    before, blo, bhi, after, alo, ahi = rows.T
    for name, x, y in (
        ("bias", before-1, after-1), ("rmse", (before-1)**2, (after-1)**2),
        ("width95", bhi-blo, ahi-alo),
        ("coverage95", ((blo<=1)&(1<=bhi)).astype(float), ((alo<=1)&(1<=ahi)).astype(float))):
        if not len(x):
            yield name, None, None, None, None
            continue
        b, a = x.mean(), y.mean()
        influence = y-x
        if name == "rmse":
            b, a = np.sqrt(b), np.sqrt(a)
            influence = y/(2*a)-x/(2*b) if min(a,b)>0 else None
        yield name, b, a, a-b, influence


def mcse(influence):
    if influence is not None and len(influence)>1:
        return influence.std(ddof=1)/np.sqrt(len(influence))
    return None


def collect(directory, output, start, repeats, e05_summary, *, summary_hash=E05_SUMMARY, expected_protocol=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    seen, meta = set(), None
    for manifest in sorted(directory.glob("shard-*/run_summary.json")):
        shard = json.loads(manifest.read_text())
        if expected_protocol is not None and shard["protocol"] != expected_protocol:
            raise ValueError("Shard differs from the expected verification protocol.")
        if meta is None:
            meta = shard
            patches = meta["protocol"]["patches"]
            strengths = meta["protocol"]["strengths"]
            # rep, assignment, model, patch, before/after mean and interval
            values = np.empty((repeats, len(strengths), len(METHODS), len(patches), 6))
            resolved = np.empty(values.shape[:-1], dtype=bool)
            canonical = np.empty((*values.shape[:3], 3))
        if (shard["protocol"] != meta["protocol"] or shard["calibration_only_ess"] != meta["calibration_only_ess"]
                or shard["protocol"]["fields"] != list(FIELDS)):
            raise ValueError("Incompatible shard protocols.")
        for rep in range(shard["start"], shard["start"] + shard["repeats"]):
            if rep in seen or not start <= rep < start + repeats:
                raise ValueError(f"Unexpected or duplicate repetition {rep}.")
            seen.add(rep)
            path = manifest.with_name(f"rep-{rep:04d}.npz")
            with np.load(e05.verified(path, shard["sha256"][path.name]), allow_pickle=False) as saved:
                result = saved["values"]
                if (int(saved["repetition"]) != rep or result.shape != (*values.shape[1:-1], len(FIELDS))
                        or saved["canonical"].shape != canonical.shape[1:] or not np.isfinite(saved["canonical"]).all()
                        or not np.isfinite(result).all() or not np.isin(result[..., 6], [0, 1]).all()):
                    raise ValueError(f"Invalid repetition file: {path}")
                values[rep-start], resolved[rep-start], canonical[rep-start] = (
                    result[..., :6], result[..., 6], saved["canonical"])
    if len(seen) != repeats:
        raise ValueError(f"Found {len(seen)} of {repeats} requested repetitions.")
    rows, baselines, contrasts = [], [], []
    for s, strength in enumerate(strengths):
        for m, method in enumerate(METHODS):
            base = canonical[:, s, m]
            baselines.append({"a1": strength, "method": method, "n": repeats,
                              "bias": np.mean(base[:, 0]-1), "rmse": np.sqrt(np.mean((base[:, 0]-1)**2)),
                              "coverage95": np.mean((base[:, 1]<=1) & (1<=base[:, 2])),
                              "width95": np.mean(base[:, 2]-base[:, 1])})
            for p, patch in enumerate(patches):
                use = resolved[:, s, m, p]
                row = {**patch, "a1": strength, "method": method, "n_requested": repeats,
                       "n_resolved": int(use.sum()), "complete": bool(use.all()),
                       "calibration_only_ess": meta["calibration_only_ess"][m][p]}
                used = values[use, s, m, p]
                for metric, b, a, delta, influence in metrics(used):
                    row.update({f"{metric}_before": b, f"{metric}_after": a,
                                f"{metric}_change": delta, f"{metric}_change_mcse": mcse(influence)})
                row["mean_absolute_effect_shift"] = np.abs(used[:, 3]-used[:, 0]).mean() if len(used) else None
                rows.append(row)
        for p, patch in enumerate(patches):
            joint = resolved[:, s, :, p].all(axis=1)
            pair = {**patch, "a1": strength, "n_pairs": int(joint.sum()), "complete": bool(joint.all())}
            primary = metrics(values[joint, s, METHODS.index("proxy_only"), p])
            scored = metrics(values[joint, s, METHODS.index("with_scores"), p])
            for (name, _, _, d0, i0), (_, _, _, d1, i1) in zip(primary, scored, strict=True):
                pair[name+"_change_difference"] = d1-d0 if d0 is not None else None
                pair[name+"_change_difference_mcse"] = mcse(i1-i0 if i0 is not None and i1 is not None else None)
            contrasts.append(pair)
    baseline_gap = None
    if repeats == e05.TOTAL:
        old = {(float(r["a1"]), r["method"]): r
               for r in e02.read_rows(e05.verified(e05_summary, summary_hash))}
        baseline_gap = max(abs(r[k]-float(old[r["a1"], r["method"]][k])) for r in baselines
                           for k in ("bias", "rmse", "width95", "coverage95"))
        if baseline_gap > 1e-8:
            raise ValueError(f"No-reveal replay differs from the reviewed causal summary by {baseline_gap:g}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"combined"
        stage.mkdir()
        e02.write_csv(stage/"patch_impacts.csv", rows)
        e02.write_csv(stage/"paired_calibrations.csv", contrasts)
        e02.write_csv(stage/"no_reveal.csv", baselines)
        e02.write_json(stage/"run_summary.json", {"protocol": meta["protocol"], "start": start, "repeats": repeats,
                       "max_e05_replay_difference": baseline_gap, "collector_sha256": e02.digest(Path(__file__)),
                       "n_unresolved": int((~resolved).sum()), "n_incomplete_patch_groups": sum(not r["complete"] for r in rows),
                       "summary_population": "Resolved pairs only; incomplete groups are explicitly flagged."})
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=Path("results/e05-posterior-390bc867.npz"))
    parser.add_argument("--test-scores", type=Path, default=Path("results/reliability_gate/test_seed0-pred-job-44220183/gate_scores.csv"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--repeats", default=e05.TOTAL, type=int)
    parser.add_argument("--collect", type=Path)
    parser.add_argument("--e05-summary", type=Path, default=Path("results/causal_e05/phase2-array-44387254/combined/summary.csv"))
    args = parser.parse_args()
    if args.start < 0 or not 1 <= args.repeats <= e05.TOTAL - args.start:
        parser.error("Expected repetitions in 0..4999.")
    if args.collect:
        collect(args.collect, args.output, args.start, args.repeats, args.e05_summary)
    else:
        run(args)
