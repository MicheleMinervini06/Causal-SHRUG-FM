"""Fit SHRUG-FM-style pretraining clusters and score cached downstream CLS vectors.

Euclidean nearest-centroid assignment; normalize each distance by that cluster's
mean Euclidean radius. NCDD = sum_j d_j - k*d_assigned (paper equations 4-5).
FAISS squared-L2 outputs are NOT used as Euclidean distances.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import cache_encoder_features as cache
    from .cache_pretraining_embeddings import DEFAULT_EMBEDDINGS, representation
else:
    import cache_encoder_features as cache
    from cache_pretraining_embeddings import DEFAULT_EMBEDDINGS, representation


def checked_matrix(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite embedding matrix.")
    return np.ascontiguousarray(values)


def distances(vectors, centroids):
    # Direct differences in float64 avoid cancellation near a centroid.
    return np.linalg.norm(vectors.astype(np.float64)[:, None, :] - centroids.astype(np.float64)[None, :, :], axis=2)


def score(vectors, centroids, radii):
    vectors, centroids = checked_matrix(vectors), checked_matrix(centroids)
    radii = np.asarray(radii, dtype=np.float64)
    if (vectors.shape[1] != centroids.shape[1] or radii.shape != (len(centroids),)
            or not np.isfinite(radii).all() or (radii <= 0).any()):
        raise ValueError("Incompatible centroid dimensions or invalid cluster radii.")
    results = []
    for start in range(0, len(vectors), 256):
        raw = distances(vectors[start:start + 256], centroids)
        assigned = raw.argmin(axis=1)  # closest in raw Euclidean space, before radius normalization
        normalized = raw / radii
        nearest = normalized[np.arange(len(raw)), assigned]
        ncdd = normalized.sum(axis=1) - len(centroids) * nearest
        results.extend(zip(assigned.tolist(), raw[np.arange(len(raw)), assigned].tolist(), nearest.tolist(), ncdd.tolist()))
    return results


def fit(vectors, *, clusters=64, seed=0, iterations=100, restarts=5, threads=4):
    import faiss
    vectors = checked_matrix(vectors)
    if clusters < 2 or len(vectors) < 39 * clusters or min(iterations, restarts, threads) < 1 or seed < 0:
        raise ValueError("Need >=39 reference images per cluster and positive fitting settings.")
    if len(np.unique(vectors, axis=0)) < clusters:
        raise ValueError("Not enough distinct reference embeddings.")
    faiss.omp_set_num_threads(threads)
    model = faiss.Kmeans(vectors.shape[1], clusters, niter=iterations, nredo=restarts,
                         seed=seed, gpu=False, spherical=False,
                         max_points_per_centroid=len(vectors), verbose=False)
    print(f"Fitting FAISS k-means: {len(vectors)} images, k={clusters}, seed={seed}", flush=True)
    model.train(vectors)
    centroids = model.centroids
    counts, totals = np.zeros(clusters, dtype=np.int64), np.zeros(clusters, dtype=np.float64)
    for start in range(0, len(vectors), 256):
        raw = distances(vectors[start:start + 256], centroids)
        assigned = raw.argmin(axis=1)
        counts += np.bincount(assigned, minlength=clusters)
        totals += np.bincount(assigned, weights=raw[np.arange(len(raw)), assigned], minlength=clusters)
    radii = totals / np.maximum(counts, 1)
    if (counts == 0).any() or (radii <= 0).any() or not np.isfinite(radii).all():
        raise ValueError("Empty or zero-radius cluster: reference cannot support these scores.")
    return centroids, radii, counts


def load_embeddings(path):
    with np.load(path, allow_pickle=False) as archive:
        matrix = checked_matrix(archive["cls_embeddings"])
        ids = archive["patch_ids"].astype(str)
        if matrix.shape != (len(ids), 384) or len(set(ids)) != len(ids):
            raise ValueError("Expected unique IDs with 384-D CLS embeddings.")
        metadata = json.loads(archive["metadata"].item())
        if "payload_sha256" in metadata and metadata["payload_sha256"] != cache.array_hash(matrix):
            raise ValueError("Embedding payload checksum mismatch.")
        tables = {k: archive[k].astype(str).copy() for k in ("event_ids", "original_splits") if k in archive}
        if any(len(v) != len(ids) for v in tables.values()):
            raise ValueError("Embedding identifier lengths differ.")
        return matrix, ids, metadata, tables


def run(reference, downstream, output, *, clusters=64, seed=0, threads=4, iterations=100, restarts=5):
    import faiss
    reference, downstream, output = Path(reference), Path(downstream), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory.")
    training, _, reference_metadata, _ = load_embeddings(reference)
    if not reference_metadata.get("reference_complete"):
        raise ValueError("Debug/incomplete reference cannot be used to fit scientific scores.")
    vectors, ids, metadata, tables = load_embeddings(downstream)
    if set(tables) != {"event_ids", "original_splits"}:
        raise ValueError("Downstream archive must preserve event IDs and original splits.")
    if representation(reference_metadata["contract"]) != representation(metadata["contract"]):
        raise ValueError("Encoder/checkpoint/preprocessing mismatch between reference and downstream embeddings.")
    centroids, radii, counts = fit(training, clusters=clusters, seed=seed, threads=threads,
                                 iterations=iterations, restarts=restarts)
    scored = score(vectors, centroids, radii)
    output.mkdir(parents=True, exist_ok=True)
    config = {"reference_sha256": cache.digest(reference), "downstream_sha256": cache.digest(downstream),
              "reference_metadata": reference_metadata, "downstream_metadata": metadata,
              "clusters": clusters, "seed": seed, "iterations": iterations, "restarts": restarts,
              "threads": threads, "faiss_version": faiss.__version__, "numpy_version": np.__version__,
              "fit_sampling": "all supplied reference vectors; FAISS internal subsampling disabled",
              "weighting": "equal per available seasonal image; location selection recorded in reference metadata",
              "distance": "Euclidean L2; no extra embedding normalization; nearest raw centroid",
              "radius": "mean Euclidean distance of assigned reference images, recomputed after fitting",
              "NCDD": "sum_j(normalized_distance_j) - k*normalized_distance_assigned; no clipping",
              "direction": {"normalized_distance": "higher means more unfamiliar", "NCDD": "lower means less separation"},
              "scope": "same score equations as SHRUG-FM; our reference sample and unreleased fitting settings differ",
              "implementation_sha256": cache.digest(Path(__file__))}
    cache.save_npz(output / "clusters.npz", centroids=centroids, mean_radii=radii, counts=counts,
                   metadata=np.asarray(json.dumps(config, sort_keys=True)))
    rows = [{"patch_id": key, "event_id": event, "original_split": split, "nearest_cluster": result[0],
             "centroid_distance": result[1], "normalized_distance": result[2], "NCDD": result[3]}
            for key, event, split, result in zip(ids, tables["event_ids"], tables["original_splits"], scored)]
    cache.write_index(output / "embedding_uncertainty.csv", rows)
    cache.write_json(output / "embedding_config.json", config)
    cache.write_json(output / "embedding_summary.json", {"n_reference_images": len(training), "n_patches": len(rows),
                     "split_counts": dict(Counter(tables["original_splits"])), "cluster_counts": counts.tolist(),
                     "mean_radii": radii.tolist(), "negative_NCDD_count": sum(r[3] < 0 for r in scored),
                     "clusters_sha256": cache.digest(output / "clusters.npz"),
                     "scores_sha256": cache.digest(output / "embedding_uncertainty.csv")})
    print(f"Embedding uncertainty complete: {len(rows)} patches; {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_EMBEDDINGS / "pretraining_cls.npz")
    parser.add_argument("--downstream", type=Path, default=cache.ROOT / "data/sen1floods11/encoder_features/ssl4eo-s12-moco/cls_embeddings.npz")
    parser.add_argument("--output", type=Path, default=cache.ROOT / "results/embedding_uncertainty/k64-seed0")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=5)
    args = parser.parse_args()
    run(args.reference, args.downstream, args.output, clusters=args.clusters, seed=args.seed,
        threads=args.threads, iterations=args.iterations, restarts=args.restarts)


if __name__ == "__main__":
    main()
