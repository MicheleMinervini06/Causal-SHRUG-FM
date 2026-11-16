"""Compute SHRUG-FM's spatial U_I feature: pretraining-location KDE per patch.

Uses the official SSL4EO-S12 location CSV and a normalized geodesic Gaussian
kernel on a sphere. Kernel/bandwidth choices are ours: SHRUG-FM does not release
these settings. No downstream labels/errors are used to fit the density.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import transform

if __package__:
    from . import cache_encoder_features as cache
else:
    import cache_encoder_features as cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COORDINATES = ROOT / "data/ssl4eo-s12/pretraining_coordinates.csv"
SOURCE_URL = "https://drive.google.com/file/d/1RyJnGznSbMparS88BhHkXxETf0K-qYqI/view"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=1RyJnGznSbMparS88BhHkXxETf0K-qYqI&export=download&confirm=t"
SOURCE_SHA256 = "e9e5e5d3d7ba977efbc19785cb5362aee412d9c66a3864ef0fae524c5d1a94f5"
SOURCE_BYTES = 8018714
EARTH_RADIUS_KM = 6371.0
SPLITS = ("train", "validation", "test", "bolivia_holdout")
FIELDS = ("patch_id", "event_id", "original_split", "longitude", "latitude",
          "pretraining_density", "pretraining_log_density")


def read_pretraining(path, download=False):
    """Verify the released headerless id,longitude,latitude file before use."""
    path = Path(path)
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            print("Downloading official SSL4EO-S12 coordinates (~8 MB).", flush=True)
            with urllib.request.urlopen(DOWNLOAD_URL, timeout=30) as response, partial.open("wb") as stream:
                received = 0
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > SOURCE_BYTES:
                        raise ValueError("Coordinate download exceeds the pinned file size.")
                    stream.write(block)
            if cache.digest(partial) != SOURCE_SHA256:
                raise ValueError("Coordinate download SHA-256 differs from the verified release.")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if not path.is_file():
        raise ValueError("Pretraining CSV is missing. Run this script with --download-only first.")
    if cache.digest(path) != SOURCE_SHA256:
        raise ValueError("Pretraining CSV SHA-256 differs from the verified official file; restore it.")
    coordinates, identifiers = [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row:  # The released CSV has blank lines and no header.
                continue
            if len(row) != 3 or not row[0].isdigit():
                raise ValueError(f"Expected headerless id,longitude,latitude at CSV line {line}.")
            identifier = int(row[0])
            if identifier in identifiers:
                raise ValueError(f"Duplicate pretraining location ID: {identifier}")
            identifiers.add(identifier)
            coordinates.append([float(row[1]), float(row[2])])
    coordinates = checked_coordinates(coordinates)
    return coordinates, {"path": str(path.resolve()), "source_url": SOURCE_URL,
                         "source_repository": "https://github.com/zhu-xlab/SSL4EO-S12",
                         "sha256": SOURCE_SHA256, "bytes": path.stat().st_size,
                         "locations": len(coordinates),
                         "unique_coordinate_pairs": len(np.unique(coordinates, axis=0)),
                         "weighting": "equal weight per released location ID; seasonal images not repeated",
                         "scope": "released SSL4EO-S12 location catalog; exact checkpoint sampling trace unavailable",
                         "format": "headerless id,longitude,latitude; WGS84 degrees"}


def checked_coordinates(values):
    coordinates = np.asarray(values, dtype=np.float64)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates)
            or not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:, 0]) > 180).any() or (np.abs(coordinates[:, 1]) > 90).any()):
        raise ValueError("Expected nonempty finite [longitude,latitude] coordinates in WGS84 degrees.")
    return coordinates


def gaussian_log_normalizer(bandwidth_km):
    """Log of the kernel's spherical surface integral (km^2).

    Z = 2*pi*R^2 * integral_0^pi exp(-theta^2/(2*b^2))*sin(theta) dtheta,
    b=h/R. Substitute u=theta/b to resolve small bandwidths. A 64-point
    Gauss-Legendre integral stops at min(pi/b,12); the omitted Gaussian tail
    is below exp(-72). Unlike planar normalization, this integrates on a sphere.
    """
    if not math.isfinite(bandwidth_km) or bandwidth_km <= 0:
        raise ValueError("Bandwidth must be finite and positive, in km.")
    b = bandwidth_km / EARTH_RADIUS_KM
    upper = min(math.pi / b, 12.)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = (nodes + 1) * upper / 2
    integral = np.dot(weights, np.exp(-.5 * u ** 2) * u * np.sinc(b * u / math.pi)) * upper / 2
    return math.log(2 * math.pi) + 2 * math.log(bandwidth_km) + math.log(integral)


def kde_log_density(pretraining, queries, bandwidth_km):
    """Equal-weight Gaussian KDE, great-circle distances, log probability/km^2.

    Process one query at a time for bounded memory; all pretraining locations
    contribute, without subsampling or truncating the fitted kernel.
    """
    locations = np.deg2rad(checked_coordinates(pretraining))
    queries = np.deg2rad(checked_coordinates(queries))
    log_normalizer = gaussian_log_normalizer(bandwidth_km)
    cos_latitude = np.cos(locations[:, 1])
    result = np.empty(len(queries), dtype=np.float64)
    for index, (longitude, latitude) in enumerate(queries):
        haversine = (np.sin((locations[:, 1] - latitude) / 2) ** 2
                     + cos_latitude * np.cos(latitude) * np.sin((locations[:, 0] - longitude) / 2) ** 2)
        angle = 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        log_kernel = -.5 * (EARTH_RADIUS_KM * angle / bandwidth_km) ** 2
        maximum = log_kernel.max()
        result[index] = maximum + np.log(np.exp(log_kernel - maximum).mean()) - log_normalizer
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite log density; check coordinate values and bandwidth scale.")
    return result


def patch_center(row, audit, root):
    """Center of the full affine image footprint, then reproject to WGS84."""
    with rasterio.open(cache.relative_path(root, row["image_path"])) as image:
        if (image.crs is None or str(image.crs) != audit["image_crs"]
                or image.width != int(audit["image_width"]) or image.height != int(audit["image_height"])
                or not np.array_equal(list(image.transform)[:6], json.loads(audit["image_transform"]))):
            raise ValueError(f"Image geometry changed since inspection: {row['patch_id']}; rerun inspection.")
        x, y = image.transform @ (image.width / 2, image.height / 2)
        longitude, latitude = transform(image.crs, "EPSG:4326", [x], [y])
    return checked_coordinates([[longitude[0], latitude[0]]])[0]


def run(args):
    pretraining, source = read_pretraining(args.pretraining_coordinates, download=args.download_only)
    print(f"Verified {source['locations']:,} pretraining locations; SHA-256 {source['sha256']}", flush=True)
    if args.download_only:
        return source
    if args.output is None:
        raise ValueError("Provide --output for density calculation, or use --download-only.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive.")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty; choose a new directory: {output}")
    usable, skipped, audit_summary = cache.selected_rows(args.manifest, args.inspection)
    if not args.splits or len(set(args.splits)) != len(args.splits) or not set(args.splits).issubset(SPLITS):
        raise ValueError("Choose distinct known split names.")
    records = [pair for split in args.splits for pair in usable if pair[0]["original_split"] == split]
    if any(not any(row["original_split"] == split for row, _ in records) for split in args.splits):
        raise ValueError("A requested split has no usable audited patches.")
    requested = len(records)
    records = records[:args.limit] if args.limit is not None else records
    centers = np.array([patch_center(row, audit, args.manifest.resolve().parent) for row, audit in records])
    print(f"Evaluating {len(centers)} patch centers; Gaussian bandwidth {args.bandwidth_km:g} km; CPU only.", flush=True)
    log_density = kde_log_density(pretraining, centers, args.bandwidth_km)
    density = np.exp(log_density)
    rows = [{**{key: row[key] for key in ("patch_id", "event_id", "original_split")},
             "longitude": float(center[0]), "latitude": float(center[1]),
             "pretraining_density": float(value), "pretraining_log_density": float(log_value)}
            for (row, _), center, value, log_value in zip(records, centers, density, log_density)]
    config = {
        "schema_version": 1, "pretraining": source,
        "kde": {"kernel": "geodesic_gaussian", "distance": "great_circle_haversine",
                "bandwidth_km": args.bandwidth_km, "earth_radius_km": EARTH_RADIUS_KM,
                "normalization": "spherical surface integral; 64-point Gauss-Legendre; tail cutoff 12 bandwidths",
                "density_units": "km^-2; normalized probability density, not a probability or local count",
                "log_density": "natural logarithm of pretraining_density; retained for numerical underflow",
                "bandwidth_selection": "fixed physical scale; not fitted to downstream labels or errors",
                "interpretation": "lower density means sparser pretraining geographic coverage"},
        "patch_location": "affine @ (width/2,height/2), then image CRS to EPSG:4326; full image resized without cropping",
        "geometry_validation": "current image CRS, dimensions and affine must equal the audit; pixel values are not read",
        "splits": args.splits, "limit": args.limit,
        "data_provenance": {"manifest_sha256": cache.digest(args.manifest),
                            "inspection_sha256": cache.digest(args.inspection),
                            "inspection_summary_sha256": cache.digest(args.inspection.parent / "inspection_summary.json"),
                            "reference_support": audit_summary["reference_support"]},
        "intended_encoder_sha256": cache.SHA256,
        "versions": {"numpy": np.__version__, "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__},
        "implementation_sha256": {path.name: cache.digest(path) for path in (Path(__file__), Path(cache.__file__))},
        "reference": "https://arxiv.org/html/2511.10370v2",
        "differences_from_shrug_fm": "Same pretraining-location KDE feature; kernel, bandwidth, spherical metric/normalization are explicit choices because original settings are unverified. Sen1Floods11 full-patch centers replace the WorldFloods exporter. Environmental percentiles are not computed.",
    }
    output.mkdir(parents=True, exist_ok=True)
    cache.write_json(output / "density_config.json", config)
    with (output / "spatial_density.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    events = {}
    for event in sorted({row["event_id"] for row in rows}):
        values = np.array([row["pretraining_density"] for row in rows if row["event_id"] == event])
        events[event] = {"patches": len(values), "min": float(values.min()),
                         "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}
    summary = {"complete": True, "patches": len(rows), "requested_patches": requested,
               "limited": len(rows) < requested, "counts_by_split": dict(Counter(row["original_split"] for row in rows)),
               "unique_patch_centers": len(np.unique(centers, axis=0)),
               "zero_density_underflows": int((density == 0).sum()), "by_event": events,
               "skipped_unusable_patches": skipped, "features": "spatial_density.csv",
               "features_sha256": cache.digest(output / "spatial_density.csv")}
    cache.write_json(output / "density_summary.json", summary)
    print(f"Saved {len(rows)} spatial-density features in {output}", flush=True)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pretraining-coordinates", type=Path, default=DEFAULT_COORDINATES)
    result.add_argument("--download-only", action="store_true", help="Download/verify the pinned pretraining CSV, then exit without reading downstream data.")
    result.add_argument("--manifest", type=Path, default=ROOT / "data/sen1floods11/manifest.csv")
    result.add_argument("--inspection", type=Path, default=ROOT / "results/segmentation_data/patches.csv")
    result.add_argument("--output", type=Path, help="Required for calculation; new/empty output directory.")
    result.add_argument("--bandwidth-km", type=float, default=100., help="Fixed Gaussian distance scale; initial choice 100 km.")
    result.add_argument("--splits", choices=SPLITS, nargs="+", default=list(SPLITS))
    result.add_argument("--limit", type=int, help="Limit total selected patches for an implementation check.")
    return result


def main():
    try:
        run(parser().parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Spatial density failed: {error}") from error


if __name__ == "__main__":
    main()
