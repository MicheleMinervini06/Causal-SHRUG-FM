"""Audit Sen1Floods11 pairs and reference water fractions on their native grid."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio

if __package__:
    from .inspect_raster_pair import CLASS_COLORS, CLASS_NAMES, stretch_rgb
else:
    from inspect_raster_pair import CLASS_COLORS, CLASS_NAMES, stretch_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = ("patch_id", "event_id", "original_split", "image_path", "label_path")
SPLIT_FILES = {
    "flood_train_data.csv": "train", "flood_valid_data.csv": "validation",
    "flood_test_data.csv": "test", "flood_bolivia_data.csv": "bolivia_holdout",
}
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
METRIC_FIELDS = (
    "image_sha256", "label_sha256", "image_files_sha256", "label_files_sha256",
    "image_exists", "label_exists", "image_width", "image_height", "label_width", "label_height",
    "image_bands", "label_bands", "image_dtypes", "label_dtypes", "band_descriptions",
    "image_crs", "label_crs", "image_transform", "label_transform", "image_bounds", "label_bounds",
    "image_nodata", "label_nodata", "max_grid_displacement_pixels", "label_value_counts",
    "total_pixels", "image_valid_pixels", "label_valid_pixels", "joint_valid_pixels",
    "label_valid_without_image_pixels", "image_nonfinite_pixels", "label_nonfinite_pixels",
    "ignored_label_pixels", "label_water_pixels", "joint_water_pixels",
    "label_water_fraction", "joint_water_fraction", "reference_valid_pixels",
    "reference_water_pixels", "reference_water_fraction", "valid_fraction",
)


def dataset_file_hashes(dataset) -> dict[str, str]:
    """Fingerprint the raster and GDAL sidecars that may change its validity."""
    parent = Path(dataset.name).resolve().parent
    fingerprints = {}
    for filename in dataset.files:
        path = Path(filename).resolve()
        name = path.relative_to(parent).as_posix()
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        fingerprints[name] = checksum.hexdigest()
    return fingerprints


def local_path(root: Path, value: str) -> Path:
    """Manifest paths must resolve inside their dataset directory."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Manifest paths must be relative to the manifest directory.")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes the dataset directory.")
    return resolved


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(MANIFEST_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest requires columns: {', '.join(MANIFEST_FIELDS)}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Manifest contains duplicate column names.")
        records = []
        for line, source in enumerate(reader, 2):
            record = {field: (source.get(field) or "").strip() for field in MANIFEST_FIELDS}
            record.update(manifest_line=line, errors=[], warnings=[])
            if None in source:
                record["errors"].append("malformed_manifest_row")
            for field in MANIFEST_FIELDS:
                if not record[field]:
                    record["errors"].append(f"missing_{field}")
            for field in ("image_path", "label_path"):
                try:
                    if record[field]:
                        record[f"_{field}"] = local_path(path.parent, record[field])
                except ValueError:
                    record["errors"].append(f"unsafe_{field}")
            if record["original_split"] not in SPLIT_FILES.values():
                record["errors"].append("unknown_original_split")
            if record["patch_id"] and record["event_id"] != record["patch_id"].rsplit("_", 1)[0]:
                record["errors"].append("event_id_mismatch")
            for field, suffix in (("image_path", "_S2Hand.tif"), ("label_path", "_LabelHand.tif")):
                if record[field] and Path(record[field]).name != record["patch_id"] + suffix:
                    record["errors"].append(f"{field}_patch_id_mismatch")
            records.append(record)
    if not records:
        raise ValueError("Manifest is empty.")
    for field in ("patch_id", "_image_path", "_label_path"):
        counts = Counter(row[field] for row in records if row.get(field))
        for row in records:
            if row.get(field) and counts[row[field]] > 1:
                row["errors"].append(f"duplicate_{field.lstrip('_')}")
    return records


def official_splits(root: Path) -> dict[str, str] | None:
    """Cross-check manifests against the downloaded original split CSVs."""
    if not all((root / "splits" / name).is_file() for name in SPLIT_FILES):
        return None
    assignments = {}
    for filename, split in SPLIT_FILES.items():
        with (root / "splits" / filename).open(newline="", encoding="utf-8-sig") as stream:
            for cells in csv.reader(stream):
                if not cells or not any(cell.strip() for cell in cells):
                    continue
                if len(cells) != 2:
                    raise ValueError(f"Invalid official split row in {filename}: {cells}")
                image, label = (Path(cell.strip()).name for cell in cells)
                key = image.removesuffix("_S1Hand.tif")
                if image != key + "_S1Hand.tif" or label != key + "_LabelHand.tif":
                    raise ValueError(f"Mismatched official split pair in {filename}: {cells}")
                if key in assignments:
                    raise ValueError(f"Duplicate/overlapping official split assignment: {key}")
                assignments[key] = split
    return assignments


def grid_displacement(image, label) -> float:
    """Compare affine grids in pixels, tolerating harmless coordinate roundoff."""
    positions = ((0, 0), (image.width, 0), (0, image.height),
                 (image.width, image.height), (image.width / 2, image.height / 2))
    inverse = ~image.transform
    return max(
        max(abs(actual - expected) for actual, expected in zip(inverse @ (label.transform @ point), point))
        for point in positions
    )


def valid_support(image_values, image_masks, labels, label_mask):
    """Known labels plus GDAL validity; joint support requires every image band."""
    image_valid = (image_masks > 0).all(axis=0) & np.isfinite(image_values).all(axis=0)
    label_valid = (label_mask > 0) & np.isfinite(labels) & np.isin(labels, (0, 1))
    return image_valid, label_valid, image_valid & label_valid


def water_fraction(labels, valid):
    count = int(np.count_nonzero(valid))
    water = int(np.count_nonzero(valid & (labels == 1)))
    return count, water, water / count if count else None


def inspect_pair(source: dict, *, support="joint", grid_tolerance=1e-4,
                 expected_size=512, assignments=None) -> dict:
    record = {key: value for key, value in source.items() if not key.startswith("_")}
    record.update({field: None for field in METRIC_FIELDS})
    record.update(errors=list(source["errors"]), warnings=list(source["warnings"]),
                  reference_support=support, usable=False)
    errors, warnings = record["errors"], record["warnings"]
    if assignments is None:
        record["official_split_check"] = "unavailable"
    elif assignments.get(record["patch_id"]) != record["original_split"]:
        record["official_split_check"] = "mismatch"
        errors.append("official_split_mismatch")
    else:
        record["official_split_check"] = "verified"
    image_path, label_path = source.get("_image_path"), source.get("_label_path")
    record["image_exists"] = bool(image_path and image_path.is_file())
    record["label_exists"] = bool(label_path and label_path.is_file())
    for field in ("image", "label"):
        if not record[f"{field}_exists"]:
            errors.append(f"missing_{field}_file")
    if not record["image_exists"] or not record["label_exists"]:
        return record
    try:
        with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
            for name, dataset in (("image", image), ("label", label)):
                fingerprints = dataset_file_hashes(dataset)
                record.update({
                    f"{name}_files_sha256": fingerprints,
                    f"{name}_sha256": fingerprints[Path(dataset.name).name],
                    f"{name}_width": dataset.width, f"{name}_height": dataset.height,
                    f"{name}_bands": dataset.count, f"{name}_dtypes": list(dataset.dtypes),
                    f"{name}_crs": str(dataset.crs) if dataset.crs else None,
                    f"{name}_transform": list(dataset.transform)[:6],
                    f"{name}_bounds": list(dataset.bounds),
                    f"{name}_nodata": str(dataset.nodata) if dataset.nodata is not None else None,
                })
                if dataset.crs is None:
                    errors.append(f"missing_{name}_crs")
                if not np.isfinite(tuple(dataset.transform)).all() or dataset.transform.determinant == 0:
                    errors.append(f"invalid_{name}_transform")
                if dataset.crs and dataset.crs.is_geographic:
                    left, bottom, right, top = dataset.bounds
                    if not (-180 <= left < right <= 180 and -90 <= bottom < top <= 90):
                        errors.append(f"invalid_{name}_geographic_bounds")
            geometry_errors = len(errors)
            if image.shape != label.shape:
                errors.append("dimension_mismatch")
            if expected_size and image.shape != (expected_size, expected_size):
                errors.append("unexpected_patch_dimensions")
            if image.crs != label.crs:
                errors.append("crs_mismatch")
            if image.count != 13:
                errors.append("unexpected_image_band_count")
            if label.count != 1:
                errors.append("unexpected_label_band_count")
            record["band_descriptions"] = list(image.descriptions)
            if any(image.descriptions) and tuple(image.descriptions) != BANDS:
                errors.append("unexpected_band_order")
            elif not any(image.descriptions):
                warnings.append("band_order_assumed_descriptions_missing")
            if image.crs and image.crs == label.crs and image.transform.determinant != 0:
                displacement = grid_displacement(image, label)
                record["max_grid_displacement_pixels"] = displacement
                if not np.isfinite(displacement) or displacement > grid_tolerance:
                    errors.append("grid_misalignment")
            # Identity/manifest errors do not prevent reading diagnostics; raster
            # structure errors do, because pixelwise operations would be unsafe.
            unsafe_structure = len(errors) > geometry_errors or any(
                error.startswith(("missing_image_crs", "missing_label_crs", "invalid_image", "invalid_label"))
                for error in errors
            )
            if label.nodata in (0, 1):
                errors.append("label_nodata_conflicts_with_valid_class")
            if unsafe_structure:
                return record
            values, labels = image.read(), label.read(1)
            image_masks, label_mask = image.read_masks(), label.read_masks(1)
            image_valid, label_valid, joint_valid = valid_support(values, image_masks, labels, label_mask)
            classes, counts = np.unique(labels, return_counts=True)
            record["label_value_counts"] = {str(value): int(count) for value, count in zip(classes, counts)}
            nonfinite_label = ~np.isfinite(labels)
            if np.any((label_mask > 0) & nonfinite_label):
                errors.append("nonfinite_valid_label_values")
            unknown = (label_mask > 0) & np.isfinite(labels) & ~np.isin(labels, (-1, 0, 1))
            if unknown.any():
                errors.append("unexpected_label_values")
            record.update(
                total_pixels=int(labels.size), image_valid_pixels=int(image_valid.sum()),
                label_valid_pixels=int(label_valid.sum()), joint_valid_pixels=int(joint_valid.sum()),
                label_valid_without_image_pixels=int((label_valid & ~image_valid).sum()),
                image_nonfinite_pixels=int((~np.isfinite(values).all(axis=0)).sum()),
                label_nonfinite_pixels=int(nonfinite_label.sum()), ignored_label_pixels=int((labels == -1).sum()),
            )
            _, record["label_water_pixels"], record["label_water_fraction"] = water_fraction(labels, label_valid)
            _, record["joint_water_pixels"], record["joint_water_fraction"] = water_fraction(labels, joint_valid)
            if not image_valid.any():
                errors.append("no_valid_image_pixels")
            if not label_valid.any():
                errors.append("no_valid_label_pixels")
            if not joint_valid.any():
                errors.append("no_joint_valid_pixels")
            if record["image_nonfinite_pixels"]:
                warnings.append("nonfinite_image_pixels_excluded")
            if record["label_valid_without_image_pixels"]:
                warnings.append("label_support_exceeds_image_support")
            selected = joint_valid if support == "joint" else label_valid
            count, water, fraction = water_fraction(labels, selected)
            record.update(reference_valid_pixels=count, reference_water_pixels=water,
                          valid_fraction=count / labels.size)
            if 0 < record["valid_fraction"] < 0.1:
                warnings.append("less_than_10_percent_valid_support")
            if not errors:
                record.update(reference_water_fraction=fraction, usable=True)
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        errors.append("raster_read_error")
        record["read_error"] = str(error)
        record.update(usable=False, reference_water_fraction=None)
    return record


def describe(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, **{name: None for name in
                ("mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")},
                "zero_count": 0, "one_count": 0}
    quantiles = np.quantile(values, (0, .05, .25, .5, .75, .95, 1))
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            **dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(float, quantiles))),
            "zero_count": int((values == 0).sum()), "one_count": int((values == 1).sum())}


def summarize(records: list[dict], keys=()) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)
    results = []
    for key, group in sorted(groups.items()):
        usable = [row for row in group if row["usable"]]
        valid = sum(row["reference_valid_pixels"] for row in usable)
        water = sum(row["reference_water_pixels"] for row in usable)
        results.append({
            **dict(zip(keys, key)), "patch_count": len(group), "usable_count": len(usable),
            "unusable_count": len(group) - len(usable),
            "event_count": len({row["event_id"] for row in group}),
            **{f"water_fraction_{name}": value for name, value in
               describe([row["reference_water_fraction"] for row in usable]).items()},
            "pooled_valid_pixels": valid, "pooled_water_pixels": water,
            "pooled_water_fraction": water / valid if valid else None,
        })
    return results


def split_audit(records: list[dict]):
    membership = defaultdict(set)
    for row in records:
        membership[row["event_id"]].add(row["original_split"])
    shared = {event: sorted(splits) for event, splits in sorted(membership.items()) if len(splits) > 1}
    overlaps = []
    for i, first in enumerate(records):
        if not first["usable"]:
            continue
        for second in records[i + 1:]:
            if (not second["usable"] or first["original_split"] == second["original_split"]
                    or first["event_id"] != second["event_id"] or first["image_crs"] != second["image_crs"]):
                continue
            a, b = first["image_bounds"], second["image_bounds"]
            tolerance = max(abs(first["image_transform"][0]), abs(first["image_transform"][4])) * 1e-4
            if min(a[2], b[2]) - max(a[0], b[0]) > tolerance and min(a[3], b[3]) - max(a[1], b[1]) > tolerance:
                overlaps.append({"event_id": first["event_id"], "patch_id_a": first["patch_id"],
                                 "split_a": first["original_split"], "patch_id_b": second["patch_id"],
                                 "split_b": second["original_split"]})
    return shared, overlaps


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def save_csv(path: Path, rows: list[dict], fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = json_safe(row)
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def select_examples(records, count):
    usable = sorted((row for row in records if row["usable"]),
                    key=lambda row: (row["reference_water_fraction"], row["patch_id"]))
    selected = []
    for split in SPLIT_FILES.values():
        group = [row for row in usable if row["original_split"] == split]
        if group and len(selected) < count:
            selected.append(group[len(group) // 2])
    # After covering splits, prioritize both global extremes before intermediate
    # quantiles; otherwise a low-water-heavy dataset can yield only dry examples.
    candidates = ([0, len(usable) - 1, *np.linspace(0, len(usable) - 1, max(count, 1)).round().astype(int)]
                  if usable else [])
    for index in candidates:
        row = usable[index]
        if row not in selected and len(selected) < count:
            selected.append(row)
    return selected


def plot_outputs(records, sources, output: Path, count: int, support: str):
    usable = [row for row in records if row["usable"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for split in SPLIT_FILES.values():
        values = [row["reference_water_fraction"] for row in usable if row["original_split"] == split]
        if values:
            axes[0].hist(values, bins=np.linspace(0, 1, 21), histtype="step", label=f"{split} (n={len(values)})")
    axes[0].set(xlabel="Reference water fraction C_i", ylabel="Patch count", xlim=(0, 1), title="By original split")
    if usable:
        axes[0].legend(fontsize=8)
        events = sorted({row["event_id"] for row in usable})
        axes[1].boxplot([[row["reference_water_fraction"] for row in usable if row["event_id"] == event]
                         for event in events], tick_labels=events, orientation="horizontal")
    else:
        axes[1].text(.5, .5, "No usable patches", ha="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Reference water fraction C_i", xlim=(0, 1), title="By event (usable patches)")
    fig.savefig(output / "water_fraction_distributions.png", dpi=150)
    plt.close(fig)

    examples = select_examples(records, count)
    if not examples:
        (output / "rgb_mask_examples.png").unlink(missing_ok=True)
        return []
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 4.5 * len(examples)), squeeze=False)
    source_map = {row["manifest_line"]: row for row in sources}
    for index, row in enumerate(examples):
        source = source_map[row["manifest_line"]]
        with rasterio.open(source["_image_path"]) as image, rasterio.open(source["_label_path"]) as label:
            rgb, alpha = stretch_rgb(image.read((4, 3, 2), masked=True))
            labels = label.read(1)
            _, label_valid, joint = valid_support(image.read(), image.read_masks(), labels, label.read_masks(1))
            selected = joint if support == "joint" else label_valid
        axes[index, 0].set_facecolor("black")
        axes[index, 0].imshow(rgb, alpha=alpha.astype(float), interpolation="nearest")
        axes[index, 0].set_title(f"{row['patch_id']} | {row['original_split']}\nB4/B3/B2; 2-98% display stretch")
        axes[index, 1].imshow(np.where(selected, labels, -1),
                              cmap=ListedColormap(list(CLASS_COLORS.values())),
                              norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), interpolation="nearest")
        axes[index, 1].set_title(f"Reference mask on {support} support\nC_i={row['reference_water_fraction']:.3f}; valid={row['valid_fraction']:.1%}")
        for axis in axes[index]:
            axis.set(xlabel="Pixel column", ylabel="Pixel row")
    fig.legend(handles=[Patch(color=CLASS_COLORS[value], label=f"{value}: {name}")
                        for value, name in CLASS_NAMES.items()], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(output / "rgb_mask_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [row["patch_id"] for row in examples]


def write_report(output, report, grouped, records):
    overall = report["overall"]
    lines = ["# Segmentation data inspection", "", f"Manifest: `{report['manifest']}`", "",
             f"Support for C_i: **{report['reference_support']}**, on the original raster grid.", "",
             f"Patches: **{overall['patch_count']}**; usable: **{overall['usable_count']}**; "
             f"unusable: **{overall['unusable_count']}**.", "",
             "C_i = water pixels / valid pixels. Missing fractions stay undefined; they are never set to zero.", "",
             "| Original split | Patches | Usable | Mean C_i | SD | Min | Median | Max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return f"{value:.4f}" if value is not None else "N/A"
    for row in grouped["by_split"]:
        lines.append(f"| {row['original_split']} | {row['patch_count']} | {row['usable_count']} | "
                     + " | ".join(number(row[f"water_fraction_{name}"]) for name in ("mean", "std", "min", "median", "max")) + " |")
    lines += ["", "## Data quality", "", f"Error flags: `{report['error_counts']}`", "",
              f"Warning flags: `{report['warning_counts']}`", "",
              f"Events shared between original splits: `{report['events_shared_between_splits']}`", "",
              f"Cross-split intersecting bounding boxes: **{report['cross_split_bbox_intersections']}** "
              "(bounding-box screening only; does not establish independent patches).", "",
              "### Unusable patches", "", "| Patch | Original split | Flags |",
              "|---|---|---|"]
    for row in records:
        if not row["usable"]:
            lines.append(f"| {row['patch_id']} | {row['original_split']} | {', '.join(row['errors'])} |")
    lines += ["", "A label NoData value of 0 or 1 conflicts with the class encoding. "
              "Such patches are withheld for review, since trusting that metadata could silently "
              "remove an entire valid class from the denominator.", "",
              "## Interpretation", "", *report["interpretation"], "",
              "See `patches.csv`, `by_event.csv`, `by_split.csv`, `by_event_split.csv`, "
              "`cross_split_bbox_intersections.csv`, and `inspection_summary.json` for full values.", "",
              "![Water fraction distributions](water_fraction_distributions.png)", ""]
    if report["example_patch_ids"]:
        lines += ["![RGB and reference mask examples](rgb_mask_examples.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_inspection(manifest: Path, output: Path, *, support="joint", examples=6,
                   grid_tolerance=1e-4, expected_size=512):
    manifest, output = manifest.resolve(), output.resolve()
    if support not in ("joint", "label"):
        raise ValueError("support must be joint or label")
    if examples < 0 or not np.isfinite(grid_tolerance) or grid_tolerance <= 0:
        raise ValueError("Examples must be nonnegative and grid tolerance positive/finite.")
    sources = read_manifest(manifest)
    assignments = official_splits(manifest.parent)
    records = []
    for index, source in enumerate(sources, 1):
        records.append(inspect_pair(source, support=support, grid_tolerance=grid_tolerance,
                                    expected_size=expected_size, assignments=assignments))
        if index % 50 == 0 or index == len(sources):
            print(f"Inspected {index}/{len(sources)} pairs", flush=True)
    grouped = {"by_split": summarize(records, ("original_split",)),
               "by_event": summarize(records, ("event_id",)),
               "by_event_split": summarize(records, ("event_id", "original_split"))}
    overall = summarize(records)[0]
    shared, overlaps = split_audit(records)
    interpretation = [
        "Label support requires GDAL-valid, finite labels in {0,1}; -1 is ignored explicitly.",
        "Joint support also requires GDAL-valid, finite observations in every one of the 13 image bands.",
        "Both label-only and joint-support fractions are retained. Reference fractions are withheld for unusable patches.",
        "All-water and all-non-water patches are retained when otherwise valid. No resizing, cropping, normalization, or label remapping is performed.",
        "Group means weight patches equally; pooled fractions weight by valid-pixel count. Fractions are pixel fractions, not geodesic area estimates.",
        "The less-than-10%-support flag is descriptive and does not exclude patches.",
        "Valid support describes data availability and known labels; no extra cloud/shadow detector is applied.",
        "Variation in C_i supports a candidate confounder but does not establish causal relevance, measurement quality, or treatment overlap.",
        "Original splits are preserved. Shared events and possible spatial overlap require care with dependent patches and later causal calibration/evaluation splits.",
        "This is Sen1Floods11, whereas SHRUG-FM's flood experiment uses WorldFloods; it is an adapted benchmark, not a dataset-identical reproduction.",
    ]
    if overall["water_fraction_count"] < 2 or overall["water_fraction_std"] == 0:
        interpretation.append("INSUFFICIENT OBSERVED VARIATION: fewer than two usable patches or constant C_i.")
    if assignments is None:
        interpretation.append("Official split CSVs are incomplete/missing: split labels were read from the manifest but could not be cross-checked.")
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "patches.csv", records)
    for name, rows in grouped.items():
        save_csv(output / f"{name}.csv", rows)
    save_csv(output / "cross_split_bbox_intersections.csv", overlaps,
             ("event_id", "patch_id_a", "split_a", "patch_id_b", "split_b"))
    example_ids = plot_outputs(records, sources, output, examples, support)
    report = {
        "inspection_schema_version": 2,
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(), "reference_support": support,
        "grid_tolerance_pixels": grid_tolerance, "expected_patch_size": expected_size,
        "official_splits_cross_checked": assignments is not None, "overall": overall,
        "official_split_inventory_count": len(assignments) if assignments is not None else None,
        "error_counts": dict(Counter(error for row in records for error in row["errors"])),
        "warning_counts": dict(Counter(warning for row in records for warning in row["warnings"])),
        "events_shared_between_splits": shared, "cross_split_bbox_intersections": len(overlaps),
        "example_patch_ids": example_ids, "groups": grouped, "interpretation": interpretation,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "rasterio": rasterio.__version__},
    }
    (output / "inspection_summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, report, grouped, records)
    print(f"Usable: {overall['usable_count']}/{overall['patch_count']}; errors: {report['error_counts']}")
    print(f"Water fraction: mean={overall['water_fraction_mean']}, SD={overall['water_fraction_std']}, "
          f"range=({overall['water_fraction_min']}, {overall['water_fraction_max']})")
    print(f"Report: {output / 'report.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/sen1floods11/manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/segmentation_data")
    parser.add_argument("--support", choices=("joint", "label"), default="joint")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--grid-tolerance", type=float, default=1e-4, help="Maximum grid displacement in pixels.")
    args = parser.parse_args()
    try:
        run_inspection(args.manifest, args.output, support=args.support,
                       examples=args.examples, grid_tolerance=args.grid_tolerance)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
