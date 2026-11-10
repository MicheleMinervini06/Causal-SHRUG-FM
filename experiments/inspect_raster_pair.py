"""Inspect and display the local Sen1Floods11 Sentinel-2 image and label pair."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
CLASS_NAMES = {-1: "Ignored / invalid", 0: "Not water", 1: "Water"}
CLASS_COLORS = {-1: "#bdbdbd", 0: "#f0e5c9", 1: "#1675ba"}


def print_properties(name: str, source: rasterio.io.DatasetReader) -> None:
    print(f"\n{name}: {source.name}")
    print(f"  Driver: {source.driver}")
    print(f"  Shape (height, width): {source.shape}")
    print(f"  Bands: {source.count}; data types: {source.dtypes}")
    print(f"  Band descriptions: {source.descriptions}")
    print(f"  CRS: {source.crs}")
    print(f"  Pixel size (CRS units): {source.res}")
    print(f"  Bounds: {source.bounds}")
    print(f"  Affine transform (a, b, c, d, e, f): {tuple(source.transform)[:6]}")
    print(f"  Declared NoData: {source.nodata}")


def stretch_rgb(rgb: np.ma.MaskedArray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [bands, rows, cols] to RGB; stretch valid pixels for display only."""
    values = np.moveaxis(rgb.astype(np.float32).filled(np.nan), 0, -1)
    valid = ~np.ma.getmaskarray(rgb).any(axis=0) & np.isfinite(values).all(axis=-1)
    if not valid.any():
        raise ValueError("No valid pixels are available in the selected RGB bands.")
    low, high = np.percentile(values[valid], [2, 98], axis=0)
    scale = np.where(high > low, high - low, 1.0)
    stretched = np.nan_to_num(np.clip((values - low) / scale, 0, 1), nan=0.0)
    return stretched, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DATA_DIR / "images" / "Bolivia_195474_S2Hand.tif")
    parser.add_argument("--mask", type=Path, default=DATA_DIR / "labels" / "Bolivia_195474_LabelHand.tif")
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=(4, 3, 2), metavar=("R", "G", "B"),
                        help="Rasterio's 1-based band indices; default 4 3 2 = Sentinel-2 B4 B3 B2.")
    parser.add_argument("--output", type=Path, default=Path("results/sen1floods11/Bolivia_103757_preview.png"))
    parser.add_argument("--show", action="store_true", help="Also open an interactive Matplotlib window.")
    args = parser.parse_args()

    # Context managers close each GeoTIFF after reading; source files are not modified.
    with rasterio.open(args.image) as image, rasterio.open(args.mask) as label:
        print_properties("Image", image)
        print_properties("Label", label)
        if label.count != 1:
            parser.error("Expected a single-band Sen1Floods11 label raster.")
        if not all(1 <= band <= image.count for band in args.rgb_bands):
            parser.error(f"RGB band indices must be between 1 and {image.count}.")

        same_grid = (
            image.shape == label.shape
            and image.crs == label.crs
            and np.allclose(tuple(image.transform), tuple(label.transform), rtol=0, atol=1e-10)
        )
        if not same_grid:
            parser.error("Image and label grids differ; align them before comparing pixels.")
        print("\nGrid check: matching dimensions, CRS and transform (absolute tolerance 1e-10).")

        # Rasterio bands start at 1. Sentinel-2 natural colour is B4 (red), B3, B2.
        rgb = image.read(args.rgb_bands, masked=True)  # [3, height, width]
        mask = label.read(1)  # [height, width]; -1 is not declared as NoData here.
        label_valid = label.read_masks(1) > 0
        classes, counts = np.unique(mask, return_counts=True)
        print(f"RGB array: {rgb.shape}; label array: {mask.shape}")
        print("\nLabel values and pixel counts:")
        for value, count in zip(classes.tolist(), counts.tolist()):
            print(f"  {value}: {CLASS_NAMES.get(value, 'Unknown class')}: {count:,}")
        if not np.isin(mask, list(CLASS_NAMES)).all():
            parser.error("Unexpected mask values: this example expects only -1, 0 and 1.")
        # Exclude both semantic invalid labels and any GDAL-masked pixels.
        valid = label_valid & (mask != -1)
        water_pixels = np.count_nonzero(valid & (mask == 1))
        if valid.any():
            print(f"Water fraction over valid labels: {water_pixels / valid.sum():.2%}")
        else:
            print("Water fraction: undefined (no valid label pixels).")

    display_rgb, image_valid = stretch_rgb(rgb)
    display_mask = np.where(label_valid, mask, -1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7))
    axes[0].set_facecolor("black")
    axes[0].imshow(display_rgb, alpha=image_valid.astype(float), interpolation="nearest")
    axes[0].set_title(f"Sentinel-2 natural colour\nBands {tuple(args.rgb_bands)}; 2-98% display stretch")
    cmap = ListedColormap(list(CLASS_COLORS.values()))
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axes[1].imshow(display_mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Reference water mask\nSen1Floods11: Bolivia_103757")
    fig.legend(
        handles=[Patch(color=CLASS_COLORS[value], label=f"{value}: {name}")
                 for value, name in CLASS_NAMES.items()],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False,
    )
    for ax in axes:
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", pad_inches=0.15)
    print(f"\nSaved preview: {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
