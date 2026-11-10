#!/usr/bin/env python3
"""Download and optionally smoke-test the SSL4EO-S12 MoCo ViT-S/16 encoder.

Download only (standard library):
  uv run scripts/get_ssl4eo_encoder.py
Download, load, and encode a real image (torch, timm, rasterio required):
  uv run --extra encoder scripts/get_ssl4eo_encoder.py --image data/sen1floods11/images/Bolivia_103757_S2Hand.tif

This downloads pretrained encoder weights, not a water-segmentation decoder.
Use --dry-run to print the plan without network access or file writes.
Use --verify-only to verify an existing local checkpoint without downloading.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

REPOSITORY = "torchgeo/vit_small_patch16_224_sentinel2_all_moco"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "ssl4eo-s12-moco"
SENTINEL2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
REVISION = "1cb683f6c14739634cdfaaceb076529adf898c74"
FILENAME = "vit_small_patch16_224_sentinel2_all_moco-67c9032d.pth"
SHA256 = "67c9032df0449ae37f93c51696bcae0c6dc3485103deb44b4669e9897b90409c"
EXPECTED_BYTES = 90668909
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def valid_checkpoint(path):
    return path.is_file() and path.stat().st_size == EXPECTED_BYTES and digest(path) == SHA256


def download(path):
    if valid_checkpoint(path):
        print("Reusing verified checkpoint.", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "SHRUG-FM-SSL4EO-encoder/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                received, announced = 0, 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    received += len(block)
                    if received - announced >= 20 * 1024 * 1024:
                        print(f"Downloaded {received / 1024**2:.0f}/{EXPECTED_BYTES / 1024**2:.0f} MiB", flush=True)
                        announced = received
            if not valid_checkpoint(partial):
                raise ValueError("Checkpoint size or SHA-256 does not match the pinned release")
            partial.replace(path)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def encoder_dependencies():
    """Import optional packages only when loading or checking the encoder."""
    try:
        import torch
        import timm
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Encoder dependencies are unavailable. Run `uv sync --extra encoder` "
            "and use `uv run --extra encoder scripts/get_ssl4eo_encoder.py --check`."
        ) from error
    return torch, timm


def load_encoder(checkpoint, device="cpu"):
    """Load the verified local checkpoint as a frozen feature encoder; no download."""
    checkpoint = Path(checkpoint)
    if not valid_checkpoint(checkpoint):
        raise ValueError("Local checkpoint is missing or does not match the pinned size/SHA-256.")
    torch, timm = encoder_dependencies()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu for local validation.")
    print("Loading pretrained encoder...", flush=True)
    model = timm.create_model("vit_small_patch16_224", pretrained=False, in_chans=13, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Unexpected checkpoint structure")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model


def read_image(image):
    """Read all Sentinel-2 bands and describe validity before any resizing."""
    import rasterio
    import numpy as np

    image = Path(image)
    with rasterio.open(image) as src:
        if src.count != 13:
            raise ValueError(f"Expected all 13 Sentinel-2 bands; found {src.count}")
        descriptions = src.descriptions
        if any(descriptions) and tuple(descriptions) != SENTINEL2_BANDS:
            raise ValueError(f"Unexpected Sentinel-2 band order: {descriptions}")
        raw = src.read().astype("float32")
        if not np.isfinite(raw).all():
            raise ValueError("The input contains non-finite values")
        valid = (src.read_masks() > 0).all(axis=0)
        if not valid.any():
            raise ValueError("The image has no valid pixels across all 13 bands.")
        info = {
            "input_source": str(image.resolve()), "image_sha256": digest(image),
            "raster_shape": list(raw.shape), "crs": str(src.crs),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "nodata": src.nodata, "valid_pixel_fraction": float(valid.mean()),
            "source_band_descriptions": list(descriptions),
            "band_order_check": "verified from descriptions" if any(descriptions)
                                else "assumed standard Sentinel-2 order; descriptions absent",
        }
        return raw, info


def smoke_test(checkpoint, image, device, threads):
    torch, timm = encoder_dependencies()
    torch.set_num_threads(threads)
    model = load_encoder(checkpoint, device)

    info = {"torch": torch.__version__, "timm": timm.__version__, "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_dict_loading": "strict; all keys matched", "frozen": True,
            "checkpoint_sha256": SHA256,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "forward-pass/shape check only; not a validated measurement pipeline"}
    if image is None:
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(1, 13, 224, 224, generator=generator)
        info["input_source"] = "synthetic smoke-test tensor"
    else:
        raw, image_info = read_image(image)
        info.update(image_info)
        # A forward-pass check only. No labels are cropped/resized or fractions computed here.
        x = torch.from_numpy(raw).unsqueeze(0).div(10000).clamp(0, 1)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        info["preprocessing"] = "13 bands in source order; divide by 10000; clip to [0,1]; resize full patch to 224x224"
        info["preprocessing_note"] = (
            "Smoke-test transform from the supplied script. TorchGeo's documented "
            "weight transform instead resizes to 256, center-crops to 224 and divides "
            "by 10000. NoData is not removed from this forward-pass check. Choose "
            "measurement support and preprocessing before downstream evaluation."
        )
    x = x.to(device)
    with torch.inference_mode():
        tokens = model.forward_features(x)
    if tuple(tokens.shape) != (1, 197, 384):
        raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
    if not torch.isfinite(tokens).all().item():
        raise ValueError("The encoder produced non-finite features")
    info.update(input_shape=list(x.shape), token_shape=list(tokens.shape),
                cls_shape=list(tokens[:, 0, :].shape), patch_token_shape=list(tokens[:, 1:, :].shape),
                feature_min=float(tokens.min()), feature_max=float(tokens.max()),
                result="ENCODER CHECK PASSED")
    return info


def checkpoint_metadata():
    return {"repository": REPOSITORY, "revision": REVISION, "url": URL,
            "filename": FILENAME, "bytes": EXPECTED_BYTES, "sha256": SHA256,
            "architecture": "vit_small_patch16_224", "pretraining": "SSL4EO-S12 MoCo",
            "bands": 13, "band_order": list(SENTINEL2_BANDS), "license": "CC-BY-4.0",
            "scope": "pretrained encoder; no segmentation head"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination (default: this project's models/ssl4eo-s12-moco).")
    parser.add_argument("--check", action="store_true", help="Load the model and run a synthetic forward pass")
    parser.add_argument("--image", type=Path, help="Use a real 13-band S2Hand image; implies --check")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the pinned checkpoint plan; no network access or file writes.")
    mode.add_argument("--verify-only", action="store_true", help="Verify the local checkpoint; no download or file writes.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    output = args.output.resolve()
    checkpoint = output / FILENAME
    metadata = checkpoint_metadata()
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        print(f"Destination: {checkpoint}")
        print(f"Verified local checkpoint: {valid_checkpoint(checkpoint)}")
        print("DRY RUN: no network access; no weights downloaded or files written.")
        return
    if args.verify_only:
        if args.check or args.image is not None:
            parser.error("--verify-only cannot be combined with --check or --image")
        if not valid_checkpoint(checkpoint):
            raise ValueError(f"Checkpoint missing or invalid: {checkpoint}")
        print(f"CHECKPOINT VERIFIED: {checkpoint}")
        return
    if args.check or args.image is not None:
        # Fail before downloading weights if the optional runtime is missing.
        torch, _ = encoder_dependencies()
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")
    download(checkpoint)
    metadata["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "checkpoint_metadata.json", metadata)
    print(f"CHECKPOINT VERIFIED: {checkpoint}", flush=True)
    if args.check or args.image is not None:
        report_path = output / "encoder_check.json"
        report_path.unlink(missing_ok=True)
        report = smoke_test(checkpoint, args.image, args.device, args.threads)
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        print("ENCODER CHECK PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
