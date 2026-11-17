"""Encode the downloaded original SSL4EO reference using our downstream transform."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

if __package__:
    from . import cache_encoder_features as cache
    from .download_ssl4eo_reference import DEFAULT_OUTPUT
else:
    import cache_encoder_features as cache
    from download_ssl4eo_reference import DEFAULT_OUTPUT


def representation(contract):
    """Cross-device compatibility concerns the encoder/transform, not cache paths."""
    return {"encoder": contract["encoder"], "preprocessing": contract["preprocessing"],
            "embedding": "forward_features CLS token; 384 float32 values; no extra L2 normalization"}


def read_reference_image(path):
    """Match the original wrapper's per-band nearest array resize to 264x264.

    Original band geotransforms have sub-native-pixel offsets. Preserve the
    released pixel arrays rather than silently correcting the pretraining data.
    """
    raw, valid = [], []
    with np.load(path, allow_pickle=False) as image:
        if set(image.files) != set(cache.SENTINEL2_BANDS):
            raise ValueError(f"Missing/extra reference bands: {path}")
        with MemoryFile(image["B2"].tobytes()) as memory, memory.open() as blue:
            shape, crs, bounds = blue.shape, blue.crs, np.asarray(blue.bounds)
            pixel_size = np.asarray([abs(blue.transform.a), abs(blue.transform.e)] * 2)
        if shape != (264, 264) or crs is None:
            raise ValueError("Expected georeferenced 264x264 original B2 grid.")
        for band in cache.SENTINEL2_BANDS:
            with MemoryFile(image[band].tobytes()) as memory, memory.open() as source:
                if (source.count != 1 or source.dtypes != ("uint16",) or source.crs != crs
                        or not np.isin(source.shape, [44, 132, 264]).all()
                        or np.max(np.abs(np.asarray(source.bounds) - bounds) / pixel_size) > 3.1):
                    raise ValueError(f"Unexpected raw band type or footprint: {path}, {band}")
                # rasterio's default read resampling is nearest, as in the released wrapper.
                raw.append(source.read(1, out_shape=shape))
                valid.append(source.read_masks(1, out_shape=shape) > 0)
    return np.stack(raw), np.stack(valid)


def encode(reference, output, checkpoint, settings, *, device="cpu", batch_size=32, threads=4):
    if batch_size < 1 or threads < 1:
        raise ValueError("Batch size and threads must be positive.")
    reference, output = Path(reference), Path(output)
    summary = json.loads((reference / "download_summary.json").read_text())
    manifest = reference / "reference_manifest.csv"
    if cache.digest(manifest) != summary["manifest_sha256"]:
        raise ValueError("Reference manifest changed since download.")
    rows = cache.read_rows(manifest, ("patch_id", "image_path", "sha256", "location_id", "season_id"))
    torch, _ = cache.encoder_dependencies()
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    contract = cache.cache_contract(cache.preprocessing_config(settings), device)
    metadata = {"contract": contract, "reference_complete": summary["reference_complete"],
                "reference_manifest_sha256": cache.digest(manifest),
                "reference_download_config": json.loads((reference / "download_config.json").read_text()),
                "reference_download_summary": summary,
                "band_alignment": "per-band nearest array resize to 264x264, as released wrapper; shared CRS; bounds within 3.1 B2 pixels",
                "georeferencing_limitation": "original inter-band offsets retained; array resize does not repair original geospatial alignment",
                "implementation_sha256": cache.digest(Path(__file__)),
                "embedding": representation(contract)["embedding"]}
    identity = cache.fingerprint(metadata)
    output.mkdir(parents=True, exist_ok=True)
    vectors, pending = {}, []
    for row in rows:
        path = cache.relative_path(reference, row["image_path"])
        if cache.digest(path) != row["sha256"]:
            raise ValueError(f"Changed reference image: {row['patch_id']}")
        cached = output / "patches" / f"{row['patch_id']}.npz"
        if cached.exists():
            with np.load(cached, allow_pickle=False) as archive:
                info, vector = json.loads(archive["metadata"].item()), archive["cls_embedding"]
                if (info != {"identity": identity, "image_sha256": row["sha256"], "payload_sha256": cache.array_hash(vector)}
                        or vector.shape != (384,) or vector.dtype != np.float32 or not np.isfinite(vector).all()):
                    raise ValueError(f"Incompatible reference cache: {cached}; choose a new output.")
                vectors[row["patch_id"]] = vector.copy()
        else:
            pending.append(row)
    if pending:
        encoder = cache.load_encoder(checkpoint, device)
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                images = [cache.preprocess_image(*read_reference_image(cache.relative_path(reference, r["image_path"])),
                                                 contract["preprocessing"]) for r in batch]
                tokens = encoder.forward_features(torch.stack(images).to(device))
                if tuple(tokens.shape[1:]) != (197, 384) or not torch.isfinite(tokens).all().item():
                    raise ValueError("Invalid encoder tokens.")
                cls = tokens[:, 0, :].float().cpu().numpy()
                for row, vector in zip(batch, cls):
                    vectors[row["patch_id"]] = vector.copy()
                    info = {"identity": identity, "image_sha256": row["sha256"], "payload_sha256": cache.array_hash(vector)}
                    cache.save_npz(output / "patches" / f"{row['patch_id']}.npz", cls_embedding=vector,
                                   metadata=np.asarray(json.dumps(info, sort_keys=True)))
                print(f"Encoded {min(start + batch_size, len(pending))}/{len(pending)} pending images", flush=True)
    matrix = np.stack([vectors[r["patch_id"]] for r in rows])
    metadata["payload_sha256"] = cache.array_hash(matrix)
    cache.save_npz(output / "pretraining_cls.npz", cls_embeddings=matrix,
                   patch_ids=np.asarray([r["patch_id"] for r in rows]),
                   location_ids=np.asarray([r["location_id"] for r in rows]),
                   season_ids=np.asarray([r["season_id"] for r in rows]),
                   metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    cache.write_json(output / "embedding_summary.json", {"n_images": len(rows), "reference_complete": summary["reference_complete"],
                     "identity": identity, "archive_sha256": cache.digest(output / "pretraining_cls.npz"),
                     "representation": representation(contract)})
    print(f"Pretraining embeddings: {output / 'pretraining_cls.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "embeddings")
    parser.add_argument("--checkpoint", type=Path, default=cache.MODEL_DIR / cache.FILENAME)
    parser.add_argument("--settings", type=Path, default=cache.ROOT / "src/config/dataset.toml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    encode(args.reference, args.output, args.checkpoint, args.settings,
           device=args.device, batch_size=args.batch_size, threads=args.threads)


if __name__ == "__main__":
    main()
