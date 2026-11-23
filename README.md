# SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation

## Abstract
Geospatial foundation models (GFMs) for Earth observation often fail to perform reliably in environments underrepresented during pretraining. We introduce SHRUG-FM, a framework for reliability-aware prediction that enables GFMs to identify and abstain from likely failures. Our approach integrates three complementary signals: geophysical out-of-distribution (OOD) detection in the input space, OOD detection in the embedding space, and task-specific predictive uncertainty. We evaluate SHRUG-FM across three high-stakes rapid-mapping tasks: burn scar segmentation, flood mapping, and landslide detection. Our results show that SHRUG-FM consistently reduces prediction risk on retained samples, outperforming established single-signal baselines like predictive entropy. Crucially, by utilizing a shallow "glass-box" decision tree for signal fusion, SHRUG-FM provides interpretable abstention thresholds. It builds a pathway toward safer and more interpretable deployment of GFMs in climate-sensitive applications, bridging the gap between benchmark performance and real-world reliability. 


## Cite
```bibtex
@misc{gonzalezcalabuig2026shrugfmreliabilityawarefoundationmodels,
      title={SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation}, 
      author={Maria Gonzalez-Calabuig and Kai-Hendrik Cohrs and Vishal Nedungadi and Zuzanna Osika and Ruben Cartuyvels and Steffen Knoblauch and Joppe Massant and Shruti Nath and Patrick Ebel and Vasileios Sitokonstantinou},
      year={2026},
      eprint={2511.10370},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.10370}, 
}
```

## Synthetic causal baseline

Use uv to create and synchronize the project virtual environment at `.venv`.
The project pins Python 3.13 in `.python-version`:

```sh
uv sync --locked
```

Run the standalone benchmark in that environment. The synthetic benchmark itself
uses only the Python standard library; no datasets, weights, or GPU are required:

```sh
uv run experiments/synthetic_confounder.py
```

It generates `C ~ N(0,1)`, `T ~ Bernoulli(sigmoid(a*C))`,
`Y = T + C + epsilon`, and `C_hat = C + sigma*Z`, with independent
standard-normal outcome and measurement noise. The true treatment effect is 1.
It compares the treatment coefficients from `Y ~ 1 + T`, `Y ~ 1 + T + C`,
and `Y ~ 1 + T + C_hat`, for `a = 1, 0` and `sigma = 0, 0.5, 1`.

Defaults are 2,000 units per population, 200 independent repetitions, and seed 42.
Each repetition shares latent draws across assignment scenarios and the same
measurement-noise draw across sigma levels. The table reports mean coefficients,
empirical bias, and RMSE for every estimator in each treatment/error scenario.
Bias is the mean signed error relative to the true effect; RMSE is the square root
of the mean squared error across repetitions. With one repetition, these reduce
to signed error and absolute error. Monte Carlo standard errors of mean bias are
calculated as `sample_SD(estimates) / sqrt(repetitions)`, using the sample SD with
denominator `R - 1`. They are unavailable (`N/A`, or an empty CSV field) for a
single repetition. These describe simulation precision, not the uncertainty of
an individual treatment-effect estimate. No-adjustment and oracle results repeat
across sigma because only the proxy changes. Individual populations need not
recover the effect exactly. No confidence intervals or reliability signals are
implemented in this first baseline.

Before simulation, the script checks the first population for each treatment
setting: array lengths, finite values, binary treatment and both treatment groups.
It prints treatment counts and means/sample SDs for `C`, `T`, `Y`, outcome noise,
measurement noise, and each proxy. It checks exact equality of `C` and `C_hat` at
sigma zero and agreement of the corresponding OLS coefficients. Sample moments
are reported for inspection, without requiring exact agreement with population
moments or claiming that empirical checks prove independence.

For diagnostics alone, a single repetition, or CSV exports:

```sh
uv run experiments/synthetic_confounder.py --diagnostics-only
uv run experiments/synthetic_confounder.py --repeats 1
uv run experiments/synthetic_confounder.py --csv results/e01/estimates.csv --summary-csv results/e01/summary.csv
```

The estimates CSV retains one row per repetition and scenario, with one column
per estimator and the sample size, seed, and true effect. The summary CSV has one
row per scenario and estimator, including bias, RMSE, repetition count, and MCSE.
Exports contain unrounded values. Results directories are ignored by Git.

The completed default run and interpretation are recorded in
[E01 results](experiments/E01_results.md).

Expected behavior: oracle adjustment recovers approximately 1; proxy adjustment
matches it exactly at sigma zero and retains increasing confounding bias as noise
increases in this model. With randomized treatment (`a = 0`), all three methods
recover approximately 1. The synthetic proxy is a stand-in for an FM prediction,
not an evaluation of SHRUG-FM or its reliability signals.

Run the coefficient and simulation checks with:

```sh
uv run python -B -m unittest discover -s tests -v
```

Manage future packages with `uv add <package>` or `uv add --dev <package>`.
Commit `pyproject.toml`, `uv.lock`, and `.python-version`; `.venv` and the local
uv cache are ignored. `uv run` uses the environment without manual activation.
The original FM pipeline's `environment.yml` remains its separate dependency
specification; the uv environment currently covers the causal benchmark.

## Download the hand-labelled Sentinel-2 subset

The adapted downloader is `scripts/download_sen1floods11.py`. It uses the Python
standard library through the project's uv environment; no additional packages,
Google Cloud CLI, or cloud credentials are needed.

First inspect the public inventory, official splits, and expected download size:

```sh
uv run scripts/download_sen1floods11.py --dry-run
```

The dry run reads object metadata and small split CSVs, downloads no rasters, and
writes no files. To download a small sample or the complete labelled S2 subset:

```sh
uv run scripts/download_sen1floods11.py --limit 2
uv run scripts/download_sen1floods11.py
```

The default destination is this project's `data/sen1floods11`, resolved relative
to the script rather than the current working directory. `--output PATH` overrides
it; `--workers 1` through `--workers 16` controls concurrency (default 4).

```text
data/sen1floods11/
  images/             # S2Hand GeoTIFFs
  labels/             # LabelHand GeoTIFFs
  splits/             # Official train, validation, test, Bolivia-holdout CSVs
  metadata/           # Event metadata GeoJSON
  manifest.csv        # Patch/event IDs, original splits and relative file paths
  download_manifest.json  # Object generations, sizes, checksums and selection
```

Existing TIFFs are reused only after checking their byte size and published MD5.
Downloads use temporary files, retries, and replacement only after verification.
Rerun the same command after an interrupted download to reuse verified completed
files. Incomplete files restart from the beginning. Manifests describe the current
selection; `--limit` chooses the first N sorted patch IDs and does not delete other
local files. Official splits, including the Bolivia holdout, remain distinct.

This downloads **hand-labelled Sentinel-2 images and reference masks**, not all
Sen1Floods11 layers. Sentinel-1 and weakly labelled subsets are outside its scope.
See the [dataset structure](https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md).
The downloader verifies files and pairing; inspect raster alignment and valid
pixels separately with the Rasterio script below. Downloaded data is ignored by Git.

## Inspect the local image and label

The Rasterio example opens the Sen1Floods11 image and label in `data/sen1floods11`,
prints their metadata and label counts, checks their grids, and saves a preview:

```sh
uv run experiments/inspect_raster_pair.py
uv run experiments/inspect_raster_pair.py --show
```

The output is `results/sen1floods11/Bolivia_103757_preview.png`. `--show` also opens
a Matplotlib window. Alternate files can be supplied with `--image` and `--mask`.
This example uses NumPy, Rasterio and Matplotlib, managed with uv.

Rasterio uses 1-based band indices: RGB bands `4 3 2` select Sentinel-2 B4/B3/B2.
The 2-98% contrast stretch affects the display only. For the mask, `-1` means
ignored/invalid, `0` means not water, and `1` means water. Water does not necessarily
mean newly flooded land. The example explicitly excludes `-1` when computing
water fraction, even if the raster does not declare it as NoData.

References: [Rasterio quickstart](https://rasterio.readthedocs.io/en/stable/quickstart.html)
and [Sen1Floods11 dataset conventions](https://github.com/cloudtostreet/Sen1Floods11#dataset-information).

## Inspect the full segmentation dataset before decoder training

```sh
uv run experiments/inspect_segmentation_data.py
```

This reads `data/sen1floods11/manifest.csv`, verifies every listed pair, and saves
an audit under `results/segmentation_data`. It runs on CPU with the existing base
dependencies; encoder weights and the optional encoder runtime are unnecessary.
Use `--manifest PATH`, `--output PATH`, or `--examples N` to override defaults.
A notebook is optional: the script produces the tables and figures needed for
interactive follow-up while keeping the underlying checks reproducible.

Checks include IDs, duplicate paths/patches, file existence/readability, 512x512
dimensions, 13-band order, single-band labels, CRS/bounds/affine alignment, label
values, GDAL validity masks, and finite values. When available, local official
split CSVs are cross-checked against the manifest. Affine grids are compared with
an explicit tolerance of `0.0001` pixels so harmless coordinate roundoff is
accepted. No raster is changed, cropped, resized, normalized or remapped.

The default reference confounder is:

```text
C_i = water pixels / valid pixels
valid = known label (0 or 1) AND GDAL-valid label
        AND GDAL-valid, finite observations in every image band
```

Label value `-1` is explicitly ignored, even when the GeoTIFF declares no NoData.
Both `label_water_fraction` (label support alone) and `joint_water_fraction` are
saved. `reference_water_fraction` uses joint support by default; `--support label`
selects label support for an alternate audit. Always match future predictions and
references on the same support. These are native-grid pixel fractions, not exact
geodesic area fractions or a measure of newly flooded land. Validity does not
provide an additional cloud/shadow quality mask.

Outputs include:

- `patches.csv`: one row per manifest entry, properties, both support counts and
  fractions, usability, and error/warning flags. Missing/unusable fractions remain
  empty rather than being converted to zero. Raster and GDAL-sidecar SHA-256
  fingerprints let later steps detect changes since inspection.
- `by_event.csv`, `by_split.csv`, `by_event_split.csv`: all/usable patch counts,
  mean, sample SD, quantiles, range, zero/one counts, and pooled water fractions.
  Patch means and valid-pixel-weighted pooled fractions are distinct quantities.
- `water_fraction_distributions.png` and `rgb_mask_examples.png`: distributions
  and deterministic examples covering original splits and fraction variation.
  RGB stretching is display-only; masks show the chosen reference support.
- `cross_split_bbox_intersections.csv`: possible geographic overlap within an
  event across splits, for review rather than automatic exclusion.
- `inspection_summary.json` and `report.md`: configuration/provenance, findings,
  shared-event warnings, and interpretation.

All-water and all-non-water patches are retained when otherwise valid. Empty
image/label/joint support is unusable. Less than 10% valid support is a descriptive
warning, not an exclusion rule. The script preserves the official splits and
reports events shared between them; patch counts do not equal independent event
counts. Original splits can support decoder comparisons, but causal calibration
and evaluation will need an explicit plan for dependent patches.

Reuse/method check: RGB stretching, mask colours and class labels are reused from
`experiments/inspect_raster_pair.py`. SHRUG-FM's
`src/signal_task/dataset/floods/dataset.py` targets **WorldFloods**, with a two-band
source mask mapped to `0/1/2`; our **Sen1Floods11** source mask is one band with
`-1/0/1`. That loader and the tester's raw `GT_Area` sum cannot be used unchanged
for this inspection. Before decoder training, review the existing `CoreDecoder`
in `src/signal_task/models/model_components/decoder.py`, the frozen-encoder wiring
in `src/signal_task/models/model_ssl4eo.py`, and training normalization/loss settings. This audit
does not train a decoder or constitute a dataset-identical SHRUG-FM reproduction.

## Acquire the SSL4EO-S12 MoCo encoder

`scripts/get_ssl4eo_encoder.py` downloads a pinned SSL4EO-S12 MoCo ViT-S/16
checkpoint. Its default destination is this project's `models/ssl4eo-s12-moco`,
resolved relative to the script. Model artifacts and reports are ignored by Git.

Inspect the plan without network access or file writes, then download when ready:

```sh
uv run scripts/get_ssl4eo_encoder.py --dry-run
uv run scripts/get_ssl4eo_encoder.py
```

The script verifies the expected 90,668,909 bytes (about 86.5 MiB) and full SHA-256,
reuses an existing checkpoint only if both match, and replaces temporary downloads
only after verification. Source repository, revision, URL, band order, checksum,
and license are recorded in `checkpoint_metadata.json`. `--output PATH` overrides
the destination. To verify an existing file without downloading or writing:

```sh
uv run scripts/get_ssl4eo_encoder.py --verify-only
```

Download-only and verification modes use the standard library. Loading and
forward-pass checks use the optional `encoder` extra (PyTorch, torchvision, timm),
locked in `uv.lock`. This extra selects the official CPU PyTorch index for local
checks; `encoder-cuda` selects CUDA 12.8 for NVIDIA GPU work (see below).
Install/run the CPU extra explicitly:

```sh
uv sync --extra encoder --locked
uv run --extra encoder scripts/get_ssl4eo_encoder.py --check
uv run --extra encoder scripts/get_ssl4eo_encoder.py --image data/sen1floods11/images/Bolivia_103757_S2Hand.tif
```

`--check` uses a seeded synthetic input; `--image` implies a check on a real raster.
Use the same selected extra on every `uv run` so uv retains that backend.
CPU is the script default; `--threads` controls CPU threads. For NVIDIA GPU work,
use `--extra encoder-cuda` together with `--device cuda`.

The reusable `load_encoder(checkpoint, device="cpu")` function verifies the local
file, uses `torch.load(weights_only=True)` and strict state-dict loading, and
returns a frozen model in evaluation mode. It does not download anything. Expected
input is `[B,13,224,224]`; `forward_features` returns `[B,197,384]`, with one CLS
token `[B,384]` and 196 patch tokens `[B,196,384]`. It has no water-segmentation head.
The optional smoke check records versions, shapes and finite-feature checks in
`encoder_check.json`; it does not save a dataset of embeddings or predictions.

Real-image checks verify the 13-band order when descriptions are present and
record valid-pixel fraction. They retain the supplied script's **smoke-test-only**
preprocessing: divide by 10,000, clip to `[0,1]`, and resize the full patch to
224x224. This still includes image NoData and is not the documented TorchGeo
weight transform (resize to 256, center-crop to 224, divide by 10,000). Choose
preprocessing and matching label support before evaluating measurements. A
successful forward pass alone does not validate either choice.

This script uses MoCo explicitly. The legacy embedding exporter still defaults
to MAE; acquiring these weights does not reconfigure that pipeline.
Sources: [pinned checkpoint](https://huggingface.co/torchgeo/vit_small_patch16_224_sentinel2_all_moco/tree/1cb683f6c14739634cdfaaceb076529adf898c74),
[TorchGeo weights/transforms](https://docs.torchgeo.org/en/v0.7.1/_modules/torchgeo/models/vit.html),
and [uv PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Cache frozen encoder features

```sh
uv run --extra encoder scripts/cache_encoder_features.py --dry-run
uv run --extra encoder scripts/cache_encoder_features.py --limit 2
uv run --extra encoder scripts/cache_encoder_features.py
```

The script loads the verified local checkpoint through the existing
`load_encoder` helper and uses the usable cohort from the segmentation audit.
It never downloads weights. An older audit without source hashes must first be
regenerated with `uv run experiments/inspect_segmentation_data.py`. Missing,
changed or mismatched images/labels/validity sidecars require a refreshed audit.

The single canonical preprocessing procedure is `shrug_flood_full_patch_v1`:

1. Read all 13 Sentinel-2 bands in verified B1...B12 order (including B8A).
2. Fill GDAL-invalid/nonfinite values with raw zero, independently per band.
3. Standardize in float32 with the fixed 13 means/SDs in
   `src/config/dataset.toml`, `flood.train.normalization`.
4. Resize the full patch to 224x224 with PyTorch `nearest-exact` interpolation.
   No cropping, clipping, division by 10,000, or random augmentation is applied.

This follows the normalization/resize pattern in SHRUG-FM's flood loader and
preserves the full spatial extent for native-grid targets. The fixed statistics
match the [ml4floods all-band normalization](https://github.com/spaceml-org/ml4floods/blob/main/ml4floods/data/worldfloods/configs.py);
no statistics are fitted on our train/test data. It deliberately differs from the
earlier forward-pass smoke transform and TorchGeo's generic center-crop transform.
NoData fill affects encoder context; those pixels remain excluded from downstream
reference/prediction fractions and losses using the original masks.

The model is frozen and in evaluation/inference mode, with float32 computation,
autocast disabled, and highest matmul precision. The spatial layout exactly matches
`SSL4EO.forward/reshape`: remove the CLS token, permute channels, then reshape the
196 row-major spatial tokens into `[384,14,14]`. The legacy `CreateEmbeddings`
exporter saves CLS only and cannot provide this decoder input.

Default output: `data/sen1floods11/encoder_features/ssl4eo-s12-moco/`:

```text
cache_config.json        # Full preprocessing, checkpoint, code/runtime fingerprint
patches/<patch_id>.npz   # spatial_features [384,14,14], cls_embedding [384], metadata
feature_manifest.csv    # Patch/event/split IDs, image/label paths and SHA-256 hashes
cls_embeddings.npz      # CLS matrix [N,384], aligned patch/event/split arrays, metadata
cache_summary.json      # Selection, skipped patches, provenance and reuse counts
```

Per-patch NPZ metadata includes the checkpoint hash, complete preprocessing/runtime
contract, source image/sidecar hashes, native grid geometry, and payload hashes.
Archives use numeric arrays and JSON strings and load with `allow_pickle=False`.
The CLS embedding is separate from spatial features and is also exported as a
standalone aligned matrix for embedding-space analysis. Label content is never
fed into the encoder; refreshed labels can reuse unchanged image features.

Compatible cache files are reused without loading/running the encoder. Shape,
dtype, finiteness, payload corruption, source identity, and checkpoint/preprocessing/
runtime compatibility are checked. An incompatible root configuration requires
a new `--output` directory. `--rebuild-incompatible` explicitly replaces individual
stale/corrupt derived files after the audit has been refreshed. Writes are atomic
per file, so interrupted runs can reuse completed patches. A failed run does not
publish a new index of incomplete results.

Limited runs write `feature_manifest.limit-N.csv`, `cls_embeddings.limit-N.npz`,
and `cache_summary.limit-N.json`, preserving a full run's index. Cached patches
are shared between limited and full runs. `--batch-size` and `--threads` control
execution; CPU is the default. Runtime/device and implementation changes are
conservatively treated as cache incompatibilities.

**Training implication:** one canonical cached grid cannot reproduce the original
loader's random image rotations/flips before the encoder. Rotating a feature grid
is not equivalent to re-encoding a rotated image. Later decoder training must use
this canonical-input setup or explicitly cache additional input augmentations.
Feed cached grids directly to `CoreDecoder`; `SSL4EO.decode` instead expects the
full token sequence. Labels/validity masks stay on their native grid. This remains
a Sen1Floods11 adaptation of the WorldFloods experiment.

## Train the segmentation decoder

`scripts/train_segmentation_head.py` imports the **unchanged original
`CoreDecoder`** from `src/signal_task/models/model_components/decoder.py`, using
the SSL4EO wrapper's explicit settings, rather than the generic decoder defaults:

| Stage | Channels | Spatial grid |
| --- | --- | --- |
| Cached encoder features | 384 | 14 x 14 |
| CNN bridge | 1280 | 14 x 14 |
| Upsampling block 1 | 640 | 28 x 28 |
| Upsampling block 2 | 320 | 56 x 56 |
| Upsampling block 3 | 160 | 112 x 112 |
| Upsampling block 4 | 160 | 224 x 224 |
| Classification head | 2 | 224 x 224 |

Depths are `[2,2,8,2]`, dimensions `[160,320,640,1280]`, with BatchNorm,
ReLU, same padding, residual/channel-attention blocks, and zero dropout.
There are **41,268,162 trainable decoder parameters**; no architecture is reduced
for debugging. The script does not instantiate the encoder or update its weights.
The optional `encoder` uv extra supplies the runtime; PyYAML is an explicit
dependency because the original decoder's utility module imports it.

Commands to run when ready (implementation tests do not execute these runs):

```sh
uv run --locked --extra encoder scripts/train_segmentation_head.py --debug
uv run --locked --extra encoder scripts/train_segmentation_head.py --seed 0
```

`--debug` uses one epoch, the first two available patches in each of train and
validation, and at most one batch per split. It still runs the complete, large
decoder; it checks plumbing and produces an explicitly marked debug checkpoint,
not a meaningful trained model. The default output is
`models/segmentation/core_decoder_seed-0-debug/`; a full run uses
`models/segmentation/core_decoder_seed-0/`. Existing nonempty output directories
are rejected. Use `--output PATH` for another run.

Before optimization, the script verifies the complete index against the current
manifest/audit and prepares **only train and validation** samples. Image/label
hashes include validity sidecars; NPZ hashes, identities, shape, payload hashes
and encoder/preprocessing contracts are checked. Training samples retain the
native reference grid and audited support (`joint` by default). Invalid targets
are `-1`; reference labels are never interpolated. Only output logits are resized
with bilinear interpolation (`align_corners=False`) to each native mask grid.

Loss is unweighted two-class cross-entropy, summed over valid pixels and divided
by their count per batch. Epoch losses are weighted by valid-pixel counts. This
deliberately replaces SHRUG-FM's configured Dice loss. Water is class `1` and
non-water class `0`. F1 and IoU aggregate TP/FP/FN over all valid pixels in the
split, at a fixed 0.5 water threshold. They are **micro pixel metrics**, not mean
patch scores. When both reference and prediction contain no water, F1/IoU are
undefined (JSON `null`, blank CSV); a completely invalid patch is rejected.

Defaults retain the flood configuration's AdamW optimizer, learning rate
`5e-5`, weight decay `5e-6`, batch size 1, 50 epochs, 3 warmup epochs, and patience
5. Warmup starts at **0.1 times the learning rate**, then rises linearly to the
full rate and stays constant. This is an explicit numerical change from the legacy
trainer's `start_factor=lr/100`, which would start at `2.5e-11` with these defaults.
There are no random input augmentations, because the cache contains canonical
images. These loss, warmup, support and dataset choices mean this is an adaptation,
not an exact reproduction of the WorldFloods training experiment.

The best head maximizes **validation water IoU**; exact ties use lower validation
cross-entropy. Test and Bolivia are never loaded for fitting/checkpoint selection.
The current usable cohort is train 250, validation 86, test 88, Bolivia holdout 15.
Original main splits share events: this is not an event-disjoint evaluation.
Patch/event IDs and original split membership are retained.

Training writes `best_head.pt`, `history.csv`, `config.json`, and
`training_summary.json`. Checkpoints record the seed, selected patch/event IDs,
architecture, source hashes (LF/CRLF normalized), cache/checkpoint/preprocessing
contract, data hashes, versions, and validation metrics. Training is float32,
seeded, with deterministic algorithms and TF32 disabled. Exact bitwise equality
across devices/software versions is not promised. There is no resume mode yet.
Prepared features and compact native targets are kept in CPU memory, and the
decoder/optimizer require substantial additional memory. CPU is the script
default; on Anunna select the `encoder-cuda` uv extra and `--device cuda`, as
described below.

## Export segmentation predictions and water fractions

```sh
uv run --locked --extra encoder scripts/predict_segmentation.py --checkpoint models/segmentation/core_decoder_seed-0/best_head.pt
```

The default evaluates all usable cached patches, retaining their original split
and event IDs. To export held-out patches only, append
`--splits test bolivia_holdout`. Train predictions are in-sample; validation
predictions come from the split used to select the head. Keep these distinctions
when using fractions in later causal experiments. No threshold search is done.

Prediction loads the head with `weights_only=True` and strict state-dict loading,
and requires its recorded encoder/cache/index/audit/support contract to match.
The encoder is represented by the verified cached features and is not rerun.
The decoder runs in evaluation/inference mode. It resizes **logits first**, then
applies two-class softmax to obtain foreground probabilities. Water is predicted
where `p_water >= 0.5` (configurable as one fixed `--threshold` for the whole run).
Both reference and predicted water fractions divide water-pixel counts by exactly
the same native valid-pixel count. Signed error is prediction minus reference.

Default output is `results/segmentation_predictions/seed-0/`:

```text
probabilities/<patch_id>.tif  # Float32 water probability on the native label grid
masks/<patch_id>.tif          # Int16: 0 non-water, 1 water, -1 invalid
predictions.csv
prediction_config.json
prediction_summary.json      # Counts and aggregate pixel metrics per split
```

Rasters preserve the label CRS/affine transform. These are evaluation exports:
outside the reference valid support, probabilities are NoData `NaN` and masks
are `-1`. They do not supply predictions for unknown/invalid reference pixels.
TIFF tags and JSON preserve head/checkpoint/cache hashes, seed, threshold and IDs.
CSV paths are relative to the prediction output directory. The table contains:

```text
patch_id,event_id,original_split,model_seed,reference_fraction,predicted_fraction,
signed_fraction_error,absolute_fraction_error,n_valid_pixels,F1,IoU,
probability_path,mask_path,threshold
```

`--limit N` supports a small prediction plumbing check. Debug heads require
`--allow-debug-head` and use a separate default output folder. Neither a debug
run nor a complete training/prediction run is launched automatically.

Implementation checks (tiny synthetic fixtures, plus an original-decoder forward
shape check; no dataset training):

```sh
uv run --locked --extra encoder python -B -m unittest discover -s tests -v
```

## CPU and Anunna CUDA environments with uv

Choose **one** of these mutually exclusive extras. Both are resolved in `uv.lock`:

| Extra | PyTorch index | Currently locked torch / torchvision |
| --- | --- | --- |
| `encoder` | CPU | `2.15.0+cpu` / `0.29.2+cpu` on Windows/Linux |
| `encoder-cuda` | CUDA 12.8 (`cu128`) | `2.11.0+cu128` / `0.26.0+cu128` |

Each extra includes the same timm/PyYAML dependencies. CUDA wheels are for NVIDIA
GPUs on supported Linux/Windows systems, not AMD GPUs or macOS. The two indexes
provide different available versions, so the GPU runtime is deliberately recorded
separately from the existing CPU runtime. Training/prediction configurations also
record their actual runtime versions. This configuration does not verify Anunna's
driver or execute a GPU operation; check on an allocated GPU node before training.

Local CPU commands remain:

```sh
uv sync --locked --extra encoder
uv run --locked --extra encoder scripts/train_segmentation_head.py --debug
```

On Anunna, place the project on your assigned **Lustre** working storage and
create a fresh Linux `.venv` there. Transfer code/data/features/audit reports;
do not transfer the Windows `.venv` or `.uv-cache`. Build the environment in an
interactive compute allocation, not on the login node. An example NVIDIA A100
allocation for setup/checks is:

```sh
sinteractive -p gpu --gres=gpu:1 --constraint='nvidia&A100' -c 4 --mem=16G --time=00:30:00
module load 2025 uv
# Change into the transferred project directory on your assigned Lustre storage.
uv sync --locked --extra encoder-cuda
```

The Anunna uv module configures its cache on Lustre (`UV_CACHE_DIR` takes
precedence over the project's `.uv-cache` default). It also loads Python 3.13,
matching this project's `.python-version`. Verify the GPU runtime in that same
allocation:

```sh
nvidia-smi
uv run --locked --extra encoder-cuda python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable: check GPU allocation and NVIDIA driver'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda').sum().item())"
```

When ready, the short training check is:

```sh
uv run --locked --extra encoder-cuda scripts/train_segmentation_head.py --device cuda --debug
```

For full training, use an `sbatch` job and omit `--debug`. Prepare the environment
once beforehand; load `2025 uv` inside the job and use
`uv run --no-sync --extra encoder-cuda ... --device cuda` to use that prepared
environment without installing packages during the job. These are instructions,
not jobs launched by this repository update.

Always keep `--extra encoder-cuda` on CUDA commands: using `--extra encoder` would
switch `.venv` back to CPU packages. Combining them (including `--all-extras`) is
rejected. Base benchmark commands still need no extra. Existing CPU-generated
cached features can be consumed by the CUDA decoder trainer; generating a new
cache under CUDA changes its runtime/device fingerprint and needs a separate
cache output directory.

References: [uv's mutually exclusive PyTorch extras](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies),
[Anunna uv setup](https://wiki.anunna.wur.nl/Uv),
[Anunna GPU allocations](https://wiki.anunna.wur.nl/Compute_Nodes),
and [batch jobs](https://wiki.anunna.wur.nl/Batch_Jobs).

## Five-epoch Anunna training check

`scripts/hpc/train_segmentation_5epochs.sbatch` requests one NVIDIA A100, four CPU
cores, 16 GiB of system RAM and a two-hour time limit. It runs the normal full
train/validation cohort for **five epochs**, seed 0, batch size 1, with the existing
AdamW/loss/warmup settings. It does **not** pass `--debug`, which would reduce the
run to one batch/epoch. Test and Bolivia remain unused. This is a pipeline and
early-learning check, not a convergence or final model-quality assessment.

Transfer the current code, `pyproject.toml`, `uv.lock`, `.python-version`,
`data/sen1floods11/` and `results/segmentation_data/` to a fresh project directory
on your assigned Lustre storage. The feature index and audit must be copied as
well as the patch NPZs; retain source TIFFs and any GDAL sidecars. Encoder weights
are not needed to train from the cached features. A Git clone alone does not
include these ignored data/report artifacts or uncommitted code changes.

After preparing `.venv` once in an interactive compute session with
`module load 2025 uv` and `uv sync --locked --extra encoder-cuda`, return to the
login node and submit **from the transferred project root**:

```sh
mkdir -p logs
sbatch scripts/hpc/train_segmentation_5epochs.sbatch
squeue -u "$USER"
```

The submission script performs a small CUDA interpolation forward/backward
preflight, then starts training. It never installs packages inside the job.
Logs are `logs/segmentation-5ep-JOBID.out` and `.err`; results are
`models/segmentation/core_decoder_seed-0-5epochs-job-JOBID/`. Each submission gets
a fresh output directory. `history.csv` includes epoch duration and CUDA peak
allocated/reserved memory, as well as losses/F1/IoU; progress is logged every
batch by default. Read the logs and history while it runs:

```sh
tail -f logs/segmentation-5ep-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The original decoder's CUDA bilinear backward has no deterministic implementation
in the locked PyTorch 2.11 runtime. CUDA therefore requests deterministic
algorithms with `warn_only=True`; unsupported operations emit a warning rather
than aborting the run. CPU remains strict. Seeds, disabled TF32, deterministic
cuDNN selection and the actual policy/device are recorded, but GPU training is
not guaranteed bitwise repeatable. This preserves the original decoder and its
bilinear interpolation. See [PyTorch's determinism documentation](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html).

## Full Anunna training

After the five-epoch check, `scripts/hpc/train_segmentation_full.sbatch` runs up to
50 epochs with the same seed 0, batch size 1, AdamW learning rate `5e-5`, weight
decay `5e-6`, three-epoch warmup and patience 5. It uses all usable train and
validation patches and selects by validation water IoU (ties: lower CE), leaving
test and Bolivia out of fitting/selection. Review the five-epoch losses and
validation scores for finite, plausible learning before the longer run.

This is a **fresh training run**, not a continuation from the five-epoch head.
The current checkpoint saves model weights, not the optimizer/scheduler state
required for exact resumption. The encoder remains frozen and uses the existing
cached features. The architecture, loss and valid support are unchanged.

The new submission script is standalone; upload it into the existing Anunna
project's `scripts/hpc/` directory. No dataset transfer or environment rebuild is
needed if the successful check's data/code/environment remain unchanged. From
that project's root on the login node:

```sh
mkdir -p logs
sbatch scripts/hpc/train_segmentation_full.sbatch
```

It requests the same one A100, four CPU cores and 16 GiB RAM, with a **four-hour
wall-time cap**. This is a limit, not an estimated duration. Use measured epoch
durations from the five-epoch `history.csv` to size it, allowing for input
verification/checkpoint writes and later variability; override at submission
with `sbatch --time=HH:MM:SS scripts/hpc/train_segmentation_full.sbatch` if needed.
There is currently no automatic restart after a timeout.

Outputs are `models/segmentation/core_decoder_seed-0-full-job-JOBID/`; logs are
`logs/segmentation-full-JOBID.out` and `.err`. Earlier check runs are preserved.
Follow the actual job ID printed by `sbatch`:

```sh
tail -F logs/segmentation-full-JOBID.out logs/segmentation-full-JOBID.err
```

The trainer writes a flushed line for every train/validation batch by default:

```text
Epoch 1/50 starting | lr=5e-06
Epoch 1/50 | Train | batch 12/250 | loss=0.41234 | mean_loss=0.43827 | F1=0.72000 | IoU=0.56250 | elapsed=3.8s | ETA=75.4s
```

Values above illustrate the format, not measured performance. `loss` is the
current batch's valid-pixel mean cross-entropy; `mean_loss` and water F1/IoU
accumulate over the current phase. ETA applies to that train/validation phase.
`n/a` denotes undefined empty-water scores. `--log-every N` reduces output to
every N batches (first/last always logged); the frequency is saved in the run
configuration. Both Slurm entrypoints explicitly use `--log-every 1`. No new
package or environment rebuild is needed for this logging change.

An already running Python process keeps its loaded training code. Runs started
before this logging update still print every 50 batches and at epoch completion.
Keep the checkout fixed while they run; deploy logging updates between runs.

When training finishes successfully, use this run's `best_head.pt` for the planned
held-out test/Bolivia evaluation. Do not select epochs or tune the fixed 0.5
prediction threshold using those held-out results.

## Synchronize local development and Anunna with Git

Develop/review/test locally, commit and push an intended code version to
`https://github.com/MicheleMinervini06/Causal-SHRUG-FM.git`, then pull it on Anunna
between runs. Keep the upstream project as the separate `upstream` remote.
Only source/configuration/job scripts travel through Git; data, features, audit
reports, `.venv`, logs and outputs stay on Lustre and remain ignored by Git.

The initial Anunna folder was uploaded as an archive. To attach it to Git while
preserving all existing working files, first commit/push the current local code,
then run this **one-time** setup on the Anunna login node or Remote SSH terminal:

```sh
cd /lustre/nobackup/WUR/AIN/miner003/SHRUG-FM
if [ -e .git ]; then
    echo "Git is already configured here; inspect git status and git remote -v instead."
else
    (
        set -e
        git init -b main
        git remote add origin https://github.com/MicheleMinervini06/Causal-SHRUG-FM.git
        git remote add upstream https://github.com/vishalned/SHRUG-FM.git
        git fetch origin
        git reset --mixed origin/main
        git branch --set-upstream-to=origin/main main
        git config pull.ff only
        git status --short --branch
    )
fi
```

`reset --mixed` sets the initial branch/index to GitHub's history while retaining
the current working files. This intentionally exposes any uploaded/HPC code
differences rather than overwriting them. Review/reconcile the reported code
differences once before pulling or submitting a new experiment. If setup stops
after `git init`, inspect the existing configuration before resuming; do not
repeat the reset against an established repository with local commits.

After the initial code alignment, routine HPC updates are:

```sh
git status --short --branch
git pull --ff-only
git rev-parse HEAD
```

Update between active runs, and retain the commit ID with the experiment.
Prepare the environment in a compute allocation if `pyproject.toml` or `uv.lock`
changed. A public HTTPS repository can be fetched without a GitHub login; private
access needs GitHub authentication, such as a personal access token at the HTTPS
password prompt. Anunna/WUR login authentication is separate.

References: [Git mixed reset](https://git-scm.com/docs/git-reset)
and [GitHub authentication](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).

## Three controlled training configurations

`scripts/hpc/train_segmentation_trials.sbatch` submits three independent array
tasks, with **at most one GPU task running at a time** (`--array=0-2%1`):

| Array task | Profile | Physical batch size | Learning rate |
| --- | --- | ---: | ---: |
| 0 | `reference_b1_lr5e-5` | 1 | `5e-5` |
| 1 | `batch8_lr5e-5` | 8 | `5e-5` |
| 2 | `lower_lr_b1_lr2e-5` | 1 | `2e-5` |

Every trial initializes a new original CoreDecoder with **seed 0**, trains for
**50 epochs**, and uses **patience 50** so early stopping cannot truncate this
50-epoch diagnostic comparison. The best validation-IoU checkpoint is still saved
(ties: lower validation CE). Encoder/cache/preprocessing, architecture, masked
cross-entropy, weight decay `5e-6`, three-epoch warmup and fixed 0.5 threshold stay
the same. Test/Bolivia are not used. The older 21-epoch baseline is preserved.

The shared runner is `scripts/run_segmentation_trial.py`. Before the full fit it
verifies a real **training-only** batch at the requested physical batch size,
using the original decoder, native masks, backward pass and an AdamW step. This
also checks optimizer-state memory allocations, including for batch 8. That model
and optimizer are discarded, GPU cache is released, and the trainer resets seed 0
before fresh initialization. No preflight weight or normalization update carries
into training. A failed memory check fails that trial; it does not silently
reduce its batch size. No new dependencies are required.

After committing/pushing local changes and pulling them into the existing Anunna
project (including the latest per-batch logging), submit from its root:

```sh
mkdir -p logs
sbatch scripts/hpc/train_segmentation_trials.sbatch
squeue -u "$USER"
```

Each task requests one A100, four CPU cores, 16 GiB system RAM and a two-hour time
limit. Array scheduling does not guarantee execution order. Keep the Anunna code
checkout unchanged until **all three tasks** finish; the Linux CUDA environment
and data are reused without package installation inside the jobs.

For the array job ID `ARRAYID`, logs and outputs are:

```text
logs/segmentation-trials-ARRAYID_0.out   # reference; .err alongside
logs/segmentation-trials-ARRAYID_1.out   # batch 8
logs/segmentation-trials-ARRAYID_2.out   # lower LR
models/segmentation/config_comparison/array-ARRAYID/<profile>/
results/segmentation_config_comparison/array-ARRAYID/<profile>/trial.json
```

Model folders contain `config.json`, `history.csv`, `training_summary.json` and
`best_head.pt`. `trial.json` records the profile, code commit/dirty state/source
hashes, data provenance, real-batch memory/gradient check, actual training
arguments, status and final summary. Follow all three logs, waiting for tasks
that have not started yet:

```sh
tail -n 20 -F logs/segmentation-trials-ARRAYID_0.out logs/segmentation-trials-ARRAYID_0.err \
             logs/segmentation-trials-ARRAYID_1.out logs/segmentation-trials-ARRAYID_1.err \
             logs/segmentation-trials-ARRAYID_2.out logs/segmentation-trials-ARRAYID_2.err
```

Batch 1 gives 250 optimizer updates per epoch; batch 8 gives 32 (31 full batches
plus a final batch of 2), retaining all 250 training patches. Use the recorded
`train_batches` in `history.csv` to compare cumulative update counts as well as
epochs. Thus the batch-size trial tests the practical batching change, not solely
BatchNorm statistics. Inspect validation quality and collapses alongside
smoothness, then repeat promising configurations with additional seeds before
final held-out evaluation. The cache is not recomputed by these trials.

## Additional ensemble members: seeds 1 and 2

Reuse the **seed-0, batch-8, LR `5e-5`** checkpoint from the completed comparison:

```text
models/segmentation/config_comparison/array-44148765/batch8_lr5e-5/best_head.pt
```

Its best validation checkpoint is epoch 46 (water IoU `0.6216474791`). Keep this
checkpoint and its `config.json`; seed 0 is not retrained or overwritten.
`scripts/hpc/train_segmentation_ensemble.sbatch` trains **only seeds 1 and 2**
using the same batch-8 profile through `run_segmentation_trial.py --trial-id 1`.
This is a standard ensemble: all members use the same training patches, with
different random initialization and training shuffle; there is no bootstrap.
The original SHRUG-FM flood standard-ensemble configuration is in
`src/config/models.toml` (ten members there; three for this preliminary study).

Both new members use AdamW, LR `5e-5`, weight decay `5e-6`, three-epoch warmup,
unweighted masked cross-entropy, 50 epochs and patience 50. Each saves its own
best validation-IoU checkpoint (ties: lower CE); best epochs can differ between
seeds. The trainer, original CoreDecoder, frozen feature cache, native reference
support, and splits stay the same. Test and Bolivia remain unused. Each member
gets the existing disposable real-batch forward/backward/AdamW preflight, then
fresh initialization with its requested seed.

Before that preflight, `--reference-config` compares the existing seed-0
configuration with the requested training settings, architecture/decoder source
hashes, trainer/common source hashes, runtime package versions, data/cache
provenance, and selected train/validation patch IDs. A mismatch stops the job
instead of silently changing the experiment. The seed must differ from the
reference. The shared runner still defaults to seed 0 and retains the previous
comparison output paths when used without the new options.

After committing/pushing the local changes, run in the **Anunna project terminal**:

```sh
git pull --ff-only
mkdir -p logs
sbatch scripts/hpc/train_segmentation_ensemble.sbatch
squeue -r -u "$USER"
```

No dependency changes or new `uv sync` are needed if the working CUDA environment
is unchanged. Each task requests one A100, four CPU cores, 16 GiB system RAM,
and two hours. The default `--array=1-2%1` runs one member at a time. To allow
two independent one-GPU tasks concurrently, submit **instead** with:

```sh
sbatch --array=1-2%2 scripts/hpc/train_segmentation_ensemble.sbatch
```

For the new array job ID `ARRAYID`, the task ID is also the model seed:

```text
logs/segmentation-ensemble-ARRAYID_1.out     # seed 1; .err alongside
logs/segmentation-ensemble-ARRAYID_2.out     # seed 2; .err alongside
models/segmentation/ensemble/array-ARRAYID/batch8_lr5e-5_seed-1/
models/segmentation/ensemble/array-ARRAYID/batch8_lr5e-5_seed-2/
results/segmentation_ensemble/array-ARRAYID/batch8_lr5e-5_seed-1/trial.json
results/segmentation_ensemble/array-ARRAYID/batch8_lr5e-5_seed-2/trial.json
```

The model folders contain `best_head.pt`, `config.json`, `history.csv` and
`training_summary.json`. Metadata includes the reference-config hash and checked
fields, seed, code version, preflight and completion status. Follow both members:

```sh
tail -n 20 -F logs/segmentation-ensemble-ARRAYID_1.out logs/segmentation-ensemble-ARRAYID_1.err \
             logs/segmentation-ensemble-ARRAYID_2.out logs/segmentation-ensemble-ARRAYID_2.err
```

Keep the Anunna code and environment fixed until both tasks finish. Uncertainty
code can be developed locally meanwhile; pull it on Anunna after training, or
use a separate checkout. These jobs do not generate predictions or uncertainty
scores. Later, combine the two new best checkpoints with the existing seed-0
checkpoint, retaining every seed rather than selecting only the best member.

## Ensemble predictions and task uncertainty

`scripts/predict_segmentation_ensemble.py` loads compatible decoder checkpoints
on the same verified feature cache. The encoder is not loaded or run. It calls
each head once per patch in evaluation/inference mode, resizes **each member's
logits** bilinearly to the native label grid, then applies softmax. It averages
the resulting water probabilities, rather than logits or thresholded masks.
At least two members are required; the current experiment uses seeds 0, 1 and 2.
Duplicate seeds/checkpoint hashes, debug heads, incompatible caches/support,
and differing training protocols or train/validation cohorts are rejected.

For every pixel, binary probabilities are `[1 - p_water, p_water]`. Following
[SHRUG-FM equations 7-9 and Table 4](https://arxiv.org/html/2511.10370v2), the
script computes:

| Output | Pixel calculation | Patch calculation |
| --- | --- | --- |
| `average_entropy` | Mean of member entropies | Mean over valid pixels with ensemble water probability >= `0.05` |
| `mutual_information` | Entropy of ensemble mean minus mean member entropy | Mean over that same region |
| `predictive_entropy` | Entropy of ensemble mean | Mean over that same region, retained as a diagnostic |

Entropy uses natural logarithms (nats), float64 calculations and float32 exported
maps. Tiny negative numerical MI is clamped to zero. The ROI is computed from
the exported float32 ensemble probability. These are uncertainty signals, not
calibrated confidence intervals for the water fraction.

The original `EnsembleUQModel` implements the mean/MI mathematics; its
`UQModel.entropy()` means **predictive entropy**, and the original tester exports
that under `Entropy`. Average member entropy exists internally in MI but is not
exported separately. The small `ensemble_maps()` helper makes all three quantities
explicit and avoids the original API's repeated model forwards for different
metrics. Analytical tests compare its results with the original class. We reuse
our strict `load_head()`, native-grid resizing, `prediction_result()` and
`write_raster()` helpers. The legacy full-model loader/checkpoint format and
fixed 256-grid nearest resizing are not used.

The **segmentation threshold remains `0.5`** (`--threshold`) for the predicted
mask and water fraction. The separate **uncertainty ROI threshold is `0.05`**
(`--roi-threshold`). It selects which pixels contribute to the patch uncertainty
average; it does not change segmentation. The ROI intersects the audited valid
support, without using reference water labels to select pixels. Reference and
predicted fractions use the same native valid support as before. All probability
and uncertainty maps cover this full valid support, with NaN outside it; masks
use -1 outside support. An empty ROI gives blank CSV scores, `roi_empty=True`,
and `n_roi_pixels=0`. This explicitly differs from original `metric_scaled()`,
which returns zero for empty regions and has no separate validity mask.

Each run writes `prediction_config.json` (member checkpoint paths/hashes, seeds,
epochs, validation metrics, matching training protocol, data/cache identity,
thresholds, runtime and code hashes), `prediction_summary.json`, and
`predictions.csv`. CSV rows preserve patch/event/original split IDs, reference
and ensemble-predicted water fractions, signed/absolute error, valid count,
F1/IoU and raster paths. They also include semicolon-separated `model_seeds`,
`n_members`, `roi_threshold`, `n_roi_pixels`, `roi_fraction`, `roi_empty`, and the
three patch scores. Outputs are relative to the run directory:

```text
probabilities/PATCH_ID.tif
masks/PATCH_ID.tif
uncertainty/average_entropy/PATCH_ID.tif
uncertainty/mutual_information/PATCH_ID.tif
uncertainty/predictive_entropy/PATCH_ID.tif
```

Validation is the default split. Training predictions are in-sample; validation
selected the heads; main splits share events. The script fits no decision tree
or calibrator, and does not automatically open test/Bolivia samples. Limit a
local implementation check with `--limit 2` and `--device cpu`, for example:

```sh
uv run --locked --extra encoder python -B scripts/predict_segmentation_ensemble.py \
  --checkpoints PATH_TO_SEED0/best_head.pt PATH_TO_SEED1/best_head.pt PATH_TO_SEED2/best_head.pt \
  --output results/segmentation_uncertainty/local-check --limit 2
```

For Anunna, **wait until both additional training tasks finish** before pulling
these code changes into their checkout. After committing/pushing locally and
pulling on Anunna, submit the validation inference job from the project root,
replacing `TRAINING_ARRAY_ID` with the ID of the seeds-1-and-2 training array:

```sh
git pull --ff-only
mkdir -p logs
sbatch scripts/hpc/predict_segmentation_ensemble.sbatch TRAINING_ARRAY_ID
```

This reuses seed 0 from comparison array `44148765` and seeds 1/2 from the supplied
training array, checks for completed training outputs, and requests one A100
with four CPU cores, 16 GiB RAM and a one-hour limit. It uses the existing uv
CUDA environment without installing dependencies. Results go to
`results/segmentation_uncertainty/array-TRAINING_ARRAY_ID/job-JOBID/`. Follow logs:

```sh
tail -n 30 -F logs/segmentation-uncertainty-JOBID.out logs/segmentation-uncertainty-JOBID.err
```

For an explicitly chosen later evaluation, pass split names after the training
array ID, e.g. `test bolivia_holdout`. Do not tune settings on those holdouts.

## Spatial input uncertainty: pretraining geographic density

`scripts/compute_spatial_density.py` implements the spatial component of
[SHRUG-FM's input signal](https://arxiv.org/html/2511.10370v2): fit a KDE using
SSL4EO-S12 pretraining locations, then evaluate it at each downstream patch
center. Lower density means sparser nearby geographic coverage during
pretraining. It is a feature for later reliability analysis, not a calibrated
probability of prediction error. Environmental percentile features are deferred.

The script uses the official location CSV linked by the
[SSL4EO-S12 repository](https://github.com/zhu-xlab/SSL4EO-S12):
**251,079 distinct location IDs and coordinate pairs**, headerless
`id,longitude,latitude`, with blank lines. Each location has equal weight;
seasonal images are not repeated. The source is pinned by SHA-256:

```text
e9e5e5d3d7ba977efbc19785cb5362aee412d9c66a3864ef0fae524c5d1a94f5
```

This is the released dataset's location catalog; the exact per-checkpoint
sampling trace is unavailable. Metadata records the intended encoder hash.

The downloaded file is about 8 MB, stored in
`data/ssl4eo-s12/pretraining_coordinates.csv`. Existing files are verified;
changed/corrupt inputs are rejected. The file is excluded from Git, so download
or copy it separately on Anunna. No new dependencies are needed: the calculation
uses NumPy and Rasterio, without PyTorch, scikit-learn or model inference.

The metric is an equal-weight KDE:

\[
\widehat\rho(s)=\frac{1}{N Z_h}\sum_{i=1}^{N}
\exp\left(-\frac{d(s,s_i)^2}{2h^2}\right),
\]

where `d` is great-circle distance on a sphere of radius **6371 km**, `h` is the
Gaussian bandwidth in km, and `Z_h` normalizes each kernel over the spherical
surface. The initial bandwidth is **100 km**, a declared physical scale chosen
without using downstream outcomes, labels or prediction errors. It is exposed
as `--bandwidth-km`; the authors' original kernel, bandwidth and metric settings
are unverified, so this reproduces their KDE feature definition rather than
claiming numerically identical scores. A future sensitivity comparison can use
predeclared 50/100/200 km scales without selecting a bandwidth on test errors.

Normalization uses the integral
`Z_h = 2*pi*R^2 * integral_0^pi exp(-theta^2/(2*(h/R)^2))*sin(theta) dtheta`.
After scaling `u=theta/(h/R)`, 64-point Gauss-Legendre quadrature integrates to
`min(pi*R/h,12)`; the omitted Gaussian tail is below `exp(-72)`. Every reference
location contributes to the KDE; the fitted kernel itself is not truncated.
This avoids both Euclidean distances on raw longitude/latitude and planar-only
density normalization. Computation is float64 and processes one query at a time
to bound memory. Tests independently check surface normalization, known kernel
values, the dateline and polar geometry.

Patch centers use `image.transform @ (width/2,height/2)` in the source CRS,
reprojected to WGS84 (`EPSG:4326`). Our encoder resizes the **whole source image**,
so that image's footprint is the model footprint. The legacy WorldFloods
coordinate exporter is not imported: its dataset setup is unrelated and its
`x1 + x2/2` expression is not a midpoint. New centers are verified against the
audit's image dimensions, CRS and affine transform. The audit's complete usable
cohort and manifest identity are reused through `cache_encoder_features` helpers;
pixel values, reference water classes and cached encoder arrays are not read.

By default the script computes features for all usable patches, preserving
`patch_id`, `event_id` and `original_split` (currently 439 patches; 7 excluded by
the audit). Computing locations for test/holdout patches does not fit or evaluate
anything on their labels. Use `--splits validation` for validation-only output,
and `--limit 2` for a small implementation check. Outputs require a new/empty
directory:

```text
spatial_density.csv
density_config.json
density_summary.json
```

The CSV contains:

```text
patch_id,event_id,original_split,longitude,latitude,pretraining_density,pretraining_log_density
```

`pretraining_density` is normalized density in **km^-2**, not a local image count
or a probability. Its natural logarithm is retained as a numerical diagnostic
when a very small bandwidth makes raw density underflow to zero. Configuration
records source hash/URL, weighting, kernel, bandwidth, units, Earth radius,
normalization, patch-center convention, input audit hashes, versions and code
hashes. The summary records completeness/limits, split counts, excluded patches,
raw-density underflows and within-event density ranges/standard deviations.
No automatic feature selection, uncertainty threshold, or classifier is fitted.

For a local implementation check in the existing CPU encoder environment:

```sh
uv run --locked --extra encoder python -B scripts/compute_spatial_density.py --download-only
uv run --locked --extra encoder python -B scripts/compute_spatial_density.py \
  --splits validation --limit 2 --output results/input_uncertainty/local-density-check
```

On **Anunna**, after committing/pushing locally and pulling the code into the
project, submit the CPU job:

```sh
git pull --ff-only
module load 2025 uv
mkdir -p logs
sbatch scripts/hpc/compute_spatial_density.sbatch
```

The Slurm job uses the existing project environment without synchronizing or
removing the installed CUDA extra. It requests **one CPU, 2 GiB RAM, 15 minutes**
in `main --constraint=gen2` (Skylake); no GPU is requested. It downloads/verifies
the small public coordinate CSV on the allocated node before computing, with no
package installation inside the job. Existing verified coordinates are reused.
Override the bandwidth, for example,
with `sbatch scripts/hpc/compute_spatial_density.sbatch 50`, using a separate
output directory automatically. Results and logs are:

```text
results/input_uncertainty/spatial_density/h-100km-job-JOBID/
logs/spatial-density-JOBID.out
logs/spatial-density-JOBID.err
```

The architecture constraint is required by the current environment:
`.venv/bin/python` resolves to the `skylake_avx512` Python 3.13.1 used on the
A100 nodes, while `login201` and CPU `gen3` nodes use Zen3. Invoking that
interpreter there fails with `Illegal instruction` / exit 132, even for
`python --version`. Loading another Python module does not change the existing
virtual environment's interpreter link. The
[Anunna hardware table](https://wiki.anunna.wur.nl/Compute_Hardware_Overview)
identifies CPU `gen2` and A100 nodes as Skylake. The batch loads modules on its
allocated node and logs both Python paths; keep the working training `.venv`.
Use the local `--download-only` examples above on a compatible machine, rather
than running this environment on the Zen3 login node.

Keep these raw features and their settings fixed when comparing them with U_T.
Inspect both within-event variation and between-event differences before
interpreting geographic density as a predictor of segmentation or fraction error.

## Embedding uncertainty U_E

The pipeline uses the **original** [SSL4EO-S12 Hugging Face mirror](https://huggingface.co/datasets/wangyi111/SSL4EO-S12),
linked by the [dataset authors](https://github.com/zhu-xlab/SSL4EO-S12), pinned to
revision `3f5ddad68ba2ea29d019b0cef6cf292ff8af0d62`. It does not substitute the
updated v1.1 dataset or embeddings from another model. The released SHRUG-FM
extractor selects CLS tokens but defaults to MAE; we reuse our verified **MoCo**
encoder and existing preprocessing helper instead. No decoder is trained here.

Three scripts perform the work:

1. `download_ssl4eo_reference.py`: keep the **first 5,000 contiguous location
   blocks in archive order**, then stop downloading. Keep **all seasons in each
   selected block**. This is a preliminary prefix sample, not a random global
   reference. Preserve each
   image's 13 original uint16 TIFFs losslessly inside one NPZ. Save IDs, source
   revision, selection order, season counts, per-image hashes and a completion summary.
2. `cache_pretraining_embeddings.py`: nearest-resize each native band array to
   264x264, as in the released SSL4EO wrapper (**264 is the output size, not a
   requirement on native TIFF dimensions; CRS is not needed for array encoding**), then use exactly the downstream
   fixed flood mean/std and full-patch nearest-exact resize to 224x224. Run the
   frozen encoder in eval/inference mode, float32, no augmentation/TF32/autocast.
   Cache **384-D CLS vectors** with source/checkpoint/preprocessing/runtime identity.
   Compatible individual caches are reused after interruption.
3. `compute_embedding_uncertainty.py`: fit CPU **FAISS k-means, k=64**, seed 0,
   100 iterations, 5 restarts, using all reference vectors (disable FAISS's
   internal subsampling). Recompute cluster counts and mean **Euclidean** radii.
   Apply SHRUG-FM equations 4–5 to our existing downstream CLS cache. Preserve
   `patch_id,event_id,original_split`; no labels or segmentation errors enter fitting.

For cluster j, `d_j = ||embedding-centroid_j||_2 / mean_radius_j`. Assign the
nearest **raw Euclidean** centroid c, report `normalized_distance=d_c`, and
`NCDD=sum_j(d_j)-64*d_c`. Higher normalized distance means more unfamiliar;
lower NCDD means less separation. NCDD can be negative because assignment occurs
before normalization; do not clip it. FAISS squared-L2 values are not radii.
No extra embedding L2 normalization is applied. The authors' reference sample,
seed and complete extraction/fitting settings are unavailable: these are a
documented implementation of the same scores, not identical published values.

**Download cost:** the full original L1C archive is **557.48 GB**, but this script
now stops at the first band header of location 5,001 instead of reading it all.
Expect roughly 20,000 seasonal images and a transfer/storage cost on the order of
10 GB; actual counts/bytes are recorded, not assumed. Small read-ahead buffers
can include bytes beyond the stopping header. Transient connections retry from
their current byte offset; restarting the process rescans the prefix. Entire
parts, if consumed, are verified against LFS SHA-256. Early stopping cannot verify
the final partial part or the gzip end CRC: its downloaded prefix hash is recorded
locally, and per-image hashes detect later changes, not source authenticity.
The supplied 24-hour limit is an allocation ceiling, not a measured runtime.

**Sampling limitation:** the first locations may have biased geographic and
embedding coverage. Check their coverage and cluster occupancy before interpreting
scores as global pretraining unfamiliarity. Noncontiguous location blocks within
the scanned prefix or incomplete images cause an error. Completion means the
requested subset is complete, not the full archive. Previous globally sampled
runs use different directories and are not silently reused.

There is no need to download 2,000 locations separately before 5,000. The summary
records location order, and the embedding archive records location IDs: a nested
2,000-location comparison can later reuse those vectors without another download
or encoder pass. Such a comparison checks reference-size sensitivity; it does not
remove the prefix sample's geographic bias.

**Data limitation:** the original band geotransforms have small offsets (visible
in the checked example). Per-band array resizing preserves the original loader's
behavior; it does not repair geographic alignment. This is recorded in metadata.
Native band dimensions may vary, and some images may have no CRS. The CPU audit
records dimension counts, missing CRS and non-264 B2 examples in
`reference_audit.json`; it still rejects wrong band sets, non-uint16/multiband
TIFFs, changed source hashes and images with no jointly valid support. These
checks validate array loading, not geographic alignment.
Pretraining patches also cover a different footprint from Sen1Floods11 patches.
Keep these limitations when interpreting embedding distance as unfamiliarity.

Small local checks, without a full reference download:

```sh
uv sync --locked --extra encoder
uv run --locked --extra encoder python -B -m unittest tests.test_embedding_uncertainty
uv run --locked --extra encoder python -u -B scripts/download_ssl4eo_reference.py \
  --debug --output data/ssl4eo-s12/debug-reference
uv run --locked --extra encoder python -u -B scripts/cache_pretraining_embeddings.py \
  --reference data/ssl4eo-s12/debug-reference \
  --output data/ssl4eo-s12/debug-reference/embeddings --batch-size 1
```

Debug downloads stop after **one first complete image**, explicitly mark the
reference incomplete, and are rejected by cluster fitting. This verifies loading
and the forward pass; it is not a representative reference sample.

On **Anunna**, synchronize code through Git first. This change adds `faiss-cpu`
to `pyproject.toml`/`uv.lock`; update the prepared environment in a compatible
compute allocation before submitting the jobs:

```sh
git pull --ff-only
module load 2025 uv slurm
srun --partition=main --constraint=gen2 --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash -lc 'module load 2025 uv; uv sync --locked --extra encoder-cuda'
mkdir -p logs
DOWNLOAD_JOB=$(sbatch --parsable scripts/hpc/download_ssl4eo_reference.sbatch)
sbatch --dependency="afterok:${DOWNLOAD_JOB}" scripts/hpc/compute_embedding_uncertainty.sbatch
```

The first job uses CPU gen2 to match the existing interpreter; the second uses
an A100 for reference encoding and then CPU FAISS within the same allocation.
After downloading, the first job audits all reference images on CPU before
releasing the dependent GPU job. If the download was already completed with an
older script, run that audit separately before resubmitting the second job:

```sh
srun --partition=main --constraint=gen2 --cpus-per-task=2 --mem=4G --time=00:30:00 \
  bash -lc 'module load 2025 uv; uv run --no-sync python -u -B scripts/cache_pretraining_embeddings.py --audit-only'
```

Wait for `REFERENCE AUDIT PASSED`, then submit
`sbatch scripts/hpc/compute_embedding_uncertainty.sbatch 5000`. The current
array reader writes `embeddings-native-v2` so older, incompatible partial caches
are preserved. The raw reference images do not need another download.
Both accept `[locations]` (default 5,000); pass identical arguments to both.
Archive-prefix selection has no random sampling seed.
The clustering seed remains 0 unless changed in the Python scoring CLI.
They install no dependencies inside the batch jobs. Outputs are:

```text
data/ssl4eo-s12/reference-prefix-n5000/{reference_manifest.csv,download_config.json,download_summary.json,images/}
data/ssl4eo-s12/reference-prefix-n5000/reference_audit.json
data/ssl4eo-s12/reference-prefix-n5000/embeddings-native-v2/{pretraining_cls.npz,embedding_summary.json,patches/}
results/embedding_uncertainty/prefix-n5000-job-JOBID/{clusters.npz,embedding_uncertainty.csv,embedding_config.json,embedding_summary.json}
logs/embedding-download-JOBID.{out,err}
logs/embedding-uncertainty-JOBID.{out,err}
```

`clusters.npz` preserves the centroids, mean radii, counts and full protocol so
future matching embeddings can be scored with the same reference. Scoring rejects
encoder/preprocessing mismatches, incomplete/debug references, empty or zero-radius
clusters, duplicate IDs and nonfinite vectors. At least `39*k` reference images
are required. Fit/scoring output directories must be new/empty. The default
downstream archive includes all usable splits; computing features does not
evaluate test/Bolivia labels. Reliability analysis should remain validation-only.

## Segmentation-failure gate and risk–coverage evaluation

`scripts/segmentation_gate.py` has two separate commands: `train` and `evaluate`.
It uses exported patch tables only: no encoder/decoder forward pass, raster
loading, or causal-effect estimation. The released repository has uncertainty
and pixel-calibration code, but no reusable selective-tree trainer was found.

The first baseline fixes the three features selected for floods in SHRUG-FM:
**average entropy at ROI 0.05, mutual information at ROI 0.05, spatial density**.
The target is **failure = 1 when patch F1 < 0.60**; F1 exactly 0.60 is a success.
The segmentation threshold remains 0.5. The tree uses Gini, maximum depth 3,
minimum leaf size 5, no class weights and seed 0. There is no feature selection
or hyperparameter search. Optional U_E scores are joined/exported and compared
as baselines, but do not enter the fitted tree.

Important implementation choices:

- Fit only on **validation**, with five `StratifiedGroupKFold` folds grouped by
  `event_id` for development out-of-fold (OOF) scores. Then fit the final tree on
  all labelled validation rows. This differs from the paper's train+validation
  classifier fit and standard flood CV. The decoder already used validation for
  checkpoint selection; these OOF diagnostics do not make the full pipeline
  independent of that selection. The final frozen gate is evaluated separately.
- Keep undefined F1 when both foreground masks are empty. Such rows receive a
  final-model score but are excluded from fitting and risk denominators. Do not
  turn undefined F1 into a failure. On our validation export: 86 rows, 78 defined
  F1 (51 failures), and 8 unlabelled rows.
- Keep U_T missing on empty predicted ROIs. Scikit-learn's native NaN routing
  scores these cases instead of dropping them or substituting zero. This retains
  the labelled F1=0, empty-ROI failure. There are 2 missing-U_T rows, so all-label
  risk curves use 78 rows while common-score baseline comparisons use 77.
- Convert density internally by a **fixed factor 1e8**. Our km^-2 densities are
  below CART's absolute feature tolerance at their original scale; leaving them
  unscaled could silently suppress density splits. This fixed linear unit
  conversion is applied identically at fit and inference and recorded in the
  model configuration. Original densities are preserved in exported tables.
- Reference fractions, predicted fractions, F1, IDs and event groups never enter
  the tree feature matrix. They remain in the output for provenance/evaluation.

Training input directories must contain complete exports:

```text
predictions-dir: predictions.csv, prediction_config.json, prediction_summary.json
density-dir: spatial_density.csv, density_config.json, density_summary.json
embedding-dir (optional): embedding_uncertainty.csv, embedding_config.json, embedding_summary.json
```

Joins verify unique patch IDs, event/split identities, ensemble members, thresholds,
feature-cache identity, reference support, dataset/audit hashes and density/KDE
settings. Evaluation must use the same ensemble and feature protocols as fitting.
Training rejects a prediction export containing non-validation rows. Evaluation
accepts one split at a time: `test` (default) or explicit `bolivia_holdout`.

For local synthetic implementation checks only:

```sh
uv sync --locked --extra encoder
uv run --locked --extra encoder python -B -m unittest tests.test_segmentation_gate
```

On Anunna, synchronize source and update the prepared environment in a compatible
compute allocation once, because this change adds scikit-learn:

```sh
git pull --ff-only
module load 2025 uv slurm
srun --partition=main --constraint=gen2 --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash -lc 'module load 2025 uv; uv sync --locked --extra encoder-cuda'
mkdir -p logs
```

Set `VALIDATION_DIR` to the existing validation ensemble output directory (the
directory containing the three prediction files); replace the placeholder:

```sh
VALIDATION_DIR="results/segmentation_uncertainty/array-44157060/job-VALIDATION_JOB_ID"
DENSITY_DIR="results/input_uncertainty/spatial_density/h-100km-job-44166500"
EMBEDDING_DIR="results/embedding_uncertainty/prefix-n5000-job-44198535"
GATE_DIR="models/reliability_gate/three_features_seed0"

sbatch scripts/hpc/segmentation_gate.sbatch train \
  --predictions-dir "$VALIDATION_DIR" --density-dir "$DENSITY_DIR" \
  --embedding-dir "$EMBEDDING_DIR" --output "$GATE_DIR"
```

The wrapper requests CPU gen2, 2 CPUs, 4 GiB and 15 minutes; it uses the existing
environment without installation. Direct local commands use
`uv run --locked --extra encoder python -B scripts/segmentation_gate.py train ...`.
Follow `logs/segmentation-gate-JOBID.out` and `.err` with `tail -n 50 -F`.

Training writes:

```text
gate.pkl                   frozen scikit-learn tree
gate_config.json           features, scale, fitting IDs/events/folds, source hashes, versions, model hash
gate_scores.csv            all validation patches; in-sample final scores and separate OOF scores
tree_rules.txt, tree_structure.json, tree.png
evaluation_summary.json    development OOF risk–coverage metrics
risk_coverage.csv, risk_coverage.png, risk_coverage.pdf
```

Use **OOF scores** for validation diagnostics. `gate_score` in this training CSV
comes from the final fitted tree and is marked `validation_in_sample`; it must
not be presented as independent validation performance. Undefined-F1 rows have
no OOF score but still receive a final-tree score. The exported leaf frequencies
are ranking scores, not calibrated probabilities of failure. Use only the gate
artifact produced by this script; evaluation checks its checksum and exact
scikit-learn version. Native missing-value branches are explicit in
`tree_structure.json`, including splits not fully described by text rules.

Before final evaluation, freeze the protocol and obtain ensemble predictions for
the untouched **test** split using the existing ensemble prediction script. Do
not evaluate the gate on decoder-training predictions or refit it on test F1.
Once a test prediction export is available, set its directory and run:

```sh
TEST_DIR="results/segmentation_uncertainty/array-44157060/job-TEST_JOB_ID"
sbatch scripts/hpc/segmentation_gate.sbatch evaluate \
  --predictions-dir "$TEST_DIR" --density-dir "$DENSITY_DIR" \
  --embedding-dir "$EMBEDDING_DIR" --gate-dir "$GATE_DIR" \
  --split test --output results/reliability_gate/test_seed0
```

Evaluation exports `gate_scores.csv` with `score_kind=held_out`, an evaluation
summary, and risk–coverage CSV/PNG/PDF. It never calls `fit`. Two cohorts are
reported: **all labelled** patches (gate vs random) and **common complete scores**
(gate vs individual scores and random on the identical rows). Coverage is
conditional on defined F1, with input/unlabelled/missing-U_T counts and included
patch IDs recorded. It is not automatically coverage of the entire input set.

Retain lowest uncertainty/failure scores; single-score directions are fixed:
low entropy, low MI, high spatial density, low normalized distance and high NCDD.
Equal tree scores are averaged **analytically over uniform within-tie ordering**,
avoiding arbitrary ID ordering (the paper approximates this with permutations).
Curves at partial leaves therefore represent randomized retention within ties.
The random baseline is the exact expected failure rate on the same cohort.
The oracle curve uses failure labels only as a diagnostic bound. AURC is mean
risk over k/n for k=1..n; risk at 50%, 75%, 90% reports actual integer coverage.
No confidence intervals or deployment threshold are inferred from those curves.

Original train/validation/test splits share events: the summary reports event
overlap explicitly. This held-out-patch evaluation does not establish generalization
to new events. Bolivia remains a separately requested evaluation. No causal
simulation, ATE estimator, or causal gain is implemented in this step.

References: [SHRUG-FM methodology and appendix](https://arxiv.org/html/2511.10370v2),
[scikit-learn missing-value routing](https://scikit-learn.org/stable/modules/tree.html#missing-values-support),
[CART feature tolerance](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_partitioner.pxd).

## E02: real confounder errors in a semi-synthetic causal benchmark

`experiments/semisynthetic_confounder.py` extends E01 using the frozen gate's
**test** export. It reuses E01's OLS treatment-coefficient estimator, including
an intercept. No encoder, decoder or reliability gate is trained or retuned.
Existing dependencies suffice; `pyproject.toml` and `uv.lock` are unchanged.

The cohort is every usable audited test patch: currently **88**, including the
six both-empty masks with undefined F1. F1 and missing U_T never filter this
causal cohort. The loader verifies complete test IDs, reference valid support,
the inspection checksum, and the reported frozen gate identity; it rejects
validation scores and overlap with gate-fitting patches. It consumes exported
scores without loading the model binary. Preserve the input export files.

Reference fractions C, predicted fractions C_hat and gate scores remain fixed.
Fit mean and population SD (`ddof=0`) once on all test reference fractions;
apply that same center/scale to both C and C_hat, including every subset.
At each repetition generate independent treatment uniforms and N(0,1) outcome
noise, then simulate:

```text
T ~ Bernoulli(sigmoid(a0 + a1*C_standardized))
Y = T + C_standardized + epsilon
```

Defaults: `a0=0`, `a1 in {0,1}`, true effect 1, **5,000 repetitions**, seed 42.
The same uniforms/noise pair both assignment scenarios. Within a scenario every
comparison uses subsets of the same full T/Y arrays. All three estimators
(unadjusted, oracle, proxy) run on the full cohort and on gate/random selections
at 75% and 50% retention: **88, 66 and 44 observations**. Exact sizes use
`floor(coverage*n)`. Gate selection keeps lowest scores with independent uniform
tie-breaking. Random rejection samples without replacement. Separate seeded
streams make the selections independent of T/Y; 50% subsets are nested within
75% subsets for each selection method. Ties and random subsets are redrawn each
repetition, while the gate's scores and ranking rule remain fixed.

Local implementation checks:

```sh
uv run --locked --extra encoder python -B -m unittest tests.test_semisynthetic_confounder tests.test_synthetic_confounder
```

After committing/pushing the reviewed code and pulling it on Anunna, submit
from `/lustre/nobackup/WUR/AIN/miner003/SHRUG-FM`:

```sh
mkdir -p logs
# Path used by the test jobs submitted after prediction job 44220183:
TEST_GATE_DIR="results/reliability_gate/test_seed0-pred-job-44220183"

# Small implementation check, with its own output directory:
CHECK_JOB=$(sbatch --parsable scripts/hpc/semisynthetic_confounder.sbatch \
  --gate-evaluation-dir "$TEST_GATE_DIR" --repeats 5 \
  --output results/causal_e02/check_seed42)
echo "$CHECK_JOB"
tail -n 50 -F "logs/semisynthetic-confounder-${CHECK_JOB}.out" \
  "logs/semisynthetic-confounder-${CHECK_JOB}.err"
```

Ctrl+C stops log monitoring without cancelling the job. Review successful
completion/output before launching the full simulation in a separate directory:

```sh
E02_JOB=$(sbatch --parsable scripts/hpc/semisynthetic_confounder.sbatch \
  --gate-evaluation-dir "$TEST_GATE_DIR" --repeats 5000 --seed 42 \
  --output results/causal_e02/test_seed42)
echo "$E02_JOB"
tail -n 50 -F "logs/semisynthetic-confounder-${E02_JOB}.out" \
  "logs/semisynthetic-confounder-${E02_JOB}.err"
# Alternatively check state/completion:
squeue -j "$E02_JOB"
sacct -j "$E02_JOB" --format=JobID,State,Elapsed,ExitCode
```

The wrapper requests CPU `main --constraint=gen2`, one CPU, 4 GiB, 30 minutes;
it reuses the prepared environment with `uv run --no-sync`. No GPU is needed.
The default `--gate-config` points to the original fitted gate configuration;
`--inspection` points to `results/segmentation_data/patches.csv`. For an exported
local copy, pass these paths explicitly and use
`uv run --locked --extra encoder python -B experiments/semisynthetic_confounder.py ...`.

Outputs in a new/empty directory:

```text
config.json          protocol, scaling, seed, source hashes and code identity
patches.csv          fixed aligned C/C_hat/scores, standardized values, IDs/support
selections.csv       retained IDs, event counts, C distribution and proxy MAE per draw
estimates.csv        three coefficients/statuses per scenario and selection draw;
                     treatment counts and known-propensity/observed-C-range diagnostics
summary.csv          signed/absolute bias, RMSE, SD, MCSE(bias), success/failure counts
run_summary.json     completion marker, failed-fit reasons and output checksums
```

Bias is mean(estimate - 1); absolute bias is abs(mean(estimate - 1)), not mean
absolute error. Compare RMSE as well: rejection can reduce bias while increasing
variance. Non-estimable regressions stay in the output with a status and are
counted; summary moments are conditional on successful fits and cannot be
interpreted without those failure counts. The propensity diagnostic flags
values outside [0.05,0.95] without clipping probabilities or trimming patches.

This experiment is conditional on 88 fixed patches and fitted measurements;
additional repetitions improve Monte Carlo precision, not geographic sample
size. The reference-based generator is a synthetic causal mechanism, not an
observed real treatment/outcome. The constant individual effect remains 1 in
every selected subset. Preserve gate/coverage choices after inspecting results;
shared events across main splits and possible changes in geographic/confounder
composition remain limitations. No Bolivia experiment or interval-coverage
analysis is included in this baseline.

## E03: fraction-error gates, in two separate phases

E03 changes the gate target while preserving the frozen segmentation ensemble,
preprocessing, three reliability inputs (average entropy, mutual information,
spatial density), depth 3, minimum leaf size 5 and seed 0. Phase 2 reuses the
E02 simulation and its archived control results.

**Phase 1, implemented:** `scripts/train_fraction_error_gates.py` trains:

- Primary: `DecisionTreeRegressor(criterion="squared_error")` with target
  `abs(predicted_fraction - reference_fraction)`. Its score predicts mean
  absolute fraction error in units of fraction, not percentage points.
- Secondary: the same classifier architecture as E02 with target
  `abs(predicted_fraction - reference_fraction) > 0.02`. This is a strict
  threshold of **two percentage points**, fixed before E03 evaluation.

Both use exactly the original F1 gate's **78 validation fitting IDs** and its
five event-grouped folds. The other eight undefined-F1 validation patches are
preserved and receive final-model scores, but no OOF score; excluding them here
keeps the fitting cohort identical across targets. Missing U_T stays NaN for
native tree routing. The classifier has 32 positive labels in the real fitting
cohort. No new fold search, feature selection or threshold tuning is performed.

The script reuses `segmentation_gate.py` for strict prediction/density joins,
protocol and support checks, feature construction, score export and binary
risk calculations. It verifies the original F1 configuration, input checksums,
scikit-learn version, fitting IDs and event separation before fitting. F1
comparison scores are the saved OOF scores; that gate is not retrained.

Review compares all three gates on the same 78 OOF patches using retained
fraction MAE, fraction RMSE and the fraction-error failure rate, with exact
expected ordering within ties. At 75/50% retention this development cohort has
58/39 patches (actual coverages are saved). The full cohort, random expectation
and an oracle error ranking are references. RMSE is the square root of expected
retained squared error, not the expectation of a subset's RMSE. Validation was
previously used for decoder checkpoint selection, so these remain development
diagnostics.

Local implementation checks use small synthetic fits:

```sh
uv run --locked --extra encoder python -B -m unittest tests.test_fraction_error_gates tests.test_segmentation_gate tests.test_semisynthetic_confounder
```

After commit/push and pulling the code on Anunna, run **phase 1 only** from the
project root:

```sh
mkdir -p logs
E03_JOB=$(sbatch --parsable scripts/hpc/train_fraction_error_gates.sbatch \
  --predictions-dir results/segmentation_uncertainty/array-44157060/job-44161703 \
  --density-dir results/input_uncertainty/spatial_density/h-100km-job-44166500 \
  --f1-gate-dir models/reliability_gate/three_features_seed0 \
  --output models/reliability_gate/e03_fraction_error_seed0)
echo "$E03_JOB"
tail -n 50 -F "logs/fraction-error-gates-${E03_JOB}.out" \
  "logs/fraction-error-gates-${E03_JOB}.err"
```

The CPU wrapper requests `main --constraint=gen2`, two CPUs, 4 GiB and 15 minutes
and uses the prepared environment with `uv run --no-sync`. Ctrl+C stops log
monitoring without cancelling the job. Check status with `squeue -j "$E03_JOB"`
or `sacct -j "$E03_JOB" --format=JobID,State,Elapsed,ExitCode`.

Outputs in the new/empty E03 directory:

```text
training_summary.json                  completion, cohort, targets and OOF metrics
validation_scores.csv                  aligned rows, errors and three OOF scores
validation_error_coverage.csv/.png/.pdf retained fraction-error diagnostics
regression/, classification/           each contains gate.pkl, gate_config.json,
                                       gate_scores.csv, tree.png, tree_rules.txt,
                                       tree_structure.json (including NaN routes)
```

Gate configurations use schema 2 and explicitly identify the new targets;
the existing F1-only evaluation/causal loaders intentionally reject them.
Final-model validation scores are in-sample; only OOF scores are used for the
development comparisons. Review `training_summary.json`, `validation_scores.csv`,
`validation_error_coverage.csv/.png` and both tree images before phase 2.

### E03 validation with embedding uncertainty

Run `scripts/validate_embedding_gate.py` to add normalized distance and raw
NCDD to the primary regressor. It checks the original three-feature OOF scores
patch by patch, then compares both regressors on the same event-grouped folds
at 75% and 50% retention (58 and 39 patches). The thresholds, tree settings,
density scaling and treatment of ties are the same as in phase 1.

From the project root on Anunna, after syncing the code:

```bash
mkdir -p logs
JOB=$(sbatch --parsable scripts/hpc/train_fraction_error_gates.sbatch --with-ue)
tail -n 50 -F "logs/fraction-error-gates-${JOB}.out" "logs/fraction-error-gates-${JOB}.err"
```

The default paths point to the archived validation predictions, density and
embedding exports, F1 gate and original E03 validation run. The new model and
its ordered feature list, hashes and folds are saved in
`models/reliability_gate/e03_regression_with_ue_seed0/`. Review
`training_summary.json` and `validation_error_coverage.png`. Only OOF scores
are used for this comparison. The frozen tree is fitted to all 78 eligible
validation patches for later use.

### E03 phase 2 on Anunna

For the **five- versus three-feature held-out comparison**, use the existing
batch script with `--with-ue` after pulling the updated code on Anunna:

```bash
mkdir -p logs
JOB=$(sbatch --parsable scripts/hpc/fraction_gate_causal.sbatch --with-ue)
echo "$JOB"
tail -n 50 -F "logs/fraction-gate-causal-${JOB}.out" "logs/fraction-gate-causal-${JOB}.err"
```

This mode loads the frozen Anunna validation model and the original E03 gates.
Normalized distance and raw NCDD come from the verified test export, aligned by patch ID.
Outputs go to `results/causal_e03/ue_test_seed42/`. In `paired_comparisons.csv`
the four rows give five minus three features for the proxy-adjusted estimator
at both coverages in each treatment scenario. Read `selection_summary.csv`
for fraction MAE/RMSE and `summary.csv` for causal bias/RMSE. The full run
checks all 54 archived E03 and 30 E02 control groups. The comparison on the
previously examined test cohort is exploratory.

`experiments.fraction_gate_causal` applies both reviewed Anunna models to the
three features in the held-out F1 export. It verifies the recorded hashes for
the two model files, the input table and the original E02 cohort. All **88**
test patches participate, including empty foreground masks and missing
entropy/MI. The frozen trees route missing features as in validation.

The full run uses the E02 seed (42), reference scaling, treatment/outcome
draws, random subsets and within-tie draws for 5,000 paired repetitions.
Regression (primary), classification, F1 and random selection each retain
66/44 patches at 75/50%. Full coverage uses all 88 patches once per scenario.
Each subset gets unadjusted, oracle and proxy OLS estimates under randomized
and confounded treatment. Completion of the full run requires agreement with
all 30 archived E02 control summaries.

After pushing the code and pulling it on Anunna, submit from the project root
in the prepared environment. The wrapper uses one CPU on `main --constraint=gen2`.

```bash
mkdir -p logs
JOB=$(sbatch --parsable scripts/hpc/fraction_gate_causal.sbatch)
echo "$JOB"
tail -n 50 -F "logs/fraction-gate-causal-${JOB}.out" "logs/fraction-gate-causal-${JOB}.err"
```

For a five-repetition check, use the same wrapper with
`--repeats 5 --output results/causal_e03/check_seed42`. Every run needs a new
output directory. Use `squeue -j "$JOB"` while queued and
`sacct -j "$JOB" --format=JobID,State,Elapsed,ExitCode` after completion.

Results go to `results/causal_e03/test_seed42/` by default.

| File | Interpretation |
|---|---|
| `run_summary.json`, `config.json` | Completion, attempts/failures, checksums, frozen models and simulation settings |
| `summary.csv` | Causal signed bias, absolute bias, RMSE and MCSE for every estimator, scenario and selection |
| `paired_comparisons.csv` | Proxy-adjusted contrasts at matched coverage on repetitions where both fits succeeded. Signed bias and RMSE differences have 95% Monte Carlo intervals. Absolute bias difference is descriptive |
| `selection_summary.csv` | Fraction MAE, pooled fraction RMSE, signed error and reference mean in fraction units, plus the share of patches above the two-point error threshold |
| `event_composition.csv`, `selection_overlap.csv`, `treatment_overlap.csv` | Retention by event, shared patches between methods, and treated/control counts and propensity support |
| `patches.csv`, `selections.csv`, `estimates.csv` | Patch-level scores, retained IDs and event counts, and individual simulated estimates and statuses |

In the exports, `gate` denotes the original F1 gate. A negative
`delta_absolute_bias` or `delta_rmse` favors the method in `selection` over
`comparator`. The Monte Carlo intervals describe simulation precision for
these fixed patches, predictions and gates.

E03 is exploratory because these test patches informed the follow-up question.
Better fraction MAE need not imply less residual confounding; preserve both
questions and the negative E02 baseline. Independent confirmation needs data
kept out of development.

Reference: [scikit-learn regression means and missing-value support](https://scikit-learn.org/stable/modules/tree.html).

## E05 phase 2 — propagate confounder uncertainty

Use the frozen Phase 1 posterior for both Bayesian corrections. The experiment replays the
5,000 E02 repetitions on all 88 patches, with randomized and confounded assignment. Each
Bayesian draw shares a calibration row across all patches. Treatment and outcome jointly
inform the latent fractions and the treatment coefficient. Oracle and raw-proxy OLS use
model-based t intervals. The two Bayesian methods use equal-tailed posterior intervals.

After syncing the source via Git, copy the reviewed Phase 1 draws from the **local project
terminal**. The filename identifies the reviewed fit, and the runner checks its full SHA-256.

```powershell
scp results/causal_e05/phase1_seed42/posterior.npz miner003@login.anunna.wur.nl:/lustre/nobackup/WUR/AIN/miner003/SHRUG-FM/results/e05-posterior-390bc867.npz
```

On **Anunna**, from the project root, submit the array and a dependent aggregation job.

```bash
git pull --ff-only
mkdir -p logs
script=scripts/hpc/fraction_calibration_causal.sbatch
job=$(sbatch --parsable --array=0-99%4 "$script" --posterior results/e05-posterior-390bc867.npz)
root="results/causal_e05/phase2-array-$job"
report=$(sbatch --parsable --dependency="afterok:$job" "$script" --collect "$root" --output "$root/combined")
echo "simulation=$job report=$report"
squeue -r -j "$job"
tail -n 30 -F "logs/e05-causal-${job}_0.out" "logs/e05-causal-${job}_0.err"
```

Each task writes 50 complete repetitions atomically. Collection requires exactly one
estimate per method, scenario and repetition, with a common protocol. The main report
is `combined/paired_comparisons.csv` (with scores minus without scores) together with
`combined/summary.csv` for absolute coverage, interval width, bias and RMSE.
`combined/estimates.csv` records individual intervals and importance-sampling diagnostics.
