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
