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
  empty rather than being converted to zero.
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
locked in `uv.lock`. The project selects the official CPU PyTorch index for this
local check. Install/run that extra explicitly:

```sh
uv sync --extra encoder --locked
uv run --extra encoder scripts/get_ssl4eo_encoder.py --check
uv run --extra encoder scripts/get_ssl4eo_encoder.py --image data/sen1floods11/images/Bolivia_103757_S2Hand.tif
```

`--check` uses a seeded synthetic input; `--image` implies a check on a real raster.
Use `--extra encoder` for encoder commands so uv retains the optional runtime.
CPU is the default; `--threads` controls CPU threads. `--device cuda` requires a
CUDA-enabled PyTorch environment, which the configured CPU extra does not provide.

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
