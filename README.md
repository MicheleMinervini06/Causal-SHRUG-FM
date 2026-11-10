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

Run the standalone benchmark in that environment; no third-party packages,
datasets, weights, or GPU are currently required:

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
