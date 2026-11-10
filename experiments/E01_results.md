# E01: synthetic confounder measurement-error baseline

Status: baseline run completed on 10 November 2026.

Configuration: 2,000 observations per population, 200 repetitions, starting seed
42 (repetition seeds 42 through 241), true treatment effect 1. Both assignment
settings (`a=1,0`) and all measurement-error levels (`sigma=0,0.5,1`) were run.
Latent draws are paired across assignment settings and error levels; fresh draws
are generated for each repetition. Execution used uv with Python 3.13.16.

Reproduce from the repository root:

```sh
uv run --locked experiments/synthetic_confounder.py --csv results/e01/estimates.csv --summary-csv results/e01/summary.csv
```

## Initial diagnostics

For the first population (seed 42), the confounded setting had 1,008 treated and
992 control observations; the randomized setting had 1,019 treated and 981
controls. All input arrays had matching lengths, finite values, and valid binary
treatments. At sigma zero, both the proxy measurements and adjusted coefficients
matched their oracle counterparts exactly.

The sample mean/SD of C was -0.0184/1.0053, outcome noise 0.0057/1.0005, and
measurement noise -0.0075/0.9917. These moments are descriptive checks; they do not
prove independence or require sample moments to equal their population values.

## Observed results

Bias is the mean signed error relative to 1. RMSE includes both bias and sampling
variation. MCSE of mean bias is the sample SD of estimates divided by sqrt(200).
The table omits duplicate oracle/unadjusted rows across sigma, and duplicate
sigma-zero proxy rows, which equal the oracle rows.

| a | sigma | Estimator | Mean effect | Bias | RMSE | MCSE of bias |
|---|---|---|---|---|---|---|
| 1 | all | No adjustment | 1.8328 | 0.8328 | 0.8348 | 0.0041 |
| 1 | all | Oracle | 1.0033 | 0.0033 | 0.0461 | 0.0033 |
| 1 | 0.5 | Proxy | 1.1947 | 0.1947 | 0.2010 | 0.0036 |
| 1 | 1 | Proxy | 1.4557 | 0.4557 | 0.4588 | 0.0038 |
| 0 | all | No adjustment | 1.0083 | 0.0083 | 0.0617 | 0.0043 |
| 0 | all | Oracle | 1.0026 | 0.0026 | 0.0414 | 0.0029 |
| 0 | 0.5 | Proxy | 1.0036 | 0.0036 | 0.0451 | 0.0032 |
| 0 | 1 | Proxy | 1.0052 | 0.0052 | 0.0507 | 0.0036 |

The oracle and randomized-control biases are within about two Monte Carlo SEs
of zero in this run. Under confounding, proxy bias and RMSE increase with
measurement-error magnitude. Under randomized treatment, the proxy remains
approximately unbiased, while its RMSE increases with measurement error.

Full local artifacts are [per-repetition estimates](../results/e01/estimates.csv),
[all 18 summaries](../results/e01/summary.csv), and the
[captured console output](../results/e01/run.txt). The results directory is
ignored by Git; the command above regenerates the CSVs.

## Scope

This validates the baseline simulation under a linear outcome model and
independent Gaussian measurement error. It does not evaluate FM predictions,
reliability scores, selection, or measurement-error correction. Treatment-effect
interval coverage and width remain part of later evaluation. The MCSE reported
here describes simulation precision, not an individual effect estimate's SE.

The generator still fixes tau=1 and the scenario grid; exports retain a wide
per-repetition estimate table. Fully configurable tau/grids and a long-form
estimate export are interface differences from the complete E01 specification,
not changes to the baseline experiment assessed here.
