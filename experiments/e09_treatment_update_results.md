# E09 · Predicting residual bias after updating with treatment

On the 88 Sen1Floods11 test patches, the frozen E05 calibration with three reliability scores gave more accurate bias predictions under confounded assignment. With the known assignment rule and scale, the scores reduced signed-bias prediction MAE by 31%. This measures accuracy of the warning about raw proxy-adjusted OLS. Both methods warn about the same effect estimator.

| Treatment model used by warning | MAE, proxy only | MAE, with scores | Paired MAE change (95% MC interval) | Brier, proxy only | Brier, with scores |
|---|---:|---:|---:|---:|---:|
| Known confounded rule and test scale | 0.04698 | 0.03221 | −0.01477 (−0.01551, −0.01403) | 0.31819 | 0.25188 |
| Logistic rule and scale from validation | 0.08529 | 0.04886 | −0.03643 (−0.03715, −0.03570) | 0.40607 | 0.30182 |
| Randomized rule and test scale | 0.04822 | 0.05150 | +0.00327 (+0.00257, +0.00397) | 0.25031 | 0.25923 |

Each row summarizes 5,000 assignments on the same fixed test cohort. Monte Carlo intervals describe variability across those simulated assignments. Under confounding, the true conditional bias averaged +0.03923. The mean warning was +0.07961 with proxy alone and +0.05076 with scores when the assignment rule was known. Using the validation rule and scale, both warnings were farther from the reference mean (+0.12362 and +0.08312).

The Brier score evaluates the probability that absolute bias exceeds 0.05. The known-rule Brier improvement was −0.06631 (95% MC interval −0.07248 to −0.06014). A constant forecast set retrospectively to the observed event prevalence (41.78%) would score 0.24324 on the confounded assignments. So the scores improved the warning relative to its proxy-only counterpart while the probability warning still scored worse than this descriptive constant benchmark. This benchmark uses test labels to estimate the prevalence.

The three scores enter the frozen E05 measurement model together. The primary benchmark uses the known assignment rule and test-reference center and SD. The validation sensitivity changes both the treatment model and the scale used to convert fraction error to outcome units, so its results describe their combined impact. The predictive fit uses the full E05 validation split, which shares events with test. The original E09 ridge excludes each test patch's event from its validation fit. Comparisons with the original ridge are historical.

## Verification and reproduction

The [protocol](e09_treatment_update_protocol.md) specifies the comparison. All reference biases agree with archived E09 to within 1e-12. The synthetic example verifies treatment-induced shifts of opposite sign, and direct enumeration checks how the whole assignment reweights shared calibration draws. Under randomization, all 8,192 calibration rows retain equal weights and the conditional node means equal the prior means.

The first 100 assignments used 2,048 risk draws to check quadrature. Moving from 16 to 32 nodes changed the confounded paired MAE differences by at most 0.00018. The full run used 16 nodes and 512 risk draws. Using only the first 256 risk draws changed absolute Brier scores by at most 0.00070. The minimum effective number of shared calibration draws was about 3,742 in the primary comparison and 789 under the validation sensitivity. The measured CPU computation took 132 seconds including grid preparation and probability forecasts.

Run `uv run --no-sync python -m experiments.residual_bias_treatment_update --output results/causal_e09/treatment_update_replay` from the main repository with the original E05 test scores, validation scores and frozen posterior. The reviewed output is in `results/causal_e09/treatment_update/`, with paired per-assignment warnings and risks, a summary and configuration hashes. The 100-assignment checks are in `treatment_check16` and `treatment_check32` under the same E09 results directory.
