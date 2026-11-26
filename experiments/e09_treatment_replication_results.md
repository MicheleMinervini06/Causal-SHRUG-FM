# E09 · Treatment update across SHRUG-FM cohorts

The replication uses the frozen E07 positive-beta calibrations and 5,000 paired assignments per scenario on each fixed test cohort. The main comparison is how accurately the two diagnostics predict the signed bias of the same raw-proxy OLS estimate. Probability forecasts target absolute bias above 0.05. A constant probability is estimated on each validation set using 5,000 independently seeded assignments. It is shared across methods and across the two confounded diagnostics within a cohort. See the [replication protocol](e09_treatment_replication_protocol.md).

## Comparison

| Cohort | Diagnostic assignment rule and scale | Proxy MAE | Score MAE | Score minus proxy MAE (95% MC interval) | Proxy Brier | Score Brier | Validation constant Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Burn Scars | Known confounded | 0.02930 | 0.02750 | −0.00181 (−0.00224, −0.00137) | 0.16675 | 0.19241 | 0.23760 |
| Burn Scars | Validation fit | 0.03043 | 0.03539 | +0.00495 (+0.00468, +0.00523) | 0.23927 | 0.31591 | 0.23760 |
| Burn Scars | Randomized | 0.04159 | 0.04274 | +0.00114 (+0.00095, +0.00134) | 0.22448 | 0.23047 | 0.23408 |
| Landslides | Known confounded | 0.37173 | 0.33990 | −0.03182 (−0.03267, −0.03097) | 0.05328 | 0.03591 | 0.00440 |
| Landslides | Validation fit | 0.12478 | 0.12562 | +0.00084 (−0.00083, +0.00251) | 0.00462 | 0.00462 | 0.00440 |
| Landslides | Randomized | 0.15216 | 0.14744 | −0.00472 (−0.00531, −0.00413) | 0.23474 | 0.21922 | 0.16993 |

Adding scores lowered MAE in all three cohorts when the confounded rule and scale were known, including the original flood result (0.04698 to 0.03221). With validation-fitted rules and scales, score gains remain on floods, reverse on burns, and are unresolved on landslides. The fixed validation forecast also provides an alternative to the retrospective flood benchmark. The flood Brier score with scores under the known rule was 0.25188, close to the validation constant's 0.25145 (paired difference +0.00043, 95% MC interval −0.00451 to +0.00536). The intervals describe Monte Carlo variability conditional on the fixed cohorts and calibration banks.

## Burn Scars

With the known rule, adding scores shifted the average warning from +0.08730 to +0.07129, closer to the true +0.07509 on average. Its advantage for signed bias prediction was about 6%. The probability forecasts with and without scores both outperformed the fixed validation constant, which assigned 0.52 probability to the event. The event occurred in 4,100 of 5,000 test assignments. Adding scores worsened Brier by 0.02567 (95% MC interval +0.02279 to +0.02855) relative to proxy alone. Under the validation-fitted rule and scale, the diagnostic with scores also had higher MAE and Brier than the diagnostic with proxy alone.

## Landslides

Under the known rule, adding scores lowered MAE from 0.37173 to 0.33990. The true mean bias was +0.48542. The mean warnings were +0.11540 and +0.14757, leaving considerable underprediction. The validation-fitted rule and scale yielded much lower absolute MAE, with the score comparison close to zero. This sensitivity changes both the treatment model and the units used in the warning. The test reference SD is 0.23231 and the validation SD is 0.09135. The sensitivity cannot attribute the difference to either change alone.

The bias event occurred in 4,978 of 5,000 confounded assignments. The fixed validation probability of 0.9908 yielded a Brier score of 0.00440. Both known-rule diagnostics and both validation-fitted diagnostics scored above this constant. The primary signed-bias MAE is more informative about performance on this cohort than the fixed-threshold event.

## Reproduction

Run from the main repository with the reviewed E07 validation calibrations and evaluation exports.

```powershell
uv run --no-sync python -m experiments.residual_bias_treatment_update --dataset burns --output results/causal_e09/replay_burns
uv run --no-sync python -m experiments.residual_bias_treatment_update --dataset landslides --output results/causal_e09/replay_landslides
```

The reviewed runs use 16 quadrature nodes and 512 joint bias draws per assignment. They are in `results/causal_e09/treatment_update_burns/` and `results/causal_e09/treatment_update_landslides/`. Each result directory contains per-assignment bias and probability forecasts, paired summaries, the fixed validation probability and source hashes.

On the first 100 assignments, increasing quadrature order from 16 to 32 changed the known-rule paired MAE contrast by 0.00012 on burns and 0.00017 on landslides. The updated runner also reproduced all original forecast and score columns exactly on the 100-assignment flood check.

All fixed input hashes and archived E09 reference biases were verified for both replications. The minimum calibration-row effective sample size across all confounded runs was 834 for burns and 408 for landslides. Reducing the risk calculation from 512 to 256 draws changed each Brier score by less than 0.00052. Full local CPU evaluation, including grid construction, took 392 seconds for burns and 121 seconds for landslides.
