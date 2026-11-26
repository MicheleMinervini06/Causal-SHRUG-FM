# E09 · How much does the bias threshold matter?

The [protocol](e09_threshold_protocol.md) fixes five common tolerances and two cutoffs derived from the validation bias distribution. Each probability predicts whether the absolute conditional bias of the raw-proxy treatment coefficient exceeds its cutoff. The same simulated assignments and joint bias draws are used across cutoffs. The constant baseline is the validation event frequency at each cutoff and assignment strength. Brier rewards probability accuracy, while AUC measures the ordering of assignments.

## Floods and landslides

For floods, across the four known-rule cutoffs with at least 5% of assignments on each side, adding scores improved AUC and Brier relative to proxy alone. At validation q75 (0.07972), AUC rose from 0.651 to 0.717 and Brier fell from 0.25230 to 0.12747. The validation constant scored 0.11560. At 0.05 and validation q50, the score forecast and the constant had indistinguishable Brier within Monte Carlo uncertainty. At 0.10 only 168 assignments exceeded the cutoff, and neither 0.20 nor 0.50 had any observed exceedances.

For landslides, raising the cutoff made discrimination assessable on substantially more assignments. At 0.50, the split is 2,438 above and 2,562 at or below. With the known rule and scale, adding scores improved AUC from 0.514 to 0.578, while Brier remained above the validation constant (0.47963 versus 0.27674). At validation q50 (0.58945), 1,175 assignments exceeded the cutoff, AUC improved from 0.505 to 0.572, and Brier with scores was 0.23364 versus 0.23491 for proxy alone and 0.25000 for the constant. At 0.50 with the rule and scale fitted on validation, adding scores reduced Brier from 0.26763 to 0.25519, also better than the 0.27674 constant. The threshold limitation and probability accuracy therefore matter separately. The comparison between assignment models changes both the rule and the outcome scale.

## Burn Scars

The known-rule AUC with scores remained between 0.364 and 0.387 at all four cutoffs with at least 5% of cases on both sides (0.05, 0.10, validation q50 and validation q75). At validation q75 (0.08539), 1,837 assignments exceeded the cutoff. With scores, the AUC was 0.387 and Brier was 0.27150, versus 0.29389 for proxy alone and 0.24620 for the constant. At 0.10 and validation q75, scores improved the Brier relative to proxy alone and the constant still had a lower Brier. At 0.05 and validation q50, adding scores worsened the Brier relative to proxy alone and both predictors beat the constant. The ordering problem persists across cutoffs with both classes well represented.

All archived 0.05 risks and bias estimates replayed, and risks decreased monotonically with threshold for every assignment and method. The 256- and 512-draw Brier estimates differed by at most 0.00070. The pattern is conditional on these cohorts, assignment and outcome laws, and calibration models. The grid offers evidence about sensitivity to the tolerance and the current diagnostic's discrimination.

## Reproduction

Run from the main repository with the verified E05/E07 calibration archives and validation and test exports.

```powershell
uv run --no-sync python -m experiments.residual_bias_treatment_update --dataset floods --threshold-sensitivity
uv run --no-sync python -m experiments.residual_bias_treatment_update --dataset burns --threshold-sensitivity
uv run --no-sync python -m experiments.residual_bias_treatment_update --dataset landslides --threshold-sensitivity
```

The results are in the `results/causal_e09/threshold_sensitivity_*` directories. Each `threshold_summary.csv` contains all cutoffs and scenarios, including both class counts, AUC, Brier and paired Brier intervals. Each `threshold_risks.npz` contains the per-assignment bias and risk at every cutoff. `config.json` records the grid, frozen input hashes, seeds and timing.
