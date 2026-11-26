# E09 — Predicting residual confounding after proxy adjustment

**Average entropy did not improve the whole-cohort bias warning.** It tied the proxy-only diagnostic on floods and slightly worsened prediction on burns and landslides under confounded assignment. This tests a different use of entropy from E06, which ranked individual patches for verification.

## Results under confounded assignment

Each of 5,000 paired assignments yields one signed conditional bias and one prediction per diagnostic for the entire cohort. MAE measures error in predicting that signed bias, in outcome units. True treatment effect is 1. Negative paired MAE change favors adding entropy. Intervals reflect simulation precision on these fixed cohorts.

| Test cohort | Mean true conditional bias | MAE predicting zero | Proxy-only MAE | Proxy + entropy MAE | Paired change (95% MC interval) |
|---|---:|---:|---:|---:|---:|
| Floods, 88 patches | +0.03923 | 0.04547 | 0.04509 | 0.04509 | −0.000002 (−0.000194, +0.000189) |
| Burns, 263 patches | +0.07509 | 0.07533 | 0.06167 | 0.06194 | +0.000266 (+0.000166, +0.000367) |
| Landslides, 68 patches | +0.48542 | 0.48566 | 0.19219 | 0.19540 | +0.003206 (+0.002480, +0.003931) |

**Decision threshold.** On floods, absolute bias exceeded 0.05 in 2,089/5,000 assignments. The entropy model detected 849 of those (40.6% sensitivity) and only 849/2,139 alerts were correct (39.7% precision, versus 41.8% prevalence). On burns its precision was 82.2% at 82.0% prevalence. On landslides 4,978/5,000 exceeded the threshold: none of the 22 below-threshold assignments was correctly cleared by the entropy diagnostic. Apparent high precision on that cohort largely reflects the base rate.

**Other scores.** With confounding, negative NCDD improved burn MAE by 0.00111 versus proxy alone (paired MC interval −0.00158 to −0.00063); it worsened MAE by 0.00039 on floods and 0.00677 on landslides. None of the five individual scores improved signed-bias MAE consistently. The normalized-distance diagnostic extrapolated badly: maximum validation/test scores were 2.34/168.18 on burns and 1.62/92.92 on landslides. The registered linear diagnostic reached MAE 0.421 and 5.140 respectively. These exports assume comparable split-specific score scales and within-split indices. The extreme ranges warrant checking that assumption before using this score in another model.

**Randomized control and scaling.** Mean conditional bias was close to zero on every cohort (+0.00087, +0.00086, +0.00161). Predicting zero had lower MAE than any fitted warning in all three. The primary warning uses validation SD to convert fraction error into outcome units. Supplying the simulator's test SD in the scale sensitivity gave proxy-only/entropy MAEs of 0.04063/0.04071 on floods, 0.06351/0.06391 on burns and 0.31895/0.31581 on landslides. The small landslide entropy contrast changes direction under this privileged conversion.

**Interpretation.** Validation labels and the observed proxy/treatment carry useful information about bias on burns and especially landslides. The primary entropy comparison adds no stable warning. The reference is the exact bias over outcome noise under the specified outcome law, conditional on a particular treatment assignment. It does not include random outcome noise in an individual estimate. Different outcome slopes, nonlinear confounding or geographic transfer can change the relationship. The small calibration samples (86, 54 and 41) also limit what the diagnostic can learn.

## Reproduce and verify

Run `uv run --no-sync python -m experiments.residual_bias_warning --output results/causal_e09/replay` in the main checkout with the reviewed E05/E07 inputs and the original landslide embedding CSV at `data/landslides/landslides_ncdd_k64.csv`. The [protocol](e09_protocol.md) fixes the primary comparison, geographic exclusions and warning threshold. The reviewed output is in `results/causal_e09/seed42/`. The summary gives all 42 comparisons, the NPZ files retain paired predictions by repetition, and the partition table and config record geographic membership, inputs and scales.

The vectorized diagnostic agreed with independently fitted ridge and least-squares models. Ten archived full-cohort E02 flood estimates reproduced, including their decomposition into true effect, conditional bias and projected outcome noise. The coefficient identity was also checked at the first and last assignment in each cohort/scenario. In burns one eight-patch test group has no nearest validation patches; in landslides one four-patch group has none. Those groups use all validation rows for fitting. The geographic blocks on landslides use the supplied projected boxes; their CRS and event independence are unverified.
