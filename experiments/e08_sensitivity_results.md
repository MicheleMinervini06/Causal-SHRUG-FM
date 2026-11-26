# E08 on Burn Scars · sensitivity results

**Finding.** Across the same 263 patches, adding pretraining density consistently helped the E06 hidden-overlap diagnostic and consistently hurt the steep-assignment diagnostic across neighborhood sizes. The pooled improvement under E06 depended heavily on one geographic fold. The negative result on steep assignment does not rule out a relationship between density and measurement error. Even a deliberately strong injected relationship was difficult for this diagnostic to exploit.

## Where real crossings come from

Weak reference overlap means that one treatment has probability below 0.10 given reference burned fraction. “Hidden” means the out-of-fold apparent overlap is at least 0.10. The reference smoother repeats the same out-of-fold neighbor averaging using the measured fraction to find neighbors. Negative Brier change favors adding density.

| Assignment | Neighbors | Reference weak | Apparently adequate | Hidden using proxy | Hidden using reference smoother | Brier change |
|---|---:|---:|---:|---:|---:|---:|
| E06 | 5 | 13 | 252 | 2 | 0 | −0.00420 |
| E06 | 15 | 13 | 262 | 12 | 10 | −0.00232 |
| E06 | 30 | 13 | 263 | 13 | 13 | −0.00243 |
| Steep | 5 | 43 | 248 | 29 | 0 | +0.00677 |
| Steep | 15 | 43 | 258 | 38 | 1 | +0.00672 |
| Steep | 30 | 43 | 258 | 38 | 34 | +0.00672 |

**Can density predict fraction error?** Real mean absolute fraction error was 3.19 percentage points in the lowest density quartile and 2.56 elsewhere. When predicting that error in held-out regions, adding log density to the proxy spline increased prediction MAE from 2.33 to 2.50 points. Four of five folds worsened. The one improving fold contained eight patches.

**Does error conceal weak overlap?** Substituting the proxy directly into the assignment rule concealed 2/13 weak cases for E06 and 8/43 for steep assignment. At 15 neighbors, both E06 direct crossings were among the 12 hidden cases. Ten of the 12 also appeared hidden with the reference smoother, including both direct crossings. For steep assignment, all eight direct crossings were among the 38 hidden cases. Switching the smoother's input from reference to proxy introduced 37 hidden cases and removed none. Thus proxy error and averaging can interact. The magnitude of a patch's error alone cannot determine whether it crosses the threshold.

**Geographic concentration.** At 15 neighbors the five folds contained 0, 0, 3, 9, 0 hidden cases under E06 and 4, 1, 4, 29, 0 under steep assignment. In the 96-patch fold, the Brier changes were −0.00709 and +0.01760, respectively. These are geographic approximations; independence of fire events is unverified.

## Could the diagnostic detect an injected signal?

Over 100 paired draws, the two synthetic proxies had exactly matched overall absolute error in each draw (mean MAE 9.23 percentage points). Low-density versus remaining-patch MAE averaged 14.95 versus 7.31 points in the positive control and 9.34 versus 9.19 in the negative. The manipulation therefore produced the intended difference in absolute error. It did not explicitly inject a threshold crossing or a density effect on treatment.

| Assignment | Neighbors | Density improved positive / 100 | Density improved negative / 100 | Positive beat negative / 100 |
|---|---:|---:|---:|---:|
| E06 | 5 | 45 | 19 | 65 |
| E06 | 15 | 0 | 1 | 28 |
| E06 | 30 | 0 | 1 | 30 |
| Steep | 5 | 1 | 8 | 11 |
| Steep | 15 | 0 | 2 | 0 |
| Steep | 30 | 0 | 2 | 0 |

None met the prespecified 80/100 criterion on both improvement in the positive arm and paired wins. These frequencies describe the sensitivity of the complete procedure to this error mechanism on this cohort. They also reflect how often the injected errors can change hidden status, the number of rare cases, and the diagnostic's ability to generalize geographically. They do not identify an intrinsic limit of the score or of every possible classifier. The controls use larger errors than the observed proxy on average. The real results support limited incremental predictive value for the chosen models. Other error directions, assignment rules and independent regions remain open.

## Reproduction and detailed results

Run `uv run --no-sync python -m experiments.burn_overlap_sensitivity --output results/causal_e08/sensitivity_replay` from the main project with the original pilot present. The [protocol](e08_sensitivity_protocol.md) was recorded before the variants. Results are in `results/causal_e08/burn_overlap_sensitivity/`.

- `fold_results.csv` gives all 7,236 pooled and fold-specific records, with weak, adequate and hidden counts, both Brier scores and their difference.
- `real_error_prediction.csv` and `real_crossings.csv` give the error and crossing diagnostics overall and by fold, including signed errors and transitions for the two smoothers.
- `control_errors.csv` and `control_frequencies.csv` document the manipulation and paired comparisons. The latter includes plug-in Monte Carlo standard errors; a boundary estimate of zero wins in 100 draws does not establish a zero probability.
- `config.json` records the seed, settings, and input and protocol hashes.
