# E08 · Does propagating fraction uncertainty reveal weak overlap?

On these 263 Burn Scars patches, switching to a positive-beta fraction model reproduced the original disagreement. Density modestly improved predictions under E06 and worsened them under steep assignment. The two rules ask the model about different tails of the *same* predictive distribution. Weak overlap occurred above 43.48% reference burned area for E06 and above 12.44% for steep assignment. Neither rule had a reachable lower-tail cutoff on this cohort.

## Real proxy

Held-out Brier score among the original pilot's apparently adequate patches (lower is better). The primary model uses ordered density quarters; continuous log density is the sensitivity check. Each fraction model was fitted on the other geographic folds, keeping all training patches. The apparent-overlap criterion was identical for every diagnostic.

| Rule | Hidden / eligible | Proxy alone | With density quarter | With log density | Original pilot with log density |
|---|---:|---:|---:|---:|---:|
| E06 | 12/262 | .02834 | .02734 | .02666 | .01725 |
| Steep | 38/258 | .05580 | .05675 | .05893 | .04805 |

Using the original importance-sampled coefficient posterior instead of the Laplace approximation gave density-quarter Brier changes of −.00109 and +.00108, and log-density changes of −.00175 and +.00382. The directions agree. The new diagnostic had higher absolute Brier scores than the direct pilot under both rules. For fraction prediction itself, adding density quarters slightly raised mean held-out log predictive density from 2.0543 to 2.0609; continuous log density lowered it to 2.0356. Overall fit to the fraction and probability in a particular tail measure different things.

| Geographic fold | E06 hidden | E06 Brier change, quarters | Steep hidden | Steep Brier change, quarters |
|---|---:|---:|---:|---:|
| 0 | 0 | <+.00001 | 4 | <+.00001 |
| 1 | 0 | <+.00001 | 1 | +.00064 |
| 2 | 3 | −.00342 | 4 | −.00014 |
| 3 | 9 | −.00118 | 29 | +.00227 |
| 4 | 0 | <+.00001 | 0 | −.00002 |

## Known error mechanism

The positive and negative controls reuse the 100 matched proxy-error draws from the previous E08 analysis. The positive arm concentrates large errors in low-density patches. The negative arm shuffles their magnitudes. In both arms density can still carry information about the true burned fraction through the composition of this cohort. The known-mechanism calculation uses the actual finite reference population and the forward error law. It is a privileged benchmark for the marginal risk conditional on the observed proxy and optionally density quarter. The learned model estimates those risks using other-fold data. Selection on apparent overlap and geographic grouping matter for both comparisons.

| Rule | Diagnostic | Mean Brier change, positive | Mean change, negative | Paired contrast ± MCSE | Positive improvements / 100 | Paired wins / 100 |
|---|---|---:|---:|---:|---:|---:|
| E06 | Learned beta | +.00043 | +.00029 | +.00013 ± .00019 | 26 | 43 |
| E06 | Known mechanism | −.00147 | −.00110 | −.00036 ± .00020 | 74 | 56 |
| Steep | Learned beta | +.00243 | −.00511 | +.00754 ± .00041 | 12 | 3 |
| Steep | Known mechanism | −.00486 | −.00415 | −.00072 ± .00043 | 95 | 55 |

Negative changes favor density. A paired win means density helped more (or hurt less) in the positive arm than in its shuffled counterpart. The known mechanism often benefits from knowing density, particularly under steep assignment, but the incremental benefit specific to injecting density-dependent error is small relative to the total benefit. Neither rule reaches the prespecified 80/100 standard for both columns. Under steep assignment the learned beta model also fails to recover the direction of the known positive-arm gain. Its mean fraction log score improves by .0297 in the positive arm despite worsening weak-overlap Brier. A single smooth beta distribution may poorly capture tails produced by a two-band error mixture; geographic transfer and training size also differ from the privileged calculation. This comparison cannot isolate these explanations.

**Interpretation.** Both the limited incremental signal of the injected mechanism and the learned diagnostic's ability to recover it matter here. A denser or sparser pretraining location alone does not give a reliable warning across rules. These are exploratory results on the same patches, with an assumed within-split density join. No effect on treatment-effect bias or RMSE was estimated.

## Reproduce

From the main repository run `uv run --no-sync python -m experiments.burn_overlap_propagation --output results/causal_e08/propagation_replay`. See the [prespecified protocol](e08_propagation_protocol.md). The reviewed run is in `results/causal_e08/burn_overlap_propagation/`. It includes patch-level real risks, pooled and fold-level Brier and fraction log scores for every arm and draw, paired frequencies, input/code hashes and fraction boundaries. Eligibility and hidden counts were checked against all corresponding original k=15 records. The forward likelihood was checked by numerical normalization and a two-candidate hand calculation.
