# E06 · Does score-guided verification beat random verification?

On the 88 Sen1Floods11 test patches, normalized embedding distance gave the lowest treatment-effect RMSE at both verification budgets in the confounded setting. At 10 checks, entropy, mutual information and negative NCDD also improved on random verification. All fits retained the 88 patches. All selection policies drew from the same 85 patches with complete scores.

## Confounded assignments

The table gives treatment-effect RMSE across 5,000 paired assignments and outcomes. Each score is used separately to rank patches. The paired differences are targeted minus random. Lower values favor targeting.

| Verification policy | RMSE at 5 checks | Paired difference (95% MC interval) | RMSE at 10 checks | Paired difference (95% MC interval) |
|---|---:|---:|---:|---:|
| Random | 0.23611 | — | 0.23570 | — |
| Average entropy | 0.23641 | +0.00030 (−0.00007, +0.00066) | 0.23449 | −0.00121 (−0.00192, −0.00049) |
| Mutual information | 0.23661 | +0.00049 (+0.00017, +0.00082) | 0.23459 | −0.00111 (−0.00178, −0.00043) |
| Spatial rarity | 0.23646 | +0.00035 (−0.00002, +0.00072) | 0.23629 | +0.00059 (+0.00005, +0.00114) |
| Normalized embedding distance | 0.23415 | −0.00196 (−0.00260, −0.00132) | 0.23303 | −0.00267 (−0.00366, −0.00168) |
| Negative NCDD | 0.23603 | −0.00008 (−0.00056, +0.00039) | 0.23407 | −0.00163 (−0.00243, −0.00083) |

No verification gave RMSE 0.23658. Full reference adjustment gave 0.23125. At 10 checks, normalized distance lowered signed bias from +0.03351 with random verification to +0.00932. Its RMSE advantage was 1.1% and coverage was 94.88% versus 95.20% for random. Coverage across all confounded policies ranged from 94.76% to 95.30%, including both controls.

## Randomized control and link to the earlier E06 result

Normalized distance also lowered RMSE under randomization (0.22355 to 0.22078 at 5 checks and 0.22327 to 0.21940 at 10), where mean bias was already small. The score can help with overall estimation error even in the randomized scenario.

The earlier E06 result concerned the association between a score and the impact of revealing one fraction inside the Bayesian joint model. This experiment evaluates several simultaneous replacements in raw-proxy OLS. The change in the preferred score can reflect the different estimator and the joint choice of patches. The two analyses do not isolate those explanations. Monte Carlo intervals in this report describe simulation precision on the fixed cohort. The comparison is exploratory and uses pointwise intervals for the five scores and two budgets.

## Reproduce and verify

Run `uv run --no-sync python -m experiments.fixed_budget_verification --output results/causal_e06/fixed_budget_replay` from the main repository with the reviewed E06 score export. The reviewed run is in `results/causal_e06/fixed_budget_seed42/`. It includes per-policy rankings, nested random reveals, paired estimates and intervals, input hashes and settings. See the [fixed protocol](e06_fixed_budget_protocol.md).

The first and last batched estimates in each policy and scenario agreed with the scalar OLS estimate and interval to within 1e-12. The full-run no-verification and all-reference summaries reproduced archived E05 for bias, RMSE, coverage and interval width. The saved selection arrays were checked for unique IDs, exact budgets and shared eligibility.
