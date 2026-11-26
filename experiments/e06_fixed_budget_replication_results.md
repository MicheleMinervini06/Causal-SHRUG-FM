# E06 · Fixed-budget verification across datasets

On the same fixed-budget OLS benchmark, uncertainty-guided verification beat random verification on all three cohorts, with different score rankings. Normalized embedding distance led on floods, average entropy on Burn Scars, and mutual information on Landslides. The matched baseline uses the same number of checks and draws from the same eligibility pool as the score policies. Every fit retains all patches.

## Confounded results

The table gives the change in treatment-effect RMSE from adding targeting at a matched budget (score-guided minus random). Negative values favor score-guided verification. Intervals quantify paired Monte Carlo precision conditional on the fixed cohort.

| Score | Burns, 5 checks | Burns, 10 checks | Landslides, 5 checks | Landslides, 10 checks |
|---|---:|---:|---:|---:|
| Average entropy | −0.00088 | −0.00927 | +0.02254 | −0.00462 |
| Mutual information | −0.00003 | −0.00699 | −0.21270 | −0.19359 |
| Spatial rarity | +0.00058 | +0.00080 | +0.02968 | +0.05667 |
| Normalized embedding distance | +0.00086 | +0.00007 | −0.12036 | −0.15770 |
| Negative NCDD | −0.00006 | +0.00005 | +0.02446 | +0.05266 |

- **Burn Scars** had 263 patches, with 254 eligible for verification. At 10 checks, random RMSE was 0.15040. Entropy lowered it to 0.14113, a 6.2% improvement (paired change −0.00927, 95% MC interval −0.00991 to −0.00864). Mutual information also helped at 10. Entropy lowered signed bias from +0.07172 to +0.05049 and increased coverage from 92.74% to 93.76%.
- **Landslides** had 68 patches, with 63 eligible. At 10 checks, mutual information lowered RMSE from 0.50634 to 0.31275, a 38.2% improvement (paired change −0.19359, 95% MC interval −0.19842 to −0.18876). The five-check result was 0.32241 versus 0.53511 with random verification. At 10 checks, signed bias fell from +0.41026 to +0.13398 and coverage increased from 76.20% to 92.48%. Full reference adjustment gave RMSE 0.26645. Normalized distance also helped at both budgets.
- **Floods** remain the original 88-patch experiment. At 10 checks, normalized embedding distance lowered RMSE from 0.23570 to 0.23303, a 1.1% improvement. See the [flood report](e06_fixed_budget_results.md).

The number of checks is the same across cohorts. Ten checks correspond to 3.8% of all Burn Scars patches and 14.7% of all Landslides patches. Compare targeting with random selection within each cohort at a matched budget. The best score in each cohort describes these test results. The policy choice would require evaluation on independent data to support selecting that score in advance. The full tables also include the randomized controls, coverage and interval width.

## Reproduction and checks

Run from the main repository with the reviewed score exports:

```powershell
uv run --no-sync python -m experiments.fixed_budget_verification --dataset burns --output results/causal_e06/fixed_budget_burns_replay
uv run --no-sync python -m experiments.fixed_budget_verification --dataset landslides --output results/causal_e06/fixed_budget_landslides_replay
```

The archived runs are `results/causal_e06/fixed_budget_burns_seed42/` and `results/causal_e06/fixed_budget_landslides_seed42/`. They include 5,000 repetitions per scenario, rankings, random reveals and paired intervals. See the [replication protocol](e06_fixed_budget_replication_protocol.md).

All four flood result files matched their original versions byte-for-byte. The two new cohorts reproduced both E07 OLS baselines for bias, RMSE, coverage and width on matching assignments. The Landslides confounded replication keeps all 5,000 assignments. Its archive check uses the E07 subset of 4,999 because that joint Bayesian run excluded repetition 3133.
