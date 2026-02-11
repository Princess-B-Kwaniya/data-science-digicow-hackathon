# DigiCow Hackathon — Predicting Dairy Technology Adoption

**Competition:** [Zindi DigiCow Hackathon](https://zindi.africa/)  
**Team:** Princess-B-Kwaniya  
**Current Best LB Score:** 0.94260  
**Metric:** Per-target `0.75*(1-LogLoss) + 0.25*AUC`, summed across 3 targets (HIGHER = BETTER, max 3.0)

## Problem

Predict whether Kenyan dairy farmers adopt technology within 7, 90, and 120 days of training on the DigiCow platform.

## Data

| File | Rows | Description |
|------|------|-------------|
| Train.csv | 13,536 | Training data with 3 binary targets |
| Test.csv | 5,621 | Test data to predict |
| Prior.csv | 44,882 | Historical farmer training sessions (KEY resource) |

- Target rates: 7d=1.1%, 90d=1.6%, 120d=2.2% (extreme imbalance)
- 62.7% of test farmers have prior history
- Zero farmer overlap between Train and Test

## Best Approach (LB 0.94260)

**90% ULTIMATE_V5.py + 10% ULTIMATE_V4.py blend** — Ensemble of two LightGBM pipelines with complementary strengths

1. **ULTIMATE_V5.py (LB 0.93892):** Deep feature engineering + Chi-square/MI feature selection + Prior deduplication + Topic adoption rates + 250 features + 10 LGB seeds + 5 XGB seeds + Stacking
2. **ULTIMATE_V4.py (LB 0.93544):** LGB-focused with 80 Optuna trials + 20 seeds + Prior deduplication
3. **Blend:** Simple 90/10 weighted average creates diversity → +0.00368 gain over pure V5
4. **DUAL Strategy:** Rank-based AUC columns + calibrated LogLoss columns
5. **Post-processing:** Zero-topic/zero-group rules, monotonicity enforcement

## Score Progression

| Submission | LB Score | Strategy | Phase |
|------------|----------|----------|-------|
| 90%V5+10%V4 | **0.94260** | Blend of two LGB pipelines | Phase 3 ★ |
| V5 pure | 0.93892 | Deep features + chi-square selection | Phase 3 |
| V4 pure | 0.93544 | LGB 80 Optuna + 20 seeds | Phase 3 |
| V2 pure | 0.93333 | LGB+XGB+Stack ensemble | Phase 3 |
| COMP_D | 0.91788 | LightGBM + Prior features + DUAL + blend | Phase 2 |
| COMP_A | 0.91748 | LightGBM + Prior features (standard) | Phase 2 |
| REFv3_D | 0.91254 | 60% D + 40% new ensemble blend | Phase 2 ✗ |
| ENS_G | 0.90705 | Training on Prior+Train combined | Phase 2 ✗ |
| REFv3_A | 0.88230 | New features with target leakage | Phase 2 ✗ |
| DUAL_f1 | 0.72828 | Bayesian hierarchical + DUAL | Phase 1 (old data) |

## Key Lessons

- **Prior as features = GOOD.** Prior as training data = BAD (different distributions)
- **Topic adoption rates = GAME CHANGER.** Per-topic adoption rates from Prior ranked #2-6 in feature importance
- **Chi-square feature selection helps.** Removing 25/279 low-signal features reduced noise
- **More Optuna trials + seeds = better.** 80 trials + 20 seeds (V4) > 50 trials + 10 seeds (V2)
- **LGB alone > multi-model ensemble.** V3 (6 models) FAILED. V4 (LGB-only) succeeded
- **Blending complementary models = huge gains.** 90%V5+10%V4 gained +0.00368 over pure V5
- **CV can mislead.** V5 CV (2.929) < V2 CV (2.933), yet V5 LB (0.93892) >> V2 LB (0.93333)

## Full Documentation

See [SOLUTION_JOURNEY.md](SOLUTION_JOURNEY.md) for complete version history, technical details, and game plan.

## Files

See [SOLUTION_EVOLUTION.txt](SOLUTION_EVOLUTION.txt) and [SOLUTION_EXPLANATION.txt](SOLUTION_EXPLANATION.txt) for detailed documentation.