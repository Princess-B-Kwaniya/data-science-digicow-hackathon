# DigiCow Hackathon — Predicting Dairy Technology Adoption

**Competition:** [Zindi DigiCow Hackathon](https://zindi.africa/)  
**Team:** Princess-B-Kwaniya  
**Current Best LB Score:** 0.91788  
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

## Best Approach (LB 0.91788)

**CompetitiveSolution.py** — LightGBM trained on Train only, with Prior.csv as feature source

1. **Prior as Features:** Extract 46 farmer/group/geo history features + 12 prior target encodings from Prior.csv
2. **LightGBM:** 205 features, 5-fold StratifiedKFold, scale_pos_weight for imbalance
3. **DUAL Strategy:** Rank-based AUC columns + calibrated LogLoss columns
4. **Prior Blend:** 70% model + 30% prior adoption rate for farmers with history
5. **Post-processing:** Zero-topic/zero-group rules, monotonicity enforcement

## Score Progression

| Submission | LB Score | Strategy | Phase |
|------------|----------|----------|-------|
| COMP_D | **0.91788** | LightGBM + Prior features + DUAL + blend | Phase 2 ★ |
| COMP_A | 0.91748 | LightGBM + Prior features (standard) | Phase 2 |
| REFv3_D | 0.91254 | 60% D + 40% new ensemble blend | Phase 2 ✗ |
| ENS_G | 0.90705 | Training on Prior+Train combined | Phase 2 ✗ |
| REFv3_A | 0.88230 | New features with target leakage | Phase 2 ✗ |
| DUAL_f1 | 0.72828 | Bayesian hierarchical + DUAL | Phase 1 (old data) |

## Key Lessons

- **Prior as features = GOOD.** Prior as training data = BAD (different distributions)
- **More features ≠ better.** Extra features caused target leakage and LB drop (0.918 → 0.882)
- **Trust proven pipelines.** Blending with worse predictions poisons good ones
- **CV can lie.** Higher CV (2.945) with leaked features → worse LB (0.882 vs 0.918)

## Files

See [SOLUTION_EVOLUTION.txt](SOLUTION_EVOLUTION.txt) and [SOLUTION_EXPLANATION.txt](SOLUTION_EXPLANATION.txt) for detailed documentation.