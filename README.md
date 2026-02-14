# DigiCow Hackathon — Predicting Dairy Technology Adoption

**Competition:** [Zindi DigiCow Hackathon](https://zindi.africa/)  
**Team:** Princess-B-Kwaniya  
**Current Best LB Score:** 0.950636  
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

## Best Approach (LB 0.950636)

Our best score comes from a **post-prediction blending pipeline** that combines two independently-trained model outputs using three key innovations:

### Source Models
1. **Our model** (`sub_ADV_pertarget_progressive.csv`, LB 0.942924): Multi-seed LightGBM with DUAL column strategy (rank-based AUC + calibrated LogLoss), per-target progressive blending of V4/V5 pipelines, Prior.csv as feature source (farmer/group/geo history)
2. **Teammate's model** (`submission_v4_ensemble (3).csv`): LGB+CatBoost+XGB ensemble with topic adoption rates, farmer history, trainer-county interactions

### Blending Pipeline (5 steps)

**Step 1 — DUAL Column Strategy (foundation)**  
AUC columns use percentile ranks (optimal for ranking). LogLoss columns use raw calibrated probabilities (optimal for calibration). We keep these separate throughout.

**Step 2 — LL-Only Blending**  
Only blend LogLoss columns (75% of metric weight). AUC columns stay from our best model since they're already rank-optimized. This avoids destroying AUC optimization while gaining diversity on LL.

**Step 3 — Rank Averaging** (+0.003 over arithmetic blending)  
Instead of averaging probabilities (which pulls predictions toward 0.5 and hurts rare-event calibration), we:
1. Compute percentile ranks for both sources per LL column
2. Take weighted average of RANKS (not values): `blended = 0.20 * rank(ours) + 0.80 * rank(teammate)`
3. Map blended ranks back to original sorted value distribution

This preserves the calibration distribution shape while using consensus ordering from both models.

**Step 4 — RANKLOGODDS Distribution** (+0.000175)  
Instead of mapping to one source's distribution, blend both distributions in logit space:
```
sorted_blend = sigmoid(0.40 * logit(sort(ours)) + 0.60 * logit(sort(teammate)))
output = sorted_blend[ordinal_rank(blended_ranks) - 1]
```
Logit space is the natural scale for calibration with extreme class imbalance (1-2% positive rates).

**Step 5 — AUC Column Blending** (+0.000817)  
Apply light rank-averaging (15% teammate weight) to AUC columns too. Even small diversity on the 25% AUC component yields significant gains. This dimension is still climbing — not yet peaked.

**Step 6 — Monotonicity Enforcement**  
Ensure `pred_7d <= pred_90d <= pred_120d` (a farmer who adopts within 7 days must have adopted within 90 and 120 days).

## Score Progression

| Date | Score | Method |
|------|-------|--------|
| Feb 10 | 0.91788 | Single LGB + Prior features + DUAL |
| Feb 11 | 0.93892 | Multi-seed LGB (V4/V5 pipelines) |
| Feb 12 | 0.942924 | Per-target progressive V4/V5 blend |
| Feb 12 | 0.946010 | LL-only arithmetic blend with teammate |
| Feb 13 | 0.947179 | Rank averaging discovery |
| Feb 13 | 0.949640 | Rank-avg gradient search (85% V4) |
| Feb 14 | 0.949819 | RANKLOGODDS distribution mapping |
| Feb 14 | **0.950636** | + AUC column blending (15% V4) |

## Key Lessons

- **Rank averaging >>> arithmetic blending** for rare events. Arithmetic mean destroys calibration for 1-2% positive rates. Rank averaging preserves distribution shape. Worth +0.003 at same blend ratios
- **Blend ALL metric components.** We initially only blended LL (75% of metric). Adding 15% V4 to AUC columns gave +0.000817 — independent diversity gains
- **Post-prediction blending > model improvements.** Our best single model scored 0.942924. Blending added +0.007712 without changing any model
- **Logit-space calibration blending** is superior for extreme class imbalance
- **Prior.csv as features, NEVER as training data.** Different adoption rate distributions cause calibration shift
- **DUAL column strategy** (rank AUC + calibrated LL) provides consistent gains

## Repository Structure

```
# Core Scripts
CompetitiveSolution.py      # Original LGB pipeline (LB 0.91788)
ULTIMATE_V4.py              # 20-seed LGB with Optuna (LB 0.93544)
ULTIMATE_V5.py              # LGB+XGB+Stack ensemble (LB 0.93892)
gen_best.py                 # RKLO + AUC blending experiments
gen_best2.py                # Full AUC gradient grid (winning approach)

# Analysis
cross_analysis.py           # Prior/Train/Test diagnostic analysis
DEEP_EDA_ANALYSIS.py        # Deep exploratory data analysis

# Data
Train.csv                   # Training data (13,536 rows)
Test.csv                    # Test data (5,621 rows)
Prior.csv                   # Historical sessions (44,882 rows)
SampleSubmission.csv        # Submission format
dataset_data_dictionary.csv # Column descriptions

# Best Submission
sub_BEST2_rklo80d60_auc15.csv  # LB 0.950636 (CURRENT BEST)

# Source Files for Blending
sub_ADV_pertarget_progressive.csv  # Our best model output
submission_v4_ensemble (3).csv     # Teammate's model output

# Untested Submissions (ready for next day)
sub_BEST2_rklo80d60_auc{20-50}.csv   # AUC gradient (still climbing)
sub_BEST2_rkloFULL_auc{10-30}.csv    # RKLO on AUC columns too
sub_BEST2_aucPT_{A-F}_*.csv          # Per-target AUC optimization

# Documentation
README.md                   # This file
SOLUTION_EXPLANATION.txt    # Detailed technical explanation
```

## What's Next

1. **Continue AUC gradient:** Test 20%, 25%, 30% V4 on AUC columns (gains still linear at ~250/step)
2. **RKLO on AUC:** Apply logodds distribution mapping to AUC columns too (not just rank-avg)
3. **Per-target AUC:** Different AUC% per target — each target may have different optimal blend ratio
4. **Fine-grid LL parameters:** Test rank_weight=78-82%, dist_ratio=55-65% for marginal gains
5. **Third model source:** Generate independent CatBoost predictions for triple-source blending