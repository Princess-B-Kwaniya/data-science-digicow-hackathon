# ULTIMATE_V6.py & sub_V6_rklo80d60_auc55.csv: Best Model Documentation

## Overview
This document describes the approach, improvements, and future directions for the best submission:
- **Model script:** ULTIMATE_V6.py
- **Submission file:** sub_V6_rklo80d60_auc55.csv

## Concept Used
- **Advanced Feature Engineering:**
  - Farmer and group history, topic overlap, recency bins, group/trainer effectiveness, and more.
  - Vectorized pandas operations for speed (no slow row-wise apply).
- **Modeling:**
  - LightGBM with Optuna hyperparameter tuning.
  - Out-of-fold (OOF) predictions for robust validation.
  - Platt scaling and isotonic regression for probability calibration.
- **Blending:**
  - RKLO (Rank-LogOdds) blending: combines model predictions by ranking, log-odds transform, and weighted sum.
  - The 80d60_auc55 blend means 80% weight on the V6 model, 60% on a teammate/alternative, with 55% AUC emphasis.

## Improvements Over Previous Versions
- **Speed:** Feature engineering reduced from hours to under a minute.
- **Feature Depth:** 323+ features, including new V6 features (topic rarity, overlap, group temporal, etc).
- **Leaderboard Score:** Outperforms all previous blends and single models on the public LB.
- **Calibration:** Better probability estimates for LogLoss.

## What's Left To Be Done
- Add more model diversity (TabNet, CatBoost, DNN) for even stronger blends.
- Automate blend weight optimization (hill climbing, Bayesian search).
- Further feature selection and redundancy reduction.
- Add more documentation and code comments.
- Automate leaderboard submission and tracking.

## File Descriptions
- **ULTIMATE_V6.py:** Full pipeline for feature engineering, model training, calibration, and RKLO blending.
- **sub_V6_rklo80d60_auc55.csv:** Final submission, 5621 rows, ID + 3 targets, best leaderboard score.

---

For any future work, see the TODOs above. This approach is the current state-of-the-art for this competition.
