================================================================================
DIGICOW HACKATHON — COMPLETE SOLUTION JOURNEY & GAME PLAN
================================================================================
Team: Princess-B-Kwaniya
Competition: Zindi DigiCow — Predicting Dairy Technology Adoption
Metric: Per-target 0.75*(1-LogLoss) + 0.25*AUC, summed across 3 targets
        HIGHER leaderboard score = BETTER (max = 3.0)
Updated: February 12, 2026

================================================================================
📊 LEADERBOARD SCORE PROGRESSION
================================================================================

  Score       | Submission                          | Source         | Status
  ------------|-------------------------------------|----------------|--------
  0.94260 ★★★ | sub_TOP_90v5_10v4.csv              | 90%V5 + 10%V4  | BEST
  0.93892     | sub_V5_A_ensemble_dual.csv          | ULTIMATE_V5.py | 2nd
  0.93544     | sub_V4_A_calibrated_dual.csv        | ULTIMATE_V4.py | 3rd
  0.93544     | sub_V4_B_bayesian_dual.csv          | ULTIMATE_V4.py | same
  0.93524     | sub_V4_E_blend30v2_70v4.csv         | 30%V2 + 70%V4  | blend
  0.93452     | sub_V4_E_blend70v2_30v4.csv         | 70%V2 + 30%V4  | blend
  0.93333     | sub_ULT_C_bayesian_dual.csv         | ULTIMATE_V2.py | V2 best
  0.91788     | sub_COMP_D_dual_prior_blend.csv     | CompSolution   | old best
  0.91748     | sub_COMP_A_lgbm_standard.csv        | CompSolution   | old
  0.91254     | sub_REFv3_D (60%D+40%new blend)     | REFINED_v3     | FAILED
  0.90705     | sub_ENS_G (Prior+Train training)    | ENHANCED_V2    | FAILED
  0.88230     | sub_REFv3_A (leaked features)       | REFINED_v3     | FAILED
  0.72828     | Phase 1 old data (invalidated)      | DUAL_FINAL     | old data

================================================================================
🔬 DETAILED VERSION HISTORY
================================================================================

────────────────────────────────────────────────────────────────────────────────
VERSION 1: CompetitiveSolution.py → LB 0.91788
────────────────────────────────────────────────────────────────────────────────
  - Original baseline approach
  - LightGBM trained on Train.csv only (13,536 rows)
  - Prior.csv used as feature source (46 farmer/group/geo features)
  - 205 total features, 5-fold StratifiedKFold
  - DUAL strategy: rank-based AUC + calibrated LogLoss columns
  - Prior rate blending: 70% model + 30% prior adoption rate
  - Zero-group post-processing rules
  - Key insight established: Prior = features, NOT training data

────────────────────────────────────────────────────────────────────────────────
VERSION 2: ULTIMATE_V2.py → LB 0.93333 (+0.01545)
────────────────────────────────────────────────────────────────────────────────
  File: ULTIMATE_V2.py (1361 lines)
  Cache: optuna_v2_cache.json

  WHAT CHANGED FROM V1:
  • Optuna hyperparameter tuning: 50 LGB trials + 25 XGB trials per target
  • Multi-seed ensembling: 10 LGB seeds + 5 XGB seeds per fold
  • Stacking meta-learner (LogisticRegression on OOF predictions)
  • Weighted ensemble: LGB×0.5 + XGB×0.3 + Stack×0.2
  • Multiple calibration methods: Platt, Isotonic, Beta, Bayesian
  • ~240 engineered features (up from 205)
  • Target encoding with proper OOF to prevent leakage
  
  CV RESULTS:
  • LGB-only: ~2.930
  • Weighted Ensemble: ~2.933
  • Bayesian calibration: ~2.933 (marginal improvement)
  
  OPTUNA TUNED PARAMS (per target):
  • Target 07: num_leaves=115, learning_rate=0.050
  • Target 90: num_leaves=90,  learning_rate=0.054
  • Target 120: num_leaves=91, learning_rate=0.041
  
  KEY FEATURES: te_group_name, num_sessions, te_ward_topic, te_county_topic,
  days_since_last_session, session_frequency, group_size

────────────────────────────────────────────────────────────────────────────────
VERSION 3: ULTIMATE_V3.py → DROPPED SCORE ✗ (ABANDONED)
────────────────────────────────────────────────────────────────────────────────
  File: ULTIMATE_V3.py (~1914 lines)

  WHAT CHANGED FROM V2:
  • 6-model ensemble: LGB + XGB + CatBoost + LogReg + ExtraTrees + Stacking
  • Automated Optuna tuning for all model types
  • More sophisticated blending with model weights
  
  WHY IT FAILED:
  • CV = 2.931 (BELOW V2's ~2.933)
  • CatBoost, LogReg, ExtraTrees got near-zero ensemble weights
  • More models ≠ better — weak models add noise, not signal
  • CatBoost subsample was incompatible with Bayesian bootstrap (bug)
  • LGB cached params had float type issues (verbose=-1.0)
  
  LESSON: LGB dominates this problem. Adding weaker model families is noise.

────────────────────────────────────────────────────────────────────────────────
VERSION 4: ULTIMATE_V4.py → LB 0.93544 (+0.00211 from V2)
────────────────────────────────────────────────────────────────────────────────
  File: ULTIMATE_V4.py (1301 lines)
  Cache: optuna_v4_cache.json

  WHAT CHANGED FROM V2:
  • LGB-focused (dropped XGB, CatBoost, etc. — learned from V3)
  • 80 Optuna trials (up from 50) per target
  • 20 LGB seeds (up from 10) for more stable predictions
  • Prior.csv deduplication: 44,882 → ~16,495 unique farmer records
  • ~30 new features including:
    - Prior-informed zero-group rules (more aggressive thresholds)
    - Beta calibration option
    - Improved feature engineering
  
  CV RESULTS:
  • Raw LGB (20 seeds): ~2.932
  • Calibrated: ~2.933
  
  KEY INSIGHT: More Optuna trials + more seeds + deduplication was the recipe.
  Pure LGB with better tuning > multi-model ensemble.

────────────────────────────────────────────────────────────────────────────────
VERSION 5: ULTIMATE_V5.py → LB 0.93892 (+0.00348 from V4)
────────────────────────────────────────────────────────────────────────────────
  File: ULTIMATE_V5.py (~1400 lines)
  Cache: optuna_v5_cache.json

  WHAT CHANGED FROM V2/V4:
  • Deep feature engineering:
    - Topic-level adoption rates from Prior.csv (per topic across all farmers)
    - Training sequence numbers (1st, 2nd, 3rd training for each farmer)
    - Recency-weighted adoption rates (exponential decay)
    - Group size buckets
    - Trainer daily load (how many farmers trainer handles per day)
    - Trainer×county combo adoption rates
    - Topic normalization (cleaned/standardized topic names)
  • Chi-square + Mutual Information feature selection:
    - Started with 279 features, selected 250
    - Dropped 25 low-signal features (age, is_multi_topic, ussd_x_coop, etc.)
  • Prior.csv deduplication (44,882 → 16,495)
  • V2's proven pipeline structure maintained
  • Multi-seed: 10 LGB seeds + 5 XGB seeds + Stacking
  
  CV RESULTS:
  • LGB-only (10 seeds): 2.925
  • XGB-only (5 seeds): 2.920
  • Stacking: 2.907
  • Weighted Ensemble: 2.927
  • Calibrated Ensemble: 2.929
  • Bayesian: 2.929 (model weight = 1.00, Bayesian didn't help)
  
  TOP FEATURES BY IMPORTANCE:
  1. te_group_name (52K)
  2. topic_adoption_rate_120 (29K) ← NEW
  3. num_sessions (25K)
  4. te_ward_topic (24K)
  5. topic_adoption_rate_90 (22K) ← NEW
  6. topic_adoption_rate_07 (21K) ← NEW
  
  OPTUNA PARAMS:
  • Target 07: num_leaves=102, learning_rate=0.051
  • Target 90: num_leaves=90,  learning_rate=0.054
  • Target 120: num_leaves=84, learning_rate=0.017
  
  KEY INSIGHT: CV (2.929) was LOWER than V2 (2.933), but LB was HIGHER!
  The topic_adoption_rate features added real generalization signal that
  CV couldn't fully capture. Feature selection removed noise features.

────────────────────────────────────────────────────────────────────────────────
BLEND: 90% V5 + 10% V4 → LB 0.94260 (CURRENT BEST ★★★)
────────────────────────────────────────────────────────────────────────────────
  File: sub_TOP_90v5_10v4.csv
  Generated by: blend_v5_best.py

  Simple weighted average: 0.90 × V5_predictions + 0.10 × V4_predictions
  
  WHY IT WORKS:
  • V5 has superior feature engineering (topic adoption rates, chi-square selection)
  • V4 has superior Optuna tuning (80 trials vs 50, 20 seeds vs 15)
  • V4 and V5 use DIFFERENT Optuna-tuned hyperparameters → complementary errors
  • V4's 10% contribution adds diversity from different param space exploration
  • Blend exploits the wisdom-of-crowds effect: independent errors cancel out
  
  BLENDING LESSONS LEARNED:
  • Pure V4 (0.93544) > any V2+V4 blend → V2 DILUTES V4
  • Pure V5 (0.93892) > pure V4 (0.93544) → V5 has better features
  • 90%V5 + 10%V4 (0.94260) >> pure V5 (0.93892) → +0.00368 from blend!
  • Blending only helps when models are comparable quality and complementary
  • More V2 = worse: 70v2+30v4 (0.93452) < 30v2+70v4 (0.93524) < pure V4

================================================================================
🔑 KEY DISCOVERIES & LESSONS
================================================================================

1. PRIOR.CSV DEDUPLICATION
   Prior.csv has 44,882 rows but only ~16,495 unique farmer records.
   Deduplicating before computing features gives cleaner statistics.

2. TOPIC ADOPTION RATES (V5's killer feature)
   Computing per-topic adoption rates from Prior.csv across ALL farmers
   (not just the current farmer) captures community-level topic effects.
   These ranked #2, #5, #6 by LightGBM importance.

3. CHI-SQUARE FEATURE SELECTION
   Removing 25/279 low-signal features reduced noise. Dropped features
   included age, is_multi_topic, topic_is_crop, ussd_x_coop,
   prior_ever_adopted_07 — all with weak chi-square / MI scores.

4. MORE OPTUNA TRIALS + SEEDS = CONSISTENT IMPROVEMENT
   V2 (50 trials, 10 seeds) → V4 (80 trials, 20 seeds) showed clear LB gain.
   More stable predictions from seed averaging, better hyperparams from Optuna.

5. SINGLE MODEL FAMILY DOMINATES
   LightGBM alone outperforms LGB+XGB+CatBoost+LogReg+ExtraTrees ensemble.
   V3 (6 models) DROPPED score. V4 (LGB-only) improved it.

6. CV ≠ LB (but blending reveals truth)
   V5 CV (2.929) < V2 CV (2.933), yet V5 LB (0.93892) >> V2 LB (0.93333).
   Cross-validation on this dataset doesn't perfectly capture generalization.

7. BLENDING SWEET SPOT
   Best model as majority (90%) + complementary model as minority (10%) = optimal.
   The blend gain (+0.00368) was larger than V4→V5 gain (+0.00348)!

================================================================================
📁 KEY FILES
================================================================================

  SUBMISSIONS:
    sub_TOP_90v5_10v4.csv           ★ CURRENT BEST (0.94260)
    sub_V5_A_ensemble_dual.csv      2nd best (0.93892)
    sub_V4_A_calibrated_dual.csv    3rd best (0.93544)
    sub_ULT_C_bayesian_dual.csv     V2 best (0.93333)

  PIPELINES:
    ULTIMATE_V5.py                  Best single model pipeline (0.93892)
    ULTIMATE_V4.py                  2nd best pipeline (0.93544)
    ULTIMATE_V2.py                  3rd best pipeline (0.93333)
    ULTIMATE_V3.py                  6-model ensemble (FAILED — abandoned)
    CompetitiveSolution.py          Original baseline (0.91788)

  BLENDING:
    blend_v5_best.py                V5-centered blends with V4
    blend_fine_grid.py              Fine-grained ratio search around 90/10
    blend_v4_best.py                V4-centered blends (obsolete)
    compare_and_blend.py            V2-centered blends (obsolete)

  OPTUNA CACHES:
    optuna_v2_cache.json            V2 tuned hyperparams (50 trials)
    optuna_v4_cache.json            V4 tuned hyperparams (80 trials)
    optuna_v5_cache.json            V5 tuned hyperparams (50 trials)

  DATA:
    Train.csv (13,536), Test.csv (5,621), Prior.csv (44,882)
    SampleSubmission.csv, dataset_data_dictionary.csv

================================================================================
🎯 NEXT GAME PLAN
================================================================================

PHASE A: FINE-TUNE THE BLEND RATIO (Tomorrow — 2 submissions)
───────────────────────────────────────────────────────────────
  PROVEN: 90%V5 + 10%V4 = 0.94260
  
  Submit:
    1. sub_FINE_85v5_15v4.csv  — More V4 (test if 15% V4 > 10% V4)
    2. sub_FINE_95v5_5v4.csv   — Less V4 (test if 5% V4 > 10% V4)
  
  EXPECTED OUTCOME:
    • If 85/15 wins → optimal blend has MORE V4 → try 80/20, 75/25
    • If 95/5 wins  → optimal blend has LESS V4 → try 97/3, 98/2
    • If 90/10 wins → we found the peak, lock it in

  Ready blends available:
    sub_FINE_95v5_5v4.csv, sub_FINE_93v5_7v4.csv, sub_FINE_92v5_8v4.csv,
    sub_FINE_91v5_9v4.csv, sub_FINE_89v5_11v4.csv, sub_FINE_88v5_12v4.csv,
    sub_FINE_87v5_13v4.csv, sub_FINE_85v5_15v4.csv

PHASE B: BUILD ULTIMATE_V6 (If time allows — 2-3 hour investment)
──────────────────────────────────────────────────────────────────
  Goal: Create a 3rd complementary model for triple-blend.
  
  V6 STRATEGY:
    • Start from V5's feature set (proven best features)
    • Use DIFFERENT model architecture: XGBoost-only (instead of LGB)
    • 80+ Optuna trials for XGBoost hyperparameters
    • 20 seeds for XGBoost
    • Same DUAL strategy + calibration
    
  WHY:
    • V5 vs V4 blend works because different Optuna params → different errors
    • XGBoost-only model would have fundamentally different splitting patterns
    • Triple blend (V5 + V4 + V6) could push even higher
    • XGB and LGB make different mistakes → more error cancellation

  ALTERNATIVE V6: CatBoost-only with V5 features
    • CatBoost handles categoricals natively (group_name, ward, county)
    • Different gradient computation than LGB/XGB
    • Could provide even more diversity for blending

PHASE C: FEATURE DISCOVERY (If V6 blend doesn't help)
──────────────────────────────────────────────────────
  Potential new features to explore:
    1. Temporal patterns: Day-of-week effects, seasonal adoption patterns
    2. Network features: Farmer co-occurrence in groups (graph features)
    3. Trainer effectiveness: Trainer's historical success rate
    4. Geographic clustering: Ward/subcounty adoption momentum
    5. Topic difficulty: Which topics are hardest to get adoption for
    6. Farmer engagement trajectory: Is adoption rate increasing/decreasing

PHASE D: ENSEMBLE OF BLENDS (Advanced)
───────────────────────────────────────
  If we have V5, V4, V6 (and potentially V7):
    • Bayesian optimization of blend weights per TARGET per COLUMN TYPE
    • AUC columns might want different weights than LogLoss columns
    • 7-day target might benefit from different blend than 120-day target
    • Rank-based blending (average ranks instead of probabilities)

================================================================================
📈 TRAJECTORY ANALYSIS
================================================================================

  Score gain breakdown:
    V1 → V2: +0.01545 (Optuna tuning + multi-seed + target encoding)
    V2 → V4: +0.00211 (More Optuna trials + more seeds + dedup)
    V4 → V5: +0.00348 (Deep features + chi-square selection)
    V5 → Blend: +0.00368 (90%V5+10%V4 wisdom-of-crowds)
    
    Total V1 → Current: +0.02472
    
  Diminishing returns curve:
    Each version adds incrementally less on its own, but blending
    AMPLIFIES the gains. The blend gain (+0.00368) was the LARGEST
    single improvement since V1→V2.
    
  Key takeaway: Building diverse models and blending them intelligently
  is now MORE valuable than improving any single model.

================================================================================
