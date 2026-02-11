"""
ULTIMATE_V4.py — LightGBM-Focused Pipeline with Deep Feature Engineering
=========================================================================
STRATEGY: Pure LGB focus (what actually scored 0.93333) + deep data cleaning 
          + new features V2 missed.

KEY INSIGHTS FROM DATA ANALYSIS:
  1. Prior covers test period (until Dec 2025!) — temporal features from recent Prior are gold
  2. Prior-adopted farmers → 25.7% Train adoption (17x signal)
  3. High-session farmers (16+) → 7.75% adoption (4x baseline)
  4. Specific topics drive adoption hugely (24% for health topics vs 0% for some)
  5. 54% of test in Prior zero-adoption groups → aggressive capping
  6. 36K near-duplicates in Prior — deduplicate for cleaner statistics
  7. 2025 rates much lower than 2024 — weight recent data more
  8. USSD + Coop + has_topic = triple signal for adoption
  9. Temporal gap: Test is May-Dec 2025, Train is Jan-Apr 2025 — seasonality matters
  10. 6 test wards NOT in train — need Prior-based fallbacks

CHANGES FROM V2 (0.93333):
  - DATA CLEANING: Deduplicate Prior, handle temporal shift
  - 30+ NEW FEATURES from deeper analysis
  - BETTER TARGET ENCODING: Hierarchical fallback for unseen categories
  - BETTER OPTUNA: 80 trials with wider search space
  - MORE SEEDS: 20 LGB seeds for more variance reduction
  - IMPROVED POST-PROCESSING: Prior-informed zero-group rules
  - STRONGER CALIBRATION: Beta calibration + per-fold Platt
  - NO XGB, NO STACKING: LGB dominates anyway (0.98 weight in V2/V3)

PROVEN RULES (never violate):
  - Train on Train ONLY; Prior ONLY as feature source
  - DUAL strategy for AUC vs LogLoss columns
  - Calibration is paramount (75% LogLoss weight)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import optuna
import ast
import json
import warnings
import os
import gc
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = r"C:\Users\USER\Desktop\DIGICOW"
os.chdir(DATA_DIR)

start_time = time.time()

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
N_FOLDS = 5
BASE_SEED = 42

def competition_score(y_true, y_pred):
    """Official competition metric per target."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    return 0.75 * (1 - ll) + 0.25 * auc

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("ULTIMATE V4 — STEP 1: Loading data...")
print("=" * 70)

train_df = pd.read_csv('Train.csv')
test_df  = pd.read_csv('Test.csv')
prior_df = pd.read_csv('Prior.csv')
ss       = pd.read_csv('SampleSubmission.csv')

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Prior: {prior_df.shape}")
for t in TARGETS:
    print(f"  Train {t}: {train_df[t].mean():.4f} ({train_df[t].sum()}/{len(train_df)})")

SS_COLS = list(ss.columns)
TARGET_TO_SS = {
    'adopted_within_07_days':  ('Target_07_AUC',  'Target_07_LogLoss'),
    'adopted_within_90_days':  ('Target_90_AUC',  'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

# ============================================================
# 2. DATA CLEANING (NEW IN V4)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Data cleaning...")
print("=" * 70)

# 2A. Parse formats
def parse_topics_nested(s):
    try:
        parsed = ast.literal_eval(s)
        all_topics = []
        for session in parsed:
            if isinstance(session, list):
                all_topics.extend(session)
            else:
                all_topics.append(str(session))
        return list(set(all_topics))
    except:
        return []

def parse_topics_flat(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return list(set(str(x) for x in parsed))
        return [str(parsed)]
    except:
        return []

def parse_trainer_list(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return str(parsed)
    except:
        return str(s)

def count_sessions_nested(s):
    try:
        parsed = ast.literal_eval(s)
        return len(parsed)
    except:
        return 1

train_df['topics_parsed'] = train_df['topics_list'].apply(parse_topics_nested)
train_df['trainer_parsed'] = train_df['trainer'].apply(parse_trainer_list)
train_df['num_sessions'] = train_df['topics_list'].apply(count_sessions_nested)

test_df['topics_parsed'] = test_df['topics_list'].apply(parse_topics_nested)
test_df['trainer_parsed'] = test_df['trainer'].apply(parse_trainer_list)
test_df['num_sessions'] = test_df['topics_list'].apply(count_sessions_nested)

prior_df['topics_parsed'] = prior_df['topics_list'].apply(parse_topics_flat)
prior_df['trainer_parsed'] = prior_df['trainer']
prior_df['num_sessions'] = 1

# 2B. Clean Prior: Deduplicate near-duplicate rows
# 36K of 44K Prior rows are near-duplicates — aggregate targets for cleaner stats
print("  2B. Deduplicating Prior...")
prior_df['training_day_dt'] = pd.to_datetime(prior_df['training_day'])
prior_orig_len = len(prior_df)

# For farmer-level features, use deduplicated Prior (take max of targets for same farmer+day)
dedup_cols = ['farmer_name', 'training_day', 'gender', 'age', 'group_name',
              'belong_to_cooperative', 'county', 'subcounty', 'ward',
              'has_topic_trained_on', 'trainer_parsed', 'registration']
prior_dedup = prior_df.groupby(dedup_cols, as_index=False).agg(
    adopted_within_07_days=('adopted_within_07_days', 'max'),
    adopted_within_90_days=('adopted_within_90_days', 'max'),
    adopted_within_120_days=('adopted_within_120_days', 'max'),
    dup_count=('ID', 'count'),
)
print(f"  Prior: {prior_orig_len} → {len(prior_dedup)} after dedup ({prior_orig_len-len(prior_dedup)} removed)")

# 2C. Temporal weight: Recent Prior data (2025) more relevant than 2024
prior_dedup['training_day_dt'] = pd.to_datetime(prior_dedup['training_day'])
prior_dedup['year'] = prior_dedup['training_day_dt'].dt.year
print(f"  Prior 2024: {(prior_dedup['year']==2024).sum()} rows, 120d rate={prior_dedup.loc[prior_dedup['year']==2024, 'adopted_within_120_days'].mean():.4f}")
print(f"  Prior 2025: {(prior_dedup['year']==2025).sum()} rows, 120d rate={prior_dedup.loc[prior_dedup['year']==2025, 'adopted_within_120_days'].mean():.4f}")

print("  Parsing complete.")

# ============================================================
# 3. BUILD FARMER HISTORY FROM PRIOR (ENHANCED)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building ENHANCED farmer history from Prior...")
print("=" * 70)

# 3A. Standard farmer history (from deduped Prior)
farmer_hist = prior_dedup.groupby('farmer_name').agg(
    prior_session_count=('training_day', 'count'),
    prior_07_adopted=('adopted_within_07_days', 'sum'),
    prior_90_adopted=('adopted_within_90_days', 'sum'),
    prior_120_adopted=('adopted_within_120_days', 'sum'),
    prior_07_rate=('adopted_within_07_days', 'mean'),
    prior_90_rate=('adopted_within_90_days', 'mean'),
    prior_120_rate=('adopted_within_120_days', 'mean'),
    prior_has_topic_rate=('has_topic_trained_on', 'mean'),
    prior_coop_rate=('belong_to_cooperative', 'mean'),
    prior_unique_groups=('group_name', 'nunique'),
    prior_unique_wards=('ward', 'nunique'),
    prior_unique_trainers=('trainer_parsed', 'nunique'),
    prior_dup_count_mean=('dup_count', 'mean'),
).reset_index()

# Topic diversity
topic_div = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))
).reset_index()
topic_div.columns = ['farmer_name', 'prior_unique_topics']
farmer_hist = farmer_hist.merge(topic_div, on='farmer_name', how='left')

# Date features
date_feats = prior_dedup.groupby('farmer_name')['training_day_dt'].agg(
    prior_first_date='min', prior_last_date='max'
).reset_index()
date_feats['prior_training_span_days'] = (date_feats['prior_last_date'] - date_feats['prior_first_date']).dt.days
farmer_hist = farmer_hist.merge(date_feats[['farmer_name', 'prior_training_span_days',
                                             'prior_first_date', 'prior_last_date']], 
                                on='farmer_name', how='left')

# Binary adoption flags
farmer_hist['prior_ever_adopted_07'] = (farmer_hist['prior_07_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_90'] = (farmer_hist['prior_90_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_120'] = (farmer_hist['prior_120_adopted'] > 0).astype(int)
farmer_hist['prior_any_adoption'] = ((farmer_hist['prior_07_adopted'] + farmer_hist['prior_90_adopted'] + farmer_hist['prior_120_adopted']) > 0).astype(int)
farmer_hist['prior_adoption_score'] = (
    farmer_hist['prior_07_rate'] * 3 + farmer_hist['prior_90_rate'] * 2 + farmer_hist['prior_120_rate'] * 1
) / 6

# Engagement features
farmer_hist['prior_engagement_intensity'] = farmer_hist['prior_session_count'] / (farmer_hist['prior_training_span_days'].clip(lower=1) / 30.0)
farmer_hist['prior_adoption_consistency'] = farmer_hist[['prior_07_rate','prior_90_rate','prior_120_rate']].std(axis=1)
farmer_hist['prior_is_loyal'] = (farmer_hist['prior_unique_groups'] == 1).astype(int)
farmer_hist['prior_high_sessions'] = (farmer_hist['prior_session_count'] >= 10).astype(int)

# 3B. V4 NEW: Recent-only history (2025 only — much more predictive for test which is May-Dec 2025)
print("  V4 NEW: Recent-only farmer history (2025)...")
recent_prior = prior_dedup[prior_dedup['year'] == 2025]
recent_farmer = recent_prior.groupby('farmer_name').agg(
    recent_session_count=('training_day', 'count'),
    recent_07_rate=('adopted_within_07_days', 'mean'),
    recent_90_rate=('adopted_within_90_days', 'mean'),
    recent_120_rate=('adopted_within_120_days', 'mean'),
    recent_07_adopted=('adopted_within_07_days', 'sum'),
    recent_90_adopted=('adopted_within_90_days', 'sum'),
    recent_120_adopted=('adopted_within_120_days', 'sum'),
    recent_has_topic_rate=('has_topic_trained_on', 'mean'),
).reset_index()
recent_farmer['recent_any_adoption'] = ((recent_farmer['recent_07_adopted'] + recent_farmer['recent_90_adopted'] + recent_farmer['recent_120_adopted']) > 0).astype(int)
recent_farmer['recent_adoption_score'] = (
    recent_farmer['recent_07_rate'] * 3 + recent_farmer['recent_90_rate'] * 2 + recent_farmer['recent_120_rate'] * 1
) / 6
farmer_hist = farmer_hist.merge(recent_farmer, on='farmer_name', how='left')
for c in recent_farmer.columns:
    if c != 'farmer_name' and c in farmer_hist.columns:
        farmer_hist[c] = farmer_hist[c].fillna(0)

print(f"  Recent farmer features: {len(recent_farmer)} farmers (from {len(recent_prior)} sessions)")

# 3C. V4 NEW: Last-session features (what happened most recently?)
print("  V4 NEW: Last-session features...")
last_session = prior_dedup.sort_values('training_day_dt').groupby('farmer_name').last()
farmer_hist['last_07'] = farmer_hist['farmer_name'].map(last_session['adopted_within_07_days']).fillna(0)
farmer_hist['last_90'] = farmer_hist['farmer_name'].map(last_session['adopted_within_90_days']).fillna(0)
farmer_hist['last_120'] = farmer_hist['farmer_name'].map(last_session['adopted_within_120_days']).fillna(0)
farmer_hist['last_has_topic'] = farmer_hist['farmer_name'].map(last_session['has_topic_trained_on']).fillna(0)
farmer_hist['last_coop'] = farmer_hist['farmer_name'].map(last_session['belong_to_cooperative']).fillna(0)

# 3D. V4 NEW: Adoption trajectory (getting better or worse over time?)
print("  V4 NEW: Adoption trajectory...")
# Split prior into first half and second half per farmer
def calc_trajectory(farmer_group):
    if len(farmer_group) < 2:
        return pd.Series({'traj_07': 0, 'traj_90': 0, 'traj_120': 0})
    mid = len(farmer_group) // 2
    first = farmer_group.iloc[:mid]
    second = farmer_group.iloc[mid:]
    return pd.Series({
        'traj_07': second['adopted_within_07_days'].mean() - first['adopted_within_07_days'].mean(),
        'traj_90': second['adopted_within_90_days'].mean() - first['adopted_within_90_days'].mean(),
        'traj_120': second['adopted_within_120_days'].mean() - first['adopted_within_120_days'].mean(),
    })

traj = prior_dedup.sort_values('training_day_dt').groupby('farmer_name').apply(calc_trajectory).reset_index()
farmer_hist = farmer_hist.merge(traj, on='farmer_name', how='left')
for c in ['traj_07', 'traj_90', 'traj_120']:
    farmer_hist[c] = farmer_hist[c].fillna(0)

test_coverage = test_df['farmer_name'].isin(farmer_hist['farmer_name']).mean()
print(f"  {len(farmer_hist)} farmers, {farmer_hist.shape[1]-3} features, test coverage: {test_coverage:.1%}")

# Drop date columns before merge
farmer_hist = farmer_hist.drop(columns=['prior_first_date', 'prior_last_date'], errors='ignore')

# ============================================================
# 4. BUILD GROUP HISTORY FROM PRIOR (ENHANCED)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Building ENHANCED group history from Prior...")
print("=" * 70)

group_hist = prior_dedup.groupby('group_name').agg(
    prior_grp_size=('training_day', 'count'),
    prior_grp_07_rate=('adopted_within_07_days', 'mean'),
    prior_grp_90_rate=('adopted_within_90_days', 'mean'),
    prior_grp_120_rate=('adopted_within_120_days', 'mean'),
    prior_grp_has_topic_rate=('has_topic_trained_on', 'mean'),
    prior_grp_coop_rate=('belong_to_cooperative', 'mean'),
    prior_grp_unique_farmers=('farmer_name', 'nunique'),
    prior_grp_unique_trainers=('trainer_parsed', 'nunique'),
).reset_index()
group_hist['prior_grp_adoption_score'] = (
    group_hist['prior_grp_07_rate'] * 3 + group_hist['prior_grp_90_rate'] * 2 + group_hist['prior_grp_120_rate'] * 1
) / 6
group_hist['prior_grp_sessions_per_farmer'] = group_hist['prior_grp_size'] / group_hist['prior_grp_unique_farmers'].clip(lower=1)
group_hist['prior_grp_any_adoption'] = ((group_hist['prior_grp_07_rate'] + group_hist['prior_grp_90_rate'] + group_hist['prior_grp_120_rate']) > 0).astype(int)

# V4 NEW: Recent group rates (2025 only)
recent_grp = recent_prior.groupby('group_name').agg(
    recent_grp_size=('training_day', 'count'),
    recent_grp_07_rate=('adopted_within_07_days', 'mean'),
    recent_grp_90_rate=('adopted_within_90_days', 'mean'),
    recent_grp_120_rate=('adopted_within_120_days', 'mean'),
).reset_index()
recent_grp['recent_grp_adoption_score'] = (
    recent_grp['recent_grp_07_rate'] * 3 + recent_grp['recent_grp_90_rate'] * 2 + recent_grp['recent_grp_120_rate'] * 1
) / 6
group_hist = group_hist.merge(recent_grp, on='group_name', how='left')
for c in recent_grp.columns:
    if c != 'group_name' and c in group_hist.columns:
        group_hist[c] = group_hist[c].fillna(0)

# V4 NEW: Zero-adoption group flag (strong rule for 54% of test)
for t in TARGETS:
    rate_col = f'prior_grp_{t.split("_")[2]}_rate'
    group_hist[f'prior_grp_zero_{t.split("_")[2]}'] = ((group_hist[rate_col] == 0) & (group_hist['prior_grp_size'] >= 15)).astype(int)

print(f"  {len(group_hist)} groups, {group_hist.shape[1]-1} features")

# ============================================================
# 5. GEOGRAPHIC + TRAINER HISTORY FROM PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Geographic & trainer history from Prior...")
print("=" * 70)

for geo_col in ['ward', 'subcounty', 'county']:
    geo_hist = prior_dedup.groupby(geo_col).agg(
        **{f'prior_{geo_col}_size': ('training_day', 'count'),
           f'prior_{geo_col}_07_rate': ('adopted_within_07_days', 'mean'),
           f'prior_{geo_col}_90_rate': ('adopted_within_90_days', 'mean'),
           f'prior_{geo_col}_120_rate': ('adopted_within_120_days', 'mean'),
           f'prior_{geo_col}_coop_rate': ('belong_to_cooperative', 'mean'),
           f'prior_{geo_col}_has_topic_rate': ('has_topic_trained_on', 'mean'),
           }
    ).reset_index()
    
    # V4 NEW: Recent geo rates
    recent_geo = recent_prior.groupby(geo_col).agg(
        **{f'recent_{geo_col}_07_rate': ('adopted_within_07_days', 'mean'),
           f'recent_{geo_col}_90_rate': ('adopted_within_90_days', 'mean'),
           f'recent_{geo_col}_120_rate': ('adopted_within_120_days', 'mean'),
           f'recent_{geo_col}_size': ('training_day', 'count'),
           }
    ).reset_index()
    geo_hist = geo_hist.merge(recent_geo, on=geo_col, how='left')
    
    train_df = train_df.merge(geo_hist, on=geo_col, how='left')
    test_df = test_df.merge(geo_hist, on=geo_col, how='left')
    print(f"  {geo_col}: {len(geo_hist)} values")

# Trainer effectiveness (smoothed)
print("  Trainer effectiveness...")
trainer_eff = prior_dedup.groupby('trainer_parsed').agg(
    prior_trainer_total=('training_day', 'count'),
    prior_trainer_07_rate=('adopted_within_07_days', 'mean'),
    prior_trainer_90_rate=('adopted_within_90_days', 'mean'),
    prior_trainer_120_rate=('adopted_within_120_days', 'mean'),
    prior_trainer_unique_farmers=('farmer_name', 'nunique'),
    prior_trainer_unique_groups=('group_name', 'nunique'),
    prior_trainer_coop_rate=('belong_to_cooperative', 'mean'),
    prior_trainer_topic_rate=('has_topic_trained_on', 'mean'),
).reset_index()
trainer_eff['prior_trainer_effectiveness'] = (
    trainer_eff['prior_trainer_07_rate'] * 3 + 
    trainer_eff['prior_trainer_90_rate'] * 2 + 
    trainer_eff['prior_trainer_120_rate'] * 1
) / 6

TRAINER_SMOOTH = 50
for rate_col in ['prior_trainer_07_rate', 'prior_trainer_90_rate', 'prior_trainer_120_rate']:
    target_name = rate_col.replace('prior_trainer_', '').replace('_rate', '')
    global_rate = prior_dedup[f'adopted_within_{target_name}_days'].mean()
    trainer_eff[f'{rate_col}_smoothed'] = (
        trainer_eff[rate_col] * trainer_eff['prior_trainer_total'] + global_rate * TRAINER_SMOOTH
    ) / (trainer_eff['prior_trainer_total'] + TRAINER_SMOOTH)

train_df = train_df.merge(trainer_eff, on='trainer_parsed', how='left')
test_df = test_df.merge(trainer_eff, on='trainer_parsed', how='left')
print(f"  {len(trainer_eff)} trainers")

# ============================================================
# 6. MERGE HISTORY FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Merging history features...")
print("=" * 70)

train_df = train_df.merge(farmer_hist, on='farmer_name', how='left')
test_df = test_df.merge(farmer_hist, on='farmer_name', how='left')
train_df = train_df.merge(group_hist, on='group_name', how='left')
test_df = test_df.merge(group_hist, on='group_name', how='left')

# Fill NaN with 0 for all history columns
hist_cols = [c for c in farmer_hist.columns if c != 'farmer_name'] + \
            [c for c in group_hist.columns if c != 'group_name'] + \
            [c for c in trainer_eff.columns if c != 'trainer_parsed']
for c in hist_cols:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

# Fill geo features NaN
for c in train_df.columns:
    if ('prior_ward' in c or 'prior_subcounty' in c or 'prior_county' in c or 
        'recent_ward' in c or 'recent_subcounty' in c or 'recent_county' in c):
        if train_df[c].isnull().any():
            fill_val = train_df[c].median() if train_df[c].dtype in ['float64', 'int64'] else 0
            train_df[c] = train_df[c].fillna(fill_val)
        if test_df[c].isnull().any():
            fill_val = test_df[c].median() if test_df[c].dtype in ['float64', 'int64'] else 0
            test_df[c] = test_df[c].fillna(fill_val)

print(f"  Train rows with farmer history: {(train_df['prior_session_count'] > 0).sum()}/{len(train_df)}")
print(f"  Test rows with farmer history: {(test_df['prior_session_count'] > 0).sum()}/{len(test_df)}")

# ============================================================
# 7. DEEP FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Deep Feature Engineering...")
print("=" * 70)

train_df['is_train'] = 1
test_df['is_train'] = 0
for t in TARGETS:
    if t not in test_df.columns:
        test_df[t] = np.nan

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_idx = df['is_train'] == 1
test_idx = df['is_train'] == 0

# 7A. TEMPORAL FEATURES (ENHANCED)
print("  7A. Temporal features...")
df['training_day_dt'] = pd.to_datetime(df['training_day'])
df['train_year'] = df['training_day_dt'].dt.year
df['train_month'] = df['training_day_dt'].dt.month
df['train_day'] = df['training_day_dt'].dt.day
df['train_dayofweek'] = df['training_day_dt'].dt.dayofweek
df['train_quarter'] = df['training_day_dt'].dt.quarter
df['train_weekofyear'] = df['training_day_dt'].dt.isocalendar().week.astype(int)
df['train_dayofyear'] = df['training_day_dt'].dt.dayofyear
df['is_weekend'] = (df['train_dayofweek'] >= 5).astype(int)
df['is_month_start'] = df['training_day_dt'].dt.is_month_start.astype(int)
df['is_month_end'] = df['training_day_dt'].dt.is_month_end.astype(int)
df['days_since_epoch'] = (df['training_day_dt'] - pd.Timestamp('2024-01-01')).dt.days
df['month_sin'] = np.sin(2 * np.pi * df['train_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['train_month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['train_dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['train_dayofweek'] / 7)

# Season (Kenya agricultural seasons)
def get_season(m):
    if m in [3, 4, 5]: return 0  # Long rains
    elif m in [6, 7, 8]: return 1  # Cool dry
    elif m in [10, 11]: return 2  # Short rains
    elif m in [12, 1, 2]: return 3  # Hot dry
    else: return 4  # September transition
df['season'] = df['train_month'].apply(get_season)

# V4 NEW: Days since last Prior session (recency = powerful signal)
prior_last_dates = prior_dedup.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last_dates.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last_dates else -1, axis=1)
df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)

# V4 NEW: High adoption months from Train data
df['is_high_adoption_month'] = df['train_month'].isin([3, 8, 9]).astype(int)

# V4 NEW: Recency buckets
df['recency_bucket'] = pd.cut(df['days_since_last_prior'], 
                               bins=[-999, -1, 0, 30, 90, 180, 365, 9999],
                               labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)

# V4 NEW: Is test period? (temporal awareness)
df['is_2025'] = (df['train_year'] == 2025).astype(int)
df['is_h2_2025'] = ((df['train_year'] == 2025) & (df['train_month'] >= 7)).astype(int)

# 7B. TOPIC FEATURES (ENHANCED)
print("  7B. Topic features...")
df['topic_count'] = df['topics_parsed'].apply(len)
df['is_multi_topic'] = (df['topic_count'] > 1).astype(int)

def extract_topic_cats(topics):
    t = ' '.join(topics).lower()
    cats = []
    if any(w in t for w in ['dairy', 'cow', 'milk', 'lactating', 'calf']): cats.append('dairy')
    if any(w in t for w in ['poultry', 'chicken', 'egg', 'chick', 'layer', 'kienyeji']): cats.append('poultry')
    if any(w in t for w in ['feed', 'nutrition', 'tyari', 'unga']): cats.append('feeding')
    if any(w in t for w in ['health', 'disease', 'vaccin', 'deworm', 'biosecurity', 'mngt']): cats.append('health')
    if any(w in t for w in ['record', 'business', 'market', 'financial']): cats.append('business')
    if any(w in t for w in ['breed', 'ai ', 'artificial', 'reproduction', 'calving']): cats.append('breeding')
    if any(w in t for w in ['housing', 'shelter']): cats.append('housing')
    if any(w in t for w in ['hygiene', 'milking', 'ppe']): cats.append('hygiene')
    if any(w in t for w in ['app', 'ndume', 'digital']): cats.append('tech')
    if any(w in t for w in ['pest', 'crop', 'maize', 'bean', 'seed', 'weed', 'fertiliz']): cats.append('crop')
    if not cats: cats.append('other')
    return cats

df['topic_cats'] = df['topics_parsed'].apply(extract_topic_cats)
df['primary_topic_cat'] = df['topic_cats'].apply(lambda x: x[0])
df['num_topic_cats'] = df['topic_cats'].apply(len)

all_cats = ['dairy', 'poultry', 'feeding', 'health', 'business', 'breeding', 'housing', 'hygiene', 'tech', 'crop', 'other']
for cat in all_cats:
    df[f'topic_is_{cat}'] = df['topic_cats'].apply(lambda x, c=cat: int(c in x))

# V4 NEW: High-adoption topic flag (topics with >5% 120d adoption rate)
high_adopt_keywords = ['health', 'disease', 'vaccin', 'deworm', 'ppe', 'hygiene', 'milking',
                       'biodeal', 'tyari', 'mineral']
df['has_high_adopt_topic'] = df['topics_parsed'].apply(
    lambda topics: int(any(any(kw in t.lower() for kw in high_adopt_keywords) for t in topics)))

# V4 NEW: Zero-adoption topic flag
zero_adopt_keywords = ['importance of mineral supplementation with ckl', 'pest', 'livestock management with biodeal']
df['has_zero_adopt_topic'] = df['topics_parsed'].apply(
    lambda topics: int(any(any(kw in t.lower() for kw in zero_adopt_keywords) for t in topics)))

# 7C. GEOGRAPHIC INTERACTIONS
print("  7C. Geographic interactions...")
df['county_subcounty'] = df['county'] + '_' + df['subcounty']
df['subcounty_ward'] = df['subcounty'] + '_' + df['ward']
df['county_ward'] = df['county'] + '_' + df['ward']
df['county_trainer'] = df['county'] + '_' + df['trainer_parsed']
df['ward_trainer'] = df['ward'] + '_' + df['trainer_parsed']
df['county_topic'] = df['county'] + '_' + df['primary_topic_cat']
df['ward_topic'] = df['ward'] + '_' + df['primary_topic_cat']
df['trainer_topic'] = df['trainer_parsed'] + '_' + df['primary_topic_cat']

# 7D. FREQUENCY ENCODING
print("  7D. Frequency encoding...")
freq_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
             'primary_topic_cat', 'county_subcounty', 'county_topic',
             'ward_topic', 'trainer_topic', 'county_trainer']
for col in freq_cols:
    df[f'{col}_freq'] = df.groupby(col)[col].transform('count')

# 7E. GROUP FEATURES
print("  7E. Group features...")
df['group_size'] = df.groupby('group_name')['group_name'].transform('count')
df['group_coop_rate'] = df.groupby('group_name')['belong_to_cooperative'].transform('mean')
df['group_female_rate'] = df.groupby('group_name')['gender'].transform(lambda x: (x == 'Female').mean())
df['group_young_rate'] = df.groupby('group_name')['age'].transform(lambda x: (x == 'Below 35').mean())
df['group_ussd_rate'] = df.groupby('group_name')['registration'].transform(lambda x: (x == 'Ussd').mean())
df['group_topic_diversity'] = df.groupby('group_name')['primary_topic_cat'].transform('nunique')
df['group_trainer_diversity'] = df.groupby('group_name')['trainer_parsed'].transform('nunique')
df['group_has_topic_rate'] = df.groupby('group_name')['has_topic_trained_on'].transform('mean')
df['group_session_mean'] = df.groupby('group_name')['num_sessions'].transform('mean')

# 7F. TRAINER FEATURES
print("  7F. Trainer features...")
df['trainer_total'] = df.groupby('trainer_parsed')['trainer_parsed'].transform('count')
df['trainer_group_diversity'] = df.groupby('trainer_parsed')['group_name'].transform('nunique')
df['trainer_county_diversity'] = df.groupby('trainer_parsed')['county'].transform('nunique')
df['trainer_coop_rate'] = df.groupby('trainer_parsed')['belong_to_cooperative'].transform('mean')
df['trainer_female_rate'] = df.groupby('trainer_parsed')['gender'].transform(lambda x: (x == 'Female').mean())

# 7G. DEMOGRAPHIC INTERACTIONS
print("  7G. Demographic interactions...")
df['gender_age'] = df['gender'] + '_' + df['age']
df['gender_coop'] = df['gender'] + '_' + df['belong_to_cooperative'].astype(str)
df['age_coop'] = df['age'] + '_' + df['belong_to_cooperative'].astype(str)
df['registration_age'] = df['registration'] + '_' + df['age']
df['gender_registration'] = df['gender'] + '_' + df['registration']
df['gender_trainer'] = df['gender'] + '_' + df['trainer_parsed']
df['age_trainer'] = df['age'] + '_' + df['trainer_parsed']
df['gender_county'] = df['gender'] + '_' + df['county']
df['gender_topic'] = df['gender'] + '_' + df['primary_topic_cat']

# Binary features
df['is_ussd'] = (df['registration'] == 'Ussd').astype(int)
df['is_coop'] = df['belong_to_cooperative']
df['is_female'] = (df['gender'] == 'Female').astype(int)
df['is_young'] = (df['age'] == 'Below 35').astype(int)

# 7H. FARMER HISTORY INTERACTION FEATURES (DEEP)
print("  7H. Deep farmer history interactions...")
df['hist_sessions_x_topics'] = df['prior_session_count'] * df['topic_count']
df['hist_adoption_x_hastopic'] = df['prior_adoption_score'] * df['has_topic_trained_on']
df['hist_grp_adoption_x_farmer_adoption'] = df['prior_grp_adoption_score'] * df['prior_adoption_score']
df['hist_ever_adopted_x_hastopic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['farmer_is_repeat'] = (df['prior_session_count'] > 0).astype(int)
df['farmer_high_engagement'] = (df['prior_session_count'] >= 5).astype(int)
df['farmer_adopted_before'] = df['prior_any_adoption']

# V4 NEW: Combined adoption signals (multiplicative)
df['ussd_x_coop'] = df['is_ussd'] * df['is_coop']
df['ussd_x_has_topic'] = df['is_ussd'] * df['has_topic_trained_on']
df['coop_x_has_topic'] = df['is_coop'] * df['has_topic_trained_on']
df['triple_signal'] = df['is_ussd'] * df['is_coop'] * df['has_topic_trained_on']

# V4 NEW: Recency-weighted adoption (recent prior → stronger signal)
df['recency_weighted_adoption'] = df['prior_adoption_score'] / (df['days_since_last_prior'].clip(lower=1) / 100.0)
df.loc[df['days_since_last_prior'] < 0, 'recency_weighted_adoption'] = 0

# V4 NEW: Recent adoption signals
df['recent_adoption_x_hastopic'] = df['recent_adoption_score'] * df['has_topic_trained_on']
df['recent_any_x_coop'] = df['recent_any_adoption'] * df['is_coop']
df['recent_any_x_ussd'] = df['recent_any_adoption'] * df['is_ussd']

# V4 NEW: High-value adopter score (strong prior signals)
df['prior_ever_any_x_sessions'] = df['prior_any_adoption'] * np.log1p(df['prior_session_count'])
df['high_value_adopter'] = (
    df['prior_any_adoption'] * 5 + 
    df['is_ussd'] * 2 + 
    df['is_coop'] * 2 + 
    df['has_topic_trained_on'] * 3 +
    df['has_high_adopt_topic'] * 2 +
    (df['prior_session_count'] >= 8).astype(int) * 3
)

# V4 NEW: Trainer × farmer interactions
df['trainer_eff_x_has_topic'] = df['prior_trainer_effectiveness'] * df['has_topic_trained_on']
df['trainer_eff_x_coop'] = df['prior_trainer_effectiveness'] * df['is_coop']
df['trainer_eff_x_ussd'] = df['prior_trainer_effectiveness'] * df['is_ussd']
df['trainer_eff_x_farmer_adoption'] = df['prior_trainer_effectiveness'] * df['prior_adoption_score']

# V4 NEW: Group × trainer effectiveness
df['grp_adopt_x_trainer_eff'] = df['prior_grp_adoption_score'] * df['prior_trainer_effectiveness']
df['farmer_engaged_good_group'] = df['farmer_high_engagement'] * (df['prior_grp_adoption_score'] > 0).astype(int)

# V4 NEW: Last session interactions
df['last_120_x_hastopic'] = df['last_120'] * df['has_topic_trained_on']
df['last_120_x_coop'] = df['last_120'] * df['is_coop']

# V4 NEW: Trajectory-based features
df['improving_farmer'] = (df['traj_120'] > 0).astype(int)
df['traj_x_sessions'] = df['traj_120'] * np.log1p(df['prior_session_count'])

# 7I. AGGREGATION FEATURES
print("  7I. Aggregation features...")
for stat_col in ['group_size', 'group_coop_rate', 'prior_grp_adoption_score']:
    for agg in ['mean', 'std']:
        col_name = f'county_{stat_col}_{agg}'
        if stat_col in df.columns:
            df[col_name] = df.groupby('county')[stat_col].transform(agg)

df['ward_coop_rate'] = df.groupby('ward')['belong_to_cooperative'].transform('mean')
df['ward_female_rate'] = df.groupby('ward')['gender'].transform(lambda x: (x == 'Female').mean())
df['ward_group_count'] = df.groupby('ward')['group_name'].transform('nunique')

# V4 NEW: County-level aggregations
df['county_trainer_density'] = df.groupby('county')['trainer_parsed'].transform('nunique')
df['county_ussd_rate'] = df.groupby('county')['is_ussd'].transform('mean')
df['county_topic_diversity'] = df.groupby('county')['primary_topic_cat'].transform('nunique')
df['ward_ussd_rate'] = df.groupby('ward')['is_ussd'].transform('mean')

for col in df.columns:
    if col.endswith('_std'):
        df[col] = df[col].fillna(0)

# 7J. TARGET ENCODING (OOF) — same proven approach
print("  7J. Smoothed target encoding (OOF)...")
SMOOTHING = 10
te_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
           'primary_topic_cat', 'gender', 'age', 'registration',
           'county_subcounty', 'county_topic', 'ward_topic',
           'trainer_topic', 'gender_age', 'county_trainer']

train_data = df[train_idx].copy()
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target in TARGETS:
    global_mean = train_data[target].mean()
    for col in te_cols:
        te_col_name = f'te_{col}_{target}'
        df[te_col_name] = np.nan
        for fold_idx, (tr_idx_te, val_idx_te) in enumerate(skf_te.split(train_data, train_data[target])):
            fold_train = train_data.iloc[tr_idx_te]
            fold_val_indices = train_data.iloc[val_idx_te].index
            stats = fold_train.groupby(col)[target].agg(['sum', 'count'])
            smoothed = (stats['sum'] + SMOOTHING * global_mean) / (stats['count'] + SMOOTHING)
            df.loc[fold_val_indices, te_col_name] = train_data.loc[fold_val_indices, col].map(smoothed)
        stats_all = train_data.groupby(col)[target].agg(['sum', 'count'])
        smoothed_all = (stats_all['sum'] + SMOOTHING * global_mean) / (stats_all['count'] + SMOOTHING)
        test_mask = df['is_train'] == 0
        df.loc[test_mask, te_col_name] = df.loc[test_mask, col].map(smoothed_all)
        df[te_col_name] = df[te_col_name].fillna(global_mean)

print(f"  OOF Target encoding: {len(te_cols)} x {len(TARGETS)} = {len(te_cols)*len(TARGETS)} features")

# 7K. PRIOR-BASED TARGET ENCODING (using deduped Prior)
print("  7K. Prior-based target encoding...")
PRIOR_SMOOTH = 20
for target in TARGETS:
    prior_global = prior_dedup[target].mean()
    for col in ['group_name', 'ward', 'subcounty', 'county']:
        prior_stats = prior_dedup.groupby(col)[target].agg(['sum', 'count'])
        prior_smoothed = (prior_stats['sum'] + PRIOR_SMOOTH * prior_global) / (prior_stats['count'] + PRIOR_SMOOTH)
        df[f'prior_te_{col}_{target}'] = df[col].map(prior_smoothed).fillna(prior_global)
    
    # V4 NEW: Prior TE for trainer (with higher smoothing)
    trainer_stats = prior_dedup.groupby('trainer_parsed')[target].agg(['sum', 'count'])
    trainer_smoothed = (trainer_stats['sum'] + 50 * prior_global) / (trainer_stats['count'] + 50)
    df[f'prior_te_trainer_{target}'] = df['trainer_parsed'].map(trainer_smoothed).fillna(prior_global)

    # V4 NEW: Recent-only Prior TE (2025 data only)
    if len(recent_prior) > 0:
        recent_global = recent_prior[target].mean()
        for col in ['group_name', 'ward', 'county']:
            r_stats = recent_prior.groupby(col)[target].agg(['sum', 'count'])
            r_smoothed = (r_stats['sum'] + PRIOR_SMOOTH * recent_global) / (r_stats['count'] + PRIOR_SMOOTH)
            df[f'recent_te_{col}_{target}'] = df[col].map(r_smoothed).fillna(recent_global)

print(f"  Prior TE: comprehensive")

# ============================================================
# 8. PREPARE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Preparing features...")
print("=" * 70)

exclude_cols = ['ID', 'farmer_name', 'is_train', 'training_day', 'training_day_dt',
                'topics_list', 'topics_parsed', 'topic_cats', 'trainer'] + TARGETS

cat_cols_to_encode = [col for col in df.select_dtypes(include='object').columns
                      if col not in exclude_cols]

label_encoders = {}
for col in cat_cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

X_train = df.loc[train_idx, feature_cols].reset_index(drop=True)
X_test = df.loc[test_idx, feature_cols].reset_index(drop=True)
test_ids_ordered = df.loc[test_idx, 'ID'].values

y_train = {}
for t in TARGETS:
    y_train[t] = df.loc[train_idx, t].reset_index(drop=True).astype(int)

# Replace inf with NaN then fill
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
for c in X_train.columns:
    if X_train[c].isnull().any():
        fill_val = X_train[c].median() if X_train[c].dtype != 'object' else 0
        X_train[c] = X_train[c].fillna(fill_val if not np.isnan(fill_val) else 0)
        X_test[c] = X_test[c].fillna(fill_val if not np.isnan(fill_val) else 0)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Save masks for post-processing
zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0
has_prior_hist = df.loc[test_idx, 'prior_session_count'].values > 0

# Compute zero-group masks from BOTH Train and Prior (V4: prior-informed)
zero_group_masks = {}
for target in TARGETS:
    # Train-based zero groups (reliable: from same distribution)
    train_grp = df[train_idx].groupby('group_name').agg(
        n=(target, 'count'), rate=(target, 'mean'))
    train_zero_groups = set(train_grp[(train_grp['n'] >= 15) & (train_grp['rate'] == 0)].index)
    
    # Prior-based zero groups (wider coverage, 54% of test)
    prior_grp_stats = prior_dedup.groupby('group_name')[target].agg(['count', 'mean'])
    prior_zero_groups = set(prior_grp_stats[(prior_grp_stats['count'] >= 15) & (prior_grp_stats['mean'] == 0)].index)
    
    combined_zero = train_zero_groups | prior_zero_groups
    test_groups_arr = df.loc[test_idx, 'group_name'].values
    zero_group_masks[target] = np.isin(test_groups_arr, list(combined_zero))
    
    print(f"  Zero-group ({target}): train={len(train_zero_groups)}, prior={len(prior_zero_groups)}, combined={len(combined_zero)}, test hits={zero_group_masks[target].sum()}")

print(f"  Zero topic in test: {zero_topic_mask.sum()}")
print(f"  Has prior history in test: {has_prior_hist.sum()}")

elapsed = time.time() - start_time
print(f"  Feature engineering took {elapsed:.0f}s")

# ============================================================
# 9. OPTUNA LGB (80 trials, wider search + warm-start)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Optuna LGB tuning (80 trials, enhanced search)...")
print("=" * 70)

V4_CACHE = os.path.join(DATA_DIR, 'optuna_v4_cache.json')
V2_CACHE = os.path.join(DATA_DIR, 'optuna_v2_cache.json')

def sanitize_lgb_params(params):
    INT_KEYS = ['num_leaves', 'bagging_freq', 'min_child_samples', 'max_depth',
                'n_estimators', 'verbose', 'random_state', 'early_stopping_rounds']
    for k in INT_KEYS:
        if k in params and params[k] is not None:
            params[k] = int(params[k])
    return params

def optuna_lgb_objective_v4(trial, X, y, target_name):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.98),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 20.0, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'random_state': BASE_SEED,
        'n_estimators': 4000,
        'verbose': -1,
    }
    
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    params['scale_pos_weight'] = pos_weight
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**sanitize_lgb_params(params))
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        
        preds = model.predict_proba(X_val)[:, 1]
        score = competition_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

N_OPTUNA_TRIALS = 80
best_lgb_params = {}

if os.path.exists(V4_CACHE):
    print("  Found V4 cached Optuna results! Loading...")
    with open(V4_CACHE, 'r') as f:
        cached = json.load(f)
    for target in TARGETS:
        best_params = cached[target]
        INT_KEYS = ['num_leaves', 'bagging_freq', 'min_child_samples', 'max_depth',
                     'n_estimators', 'verbose', 'random_state', 'early_stopping_rounds']
        for k in INT_KEYS:
            if k in best_params: best_params[k] = int(best_params[k])
        for k in ['learning_rate', 'feature_fraction', 'bagging_fraction', 'reg_alpha', 
                   'reg_lambda', 'scale_pos_weight', 'min_gain_to_split']:
            if k in best_params: best_params[k] = float(best_params[k])
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        best_lgb_params[target] = best_params
        print(f"    {target}: leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")
else:
    # Load V2 cache for warm-start
    v2_cache = {}
    if os.path.exists(V2_CACHE):
        with open(V2_CACHE) as f:
            v2_cache = json.load(f)
        print("  Loaded V2 params for warm-start")
    
    for target in TARGETS:
        print(f"\n  Tuning {target} ({N_OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        
        # Warm-start from V2 best
        if target in v2_cache:
            v2_p = v2_cache[target]
            enqueue_params = {}
            for k in ['num_leaves', 'bagging_freq', 'min_child_samples', 'max_depth']:
                if k in v2_p: enqueue_params[k] = int(float(v2_p[k]))
            for k in ['learning_rate', 'feature_fraction', 'bagging_fraction', 'reg_alpha', 
                       'reg_lambda', 'min_gain_to_split']:
                if k in v2_p: enqueue_params[k] = float(v2_p[k])
            if 'min_gain_to_split' not in enqueue_params:
                enqueue_params['min_gain_to_split'] = 0.0
            study.enqueue_trial(enqueue_params)
            print(f"    Warm-started from V2 params")
        
        study.optimize(
            lambda trial: optuna_lgb_objective_v4(trial, X_train, y_train[target], target),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        
        best_params = study.best_params.copy()
        best_params['objective'] = 'binary'
        best_params['metric'] = 'binary_logloss'
        best_params['boosting_type'] = 'gbdt'
        best_params['n_estimators'] = 4000
        best_params['verbose'] = -1
        best_params['random_state'] = BASE_SEED
        
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        
        best_lgb_params[target] = best_params
        print(f"    Best score: {study.best_value:.6f}")
        print(f"    Best params: leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")
    
    cache_data = {}
    for target in TARGETS:
        cache_data[target] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                              for k, v in best_lgb_params[target].items()}
    with open(V4_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\n  Cached V4 Optuna results")

elapsed = time.time() - start_time
print(f"\n  Optuna took {elapsed:.0f}s total")

# ============================================================
# 10. LGB MULTI-SEED (20 seeds for maximum variance reduction)
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: LightGBM (Optuna-tuned, 20 seeds)...")
print("=" * 70)

LGB_SEEDS = [42, 123, 456, 789, 2025, 1337, 7777, 31415, 99, 555,
             888, 12345, 54321, 2026, 3141, 9876, 1111, 4242, 6789, 77777]

lgb_oof_preds = {t: np.zeros(len(X_train)) for t in TARGETS}
lgb_test_preds = {t: np.zeros(len(X_test)) for t in TARGETS}

for target in TARGETS:
    print(f"\n  Target: {target}")
    oof_accumulated = np.zeros(len(X_train))
    test_accumulated = np.zeros(len(X_test))
    
    for seed_idx, seed in enumerate(LGB_SEEDS):
        params = best_lgb_params[target].copy()
        params['random_state'] = seed
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(X_train))
        test_seed = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train[target])):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[target].iloc[tr_idx], y_train[target].iloc[val_idx]
            
            model = lgb.LGBMClassifier(**sanitize_lgb_params(params))
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
            
            oof_seed[val_idx] = model.predict_proba(X_val)[:, 1]
            test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        seed_score = competition_score(y_train[target], oof_seed)
        if seed_idx < 3 or seed_idx == len(LGB_SEEDS) - 1:
            print(f"    Seed {seed}: comp_score={seed_score:.6f}")
        oof_accumulated += oof_seed
        test_accumulated += test_seed
    
    lgb_oof_preds[target] = oof_accumulated / len(LGB_SEEDS)
    lgb_test_preds[target] = test_accumulated / len(LGB_SEEDS)
    
    final_score = competition_score(y_train[target], lgb_oof_preds[target])
    auc = roc_auc_score(y_train[target], lgb_oof_preds[target])
    ll = log_loss(y_train[target], lgb_oof_preds[target])
    print(f"  LGB Ensemble ({len(LGB_SEEDS)} seeds): AUC={auc:.6f}, LL={ll:.6f}, Comp={final_score:.6f}")

lgb_total = sum(competition_score(y_train[t], lgb_oof_preds[t]) for t in TARGETS)
print(f"\n  LGB TOTAL CV: {lgb_total:.6f}")

elapsed = time.time() - start_time
print(f"  Training took {elapsed:.0f}s total")

# ============================================================
# 11. CALIBRATION: PLATT vs ISOTONIC (pick best per target)
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Calibration (Platt vs Isotonic)...")
print("=" * 70)

calibrated_oof = {}
calibrated_test = {}
calibration_method = {}

for target in TARGETS:
    oof_blend = lgb_oof_preds[target].copy()
    test_blend = lgb_test_preds[target].copy()
    
    raw_score = competition_score(y_train[target], oof_blend)
    raw_ll = log_loss(y_train[target], oof_blend)
    
    # --- PLATT SCALING ---
    oof_logodds = np.log(np.clip(oof_blend, 1e-7, 1-1e-7) / (1 - np.clip(oof_blend, 1e-7, 1-1e-7)))
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(oof_logodds.reshape(-1, 1), y_train[target])
    platt_oof = platt.predict_proba(oof_logodds.reshape(-1, 1))[:, 1]
    platt_score = competition_score(y_train[target], platt_oof)
    
    test_logodds = np.log(np.clip(test_blend, 1e-7, 1-1e-7) / (1 - np.clip(test_blend, 1e-7, 1-1e-7)))
    platt_test = platt.predict_proba(test_logodds.reshape(-1, 1))[:, 1]
    
    # --- ISOTONIC (OOF cross-validated) ---
    skf_iso = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iso_oof = np.zeros(len(oof_blend))
    iso_test_folds = np.zeros(len(test_blend))
    
    for fold, (tr_idx_i, val_idx_i) in enumerate(skf_iso.split(np.arange(len(oof_blend)), y_train[target])):
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso.fit(oof_blend[tr_idx_i], y_train[target].iloc[tr_idx_i])
        iso_oof[val_idx_i] = iso.predict(oof_blend[val_idx_i])
        iso_test_folds += iso.predict(test_blend) / 5
    
    iso_score = competition_score(y_train[target], iso_oof)
    
    print(f"\n  {target}:")
    print(f"    Raw:      LL={raw_ll:.6f}, Comp={raw_score:.6f}")
    print(f"    Platt:    Comp={platt_score:.6f}")
    print(f"    Isotonic: Comp={iso_score:.6f}")
    
    best_method = 'raw'
    best_score = raw_score
    best_oof = oof_blend
    best_test = test_blend
    
    if platt_score > best_score:
        best_method = 'platt'
        best_score = platt_score
        best_oof = platt_oof
        best_test = platt_test
    
    if iso_score > best_score:
        best_method = 'isotonic'
        best_score = iso_score
        best_oof = iso_oof
        best_test = iso_test_folds
    
    calibrated_oof[target] = best_oof
    calibrated_test[target] = best_test
    calibration_method[target] = best_method
    print(f"    WINNER: {best_method} ({best_score:.6f})")

cal_total = sum(competition_score(y_train[t], calibrated_oof[t]) for t in TARGETS)
print(f"\n  Calibrated TOTAL CV: {cal_total:.6f}")

# ============================================================
# 12. BAYESIAN CALIBRATION WITH PRIOR HISTORY
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Bayesian calibration with Prior farmer history...")
print("=" * 70)

farmer_prior_data = prior_dedup.groupby('farmer_name').agg(
    n_sessions=('training_day', 'count'),
    k_07=('adopted_within_07_days', 'sum'),
    k_90=('adopted_within_90_days', 'sum'),
    k_120=('adopted_within_120_days', 'sum'),
).reset_index()

test_farmer_names = df.loc[test_idx, 'farmer_name'].values
farmer_prior_dict = farmer_prior_data.set_index('farmer_name').to_dict('index')

bayesian_posteriors = {}
for target in TARGETS:
    train_rate = y_train[target].mean()
    PRIOR_STRENGTH = 5
    alpha_0 = train_rate * PRIOR_STRENGTH
    beta_0 = (1 - train_rate) * PRIOR_STRENGTH
    
    k_col = {'adopted_within_07_days': 'k_07',
             'adopted_within_90_days': 'k_90',
             'adopted_within_120_days': 'k_120'}[target]
    
    posteriors = np.full(len(test_farmer_names), train_rate)
    for i, fname in enumerate(test_farmer_names):
        if fname in farmer_prior_dict:
            data = farmer_prior_dict[fname]
            n = data['n_sessions']
            k = data[k_col]
            posteriors[i] = (alpha_0 + k) / (alpha_0 + beta_0 + n)
    
    bayesian_posteriors[target] = posteriors

# Optimize on OOF
train_farmer_names = df.loc[train_idx, 'farmer_name'].values
has_train_hist = np.array([fn in farmer_prior_dict for fn in train_farmer_names])

bayesian_oof = {}
optimal_bayes_weights = {}

for target in TARGETS:
    train_rate = y_train[target].mean()
    alpha_0 = train_rate * 5
    beta_0 = (1 - train_rate) * 5
    k_col = {'adopted_within_07_days': 'k_07',
             'adopted_within_90_days': 'k_90',
             'adopted_within_120_days': 'k_120'}[target]
    
    oof_posteriors = np.full(len(train_farmer_names), train_rate)
    for i, fname in enumerate(train_farmer_names):
        if fname in farmer_prior_dict:
            data = farmer_prior_dict[fname]
            oof_posteriors[i] = (alpha_0 + data[k_col]) / (alpha_0 + beta_0 + data['n_sessions'])
    bayesian_oof[target] = oof_posteriors
    
    model_oof = calibrated_oof[target]
    best_w_m = 1.0
    best_score = competition_score(y_train[target], model_oof)
    
    for w_m in np.arange(0.50, 1.01, 0.01):
        blended = model_oof.copy()
        blended[has_train_hist] = w_m * model_oof[has_train_hist] + (1 - w_m) * oof_posteriors[has_train_hist]
        score = competition_score(y_train[target], blended)
        if score > best_score:
            best_score = score
            best_w_m = w_m
    
    optimal_bayes_weights[target] = best_w_m
    print(f"  {target}: model_weight={best_w_m:.2f}, score={best_score:.6f}")

# ============================================================
# 13. GENERATE PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: Generating predictions...")
print("=" * 70)

final_preds_raw = {}
final_preds_bayesian = {}
lgb_only_preds = {}

for target in TARGETS:
    # Raw calibrated
    final_preds_raw[target] = calibrated_test[target].copy()
    
    # Bayesian blend
    w_m = optimal_bayes_weights[target]
    blended = calibrated_test[target].copy()
    blended[has_prior_hist] = w_m * calibrated_test[target][has_prior_hist] + (1 - w_m) * bayesian_posteriors[target][has_prior_hist]
    final_preds_bayesian[target] = blended
    
    # LGB raw
    lgb_only_preds[target] = lgb_test_preds[target].copy()

# ============================================================
# 14. POST-PROCESSING (ENHANCED)
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: Post-processing...")
print("=" * 70)

def post_process(preds_dict, label):
    for target in TARGETS:
        preds_dict[target] = np.clip(preds_dict[target], 0.001, 0.999)
    
    # Enforce monotonicity: 120d >= 90d >= 7d
    preds_dict['adopted_within_90_days'] = np.maximum(
        preds_dict['adopted_within_07_days'], preds_dict['adopted_within_90_days'])
    preds_dict['adopted_within_120_days'] = np.maximum(
        preds_dict['adopted_within_90_days'], preds_dict['adopted_within_120_days'])
    
    # Zero-topic rule (PERFECT: has_topic=0 → 0 adoptions in all train data)
    for target in TARGETS:
        preds_dict[target][zero_topic_mask] = 0.001
    
    # Zero-group rule (V4: uses BOTH train and prior zero-groups)
    for target in TARGETS:
        zmask = zero_group_masks[target]
        if zmask.sum() > 0:
            preds_dict[target][zmask] = np.minimum(preds_dict[target][zmask], 0.003)
            print(f"  {label} {target}: capped {zmask.sum()} zero-group rows")
    
    print(f"  {label}: Mean preds: " +
          ", ".join(f"{t.split('_')[2]}d={preds_dict[t].mean():.5f}" for t in TARGETS))
    
    return preds_dict

final_preds_raw = post_process(final_preds_raw, "Raw")
final_preds_bayesian = post_process(final_preds_bayesian, "Bayesian")
lgb_only_preds = post_process(lgb_only_preds, "LGB-Only")

# ============================================================
# 15. CREATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 15: Creating submissions...")
print("=" * 70)

def create_dual_submission(preds, test_ids, filename):
    sub = pd.DataFrame({'ID': test_ids})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        raw = preds[target].copy()
        ranks = pd.Series(raw).rank(pct=True)
        auc_preds = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
        auc_preds[zero_topic_mask] = 0.001
        sub[auc_col] = auc_preds
        sub[ll_col] = raw
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    assert len(sub) == len(ss)
    assert sub.isnull().sum().sum() == 0
    sub.to_csv(filename, index=False)
    print(f"  SAVED: {filename}")

def create_standard_submission(preds, test_ids, filename):
    sub = pd.DataFrame({'ID': test_ids})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        sub[auc_col] = preds[target]
        sub[ll_col] = preds[target]
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    assert len(sub) == len(ss)
    assert sub.isnull().sum().sum() == 0
    sub.to_csv(filename, index=False)
    print(f"  SAVED: {filename}")

# Primary submissions
create_dual_submission(final_preds_raw, test_ids_ordered, 'sub_V4_A_calibrated_dual.csv')
create_dual_submission(final_preds_bayesian, test_ids_ordered, 'sub_V4_B_bayesian_dual.csv')
create_dual_submission(lgb_only_preds, test_ids_ordered, 'sub_V4_C_lgb_dual.csv')
create_standard_submission(final_preds_bayesian, test_ids_ordered, 'sub_V4_D_bayesian_standard.csv')

# Blend with V2 best (sub_ULT_C_bayesian_dual.csv)
v2_best_path = os.path.join(DATA_DIR, 'sub_ULT_C_bayesian_dual.csv')
if os.path.exists(v2_best_path):
    v2_sub = pd.read_csv(v2_best_path)
    for blend_ratio in [0.3, 0.5, 0.7]:
        blend_sub = v2_sub.copy()
        for col in v2_sub.columns:
            if col != 'ID':
                blend_sub[col] = blend_ratio * v2_sub[col] + (1 - blend_ratio) * \
                    pd.read_csv('sub_V4_B_bayesian_dual.csv').set_index('ID').loc[v2_sub['ID'], col].values
        fname = f'sub_V4_E_blend{int(blend_ratio*100)}v2_{int((1-blend_ratio)*100)}v4.csv'
        blend_sub.to_csv(fname, index=False)
        print(f"  SAVED: {fname}")

# ============================================================
# 16. FINAL CV SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("STEP 16: FINAL CV SUMMARY")
print("=" * 70)

print("\nPer-target scores (OOF):")
for target in TARGETS:
    lgb_s = competition_score(y_train[target], lgb_oof_preds[target])
    cal_s = competition_score(y_train[target], calibrated_oof[target])
    print(f"  {target}: LGB={lgb_s:.6f}, Calibrated={cal_s:.6f}")

lgb_total = sum(competition_score(y_train[t], lgb_oof_preds[t]) for t in TARGETS)
cal_total = sum(competition_score(y_train[t], calibrated_oof[t]) for t in TARGETS)

bayes_total = 0
for target in TARGETS:
    w_m = optimal_bayes_weights[target]
    blended = calibrated_oof[target].copy()
    blended[has_train_hist] = w_m * calibrated_oof[target][has_train_hist] + (1 - w_m) * bayesian_oof[target][has_train_hist]
    bayes_total += competition_score(y_train[target], blended)

print(f"\nTotal competition scores (OOF):")
print(f"  LGB-only (20 seeds):   {lgb_total:.6f}")
print(f"  Calibrated:            {cal_total:.6f}")
print(f"  +Bayesian:             {bayes_total:.6f}")
print(f"  --")
print(f"  V2 best CV:            ~2.933 (LB 0.93333)")

print(f"\nCalibration methods:")
for target in TARGETS:
    print(f"  {target}: {calibration_method[target]}")

print(f"\nBayesian blend weights:")
for target in TARGETS:
    print(f"  {target}: model={optimal_bayes_weights[target]:.2f}")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"TOTAL PIPELINE TIME: {total_time/60:.1f} minutes")
print(f"{'='*70}")

print(f"\nSUBMISSIONS:")
print(f"  sub_V4_A_calibrated_dual.csv  - DUAL calibrated LGB")
print(f"  sub_V4_B_bayesian_dual.csv    - Bayesian + DUAL (RECOMMENDED)")
print(f"  sub_V4_C_lgb_dual.csv         - LGB-only DUAL (safe)")
print(f"  sub_V4_D_bayesian_standard.csv - Standard Bayesian")
print(f"  sub_V4_E_blend*.csv           - V2+V4 blends")
print(f"\nSUBMIT ORDER: sub_V4_B_bayesian_dual.csv first!")
print(f"If V4_B beats V2: try V4_A. If not: try V4_E blends.")
