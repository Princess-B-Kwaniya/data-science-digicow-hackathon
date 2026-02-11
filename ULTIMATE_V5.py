"""
ULTIMATE_V5.py - Deep Feature Engineering + Chi-Square Selection + Clean Data
================================================================
BASED ON V2 (scored 0.93333 on LB, CV ~2.933):
  - SAME backbone: LGB 10seeds + XGB 5seeds + Stack + DUAL
  - SAME Optuna warm-start from V2 cache
  - SAME calibration pipeline (Platt vs Isotonic)

NEW IN V5:
  1. PRIOR DATA CLEANING: Deduplicate 40K+ same-farmer-same-day rows
  2. TOPIC NORMALIZATION: Merge near-duplicate topic strings  
  3. CHI-SQUARE FEATURE SELECTION: Score all features, drop bottom N
  4. MUTUAL INFORMATION: Cross-validate with chi-square
  5. DEEPER FEATURES (from data analysis):
     - Training sequence number (nth session for farmer)
     - Days gap from previous training
     - Group size buckets (small/medium/large)
     - Trainer daily load pattern
     - Topic-level adoption rates (smoothed from Prior)
     - First adopter effect  
     - Trainer-county combination rates
     - Month-DOW interaction
     - Recency decay adoption
     - Prior temporal features (last session features)
     - County-month seasonality
  6. FEATURE IMPORTANCE PRUNING: Drop features with zero/near-zero importance
  7. LEAKAGE CHECKS: Explicit validation
  8. PRIOR-INFORMED ZERO-GROUP RULES (from Prior + Train)

RULES: Train on Train ONLY. Prior ONLY as feature source. No target leakage.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import chi2, mutual_info_classif
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

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("ULTIMATE V5 - Deep Features + Chi-Square Selection")
print("=" * 70)
print("\nSTEP 1: Loading data...")

train_df = pd.read_csv('Train.csv')
test_df  = pd.read_csv('Test.csv')
prior_df = pd.read_csv('Prior.csv')
ss       = pd.read_csv('SampleSubmission.csv')

print(f"  Train: {train_df.shape}, Test: {test_df.shape}, Prior: {prior_df.shape}")

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
SS_COLS = list(ss.columns)
TARGET_TO_SS = {
    'adopted_within_07_days':  ('Target_07_AUC',  'Target_07_LogLoss'),
    'adopted_within_90_days':  ('Target_90_AUC',  'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

for t in TARGETS:
    print(f"  Train {t}: {train_df[t].mean():.4f} ({train_df[t].sum()}/{len(train_df)})")

# ============================================================
# 2. DATA CLEANING & PARSING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Data cleaning, dedup, topic normalization...")
print("=" * 70)

# 2A. PRIOR DEDUPLICATION
# 40,574 rows are same farmer + same day → keep the one with most info
prior_df['training_day_dt'] = pd.to_datetime(prior_df['training_day'])

# Sort by most informative (has_topic=1 first, then latest ID)
prior_df = prior_df.sort_values(
    ['farmer_name', 'training_day_dt', 'has_topic_trained_on', 'ID'],
    ascending=[True, True, False, False]
)
prior_orig_len = len(prior_df)
prior_df = prior_df.drop_duplicates(subset=['farmer_name', 'training_day_dt'], keep='first')
print(f"  Prior dedup: {prior_orig_len} → {len(prior_df)} rows (removed {prior_orig_len - len(prior_df)})")

# 2B. TOPIC NORMALIZATION MAP
TOPIC_NORMALIZE = {
    'herd health. management': 'herd health management',
    'herd health.management': 'herd health management',
    'herd health management': 'herd health management',
    'dairy cow nutrition': 'dairy cow nutrition',
    'dairy cow husbandry': 'dairy cow husbandry',
    'poultry management': 'poultry management',
    'calf management': 'calf management',
    'calf rearing': 'calf rearing',
    'milking & hygiene': 'milking and hygiene',
    'milking and hygiene': 'milking and hygiene',
    'milking &amp; hygiene': 'milking and hygiene',
}

def normalize_topic(t):
    t_lower = str(t).strip().lower()
    return TOPIC_NORMALIZE.get(t_lower, t_lower)

# 2C. PARSE FORMATS
def parse_topics_nested(s):
    try:
        parsed = ast.literal_eval(s)
        all_topics = []
        for session in parsed:
            if isinstance(session, list):
                all_topics.extend(session)
            else:
                all_topics.append(str(session))
        return list(set(normalize_topic(t) for t in all_topics))
    except:
        return []

def parse_topics_flat(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return list(set(normalize_topic(t) for t in parsed))
        return [normalize_topic(parsed)]
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

# Parse dates
train_df['training_day_dt'] = pd.to_datetime(train_df['training_day'])
test_df['training_day_dt'] = pd.to_datetime(test_df['training_day'])

print(f"  Parsing and normalization complete.")

# ============================================================
# 3. BUILD FARMER HISTORY FROM PRIOR (cleaned)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building farmer history from Prior (cleaned)...")
print("=" * 70)

farmer_hist = prior_df.groupby('farmer_name').agg(
    prior_session_count=('ID', 'count'),
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
).reset_index()

# Topic diversity
topic_diversity = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))
).reset_index()
topic_diversity.columns = ['farmer_name', 'prior_unique_topics']
farmer_hist = farmer_hist.merge(topic_diversity, on='farmer_name', how='left')

# Date features  
date_feats = prior_df.groupby('farmer_name')['training_day_dt'].agg(
    prior_first_date='min', prior_last_date='max',
).reset_index()
date_feats['prior_training_span_days'] = (date_feats['prior_last_date'] - date_feats['prior_first_date']).dt.days
farmer_hist = farmer_hist.merge(date_feats[['farmer_name', 'prior_training_span_days']], on='farmer_name', how='left')

# Derived farmer features
farmer_hist['prior_ever_adopted_07'] = (farmer_hist['prior_07_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_90'] = (farmer_hist['prior_90_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_120'] = (farmer_hist['prior_120_adopted'] > 0).astype(int)
farmer_hist['prior_any_adoption'] = ((farmer_hist['prior_07_adopted'] + farmer_hist['prior_90_adopted'] + farmer_hist['prior_120_adopted']) > 0).astype(int)
farmer_hist['prior_adoption_score'] = (
    farmer_hist['prior_07_rate'] * 3 + farmer_hist['prior_90_rate'] * 2 + farmer_hist['prior_120_rate'] * 1
) / 6

# Engagement intensity (sessions per month)
farmer_hist['prior_engagement_intensity'] = farmer_hist['prior_session_count'] / (farmer_hist['prior_training_span_days'].clip(lower=1) / 30.0)
# Consistency
farmer_hist['prior_adoption_consistency'] = farmer_hist[['prior_07_rate','prior_90_rate','prior_120_rate']].std(axis=1)
# Loyalty
farmer_hist['prior_is_loyal'] = (farmer_hist['prior_unique_groups'] == 1).astype(int)
# High engagement flag
farmer_hist['prior_high_sessions'] = (farmer_hist['prior_session_count'] >= 10).astype(int)

# --- NEW V5 FARMER FEATURES ---
# 3A. Recency-weighted adoption rates (recent sessions matter more)
prior_sorted = prior_df.sort_values(['farmer_name', 'training_day_dt'])
prior_sorted['session_rank'] = prior_sorted.groupby('farmer_name').cumcount()
prior_sorted['session_total'] = prior_sorted.groupby('farmer_name')['ID'].transform('count')
prior_sorted['recency_weight'] = (prior_sorted['session_rank'] + 1) / prior_sorted['session_total']

for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    weighted_rates = prior_sorted.groupby('farmer_name').apply(
        lambda x: np.average(x[target], weights=x['recency_weight']) if len(x) > 0 else 0
    ).reset_index()
    weighted_rates.columns = ['farmer_name', f'prior_{col_short}_recency_rate']
    farmer_hist = farmer_hist.merge(weighted_rates, on='farmer_name', how='left')

# 3B. Last session features
last_sessions = prior_sorted.groupby('farmer_name').last().reset_index()
farmer_hist = farmer_hist.merge(
    last_sessions[['farmer_name', 'has_topic_trained_on', 'belong_to_cooperative']].rename(
        columns={'has_topic_trained_on': 'prior_last_has_topic', 'belong_to_cooperative': 'prior_last_coop'}),
    on='farmer_name', how='left')

# 3C. Temporal decay adoption rate (exponential decay)
ref_date = prior_df['training_day_dt'].max()
prior_sorted['days_from_ref'] = (ref_date - prior_sorted['training_day_dt']).dt.days
prior_sorted['decay_weight'] = np.exp(-prior_sorted['days_from_ref'] / 365.0)

for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    decay_rates = prior_sorted.groupby('farmer_name').apply(
        lambda x: np.average(x[target], weights=x['decay_weight']) if len(x) > 0 else 0
    ).reset_index()
    decay_rates.columns = ['farmer_name', f'prior_{col_short}_decay_rate']
    farmer_hist = farmer_hist.merge(decay_rates, on='farmer_name', how='left')

# 3D. Training sequence features
seq_feats = prior_sorted.groupby('farmer_name').agg(
    prior_last_session_date=('training_day_dt', 'max'),
    prior_first_session_date=('training_day_dt', 'min'),
).reset_index()

# Days between consecutive sessions (mean gap)
def calc_mean_gap(group):
    dates = group['training_day_dt'].sort_values()
    if len(dates) < 2:
        return 0
    gaps = dates.diff().dt.days.dropna()
    return gaps.mean()

day_gaps = prior_sorted.groupby('farmer_name').apply(calc_mean_gap).reset_index()
day_gaps.columns = ['farmer_name', 'prior_mean_day_gap']
farmer_hist = farmer_hist.merge(day_gaps, on='farmer_name', how='left')

# 3E. First adopter effect (did farmer adopt in their FIRST session?)
first_sessions = prior_sorted.groupby('farmer_name').first().reset_index()
for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    farmer_hist = farmer_hist.merge(
        first_sessions[['farmer_name', target]].rename(columns={target: f'prior_first_session_{col_short}'}),
        on='farmer_name', how='left')

# 3F. Adoption improvement (last session vs first session)  
for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    farmer_hist[f'prior_adoption_improvement_{col_short}'] = (
        farmer_hist.get(f'prior_{col_short}_recency_rate', 0) - farmer_hist[f'prior_{col_short}_rate']
    )

test_coverage = test_df['farmer_name'].isin(farmer_hist['farmer_name']).mean()
print(f"  {len(farmer_hist)} farmers, {farmer_hist.shape[1]-1} features, test coverage: {test_coverage:.1%}")

# ============================================================
# 4. BUILD GROUP HISTORY FROM PRIOR (cleaned)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Building group history from Prior...")
print("=" * 70)

group_hist = prior_df.groupby('group_name').agg(
    prior_grp_size=('ID', 'count'),
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

# V5 NEW: Group size buckets from prior
group_hist['prior_grp_size_bucket'] = pd.cut(group_hist['prior_grp_size'], 
                                              bins=[0, 5, 15, 50, 200, np.inf], 
                                              labels=[0, 1, 2, 3, 4]).astype(float)
# V5 NEW: Group adopter density (what fraction of unique farmers adopted?)
grp_adopters = prior_df.groupby('group_name').apply(
    lambda x: x.groupby('farmer_name')['adopted_within_120_days'].max().mean()
).reset_index()
grp_adopters.columns = ['group_name', 'prior_grp_adopter_density']
group_hist = group_hist.merge(grp_adopters, on='group_name', how='left')

print(f"  {len(group_hist)} groups, {group_hist.shape[1]-1} features")

# ============================================================
# 5. BUILD GEO HISTORY FROM PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Geographic history from Prior...")
print("=" * 70)

for geo_col in ['ward', 'subcounty', 'county']:
    geo_hist = prior_df.groupby(geo_col).agg(
        **{f'prior_{geo_col}_size': ('ID', 'count'),
           f'prior_{geo_col}_07_rate': ('adopted_within_07_days', 'mean'),
           f'prior_{geo_col}_90_rate': ('adopted_within_90_days', 'mean'),
           f'prior_{geo_col}_120_rate': ('adopted_within_120_days', 'mean'),
           f'prior_{geo_col}_coop_rate': ('belong_to_cooperative', 'mean'),
           f'prior_{geo_col}_has_topic_rate': ('has_topic_trained_on', 'mean'),
           }
    ).reset_index()
    train_df = train_df.merge(geo_hist, on=geo_col, how='left')
    test_df = test_df.merge(geo_hist, on=geo_col, how='left')
    print(f"  {geo_col}: {len(geo_hist)} values")

# ============================================================
# 5B. TRAINER EFFECTIVENESS FROM PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 5B: Trainer effectiveness from Prior...")
print("=" * 70)

trainer_eff = prior_df.groupby('trainer_parsed').agg(
    prior_trainer_total=('ID', 'count'),
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

# Smoothed rates (avoid overfitting small trainers)
TRAINER_SMOOTH = 50
for rate_col in ['prior_trainer_07_rate', 'prior_trainer_90_rate', 'prior_trainer_120_rate']:
    target_name = rate_col.replace('prior_trainer_', '').replace('_rate', '')
    global_rate = prior_df[f'adopted_within_{target_name}_days'].mean()
    trainer_eff[f'{rate_col}_smoothed'] = (
        trainer_eff[rate_col] * trainer_eff['prior_trainer_total'] + global_rate * TRAINER_SMOOTH
    ) / (trainer_eff['prior_trainer_total'] + TRAINER_SMOOTH)

# V5 NEW: Trainer daily load (avg sessions per unique day)
trainer_days = prior_df.groupby('trainer_parsed')['training_day_dt'].nunique().reset_index()
trainer_days.columns = ['trainer_parsed', 'prior_trainer_active_days']
trainer_eff = trainer_eff.merge(trainer_days, on='trainer_parsed', how='left')
trainer_eff['prior_trainer_daily_load'] = trainer_eff['prior_trainer_total'] / trainer_eff['prior_trainer_active_days'].clip(lower=1)

# V5 NEW: Trainer county diversity
trainer_county_div = prior_df.groupby('trainer_parsed')['county'].nunique().reset_index()
trainer_county_div.columns = ['trainer_parsed', 'prior_trainer_county_diversity']
trainer_eff = trainer_eff.merge(trainer_county_div, on='trainer_parsed', how='left')

train_df = train_df.merge(trainer_eff, on='trainer_parsed', how='left')
test_df = test_df.merge(trainer_eff, on='trainer_parsed', how='left')
print(f"  {len(trainer_eff)} trainers, {trainer_eff.shape[1]-1} features")

# ============================================================
# 5C. NEW: TOPIC-LEVEL ADOPTION RATES FROM PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 5C: Topic-level adoption rates from Prior (NEW)...")
print("=" * 70)

# Build topic-specific adoption rates using exploded Prior data
topic_rows = []
for _, row in prior_df.iterrows():
    for topic in row['topics_parsed']:
        topic_rows.append({
            'topic': topic,
            'adopted_within_07_days': row['adopted_within_07_days'],
            'adopted_within_90_days': row['adopted_within_90_days'],
            'adopted_within_120_days': row['adopted_within_120_days'],
        })

if topic_rows:
    topic_df = pd.DataFrame(topic_rows)
    TOPIC_SMOOTH = 30
    topic_rates = topic_df.groupby('topic').agg(
        topic_n=('adopted_within_07_days', 'count'),
        topic_07_rate=('adopted_within_07_days', 'mean'),
        topic_90_rate=('adopted_within_90_days', 'mean'),
        topic_120_rate=('adopted_within_120_days', 'mean'),
    ).reset_index()
    
    # Smooth with global rates
    for col in ['topic_07_rate', 'topic_90_rate', 'topic_120_rate']:
        tgt = col.replace('topic_', 'adopted_within_').replace('_rate', '_days')
        g = prior_df[tgt].mean()
        topic_rates[f'{col}_smoothed'] = (
            topic_rates[col] * topic_rates['topic_n'] + g * TOPIC_SMOOTH
        ) / (topic_rates['topic_n'] + TOPIC_SMOOTH)
    
    topic_rate_dict = topic_rates.set_index('topic').to_dict('index')
    print(f"  {len(topic_rates)} unique topics with adoption rates")
else:
    topic_rate_dict = {}
    print("  No topic rows found")

# ============================================================
# 5D. NEW: TRAINER-COUNTY COMBINATION RATES FROM PRIOR
# ============================================================
print("\nSTEP 5D: Trainer-county combination rates...")

trainer_county_rates = prior_df.groupby(['trainer_parsed', 'county']).agg(
    tc_size=('ID', 'count'),
    tc_07_rate=('adopted_within_07_days', 'mean'),
    tc_90_rate=('adopted_within_90_days', 'mean'),
    tc_120_rate=('adopted_within_120_days', 'mean'),
).reset_index()

TC_SMOOTH = 20
for col in ['tc_07_rate', 'tc_90_rate', 'tc_120_rate']:
    tgt = col.replace('tc_', 'adopted_within_').replace('_rate', '_days')
    g = prior_df[tgt].mean()
    trainer_county_rates[f'{col}_smoothed'] = (
        trainer_county_rates[col] * trainer_county_rates['tc_size'] + g * TC_SMOOTH
    ) / (trainer_county_rates['tc_size'] + TC_SMOOTH)

train_df = train_df.merge(trainer_county_rates[['trainer_parsed', 'county', 
    'tc_size', 'tc_07_rate_smoothed', 'tc_90_rate_smoothed', 'tc_120_rate_smoothed']], 
    on=['trainer_parsed', 'county'], how='left')
test_df = test_df.merge(trainer_county_rates[['trainer_parsed', 'county',
    'tc_size', 'tc_07_rate_smoothed', 'tc_90_rate_smoothed', 'tc_120_rate_smoothed']], 
    on=['trainer_parsed', 'county'], how='left')
print(f"  {len(trainer_county_rates)} trainer-county combos")

# ============================================================
# 6. MERGE HISTORY
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Merging history features...")
print("=" * 70)

train_df = train_df.merge(farmer_hist, on='farmer_name', how='left')
test_df = test_df.merge(farmer_hist, on='farmer_name', how='left')
train_df = train_df.merge(group_hist, on='group_name', how='left')
test_df = test_df.merge(group_hist, on='group_name', how='left')

hist_cols = [c for c in farmer_hist.columns if c != 'farmer_name'] + \
            [c for c in group_hist.columns if c != 'group_name'] + \
            [c for c in trainer_eff.columns if c != 'trainer_parsed']
for c in hist_cols:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

# Fill trainer-county NaN
for c in ['tc_size', 'tc_07_rate_smoothed', 'tc_90_rate_smoothed', 'tc_120_rate_smoothed']:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

print(f"  Train with history: {(train_df['prior_session_count'] > 0).sum()}/{len(train_df)}")
print(f"  Test with history: {(test_df['prior_session_count'] > 0).sum()}/{len(test_df)}")

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Feature Engineering...")
print("=" * 70)

train_df['is_train'] = 1
test_df['is_train'] = 0
for t in TARGETS:
    if t not in test_df.columns:
        test_df[t] = np.nan

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_idx = df['is_train'] == 1
test_idx = df['is_train'] == 0

# --- 7A. TEMPORAL ---
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
df['is_sunday'] = (df['train_dayofweek'] == 6).astype(int)  # V5: Sunday has 2.8x lift
df['is_month_start'] = df['training_day_dt'].dt.is_month_start.astype(int)
df['is_month_end'] = df['training_day_dt'].dt.is_month_end.astype(int)
df['days_since_epoch'] = (df['training_day_dt'] - pd.Timestamp('2024-01-01')).dt.days
df['month_sin'] = np.sin(2 * np.pi * df['train_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['train_month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['train_dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['train_dayofweek'] / 7)

def get_season(m):
    if m in [3, 4, 5]: return 0    # Long rains
    elif m in [6, 7, 8]: return 1   # Cool dry
    elif m in [10, 11]: return 2    # Short rains
    elif m in [12, 1, 2]: return 3  # Hot dry
    else: return 4
df['season'] = df['train_month'].apply(get_season)

# V5 NEW: Month-DOW interaction (certain combos have higher adoption)
df['month_dow'] = df['train_month'] * 10 + df['train_dayofweek']
df['is_high_adoption_month'] = df['train_month'].isin([3, 9, 11]).astype(int)

# V5 NEW: Days since last prior session (recency signal)
prior_last_dates = prior_df.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last_dates.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last_dates else -1, axis=1)
df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)

# V5 NEW: Training sequence number for this farmer (1st, 2nd, 3rd training in current data)
# Count how many prior sessions they had
prior_session_counts = prior_df.groupby('farmer_name').size().to_dict()
df['farmer_total_prior_sessions'] = df['farmer_name'].map(prior_session_counts).fillna(0)
df['training_sequence_num'] = df['farmer_total_prior_sessions'] + 1  # This training is next in sequence

# --- 7B. TOPIC FEATURES ---
print("  7B. Topic features...")
df['topic_count'] = df['topics_parsed'].apply(len)
df['is_multi_topic'] = (df['topic_count'] > 1).astype(int)
df['is_single_topic'] = (df['topic_count'] == 1).astype(int)  # V5: single topic has 4x lift

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

# V5 NEW: Topic-level adoption rate features (from Prior)
def get_topic_rate(topics, rate_key, default=0):
    rates = []
    for t in topics:
        if t in topic_rate_dict and rate_key in topic_rate_dict[t]:
            rates.append(topic_rate_dict[t][rate_key])
    return np.mean(rates) if rates else default

for suffix in ['07', '90', '120']:
    df[f'topic_adoption_rate_{suffix}'] = df['topics_parsed'].apply(
        lambda x: get_topic_rate(x, f'topic_{suffix}_rate_smoothed'))

# --- 7C. GEOGRAPHIC INTERACTIONS ---
print("  7C. Geographic interactions...")
df['county_subcounty'] = df['county'] + '_' + df['subcounty']
df['subcounty_ward'] = df['subcounty'] + '_' + df['ward']
df['county_ward'] = df['county'] + '_' + df['ward']
df['county_trainer'] = df['county'] + '_' + df['trainer_parsed']
df['ward_trainer'] = df['ward'] + '_' + df['trainer_parsed']
df['county_topic'] = df['county'] + '_' + df['primary_topic_cat']
df['ward_topic'] = df['ward'] + '_' + df['primary_topic_cat']
df['trainer_topic'] = df['trainer_parsed'] + '_' + df['primary_topic_cat']

# --- 7D. FREQUENCY ENCODING ---
print("  7D. Frequency encoding...")
freq_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
             'primary_topic_cat', 'county_subcounty', 'county_topic',
             'ward_topic', 'trainer_topic', 'county_trainer']
for col in freq_cols:
    df[f'{col}_freq'] = df.groupby(col)[col].transform('count')

# --- 7E. GROUP FEATURES ---
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

# V5 NEW: Group size buckets in current data
df['group_size_bucket'] = pd.cut(df['group_size'], bins=[0, 3, 10, 30, 100, np.inf], 
                                  labels=[0, 1, 2, 3, 4]).astype(float)

# --- 7F. TRAINER FEATURES ---
print("  7F. Trainer features...")
df['trainer_total'] = df.groupby('trainer_parsed')['trainer_parsed'].transform('count')
df['trainer_group_diversity'] = df.groupby('trainer_parsed')['group_name'].transform('nunique')
df['trainer_county_diversity'] = df.groupby('trainer_parsed')['county'].transform('nunique')
df['trainer_coop_rate'] = df.groupby('trainer_parsed')['belong_to_cooperative'].transform('mean')
df['trainer_female_rate'] = df.groupby('trainer_parsed')['gender'].transform(lambda x: (x == 'Female').mean())

# --- 7G. DEMOGRAPHIC INTERACTIONS ---
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

# --- 7H. FARMER HISTORY INTERACTIONS ---
print("  7H. History interactions...")
df['hist_sessions_x_topics'] = df['prior_session_count'] * df['topic_count']
df['hist_adoption_x_hastopic'] = df['prior_adoption_score'] * df['has_topic_trained_on']
df['hist_grp_adoption_x_farmer_adoption'] = df['prior_grp_adoption_score'] * df['prior_adoption_score']
df['hist_ever_adopted_x_hastopic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['farmer_is_repeat'] = (df['prior_session_count'] > 0).astype(int)
df['farmer_high_engagement'] = (df['prior_session_count'] >= 5).astype(int)
df['farmer_adopted_before'] = df['prior_any_adoption']

# --- 7I. AGGREGATIONS ---
print("  7I. Aggregation features...")
for stat_col in ['group_size', 'group_coop_rate', 'prior_grp_adoption_score']:
    for agg in ['mean', 'std']:
        col_name = f'county_{stat_col}_{agg}'
        if stat_col in df.columns:
            df[col_name] = df.groupby('county')[stat_col].transform(agg)

df['ward_coop_rate'] = df.groupby('ward')['belong_to_cooperative'].transform('mean')
df['ward_female_rate'] = df.groupby('ward')['gender'].transform(lambda x: (x == 'Female').mean())
df['ward_group_count'] = df.groupby('ward')['group_name'].transform('nunique')

for col in df.columns:
    if col.endswith('_std'):
        df[col] = df[col].fillna(0)

# --- 7J. ADVANCED INTERACTIONS (V2 proven + V5 new) ---
print("  7J. Advanced interactions...")

# V2 PROVEN (keep all of these)
df['trainer_eff_x_has_topic'] = df['prior_trainer_effectiveness'] * df['has_topic_trained_on']
df['trainer_eff_x_coop'] = df['prior_trainer_effectiveness'] * df['belong_to_cooperative']
df['trainer_eff_x_ussd'] = df['prior_trainer_effectiveness'] * (df['registration'] == 'Ussd').astype(int)
df['is_ussd'] = (df['registration'] == 'Ussd').astype(int)
df['is_coop'] = df['belong_to_cooperative']
df['ussd_x_coop'] = df['is_ussd'] * df['is_coop']
df['ussd_x_has_topic'] = df['is_ussd'] * df['has_topic_trained_on']
df['coop_x_has_topic'] = df['is_coop'] * df['has_topic_trained_on']
df['triple_signal'] = df['is_ussd'] * df['is_coop'] * df['has_topic_trained_on']
df['recency_weighted_adoption'] = df['prior_adoption_score'] / (df['days_since_last_prior'].clip(lower=1) / 100.0)
df.loc[df['days_since_last_prior'] < 0, 'recency_weighted_adoption'] = 0
df['grp_adopt_x_trainer_eff'] = df['prior_grp_adoption_score'] * df['prior_trainer_effectiveness']
df['farmer_engaged_good_group'] = df['farmer_high_engagement'] * (df['prior_grp_adoption_score'] > 0).astype(int)
df['county_trainer_density'] = df.groupby('county')['trainer_parsed'].transform('nunique')

# V5 NEW INTERACTIONS
# Prior adopter × has_topic (17x · 30x = extremely powerful)
df['prior_adopter_x_has_topic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
# Group social proof × farmer history
df['group_proof_x_farmer_hist'] = df['prior_grp_adopter_density'].fillna(0) * df['prior_any_adoption']
# Trainer quality × group quality
df['trainer_x_group_quality'] = df['prior_trainer_effectiveness'] * df['prior_grp_adoption_score']
# Sunday × has_topic (Sunday training + topic = motivated)
df['sunday_x_has_topic'] = df['is_sunday'] * df['has_topic_trained_on']
# Recency × triple signal
df['recency_x_triple'] = df['recency_weighted_adoption'] * df['triple_signal']
# Sequence × adoption (repeat farmers with adoption history)
df['sequence_x_adoption'] = df['training_sequence_num'] * df['prior_adoption_score']
# Topic adoption rate × trainer effectiveness
df['topic_rate_x_trainer'] = df.get('topic_adoption_rate_120', pd.Series(0, index=df.index)) * df['prior_trainer_effectiveness']
# County adoption rate × farmer history
for geo in ['county', 'ward']:
    if f'prior_{geo}_120_rate' in df.columns:
        df[f'{geo}_rate_x_farmer_adopt'] = df[f'prior_{geo}_120_rate'] * df['prior_adoption_score']

# --- 7K. TARGET ENCODING (OOF) ---
print("  7K. Smoothed target encoding (OOF)...")
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
        for fold_idx, (tr_idx, val_idx) in enumerate(skf_te.split(train_data, train_data[target])):
            fold_train = train_data.iloc[tr_idx]
            fold_val_indices = train_data.iloc[val_idx].index
            stats = fold_train.groupby(col)[target].agg(['sum', 'count'])
            smoothed = (stats['sum'] + SMOOTHING * global_mean) / (stats['count'] + SMOOTHING)
            df.loc[fold_val_indices, te_col_name] = train_data.loc[fold_val_indices, col].map(smoothed)
        stats_all = train_data.groupby(col)[target].agg(['sum', 'count'])
        smoothed_all = (stats_all['sum'] + SMOOTHING * global_mean) / (stats_all['count'] + SMOOTHING)
        test_mask = df['is_train'] == 0
        df.loc[test_mask, te_col_name] = df.loc[test_mask, col].map(smoothed_all)
        df[te_col_name] = df[te_col_name].fillna(global_mean)

print(f"  OOF Target encoding: {len(te_cols)} x {len(TARGETS)} = {len(te_cols)*len(TARGETS)} features")

# 7L. PRIOR-BASED TARGET ENCODING (no leakage - uses Prior only)
print("  7L. Prior-based target encoding...")
PRIOR_SMOOTH = 20
for target in TARGETS:
    prior_global = prior_df[target].mean()
    for col in ['group_name', 'ward', 'subcounty', 'county']:
        prior_stats = prior_df.groupby(col)[target].agg(['sum', 'count'])
        prior_smoothed = (prior_stats['sum'] + PRIOR_SMOOTH * prior_global) / (prior_stats['count'] + PRIOR_SMOOTH)
        prior_te_col = f'prior_te_{col}_{target}'
        df[prior_te_col] = df[col].map(prior_smoothed).fillna(prior_global)

print(f"  Prior TE: 4 x {len(TARGETS)} = {4*len(TARGETS)} features")

# ============================================================
# 8. PREPARE FEATURES + CHI-SQUARE SELECTION
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Preparing features + Chi-Square selection...")
print("=" * 70)

exclude_cols = ['ID', 'farmer_name', 'is_train', 'training_day', 'training_day_dt',
                'topics_list', 'topics_parsed', 'topic_cats', 'trainer',
                'prior_last_session_date', 'prior_first_session_date',
                'prior_first_date', 'prior_last_date'] + TARGETS

cat_cols_to_encode = [col for col in df.select_dtypes(include='object').columns
                      if col not in exclude_cols]

label_encoders = {}
for col in cat_cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

feature_cols = [col for col in df.columns if col not in exclude_cols 
                and df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'uint8']]
print(f"  Total candidate features: {len(feature_cols)}")

X_all_train = df.loc[train_idx, feature_cols].reset_index(drop=True)

# ---- CHI-SQUARE FEATURE SELECTION ----
print("\n  Running Chi-Square + Mutual Information feature selection...")

# Replace NaN/inf for chi-square (requires non-negative)
X_chi = X_all_train.copy()
for c in X_chi.columns:
    X_chi[c] = X_chi[c].fillna(0)
    X_chi[c] = X_chi[c].replace([np.inf, -np.inf], 0)
    # Shift to non-negative for chi2
    if X_chi[c].min() < 0:
        X_chi[c] = X_chi[c] - X_chi[c].min()

# Chi-square scores for each target
chi2_scores = {}
mi_scores = {}
for target in TARGETS:
    y_t = df.loc[train_idx, target].reset_index(drop=True).astype(int)
    
    # Chi-square
    chi_vals, chi_pvals = chi2(X_chi, y_t)
    chi2_scores[target] = pd.Series(chi_vals, index=feature_cols)
    
    # Mutual Information (more robust for non-linear)
    mi_vals = mutual_info_classif(X_chi, y_t, random_state=42, n_neighbors=5)
    mi_scores[target] = pd.Series(mi_vals, index=feature_cols)

# Average importance across all 3 targets
avg_chi2 = sum(chi2_scores[t] for t in TARGETS) / len(TARGETS)
avg_mi = sum(mi_scores[t] for t in TARGETS) / len(TARGETS)

# Normalize to 0-1 scale
chi2_norm = (avg_chi2 - avg_chi2.min()) / (avg_chi2.max() - avg_chi2.min() + 1e-8)
mi_norm = (avg_mi - avg_mi.min()) / (avg_mi.max() - avg_mi.min() + 1e-8)

# Combined score (50% chi2 + 50% MI)
combined_importance = 0.5 * chi2_norm + 0.5 * mi_norm
combined_importance = combined_importance.sort_values(ascending=False)

# Print top 30 features
print("\n  TOP 30 FEATURES (Chi2 + MI combined):")
for i, (feat, score) in enumerate(combined_importance.head(30).items()):
    print(f"    {i+1:2d}. {feat:45s} score={score:.4f}")

# Drop features with zero combined importance (corrupting/useless)
# Be conservative — keep features above 1% of max score
SELECTION_THRESHOLD = 0.005  # Keep features with >0.5% of max importance
selected_features = combined_importance[combined_importance > SELECTION_THRESHOLD].index.tolist()
dropped_features = combined_importance[combined_importance <= SELECTION_THRESHOLD].index.tolist()

print(f"\n  Features selected: {len(selected_features)}/{len(feature_cols)}")
print(f"  Features dropped: {len(dropped_features)}")
if dropped_features:
    print(f"  Dropped: {dropped_features[:20]}{'...' if len(dropped_features) > 20 else ''}")

feature_cols = selected_features

# ---- LEAKAGE CHECK ----
print("\n  Leakage check...")
leakage_terms = ['adopted', 'target', 'label', 'y_true']
leaking_cols = [c for c in feature_cols if any(term in c.lower() for term in leakage_terms)
                and not c.startswith('prior_') and not c.startswith('te_')]
if leaking_cols:
    print(f"  WARNING: Potential leakage columns: {leaking_cols}")
    feature_cols = [c for c in feature_cols if c not in leaking_cols]
else:
    print(f"  OK - No leakage detected")

# Build final matrices
X_train = df.loc[train_idx, feature_cols].reset_index(drop=True)
X_test = df.loc[test_idx, feature_cols].reset_index(drop=True)
test_ids_ordered = df.loc[test_idx, 'ID'].values

y_train = {}
for t in TARGETS:
    y_train[t] = df.loc[train_idx, t].reset_index(drop=True).astype(int)

print(f"\n  X_train: {X_train.shape}, X_test: {X_test.shape}")

# Save masks for post-processing
zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0
has_prior_hist = df.loc[test_idx, 'prior_session_count'].values > 0
prior_rates = {}
for target in TARGETS:
    rate_map = {
        'adopted_within_07_days': 'prior_07_rate',
        'adopted_within_90_days': 'prior_90_rate',
        'adopted_within_120_days': 'prior_120_rate',
    }
    prior_rates[target] = df.loc[test_idx, rate_map[target]].values

print(f"  Zero topic in test: {zero_topic_mask.sum()}")
print(f"  Has prior history in test: {has_prior_hist.sum()}")

elapsed = time.time() - start_time
print(f"  Feature engineering took {elapsed:.0f}s")

# ============================================================
# 9. OPTUNA LGB (50 trials, warm-start from V2 cache — PROVEN)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Optuna LGB tuning (50 trials, V2 warm-start)...")
print("=" * 70)

N_FOLDS = 5
BASE_SEED = 42

def competition_score(y_true, y_pred):
    """Score = 0.75*(1-LogLoss) + 0.25*AUC"""
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    return 0.75 * (1 - ll) + 0.25 * auc

def sanitize_lgb_params(params):
    """Fix type issues from cached Optuna params"""
    int_keys = ['num_leaves', 'bagging_freq', 'min_child_samples', 'max_depth', 
                'n_estimators', 'random_state', 'verbose']
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
    float_keys = ['learning_rate', 'feature_fraction', 'bagging_fraction', 
                  'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'min_gain_to_split']
    for k in float_keys:
        if k in params:
            params[k] = float(params[k])
    return params

def optuna_lgb_objective(trial, X, y, target_name):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 30, 130),
        'learning_rate': trial.suggest_float('learning_rate', 0.006, 0.09, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.40, 0.90),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.65, 0.99),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.003, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.03, 20.0, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 14),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.6),
        'random_state': BASE_SEED,
        'n_estimators': 3000,
        'verbose': -1,
    }
    
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    params['scale_pos_weight'] = pos_weight
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        
        preds = model.predict_proba(X_val)[:, 1]
        score = competition_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

# Cache path for V5
OPTUNA_V5_CACHE = os.path.join(DATA_DIR, 'optuna_v5_cache.json')
best_lgb_params = {}

if os.path.exists(OPTUNA_V5_CACHE):
    print("  Found V5 cached Optuna results! Loading...")
    with open(OPTUNA_V5_CACHE, 'r') as f:
        cached = json.load(f)
    for target in TARGETS:
        best_params = sanitize_lgb_params(cached[target].copy())
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        best_lgb_params[target] = best_params
        print(f"    {target}: leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")
else:
    # Load V2 cache for warm-start (the 0.93333 scorer)
    v2_cache = {}
    OPTUNA_V2_CACHE = os.path.join(DATA_DIR, 'optuna_v2_cache.json')
    if os.path.exists(OPTUNA_V2_CACHE):
        with open(OPTUNA_V2_CACHE) as f:
            v2_cache = json.load(f)
        print("  Loaded V2 best params for warm-start (from 0.93333 scorer)")
    
    N_OPTUNA_TRIALS = 50
    
    for target in TARGETS:
        print(f"\n  Tuning {target} ({N_OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        
        # Enqueue V2 best params as first trial
        if target in v2_cache:
            v2_p = v2_cache[target]
            study.enqueue_trial({
                'num_leaves': int(float(v2_p['num_leaves'])),
                'learning_rate': float(v2_p['learning_rate']),
                'feature_fraction': float(v2_p['feature_fraction']),
                'bagging_fraction': float(v2_p['bagging_fraction']),
                'bagging_freq': int(float(v2_p['bagging_freq'])),
                'min_child_samples': int(float(v2_p['min_child_samples'])),
                'reg_alpha': float(v2_p['reg_alpha']),
                'reg_lambda': float(v2_p['reg_lambda']),
                'max_depth': int(float(v2_p['max_depth'])),
                'min_gain_to_split': float(v2_p.get('min_gain_to_split', 0.15)),
            })
            print(f"    Warm-started from V2 (0.93333 scorer)")
        
        study.optimize(
            lambda trial: optuna_lgb_objective(trial, X_train, y_train[target], target),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        
        best_params = study.best_params.copy()
        best_params['objective'] = 'binary'
        best_params['metric'] = 'binary_logloss'
        best_params['boosting_type'] = 'gbdt'
        best_params['n_estimators'] = 3000
        best_params['verbose'] = -1
        best_params['random_state'] = BASE_SEED
        
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        
        best_lgb_params[target] = best_params
        print(f"    Best score: {study.best_value:.6f}")
        print(f"    Best params: leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}, "
              f"ff={best_params['feature_fraction']:.2f}, bf={best_params['bagging_fraction']:.2f}")
    
    # Cache V5 results
    cache_data = {}
    for target in TARGETS:
        cache_data[target] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                              for k, v in best_lgb_params[target].items()}
    with open(OPTUNA_V5_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\n  Cached V5 Optuna results → {OPTUNA_V5_CACHE}")

elapsed = time.time() - start_time
print(f"\n  Optuna LGB took {elapsed:.0f}s total")

# ============================================================
# 10. OPTUNA XGB (25 trials)
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Optuna XGB tuning (25 trials)...")
print("=" * 70)

XGB_V5_CACHE = os.path.join(DATA_DIR, 'optuna_v5_xgb_cache.json')
best_xgb_params = {}

def sanitize_xgb_params(params):
    int_keys = ['max_depth', 'min_child_weight', 'n_estimators', 'early_stopping_rounds', 'random_state', 'verbosity']
    for k in int_keys:
        if k in params: params[k] = int(params[k])
    float_keys = ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'scale_pos_weight']
    for k in float_keys:
        if k in params: params[k] = float(params[k])
    return params

def optuna_xgb_objective(trial, X, y, target_name):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'random_state': BASE_SEED,
        'n_estimators': 3000,
        'verbosity': 0,
        'early_stopping_rounds': 100,
    }
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    params['scale_pos_weight'] = pos_weight
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        score = competition_score(y_val, preds)
        scores.append(score)
    return np.mean(scores)

if os.path.exists(XGB_V5_CACHE):
    print("  Found cached XGB V5 Optuna results! Loading...")
    with open(XGB_V5_CACHE, 'r') as f:
        cached = json.load(f)
    for target in TARGETS:
        best_params = sanitize_xgb_params(cached[target].copy())
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        best_xgb_params[target] = best_params
        print(f"    {target}: depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}")
else:
    # Load V2 XGB cache for warm-start
    v2_xgb_cache = {}
    XGB_V2_CACHE = os.path.join(DATA_DIR, 'optuna_xgb_cache.json')
    if os.path.exists(XGB_V2_CACHE):
        with open(XGB_V2_CACHE) as f:
            v2_xgb_cache = json.load(f)
        print("  Loaded V2 XGB params for warm-start")
    
    N_XGB_TRIALS = 25
    for target in TARGETS:
        print(f"\n  Tuning {target} ({N_XGB_TRIALS} trials)...")
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        
        if target in v2_xgb_cache:
            v2_p = v2_xgb_cache[target]
            study.enqueue_trial({
                'max_depth': int(float(v2_p['max_depth'])),
                'learning_rate': float(v2_p['learning_rate']),
                'subsample': float(v2_p['subsample']),
                'colsample_bytree': float(v2_p['colsample_bytree']),
                'min_child_weight': int(float(v2_p['min_child_weight'])),
                'reg_alpha': float(v2_p['reg_alpha']),
                'reg_lambda': float(v2_p['reg_lambda']),
                'gamma': float(v2_p.get('gamma', 0.0)),
            })
        
        study.optimize(
            lambda trial: optuna_xgb_objective(trial, X_train, y_train[target], target),
            n_trials=N_XGB_TRIALS,
            show_progress_bar=False,
        )
        
        best_params = study.best_params.copy()
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'logloss'
        best_params['tree_method'] = 'hist'
        best_params['n_estimators'] = 3000
        best_params['verbosity'] = 0
        best_params['early_stopping_rounds'] = 100
        best_params['random_state'] = BASE_SEED
        pos_weight = (len(y_train[target]) - y_train[target].sum()) / max(y_train[target].sum(), 1)
        best_params['scale_pos_weight'] = pos_weight
        best_xgb_params[target] = best_params
        print(f"    Best score: {study.best_value:.6f}")
        print(f"    Best params: depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}")
    
    cache_data = {}
    for target in TARGETS:
        cache_data[target] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                              for k, v in best_xgb_params[target].items()}
    with open(XGB_V5_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\n  Cached V5 XGB results → {XGB_V5_CACHE}")

elapsed = time.time() - start_time
print(f"  Optuna XGB took {elapsed:.0f}s total")

# ============================================================
# 11A. LGB MULTI-SEED (10 seeds) + FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("STEP 11A: LightGBM (Optuna-tuned, 10 seeds)...")
print("=" * 70)

LGB_SEEDS = [42, 123, 456, 789, 2025, 1337, 7777, 31415, 99, 555]
lgb_oof_preds = {t: np.zeros(len(X_train)) for t in TARGETS}
lgb_test_preds = {t: np.zeros(len(X_test)) for t in TARGETS}
feature_importance_all = {t: np.zeros(len(feature_cols)) for t in TARGETS}

for target in TARGETS:
    print(f"\n  Target: {target}")
    oof_accumulated = np.zeros(len(X_train))
    test_accumulated = np.zeros(len(X_test))
    
    params = sanitize_lgb_params(best_lgb_params[target].copy())
    
    for seed_idx, seed in enumerate(LGB_SEEDS):
        params_seed = params.copy()
        params_seed['random_state'] = seed
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(X_train))
        test_seed = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train[target])):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[target].iloc[tr_idx], y_train[target].iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params_seed)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            oof_seed[val_idx] = model.predict_proba(X_val)[:, 1]
            test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS
            feature_importance_all[target] += model.feature_importances_
        
        seed_score = competition_score(y_train[target], oof_seed)
        print(f"    Seed {seed}: comp_score={seed_score:.6f}")
        oof_accumulated += oof_seed
        test_accumulated += test_seed
    
    lgb_oof_preds[target] = oof_accumulated / len(LGB_SEEDS)
    lgb_test_preds[target] = test_accumulated / len(LGB_SEEDS)
    
    final_score = competition_score(y_train[target], lgb_oof_preds[target])
    auc = roc_auc_score(y_train[target], lgb_oof_preds[target])
    ll = log_loss(y_train[target], lgb_oof_preds[target])
    print(f"  LGB ({len(LGB_SEEDS)} seeds): AUC={auc:.6f}, LL={ll:.6f}, Comp={final_score:.6f}")

lgb_total = sum(competition_score(y_train[t], lgb_oof_preds[t]) for t in TARGETS)
print(f"\n  LGB TOTAL CV: {lgb_total:.6f}")

# Print feature importance ranking
print("\n  TOP 25 FEATURES BY LGB IMPORTANCE:")
total_imp = sum(feature_importance_all[t] for t in TARGETS)
imp_series = pd.Series(total_imp, index=feature_cols).sort_values(ascending=False)
for i, (feat, imp) in enumerate(imp_series.head(25).items()):
    print(f"    {i+1:2d}. {feat:45s} importance={imp:.0f}")

# ============================================================
# 11B. XGB MULTI-SEED (5 seeds, Optuna-tuned)
# ============================================================
print("\n" + "=" * 70)
print("STEP 11B: XGBoost (Optuna-tuned, 5 seeds)...")
print("=" * 70)

XGB_SEEDS = [42, 123, 456, 789, 2025]
xgb_oof_preds = {t: np.zeros(len(X_train)) for t in TARGETS}
xgb_test_preds = {t: np.zeros(len(X_test)) for t in TARGETS}

for target in TARGETS:
    print(f"\n  Target: {target}")
    oof_accumulated = np.zeros(len(X_train))
    test_accumulated = np.zeros(len(X_test))
    
    for seed in XGB_SEEDS:
        xgb_params = sanitize_xgb_params(best_xgb_params[target].copy())
        xgb_params['random_state'] = seed
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(X_train))
        test_seed = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train[target])):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[target].iloc[tr_idx], y_train[target].iloc[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            oof_seed[val_idx] = model.predict_proba(X_val)[:, 1]
            test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        seed_score = competition_score(y_train[target], oof_seed)
        print(f"    Seed {seed}: comp_score={seed_score:.6f}")
        oof_accumulated += oof_seed
        test_accumulated += test_seed
    
    xgb_oof_preds[target] = oof_accumulated / len(XGB_SEEDS)
    xgb_test_preds[target] = test_accumulated / len(XGB_SEEDS)
    
    final_score = competition_score(y_train[target], xgb_oof_preds[target])
    auc = roc_auc_score(y_train[target], xgb_oof_preds[target])
    ll = log_loss(y_train[target], xgb_oof_preds[target])
    print(f"  XGB ({len(XGB_SEEDS)} seeds): AUC={auc:.6f}, LL={ll:.6f}, Comp={final_score:.6f}")

xgb_total = sum(competition_score(y_train[t], xgb_oof_preds[t]) for t in TARGETS)
print(f"\n  XGB TOTAL CV: {xgb_total:.6f}")

elapsed = time.time() - start_time
print(f"\n  Model training took {elapsed:.0f}s total")

# ============================================================
# 12. STACKING META-LEARNER (V2 exact approach)
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Stacking (LGB+XGB OOF → LogReg)...")
print("=" * 70)

stack_oof_preds = {t: np.zeros(len(X_train)) for t in TARGETS}
stack_test_preds = {t: np.zeros(len(X_test)) for t in TARGETS}

for target in TARGETS:
    meta_train = np.column_stack([
        lgb_oof_preds[target],
        xgb_oof_preds[target],
        lgb_oof_preds[target] * xgb_oof_preds[target],
        np.abs(lgb_oof_preds[target] - xgb_oof_preds[target]),
    ])
    meta_test = np.column_stack([
        lgb_test_preds[target],
        xgb_test_preds[target],
        lgb_test_preds[target] * xgb_test_preds[target],
        np.abs(lgb_test_preds[target] - xgb_test_preds[target]),
    ])
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    stack_oof = np.zeros(len(X_train))
    stack_test_folds = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(meta_train, y_train[target])):
        meta_tr, meta_val = meta_train[tr_idx], meta_train[val_idx]
        y_tr = y_train[target].iloc[tr_idx]
        
        meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        meta_model.fit(meta_tr, y_tr)
        
        stack_oof[val_idx] = meta_model.predict_proba(meta_val)[:, 1]
        stack_test_folds += meta_model.predict_proba(meta_test)[:, 1] / N_FOLDS
    
    stack_oof_preds[target] = stack_oof
    stack_test_preds[target] = stack_test_folds
    
    stack_score = competition_score(y_train[target], stack_oof)
    lgb_score = competition_score(y_train[target], lgb_oof_preds[target])
    xgb_score = competition_score(y_train[target], xgb_oof_preds[target])
    print(f"  {target}: LGB={lgb_score:.6f}, XGB={xgb_score:.6f}, Stack={stack_score:.6f}")

stack_total = sum(competition_score(y_train[t], stack_oof_preds[t]) for t in TARGETS)
print(f"\n  Stack TOTAL CV: {stack_total:.6f}")

# ============================================================
# 13. OPTIMIZE ENSEMBLE WEIGHTS (LGB + XGB + Stack)
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: Optimizing ensemble weights...")
print("=" * 70)

optimal_weights = {}

for target in TARGETS:
    lgb_s = competition_score(y_train[target], lgb_oof_preds[target])
    xgb_s = competition_score(y_train[target], xgb_oof_preds[target])
    stk_s = competition_score(y_train[target], stack_oof_preds[target])
    print(f"\n  {target}: LGB={lgb_s:.6f}, XGB={xgb_s:.6f}, Stack={stk_s:.6f}")
    
    def neg_comp_score(weights):
        w1, w2, w3 = weights
        w_sum = w1 + w2 + w3
        w1, w2, w3 = w1/w_sum, w2/w_sum, w3/w_sum
        blend = w1 * lgb_oof_preds[target] + w2 * xgb_oof_preds[target] + w3 * stack_oof_preds[target]
        return -competition_score(y_train[target], blend)
    
    best_result = None
    best_neg = 0
    for init_w in [[0.5, 0.2, 0.3], [0.6, 0.1, 0.3], [0.7, 0.1, 0.2],
                   [0.4, 0.1, 0.5], [0.3, 0.1, 0.6], [0.8, 0.1, 0.1],
                   [0.5, 0.0, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]:
        result = minimize(neg_comp_score, init_w,
                         bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                         method='Nelder-Mead')
        if best_result is None or result.fun < best_neg:
            best_result = result
            best_neg = result.fun
    
    w = best_result.x
    w = w / w.sum()
    optimal_weights[target] = w
    
    blend_score = -best_neg
    print(f"  Optimal: LGB={w[0]:.3f}, XGB={w[1]:.3f}, Stack={w[2]:.3f} → {blend_score:.6f}")

# ============================================================
# 14. CALIBRATION: PLATT vs ISOTONIC
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: Calibration (Platt vs Isotonic)...")
print("=" * 70)

calibrated_oof = {}
calibrated_test = {}
calibration_method = {}

for target in TARGETS:
    w = optimal_weights[target]
    oof_blend = w[0]*lgb_oof_preds[target] + w[1]*xgb_oof_preds[target] + w[2]*stack_oof_preds[target]
    test_blend = w[0]*lgb_test_preds[target] + w[1]*xgb_test_preds[target] + w[2]*stack_test_preds[target]
    
    raw_score = competition_score(y_train[target], oof_blend)
    raw_ll = log_loss(y_train[target], oof_blend)
    
    # PLATT
    oof_logodds = np.log(np.clip(oof_blend, 1e-7, 1-1e-7) / (1 - np.clip(oof_blend, 1e-7, 1-1e-7)))
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(oof_logodds.reshape(-1, 1), y_train[target])
    platt_oof = platt.predict_proba(oof_logodds.reshape(-1, 1))[:, 1]
    platt_score = competition_score(y_train[target], platt_oof)
    platt_ll = log_loss(y_train[target], platt_oof)
    
    test_logodds = np.log(np.clip(test_blend, 1e-7, 1-1e-7) / (1 - np.clip(test_blend, 1e-7, 1-1e-7)))
    platt_test = platt.predict_proba(test_logodds.reshape(-1, 1))[:, 1]
    
    # ISOTONIC (OOF cross-validated)
    skf_iso = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iso_oof = np.zeros(len(oof_blend))
    iso_test_folds = np.zeros(len(test_blend))
    
    for fold, (tr_idx, val_idx) in enumerate(skf_iso.split(np.arange(len(oof_blend)), y_train[target])):
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso.fit(oof_blend[tr_idx], y_train[target].iloc[tr_idx])
        iso_oof[val_idx] = iso.predict(oof_blend[val_idx])
        iso_test_folds += iso.predict(test_blend) / 5
    
    iso_score = competition_score(y_train[target], iso_oof)
    iso_ll = log_loss(y_train[target], iso_oof)
    
    print(f"\n  {target}:")
    print(f"    Raw:      LL={raw_ll:.6f}, Comp={raw_score:.6f}, mean={oof_blend.mean():.5f}")
    print(f"    Platt:    LL={platt_ll:.6f}, Comp={platt_score:.6f}, mean={platt_oof.mean():.5f}")
    print(f"    Isotonic: LL={iso_ll:.6f}, Comp={iso_score:.6f}, mean={iso_oof.mean():.5f}")
    
    best_method = 'raw'
    best_cal_score = raw_score
    best_cal_oof = oof_blend
    best_cal_test = test_blend
    
    if platt_score > best_cal_score:
        best_method = 'platt'
        best_cal_score = platt_score
        best_cal_oof = platt_oof
        best_cal_test = platt_test
    
    if iso_score > best_cal_score:
        best_method = 'isotonic'
        best_cal_score = iso_score
        best_cal_oof = iso_oof
        best_cal_test = iso_test_folds
    
    calibrated_oof[target] = best_cal_oof
    calibrated_test[target] = best_cal_test
    calibration_method[target] = best_method
    print(f"    WINNER: {best_method} (comp={best_cal_score:.6f})")

cal_total = sum(competition_score(y_train[t], calibrated_oof[t]) for t in TARGETS)
print(f"\n  Calibrated TOTAL CV: {cal_total:.6f}")

# ============================================================
# 15. BAYESIAN CALIBRATION WITH PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 15: Bayesian calibration with Prior farmer history...")
print("=" * 70)

farmer_prior_data = prior_df.groupby('farmer_name').agg(
    n_sessions=('ID', 'count'),
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
            post_mean = (alpha_0 + k) / (alpha_0 + beta_0 + n)
            posteriors[i] = post_mean
    
    bayesian_posteriors[target] = posteriors
    with_hist = np.array([fn in farmer_prior_dict for fn in test_farmer_names])
    print(f"  {target}: posterior mean (w/ hist)={posteriors[with_hist].mean():.5f}")

# OOF Bayesian optimization
train_farmer_names = df.loc[train_idx, 'farmer_name'].values
has_train_hist = np.array([fn in farmer_prior_dict for fn in train_farmer_names])

bayesian_oof = {}
optimal_bayes_weights = {}

for target in TARGETS:
    train_rate = y_train[target].mean()
    PRIOR_STRENGTH = 5
    alpha_0 = train_rate * PRIOR_STRENGTH
    beta_0 = (1 - train_rate) * PRIOR_STRENGTH
    k_col = {'adopted_within_07_days': 'k_07',
             'adopted_within_90_days': 'k_90',
             'adopted_within_120_days': 'k_120'}[target]
    
    oof_posteriors = np.full(len(train_farmer_names), train_rate)
    for i, fname in enumerate(train_farmer_names):
        if fname in farmer_prior_dict:
            data = farmer_prior_dict[fname]
            n = data['n_sessions']
            k = data[k_col]
            post_mean = (alpha_0 + k) / (alpha_0 + beta_0 + n)
            oof_posteriors[i] = post_mean
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
# 16. GENERATE FINAL PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 16: Generating final predictions...")
print("=" * 70)

final_preds_raw = {}
final_preds_bayesian = {}

for target in TARGETS:
    model_test = calibrated_test[target].copy()
    final_preds_raw[target] = model_test.copy()
    
    w_m = optimal_bayes_weights[target]
    blended = model_test.copy()
    blended[has_prior_hist] = w_m * model_test[has_prior_hist] + (1 - w_m) * bayesian_posteriors[target][has_prior_hist]
    final_preds_bayesian[target] = blended

lgb_only_preds = {}
for target in TARGETS:
    lgb_only_preds[target] = lgb_test_preds[target].copy()

# ============================================================
# 17. POST-PROCESSING (enhanced with Prior-informed rules)
# ============================================================
print("\n" + "=" * 70)
print("STEP 17: Post-processing (enhanced)...")
print("=" * 70)

def post_process(preds_dict, label):
    for target in TARGETS:
        preds_dict[target] = np.clip(preds_dict[target], 0.001, 0.999)
    
    # Enforce monotonicity: 120d >= 90d >= 7d
    preds_dict['adopted_within_90_days'] = np.maximum(
        preds_dict['adopted_within_07_days'], preds_dict['adopted_within_90_days'])
    preds_dict['adopted_within_120_days'] = np.maximum(
        preds_dict['adopted_within_90_days'], preds_dict['adopted_within_120_days'])
    
    # RULE 1: Zero-topic → NO adoption (perfect rule, 0/4044 in train)
    for target in TARGETS:
        preds_dict[target][zero_topic_mask] = 0.001
    
    # RULE 2: Zero-group rule from TRAIN
    train_data_rules = df[train_idx].copy()
    for target in TARGETS:
        group_stats = train_data_rules.groupby('group_name').agg(
            n=(target, 'count'), rate=(target, 'mean'))
        zero_groups = group_stats[(group_stats['n'] >= 30) & (group_stats['rate'] == 0)].index
        if len(zero_groups) > 0:
            test_vals = df.loc[test_idx, 'group_name'].values
            zmask = np.isin(test_vals, zero_groups)
            if zmask.sum() > 0:
                preds_dict[target][zmask] = np.minimum(preds_dict[target][zmask], 0.005)
                print(f"  {label} {target}: capped {zmask.sum()} train-zero-group rows")
    
    # RULE 3: Prior-informed zero-group rule (groups with 0 adoption in Prior, n>=20)
    for target in TARGETS:
        prior_grp_stats = prior_df.groupby('group_name').agg(
            n=(target, 'count'), rate=(target, 'mean'))
        prior_zero_grps = prior_grp_stats[(prior_grp_stats['n'] >= 20) & (prior_grp_stats['rate'] == 0)].index
        # Only apply to groups NOT in train (avoid conflict)
        train_groups = set(train_data_rules['group_name'].unique())
        prior_only_zero = [g for g in prior_zero_grps if g not in train_groups]
        if prior_only_zero:
            test_vals = df.loc[test_idx, 'group_name'].values
            zmask = np.isin(test_vals, prior_only_zero)
            if zmask.sum() > 0:
                preds_dict[target][zmask] = np.minimum(preds_dict[target][zmask], 0.008)
                print(f"  {label} {target}: capped {zmask.sum()} prior-zero-group rows")
    
    print(f"  {label}: Mean preds: " +
          ", ".join(f"{t.split('_')[2]}d={preds_dict[t].mean():.5f}" for t in TARGETS))
    
    return preds_dict

final_preds_raw = post_process(final_preds_raw, "Raw")
final_preds_bayesian = post_process(final_preds_bayesian, "Bayesian")
lgb_only_preds = post_process(lgb_only_preds, "LGB-Only")

# ============================================================
# 18. CREATE SUBMISSIONS (DUAL + Standard)
# ============================================================
print("\n" + "=" * 70)
print("STEP 18: Creating submissions...")
print("=" * 70)

def create_submission(preds, test_ids, filename):
    sub = pd.DataFrame({'ID': test_ids})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        sub[auc_col] = preds[target]
        sub[ll_col] = preds[target]
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    assert len(sub) == len(ss)
    assert list(sub.columns) == list(ss.columns)
    assert sub.isnull().sum().sum() == 0
    sub.to_csv(filename, index=False)
    print(f"  SAVED: {filename}")
    return sub

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
    return sub

# V5 submissions
create_dual_submission(final_preds_raw, test_ids_ordered, 'sub_V5_A_ensemble_dual.csv')
create_dual_submission(final_preds_bayesian, test_ids_ordered, 'sub_V5_B_bayesian_dual.csv')
create_dual_submission(lgb_only_preds, test_ids_ordered, 'sub_V5_C_lgb_dual.csv')
create_submission(final_preds_raw, test_ids_ordered, 'sub_V5_D_ensemble_standard.csv')
create_submission(final_preds_bayesian, test_ids_ordered, 'sub_V5_E_bayesian_standard.csv')

# ============================================================
# 19. BLEND WITH V2 BEST (if available)
# ============================================================
print("\n" + "=" * 70)
print("STEP 19: Blending with V2 best submission...")
print("=" * 70)

v2_best_file = 'sub_ULT_C_bayesian_dual.csv'
if os.path.exists(v2_best_file):
    v2_best = pd.read_csv(v2_best_file)
    v5_best = pd.read_csv('sub_V5_A_ensemble_dual.csv')
    
    for blend_w in [0.3, 0.5, 0.7]:
        blended = v2_best.copy()
        for col in SS_COLS[1:]:  # Skip ID
            blended[col] = blend_w * v5_best[col] + (1 - blend_w) * v2_best[col]
        fname = f'sub_V5_blend{int(blend_w*100)}v5_{int((1-blend_w)*100)}v2.csv'
        blended.to_csv(fname, index=False)
        print(f"  SAVED: {fname} ({int(blend_w*100)}% V5 + {int((1-blend_w)*100)}% V2)")
else:
    print(f"  V2 best file not found ({v2_best_file}), skipping blends")

# ============================================================
# 20. FINAL CV SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("STEP 20: FINAL V5 CV SUMMARY")
print("=" * 70)

print("\nPer-target scores (OOF):")
for target in TARGETS:
    lgb_s = competition_score(y_train[target], lgb_oof_preds[target])
    xgb_s = competition_score(y_train[target], xgb_oof_preds[target])
    stk_s = competition_score(y_train[target], stack_oof_preds[target])
    cal_s = competition_score(y_train[target], calibrated_oof[target])
    print(f"  {target}: LGB={lgb_s:.6f}, XGB={xgb_s:.6f}, Stack={stk_s:.6f}, Cal={cal_s:.6f}")

print("\nTotal competition scores (OOF):")
lgb_total = sum(competition_score(y_train[t], lgb_oof_preds[t]) for t in TARGETS)
xgb_total = sum(competition_score(y_train[t], xgb_oof_preds[t]) for t in TARGETS)
stack_total = sum(competition_score(y_train[t], stack_oof_preds[t]) for t in TARGETS)
ens_total = sum(competition_score(y_train[t], 
    optimal_weights[t][0]*lgb_oof_preds[t] + optimal_weights[t][1]*xgb_oof_preds[t] + optimal_weights[t][2]*stack_oof_preds[t]) 
    for t in TARGETS)
cal_total = sum(competition_score(y_train[t], calibrated_oof[t]) for t in TARGETS)

# Bayesian
bayes_total = 0
for target in TARGETS:
    model_oof = calibrated_oof[target]
    w_m = optimal_bayes_weights[target]
    blended = model_oof.copy()
    blended[has_train_hist] = w_m * model_oof[has_train_hist] + (1 - w_m) * bayesian_oof[target][has_train_hist]
    bayes_total += competition_score(y_train[target], blended)

print(f"  LGB-only (10 seeds):   {lgb_total:.6f}")
print(f"  XGB-only (5 seeds):    {xgb_total:.6f}")
print(f"  Stacking:              {stack_total:.6f}")
print(f"  Weighted Ensemble:     {ens_total:.6f}")
print(f"  Calibrated Ensemble:   {cal_total:.6f}")
print(f"  +Bayesian:             {bayes_total:.6f}")
print(f"  ---")
print(f"  V2 best CV:            ~2.933 (LB 0.93333)")
print(f"  V3 best CV:            ~2.931 (DROPPED)")

print(f"\nFeature Selection:")
print(f"  Candidates: {len(combined_importance)}, Selected: {len(feature_cols)}, Dropped: {len(dropped_features)}")
print(f"  Chi2+MI threshold: {SELECTION_THRESHOLD}")

print(f"\nEnsemble weights:")
for target in TARGETS:
    w = optimal_weights[target]
    print(f"  {target}: LGB={w[0]:.3f}, XGB={w[1]:.3f}, Stack={w[2]:.3f}")

print(f"\nCalibration:")
for target in TARGETS:
    print(f"  {target}: {calibration_method[target]}")

print(f"\nBayesian blend:")
for target in TARGETS:
    print(f"  {target}: model={optimal_bayes_weights[target]:.2f}")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"V5 TOTAL TIME: {total_time/60:.1f} minutes")
print(f"{'='*70}")

print(f"\nSUBMISSIONS:")
print(f"  sub_V5_A_ensemble_dual.csv    - DUAL calibrated ensemble (PRIMARY)")
print(f"  sub_V5_B_bayesian_dual.csv    - Bayesian + DUAL (if Bayesian helps)")
print(f"  sub_V5_C_lgb_dual.csv         - LGB-only DUAL (safe fallback)")
print(f"  sub_V5_D_ensemble_standard.csv - Standard ensemble")
print(f"  sub_V5_E_bayesian_standard.csv - Standard Bayesian")
if os.path.exists(v2_best_file):
    print(f"  sub_V5_blend30v5_70v2.csv     - 30% V5 + 70% V2 best")
    print(f"  sub_V5_blend50v5_50v2.csv     - 50% V5 + 50% V2 best")
    print(f"  sub_V5_blend70v5_30v2.csv     - 70% V5 + 30% V2 best")
print(f"\nSUBMIT: sub_V5_A_ensemble_dual.csv first!")
print(f"If V5 CV > V2 CV, submit V5_A. If not, try blend50.")
