"""
DUAL_FINAL.py — Final Combined Solution
=========================================
LESSONS LEARNED from 15+ LB submissions:
  - V9 Bayesian hierarchy = LB 0.7247 (BEST)
  - LGBM replacing AUC column = WORSE (SUPER_A/B/D all failed)
  - LGBM replacing LL column = TERRIBLE (V10 = 0.720)
  - V11 sqrt smoothing = WORSE despite better CV
  - Topic=0 fix was the ONLY improvement that worked on LB

NEW STRATEGY:
  1. Recreate teammate's V4.1f LGBM faithfully
  2. Instead of REPLACING V9, use LGBM as a CONFIDENCE SIGNAL
  3. Only adjust V9 where LGBM strongly agrees (both predict same extreme)
  4. Apply all zero-group rules (V12 — untested on LB, same logic as topic fix)
  5. Try rank-averaged AUC (preserves V9's ordering, adds LGBM diversity)
  6. Calibrated blend for LL (very conservative — 95%+ V9)

Submissions generated:
  A. V9 + V12 zero-group rules only (NO LGBM — safest)
  B. V9 + V12 rules + LGBM confidence-gated adjustments
  C. Rank-averaged AUC + V9 LL + rules
  D. 95% V9 + 5% LGBM both columns + rules
  E. V9 + zero-rule for LL only (AUC untouched)
  F. Agreement ensemble (only change where V9 & LGBM agree)
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import lightgbm as lgb

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
TC = {
    'adopted_within_07_days':  ('Target_07_AUC', 'Target_07_LogLoss'),
    'adopted_within_90_days':  ('Target_90_AUC', 'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

print("=" * 80)
print("  DUAL FINAL — Confidence-Gated Ensemble + Zero-Group Rules")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")
train = pd.read_csv('Train.csv', parse_dates=['training_date'])
test = pd.read_csv('Test.csv', parse_dates=['training_date'])
v9 = pd.read_csv('sub_V9_A_v7_topicfix.csv')
N_TRAIN, N_TEST = len(train), len(test)
print(f"  Train: {N_TRAIN}, Test: {N_TEST}")

# ══════════════════════════════════════════════════════════════════════════
# 2. TEAMMATE'S V4.1f FEATURE ENGINEERING (Faithful recreation)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/7] Building V4.1f features (teammate's exact pipeline)...")
t0 = time.time()

# Parse dates (already done on load)
for df in [train, test]:
    df['train_month'] = df['training_date'].dt.month
    df['train_day'] = df['training_date'].dt.day
    df['train_dayofweek'] = df['training_date'].dt.dayofweek

# ── V2.1d: Location features ──
def apply_v21d_location(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    for col in ['county', 'subcounty', 'ward']:
        freq_map = train_df[col].value_counts().to_dict()
        train_out[f'{col}_count'] = train_df[col].map(freq_map)
        test_out[f'{col}_count'] = test_df[col].map(freq_map).fillna(0)
    
    train_out['county_subcounty'] = train_df['county'].astype(str) + '_' + train_df['subcounty'].astype(str)
    test_out['county_subcounty'] = test_df['county'].astype(str) + '_' + test_df['subcounty'].astype(str)
    train_out['subcounty_ward'] = train_df['subcounty'].astype(str) + '_' + train_df['ward'].astype(str)
    test_out['subcounty_ward'] = test_df['subcounty'].astype(str) + '_' + test_df['ward'].astype(str)
    
    for col in ['county_subcounty', 'subcounty_ward']:
        le = LabelEncoder()
        combined = pd.concat([train_out[col], test_out[col]])
        le.fit(combined)
        train_out[col] = le.transform(train_out[col])
        test_out[col] = le.transform(test_out[col])
    
    for pair, col_name in [(('county', 'subcounty'), 'county_subcounty'), 
                            (('subcounty', 'ward'), 'subcounty_ward')]:
        counts = train_df.groupby(list(pair)).size().reset_index(name='_count')
        train_out = train_out.merge(counts.rename(columns={'_count': f'{col_name}_count'}), 
                                     on=list(pair), how='left')
        test_out = test_out.merge(counts.rename(columns={'_count': f'{col_name}_count'}), 
                                   on=list(pair), how='left')
        test_out[f'{col_name}_count'] = test_out[f'{col_name}_count'].fillna(0)
    
    county_counts = train_df['county'].value_counts().to_dict()
    subcounty_counts = train_df['subcounty'].value_counts().to_dict()
    ward_counts = train_df['ward'].value_counts().to_dict()
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        c = df_in['county'].map(county_counts).fillna(1)
        s = df_in['subcounty'].map(subcounty_counts).fillna(1)
        w = df_in['ward'].map(ward_counts).fillna(1)
        df_out['subcounty_county_ratio'] = s / c
        df_out['ward_subcounty_ratio'] = w / s
        df_out['ward_county_ratio'] = w / c
    return train_out, test_out

# ── V2.2d: Trainer features ──
def apply_v22d_trainer(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    trainer_count = train_df.groupby('trainer').size().to_dict()
    trainer_unique_groups = train_df.groupby('trainer')['group_name'].nunique().to_dict()
    trainer_unique_counties = train_df.groupby('trainer')['county'].nunique().to_dict()
    trainer_unique_wards = train_df.groupby('trainer')['ward'].nunique().to_dict()
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['trainer_count'] = df_in['trainer'].map(trainer_count).fillna(1)
        df_out['trainer_unique_groups'] = df_in['trainer'].map(trainer_unique_groups).fillna(1)
        df_out['trainer_unique_counties'] = df_in['trainer'].map(trainer_unique_counties).fillna(1)
        df_out['trainer_unique_wards'] = df_in['trainer'].map(trainer_unique_wards).fillna(1)
        df_out['trainer_groups_per_county'] = df_out['trainer_unique_groups'] / df_out['trainer_unique_counties']
    
    trainer_first = train_df.groupby('trainer')['training_date'].min().to_dict()
    trainer_last = train_df.groupby('trainer')['training_date'].max().to_dict()
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        first = df_in['trainer'].map(trainer_first)
        last = df_in['trainer'].map(trainer_last)
        df_out['trainer_days_active'] = (df_in['training_date'] - first).dt.days.fillna(0)
        df_out['is_trainer_first'] = (df_out['trainer_days_active'] == 0).astype(int)
        df_out['trainer_tenure'] = (last - first).dt.days.fillna(0)
    
    def get_topic_category(topic):
        if pd.isna(topic): return 'unknown'
        topic = str(topic).lower()
        if any(w in topic for w in ['dairy', 'cow', 'milk']): return 'dairy'
        if any(w in topic for w in ['poultry', 'chicken', 'egg']): return 'poultry'
        if any(w in topic for w in ['health', 'disease', 'vaccin']): return 'health'
        if any(w in topic for w in ['feed', 'nutrition']): return 'feeding'
        return 'other'
    
    train_out['topic_category'] = train_df['topics'].apply(get_topic_category)
    test_out['topic_category'] = test_df['topics'].apply(get_topic_category)
    
    trainer_topics = train_df.groupby('trainer')['topics'].nunique().to_dict()
    trainer_specialty = train_out.groupby('trainer')['topic_category'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
    ).to_dict()
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['trainer_unique_topics'] = df_in['trainer'].map(trainer_topics).fillna(1)
        df_out['trainer_specialty'] = df_in['trainer'].map(trainer_specialty).fillna('unknown')
        df_out['is_trainer_specialty'] = (df_out['topic_category'] == df_out['trainer_specialty']).astype(int)
        df_out['trainer_topic_specialization'] = 1 / df_out['trainer_unique_topics']
    
    for col in ['topic_category', 'trainer_specialty']:
        le = LabelEncoder()
        combined = pd.concat([train_out[col].astype(str), test_out[col].astype(str)])
        le.fit(combined)
        train_out[col] = le.transform(train_out[col].astype(str))
        test_out[col] = le.transform(test_out[col].astype(str))
    
    train_out['_tm'] = train_df['trainer'].astype(str) + '_' + train_df['train_month'].astype(str)
    test_out['_tm'] = test_df['trainer'].astype(str) + '_' + test_df['train_month'].astype(str)
    train_out['_td'] = train_df['trainer'].astype(str) + '_' + train_df['training_date'].astype(str)
    test_out['_td'] = test_df['trainer'].astype(str) + '_' + test_df['training_date'].astype(str)
    
    monthly = train_out.groupby('_tm').size().to_dict()
    daily = train_out.groupby('_td').size().to_dict()
    monthly_med = np.median(list(monthly.values()))
    for df_out in [train_out, test_out]:
        df_out['trainer_monthly_count'] = df_out['_tm'].map(monthly).fillna(1)
        df_out['trainer_daily_count'] = df_out['_td'].map(daily).fillna(1)
        df_out['trainer_is_busy'] = (df_out['trainer_monthly_count'] > monthly_med).astype(int)
        df_out.drop(columns=['_tm', '_td'], inplace=True)
    return train_out, test_out

# ── V2.3d: Group features (reconstructed from feature names) ──
def apply_v23d_group(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    
    # Group size
    group_size = train_df.groupby('group_name').size().to_dict()
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['group_size'] = df_in['group_name'].map(group_size).fillna(1)
        df_out['is_large_group'] = (df_out['group_size'] > df_out['group_size'].quantile(0.75)).astype(int)
        df_out['is_small_group'] = (df_out['group_size'] < df_out['group_size'].quantile(0.25)).astype(int)
        df_out['group_size_log'] = np.log1p(df_out['group_size'])
    
    # Group demographics
    group_coop = train_df.groupby('group_name')['belong_to_cooperative'].mean().to_dict()
    group_male = train_df.groupby('group_name')['gender'].apply(lambda x: (x == 'Male').mean()).to_dict()
    group_age_div = train_df.groupby('group_name')['age'].nunique().to_dict()
    group_reg_div = train_df.groupby('group_name')['registration'].nunique().to_dict()
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['group_coop_rate'] = df_in['group_name'].map(group_coop).fillna(0.5)
        df_out['group_male_rate'] = df_in['group_name'].map(group_male).fillna(0.5)
        df_out['group_age_diversity'] = df_in['group_name'].map(group_age_div).fillna(1)
        df_out['group_registration_diversity'] = df_in['group_name'].map(group_reg_div).fillna(1)
        df_out['group_homogeneity'] = 1 / (df_out['group_age_diversity'] * df_out['group_registration_diversity'])
    
    # Group name text features
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['group_name_length'] = df_in['group_name'].str.len()
        df_out['group_word_count'] = df_in['group_name'].str.split().str.len()
        df_out['group_has_number'] = df_in['group_name'].str.contains(r'\d', regex=True).astype(int)
    
    # Group geographic spread
    group_wards = train_df.groupby('group_name')['ward'].nunique().to_dict()
    group_counties = train_df.groupby('group_name')['county'].nunique().to_dict()
    group_subcounties = train_df.groupby('group_name')['subcounty'].nunique().to_dict()
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['group_wards'] = df_in['group_name'].map(group_wards).fillna(1)
        df_out['group_counties'] = df_in['group_name'].map(group_counties).fillna(1)
        df_out['group_subcounties'] = df_in['group_name'].map(group_subcounties).fillna(1)
        df_out['group_is_local'] = (df_out['group_counties'] == 1).astype(int)
        df_out['group_spread'] = df_out['group_wards'] / df_out['group_counties'].clip(lower=1)
    
    # Group name prefixes
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['group_prefix_3'] = df_in['group_name'].str[:3].fillna('UNK')
        df_out['group_prefix_4'] = df_in['group_name'].str[:4].fillna('UNK')
    
    for col in ['group_prefix_3', 'group_prefix_4']:
        le = LabelEncoder()
        combined = pd.concat([train_out[col].astype(str), test_out[col].astype(str)])
        le.fit(combined)
        train_out[col] = le.transform(train_out[col].astype(str))
        test_out[col] = le.transform(test_out[col].astype(str))
    
    return train_out, test_out

# ── V2.4a: Demographic interactions ──
def apply_v24a_demographic(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['gender_age'] = df_in['gender'].astype(str) + '_' + df_in['age'].astype(str)
        df_out['gender_coop'] = df_in['gender'].astype(str) + '_' + df_in['belong_to_cooperative'].astype(str)
        df_out['age_coop'] = df_in['age'].astype(str) + '_' + df_in['belong_to_cooperative'].astype(str)
    for col in ['gender_age', 'gender_coop', 'age_coop']:
        le = LabelEncoder()
        combined = pd.concat([train_out[col].astype(str), test_out[col].astype(str)])
        le.fit(combined)
        train_out[col] = le.transform(train_out[col].astype(str))
        test_out[col] = le.transform(test_out[col].astype(str))
    return train_out, test_out

# ── V2.5d: Temporal features ──
def apply_v25d_temporal(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()
    
    def get_season(month):
        if month in [3, 4, 5]: return 'long_rains'
        elif month in [6, 7, 8]: return 'harvest_1'
        elif month in [10, 11]: return 'short_rains'
        elif month in [12, 1, 2]: return 'harvest_2'
        else: return 'dry'
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['season'] = df_in['train_month'].apply(get_season)
        df_out['is_planting_season'] = df_out['season'].isin(['long_rains', 'short_rains']).astype(int)
        df_out['is_harvest_season'] = df_out['season'].isin(['harvest_1', 'harvest_2']).astype(int)
    
    le = LabelEncoder()
    combined = pd.concat([train_out['season'], test_out['season']])
    le.fit(combined)
    train_out['season'] = le.transform(train_out['season'])
    test_out['season'] = le.transform(test_out['season'])
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['season_topic'] = df_out['season'].astype(str) + '_' + df_out['topic_category'].astype(str)
        df_out['season_county'] = df_out['season'].astype(str) + '_' + df_in['county'].astype(str)
        df_out['season_trainer'] = df_out['season'].astype(str) + '_' + df_in['trainer'].astype(str)
    
    for col in ['season_topic', 'season_county', 'season_trainer']:
        le = LabelEncoder()
        combined = pd.concat([train_out[col].astype(str), test_out[col].astype(str)])
        le.fit(combined)
        train_out[col] = le.transform(train_out[col].astype(str))
        test_out[col] = le.transform(test_out[col].astype(str))
    
    ref_date = train_df['training_date'].min()
    ref_year = ref_date.year
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['days_since_start'] = (df_in['training_date'] - ref_date).dt.days.fillna(0)
        df_out['month_number'] = ((df_in['training_date'].dt.year - ref_year) * 12 + df_in['training_date'].dt.month).fillna(0)
        df_out['week_of_year'] = df_in['training_date'].dt.isocalendar().week.fillna(1).astype(int)
        df_out['quarter'] = df_in['training_date'].dt.quarter.fillna(1).astype(int)
        df_out['is_weekend'] = (df_in['training_date'].dt.dayofweek >= 5).astype(int)
    
    for df_out, df_in in [(train_out, train_df), (test_out, test_df)]:
        df_out['month_sin'] = np.sin(2 * np.pi * df_in['train_month'] / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * df_in['train_month'] / 12)
        df_out['dow_sin'] = np.sin(2 * np.pi * df_in['train_dayofweek'] / 7)
        df_out['dow_cos'] = np.cos(2 * np.pi * df_in['train_dayofweek'] / 7)
        df_out['day_sin'] = np.sin(2 * np.pi * df_in['train_day'] / 31)
        df_out['day_cos'] = np.cos(2 * np.pi * df_in['train_day'] / 31)
        week = df_in['training_date'].dt.isocalendar().week.fillna(1)
        df_out['week_sin'] = np.sin(2 * np.pi * week / 52)
        df_out['week_cos'] = np.cos(2 * np.pi * week / 52)
    return train_out, test_out

# Apply all feature stages
train_v21d, test_v21d = apply_v21d_location(train, test)
train_v22d, test_v22d = apply_v22d_trainer(train_v21d, test_v21d)
train_v23d, test_v23d = apply_v23d_group(train_v22d, test_v22d)
train_v24a, test_v24a = apply_v24a_demographic(train_v23d, test_v23d)
train_base, test_base = apply_v25d_temporal(train_v24a, test_v24a)

print(f"  Features built in {time.time()-t0:.1f}s")
print(f"  Train: {train_base.shape}, Test: {test_base.shape}")

# ── V4.1f Target Encoding (OOF + Smoothed) ──
TARGET_MAP = {'adopted_within_07_days': '07', 'adopted_within_90_days': '90', 
              'adopted_within_120_days': '120'}
TE_COLUMNS = ['county', 'subcounty', 'ward', 'trainer', 'group_name', 'topic_category']
ALPHA = 10

def target_encode_oof_smoothed(train_te, test_te, col, target, n_splits=5, alpha=10):
    feat_name = f'{col}_{target}_te'
    train_te[feat_name] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = train_te[target].mean()
    for train_idx, val_idx in kf.split(train_te):
        fold_train = train_te.iloc[train_idx]
        stats = fold_train.groupby(col)[target].agg(['sum', 'count'])
        smoothed = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)
        train_te.iloc[val_idx, train_te.columns.get_loc(feat_name)] = \
            train_te.iloc[val_idx][col].map(smoothed)
    train_te[feat_name].fillna(global_mean, inplace=True)
    stats_full = train_te.groupby(col)[target].agg(['sum', 'count'])
    smoothed_full = (stats_full['sum'] + alpha * global_mean) / (stats_full['count'] + alpha)
    test_te[feat_name] = test_te[col].map(smoothed_full)
    test_te[feat_name].fillna(global_mean, inplace=True)
    return train_te, test_te, feat_name

print("  Applying target encoding...")
train_te = train_base.copy()
test_te = test_base.copy()
te_features = []
for target_col in TARGETS:
    target_key = TARGET_MAP[target_col]
    train_te[f'_target_{target_key}'] = train[target_col].astype(int)
    for col in TE_COLUMNS:
        train_te, test_te, feat = target_encode_oof_smoothed(
            train_te, test_te, col, f'_target_{target_key}', n_splits=5, alpha=ALPHA)
        te_features.append(feat)
te_features = list(set(te_features))
print(f"  Added {len(te_features)} target encoding features")

# ── Define feature sets (exact V4.1f) ──
BASE_CAT = ['gender', 'registration', 'age', 'trainer', 'belong_to_cooperative',
            'county', 'subcounty', 'ward', 'group_name', 'topics']
V25D_CAT = BASE_CAT + ['county_subcounty', 'subcounty_ward', 'topic_category', 'trainer_specialty',
                       'group_prefix_3', 'group_prefix_4', 'gender_age', 'gender_coop', 'age_coop',
                       'season', 'season_topic', 'season_county', 'season_trainer']
V25D_NUM = ['has_topic_trained_on', 'train_month', 'train_day', 'train_dayofweek',
            'county_count', 'subcounty_count', 'ward_count',
            'county_subcounty_count', 'subcounty_ward_count',
            'subcounty_county_ratio', 'ward_subcounty_ratio', 'ward_county_ratio',
            'trainer_count', 'trainer_unique_groups', 'trainer_unique_counties',
            'trainer_unique_wards', 'trainer_groups_per_county',
            'trainer_days_active', 'is_trainer_first', 'trainer_tenure',
            'trainer_unique_topics', 'is_trainer_specialty', 'trainer_topic_specialization',
            'trainer_monthly_count', 'trainer_daily_count', 'trainer_is_busy',
            'group_size', 'is_large_group', 'is_small_group', 'group_size_log',
            'group_coop_rate', 'group_male_rate', 'group_age_diversity',
            'group_registration_diversity', 'group_homogeneity',
            'group_name_length', 'group_word_count', 'group_has_number',
            'group_wards', 'group_counties', 'group_subcounties',
            'group_is_local', 'group_spread',
            'is_planting_season', 'is_harvest_season',
            'days_since_start', 'month_number', 'week_of_year', 'quarter', 'is_weekend',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'day_sin', 'day_cos', 'week_sin', 'week_cos']
ALL_NUM = V25D_NUM + te_features

# Prepare features with label + frequency encoding 
def prepare_features(train_df, test_df, cat_features, num_features):
    all_features = cat_features + num_features
    available = [f for f in all_features if f in train_df.columns]
    X_train = train_df[available].copy()
    X_test = test_df[available].copy()
    for col in [c for c in cat_features if c in available]:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        X_train[f'{col}_freq'] = train_df[col].map(freq_map)
        X_test[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)
    for col in [c for c in num_features if c in available]:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    return X_train, X_test

X_train, X_test = prepare_features(train_te, test_te, V25D_CAT, ALL_NUM)
print(f"  Final feature count: {X_train.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# 3. TRAIN V4.1f LightGBM (teammate's exact params)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/7] Training V4.1f LightGBM (teammate's exact params)...")

LGB_PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1
}

lgbm_oof = {}
lgbm_test = {}

for target_col in TARGETS:
    short = target_col.split('_')[-2] + 'd'
    y = train[target_col].astype(int)
    pos_weight = (len(y) - y.sum()) / y.sum()
    params = LGB_PARAMS.copy()
    params['scale_pos_weight'] = pos_weight
    
    oof = np.zeros(N_TRAIN)
    test_preds = np.zeros(N_TEST)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5
    
    oof_auc = roc_auc_score(y, oof)
    oof_ll = log_loss(y, oof)
    print(f"  {short}: AUC={oof_auc:.5f}  LL={oof_ll:.5f}")
    lgbm_oof[target_col] = oof
    lgbm_test[target_col] = np.clip(test_preds, 0.001, 0.999)

# ══════════════════════════════════════════════════════════════════════════
# 4. BAYESIAN HIERARCHY (V9's proven approach)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/7] Building V9 Bayesian hierarchy...")

for df in [train, test]:
    df['trainer_ward_topic'] = df['trainer'].astype(str)+'|'+df['ward'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)
    df['trainer_ward'] = df['trainer'].astype(str)+'|'+df['ward'].astype(str)
    df['trainer_county_topic'] = df['trainer'].astype(str)+'|'+df['county'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)
    df['ward_topic'] = df['ward'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)
    df['county_topic'] = df['county'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)
    df['ward_group'] = df['ward'].astype(str)+'|'+df['group_name'].astype(str)
    df['group_topic'] = df['group_name'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)
    df['subcounty_topic'] = df['subcounty'].astype(str)+'|'+df['has_topic_trained_on'].astype(str)

s3 = lambda n: 3
V3_HIER = [('trainer_ward_topic', s3), ('trainer_ward', s3), ('trainer_county_topic', s3),
           ('ward_topic', s3), ('county_topic', s3), ('trainer', s3), ('ward', s3), ('county', s3)]
V7_FINE = [('ward_group', s3), ('group_name', s3), ('ward_topic', s3),
           ('county_topic', s3), ('ward', s3), ('county', s3), ('trainer', s3)]

def build_hier(hierarchy, target):
    result = pd.Series(np.nan, index=test.index)
    gm = train[target].mean()
    for gk, sfn in hierarchy:
        g = train.groupby(gk)[target].agg(['mean', 'count']).reset_index()
        g.columns = [gk, 'gmean', 'gcount']
        g['s'] = g['gcount'].apply(sfn)
        g['smoothed'] = (g['gmean'] * g['gcount'] + gm * g['s']) / (g['gcount'] + g['s'])
        mapping = dict(zip(g[gk], g['smoothed']))
        vals = test[gk].map(mapping)
        mask = result.isna() & vals.notna()
        result[mask] = vals[mask]
    return result.fillna(gm).values

hier_auc = {}  # V7 for AUC
hier_ll = {}   # V3 for LL
for target in TARGETS:
    hier_auc[target] = build_hier(V7_FINE, target)
    hier_ll[target] = build_hier(V3_HIER, target)
    short = target.split('_')[-2] + 'd'
    print(f"  {short}: AUC_hier={hier_auc[target].mean():.4f}, LL_hier={hier_ll[target].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. ZERO-GROUP RULES (proven concept — same as topic=0 fix)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/7] Mining zero-group rules...")

MIN_N = 30
ZERO_VAL = 0.001

zero_rules = {}
farmer_zero = {}

for target in TARGETS:
    short = target.split('_')[-2] + 'd'
    zero_rules[target] = []
    farmer_zero[target] = []
    
    for gk in ['group_topic', 'subcounty_topic', 'ward_topic']:
        stats = train.groupby(gk)[target].agg(['sum', 'count']).reset_index()
        stats.columns = [gk, 'adopted', 'total']
        zeros = stats[(stats['adopted'] == 0) & (stats['total'] >= MIN_N)]
        for _, row in zeros.iterrows():
            gval = row[gk]
            test_mask = test[gk] == gval
            if test_mask.sum() > 0:
                zero_rules[target].append((gk, gval, int(row['total']),
                                           test.index[test_mask].tolist()))
    
    # Farmer zero (5+ trainings, zero adoption)
    fstats = train.groupby('farmer_id')[target].agg(['sum', 'count']).reset_index()
    for _, row in fstats[(fstats['sum'] == 0) & (fstats['count'] >= 5)].iterrows():
        fid = row['farmer_id']
        test_mask = test['farmer_id'] == fid
        if test_mask.sum() > 0:
            farmer_zero[target].append((fid, test.index[test_mask].tolist()))
    
    n_grp = sum(len(r[3]) for r in zero_rules[target])
    n_fz = sum(len(r[1]) for r in farmer_zero[target])
    print(f"  {short}: {len(zero_rules[target])} group rules ({n_grp} rows), "
          f"{len(farmer_zero[target])} farmer-zero ({n_fz} rows)")

# ══════════════════════════════════════════════════════════════════════════
# 6. CONFIDENCE-GATED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/7] Confidence analysis (where LGBM & V9 agree/disagree)...")

for target in TARGETS:
    short = target.split('_')[-2] + 'd'
    ac, lc = TC[target]
    v9_auc_vals = v9[ac].values
    lgb_vals = lgbm_test[target]
    
    # Where both predict very low (<0.01)
    both_low = ((v9_auc_vals < 0.02) & (lgb_vals < 0.02)).sum()
    # Where both predict high (>0.5)
    both_high = ((v9_auc_vals > 0.5) & (lgb_vals > 0.5)).sum()
    # Where they disagree strongly
    disagree = (np.abs(v9_auc_vals - lgb_vals) > 0.3).sum()
    
    lgb_rank = rankdata(lgb_vals) / len(lgb_vals)
    v9_rank = rankdata(v9_auc_vals) / len(v9_auc_vals)
    rank_corr = np.corrcoef(lgb_rank, v9_rank)[0, 1]
    
    print(f"  {short}: both_low={both_low}, both_high={both_high}, "
          f"disagree={disagree}, rank_corr={rank_corr:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. GENERATE SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n[7/7] Generating submissions...")

def make_sub(name, auc_dict, ll_dict, 
             apply_topic=True, apply_group_zero='none', apply_farmer_zero=False):
    """
    apply_group_zero: 'none', 'group_only', 'group+sub', 'all'
    """
    out = pd.DataFrame({'ID': test['ID']})
    
    for target in TARGETS:
        ac, lc = TC[target]
        auc_v = np.clip(np.array(auc_dict[target], dtype=float), 0.001, 0.999)
        ll_v = np.clip(np.array(ll_dict[target], dtype=float), 0.001, 0.999)
        
        # Topic=0 fix
        if apply_topic:
            mask = (test['has_topic_trained_on'] == 0).values
            auc_v[mask] = ZERO_VAL
            ll_v[mask] = ZERO_VAL
        
        # Zero-group rules
        for gk, gval, tn, idxs in zero_rules[target]:
            if apply_group_zero == 'none':
                break
            if apply_group_zero == 'group_only' and gk != 'group_topic':
                continue
            if apply_group_zero == 'group+sub' and gk not in ['group_topic', 'subcounty_topic']:
                continue
            for idx in idxs:
                auc_v[idx] = min(auc_v[idx], ZERO_VAL)
                ll_v[idx] = min(ll_v[idx], ZERO_VAL)
        
        # Farmer zero
        if apply_farmer_zero:
            for fid, idxs in farmer_zero[target]:
                for idx in idxs:
                    auc_v[idx] = min(auc_v[idx], ZERO_VAL)
                    ll_v[idx] = min(ll_v[idx], ZERO_VAL)
        
        out[ac] = auc_v
        out[lc] = ll_v
    
    # Monotonicity
    for prefix in ['AUC', 'LogLoss']:
        p07 = out[f'Target_07_{prefix}'].values.copy()
        p90 = out[f'Target_90_{prefix}'].values.copy()
        p120 = out[f'Target_120_{prefix}'].values.copy()
        p90 = np.maximum(p90, p07)
        p120 = np.maximum(p120, p90)
        p07 = np.minimum(p07, p90)
        out[f'Target_07_{prefix}'] = np.clip(p07, 0.001, 0.999)
        out[f'Target_90_{prefix}'] = np.clip(p90, 0.001, 0.999)
        out[f'Target_120_{prefix}'] = np.clip(p120, 0.001, 0.999)
    
    fname = f'sub_DUAL_{name}.csv'
    out.to_csv(fname, index=False)
    
    # Compare to V9
    changes = {}
    for target in TARGETS:
        short = target.split('_')[-2] + 'd'
        ac, lc = TC[target]
        ll_diff = (np.abs(out[lc].values - v9[lc].values) > 1e-10).sum()
        auc_diff = (np.abs(out[ac].values - v9[ac].values) > 1e-10).sum()
        changes[short] = (auc_diff, ll_diff)
        print(f"    {short}: AUC Δ={auc_diff:5d}  LL Δ={ll_diff:5d}")
    return fname, changes

# Load V9 column values
v9_auc_d, v9_ll_d = {}, {}
for target in TARGETS:
    ac, lc = TC[target]
    v9_auc_d[target] = v9[ac].values.copy()
    v9_ll_d[target] = v9[lc].values.copy()

# ────────────────────────────────────────────────────────────────────────
# SUB A: Pure V9 + group_topic+subcounty zero rules (NO LGBM)
#   Safest possible — just V12_A logic
# ────────────────────────────────────────────────────────────────────────
print("\n── A: V9 + group+sub zero rules (no LGBM, purest test of rules) ──")
make_sub('f1_g3_w3', v9_auc_d, v9_ll_d, apply_group_zero='group+sub')

# ────────────────────────────────────────────────────────────────────────
# SUB B: V9 + ALL zero rules + farmer zero
# ────────────────────────────────────────────────────────────────────────
print("\n── B: V9 + all zero rules + farmer zero ──")
make_sub('f2_g5_w3_b', v9_auc_d, v9_ll_d, apply_group_zero='all', apply_farmer_zero=True)

# ────────────────────────────────────────────────────────────────────────
# SUB C: V9_AUC + V9_LL with group_topic zero LL-only
#   Only change LL column, keep AUC exactly V9
# ────────────────────────────────────────────────────────────────────────
print("\n── C: V9 + group zero LL-only (AUC untouched) ──")
ll_only = {}
for target in TARGETS:
    ll_only[target] = v9_ll_d[target].copy()
    for gk, gval, tn, idxs in zero_rules[target]:
        if gk in ['group_topic', 'subcounty_topic']:
            for idx in idxs:
                ll_only[target][idx] = min(ll_only[target][idx], ZERO_VAL)
out = pd.DataFrame({'ID': test['ID']})
for target in TARGETS:
    ac, lc = TC[target]
    out[ac] = v9_auc_d[target]  # V9 AUC EXACT
    out[lc] = np.clip(ll_only[target], 0.001, 0.999)
# Monotonicity on LL only
for prefix in ['LogLoss']:
    p07 = out[f'Target_07_{prefix}'].values.copy()
    p90 = out[f'Target_90_{prefix}'].values.copy()
    p120 = out[f'Target_120_{prefix}'].values.copy()
    p90 = np.maximum(p90, p07)
    p120 = np.maximum(p120, p90)
    p07 = np.minimum(p07, p90)
    out[f'Target_07_{prefix}'] = np.clip(p07, 0.001, 0.999)
    out[f'Target_90_{prefix}'] = np.clip(p90, 0.001, 0.999)
    out[f'Target_120_{prefix}'] = np.clip(p120, 0.001, 0.999)
out.to_csv('sub_DUAL_f2_g5_w3.csv', index=False)
for target in TARGETS:
    short = target.split('_')[-2] + 'd'
    ac, lc = TC[target]
    ll_diff = (np.abs(out[lc].values - v9[lc].values) > 1e-10).sum()
    print(f"    {short}: LL Δ={ll_diff:5d}  (AUC=V9 exact)")

# ────────────────────────────────────────────────────────────────────────
# SUB D: Rank-averaged AUC + V9 LL + rules
#   Average the RANKS (not raw values) of LGBM and V9
# ────────────────────────────────────────────────────────────────────────
print("\n── D: Rank-averaged AUC + V9_LL + group+sub rules ──")
rank_auc = {}
for target in TARGETS:
    lgb_rank = rankdata(lgbm_test[target])
    v9_rank = rankdata(v9_auc_d[target])
    avg_rank = (lgb_rank + v9_rank) / 2.0
    # Scale to [0.001, 0.999] preserving rank order
    rank_auc[target] = 0.001 + 0.998 * (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min())
make_sub('f3_g5_w3_d', rank_auc, v9_ll_d, apply_group_zero='group+sub')

# ────────────────────────────────────────────────────────────────────────
# SUB E: Agreement ensemble — only change V9 where LGBM agrees it's wrong
#   If LGBM predicts <0.01 AND V9 predicts >0.05, lower V9 slightly
#   If LGBM predicts >0.5 AND V9 predicts <0.3, raise V9 slightly
# ────────────────────────────────────────────────────────────────────────
print("\n── E: Agreement-gated V9 + rules ──")
agree_auc = {}
agree_ll = {}
for target in TARGETS:
    short = target.split('_')[-2] + 'd'
    a = v9_auc_d[target].copy()
    l = v9_ll_d[target].copy()
    lgb = lgbm_test[target]
    
    # Where LGBM says near-zero AND V9 says moderate -> pull V9 down
    pull_down = (lgb < 0.01) & (a > 0.05)
    a[pull_down] = a[pull_down] * 0.5  # Halve the prediction
    l[pull_down] = l[pull_down] * 0.5
    
    # Where LGBM says high AND V9 says moderate -> pull V9 up  
    pull_up = (lgb > 0.7) & (a < 0.3) & (a > 0.1)
    a[pull_up] = a[pull_up] * 1.3  # Boost by 30%
    l[pull_up] = l[pull_up] * 1.3
    
    n_down = pull_down.sum()
    n_up = pull_up.sum()
    agree_auc[target] = a
    agree_ll[target] = l
    print(f"    {short}: pulled_down={n_down}, pulled_up={n_up}")
make_sub('f2_g3_w3', agree_auc, agree_ll, apply_group_zero='group+sub')

# ────────────────────────────────────────────────────────────────────────
# SUB F: Teammate's V4.1f (improved with DUAL + topic fix + rules)
#   Same predictions for both columns (teammate's original approach)
#   + our proven fixes
# ────────────────────────────────────────────────────────────────────────
print("\n── F: Teammate V4.1f + DUAL topic fix + group rules ──")
# Apply monotonicity to LGBM predictions  
lgbm_mono = {}
for target in TARGETS:
    lgbm_mono[target] = lgbm_test[target].copy()
preds_90 = np.maximum(lgbm_mono[TARGETS[0]], lgbm_mono[TARGETS[1]])
preds_120 = np.maximum(preds_90, lgbm_mono[TARGETS[2]])
lgbm_mono[TARGETS[1]] = preds_90
lgbm_mono[TARGETS[2]] = preds_120
make_sub('f3_g3_w3', lgbm_mono, lgbm_mono, apply_group_zero='group+sub')

# ════════════════════════════════════════════════════════════════════════
# SUB G: Conservative blend: 95% V9 + 5% LGBM for BOTH columns + rules
# ════════════════════════════════════════════════════════════════════════
print("\n── G: 95%V9 + 5%LGBM both columns + group+sub rules ──")
blend_auc_95 = {}
blend_ll_95 = {}
for target in TARGETS:
    blend_auc_95[target] = 0.95 * v9_auc_d[target] + 0.05 * lgbm_test[target]
    blend_ll_95[target] = 0.95 * v9_ll_d[target] + 0.05 * lgbm_test[target]
make_sub('f2_g5_w3_g', blend_auc_95, blend_ll_95, apply_group_zero='group+sub')

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SUBMISSION PRIORITY")
print("=" * 80)
print("""
  ★ #1: sub_DUAL_f1_g3_w3.csv
     V9 exact + group+subcounty zero rules (NO LGBM, safest)
     Changes ONLY ~700-900 rows per target with zero-adoption groups
     Same logic as topic=0 fix that took V7→V9
     
  #2: sub_DUAL_f2_g5_w3.csv
     V9 + group+sub zero rules on LL ONLY (AUC=V9 exact)
     If #1 hurts, tests LL-only changes
     
  #3: sub_DUAL_f2_g5_w3_b.csv
     V9 + ALL zero rules (group+sub+ward+farmer)
     More aggressive rules
     
  #4: sub_DUAL_f3_g5_w3_d.csv
     Rank-averaged AUC (LGBM+V9) + V9 LL + group rules
     
  #5: sub_DUAL_f2_g3_w3.csv
     Agreement-gated: adjust V9 only where LGBM strongly agrees
     
  #6: sub_DUAL_f3_g3_w3.csv
     Teammate V4.1f LGBM + topic fix + group rules
     
  #7: sub_DUAL_f2_g5_w3_g.csv
     95%V9 + 5%LGBM both columns + rules

  KEY: #1-#3 change ONLY zero-adoption group rows.
  Same approach that took V7 (0.723) → V9 (0.7247).
""")

print("DONE!")
