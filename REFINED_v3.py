"""
REFINED_v3.py - Refined pipeline building on D's success (0.91788)
=========================================================================
KEY RULES (learned from LB):
  1. Train on TRAIN ONLY (13.5K rows) - Prior training HURTS
  2. Use Prior only as FEATURES (farmer/group/geo history)  
  3. DUAL strategy works (rank AUC + calibrated LL)
  4. Prior blend at 70/30 helps slightly
  5. Push low predictions DOWN aggressively (true rate ~1-2%)

Improvements over D:
  - 3-model ensemble (different seeds/params, all Train-only)
  - Platt scaling for LogLoss calibration
  - Blend with proven D predictions (safest approach)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import ast
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = r"C:\Users\USER\Desktop\DIGICOW"
os.chdir(DATA_DIR)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Loading data...")
print("=" * 70)

train_df = pd.read_csv('Train.csv')
test_df  = pd.read_csv('Test.csv')
prior_df = pd.read_csv('Prior.csv')
ss       = pd.read_csv('SampleSubmission.csv')

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
SS_COLS = list(ss.columns)
TARGET_TO_SS = {
    'adopted_within_07_days':  ('Target_07_AUC',  'Target_07_LogLoss'),
    'adopted_within_90_days':  ('Target_90_AUC',  'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Prior: {prior_df.shape}")

# ============================================================
# 2. PARSE & BUILD PRIOR FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Parsing & building Prior features...")
print("=" * 70)

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

def parse_trainer(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return str(parsed)
    except:
        return str(s)

def count_sessions(s):
    try:
        return len(ast.literal_eval(s))
    except:
        return 1

# Parse Train/Test
for data in [train_df, test_df]:
    data['topics_parsed'] = data['topics_list'].apply(parse_topics_nested)
    data['trainer_parsed'] = data['trainer'].apply(parse_trainer)
    data['num_sessions'] = data['topics_list'].apply(count_sessions)

# Parse Prior
prior_df['topics_parsed'] = prior_df['topics_list'].apply(parse_topics_flat)
prior_df['trainer_parsed'] = prior_df['trainer']
prior_df['num_sessions'] = 1

# --- Farmer history from Prior ---
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

topic_div = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))).reset_index()
topic_div.columns = ['farmer_name', 'prior_unique_topics']
farmer_hist = farmer_hist.merge(topic_div, on='farmer_name', how='left')

prior_df['training_day_dt'] = pd.to_datetime(prior_df['training_day'])
date_feats = prior_df.groupby('farmer_name')['training_day_dt'].agg(
    prior_first_date='min', prior_last_date='max').reset_index()
date_feats['prior_training_span_days'] = (date_feats['prior_last_date'] - date_feats['prior_first_date']).dt.days
farmer_hist = farmer_hist.merge(date_feats[['farmer_name', 'prior_training_span_days']], on='farmer_name', how='left')

farmer_hist['prior_ever_adopted_07'] = (farmer_hist['prior_07_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_90'] = (farmer_hist['prior_90_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_120'] = (farmer_hist['prior_120_adopted'] > 0).astype(int)
farmer_hist['prior_any_adoption'] = ((farmer_hist['prior_07_adopted'] + farmer_hist['prior_90_adopted'] + farmer_hist['prior_120_adopted']) > 0).astype(int)
farmer_hist['prior_adoption_score'] = (
    farmer_hist['prior_07_rate'] * 3 +
    farmer_hist['prior_90_rate'] * 2 +
    farmer_hist['prior_120_rate'] * 1) / 6

# Group history from Prior
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
    group_hist['prior_grp_07_rate'] * 3 +
    group_hist['prior_grp_90_rate'] * 2 +
    group_hist['prior_grp_120_rate'] * 1) / 6

# Geo history from Prior
geo_hists = {}
for geo_col in ['ward', 'subcounty', 'county']:
    geo_hists[geo_col] = prior_df.groupby(geo_col).agg(
        **{f'prior_{geo_col}_size': ('ID', 'count'),
           f'prior_{geo_col}_07_rate': ('adopted_within_07_days', 'mean'),
           f'prior_{geo_col}_90_rate': ('adopted_within_90_days', 'mean'),
           f'prior_{geo_col}_120_rate': ('adopted_within_120_days', 'mean'),
           f'prior_{geo_col}_coop_rate': ('belong_to_cooperative', 'mean'),
           f'prior_{geo_col}_has_topic_rate': ('has_topic_trained_on', 'mean'),
           }).reset_index()

print("  Farmer history: done")
print(f"  Test farmer coverage: {test_df['farmer_name'].isin(farmer_hist['farmer_name']).mean():.1%}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Feature engineering...")
print("=" * 70)

# Merge histories
train_df = train_df.merge(farmer_hist, on='farmer_name', how='left', suffixes=('', '_fh'))
test_df = test_df.merge(farmer_hist, on='farmer_name', how='left', suffixes=('', '_fh'))
train_df = train_df.merge(group_hist, on='group_name', how='left', suffixes=('', '_gh'))
test_df = test_df.merge(group_hist, on='group_name', how='left', suffixes=('', '_gh'))

for geo_col in ['ward', 'subcounty', 'county']:
    train_df = train_df.merge(geo_hists[geo_col], on=geo_col, how='left', suffixes=('', f'_{geo_col}h'))
    test_df = test_df.merge(geo_hists[geo_col], on=geo_col, how='left', suffixes=('', f'_{geo_col}h'))

# Fill NaN history
for data in [train_df, test_df]:
    hist_cols = [c for c in data.columns if c.startswith('prior_')]
    for c in hist_cols:
        data[c] = data[c].fillna(0)

# Combine train+test for transforms
train_df['is_train'] = 1
test_df['is_train'] = 0
for t in TARGETS:
    if t not in test_df.columns:
        test_df[t] = np.nan

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_idx = df['is_train'] == 1
test_idx = df['is_train'] == 0

# --- Temporal ---
print("  Temporal features...")
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

def get_season(m):
    if m in [3, 4, 5]: return 0
    elif m in [6, 7, 8]: return 1
    elif m in [10, 11]: return 2
    elif m in [12, 1, 2]: return 3
    else: return 4
df['season'] = df['train_month'].apply(get_season)

prior_last = prior_df.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last else -1, axis=1)
df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)

# --- Topics ---
print("  Topic features...")
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

# --- Interactions ---
print("  Interaction features...")
df['county_subcounty'] = df['county'] + '_' + df['subcounty']
df['subcounty_ward'] = df['subcounty'] + '_' + df['ward']
df['county_ward'] = df['county'] + '_' + df['ward']
df['county_trainer'] = df['county'] + '_' + df['trainer_parsed']
df['ward_trainer'] = df['ward'] + '_' + df['trainer_parsed']
df['county_topic'] = df['county'] + '_' + df['primary_topic_cat']
df['ward_topic'] = df['ward'] + '_' + df['primary_topic_cat']
df['trainer_topic'] = df['trainer_parsed'] + '_' + df['primary_topic_cat']

# --- Frequency encoding ---
print("  Frequency encoding...")
freq_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
             'primary_topic_cat', 'county_subcounty', 'county_topic',
             'ward_topic', 'trainer_topic', 'county_trainer']
for col in freq_cols:
    df[f'{col}_freq'] = df.groupby(col)[col].transform('count')

# --- Group features ---
print("  Group features...")
df['group_size'] = df.groupby('group_name')['group_name'].transform('count')
df['group_coop_rate'] = df.groupby('group_name')['belong_to_cooperative'].transform('mean')
df['group_female_rate'] = df.groupby('group_name')['gender'].transform(lambda x: (x == 'Female').mean())
df['group_young_rate'] = df.groupby('group_name')['age'].transform(lambda x: (x == 'Below 35').mean())
df['group_ussd_rate'] = df.groupby('group_name')['registration'].transform(lambda x: (x == 'Ussd').mean())
df['group_topic_diversity'] = df.groupby('group_name')['primary_topic_cat'].transform('nunique')
df['group_trainer_diversity'] = df.groupby('group_name')['trainer_parsed'].transform('nunique')
df['group_has_topic_rate'] = df.groupby('group_name')['has_topic_trained_on'].transform('mean')
df['group_session_mean'] = df.groupby('group_name')['num_sessions'].transform('mean')

# --- Trainer features ---
print("  Trainer features...")
df['trainer_total'] = df.groupby('trainer_parsed')['trainer_parsed'].transform('count')
df['trainer_group_diversity'] = df.groupby('trainer_parsed')['group_name'].transform('nunique')
df['trainer_county_diversity'] = df.groupby('trainer_parsed')['county'].transform('nunique')
df['trainer_coop_rate'] = df.groupby('trainer_parsed')['belong_to_cooperative'].transform('mean')
df['trainer_female_rate'] = df.groupby('trainer_parsed')['gender'].transform(lambda x: (x == 'Female').mean())

# --- Demographics ---
print("  Demographic interactions...")
df['gender_age'] = df['gender'] + '_' + df['age']
df['gender_coop'] = df['gender'] + '_' + df['belong_to_cooperative'].astype(str)
df['age_coop'] = df['age'] + '_' + df['belong_to_cooperative'].astype(str)
df['registration_age'] = df['registration'] + '_' + df['age']
df['gender_registration'] = df['gender'] + '_' + df['registration']
df['gender_trainer'] = df['gender'] + '_' + df['trainer_parsed']
df['age_trainer'] = df['age'] + '_' + df['trainer_parsed']
df['gender_county'] = df['gender'] + '_' + df['county']
df['gender_topic'] = df['gender'] + '_' + df['primary_topic_cat']

# --- History interactions ---
print("  History interactions...")
df['hist_sessions_x_topics'] = df['prior_session_count'] * df['topic_count']
df['hist_adoption_x_hastopic'] = df['prior_adoption_score'] * df['has_topic_trained_on']
df['hist_grp_x_farmer_adoption'] = df['prior_grp_adoption_score'] * df['prior_adoption_score']
df['hist_ever_adopted_x_hastopic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['farmer_high_engagement'] = (df['prior_session_count'] >= 5).astype(int)

# NEW: More history features to differentiate from D
df['prior_sessions_log'] = np.log1p(df['prior_session_count'])
df['prior_grp_size_log'] = np.log1p(df['prior_grp_size'])
df['prior_adoption_consistency'] = (df['prior_ever_adopted_07'] + df['prior_ever_adopted_90'] + df['prior_ever_adopted_120']) / 3
df['farmer_recency_weight'] = np.where(df['days_since_last_prior'] >= 0, 
                                        1 / (1 + df['days_since_last_prior']/30), 0)
df['weighted_07_rate'] = df['prior_07_rate'] * df['farmer_recency_weight']
df['weighted_90_rate'] = df['prior_90_rate'] * df['farmer_recency_weight']
df['weighted_120_rate'] = df['prior_120_rate'] * df['farmer_recency_weight']

# Train group adoption rates (OOF)
print("  Train group adoption stats...")
for target in TARGETS:
    short = target.replace('adopted_within_', '').replace('_days', '')
    grp_stats = df[train_idx].groupby('group_name')[target].agg(['mean', 'sum', 'count']).reset_index()
    grp_stats.columns = ['group_name', f'train_grp_{short}_rate', f'train_grp_{short}_sum', f'train_grp_{short}_n']
    df = df.merge(grp_stats, on='group_name', how='left')
    df[f'train_grp_{short}_rate'] = df[f'train_grp_{short}_rate'].fillna(df[train_idx][target].mean())
    df[f'train_grp_{short}_sum'] = df[f'train_grp_{short}_sum'].fillna(0)
    df[f'train_grp_{short}_n'] = df[f'train_grp_{short}_n'].fillna(0)

# --- Aggregations ---
print("  Aggregations...")
for stat_col in ['group_size', 'group_coop_rate', 'prior_grp_adoption_score']:
    for agg in ['mean', 'std']:
        if stat_col in df.columns:
            df[f'county_{stat_col}_{agg}'] = df.groupby('county')[stat_col].transform(agg)

df['ward_coop_rate'] = df.groupby('ward')['belong_to_cooperative'].transform('mean')
df['ward_female_rate'] = df.groupby('ward')['gender'].transform(lambda x: (x == 'Female').mean())
df['ward_group_count'] = df.groupby('ward')['group_name'].transform('nunique')

for col in df.columns:
    if col.endswith('_std'):
        df[col] = df[col].fillna(0)

# --- Target encoding (OOF on TRAIN ONLY) ---
print("  Target encoding (OOF)...")
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

# Prior-based TE
PRIOR_SMOOTH = 20
for target in TARGETS:
    prior_global = prior_df[target].mean()
    for col in ['group_name', 'ward', 'subcounty', 'county']:
        prior_stats = prior_df.groupby(col)[target].agg(['sum', 'count'])
        prior_smoothed = (prior_stats['sum'] + PRIOR_SMOOTH * prior_global) / (prior_stats['count'] + PRIOR_SMOOTH)
        df[f'prior_te_{col}_{target}'] = df[col].map(prior_smoothed).fillna(prior_global)

print(f"  Features ready")

# ============================================================
# 4. PREPARE FEATURE MATRIX
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Preparing feature matrix...")
print("=" * 70)

exclude_cols = ['ID', 'farmer_name', 'is_train', 'training_day', 'training_day_dt',
                'topics_list', 'topics_parsed', 'topic_cats', 'trainer'] + TARGETS

dup_cols = [c for c in df.columns if c.endswith('_fh') or c.endswith('_gh') or 
            c.endswith('_wardh') or c.endswith('_subcountyh') or c.endswith('_countyh')]
exclude_cols.extend(dup_cols)

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

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 5. TRAIN 3-MODEL ENSEMBLE (all Train-only)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Training 3-model ensemble...")
print("=" * 70)

N_FOLDS = 5

configs = {
    'v1': {  # Close to D's original params
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.03,
        'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'max_depth': -1, 'min_child_samples': 20,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'n_estimators': 3000, 'verbose': -1,
    },
    'v2': {  # Different regularization & seed
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.03,
        'feature_fraction': 0.7, 'bagging_fraction': 0.75, 'bagging_freq': 3,
        'max_depth': -1, 'min_child_samples': 25,
        'reg_alpha': 0.2, 'reg_lambda': 1.5,
        'random_state': 123, 'n_estimators': 3000, 'verbose': -1,
    },
    'v3': {  # Shallower, more conservative
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 47, 'learning_rate': 0.025,
        'feature_fraction': 0.8, 'bagging_fraction': 0.85, 'bagging_freq': 5,
        'max_depth': 7, 'min_child_samples': 15,
        'reg_alpha': 0.05, 'reg_lambda': 0.5,
        'random_state': 456, 'n_estimators': 3000, 'verbose': -1,
    },
}

all_oof = {c: {} for c in configs}
all_test = {c: {} for c in configs}
feature_importance = {}

for cfg_name, base_params in configs.items():
    print(f"\n  --- Model {cfg_name} ---")
    for target in TARGETS:
        y = y_train[target]
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        params = base_params.copy()
        params['scale_pos_weight'] = pos_weight
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        fold_imp = np.zeros(len(feature_cols))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
                     eval_set=[(X_train.iloc[val_idx], y.iloc[val_idx])],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
            
            oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
            fold_imp += model.feature_importances_ / N_FOLDS
        
        auc = roc_auc_score(y, oof)
        ll = log_loss(y, oof)
        comp = 0.75 * (1 - ll) + 0.25 * auc
        print(f"    {target}: AUC={auc:.6f}, LL={ll:.6f}, Comp={comp:.6f}")
        
        all_oof[cfg_name][target] = oof
        all_test[cfg_name][target] = test_preds
        
        if cfg_name == 'v1':
            feature_importance[target] = pd.Series(fold_imp, index=feature_cols).sort_values(ascending=False)

# Find best ensemble weights via CV
print("\n  Finding best ensemble weights...")
best_score = -1
best_w = None
for w1 in np.arange(0.2, 0.9, 0.05):
    for w2 in np.arange(0.05, 0.6, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0.05 or w3 > 0.6: continue
        total = 0
        for target in TARGETS:
            blended = w1*all_oof['v1'][target] + w2*all_oof['v2'][target] + w3*all_oof['v3'][target]
            auc = roc_auc_score(y_train[target], blended)
            ll = log_loss(y_train[target], blended)
            total += 0.75 * (1 - ll) + 0.25 * auc
        if total > best_score:
            best_score = total
            best_w = (w1, w2, w3)

print(f"  Best: v1={best_w[0]:.2f}, v2={best_w[1]:.2f}, v3={best_w[2]:.2f} -> CV={best_score:.6f}")

# Also get best single model CV
for cfg_name in configs:
    total = 0
    for target in TARGETS:
        auc = roc_auc_score(y_train[target], all_oof[cfg_name][target])
        ll = log_loss(y_train[target], all_oof[cfg_name][target])
        total += 0.75 * (1 - ll) + 0.25 * auc
    print(f"  Single {cfg_name} CV: {total:.6f}")

# Build ensemble predictions
ens_oof = {}
ens_test = {}
for target in TARGETS:
    ens_oof[target] = best_w[0]*all_oof['v1'][target] + best_w[1]*all_oof['v2'][target] + best_w[2]*all_oof['v3'][target]
    ens_test[target] = best_w[0]*all_test['v1'][target] + best_w[1]*all_test['v2'][target] + best_w[2]*all_test['v3'][target]

# ============================================================
# 6. PLATT SCALING FOR LOGLOSS CALIBRATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Platt scaling...")
print("=" * 70)

platt_test = {}
for target in TARGETS:
    oof = ens_oof[target].reshape(-1, 1)
    y = y_train[target].values
    
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(oof, y)
    
    platt_oof = lr.predict_proba(oof)[:, 1]
    platt_pred = lr.predict_proba(ens_test[target].reshape(-1, 1))[:, 1]
    
    raw_ll = log_loss(y, ens_oof[target])
    platt_ll = log_loss(y, platt_oof)
    raw_auc = roc_auc_score(y, ens_oof[target])
    platt_auc = roc_auc_score(y, platt_oof)
    
    print(f"  {target}:")
    print(f"    Raw:   LL={raw_ll:.6f}, AUC={raw_auc:.6f}")
    print(f"    Platt: LL={platt_ll:.6f}, AUC={platt_auc:.6f} {'BETTER' if platt_ll < raw_ll else 'same/worse'}")
    
    if platt_ll < raw_ll:
        platt_test[target] = platt_pred
    else:
        platt_test[target] = ens_test[target]

# ============================================================
# 7. POST-PROCESSING & SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Post-processing & submissions...")
print("=" * 70)

zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0
print(f"  Zero-topic test rows: {zero_topic_mask.sum()}")

def apply_postprocessing(preds_dict, name=""):
    """Apply clipping, monotonicity, zero-group rules"""
    processed = {}
    for target in TARGETS:
        processed[target] = np.clip(preds_dict[target], 0.001, 0.999)
    
    # Monotonicity: 7d <= 90d <= 120d
    processed['adopted_within_90_days'] = np.maximum(
        processed['adopted_within_07_days'], processed['adopted_within_90_days'])
    processed['adopted_within_120_days'] = np.maximum(
        processed['adopted_within_90_days'], processed['adopted_within_120_days'])
    
    # Zero-topic rule
    for target in TARGETS:
        processed[target][zero_topic_mask] = 0.001
    
    # Zero-group rules
    train_rules = df[train_idx].copy()
    for target in TARGETS:
        gstats = train_rules.groupby('group_name').agg(n=(target, 'count'), rate=(target, 'mean'))
        zg = gstats[(gstats['n'] >= 30) & (gstats['rate'] == 0)].index
        if len(zg) > 0:
            tv = df.loc[test_idx, 'group_name'].values
            zm = np.isin(tv, zg)
            if zm.sum() > 0:
                processed[target][zm] = np.minimum(processed[target][zm], 0.005)
                if name:
                    print(f"    [{name}] Zero-group {target}: {len(zg)} groups, {zm.sum()} rows -> cap 0.005")
    
    return processed

def make_dual_auc(preds):
    """Rank-based AUC predictions"""
    auc_preds = {}
    for target in TARGETS:
        ranks = pd.Series(preds[target]).rank(pct=True)
        auc = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
        auc[zero_topic_mask] = 0.001
        auc_preds[target] = auc.values if hasattr(auc, 'values') else auc
    return auc_preds

def apply_prior_blend(preds, alpha=0.7):
    """Blend with Prior farmer rates"""
    rate_map = {
        'adopted_within_07_days': 'prior_07_rate',
        'adopted_within_90_days': 'prior_90_rate',
        'adopted_within_120_days': 'prior_120_rate',
    }
    has_hist = df.loc[test_idx, 'prior_session_count'].values > 0
    
    blended = {}
    for target in TARGETS:
        raw = preds[target].copy()
        prior_rates = df.loc[test_idx, rate_map[target]].values
        b = raw.copy()
        b[has_hist] = alpha * raw[has_hist] + (1 - alpha) * prior_rates[has_hist]
        blended[target] = b
    return blended

def save_sub(preds_auc, preds_ll, filename):
    sub = pd.DataFrame({'ID': test_ids_ordered})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        sub[auc_col] = preds_auc[target]
        sub[ll_col] = preds_ll[target]
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    assert len(sub) == len(ss), f"Length mismatch: {len(sub)} vs {len(ss)}"
    assert sub.isnull().sum().sum() == 0, "NaN values found!"
    sub.to_csv(filename, index=False)
    
    for col in sub.columns[1:]:
        if 'LogLoss' in col:
            print(f"    {col}: mean={sub[col].mean():.6f}, <=0.005: {(sub[col]<=0.005).sum()}")
    print(f"  -> SAVED: {filename}")
    return sub

# === VARIANT 1: Ensemble DUAL + Prior blend ===
print("\n  === V1: Ensemble + Prior blend + DUAL (like D improved) ===")
raw_pp = apply_postprocessing({t: ens_test[t].copy() for t in TARGETS}, "V1raw")
blend1 = apply_prior_blend(raw_pp, alpha=0.7)
blend1_pp = apply_postprocessing(blend1, "V1blend")
auc1 = make_dual_auc(blend1_pp)
sub1 = save_sub(auc1, blend1_pp, 'sub_REFv3_A_ens_blend_dual.csv')

# === VARIANT 2: Ensemble DUAL (no blend) ===
print("\n  === V2: Ensemble DUAL (no Prior blend) ===")
auc2 = make_dual_auc(raw_pp)
sub2 = save_sub(auc2, raw_pp, 'sub_REFv3_B_ens_dual.csv')

# === VARIANT 3: Platt-calibrated + Prior blend + DUAL ===
print("\n  === V3: Platt + Prior blend + DUAL ===")
platt_pp = apply_postprocessing({t: platt_test[t].copy() for t in TARGETS}, "V3raw")
platt_blend = apply_prior_blend(platt_pp, alpha=0.7)
platt_blend_pp = apply_postprocessing(platt_blend, "V3blend")
auc3 = make_dual_auc(platt_blend_pp)
sub3 = save_sub(auc3, platt_blend_pp, 'sub_REFv3_C_platt_blend_dual.csv')

# === VARIANT 4: 60% D (proven) + 40% new ensemble (safest!) ===
print("\n  === V4: 60% D (proven 0.918) + 40% new ensemble (SAFEST) ===")
d_sub = pd.read_csv('sub_COMP_D_dual_prior_blend.csv')
d_sub = d_sub.set_index('ID')

new_sub = pd.DataFrame({'ID': ss['ID'].values})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    d_ll = d_sub.loc[ss['ID'], ll_col].values
    new_ll = blend1_pp[target].copy()
    new_ll_s = pd.Series(new_ll, index=test_ids_ordered)
    new_ll_ordered = new_ll_s.loc[ss['ID']].values
    
    blended_ll = 0.6 * d_ll + 0.4 * new_ll_ordered
    blended_ll = np.clip(blended_ll, 0.001, 0.999)
    
    # Zero-topic
    zt_ss = pd.Series(zero_topic_mask, index=test_ids_ordered)
    zt_ordered = zt_ss.loc[ss['ID']].values
    blended_ll[zt_ordered] = 0.001
    
    ranks = pd.Series(blended_ll).rank(pct=True)
    blended_auc = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
    blended_auc[zt_ordered] = 0.001
    
    new_sub[auc_col] = blended_auc
    new_sub[ll_col] = blended_ll

new_sub = new_sub[SS_COLS]
new_sub.to_csv('sub_REFv3_D_blend60D_40new.csv', index=False)
for col in new_sub.columns[1:]:
    if 'LogLoss' in col:
        print(f"    {col}: mean={new_sub[col].mean():.6f}, <=0.005: {(new_sub[col]<=0.005).sum()}")
print(f"  -> SAVED: sub_REFv3_D_blend60D_40new.csv")

# === VARIANT 5: 50% D + 50% new (more aggressive) ===
print("\n  === V5: 50% D + 50% new ensemble ===")
new_sub5 = pd.DataFrame({'ID': ss['ID'].values})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    d_ll = d_sub.loc[ss['ID'], ll_col].values
    new_ll = blend1_pp[target].copy()
    new_ll_s = pd.Series(new_ll, index=test_ids_ordered)
    new_ll_ordered = new_ll_s.loc[ss['ID']].values
    
    blended_ll = 0.5 * d_ll + 0.5 * new_ll_ordered
    blended_ll = np.clip(blended_ll, 0.001, 0.999)
    
    zt_ss = pd.Series(zero_topic_mask, index=test_ids_ordered)
    zt_ordered = zt_ss.loc[ss['ID']].values
    blended_ll[zt_ordered] = 0.001
    
    ranks = pd.Series(blended_ll).rank(pct=True)
    blended_auc = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
    blended_auc[zt_ordered] = 0.001
    
    new_sub5[auc_col] = blended_auc
    new_sub5[ll_col] = blended_ll

new_sub5 = new_sub5[SS_COLS]
new_sub5.to_csv('sub_REFv3_E_blend50D_50new.csv', index=False)
for col in new_sub5.columns[1:]:
    if 'LogLoss' in col:
        print(f"    {col}: mean={new_sub5[col].mean():.6f}, <=0.005: {(new_sub5[col]<=0.005).sum()}")
print(f"  -> SAVED: sub_REFv3_E_blend50D_50new.csv")

# ============================================================
# 8. COMPARE WITH D
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON WITH D (proven 0.91788)")
print("=" * 70)

for ll_col in ['Target_07_LogLoss', 'Target_90_LogLoss', 'Target_120_LogLoss']:
    d_vals = d_sub.loc[ss['ID'], ll_col].values
    v1_vals = sub1.set_index('ID').loc[ss['ID'], ll_col].values
    
    print(f"\n  {ll_col}:")
    print(f"    D mean: {d_vals.mean():.6f},   V1 mean: {v1_vals.mean():.6f}")
    print(f"    D <=0.005: {(d_vals<=0.005).sum()},  V1 <=0.005: {(v1_vals<=0.005).sum()}")
    corr = np.corrcoef(d_vals, v1_vals)[0, 1]
    print(f"    Correlation: {corr:.4f}")

# Top features
print("\n  Top 20 features (v1 model):")
for target in TARGETS:
    print(f"\n  {target}:")
    for feat, imp in feature_importance[target].head(20).items():
        print(f"    {feat}: {imp:.1f}")

print("\n" + "=" * 70)
print("SUBMISSION PRIORITY (limited submissions!):")
print("=" * 70)
print("1. sub_REFv3_D_blend60D_40new.csv   SAFEST: 60% proven D + 40% new ensemble")
print("2. sub_REFv3_A_ens_blend_dual.csv   NEW: Full ensemble + Prior blend + DUAL")
print("3. sub_REFv3_E_blend50D_50new.csv   50/50 blend of D + new ensemble")
print("4. sub_REFv3_C_platt_blend_dual.csv Platt-calibrated version")
print("5. sub_REFv3_B_ens_dual.csv         Ensemble DUAL (no Prior blend)")
print("\nDONE!")
