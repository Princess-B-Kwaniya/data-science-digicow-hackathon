"""
ENHANCED_V2.py - Maximum Power Pipeline for DigiCow
====================================================
Key improvements over CompetitiveSolution.py:
1. Train on Prior + Train combined (58K rows vs 13.5K)
2. Harmonize Prior format to match Train
3. Multiple model configs + blending
4. Stronger farmer history features
5. Better DUAL calibration

Current best: D = 0.91788 (DUAL + Prior blend)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import ast
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = r"C:\Users\USER\Desktop\DIGICOW"
os.chdir(DATA_DIR)

# ============================================================
# 1. LOAD & HARMONIZE DATA
# ============================================================
print("=" * 70)
print("STEP 1: Loading and harmonizing all data...")
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

# --- Harmonize Prior to match Train format ---
# Prior topics_list: flat ['t1','t2'] -> wrap to nested [['t1','t2']]
def harmonize_prior_topics(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], str):
                return str([parsed])  # wrap in outer list
            return s
        return s
    except:
        return s

print("  Harmonizing Prior topics format...")
prior_df['topics_list'] = prior_df['topics_list'].apply(harmonize_prior_topics)

# Prior trainer: plain 'TRA_xxx' -> "['TRA_xxx']"
print("  Harmonizing Prior trainer format...")
prior_df['trainer'] = prior_df['trainer'].apply(
    lambda x: str([x]) if not str(x).startswith('[') else x)

# --- Parse all data consistently ---
def parse_topics(s):
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
        parsed = ast.literal_eval(s)
        return len(parsed)
    except:
        return 1

for df_name, data in [('train', train_df), ('test', test_df), ('prior', prior_df)]:
    data['topics_parsed'] = data['topics_list'].apply(parse_topics)
    data['trainer_parsed'] = data['trainer'].apply(parse_trainer)
    data['num_sessions'] = data['topics_list'].apply(count_sessions)
    print(f"  {df_name}: parsed OK")

# ============================================================
# 2. FARMER HISTORY (from Prior only - separate from training)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Building farmer history from Prior...")
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

# Recency features
farmer_hist = farmer_hist.merge(
    date_feats[['farmer_name', 'prior_last_date']], on='farmer_name', how='left')

print(f"  {len(farmer_hist)} farmers, {farmer_hist.shape[1]-1} features")
print(f"  Test coverage: {test_df['farmer_name'].isin(farmer_hist['farmer_name']).mean():.1%}")

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

# ============================================================
# 3. COMBINE PRIOR + TRAIN FOR MEGA TRAINING SET
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building mega training set (Prior + Train)...")
print("=" * 70)

# Mark source
prior_df['data_source'] = 'prior'
train_df['data_source'] = 'train'
test_df['data_source'] = 'test'

# Add targets placeholder for test
for t in TARGETS:
    if t not in test_df.columns:
        test_df[t] = np.nan

# Combine Prior + Train as training data, Test separate
mega_train = pd.concat([prior_df, train_df], axis=0, ignore_index=True)
print(f"  Mega train: {len(mega_train)} rows ({len(prior_df)} Prior + {len(train_df)} Train)")

for t in TARGETS:
    print(f"  Mega {t}: {mega_train[t].mean():.4f} ({mega_train[t].sum()}/{len(mega_train)})")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Feature Engineering...")
print("=" * 70)

# Combine all for consistent feature engineering
mega_train['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([mega_train, test_df], axis=0, ignore_index=True)
train_idx = df['is_train'] == 1
test_idx = df['is_train'] == 0

# --- Temporal ---
print("  Temporal...")
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

# Days since last prior session
prior_last = prior_df.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last else -1, axis=1)

# --- Topics ---
print("  Topics...")
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

# --- Geographic interactions ---
print("  Geographic interactions...")
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

# --- Demographic interactions ---
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

# --- Merge farmer/group/geo history ---
print("  Merging history features...")
df = df.merge(farmer_hist.drop(columns=['prior_last_date'], errors='ignore'), on='farmer_name', how='left')
df = df.merge(group_hist, on='group_name', how='left')
for geo_col in ['ward', 'subcounty', 'county']:
    df = df.merge(geo_hists[geo_col], on=geo_col, how='left')

# Fill NaN history
hist_fill_cols = [c for c in df.columns if c.startswith('prior_')]
for c in hist_fill_cols:
    df[c] = df[c].fillna(0)

df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)
df['hist_sessions_x_topics'] = df['prior_session_count'] * df['topic_count']
df['hist_adoption_x_hastopic'] = df['prior_adoption_score'] * df['has_topic_trained_on']
df['hist_grp_x_farmer_adoption'] = df['prior_grp_adoption_score'] * df['prior_adoption_score']
df['hist_ever_adopted_x_hastopic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['farmer_high_engagement'] = (df['prior_session_count'] >= 5).astype(int)

# --- Contextual ---
print("  Contextual...")
df['topic_season'] = df['primary_topic_cat'] + '_' + df['season'].astype(str)
df['county_season'] = df['county'] + '_' + df['season'].astype(str)

# --- Aggregations ---
print("  Aggregations...")
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

# --- Target encoding (OOF on mega train) ---
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

print(f"  TE: {len(te_cols)} x {len(TARGETS)} = {len(te_cols) * len(TARGETS)} features")

# --- is_from_prior indicator ---
df['is_from_prior'] = (df['data_source'] == 'prior').astype(int)

# ============================================================
# 5. PREPARE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Preparing features...")
print("=" * 70)

exclude_cols = ['ID', 'farmer_name', 'is_train', 'training_day', 'training_day_dt',
                'topics_list', 'topics_parsed', 'topic_cats', 'trainer',
                'data_source'] + TARGETS

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
# 6. TRAIN MULTIPLE LGBM CONFIGS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Training LightGBM (multiple configs)...")
print("=" * 70)

N_FOLDS = 5
SEED = 42

configs = {
    'main': {
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.03,
        'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'max_depth': -1, 'min_child_samples': 20,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': SEED, 'n_estimators': 3000, 'verbose': -1,
    },
    'deep': {
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 127, 'learning_rate': 0.02,
        'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 3,
        'max_depth': 8, 'min_child_samples': 30,
        'reg_alpha': 0.5, 'reg_lambda': 2.0,
        'random_state': SEED + 1, 'n_estimators': 3000, 'verbose': -1,
    },
    'shallow': {
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5,
        'max_depth': 5, 'min_child_samples': 50,
        'reg_alpha': 0.05, 'reg_lambda': 0.5,
        'random_state': SEED + 2, 'n_estimators': 3000, 'verbose': -1,
    },
}

all_config_preds = {}
all_config_oof = {}
best_cv = {}

for config_name, base_params in configs.items():
    print(f"\n  --- Config: {config_name} ---")
    config_preds = {}
    config_oof = {}
    
    for target in TARGETS:
        y = y_train[target]
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        params = base_params.copy()
        params['scale_pos_weight'] = pos_weight
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
            
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        auc = roc_auc_score(y, oof_preds)
        ll = log_loss(y, oof_preds)
        comp = 0.75 * (1 - ll) + 0.25 * auc
        print(f"    {target}: AUC={auc:.6f}, LL={ll:.6f}, Comp={comp:.6f}")
        
        config_preds[target] = test_preds
        config_oof[target] = oof_preds
    
    total = sum(0.75*(1-log_loss(y_train[t], config_oof[t])) + 0.25*roc_auc_score(y_train[t], config_oof[t]) for t in TARGETS)
    print(f"    TOTAL: {total:.6f}")
    
    all_config_preds[config_name] = config_preds
    all_config_oof[config_name] = config_oof
    best_cv[config_name] = total

# ============================================================
# 7. ENSEMBLE CONFIGS
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Finding best ensemble weights...")
print("=" * 70)

# Try different blend weights
best_blend_score = -1
best_weights = None

weight_options = [
    {'main': 0.5, 'deep': 0.25, 'shallow': 0.25},
    {'main': 0.6, 'deep': 0.2, 'shallow': 0.2},
    {'main': 0.4, 'deep': 0.3, 'shallow': 0.3},
    {'main': 0.7, 'deep': 0.15, 'shallow': 0.15},
    {'main': 0.33, 'deep': 0.33, 'shallow': 0.34},
]

for weights in weight_options:
    total = 0
    for target in TARGETS:
        blended_oof = sum(weights[c] * all_config_oof[c][target] for c in configs)
        auc = roc_auc_score(y_train[target], blended_oof)
        ll = log_loss(y_train[target], blended_oof)
        total += 0.75 * (1 - ll) + 0.25 * auc
    
    if total > best_blend_score:
        best_blend_score = total
        best_weights = weights
    print(f"  Weights {weights}: {total:.6f}")

print(f"\n  Best blend: {best_weights} -> {best_blend_score:.6f}")

# Build ensemble predictions
ensemble_preds = {}
for target in TARGETS:
    ensemble_preds[target] = sum(best_weights[c] * all_config_preds[c][target] for c in configs)

# ============================================================
# 8. POST-PROCESSING
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Post-processing...")
print("=" * 70)

for target in TARGETS:
    ensemble_preds[target] = np.clip(ensemble_preds[target], 0.001, 0.999)

# Monotonicity
p07 = ensemble_preds['adopted_within_07_days'].copy()
p90 = ensemble_preds['adopted_within_90_days'].copy()
p120 = ensemble_preds['adopted_within_120_days'].copy()
p90 = np.maximum(p07, p90)
p120 = np.maximum(p90, p120)
ensemble_preds['adopted_within_90_days'] = p90
ensemble_preds['adopted_within_120_days'] = p120

# Zero-group rules
zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0
print(f"  has_topic_trained_on=0: {zero_topic_mask.sum()} test rows")
for target in TARGETS:
    ensemble_preds[target][zero_topic_mask] = 0.001

# Zero-adoption groups from mega training
train_data_rules = df[train_idx].copy()
for target in TARGETS:
    group_stats = train_data_rules.groupby('group_name').agg(
        n=(target, 'count'), rate=(target, 'mean'))
    zero_groups = group_stats[(group_stats['n'] >= 30) & (group_stats['rate'] == 0)].index
    if len(zero_groups) > 0:
        test_vals = df.loc[test_idx, 'group_name'].values
        zmask = np.isin(test_vals, zero_groups)
        if zmask.sum() > 0:
            print(f"  Zero-group {target}: {len(zero_groups)} groups, {zmask.sum()} test rows -> cap at 0.005")
            ensemble_preds[target][zmask] = np.minimum(ensemble_preds[target][zmask], 0.005)

# ============================================================
# 9. GENERATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Generating submissions...")
print("=" * 70)

def save_submission(preds_dict, test_ids, filename, dual=False):
    sub = pd.DataFrame({'ID': test_ids})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        raw = preds_dict[target].copy()
        if dual:
            ranks = pd.Series(raw).rank(pct=True)
            auc_vals = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
            auc_vals[zero_topic_mask] = 0.001
            sub[auc_col] = auc_vals
            sub[ll_col] = raw
        else:
            sub[auc_col] = raw
            sub[ll_col] = raw
    
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    assert len(sub) == len(ss)
    assert list(sub.columns) == list(ss.columns)
    assert sub.isnull().sum().sum() == 0
    sub.to_csv(filename, index=False)
    print(f"  SAVED: {filename} ({len(sub)} rows)")
    return sub

# E: Ensemble Standard
save_submission(ensemble_preds, test_ids_ordered, 'sub_ENS_E_standard.csv', dual=False)

# F: Ensemble DUAL
save_submission(ensemble_preds, test_ids_ordered, 'sub_ENS_F_dual.csv', dual=True)

# G: Ensemble + Prior blend + DUAL (best strategy from before)
blend_preds = {}
rate_map = {
    'adopted_within_07_days': 'prior_07_rate',
    'adopted_within_90_days': 'prior_90_rate',
    'adopted_within_120_days': 'prior_120_rate',
}
for target in TARGETS:
    raw = ensemble_preds[target].copy()
    prior_rates = df.loc[test_idx, rate_map[target]].values
    has_hist = df.loc[test_idx, 'prior_session_count'].values > 0
    blended = raw.copy()
    blended[has_hist] = 0.7 * raw[has_hist] + 0.3 * prior_rates[has_hist]
    blended = np.clip(blended, 0.001, 0.999)
    blended[zero_topic_mask] = 0.001
    blend_preds[target] = blended

# Re-enforce monotonicity on blend
bp07 = blend_preds['adopted_within_07_days']
bp90 = np.maximum(bp07, blend_preds['adopted_within_90_days'])
bp120 = np.maximum(bp90, blend_preds['adopted_within_120_days'])
blend_preds['adopted_within_90_days'] = bp90
blend_preds['adopted_within_120_days'] = bp120

save_submission(blend_preds, test_ids_ordered, 'sub_ENS_G_dual_prior_blend.csv', dual=True)

# H: Best single config + DUAL (if one config is clearly better)
best_config = max(best_cv, key=best_cv.get)
print(f"\n  Best single config: {best_config} ({best_cv[best_config]:.6f})")
best_single_preds = {}
for target in TARGETS:
    p = all_config_preds[best_config][target].copy()
    p = np.clip(p, 0.001, 0.999)
    best_single_preds[target] = p

# Monotonicity
bs07 = best_single_preds['adopted_within_07_days'].copy()
bs90 = np.maximum(bs07, best_single_preds['adopted_within_90_days'].copy())
bs120 = np.maximum(bs90, best_single_preds['adopted_within_120_days'].copy())
best_single_preds['adopted_within_90_days'] = bs90
best_single_preds['adopted_within_120_days'] = bs120

for target in TARGETS:
    best_single_preds[target][zero_topic_mask] = 0.001

save_submission(best_single_preds, test_ids_ordered, 'sub_ENS_H_best_single_dual.csv', dual=True)

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Features: {len(feature_cols)}")
print(f"Training rows: {len(X_train)} (Prior + Train)")
print(f"\nConfig CV scores:")
for c, s in sorted(best_cv.items(), key=lambda x: -x[1]):
    print(f"  {c}: {s:.6f}")
print(f"\nBest ensemble: {best_blend_score:.6f} (weights: {best_weights})")
print(f"\nSubmissions:")
print(f"  sub_ENS_E_standard.csv         - Ensemble, same AUC/LL")
print(f"  sub_ENS_F_dual.csv             - Ensemble, DUAL strategy")
print(f"  sub_ENS_G_dual_prior_blend.csv - Ensemble + Prior blend + DUAL")
print(f"  sub_ENS_H_best_single_dual.csv - Best single config + DUAL")
print("\nRECOMMENDation: Submit G first (ensemble + prior + DUAL), it combines all winning strategies")
print("DONE!")
