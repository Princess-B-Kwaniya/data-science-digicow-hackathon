"""
CompetitiveSolution.py - Full ML Pipeline for DigiCow Hackathon (NEW DATA v2)
==============================================================================
KEY INSIGHT: Prior.csv has 44,882 rows with farmer history.
  - 62.7% of TEST farmers appear in Prior (3,526/5,621)
  - Avg 8.1 prior sessions per test farmer
  - Prior adoption rates are HIGHER than train rates
  - This is the #1 source of predictive signal

Data: Train 13,536 rows, Test 5,621 rows, Prior 44,882 rows
Targets: 7d (~1.1%), 90d (~1.6%), 120d (~2.2%) - EXTREME imbalance
Metric: Per target = 0.75*LogLoss + 0.25*(1-AUC), sum across 3. HIGHER=BETTER
SS cols: ID, Target_07_AUC, Target_90_AUC, Target_120_AUC, Target_07_LogLoss, Target_90_LogLoss, Target_120_LogLoss
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
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Loading data...")
print("=" * 70)

train_df = pd.read_csv('Train.csv')
test_df  = pd.read_csv('Test.csv')
prior_df = pd.read_csv('Prior.csv')
ss       = pd.read_csv('SampleSubmission.csv')

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Prior: {prior_df.shape}, SS: {ss.shape}")

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
SS_COLS = list(ss.columns)
TARGET_TO_SS = {
    'adopted_within_07_days':  ('Target_07_AUC',  'Target_07_LogLoss'),
    'adopted_within_90_days':  ('Target_90_AUC',  'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

for t in TARGETS:
    print(f"  Train {t}: {train_df[t].mean():.4f} ({train_df[t].sum()}/{len(train_df)})")
    print(f"  Prior {t}: {prior_df[t].mean():.4f} ({prior_df[t].sum()}/{len(prior_df)})")

# ============================================================
# 2. PARSE FORMATS & HARMONIZE PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Parsing & harmonizing formats...")
print("=" * 70)

def parse_topics_nested(s):
    """Parse Train/Test topics_list: nested list like [['t1','t2'],['t3']]"""
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
    """Parse Prior topics_list: flat list like ['t1','t2']"""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return list(set(str(x) for x in parsed))
        return [str(parsed)]
    except:
        return []

def parse_trainer_list(s):
    """Parse Train/Test trainer: list like ['TRA_xxx']"""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return str(parsed)
    except:
        return str(s)

def count_sessions_nested(s):
    """Count sessions from Train/Test nested topics_list"""
    try:
        parsed = ast.literal_eval(s)
        return len(parsed)
    except:
        return 1

# --- Parse Train/Test ---
print("  Parsing Train topics (nested)...")
train_df['topics_parsed'] = train_df['topics_list'].apply(parse_topics_nested)
train_df['trainer_parsed'] = train_df['trainer'].apply(parse_trainer_list)
train_df['num_sessions'] = train_df['topics_list'].apply(count_sessions_nested)

print("  Parsing Test topics (nested)...")
test_df['topics_parsed'] = test_df['topics_list'].apply(parse_topics_nested)
test_df['trainer_parsed'] = test_df['trainer'].apply(parse_trainer_list)
test_df['num_sessions'] = test_df['topics_list'].apply(count_sessions_nested)

# --- Parse Prior ---
print("  Parsing Prior topics (flat) & trainer (plain string)...")
prior_df['topics_parsed'] = prior_df['topics_list'].apply(parse_topics_flat)
prior_df['trainer_parsed'] = prior_df['trainer']  # Already plain string
prior_df['num_sessions'] = 1  # Prior = 1 session per row

print(f"  Train topics per row: mean={train_df['topics_parsed'].apply(len).mean():.1f}")
print(f"  Prior topics per row: mean={prior_df['topics_parsed'].apply(len).mean():.1f}")

# ============================================================
# 3. BUILD FARMER HISTORY FROM PRIOR (KEY FEATURE SOURCE)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building farmer history from Prior...")
print("=" * 70)

# Farmer-level aggregations from Prior
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

# Topic diversity from prior
topic_diversity = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))
).reset_index()
topic_diversity.columns = ['farmer_name', 'prior_unique_topics']
farmer_hist = farmer_hist.merge(topic_diversity, on='farmer_name', how='left')

# Date features from prior
prior_df['training_day_dt'] = pd.to_datetime(prior_df['training_day'])
date_feats = prior_df.groupby('farmer_name')['training_day_dt'].agg(
    prior_first_date='min',
    prior_last_date='max',
).reset_index()
date_feats['prior_training_span_days'] = (date_feats['prior_last_date'] - date_feats['prior_first_date']).dt.days
farmer_hist = farmer_hist.merge(date_feats[['farmer_name', 'prior_training_span_days']], on='farmer_name', how='left')

# Ever adopted flag
farmer_hist['prior_ever_adopted_07'] = (farmer_hist['prior_07_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_90'] = (farmer_hist['prior_90_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_120'] = (farmer_hist['prior_120_adopted'] > 0).astype(int)
farmer_hist['prior_any_adoption'] = ((farmer_hist['prior_07_adopted'] + farmer_hist['prior_90_adopted'] + farmer_hist['prior_120_adopted']) > 0).astype(int)

# Adoption intensity - weighted average
farmer_hist['prior_adoption_score'] = (
    farmer_hist['prior_07_rate'] * 3 +  # 7d adoption = strongest signal
    farmer_hist['prior_90_rate'] * 2 +
    farmer_hist['prior_120_rate'] * 1
) / 6

print(f"  Farmer history features: {farmer_hist.shape[1]-1} features for {len(farmer_hist)} farmers")

# Coverage
train_coverage = train_df['farmer_name'].isin(farmer_hist['farmer_name']).mean()
test_coverage = test_df['farmer_name'].isin(farmer_hist['farmer_name']).mean()
print(f"  Train farmer coverage from Prior: {train_coverage:.1%}")
print(f"  Test farmer coverage from Prior: {test_coverage:.1%}")

# ============================================================
# 4. BUILD GROUP HISTORY FROM PRIOR
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

# Group adoption score
group_hist['prior_grp_adoption_score'] = (
    group_hist['prior_grp_07_rate'] * 3 +
    group_hist['prior_grp_90_rate'] * 2 +
    group_hist['prior_grp_120_rate'] * 1
) / 6

train_grp_cov = train_df['group_name'].isin(group_hist['group_name']).mean()
test_grp_cov = test_df['group_name'].isin(group_hist['group_name']).mean()
print(f"  Group history: {group_hist.shape[1]-1} features for {len(group_hist)} groups")
print(f"  Train group coverage: {train_grp_cov:.1%}")
print(f"  Test group coverage: {test_grp_cov:.1%}")

# ============================================================
# 5. BUILD WARD/GEO HISTORY FROM PRIOR
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Building geographic history from Prior...")
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
    
    cov = test_df[f'prior_{geo_col}_size'].notna().mean()
    print(f"  {geo_col}: {len(geo_hist)} values, test coverage={cov:.1%}")

# ============================================================
# 6. MERGE FARMER & GROUP HISTORY
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Merging history features...")
print("=" * 70)

train_df = train_df.merge(farmer_hist, on='farmer_name', how='left')
test_df = test_df.merge(farmer_hist, on='farmer_name', how='left')

train_df = train_df.merge(group_hist, on='group_name', how='left')
test_df = test_df.merge(group_hist, on='group_name', how='left')

# Fill NaN for farmers/groups not in Prior
hist_cols = [c for c in farmer_hist.columns if c != 'farmer_name'] + \
            [c for c in group_hist.columns if c != 'group_name']
for c in hist_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(0)

has_hist_train = (train_df['prior_session_count'] > 0).sum()
has_hist_test = (test_df['prior_session_count'] > 0).sum()
print(f"  Train rows with farmer history: {has_hist_train}/{len(train_df)} ({has_hist_train/len(train_df):.1%})")
print(f"  Test rows with farmer history: {has_hist_test}/{len(test_df)} ({has_hist_test/len(test_df):.1%})")

# ============================================================
# 7. FEATURE ENGINEERING ON TRAIN+TEST
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Feature Engineering on Train+Test...")
print("=" * 70)

# Combine
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
df['is_month_start'] = df['training_day_dt'].dt.is_month_start.astype(int)
df['is_month_end'] = df['training_day_dt'].dt.is_month_end.astype(int)
df['days_since_epoch'] = (df['training_day_dt'] - pd.Timestamp('2024-01-01')).dt.days

# Cyclic
df['month_sin'] = np.sin(2 * np.pi * df['train_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['train_month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['train_dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['train_dayofweek'] / 7)

# Kenya seasons
def get_season(m):
    if m in [3, 4, 5]: return 0
    elif m in [6, 7, 8]: return 1
    elif m in [10, 11]: return 2
    elif m in [12, 1, 2]: return 3
    else: return 4
df['season'] = df['train_month'].apply(get_season)

# Days since farmer's last prior session (if available)
prior_last_dates = prior_df.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last_dates.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last_dates else -1, axis=1)
df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)

# --- 7B. TOPIC FEATURES ---
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

# --- 7E. GROUP FEATURES (from current data) ---
print("  7E. Group features (current data)...")
df['group_size'] = df.groupby('group_name')['group_name'].transform('count')
df['group_coop_rate'] = df.groupby('group_name')['belong_to_cooperative'].transform('mean')
df['group_female_rate'] = df.groupby('group_name')['gender'].transform(lambda x: (x == 'Female').mean())
df['group_young_rate'] = df.groupby('group_name')['age'].transform(lambda x: (x == 'Below 35').mean())
df['group_ussd_rate'] = df.groupby('group_name')['registration'].transform(lambda x: (x == 'Ussd').mean())
df['group_topic_diversity'] = df.groupby('group_name')['primary_topic_cat'].transform('nunique')
df['group_trainer_diversity'] = df.groupby('group_name')['trainer_parsed'].transform('nunique')
df['group_has_topic_rate'] = df.groupby('group_name')['has_topic_trained_on'].transform('mean')
df['group_session_mean'] = df.groupby('group_name')['num_sessions'].transform('mean')

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
print("  7H. Farmer history interaction features...")
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

# --- 7J. TARGET ENCODING (OOF) ---
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
        
        # OOF for train
        for fold_idx, (tr_idx, val_idx) in enumerate(skf_te.split(train_data, train_data[target])):
            fold_train = train_data.iloc[tr_idx]
            fold_val_indices = train_data.iloc[val_idx].index
            
            stats = fold_train.groupby(col)[target].agg(['sum', 'count'])
            smoothed = (stats['sum'] + SMOOTHING * global_mean) / (stats['count'] + SMOOTHING)
            
            df.loc[fold_val_indices, te_col_name] = train_data.loc[fold_val_indices, col].map(smoothed)
        
        # Full train stats for test
        stats_all = train_data.groupby(col)[target].agg(['sum', 'count'])
        smoothed_all = (stats_all['sum'] + SMOOTHING * global_mean) / (stats_all['count'] + SMOOTHING)
        
        test_mask = df['is_train'] == 0
        df.loc[test_mask, te_col_name] = df.loc[test_mask, col].map(smoothed_all)
        df[te_col_name] = df[te_col_name].fillna(global_mean)

print(f"  Target encoding: {len(te_cols)} cols x {len(TARGETS)} targets = {len(te_cols) * len(TARGETS)} features")

# --- 7K. PRIOR-BASED TARGET ENCODING (using Prior data directly) ---
print("  7K. Prior-based target encoding...")
PRIOR_SMOOTH = 20  # heavier smoothing for prior
for target in TARGETS:
    prior_global = prior_df[target].mean()
    
    for col in ['group_name', 'ward', 'subcounty', 'county']:
        prior_stats = prior_df.groupby(col)[target].agg(['sum', 'count'])
        prior_smoothed = (prior_stats['sum'] + PRIOR_SMOOTH * prior_global) / (prior_stats['count'] + PRIOR_SMOOTH)
        
        prior_te_col = f'prior_te_{col}_{target}'
        df[prior_te_col] = df[col].map(prior_smoothed).fillna(prior_global)

print(f"  Prior TE: 4 cols x {len(TARGETS)} targets = {4 * len(TARGETS)} features")

# ============================================================
# 8. PREPARE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Preparing features for modeling...")
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

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 9. TRAIN LIGHTGBM (with Prior-augmented features)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Training LightGBM models...")
print("=" * 70)

N_FOLDS = 5
SEED = 42

lgb_params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'n_estimators': 3000,
    'verbose': -1,
}

all_test_preds = {}
cv_scores = {}
all_models = {}
oof_preds_all = {}

for target in TARGETS:
    print(f"\n{'='*50}")
    print(f"Training for: {target}")
    print(f"{'='*50}")
    
    y = y_train[target]
    pos_rate = y.mean()
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    print(f"  Positive rate: {pos_rate:.4f}, scale_pos_weight: {pos_weight:.1f}")
    
    params = lgb_params.copy()
    params['scale_pos_weight'] = pos_weight
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_models = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")
        
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_ll = log_loss(y_val, oof_preds[val_idx])
        fold_models.append(model)
        print(f"    AUC: {fold_auc:.6f}, LogLoss: {fold_ll:.6f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    overall_ll = log_loss(y, oof_preds)
    comp_score = 0.75 * (1 - overall_ll) + 0.25 * overall_auc
    
    cv_scores[target] = {'auc': overall_auc, 'logloss': overall_ll, 'comp': comp_score}
    all_test_preds[target] = test_preds
    all_models[target] = fold_models
    oof_preds_all[target] = oof_preds
    
    print(f"\n  {target} CV: AUC={overall_auc:.6f}, LL={overall_ll:.6f}, Comp={comp_score:.6f}")

total_comp = sum(cv_scores[t]['comp'] for t in TARGETS)
print(f"\n{'='*50}")
print(f"ESTIMATED TOTAL SCORE: {total_comp:.6f}")
print(f"{'='*50}")

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Feature importance (top 20 per target)...")
print("=" * 70)

for target in TARGETS:
    importances = np.zeros(len(feature_cols))
    for model in all_models[target]:
        importances += model.feature_importances_ / len(all_models[target])
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    print(f"\n  {target}:")
    for _, row in imp_df.head(20).iterrows():
        print(f"    {row['feature']}: {row['importance']:.1f}")

# ============================================================
# 11. POST-PROCESSING & SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Post-processing and generating submissions...")
print("=" * 70)

# Clip predictions
for target in TARGETS:
    all_test_preds[target] = np.clip(all_test_preds[target], 0.001, 0.999)

# Enforce monotonicity: 120d >= 90d >= 7d
pred_07 = all_test_preds['adopted_within_07_days'].copy()
pred_90 = all_test_preds['adopted_within_90_days'].copy()
pred_120 = all_test_preds['adopted_within_120_days'].copy()
pred_90 = np.maximum(pred_07, pred_90)
pred_120 = np.maximum(pred_90, pred_120)
all_test_preds['adopted_within_90_days'] = pred_90
all_test_preds['adopted_within_120_days'] = pred_120

# Zero-group rules
print("  Applying zero-group rules...")
zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0
print(f"    has_topic_trained_on=0 in test: {zero_topic_mask.sum()} rows")
for target in TARGETS:
    all_test_preds[target][zero_topic_mask] = 0.001

# Check for zero-adoption groups in train
train_data_rules = df[train_idx].copy()
for group_col in ['group_name']:
    for target in TARGETS:
        le = label_encoders.get(group_col, None)
        group_stats = train_data_rules.groupby(group_col).agg(
            n=(target, 'count'), rate=(target, 'mean'))
        zero_groups = group_stats[(group_stats['n'] >= 30) & (group_stats['rate'] == 0)].index
        if len(zero_groups) > 0:
            test_vals = df.loc[test_idx, group_col].values
            zmask = np.isin(test_vals, zero_groups)
            if zmask.sum() > 0:
                print(f"    {group_col} zero for {target}: {len(zero_groups)} groups, {zmask.sum()} test rows")
                all_test_preds[target][zmask] = np.minimum(all_test_preds[target][zmask], 0.005)

def create_submission(preds, test_ids, filename):
    """Create submission with correct SS column order"""
    sub = pd.DataFrame({'ID': test_ids})
    for target, (auc_col, ll_col) in TARGET_TO_SS.items():
        sub[auc_col] = preds[target]
        sub[ll_col] = preds[target]
    sub = sub[SS_COLS]
    sub = sub.set_index('ID').loc[ss['ID']].reset_index()
    
    assert len(sub) == len(ss), f"Row count: {len(sub)} vs {len(ss)}"
    assert list(sub.columns) == list(ss.columns), f"Cols mismatch"
    assert sub.isnull().sum().sum() == 0, "NaN found!"
    assert set(sub['ID']) == set(ss['ID']), "ID mismatch!"
    
    sub.to_csv(filename, index=False)
    print(f"  SAVED: {filename} ({len(sub)} rows, validated OK)")
    return sub

# --- Submission A: Standard LightGBM ---
sub_a = create_submission(all_test_preds, test_ids_ordered, 'sub_COMP_A_lgbm_standard.csv')

# --- Submission B: DUAL strategy (rank AUC + calibrated LL) ---
sub_dual = pd.DataFrame({'ID': test_ids_ordered})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    raw = all_test_preds[target].copy()
    ranks = pd.Series(raw).rank(pct=True)
    auc_preds = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
    auc_preds[zero_topic_mask] = 0.001
    sub_dual[auc_col] = auc_preds
    sub_dual[ll_col] = raw

sub_dual = sub_dual[SS_COLS]
sub_dual = sub_dual.set_index('ID').loc[ss['ID']].reset_index()
assert len(sub_dual) == len(ss)
assert sub_dual.isnull().sum().sum() == 0
sub_dual.to_csv('sub_COMP_B_lgbm_dual.csv', index=False)
print(f"  SAVED: sub_COMP_B_lgbm_dual.csv ({len(sub_dual)} rows, validated OK)")

# --- Submission C: Blend with Prior rates for covered farmers ---
blend_preds = {}
for target in TARGETS:
    raw = all_test_preds[target].copy()
    # For farmers with prior history, blend model preds with prior adoption rate
    prior_rate_col = f'prior_{target.replace("adopted_within_", "").replace("_days", "")}_rate'
    # Map the right column
    rate_map = {
        'adopted_within_07_days': 'prior_07_rate',
        'adopted_within_90_days': 'prior_90_rate',
        'adopted_within_120_days': 'prior_120_rate',
    }
    prior_rates = df.loc[test_idx, rate_map[target]].values
    has_hist = df.loc[test_idx, 'prior_session_count'].values > 0
    
    # Blend: 70% model + 30% prior rate for farmers with history
    blended = raw.copy()
    blended[has_hist] = 0.7 * raw[has_hist] + 0.3 * prior_rates[has_hist]
    blended = np.clip(blended, 0.001, 0.999)
    blended[zero_topic_mask] = 0.001
    blend_preds[target] = blended

# Enforce monotonicity on blend
bp07 = blend_preds['adopted_within_07_days']
bp90 = blend_preds['adopted_within_90_days']
bp120 = blend_preds['adopted_within_120_days']
bp90 = np.maximum(bp07, bp90)
bp120 = np.maximum(bp90, bp120)
blend_preds['adopted_within_90_days'] = bp90
blend_preds['adopted_within_120_days'] = bp120

sub_c = create_submission(blend_preds, test_ids_ordered, 'sub_COMP_C_blend_prior.csv')

# --- Submission D: DUAL with Prior blend ---
sub_dual_blend = pd.DataFrame({'ID': test_ids_ordered})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    raw = blend_preds[target].copy()
    ranks = pd.Series(raw).rank(pct=True)
    auc_preds = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
    auc_preds[zero_topic_mask] = 0.001
    sub_dual_blend[auc_col] = auc_preds
    sub_dual_blend[ll_col] = raw

sub_dual_blend = sub_dual_blend[SS_COLS]
sub_dual_blend = sub_dual_blend.set_index('ID').loc[ss['ID']].reset_index()
assert len(sub_dual_blend) == len(ss)
assert sub_dual_blend.isnull().sum().sum() == 0
sub_dual_blend.to_csv('sub_COMP_D_dual_prior_blend.csv', index=False)
print(f"  SAVED: sub_COMP_D_dual_prior_blend.csv ({len(sub_dual_blend)} rows, validated OK)")

# ============================================================
# 12. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Features: {len(feature_cols)}")
print(f"Prior coverage: {test_coverage:.1%} of test farmers")
for t in TARGETS:
    s = cv_scores[t]
    print(f"  {t}: AUC={s['auc']:.6f}, LL={s['logloss']:.6f}, Comp={s['comp']:.6f}")
print(f"\nEstimated total score: {total_comp:.6f}")
print(f"\nSubmissions:")
print(f"  1. sub_COMP_A_lgbm_standard.csv     - Standard (same AUC/LL)")
print(f"  2. sub_COMP_B_lgbm_dual.csv         - DUAL (rank AUC + calibrated LL)")
print(f"  3. sub_COMP_C_blend_prior.csv        - 70/30 blend with prior farmer rates")
print(f"  4. sub_COMP_D_dual_prior_blend.csv   - DUAL on prior blend")
print("DONE!")
