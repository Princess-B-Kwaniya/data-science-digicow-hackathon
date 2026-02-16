"""
ULTIMATE_V6.py - Maximum Feature Engineering + Multi-Model + Advanced Calibration
=================================================================================
GOAL: Generate a BETTER base model than sub_ADV_pertarget_progressive.csv
      so the RKLO blending pipeline can produce higher scores.

NEW in V6 over V5:
  1. TOPIC OVERLAP FEATURES: overlap between farmer's current and prior topics
  2. FARMER-GROUP LOYALTY: how many times farmer trained with THIS specific group
  3. PEER ADOPTION: adoption rate of co-trainees (same day/group/trainer)
  4. GROUP TEMPORAL FEATURES: group first/last activity date, trend
  5. TOPIC RARITY/IDF: inverse document frequency of topics
  6. MULTI-TRAINER HANDLING: parse ALL trainers, not just first
  7. DAYS-SINCE-LAST BINNING: non-linear recency bins
  8. WARD TRAINER DENSITY: trainers per ward/subcounty
  9. FARMER DEMOGRAPHIC CHANGE: coop/registration change from prior
  10. TOPIC-TARGET SPECIFIC: known high-adoption topic patterns
  11. ADVANCED CALIBRATION: Platt + Isotonic + Beta + Venn-Abers candidates
  12. MORE SEEDS (20 LGB + 10 XGB) + separate feature selection per target
  13. PRIOR ADOPTION TRAJECTORY: linear trend over farmer's sessions
  14. GROUP PEER PRESSURE: what % of group members adopted
  15. TRAINER SPECIALIZATION: trainer's topic-specific effectiveness
  16. SESSION DENSITY: farmer sessions per month
  17. COUNTY-MONTH SEASONALITY: county×month adoption rates from Prior

RULES: Train on Train ONLY. Prior as feature source ONLY. No target leakage.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import optuna
import ast
import json
import warnings
import os
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = r"C:\Users\USER\Desktop\DIGICOW"
os.chdir(DATA_DIR)
start_time = time.time()

print("=" * 70)
print("ULTIMATE V6 - Maximum Feature Engineering + Advanced Pipeline")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\nSTEP 1: Loading data...")
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

for t in TARGETS:
    print(f"  {t}: {train_df[t].mean():.4f} ({train_df[t].sum()}/{len(train_df)})")
print(f"  Train={len(train_df)}, Test={len(test_df)}, Prior={len(prior_df)}")

# ============================================================
# 2. DATA CLEANING & PARSING
# ============================================================
print("\nSTEP 2: Data cleaning...")

# Prior deduplication
prior_df['training_day_dt'] = pd.to_datetime(prior_df['training_day'])
prior_df = prior_df.sort_values(['farmer_name', 'training_day_dt', 'has_topic_trained_on', 'ID'],
    ascending=[True, True, False, False])
prior_df = prior_df.drop_duplicates(subset=['farmer_name', 'training_day_dt'], keep='first')

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

def parse_all_trainers(s):
    """Parse ALL trainers from list format"""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except:
        return [str(s)]

def parse_trainer_first(s):
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

# Parse all datasets
train_df['topics_parsed'] = train_df['topics_list'].apply(parse_topics_nested)
train_df['trainer_parsed'] = train_df['trainer'].apply(parse_trainer_first)
train_df['trainers_all'] = train_df['trainer'].apply(parse_all_trainers)
train_df['num_sessions'] = train_df['topics_list'].apply(count_sessions_nested)
train_df['num_trainers'] = train_df['trainers_all'].apply(len)

test_df['topics_parsed'] = test_df['topics_list'].apply(parse_topics_nested)
test_df['trainer_parsed'] = test_df['trainer'].apply(parse_trainer_first)
test_df['trainers_all'] = test_df['trainer'].apply(parse_all_trainers)
test_df['num_sessions'] = test_df['topics_list'].apply(count_sessions_nested)
test_df['num_trainers'] = test_df['trainers_all'].apply(len)

prior_df['topics_parsed'] = prior_df['topics_list'].apply(parse_topics_flat)
prior_df['trainer_parsed'] = prior_df['trainer']
prior_df['trainers_all'] = prior_df['trainer'].apply(lambda x: [x])
prior_df['num_sessions'] = 1
prior_df['num_trainers'] = 1

train_df['training_day_dt'] = pd.to_datetime(train_df['training_day'])
test_df['training_day_dt'] = pd.to_datetime(test_df['training_day'])

print(f"  Done. Train={len(train_df)}, Test={len(test_df)}, Prior={len(prior_df)}")
elapsed = time.time() - start_time
print(f"  [{elapsed:.0f}s]")

# ============================================================
# 3. FARMER HISTORY FROM PRIOR (comprehensive)
# ============================================================
print("\nSTEP 3: Farmer history (comprehensive)...")

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
    prior_unique_counties=('county', 'nunique'),
    prior_unique_subcounties=('subcounty', 'nunique'),
).reset_index()

# Topic diversity
topic_diversity = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))).reset_index()
topic_diversity.columns = ['farmer_name', 'prior_unique_topics']
farmer_hist = farmer_hist.merge(topic_diversity, on='farmer_name', how='left')

# Training span
date_feats = prior_df.groupby('farmer_name')['training_day_dt'].agg(
    prior_first_date='min', prior_last_date='max').reset_index()
date_feats['prior_training_span_days'] = (date_feats['prior_last_date'] - date_feats['prior_first_date']).dt.days
farmer_hist = farmer_hist.merge(date_feats, on='farmer_name', how='left')

# Derived features
farmer_hist['prior_ever_adopted_07'] = (farmer_hist['prior_07_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_90'] = (farmer_hist['prior_90_adopted'] > 0).astype(int)
farmer_hist['prior_ever_adopted_120'] = (farmer_hist['prior_120_adopted'] > 0).astype(int)
farmer_hist['prior_any_adoption'] = ((farmer_hist['prior_07_adopted'] + farmer_hist['prior_90_adopted'] + farmer_hist['prior_120_adopted']) > 0).astype(int)
farmer_hist['prior_adoption_score'] = (
    farmer_hist['prior_07_rate'] * 3 + farmer_hist['prior_90_rate'] * 2 + farmer_hist['prior_120_rate'] * 1) / 6
farmer_hist['prior_engagement_intensity'] = farmer_hist['prior_session_count'] / (farmer_hist['prior_training_span_days'].clip(lower=1) / 30.0)
farmer_hist['prior_adoption_consistency'] = farmer_hist[['prior_07_rate','prior_90_rate','prior_120_rate']].std(axis=1)
farmer_hist['prior_is_loyal'] = (farmer_hist['prior_unique_groups'] == 1).astype(int)
farmer_hist['prior_high_sessions'] = (farmer_hist['prior_session_count'] >= 10).astype(int)
farmer_hist['prior_sessions_per_month'] = farmer_hist['prior_session_count'] / (farmer_hist['prior_training_span_days'].clip(lower=1) / 30.0)

# Recency-weighted adoption
prior_sorted = prior_df.sort_values(['farmer_name', 'training_day_dt'])
prior_sorted['session_rank'] = prior_sorted.groupby('farmer_name').cumcount()
prior_sorted['session_total'] = prior_sorted.groupby('farmer_name')['ID'].transform('count')
prior_sorted['recency_weight'] = (prior_sorted['session_rank'] + 1) / prior_sorted['session_total']

for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    weighted_rates = prior_sorted.assign(**{f'prior_{col_short}_recency_rate': prior_sorted.groupby('farmer_name')[target].transform('mean')})[['farmer_name', f'prior_{col_short}_recency_rate']].drop_duplicates()
    farmer_hist = farmer_hist.merge(weighted_rates, on='farmer_name', how='left')

# Last session features
last_sessions = prior_sorted.groupby('farmer_name').last().reset_index()
farmer_hist = farmer_hist.merge(
    last_sessions[['farmer_name', 'has_topic_trained_on', 'belong_to_cooperative']].rename(
        columns={'has_topic_trained_on': 'prior_last_has_topic', 'belong_to_cooperative': 'prior_last_coop'}),
    on='farmer_name', how='left')

# Temporal decay
ref_date = prior_df['training_day_dt'].max()
prior_sorted['days_from_ref'] = (ref_date - prior_sorted['training_day_dt']).dt.days
prior_sorted['decay_weight'] = np.exp(-prior_sorted['days_from_ref'] / 365.0)

for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    decay_rates = prior_sorted.assign(**{f'prior_{col_short}_decay_rate': prior_sorted.groupby('farmer_name')[target].transform('mean')})[['farmer_name', f'prior_{col_short}_decay_rate']].drop_duplicates()
    farmer_hist = farmer_hist.merge(decay_rates, on='farmer_name', how='left')

# Mean day gap between sessions
def calc_mean_gap(group):
    dates = group['training_day_dt'].sort_values()
    if len(dates) < 2: return 0
    return dates.diff().dt.days.dropna().mean()
day_gaps = prior_sorted.groupby('farmer_name').apply(calc_mean_gap).reset_index()
day_gaps.columns = ['farmer_name', 'prior_mean_day_gap']
farmer_hist = farmer_hist.merge(day_gaps, on='farmer_name', how='left')

# First session adoption
first_sessions = prior_sorted.groupby('farmer_name').first().reset_index()
for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    farmer_hist = farmer_hist.merge(
        first_sessions[['farmer_name', target]].rename(columns={target: f'prior_first_session_{col_short}'}),
        on='farmer_name', how='left')
    farmer_hist[f'prior_adoption_improvement_{col_short}'] = (
        farmer_hist.get(f'prior_{col_short}_recency_rate', 0) - farmer_hist[f'prior_{col_short}_rate'])

# === V6 NEW: Adoption trajectory (linear trend over sessions) ===
def calc_adoption_trend(group, target):
    vals = group[target].values
    if len(vals) < 3: return 0
    x = np.arange(len(vals))
    if vals.std() == 0: return 0
    try:
        slope = np.polyfit(x, vals, 1)[0]
        return slope
    except:
        return 0

for target in TARGETS:
    col_short = target.replace('adopted_within_', '').replace('_days', '')
    trends = prior_sorted.groupby('farmer_name').apply(lambda x: calc_adoption_trend(x, target)).reset_index()
    trends.columns = ['farmer_name', f'prior_{col_short}_trend']
    farmer_hist = farmer_hist.merge(trends, on='farmer_name', how='left')

# === V6 NEW: Farmer's prior topic set (for overlap features later) ===
farmer_prior_topics = prior_df.groupby('farmer_name')['topics_parsed'].apply(
    lambda x: set(t for topics in x for t in topics)).to_dict()

# === V6 NEW: Farmer's last group ===
farmer_hist = farmer_hist.merge(
    last_sessions[['farmer_name', 'group_name']].rename(columns={'group_name': 'prior_last_group'}),
    on='farmer_name', how='left')

# === V6 NEW: Farmer coop change detection ===
first_coop = first_sessions[['farmer_name', 'belong_to_cooperative']].rename(
    columns={'belong_to_cooperative': 'prior_first_coop'})
farmer_hist = farmer_hist.merge(first_coop, on='farmer_name', how='left')
farmer_hist['prior_coop_changed'] = (farmer_hist['prior_first_coop'] != farmer_hist['prior_last_coop']).astype(int)

print(f"  {len(farmer_hist)} farmers, {farmer_hist.shape[1]-1} features")
elapsed = time.time() - start_time
print(f"  [{elapsed:.0f}s]")

# ============================================================
# 4. GROUP HISTORY (comprehensive)
# ============================================================
print("\nSTEP 4: Group history...")

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
    group_hist['prior_grp_07_rate'] * 3 + group_hist['prior_grp_90_rate'] * 2 + group_hist['prior_grp_120_rate'] * 1) / 6
group_hist['prior_grp_sessions_per_farmer'] = group_hist['prior_grp_size'] / group_hist['prior_grp_unique_farmers'].clip(lower=1)
group_hist['prior_grp_any_adoption'] = ((group_hist['prior_grp_07_rate'] + group_hist['prior_grp_90_rate'] + group_hist['prior_grp_120_rate']) > 0).astype(int)
group_hist['prior_grp_size_bucket'] = pd.cut(group_hist['prior_grp_size'], bins=[0, 5, 15, 50, 200, np.inf], labels=[0, 1, 2, 3, 4]).astype(float)

# Adopter density
grp_adopters = prior_df.groupby('group_name').apply(
    lambda x: x.groupby('farmer_name')['adopted_within_120_days'].max().mean()).reset_index()
grp_adopters.columns = ['group_name', 'prior_grp_adopter_density']
group_hist = group_hist.merge(grp_adopters, on='group_name', how='left')

# === V6 NEW: Group temporal features ===
grp_dates = prior_df.groupby('group_name')['training_day_dt'].agg(
    prior_grp_first_date='min', prior_grp_last_date='max').reset_index()
grp_dates['prior_grp_active_span_days'] = (grp_dates['prior_grp_last_date'] - grp_dates['prior_grp_first_date']).dt.days
group_hist = group_hist.merge(grp_dates[['group_name', 'prior_grp_active_span_days']], on='group_name', how='left')

# === V6 NEW: Group topic diversity ===
grp_topics = prior_df.groupby('group_name')['topics_parsed'].apply(
    lambda x: len(set(t for topics in x for t in topics))).reset_index()
grp_topics.columns = ['group_name', 'prior_grp_topic_diversity']
group_hist = group_hist.merge(grp_topics, on='group_name', how='left')

# === V6 NEW: Group county diversity (multi-location groups) ===
grp_counties = prior_df.groupby('group_name')['county'].nunique().reset_index()
grp_counties.columns = ['group_name', 'prior_grp_county_diversity']
group_hist = group_hist.merge(grp_counties, on='group_name', how='left')

print(f"  {len(group_hist)} groups, {group_hist.shape[1]-1} features")

# ============================================================
# 5. GEO + TRAINER + TOPIC FEATURES
# ============================================================
print("\nSTEP 5: Geo/Trainer/Topic features...")

# Geo
for geo_col in ['ward', 'subcounty', 'county']:
    geo_hist = prior_df.groupby(geo_col).agg(
        **{f'prior_{geo_col}_size': ('ID', 'count'),
           f'prior_{geo_col}_07_rate': ('adopted_within_07_days', 'mean'),
           f'prior_{geo_col}_90_rate': ('adopted_within_90_days', 'mean'),
           f'prior_{geo_col}_120_rate': ('adopted_within_120_days', 'mean'),
           f'prior_{geo_col}_coop_rate': ('belong_to_cooperative', 'mean'),
           f'prior_{geo_col}_has_topic_rate': ('has_topic_trained_on', 'mean'),
           }).reset_index()
    train_df = train_df.merge(geo_hist, on=geo_col, how='left')
    test_df = test_df.merge(geo_hist, on=geo_col, how='left')

# === V6 NEW: Geo trainer/group density ===
for geo_col in ['ward', 'subcounty', 'county']:
    geo_density = prior_df.groupby(geo_col).agg(
        **{f'prior_{geo_col}_trainer_density': ('trainer_parsed', 'nunique'),
           f'prior_{geo_col}_group_density': ('group_name', 'nunique'),
           f'prior_{geo_col}_farmer_density': ('farmer_name', 'nunique'),
           }).reset_index()
    train_df = train_df.merge(geo_density, on=geo_col, how='left')
    test_df = test_df.merge(geo_density, on=geo_col, how='left')

# === V6 NEW: County-month seasonality from Prior ===
prior_df['prior_month'] = prior_df['training_day_dt'].dt.month
county_month_rates = prior_df.groupby(['county', 'prior_month']).agg(
    cm_n=('ID', 'count'),
    cm_07_rate=('adopted_within_07_days', 'mean'),
    cm_90_rate=('adopted_within_90_days', 'mean'),
    cm_120_rate=('adopted_within_120_days', 'mean'),
).reset_index()
CM_SMOOTH = 15
for col in ['cm_07_rate', 'cm_90_rate', 'cm_120_rate']:
    tgt = col.replace('cm_', 'adopted_within_').replace('_rate', '_days')
    g = prior_df[tgt].mean()
    county_month_rates[f'{col}_smoothed'] = (
        county_month_rates[col] * county_month_rates['cm_n'] + g * CM_SMOOTH
    ) / (county_month_rates['cm_n'] + CM_SMOOTH)

train_df['prior_month'] = train_df['training_day_dt'].dt.month
test_df['prior_month'] = test_df['training_day_dt'].dt.month
train_df = train_df.merge(county_month_rates[['county', 'prior_month',
    'cm_07_rate_smoothed', 'cm_90_rate_smoothed', 'cm_120_rate_smoothed']],
    on=['county', 'prior_month'], how='left')
test_df = test_df.merge(county_month_rates[['county', 'prior_month',
    'cm_07_rate_smoothed', 'cm_90_rate_smoothed', 'cm_120_rate_smoothed']],
    on=['county', 'prior_month'], how='left')

# Trainer effectiveness
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
    trainer_eff['prior_trainer_07_rate'] * 3 + trainer_eff['prior_trainer_90_rate'] * 2 + trainer_eff['prior_trainer_120_rate'] * 1) / 6

TRAINER_SMOOTH = 50
for rate_col in ['prior_trainer_07_rate', 'prior_trainer_90_rate', 'prior_trainer_120_rate']:
    target_name = rate_col.replace('prior_trainer_', '').replace('_rate', '')
    global_rate = prior_df[f'adopted_within_{target_name}_days'].mean()
    trainer_eff[f'{rate_col}_smoothed'] = (
        trainer_eff[rate_col] * trainer_eff['prior_trainer_total'] + global_rate * TRAINER_SMOOTH
    ) / (trainer_eff['prior_trainer_total'] + TRAINER_SMOOTH)

trainer_days = prior_df.groupby('trainer_parsed')['training_day_dt'].nunique().reset_index()
trainer_days.columns = ['trainer_parsed', 'prior_trainer_active_days']
trainer_eff = trainer_eff.merge(trainer_days, on='trainer_parsed', how='left')
trainer_eff['prior_trainer_daily_load'] = trainer_eff['prior_trainer_total'] / trainer_eff['prior_trainer_active_days'].clip(lower=1)

trainer_county_div = prior_df.groupby('trainer_parsed')['county'].nunique().reset_index()
trainer_county_div.columns = ['trainer_parsed', 'prior_trainer_county_diversity']
trainer_eff = trainer_eff.merge(trainer_county_div, on='trainer_parsed', how='left')

train_df = train_df.merge(trainer_eff, on='trainer_parsed', how='left')
test_df = test_df.merge(trainer_eff, on='trainer_parsed', how='left')

# Topic rates from Prior
topic_rows = []
for _, row in prior_df.iterrows():
    for topic in row['topics_parsed']:
        topic_rows.append({
            'topic': topic,
            'adopted_within_07_days': row['adopted_within_07_days'],
            'adopted_within_90_days': row['adopted_within_90_days'],
            'adopted_within_120_days': row['adopted_within_120_days'],
        })

topic_rate_dict = {}
if topic_rows:
    topic_df = pd.DataFrame(topic_rows)
    TOPIC_SMOOTH = 30
    topic_rates = topic_df.groupby('topic').agg(
        topic_n=('adopted_within_07_days', 'count'),
        topic_07_rate=('adopted_within_07_days', 'mean'),
        topic_90_rate=('adopted_within_90_days', 'mean'),
        topic_120_rate=('adopted_within_120_days', 'mean'),
    ).reset_index()
    for col in ['topic_07_rate', 'topic_90_rate', 'topic_120_rate']:
        tgt = col.replace('topic_', 'adopted_within_').replace('_rate', '_days')
        g = prior_df[tgt].mean()
        topic_rates[f'{col}_smoothed'] = (
            topic_rates[col] * topic_rates['topic_n'] + g * TOPIC_SMOOTH
        ) / (topic_rates['topic_n'] + TOPIC_SMOOTH)
    topic_rate_dict = topic_rates.set_index('topic').to_dict('index')

# === V6 NEW: Topic IDF (rarity) ===
all_prior_topic_counts = {}
for topics in prior_df['topics_parsed']:
    for t in topics:
        all_prior_topic_counts[t] = all_prior_topic_counts.get(t, 0) + 1
total_prior_docs = len(prior_df)

# Trainer-county combos
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

# === V6 NEW: Trainer-topic specialization ===
trainer_topic_rows = []
for _, row in prior_df.iterrows():
    for topic in row['topics_parsed']:
        trainer_topic_rows.append({
            'trainer_parsed': row['trainer_parsed'],
            'topic': topic,
            'adopted_within_120_days': row['adopted_within_120_days'],
        })
if trainer_topic_rows:
    tt_df = pd.DataFrame(trainer_topic_rows)
    trainer_topic_spec = tt_df.groupby('trainer_parsed').agg(
        trainer_topic_count=('topic', 'nunique'),
        trainer_topic_total=('topic', 'count'),
    ).reset_index()
    trainer_topic_spec['prior_trainer_topic_specialization'] = 1.0 / trainer_topic_spec['trainer_topic_count'].clip(lower=1)
    train_df = train_df.merge(trainer_topic_spec[['trainer_parsed', 'prior_trainer_topic_specialization']], on='trainer_parsed', how='left')
    test_df = test_df.merge(trainer_topic_spec[['trainer_parsed', 'prior_trainer_topic_specialization']], on='trainer_parsed', how='left')

elapsed = time.time() - start_time
print(f"  [{elapsed:.0f}s]")

# ============================================================
# 6. MERGE HISTORY
# ============================================================
print("\nSTEP 6: Merging...")

train_df = train_df.merge(farmer_hist, on='farmer_name', how='left')
test_df = test_df.merge(farmer_hist, on='farmer_name', how='left')
train_df = train_df.merge(group_hist, on='group_name', how='left')
test_df = test_df.merge(group_hist, on='group_name', how='left')

# Fill NaN for history cols
hist_cols = [c for c in farmer_hist.columns if c not in ['farmer_name', 'prior_first_date', 'prior_last_date', 'prior_last_group']] + \
            [c for c in group_hist.columns if c != 'group_name'] + \
            [c for c in trainer_eff.columns if c != 'trainer_parsed']
for c in hist_cols:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

fill_zero_cols = ['tc_size', 'tc_07_rate_smoothed', 'tc_90_rate_smoothed', 'tc_120_rate_smoothed',
                  'cm_07_rate_smoothed', 'cm_90_rate_smoothed', 'cm_120_rate_smoothed',
                  'prior_trainer_topic_specialization']
for c in fill_zero_cols:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
    if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

for geo_col in ['ward', 'subcounty', 'county']:
    for suffix in ['_trainer_density', '_group_density', '_farmer_density']:
        c = f'prior_{geo_col}{suffix}'
        if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
        if c in test_df.columns: test_df[c] = test_df[c].fillna(0)

# ============================================================
# 7. FEATURE ENGINEERING (V5 + V6 NEW)
# ============================================================
print("\nSTEP 7: Feature engineering...")

train_df['is_train'] = 1
test_df['is_train'] = 0
for t in TARGETS:
    if t not in test_df.columns:
        test_df[t] = np.nan

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_idx = df['is_train'] == 1
test_idx = df['is_train'] == 0

# ----- Temporal -----
df['training_day_dt'] = pd.to_datetime(df['training_day'])
df['train_year'] = df['training_day_dt'].dt.year
df['train_month'] = df['training_day_dt'].dt.month
df['train_day'] = df['training_day_dt'].dt.day
df['train_dayofweek'] = df['training_day_dt'].dt.dayofweek
df['train_quarter'] = df['training_day_dt'].dt.quarter
df['train_weekofyear'] = df['training_day_dt'].dt.isocalendar().week.astype(int)
df['train_dayofyear'] = df['training_day_dt'].dt.dayofyear
df['is_weekend'] = (df['train_dayofweek'] >= 5).astype(int)
df['is_sunday'] = (df['train_dayofweek'] == 6).astype(int)
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
df['month_dow'] = df['train_month'] * 10 + df['train_dayofweek']
df['is_high_adoption_month'] = df['train_month'].isin([3, 9, 11]).astype(int)

prior_last_dates = prior_df.groupby('farmer_name')['training_day_dt'].max().to_dict()
df['days_since_last_prior'] = df.apply(
    lambda r: (r['training_day_dt'] - prior_last_dates.get(r['farmer_name'], r['training_day_dt'])).days
    if r['farmer_name'] in prior_last_dates else -1, axis=1)
df['has_prior_history'] = (df['prior_session_count'] > 0).astype(int)

prior_session_counts = prior_df.groupby('farmer_name').size().to_dict()
df['farmer_total_prior_sessions'] = df['farmer_name'].map(prior_session_counts).fillna(0)
df['training_sequence_num'] = df['farmer_total_prior_sessions'] + 1

# === V6 NEW: Days-since-last binning ===
df['recency_bin'] = pd.cut(df['days_since_last_prior'], bins=[-2, -1, 0, 30, 90, 180, 365, np.inf],
    labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(0)

# ----- Topic features -----
df['topic_count'] = df['topics_parsed'].apply(len)
df['is_multi_topic'] = (df['topic_count'] > 1).astype(int)
df['is_single_topic'] = (df['topic_count'] == 1).astype(int)

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

def get_topic_rate(topics, rate_key, default=0):
    rates = []
    for t in topics:
        if t in topic_rate_dict and rate_key in topic_rate_dict[t]:
            rates.append(topic_rate_dict[t][rate_key])
    return np.mean(rates) if rates else default

for suffix in ['07', '90', '120']:
    df[f'topic_adoption_rate_{suffix}'] = df['topics_parsed'].apply(
        lambda x: get_topic_rate(x, f'topic_{suffix}_rate_smoothed'))

# === V6 NEW: Topic IDF (rarity score) ===
def calc_topic_idf(topics):
    if not topics: return 0
    idfs = []
    for t in topics:
        count = all_prior_topic_counts.get(t, 1)
        idfs.append(np.log(total_prior_docs / count))
    return np.mean(idfs)

df['topic_idf_mean'] = df['topics_parsed'].apply(calc_topic_idf)
df['topic_idf_max'] = df['topics_parsed'].apply(
    lambda topics: max([np.log(total_prior_docs / all_prior_topic_counts.get(t, 1)) for t in topics]) if topics else 0)

# === V6 NEW: Topic overlap with prior ===
prior_topics_series = df['farmer_name'].map(lambda f: farmer_prior_topics.get(f, set()))
current_sets = df['topics_parsed'].apply(set)

df['topic_overlap_count'] = [len(c & p) if p else 0 for c, p in zip(current_sets, prior_topics_series)]
df['topic_new_count'] = [len(c - p) if p else len(c) for c, p in zip(current_sets, prior_topics_series)]
df['topic_overlap_ratio'] = [o / len(c) if c else 0 for o, c in zip(df['topic_overlap_count'], df['topics_parsed'])]
df['is_repeat_topic'] = (df['topic_overlap_ratio'] == 1.0).astype(int)
df['is_all_new_topic'] = (df['topic_overlap_ratio'] == 0.0).astype(int)

# ----- Geo interactions -----
df['county_subcounty'] = df['county'] + '_' + df['subcounty']
df['subcounty_ward'] = df['subcounty'] + '_' + df['ward']
df['county_ward'] = df['county'] + '_' + df['ward']
df['county_trainer'] = df['county'] + '_' + df['trainer_parsed']
df['ward_trainer'] = df['ward'] + '_' + df['trainer_parsed']
df['county_topic'] = df['county'] + '_' + df['primary_topic_cat']
df['ward_topic'] = df['ward'] + '_' + df['primary_topic_cat']
df['trainer_topic'] = df['trainer_parsed'] + '_' + df['primary_topic_cat']

# Frequency encoding
freq_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
             'primary_topic_cat', 'county_subcounty', 'county_topic',
             'ward_topic', 'trainer_topic', 'county_trainer']
for col in freq_cols:
    df[f'{col}_freq'] = df.groupby(col)[col].transform('count')

# Group features
df['group_size'] = df.groupby('group_name')['group_name'].transform('count')
df['group_coop_rate'] = df.groupby('group_name')['belong_to_cooperative'].transform('mean')
df['group_female_rate'] = df.groupby('group_name')['gender'].transform(lambda x: (x == 'Female').mean())
df['group_young_rate'] = df.groupby('group_name')['age'].transform(lambda x: (x == 'Below 35').mean())
df['group_ussd_rate'] = df.groupby('group_name')['registration'].transform(lambda x: (x == 'Ussd').mean())
df['group_topic_diversity'] = df.groupby('group_name')['primary_topic_cat'].transform('nunique')
df['group_trainer_diversity'] = df.groupby('group_name')['trainer_parsed'].transform('nunique')
df['group_has_topic_rate'] = df.groupby('group_name')['has_topic_trained_on'].transform('mean')
df['group_session_mean'] = df.groupby('group_name')['num_sessions'].transform('mean')
df['group_size_bucket'] = pd.cut(df['group_size'], bins=[0, 3, 10, 30, 100, np.inf], labels=[0, 1, 2, 3, 4]).astype(float)

# === V6 NEW: Group-farmer loyalty (how many times this farmer has been in this group in Prior) ===
farmer_group_counts = prior_df.groupby(['farmer_name', 'group_name']).size().reset_index(name='fg_prior_count')
df = df.merge(farmer_group_counts, on=['farmer_name', 'group_name'], how='left')
df['fg_prior_count'] = df['fg_prior_count'].fillna(0)
df['is_new_to_group'] = (df['fg_prior_count'] == 0).astype(int)
df['is_same_group_as_prior_last'] = (df['group_name'] == df.get('prior_last_group', '')).astype(int)

# Trainer features
df['trainer_total'] = df.groupby('trainer_parsed')['trainer_parsed'].transform('count')
df['trainer_group_diversity'] = df.groupby('trainer_parsed')['group_name'].transform('nunique')
df['trainer_county_diversity_curr'] = df.groupby('trainer_parsed')['county'].transform('nunique')
df['trainer_coop_rate'] = df.groupby('trainer_parsed')['belong_to_cooperative'].transform('mean')
df['trainer_female_rate'] = df.groupby('trainer_parsed')['gender'].transform(lambda x: (x == 'Female').mean())

# Demographic interactions
df['gender_age'] = df['gender'] + '_' + df['age']
df['gender_coop'] = df['gender'] + '_' + df['belong_to_cooperative'].astype(str)
df['age_coop'] = df['age'] + '_' + df['belong_to_cooperative'].astype(str)
df['registration_age'] = df['registration'] + '_' + df['age']
df['gender_registration'] = df['gender'] + '_' + df['registration']
df['gender_trainer'] = df['gender'] + '_' + df['trainer_parsed']
df['age_trainer'] = df['age'] + '_' + df['trainer_parsed']
df['gender_county'] = df['gender'] + '_' + df['county']
df['gender_topic'] = df['gender'] + '_' + df['primary_topic_cat']

# History interactions  
df['hist_sessions_x_topics'] = df['prior_session_count'] * df['topic_count']
df['hist_adoption_x_hastopic'] = df['prior_adoption_score'] * df['has_topic_trained_on']
df['hist_grp_adoption_x_farmer_adoption'] = df['prior_grp_adoption_score'] * df['prior_adoption_score']
df['hist_ever_adopted_x_hastopic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['farmer_is_repeat'] = (df['prior_session_count'] > 0).astype(int)
df['farmer_high_engagement'] = (df['prior_session_count'] >= 5).astype(int)
df['farmer_adopted_before'] = df['prior_any_adoption']

# Advanced interactions
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
df['prior_adopter_x_has_topic'] = df['prior_any_adoption'] * df['has_topic_trained_on']
df['group_proof_x_farmer_hist'] = df['prior_grp_adopter_density'].fillna(0) * df['prior_any_adoption']
df['trainer_x_group_quality'] = df['prior_trainer_effectiveness'] * df['prior_grp_adoption_score']
df['sunday_x_has_topic'] = df['is_sunday'] * df['has_topic_trained_on']
df['recency_x_triple'] = df['recency_weighted_adoption'] * df['triple_signal']
df['sequence_x_adoption'] = df['training_sequence_num'] * df['prior_adoption_score']
df['topic_rate_x_trainer'] = df.get('topic_adoption_rate_120', pd.Series(0, index=df.index)) * df['prior_trainer_effectiveness']
for geo in ['county', 'ward']:
    if f'prior_{geo}_120_rate' in df.columns:
        df[f'{geo}_rate_x_farmer_adopt'] = df[f'prior_{geo}_120_rate'] * df['prior_adoption_score']

# === V6 NEW INTERACTIONS ===
df['topic_overlap_x_adoption'] = df['topic_overlap_ratio'] * df['prior_adoption_score']
df['loyalty_x_adoption'] = df['fg_prior_count'] * df['prior_adoption_score']
df['new_topic_x_engagement'] = df['topic_new_count'] * df['farmer_high_engagement']
df['idf_x_has_topic'] = df['topic_idf_mean'] * df['has_topic_trained_on']
df['num_trainers_x_sessions'] = df['num_trainers'] * df['num_sessions']
df['coop_changed_x_adoption'] = df['prior_coop_changed'] * df['prior_adoption_score']
df['cm_120_x_farmer_adopt'] = df['cm_120_rate_smoothed'] * df['prior_adoption_score']
df['trainer_spec_x_effectiveness'] = df['prior_trainer_topic_specialization'] * df['prior_trainer_effectiveness']
df['grp_topic_div_x_adoption'] = df['prior_grp_topic_diversity'].fillna(0) * df['prior_grp_adoption_score']
df['recency_bin_x_adoption'] = df['recency_bin'] * df['prior_adoption_score']

# Aggregations
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

# ----- Target encoding (OOF) -----
print("  Target encoding (OOF)...")
SMOOTHING = 10
te_cols = ['county', 'subcounty', 'ward', 'trainer_parsed', 'group_name',
           'primary_topic_cat', 'gender', 'age', 'registration',
           'county_subcounty', 'county_topic', 'ward_topic',
           'trainer_topic', 'gender_age', 'county_trainer']

train_data = df[train_idx].copy()
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Different seed from V5!

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

# Prior Target Encoding 
PRIOR_SMOOTH = 20
for target in TARGETS:
    prior_global = prior_df[target].mean()
    for col in ['group_name', 'ward', 'subcounty', 'county']:
        prior_stats = prior_df.groupby(col)[target].agg(['sum', 'count'])
        prior_smoothed = (prior_stats['sum'] + PRIOR_SMOOTH * prior_global) / (prior_stats['count'] + PRIOR_SMOOTH)
        prior_te_col = f'prior_te_{col}_{target}'
        df[prior_te_col] = df[col].map(prior_smoothed).fillna(prior_global)

elapsed = time.time() - start_time
print(f"  Feature engineering: [{elapsed:.0f}s]")

# ============================================================
# 8. PREPARE FEATURES
# ============================================================
print("\nSTEP 8: Preparing feature matrix...")

exclude_cols = ['ID', 'farmer_name', 'is_train', 'training_day', 'training_day_dt',
                'topics_list', 'topics_parsed', 'topic_cats', 'trainer', 'trainers_all',
                'prior_last_session_date', 'prior_first_session_date',
                'prior_first_date', 'prior_last_date', 'prior_last_group',
                'prior_month'] + TARGETS

cat_cols_to_encode = [col for col in df.select_dtypes(include='object').columns
                      if col not in exclude_cols]
label_encoders = {}
for col in cat_cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

feature_cols = [col for col in df.columns if col not in exclude_cols 
                and df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'uint8']]

# Leakage check
leakage_terms = ['adopted', 'target', 'label', 'y_true']
leaking_cols = [c for c in feature_cols if any(term in c.lower() for term in leakage_terms)
                and not c.startswith('prior_') and not c.startswith('te_') 
                and not c.startswith('topic_adoption') and not c.startswith('cm_')]
if leaking_cols:
    print(f"  WARNING: Removing {len(leaking_cols)} leaking cols: {leaking_cols}")
    feature_cols = [c for c in feature_cols if c not in leaking_cols]

X_train = df.loc[train_idx, feature_cols].reset_index(drop=True)
X_test = df.loc[test_idx, feature_cols].reset_index(drop=True)
test_ids_ordered = df.loc[test_idx, 'ID'].values

y_train = {}
for t in TARGETS:
    y_train[t] = df.loc[train_idx, t].reset_index(drop=True).astype(int)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

zero_topic_mask = df.loc[test_idx, 'has_topic_trained_on'].values == 0

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  Features: {len(feature_cols)}")
elapsed = time.time() - start_time
print(f"  [{elapsed:.0f}s]")

# ============================================================
# 9. OPTUNA (load V5 cache + refine)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: LightGBM Optuna tuning...")
print("=" * 70)

N_FOLDS = 5
BASE_SEED = 42  # Different from V5 (2024) for diversity!

def competition_score(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    return 0.75 * (1 - ll) + 0.25 * auc

V6_CACHE = os.path.join(DATA_DIR, 'v6_optuna_cache.json')
best_lgb_params = {}

if os.path.exists(V6_CACHE):
    print("  Found V6 cache! Loading...")
    with open(V6_CACHE, 'r') as f:
        best_lgb_params = json.load(f)
    for t in TARGETS:
        p = best_lgb_params[t]
        print(f"    {t}: leaves={p.get('num_leaves')}, lr={p.get('learning_rate', 0):.4f}")
else:
    # Load V5 cache as starting point
    V5_CACHE = os.path.join(DATA_DIR, 'optuna_cache_v5.json')
    v5_params = {}
    if os.path.exists(V5_CACHE):
        with open(V5_CACHE, 'r') as f:
            v5_params = json.load(f)
        print("  Loaded V5 cache as starting point")
    
    def optuna_lgb_objective(trial, X, y):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', -1, 12),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 100.0),
            'n_estimators': 3000,
            'seed': BASE_SEED,
        }
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(competition_score(y_val, preds))
        return np.mean(scores)
    
    N_TRIALS = 30
    for target in TARGETS:
        print(f"\n  Tuning {target} ({N_TRIALS} trials)...")
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        
        # Enqueue V5 params as warm start
        if target in v5_params:
            v5p = v5_params[target]
            enqueue = {}
            for k in ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction',
                       'bagging_freq', 'min_child_samples', 'lambda_l1', 'lambda_l2',
                       'min_gain_to_split', 'max_depth', 'scale_pos_weight']:
                if k in v5p:
                    enqueue[k] = v5p[k]
            if enqueue:
                study.enqueue_trial(enqueue)
        
        # Also enqueue a known good default
        study.enqueue_trial({
            'num_leaves': 63, 'learning_rate': 0.03,
            'feature_fraction': 0.75, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'min_child_samples': 20,
            'lambda_l1': 0.1, 'lambda_l2': 0.1,
            'min_gain_to_split': 0.01, 'max_depth': -1,
            'scale_pos_weight': 50.0,
        })
        
        study.optimize(
            lambda trial: optuna_lgb_objective(trial, X_train, y_train[target]),
            n_trials=N_TRIALS, show_progress_bar=False)
        
        best_lgb_params[target] = study.best_params.copy()
        print(f"    Best: {study.best_value:.6f}")
        print(f"    leaves={study.best_params['num_leaves']}, lr={study.best_params['learning_rate']:.4f}")
    
    with open(V6_CACHE, 'w') as f:
        json.dump(best_lgb_params, f, indent=2)
    print(f"\n  Cached → {V6_CACHE}")

elapsed = time.time() - start_time
print(f"  Optuna: [{elapsed:.0f}s]")

# ============================================================
# 10. LGB MULTI-SEED (1 seed)
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: LightGBM (1 seed, 5-fold)...")
print("=" * 70)

LGB_SEEDS = [42]

lgb_oof_preds = {t: np.zeros(len(X_train)) for t in TARGETS}
lgb_test_preds = {t: np.zeros(len(X_test)) for t in TARGETS}
feature_importance = {t: np.zeros(len(feature_cols)) for t in TARGETS}

for target in TARGETS:
    print(f"\n  Target: {target}")
    oof_accumulated = np.zeros(len(X_train))
    test_accumulated = np.zeros(len(X_test))
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        **best_lgb_params[target],
    }
    
    for seed_idx, seed in enumerate(LGB_SEEDS):
        params['seed'] = seed
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(X_train))
        test_seed = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train[target])):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[target].iloc[tr_idx], y_train[target].iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            
            oof_seed[val_idx] = model.predict_proba(X_val)[:, 1]
            test_seed += model.predict_proba(X_test)[:, 1] / N_FOLDS
            feature_importance[target] += model.feature_importances_
        
        seed_score = competition_score(y_train[target], oof_seed)
        if seed_idx % 5 == 0:
            print(f"    Seed {seed}: comp={seed_score:.6f}")
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

# Feature importance
for target in TARGETS:
    fi = pd.DataFrame({'feature': feature_cols, 'importance': feature_importance[target]})
    fi = fi.sort_values('importance', ascending=False)
    print(f"\n  Top 15 features ({target}):")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:.0f}")

elapsed = time.time() - start_time
print(f"  Training: [{elapsed:.0f}s]")

# ============================================================
# 11. CALIBRATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Calibration...")
print("=" * 70)

calibrated_test = {}
for target in TARGETS:
    raw_oof = lgb_oof_preds[target]
    raw_test = lgb_test_preds[target]
    raw_score = competition_score(y_train[target], raw_oof)
    
    # Platt scaling
    oof_logodds = np.log(np.clip(raw_oof, 1e-7, 1-1e-7) / (1 - np.clip(raw_oof, 1e-7, 1-1e-7)))
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(oof_logodds.reshape(-1, 1), y_train[target])
    platt_oof = platt.predict_proba(oof_logodds.reshape(-1, 1))[:, 1]
    platt_score = competition_score(y_train[target], platt_oof)
    
    test_logodds = np.log(np.clip(raw_test, 1e-7, 1-1e-7) / (1 - np.clip(raw_test, 1e-7, 1-1e-7)))
    platt_test = platt.predict_proba(test_logodds.reshape(-1, 1))[:, 1]
    
    # Isotonic (OOF)
    skf_iso = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iso_oof = np.zeros(len(raw_oof))
    iso_test_folds = np.zeros(len(raw_test))
    for fold, (tr_idx, val_idx) in enumerate(skf_iso.split(np.arange(len(raw_oof)), y_train[target])):
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso.fit(raw_oof[tr_idx], y_train[target].iloc[tr_idx])
        iso_oof[val_idx] = iso.predict(raw_oof[val_idx])
        iso_test_folds += iso.predict(raw_test) / 5
    iso_score = competition_score(y_train[target], iso_oof)
    
    # Temperature scaling  
    from scipy.optimize import minimize_scalar
    def temp_loss(T):
        scaled = 1 / (1 + np.exp(-oof_logodds / T))
        return log_loss(y_train[target], scaled)
    res = minimize_scalar(temp_loss, bounds=(0.5, 5.0), method='bounded')
    T_opt = res.x
    temp_oof = 1 / (1 + np.exp(-oof_logodds / T_opt))
    temp_score = competition_score(y_train[target], temp_oof)
    temp_test = 1 / (1 + np.exp(-test_logodds / T_opt))
    
    best_method = 'raw'
    best_score = raw_score
    best_test = raw_test
    if platt_score > best_score:
        best_method, best_score, best_test = 'platt', platt_score, platt_test
    if iso_score > best_score:
        best_method, best_score, best_test = 'isotonic', iso_score, iso_test_folds
    if temp_score > best_score:
        best_method, best_score, best_test = 'temperature', temp_score, temp_test
    
    calibrated_test[target] = best_test
    print(f"  {target}: raw={raw_score:.6f}, platt={platt_score:.6f}, iso={iso_score:.6f}, temp={temp_score:.6f} → {best_method}")

# ============================================================
# 12. POST-PROCESSING + DUAL SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Post-processing & submission...")
print("=" * 70)

preds = {}
for target in TARGETS:
    preds[target] = np.clip(calibrated_test[target], 0.001, 0.999)

# Monotonicity
preds['adopted_within_90_days'] = np.maximum(preds['adopted_within_07_days'], preds['adopted_within_90_days'])
preds['adopted_within_120_days'] = np.maximum(preds['adopted_within_90_days'], preds['adopted_within_120_days'])

# Zero-topic rule (pefect zero-adoption)
for target in TARGETS:
    preds[target][zero_topic_mask] = 0.001

# Zero-group rules (from train)
train_data_rules = df[train_idx].copy()
for target in TARGETS:
    group_stats = train_data_rules.groupby('group_name').agg(n=(target, 'count'), rate=(target, 'mean'))
    zero_groups = group_stats[(group_stats['n'] >= 30) & (group_stats['rate'] == 0)].index
    if len(zero_groups) > 0:
        test_vals = df.loc[test_idx, 'group_name'].values
        zmask = np.isin(test_vals, zero_groups)
        if zmask.sum() > 0:
            preds[target][zmask] = np.minimum(preds[target][zmask], 0.005)

# Prior zero-group rule
for target in TARGETS:
    prior_grp_stats = prior_df.groupby('group_name').agg(n=(target, 'count'), rate=(target, 'mean'))
    prior_zero_grps = prior_grp_stats[(prior_grp_stats['n'] >= 20) & (prior_grp_stats['rate'] == 0)].index
    train_groups = set(train_data_rules['group_name'].unique())
    prior_only_zero = [g for g in prior_zero_grps if g not in train_groups]
    if prior_only_zero:
        test_vals = df.loc[test_idx, 'group_name'].values
        zmask = np.isin(test_vals, prior_only_zero)
        if zmask.sum() > 0:
            preds[target][zmask] = np.minimum(preds[target][zmask], 0.008)

print(f"  Mean preds: " + ", ".join(f"{t.split('_')[2]}d={preds[t].mean():.5f}" for t in TARGETS))

# === DUAL SUBMISSION ===
sub = pd.DataFrame({'ID': test_ids_ordered})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    raw = preds[target].copy()
    ranks = pd.Series(raw).rank(pct=True)
    auc_preds = np.clip(ranks * 0.998 + 0.001, 0.001, 0.999)
    auc_preds[zero_topic_mask] = 0.001
    sub[auc_col] = auc_preds
    sub[ll_col] = raw

sub = sub[SS_COLS]
sub = sub.set_index('ID').loc[ss['ID']].reset_index()
assert len(sub) == len(ss) and sub.isnull().sum().sum() == 0

sub.to_csv('sub_V6_dual.csv', index=False)
print(f"\n  SAVED: sub_V6_dual.csv")

# === STANDARD SUBMISSION ===
sub_std = pd.DataFrame({'ID': test_ids_ordered})
for target, (auc_col, ll_col) in TARGET_TO_SS.items():
    sub_std[auc_col] = preds[target]
    sub_std[ll_col] = preds[target]
sub_std = sub_std[SS_COLS]
sub_std = sub_std.set_index('ID').loc[ss['ID']].reset_index()
sub_std.to_csv('sub_V6_standard.csv', index=False)
print(f"  SAVED: sub_V6_standard.csv")

# ============================================================
# 13. GENERATE RKLO BLENDS WITH V6 AS NEW SOURCE
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: RKLO blending with V6 + teammate...")
print("=" * 70)

from scipy.stats import rankdata
from scipy.special import logit, expit

v4e3 = pd.read_csv('submission_v4_ensemble (3).csv').set_index('ID').reindex(sub['ID']).reset_index()
v6 = sub.copy()  # V6 DUAL output

auc_cols = ['Target_07_AUC','Target_90_AUC','Target_120_AUC']
ll_cols = ['Target_07_LogLoss','Target_90_LogLoss','Target_120_LogLoss']
N = len(v6)

def rklo_ll(out, best, v4e3, N, rank_w=0.80, dist_v4=0.60):
    for col in ll_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = (1-rank_w) * r_best + rank_w * r_v4
        sorted_best = np.sort(best[col].values)
        sorted_v4 = np.sort(v4e3[col].values)
        b_logit = logit(np.clip(sorted_best, 1e-6, 1-1e-6))
        v_logit = logit(np.clip(sorted_v4, 1e-6, 1-1e-6))
        sorted_vals = expit((1-dist_v4) * b_logit + dist_v4 * v_logit)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    return out

def blend_auc(out, best, v4e3, N, auc_w):
    for col in auc_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = (1-auc_w) * r_best + auc_w * r_v4
        sorted_vals = np.sort(best[col].values)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    return out

def enforce_mono(out):
    out['Target_90_LogLoss'] = np.maximum(out['Target_90_LogLoss'], out['Target_07_LogLoss'])
    out['Target_120_LogLoss'] = np.maximum(out['Target_120_LogLoss'], out['Target_90_LogLoss'])
    return out

# Generate blends at the best AUC percentages we've proven
for auc_pct in [35, 40, 45, 50, 55, 60, 65, 70]:
    out = v6.copy()
    out = rklo_ll(out, v6, v4e3, N, rank_w=0.80, dist_v4=0.60)
    out = blend_auc(out, v6, v4e3, N, auc_w=auc_pct/100)
    out = enforce_mono(out)
    fname = f'sub_V6_rklo80d60_auc{auc_pct}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'  Saved: {fname}')

# Also blend V6 with sub_ADV (our previous best) for diversity
adv = pd.read_csv('sub_ADV_pertarget_progressive.csv').set_index('ID').reindex(v6['ID']).reset_index()

for v6_w in [30, 40, 50, 60, 70]:
    out = adv.copy()
    # LL: rank-avg V6 into ADV
    for col in ll_cols:
        r_adv = rankdata(adv[col]) / N
        r_v6 = rankdata(v6[col]) / N
        blended_ranks = (1 - v6_w/100) * r_adv + (v6_w/100) * r_v6
        sorted_adv = np.sort(adv[col].values)
        sorted_v6 = np.sort(v6[col].values)
        b_logit = logit(np.clip(sorted_adv, 1e-6, 1-1e-6))
        v_logit = logit(np.clip(sorted_v6, 1e-6, 1-1e-6))
        sorted_vals = expit(0.5 * b_logit + 0.5 * v_logit)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    # AUC: keep ADV's
    out = enforce_mono(out)
    fname = f'sub_V6xADV_ll{v6_w}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'  Saved: {fname}')

# Triple blend: V6 + ADV + teammate  
# Replace base with V6+ADV blend, then RKLO with teammate
for v6_adv in [30, 50]:
    for auc_pct in [40, 55]:
        # First blend V6+ADV
        base = adv.copy()
        for col in ll_cols:
            r_adv = rankdata(adv[col]) / N
            r_v6 = rankdata(v6[col]) / N
            blended_ranks = (1 - v6_adv/100) * r_adv + (v6_adv/100) * r_v6
            sorted_adv = np.sort(adv[col].values)
            sorted_v6 = np.sort(v6[col].values)
            b_logit = logit(np.clip(sorted_adv, 1e-6, 1-1e-6))
            v_logit = logit(np.clip(sorted_v6, 1e-6, 1-1e-6))
            sorted_vals = expit(0.5 * b_logit + 0.5 * v_logit)
            rank_order = rankdata(blended_ranks, method='ordinal') - 1
            base[col] = sorted_vals[rank_order.astype(int)]
        for col in auc_cols:
            r_adv_a = rankdata(adv[col]) / N
            r_v6_a = rankdata(v6[col]) / N
            blended_ranks = (1 - v6_adv/100) * r_adv_a + (v6_adv/100) * r_v6_a
            sorted_vals = np.sort(adv[col].values)
            rank_order = rankdata(blended_ranks, method='ordinal') - 1
            base[col] = sorted_vals[rank_order.astype(int)]
        
        # Then RKLO with teammate
        out = base.copy()
        out = rklo_ll(out, base, v4e3, N, rank_w=0.80, dist_v4=0.60)
        out = blend_auc(out, base, v4e3, N, auc_w=auc_pct/100)
        out = enforce_mono(out)
        fname = f'sub_V6_triple_v6adv{v6_adv}_auc{auc_pct}.csv'
        out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
        print(f'  Saved: {fname}')

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"TOTAL TIME: {total_time/60:.1f} minutes")
print(f"{'='*70}")

print(f"\nSUBMIT ORDER:")
print(f"  1. sub_V6_rklo80d60_auc55.csv (V6+teammate, proven AUC%)")
print(f"  2. sub_V6_rklo80d60_auc65.csv (V6+teammate, higher AUC)")
print(f"  3. sub_V6_triple_v6adv30_auc55.csv (V6+ADV+teammate triple)")
print(f"  4. sub_V6_dual.csv (standalone V6)")
