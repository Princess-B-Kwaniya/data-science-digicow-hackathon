"""
V7 - SAFE IMPROVEMENT over V3 (0.722)
=====================================
LESSON FROM FARMER FAILURE (0.702):
  farmer_id s=5 changed 2175 rows TOO MUCH → LogLoss destroyed.
  
NEW STRATEGY: 
1. Tie-breaking: V3 has only 39 unique values. Add TINY meaningful 
   perturbations to break ties for AUC without hurting LogLoss
2. Ultra-high smoothing: farmer/group at s=100+ barely moves predictions
   but creates unique values for AUC ranking
3. Different base s values: maybe s=3 isn't LB-optimal
4. DUAL columns: risky but could be huge (75% LogLoss + 25% AUC)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
v3 = pd.read_csv('sub_V3_A_optimal.csv')
farmer_sub = pd.read_csv('sub_FINAL_farmer_s5.csv')
targets = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
target_cols = {
    'adopted_within_07_days': ('Target_07_AUC', 'Target_07_LogLoss'),
    'adopted_within_90_days': ('Target_90_AUC', 'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

train['training_date'] = pd.to_datetime(train['training_date'])
test['training_date'] = pd.to_datetime(test['training_date'])
train = train.sort_values('training_date').reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════
# PART 1: DIAGNOSE WHY FARMER FAILED
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: DIAGNOSING FARMER FAILURE")
print("=" * 70)

# Compare V3 (0.722) vs farmer (0.702) predictions
for target in targets:
    col = target_cols[target][0]
    v3_vals = v3[col].values
    f_vals = farmer_sub[col].values
    diff = f_vals - v3_vals
    changed = diff != 0
    
    short = '07d' if '07' in target else ('90d' if '90' in target else '120d')
    print(f"\n  {short}:")
    print(f"    Changed rows: {changed.sum()}")
    print(f"    Mean absolute change: {np.abs(diff[changed]).mean():.4f}")
    print(f"    Max absolute change: {np.abs(diff[changed]).max():.4f}")
    print(f"    V3 mean (changed rows): {v3_vals[changed].mean():.4f}")
    print(f"    Farmer mean (changed rows): {f_vals[changed].mean():.4f}")
    print(f"    V3 mean (unchanged rows): {v3_vals[~changed].mean():.4f}")

# Key insight: farmer changed predictions by how much?
print("\n  Distribution of changes:")
for target in targets:
    col = target_cols[target][0]
    diff = farmer_sub[col].values - v3[col].values
    changed = diff != 0
    short = '07d' if '07' in target else ('90d' if '90' in target else '120d')
    abs_diff = np.abs(diff[changed])
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.3]:
        pct = (abs_diff > threshold).mean()
        print(f"    {short}: >{threshold:.2f} change: {pct:.1%}")

# ══════════════════════════════════════════════════════════════════════
# PART 2: STRATEGY 1 - TIE-BREAKING V3
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: TIE-BREAKING STRATEGY")
print("=" * 70)

# V3 has ~39 unique values for 07d. Within each group, rows are tied.
# We add tiny perturbations using SECONDARY features to break ties.
# This improves AUC ranking with negligible LogLoss impact.

# Use multiple secondary signals to create a composite tie-breaker
def make_keys(df):
    df = df.copy()
    df['ward_topic'] = df['ward'].astype(str) + '|' + df['has_topic_trained_on'].astype(str)
    df['county_topic'] = df['county'].astype(str) + '|' + df['has_topic_trained_on'].astype(str)
    df['ward_group'] = df['ward'].astype(str) + '|' + df['group_name'].astype(str)
    df['trainer_ward'] = df['trainer'].astype(str) + '|' + df['ward'].astype(str)
    return df

train = make_keys(train)
test = make_keys(test)

def bayesian_pred(gr_mean, gr_count, global_rate, s):
    return (gr_count * gr_mean + s * global_rate) / (gr_count + s)

# For each target, compute secondary predictions from group_name, county_topic, etc.
# Then use them as MICRO tie-breakers within V3 prediction groups

def add_tiebreaker(v3_preds, train_df, test_df, target, group_key, epsilon=0.005):
    """Add tiny perturbation to V3 predictions using a secondary group key.
    epsilon controls maximum perturbation size."""
    global_rate = train_df[target].mean()
    gs = train_df.groupby(group_key)[target].agg(['mean', 'count']).reset_index()
    gs.columns = [group_key, 'gr_mean', 'gr_count']
    gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, 3)
    
    merged = test_df[[group_key]].merge(gs[[group_key, 'pred']], on=group_key, how='left')
    secondary = merged['pred'].fillna(global_rate).values
    
    # Normalize secondary to [-1, 1] relative to V3 prediction
    # Then scale by epsilon
    preds = v3_preds.copy()
    for i in range(len(preds)):
        # perturbation = epsilon * (secondary[i] - preds[i]) / max(abs range)
        delta = secondary[i] - preds[i]
        preds[i] = preds[i] + epsilon * np.tanh(delta * 5)  # bounded perturbation
    
    return np.clip(preds, 0.001, 0.999)

# Test different epsilon values
print("\nTie-breaking CV (adding secondary signal with epsilon perturbation):")
print("  Epsilon 0 = pure V3")

n = len(train)
folds = [(0.5, 0.7), (0.6, 0.8), (0.8, 1.0)]

def predict_v3(tr, va, target, s=3):
    global_rate = tr[target].mean()
    preds = np.full(len(va), np.nan)
    hierarchy = ['ward_topic', 'trainer_ward', 'county_topic', 'trainer', 'county', 'ward']
    for gk in hierarchy:
        if np.isnan(preds).sum() == 0:
            break
        gs = tr.groupby(gk)[target].agg(['mean', 'count']).reset_index()
        gs.columns = [gk, 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
        merged = va[[gk]].merge(gs[[gk, 'pred']], on=gk, how='left')
        gp = merged['pred'].values
        missing = np.isnan(preds)
        preds = np.where(missing & ~np.isnan(gp), gp, preds)
    preds = np.where(np.isnan(preds), global_rate, preds)
    return np.clip(preds, 0.001, 0.999)

def cv_approach(approach_fn, label):
    """3-fold temporal CV."""
    target_avgs = []
    for target in targets:
        fold_scores = []
        for fold_start, fold_end in folds:
            split_s = int(n * fold_start)
            split_e = int(n * fold_end)
            tr = train.iloc[:split_s]
            va = train.iloc[split_s:split_e]
            if len(tr) < 100 or len(va) < 100 or va[target].nunique() < 2:
                continue
            preds = approach_fn(tr, va, target)
            try:
                ll = log_loss(va[target], preds)
                auc = roc_auc_score(va[target], preds)
                score = 0.75 * ll + 0.25 * (1 - auc)
                fold_scores.append(score)
            except:
                pass
        target_avgs.append(np.mean(fold_scores) if fold_scores else 999)
    avg = np.mean(target_avgs)
    parts = [f"{a:.4f}" for a in target_avgs]
    print(f"  {label:45s} {parts[0]:>7s} {parts[1]:>7s} {parts[2]:>7s} {avg:7.4f}")
    return avg

print(f"\n  {'Config':45s} {'07d':>7s} {'90d':>7s} {'120d':>7s} {'avg':>7s}")
print("  " + "-" * 70)

# Baseline: V3 with s=3
cv_approach(lambda tr, va, t: predict_v3(tr, va, t, s=3), "V3 s=3 (baseline, LB=0.722)")

# Different s values
for s in [1, 2, 4, 5, 8, 10, 15]:
    cv_approach(lambda tr, va, t, _s=s: predict_v3(tr, va, t, s=_s), f"V3 s={s}")

# Tie-breaking with group_name
for eps in [0.001, 0.003, 0.005, 0.01, 0.02]:
    def tb_approach(tr, va, t, _eps=eps):
        base = predict_v3(tr, va, t, s=3)
        global_rate = tr[t].mean()
        gs = tr.groupby('group_name')[t].agg(['mean', 'count']).reset_index()
        gs.columns = ['group_name', 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, 10)
        merged = va[['group_name']].merge(gs[['group_name', 'pred']], on='group_name', how='left')
        secondary = merged['pred'].fillna(global_rate).values
        preds = base.copy()
        for i in range(len(preds)):
            delta = secondary[i] - preds[i]
            preds[i] = preds[i] + _eps * np.tanh(delta * 3)
        return np.clip(preds, 0.001, 0.999)
    cv_approach(tb_approach, f"V3 s=3 + group tiebreak eps={eps}")

# Tie-breaking with ward_group
for eps in [0.001, 0.005, 0.01]:
    def tb_wg(tr, va, t, _eps=eps):
        base = predict_v3(tr, va, t, s=3)
        global_rate = tr[t].mean()
        gs = tr.groupby('ward_group')[t].agg(['mean', 'count']).reset_index()
        gs.columns = ['ward_group', 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, 10)
        merged = va[['ward_group']].merge(gs[['ward_group', 'pred']], on='ward_group', how='left')
        secondary = merged['pred'].fillna(global_rate).values
        preds = base.copy()
        for i in range(len(preds)):
            delta = secondary[i] - preds[i]
            preds[i] = preds[i] + _eps * np.tanh(delta * 3)
        return np.clip(preds, 0.001, 0.999)
    cv_approach(tb_wg, f"V3 s=3 + ward_group tiebreak eps={eps}")

# ══════════════════════════════════════════════════════════════════════
# PART 3: ULTRA-HIGH SMOOTHING FARMER
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: ULTRA-HIGH SMOOTHING FARMER (BARELY MOVES PREDICTIONS)")
print("=" * 70)

print(f"\n  {'Config':45s} {'07d':>7s} {'90d':>7s} {'120d':>7s} {'avg':>7s}")
print("  " + "-" * 70)

for sf in [30, 50, 100, 200, 500]:
    def farmer_ultra(tr, va, t, _sf=sf):
        base = predict_v3(tr, va, t, s=3)
        fs = tr.groupby('farmer_id')[t].agg(['mean', 'count'])
        fs.columns = ['f_mean', 'f_count']
        preds = base.copy()
        fids = va['farmer_id'].values
        for i in range(len(va)):
            if fids[i] in fs.index:
                fm = fs.loc[fids[i], 'f_mean']
                fc = fs.loc[fids[i], 'f_count']
                preds[i] = (fc * fm + _sf * base[i]) / (fc + _sf)
        return np.clip(preds, 0.001, 0.999)
    cv_approach(farmer_ultra, f"V3 s=3 + farmer s={sf}")

# ══════════════════════════════════════════════════════════════════════
# PART 4: DUAL COLUMN ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: DUAL COLUMN STRATEGY")
print("=" * 70)

# Competition has SEPARATE columns: Target_XX_AUC and Target_XX_LogLoss
# WHY would they create separate columns if same value is expected?
# Hypothesis: AUC column evaluated for AUC, LogLoss column for LogLoss
# If true: put RANKING-optimized values in AUC, CALIBRATED in LogLoss

# For AUC: we want GOOD RANKING - more unique values, good separation
# For LogLoss: we want WELL-CALIBRATED probabilities

# Simulate DUAL scoring in CV
print("\n  Simulating DUAL column scoring:")
print(f"  {'Config':45s} {'07d':>7s} {'90d':>7s} {'120d':>7s} {'avg':>7s}")
print("  " + "-" * 70)

def cv_dual(auc_fn, ll_fn, label):
    """CV with separate AUC and LogLoss column functions."""
    target_avgs = []
    for target in targets:
        fold_scores = []
        for fold_start, fold_end in folds:
            split_s = int(n * fold_start)
            split_e = int(n * fold_end)
            tr = train.iloc[:split_s]
            va = train.iloc[split_s:split_e]
            if len(tr) < 100 or len(va) < 100 or va[target].nunique() < 2:
                continue
            
            preds_auc = auc_fn(tr, va, target)
            preds_ll = ll_fn(tr, va, target)
            
            try:
                auc = roc_auc_score(va[target], preds_auc)
                ll = log_loss(va[target], preds_ll)
                score = 0.75 * ll + 0.25 * (1 - auc)
                fold_scores.append(score)
            except:
                pass
        target_avgs.append(np.mean(fold_scores) if fold_scores else 999)
    avg = np.mean(target_avgs)
    parts = [f"{a:.4f}" for a in target_avgs]
    print(f"  {label:45s} {parts[0]:>7s} {parts[1]:>7s} {parts[2]:>7s} {avg:7.4f}")
    return avg

# Standard (same for both)
cv_dual(
    lambda tr, va, t: predict_v3(tr, va, t, s=3),
    lambda tr, va, t: predict_v3(tr, va, t, s=3),
    "STANDARD: V3 for both (baseline)"
)

# DUAL: V3 for LogLoss, fine-grained for AUC
def predict_fine(tr, va, target, s=3):
    """Fine-grained predictions with more unique values."""
    global_rate = tr[target].mean()
    preds = np.full(len(va), np.nan)
    # Use a finer hierarchy with group_name
    hierarchy = ['ward_group', 'group_name', 'ward_topic', 'county_topic', 'ward', 'county', 'trainer']
    for gk in hierarchy:
        if np.isnan(preds).sum() == 0:
            break
        gs = tr.groupby(gk)[target].agg(['mean', 'count']).reset_index()
        gs.columns = [gk, 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
        merged = va[[gk]].merge(gs[[gk, 'pred']], on=gk, how='left')
        gp = merged['pred'].values
        missing = np.isnan(preds)
        preds = np.where(missing & ~np.isnan(gp), gp, preds)
    preds = np.where(np.isnan(preds), global_rate, preds)
    return np.clip(preds, 0.001, 0.999)

def predict_farmer_fine(tr, va, target, s_base=3, s_farmer=10):
    """Fine-grained with farmer layer."""
    base = predict_fine(tr, va, target, s=s_base)
    fs = tr.groupby('farmer_id')[target].agg(['mean', 'count'])
    fs.columns = ['f_mean', 'f_count']
    preds = base.copy()
    fids = va['farmer_id'].values
    for i in range(len(va)):
        if fids[i] in fs.index:
            fm = fs.loc[fids[i], 'f_mean']
            fc = fs.loc[fids[i], 'f_count']
            preds[i] = (fc * fm + s_farmer * base[i]) / (fc + s_farmer)
    return np.clip(preds, 0.001, 0.999)

cv_dual(
    lambda tr, va, t: predict_fine(tr, va, t, s=3),
    lambda tr, va, t: predict_v3(tr, va, t, s=3),
    "DUAL: fine AUC + V3 LogLoss"
)

cv_dual(
    lambda tr, va, t: predict_farmer_fine(tr, va, t, s_base=3, s_farmer=10),
    lambda tr, va, t: predict_v3(tr, va, t, s=3),
    "DUAL: farmer_fine AUC + V3 LogLoss"
)

cv_dual(
    lambda tr, va, t: predict_farmer_fine(tr, va, t, s_base=3, s_farmer=50),
    lambda tr, va, t: predict_v3(tr, va, t, s=3),
    "DUAL: farmer_fine_s50 AUC + V3 LogLoss"
)

# What if we use county_topic (best calibrated) for LogLoss?
def predict_county_topic(tr, va, target, s=1):
    global_rate = tr[target].mean()
    gs = tr.groupby('county_topic')[target].agg(['mean', 'count']).reset_index()
    gs.columns = ['county_topic', 'gr_mean', 'gr_count']
    gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
    merged = va[['county_topic']].merge(gs[['county_topic', 'pred']], on='county_topic', how='left')
    return np.clip(merged['pred'].fillna(global_rate).values, 0.001, 0.999)

cv_dual(
    lambda tr, va, t: predict_farmer_fine(tr, va, t, s_base=3, s_farmer=10),
    lambda tr, va, t: predict_county_topic(tr, va, t, s=1),
    "DUAL: farmer_fine AUC + county_topic LL"
)

cv_dual(
    lambda tr, va, t: predict_fine(tr, va, t, s=3),
    lambda tr, va, t: predict_county_topic(tr, va, t, s=1),
    "DUAL: fine AUC + county_topic LL"
)

# ══════════════════════════════════════════════════════════════════════
# PART 5: GENERATE ALL PROMISING SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: GENERATING SUBMISSIONS")
print("=" * 70)

def save_sub(name, preds_dict):
    out = pd.DataFrame({'ID': test['ID']})
    for target in targets:
        auc_col, ll_col = target_cols[target]
        if isinstance(preds_dict[target], tuple):
            out[auc_col] = np.clip(preds_dict[target][0], 0.001, 0.999)
            out[ll_col] = np.clip(preds_dict[target][1], 0.001, 0.999)
        else:
            p = np.clip(preds_dict[target], 0.001, 0.999)
            out[auc_col] = p
            out[ll_col] = p
    fname = f'sub_{name}.csv'
    out.to_csv(fname, index=False)
    print(f"\n  {fname}:")
    for target in targets:
        auc_col = target_cols[target][0]
        ll_col = target_cols[target][1]
        va = out[auc_col]
        vl = out[ll_col]
        short = '07d' if '07' in target else ('90d' if '90' in target else '120d')
        same = (va == vl).all()
        print(f"    {short}: AUC(u={va.nunique():4d}, m={va.mean():.4f}) LL(u={vl.nunique():4d}, m={vl.mean():.4f}) same={same}")
    return fname

# A. V3 tie-broken with group_name (SAFE - tiny perturbation)
for eps in [0.002, 0.005, 0.01]:
    pA = {}
    for target in targets:
        v3_vals = v3[target_cols[target][0]].values.copy()
        global_rate = train[target].mean()
        gs = train.groupby('group_name')[target].agg(['mean', 'count']).reset_index()
        gs.columns = ['group_name', 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, 10)
        lookup = dict(zip(gs['group_name'], gs['pred']))
        
        for i in range(len(test)):
            gn = test['group_name'].iloc[i]
            if gn in lookup:
                delta = lookup[gn] - v3_vals[i]
                v3_vals[i] += eps * np.tanh(delta * 3)
        pA[target] = v3_vals
    save_sub(f'V7_A_tiebreak_group_e{int(eps*1000)}', pA)

# B. V3 tie-broken with ward_group
for eps in [0.002, 0.005]:
    pB = {}
    for target in targets:
        v3_vals = v3[target_cols[target][0]].values.copy()
        global_rate = train[target].mean()
        gs = train.groupby('ward_group')[target].agg(['mean', 'count']).reset_index()
        gs.columns = ['ward_group', 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, 10)
        lookup = dict(zip(gs['ward_group'], gs['pred']))
        
        for i in range(len(test)):
            wg = test['ward_group'].iloc[i]
            if wg in lookup:
                delta = lookup[wg] - v3_vals[i]
                v3_vals[i] += eps * np.tanh(delta * 3)
        pB[target] = v3_vals
    save_sub(f'V7_B_tiebreak_wgroup_e{int(eps*1000)}', pB)

# C. V3 + ultra-high smoothing farmer (barely changes anything)
for sf in [50, 100, 200]:
    pC = {}
    for target in targets:
        v3_vals = v3[target_cols[target][0]].values.copy()
        fs = train.groupby('farmer_id')[target].agg(['mean', 'count'])
        fs.columns = ['f_mean', 'f_count']
        fids = test['farmer_id'].values
        for i in range(len(test)):
            if fids[i] in fs.index:
                fm = fs.loc[fids[i], 'f_mean']
                fc = fs.loc[fids[i], 'f_count']
                v3_vals[i] = (fc * fm + sf * v3_vals[i]) / (fc + sf)
        pC[target] = v3_vals
    save_sub(f'V7_C_farmer_s{sf}', pC)

# D. DUAL: V3 for LogLoss, fine-grained for AUC
def predict_fine_test(target, s=3):
    global_rate = train[target].mean()
    preds = np.full(len(test), np.nan)
    hierarchy = ['ward_group', 'group_name', 'ward_topic', 'county_topic', 'ward', 'county', 'trainer']
    for gk in hierarchy:
        if np.isnan(preds).sum() == 0:
            break
        gs = train.groupby(gk)[target].agg(['mean', 'count']).reset_index()
        gs.columns = [gk, 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
        merged = test[[gk]].merge(gs[[gk, 'pred']], on=gk, how='left')
        gp = merged['pred'].values
        missing = np.isnan(preds)
        preds = np.where(missing & ~np.isnan(gp), gp, preds)
    preds = np.where(np.isnan(preds), global_rate, preds)
    return np.clip(preds, 0.001, 0.999)

def predict_farmer_fine_test(target, s_base=3, s_farmer=10):
    base = predict_fine_test(target, s=s_base)
    fs = train.groupby('farmer_id')[target].agg(['mean', 'count'])
    fs.columns = ['f_mean', 'f_count']
    preds = base.copy()
    fids = test['farmer_id'].values
    for i in range(len(test)):
        if fids[i] in fs.index:
            fm = fs.loc[fids[i], 'f_mean']
            fc = fs.loc[fids[i], 'f_count']
            preds[i] = (fc * fm + s_farmer * base[i]) / (fc + s_farmer)
    return np.clip(preds, 0.001, 0.999)

# D1. DUAL: fine AUC + V3 LogLoss
pD1 = {}
for target in targets:
    auc_pred = predict_fine_test(target, s=3)
    ll_pred = v3[target_cols[target][0]].values
    pD1[target] = (auc_pred, ll_pred)
save_sub('V7_D1_DUAL_fine_v3ll', pD1)

# D2. DUAL: farmer_fine AUC + V3 LogLoss
pD2 = {}
for target in targets:
    auc_pred = predict_farmer_fine_test(target, s_base=3, s_farmer=10)
    ll_pred = v3[target_cols[target][0]].values
    pD2[target] = (auc_pred, ll_pred)
save_sub('V7_D2_DUAL_farmerfine_v3ll', pD2)

# D3. DUAL: farmer_fine s=50 AUC + V3 LogLoss (conservative)
pD3 = {}
for target in targets:
    auc_pred = predict_farmer_fine_test(target, s_base=3, s_farmer=50)
    ll_pred = v3[target_cols[target][0]].values
    pD3[target] = (auc_pred, ll_pred)
save_sub('V7_D3_DUAL_farmerfine50_v3ll', pD3)

# E. Different base smoothing (maybe s=3 isn't LB-optimal)
def predict_v3_test(target, s=3):
    global_rate = train[target].mean()
    preds = np.full(len(test), np.nan)
    hierarchy = ['ward_topic', 'trainer_ward', 'county_topic', 'trainer', 'county', 'ward']
    for gk in hierarchy:
        if np.isnan(preds).sum() == 0:
            break
        gs = train.groupby(gk)[target].agg(['mean', 'count']).reset_index()
        gs.columns = [gk, 'gr_mean', 'gr_count']
        gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
        merged = test[[gk]].merge(gs[[gk, 'pred']], on=gk, how='left')
        gp = merged['pred'].values
        missing = np.isnan(preds)
        preds = np.where(missing & ~np.isnan(gp), gp, preds)
    preds = np.where(np.isnan(preds), global_rate, preds)
    return np.clip(preds, 0.001, 0.999)

for s in [2, 4, 5]:
    pE = {}
    for target in targets:
        pE[target] = predict_v3_test(target, s=s)
    save_sub(f'V7_E_v3_s{s}', pE)

# F. Blend V3 with county_topic (different architecture, might capture different signal)
def predict_ct_test(target, s=1):
    global_rate = train[target].mean()
    gs = train.groupby('county_topic')[target].agg(['mean', 'count']).reset_index()
    gs.columns = ['county_topic', 'gr_mean', 'gr_count']
    gs['pred'] = bayesian_pred(gs['gr_mean'], gs['gr_count'], global_rate, s)
    merged = test[['county_topic']].merge(gs[['county_topic', 'pred']], on='county_topic', how='left')
    return np.clip(merged['pred'].fillna(global_rate).values, 0.001, 0.999)

for w_ct in [0.1, 0.2, 0.3]:
    pF = {}
    for target in targets:
        ct = predict_ct_test(target, s=1)
        v3_vals = v3[target_cols[target][0]].values
        pF[target] = np.clip(w_ct * ct + (1 - w_ct) * v3_vals, 0.001, 0.999)
    save_sub(f'V7_F_v3_ct_blend_w{int(w_ct*100)}', pF)

# ══════════════════════════════════════════════════════════════════════
# PART 6: COMPARE ALL V7 SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 6: ALL V7 SUBMISSIONS COMPARISON")
print("=" * 70)

import glob
v7_files = sorted(glob.glob('sub_V7_*.csv'))

# For each, compute how different from V3
print(f"\n  {'File':45s} {'07u':>5s} {'90u':>5s} {'12u':>5s} {'DUAL':>5s} {'diff_rows':>9s} {'max_diff':>8s}")
print("  " + "-" * 82)
for f in v7_files:
    df = pd.read_csv(f)
    u07 = df['Target_07_AUC'].nunique()
    u90 = df['Target_90_AUC'].nunique()
    u120 = df['Target_120_AUC'].nunique()
    is_dual = not (df['Target_07_AUC'] == df['Target_07_LogLoss']).all()
    
    # Diff from V3
    diff = np.abs(df['Target_07_AUC'].values - v3['Target_07_AUC'].values)
    n_diff = (diff > 1e-10).sum()
    max_diff = diff.max()
    
    dual_str = "YES" if is_dual else "no"
    print(f"  {f:45s} {u07:5d} {u90:5d} {u120:5d} {dual_str:>5s} {n_diff:9d} {max_diff:8.5f}")

print(f"\n  Reference: sub_V3_A_optimal.csv               39    42    44    no         0  0.00000")

# ══════════════════════════════════════════════════════════════════════
# FINAL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL RECOMMENDATIONS (RANKED BY SAFETY)")
print("=" * 70)

print("""
SAFEST → RISKIEST:

1. sub_V7_A_tiebreak_group_e2.csv (SAFEST IMPROVEMENT)
   - V3 base + ε=0.002 group_name perturbation
   - Max change from V3: ~0.002 per prediction
   - LogLoss impact: NEGLIGIBLE
   - AUC impact: More unique values from group_name signal
   - Risk: VERY LOW

2. sub_V7_C_farmer_s200.csv (ULTRA-CONSERVATIVE FARMER)
   - V3 base + farmer_id with s=200 smoothing
   - Avg change: ~0.001 per modified row
   - Creates unique values for 36% of rows
   - Risk: LOW (s=200 keeps predictions very close to V3)

3. sub_V7_E_v3_s2.csv or sub_V7_E_v3_s4.csv
   - Same V3 hierarchy but different smoothing
   - Might or might not beat s=3 on LB
   - Risk: LOW-MEDIUM

4. sub_V7_F_v3_ct_blend_w10.csv
   - 90% V3 + 10% county_topic
   - Small diversification
   - Risk: LOW-MEDIUM

5. sub_V7_D1_DUAL_fine_v3ll.csv (HIGHEST POTENTIAL, HIGHEST RISK)
   - IF Zindi evaluates columns separately: BIG improvement
   - IF Zindi uses one column for both: NEUTRAL or CATASTROPHIC
   - Risk: HIGH
""")

print("DONE!")
