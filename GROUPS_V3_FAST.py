"""
GROUPS V3 FAST - Optimized hierarchical groups (no slow recency CV)
===================================================================
V2 hierarchical scored 0.718. This improves with:
  1. More group keys (16 total)
  2. Per-target best hierarchy from CV
  3. Optimized smoothing per group
  4. No recency in CV (too slow, marginal benefit)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("GROUPS V3 FAST")
print("=" * 70)

train = pd.read_csv('Train.csv', parse_dates=['training_date'])
test = pd.read_csv('Test.csv', parse_dates=['training_date'])
ss = pd.read_csv('SampleSubmission.csv')

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']

# ---- Expanded group keys ----
GROUP_DEFS = {
    'trainer_ward':           ['trainer', 'ward'],
    'trainer_subcounty':      ['trainer', 'subcounty'],
    'trainer_county':         ['trainer', 'county'],
    'trainer_topic':          ['trainer', 'has_topic_trained_on'],
    'ward_topic':             ['ward', 'has_topic_trained_on'],
    'subcounty_topic':        ['subcounty', 'has_topic_trained_on'],
    'county_topic':           ['county', 'has_topic_trained_on'],
    'trainer_ward_topic':     ['trainer', 'ward', 'has_topic_trained_on'],
    'trainer_county_topic':   ['trainer', 'county', 'has_topic_trained_on'],
    'trainer':                ['trainer'],
    'county':                 ['county'],
    'subcounty':              ['subcounty'],
    'ward':                   ['ward'],
    'ward_coop':              ['ward', 'belong_to_cooperative'],
    'county_coop':            ['county', 'belong_to_cooperative'],
    'trainer_coop':           ['trainer', 'belong_to_cooperative'],
}

for name, cols in GROUP_DEFS.items():
    train[name] = train[cols].astype(str).agg('_'.join, axis=1)
    test[name] = test[cols].astype(str).agg('_'.join, axis=1)

# Coverage
print("\nTest coverage:")
for name in GROUP_DEFS:
    train_groups = set(train[name].unique())
    covered = test[name].isin(train_groups).sum()
    pct = 100 * covered / len(test)
    print(f"  {name:25s}: {covered:5d}/{len(test)} ({pct:.1f}%)")

# ---- Fast CV (no recency) ----
print("\n" + "=" * 70)
print("CV: Finding optimal smoothing per group key")
print("=" * 70)

SMOOTH_VALUES = [3, 5, 8, 10, 15, 20, 30, 50]
best_configs = {}

for target in TARGETS:
    print(f"\n--- {target} ---")
    y = train[target].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(train, y))
    
    results = []
    for group_name in GROUP_DEFS.keys():
        for smooth in SMOOTH_VALUES:
            oof_preds = np.full(len(train), np.nan)
            
            for tr_idx, val_idx in folds:
                tr_fold = train.iloc[tr_idx]
                val_fold = train.iloc[val_idx]
                global_mean = tr_fold[target].mean()
                
                g = tr_fold.groupby(group_name)[target].agg(['mean', 'count']).reset_index()
                g.columns = [group_name, 'gmean', 'gcount']
                g['smoothed'] = (g['gmean'] * g['gcount'] + global_mean * smooth) / (g['gcount'] + smooth)
                mapping = dict(zip(g[group_name], g['smoothed']))
                oof_preds[val_idx] = val_fold[group_name].map(mapping).fillna(global_mean).values
            
            oof_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
            ll = log_loss(y, oof_clipped)
            auc = roc_auc_score(y, oof_preds)
            score = 0.75 * ll + 0.25 * (1 - auc)
            results.append((group_name, smooth, ll, auc, score))
    
    results.sort(key=lambda x: x[4])
    print(f"  Top 10:")
    for i, (gn, sm, ll, auc, sc) in enumerate(results[:10]):
        print(f"    {i+1:2d}. {gn:25s}(s={sm:2d}): LL={ll:.4f} AUC={auc:.4f} Score={sc:.4f}")
    
    best_configs[target] = results

# ---- Helper ----
def smoothed_group_preds(train_df, test_df, group_name, target, smooth):
    gm = train_df[target].mean()
    g = train_df.groupby(group_name)[target].agg(['mean', 'count']).reset_index()
    g.columns = [group_name, 'gmean', 'gcount']
    g['smoothed'] = (g['gmean'] * g['gcount'] + gm * smooth) / (g['gcount'] + smooth)
    mapping = dict(zip(g[group_name], g['smoothed']))
    return test_df[group_name].map(mapping)

def build_hier_preds(hierarchy_list, target):
    """hierarchy_list: list of (group_name, smooth)"""
    result = pd.Series(np.nan, index=test.index)
    for gn, sm in hierarchy_list:
        vals = smoothed_group_preds(train, test, gn, target, sm)
        mask = result.isna() & vals.notna()
        result[mask] = vals[mask]
        filled = mask.sum()
        if filled > 0:
            print(f"    {gn}(s={sm}): +{filled} rows")
    n_nan = result.isna().sum()
    if n_nan > 0:
        result = result.fillna(train[target].mean())
        print(f"    global_mean: +{n_nan} rows")
    return result.values

# ---- Get per-group best smoothing ----
def get_best_smooth(target):
    best = {}
    for gn, sm, ll, auc, sc in best_configs[target]:
        if gn not in best:
            best[gn] = sm
    return best

# ======================================================================
# APPROACH A: Per-target CV-optimal hierarchy (all groups, ranked by CV)
# ======================================================================
print("\n" + "=" * 70)
print("APPROACH A: Per-target CV-optimal hierarchy")
print("=" * 70)

approachA = {}
for target in TARGETS:
    print(f"\n  {target}:")
    bs = get_best_smooth(target)
    # Order groups by their best CV score
    seen = set()
    ordered = []
    for gn, sm, ll, auc, sc in best_configs[target]:
        if gn not in seen:
            seen.add(gn)
            ordered.append((gn, bs[gn]))
    approachA[target] = build_hier_preds(ordered, target)

# ======================================================================
# APPROACH B: Expanded V2-style (what scored 0.718 + new groups)
# ======================================================================
print("\n" + "=" * 70)
print("APPROACH B: Expanded V2-style hierarchy")
print("=" * 70)

approachB = {}
for target in TARGETS:
    print(f"\n  {target}:")
    bs = get_best_smooth(target)
    hierarchy = [
        ('trainer_ward_topic', bs.get('trainer_ward_topic', 5)),
        ('trainer_ward', bs.get('trainer_ward', 5)),
        ('trainer_county_topic', bs.get('trainer_county_topic', 5)),
        ('trainer_subcounty', bs.get('trainer_subcounty', 10)),
        ('trainer_county', bs.get('trainer_county', 15)),
        ('trainer_topic', bs.get('trainer_topic', 10)),
        ('ward_topic', bs.get('ward_topic', 10)),
        ('subcounty_topic', bs.get('subcounty_topic', 15)),
        ('county_topic', bs.get('county_topic', 10)),
        ('trainer', bs.get('trainer', 20)),
        ('ward', bs.get('ward', 15)),
        ('county', bs.get('county', 30)),
        ('subcounty', bs.get('subcounty', 30)),
        ('ward_coop', bs.get('ward_coop', 15)),
        ('county_coop', bs.get('county_coop', 30)),
        ('trainer_coop', bs.get('trainer_coop', 20)),
    ]
    approachB[target] = build_hier_preds(hierarchy, target)

# ======================================================================
# APPROACH C: Minimal hierarchy (only top-3 CV groups + fallback)
# ======================================================================
print("\n" + "=" * 70)
print("APPROACH C: Minimal hierarchy (top-3 groups)")
print("=" * 70)

approachC = {}
for target in TARGETS:
    print(f"\n  {target}:")
    bs = get_best_smooth(target)
    seen = set()
    top3 = []
    for gn, sm, ll, auc, sc in best_configs[target]:
        if gn not in seen:
            seen.add(gn)
            top3.append((gn, bs[gn]))
        if len(top3) == 3:
            break
    # Add broad fallbacks
    for fb in ['trainer', 'county', 'ward']:
        if fb not in seen:
            top3.append((fb, bs.get(fb, 30)))
    approachC[target] = build_hier_preds(top3, target)

# ======================================================================
# APPROACH D: Blend of A + V2 hierarchical (0.718 winner)
# ======================================================================
print("\n" + "=" * 70)
print("APPROACH D: Blend A (new) + V2 hier (0.718)")
print("=" * 70)

v2 = pd.read_csv('sub_GROUPSv2_hierarchical.csv')
v2_vals = {
    TARGETS[0]: v2['Target_07_AUC'].values,
    TARGETS[1]: v2['Target_90_AUC'].values,
    TARGETS[2]: v2['Target_120_AUC'].values,
}

approachD = {}
for ratio in [0.3, 0.5, 0.7]:
    key = f'D_{int(ratio*100)}'
    approachD[key] = {}
    for target in TARGETS:
        approachD[key][target] = ratio * approachA[target] + (1 - ratio) * v2_vals[target]
    print(f"  {int(ratio*100)}% A + {int((1-ratio)*100)}% V2: "
          f"7d={approachD[key][TARGETS[0]].mean():.4f}")

# ======================================================================
# APPROACH E: Blend A + B + C (diversity)
# ======================================================================
print("\n" + "=" * 70)
print("APPROACH E: Blend A + B + C")
print("=" * 70)

approachE = {}
for target in TARGETS:
    approachE[target] = (approachA[target] + approachB[target] + approachC[target]) / 3
    print(f"  {target}: mean={approachE[target].mean():.4f}")

# ======================================================================
# SAVE ALL
# ======================================================================
print("\n" + "=" * 70)
print("SAVING SUBMISSIONS")
print("=" * 70)

def save_sub(p07, p90, p120, filename):
    sub = ss.copy()
    p07 = np.clip(np.array(p07, dtype=float), 0.005, 0.995)
    p90 = np.clip(np.array(p90, dtype=float), 0.005, 0.995)
    p120 = np.clip(np.array(p120, dtype=float), 0.005, 0.995)
    p90 = np.maximum(p90, p07)
    p120 = np.maximum(p120, p90)
    p07 = np.minimum(p07, p90)
    sub['Target_07_AUC'] = p07
    sub['Target_07_LogLoss'] = p07
    sub['Target_90_AUC'] = p90
    sub['Target_90_LogLoss'] = p90
    sub['Target_120_AUC'] = p120
    sub['Target_120_LogLoss'] = p120
    sub.to_csv(filename, index=False)
    u7 = len(np.unique(np.round(p07, 6)))
    u90 = len(np.unique(np.round(p90, 6)))
    u120 = len(np.unique(np.round(p120, 6)))
    print(f"  {filename}")
    print(f"    7d:  mean={p07.mean():.4f} range=[{p07.min():.4f},{p07.max():.4f}] uniq={u7}")
    print(f"   90d:  mean={p90.mean():.4f} range=[{p90.min():.4f},{p90.max():.4f}] uniq={u90}")
    print(f"  120d:  mean={p120.mean():.4f} range=[{p120.min():.4f},{p120.max():.4f}] uniq={u120}")

save_sub(approachA[TARGETS[0]], approachA[TARGETS[1]], approachA[TARGETS[2]], 'sub_V3_A_optimal.csv')
save_sub(approachB[TARGETS[0]], approachB[TARGETS[1]], approachB[TARGETS[2]], 'sub_V3_B_expanded.csv')
save_sub(approachC[TARGETS[0]], approachC[TARGETS[1]], approachC[TARGETS[2]], 'sub_V3_C_minimal.csv')

for ratio in [0.3, 0.5, 0.7]:
    key = f'D_{int(ratio*100)}'
    save_sub(approachD[key][TARGETS[0]], approachD[key][TARGETS[1]], approachD[key][TARGETS[2]],
             f'sub_V3_D_blend{int(ratio*100)}.csv')

save_sub(approachE[TARGETS[0]], approachE[TARGETS[1]], approachE[TARGETS[2]], 'sub_V3_E_triple.csv')

# Also blend B + V2 (50/50) as bonus
for target in TARGETS:
    approachD['BV2'] = approachD.get('BV2', {})
    approachD['BV2'][target] = 0.5 * approachB[target] + 0.5 * v2_vals[target]
save_sub(approachD['BV2'][TARGETS[0]], approachD['BV2'][TARGETS[1]], approachD['BV2'][TARGETS[2]],
         'sub_V3_BV2_blend50.csv')

print("\n" + "=" * 70)
print("V2 hier (0.718): 7d=%.4f, 90d=%.4f, 120d=%.4f" % (
    v2_vals[TARGETS[0]].mean(), v2_vals[TARGETS[1]].mean(), v2_vals[TARGETS[2]].mean()))
print("DONE! 8 files generated.")
print("=" * 70)
