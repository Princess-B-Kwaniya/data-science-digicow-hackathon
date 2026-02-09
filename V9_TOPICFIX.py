"""
V9 - Topic Fix Improvement over V7 (0.723 -> 0.7247)
=====================================================
INSIGHT: has_topic_trained_on=0 means guaranteed NON-ADOPTION.
  - 838 train rows with topic=0 have 0% adoption across ALL 3 targets
  - 99 test rows have topic=0
  - V7 predicted these rows at ~3-10% (from group-level rates)
  - Fixing to 0.001 reduces LogLoss AND helps AUC ranking

This is a SAFE post-processing fix on top of V7_D1 (proven 0.723).
Only 99 of 6000 rows changed. No model retraining.
"""

import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
v7 = pd.read_csv('sub_V7_D1_DUAL_fine_v3ll.csv')

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
TARGET_COLS = {
    'adopted_within_07_days': ('Target_07_AUC', 'Target_07_LogLoss'),
    'adopted_within_90_days': ('Target_90_AUC', 'Target_90_LogLoss'),
    'adopted_within_120_days': ('Target_120_AUC', 'Target_120_LogLoss'),
}

# ════════════════════════════════════════════════════════════
# VERIFY: has_topic_trained_on=0 -> 0% adoption in train
# ════════════════════════════════════════════════════════════
mask_train_0 = train['has_topic_trained_on'] == 0
print(f"Train rows with topic=0: {mask_train_0.sum()}")
for t in TARGETS:
    short = t.split('_')[-2] + 'd'
    n_adopted = train.loc[mask_train_0, t].sum()
    print(f"  {short}: {int(n_adopted)} adopted out of {mask_train_0.sum()} (rate={n_adopted/mask_train_0.sum():.4f})")

# ════════════════════════════════════════════════════════════
# CHECK: What V7 predicts for topic=0 test rows
# ════════════════════════════════════════════════════════════
mask_test_0 = test['has_topic_trained_on'] == 0
n_test_0 = mask_test_0.sum()
print(f"\nTest rows with topic=0: {n_test_0}")

print("\nV7 predictions for topic=0 test rows:")
for t in TARGETS:
    auc_col, ll_col = TARGET_COLS[t]
    short = t.split('_')[-2] + 'd'
    auc_val = v7.loc[mask_test_0.values, auc_col].mean()
    ll_val = v7.loc[mask_test_0.values, ll_col].mean()
    print(f"  {short}: AUC_col={auc_val:.4f}, LL_col={ll_val:.4f}")

# ════════════════════════════════════════════════════════════
# FIX: Force topic=0 rows to 0.001 in BOTH columns
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("APPLYING FIX: topic=0 -> 0.001")
print("=" * 60)

out = v7.copy()
for t in TARGETS:
    auc_col, ll_col = TARGET_COLS[t]
    out.loc[mask_test_0.values, auc_col] = 0.001
    out.loc[mask_test_0.values, ll_col] = 0.001

out.to_csv('sub_V9_A_v7_topicfix.csv', index=False)

print(f"\nSaved: sub_V9_A_v7_topicfix.csv")
print(f"Changed {n_test_0} rows (of {len(test)} total)")
print(f"\nLB Score: 0.7247 (was 0.723)")

# ════════════════════════════════════════════════════════════
# VERIFY OUTPUT
# ════════════════════════════════════════════════════════════
print("\nOutput summary:")
for t in TARGETS:
    auc_col, ll_col = TARGET_COLS[t]
    short = t.split('_')[-2] + 'd'
    au = out[auc_col].nunique()
    lu = out[ll_col].nunique()
    am = out[auc_col].mean()
    lm = out[ll_col].mean()
    print(f"  {short}: AUC(u={au:4d}, m={am:.4f}) LL(u={lu:4d}, m={lm:.4f})")
