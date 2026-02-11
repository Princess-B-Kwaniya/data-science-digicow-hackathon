"""V5 is BEST (0.93892). Create V5-centered blends with V4 (0.93544)."""
import pandas as pd
import numpy as np

ss = pd.read_csv('SampleSubmission.csv')
cols = [c for c in ss.columns if c != 'ID']

v2 = pd.read_csv('sub_ULT_C_bayesian_dual.csv')       # LB 0.93333
v4 = pd.read_csv('sub_V4_A_calibrated_dual.csv')       # LB 0.93544
v5a = pd.read_csv('sub_V5_A_ensemble_dual.csv')        # LB 0.93892 (BEST!)
v5b = pd.read_csv('sub_V5_B_bayesian_dual.csv')
v5c = pd.read_csv('sub_V5_C_lgb_dual.csv')

print("=== V5-CENTERED BLENDS (V5 = 0.93892 is BEST) ===\n")

# V5 + V4 blends (V5-heavy)
for wv5, wv4 in [(0.90, 0.10), (0.85, 0.15), (0.80, 0.20), (0.75, 0.25),
                  (0.70, 0.30), (0.60, 0.40), (0.50, 0.50)]:
    blend = v5a.copy()
    for c in cols:
        blend[c] = wv5 * v5a[c] + wv4 * v4[c]
    fname = f'sub_TOP_{int(wv5*100)}v5_{int(wv4*100)}v4.csv'
    blend.to_csv(fname, index=False)
    print(f"  SAVED: {fname}")

# V5 + V4 + V2 triple (V5-heavy)
for wv5, wv4, wv2 in [(0.70, 0.20, 0.10), (0.60, 0.25, 0.15), (0.80, 0.15, 0.05)]:
    blend = v5a.copy()
    for c in cols:
        blend[c] = wv5 * v5a[c] + wv4 * v4[c] + wv2 * v2[c]
    fname = f'sub_TOP_triple_{int(wv5*100)}v5_{int(wv4*100)}v4_{int(wv2*100)}v2.csv'
    blend.to_csv(fname, index=False)
    print(f"  SAVED: {fname}")

# Cherry-pick: V5 LogLoss + V4 AUC (V5 has better calibrated probs, V4 might have different ranking)
cherry1 = v5a.copy()
for c in cols:
    if 'AUC' in c:
        cherry1[c] = v4[c]
cherry1.to_csv('sub_TOP_cherry_v5ll_v4auc.csv', index=False)
print(f"  SAVED: sub_TOP_cherry_v5ll_v4auc.csv")

# Cherry-pick reverse: V4 LogLoss + V5 AUC
cherry2 = v5a.copy()
for c in cols:
    if 'AUC' in c:
        cherry2[c] = v5a[c]
    else:
        cherry2[c] = v4[c]
cherry2.to_csv('sub_TOP_cherry_v4ll_v5auc.csv', index=False)
print(f"  SAVED: sub_TOP_cherry_v4ll_v5auc.csv")

# Blend only LogLoss columns (keep V5 AUC intact since it's best)
for wv5 in [0.85, 0.80, 0.75]:
    blend = v5a.copy()
    for c in cols:
        if 'LogLoss' in c:
            blend[c] = wv5 * v5a[c] + (1-wv5) * v4[c]
    fname = f'sub_TOP_llonly_{int(wv5*100)}v5_{int((1-wv5)*100)}v4.csv'
    blend.to_csv(fname, index=False)
    print(f"  SAVED: {fname}")

print("\n=== PRIORITY SUBMISSION ORDER ===")
print("PROVEN:  sub_V5_A_ensemble_dual.csv = 0.93892 (CURRENT BEST)")
print()
print("NEXT SUBMISSIONS:")
print("  1. sub_TOP_90v5_10v4.csv           - 90% V5 + 10% V4 diversity")
print("  2. sub_TOP_85v5_15v4.csv           - 85% V5 + 15% V4")
print("  3. sub_TOP_80v5_20v4.csv           - 80% V5 + 20% V4")
print("  4. sub_TOP_cherry_v5ll_v4auc.csv   - V5 LogLoss + V4 AUC ranking")
print("  5. sub_V5_B_bayesian_dual.csv      - V5 Bayesian variant")
print()
print("RATIONALE: V5 dominates. Small V4 blend (10-20%) adds diversity")
print("from different Optuna params without diluting V5's superior features.")
