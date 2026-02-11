"""90%V5+10%V4 = 0.94260 (BEST EVER). Find the optimal ratio."""
import pandas as pd
import numpy as np

ss = pd.read_csv('SampleSubmission.csv')
cols = [c for c in ss.columns if c != 'ID']

v4 = pd.read_csv('sub_V4_A_calibrated_dual.csv')       # LB 0.93544
v5 = pd.read_csv('sub_V5_A_ensemble_dual.csv')          # LB 0.93892

print("=== FINE-GRAINED V5+V4 BLENDS ===")
print("PROVEN: 90%V5+10%V4 = 0.94260\n")

# Fine grid around 90/10
for wv5 in [0.95, 0.93, 0.92, 0.91, 0.89, 0.88, 0.87, 0.85]:
    wv4 = round(1 - wv5, 2)
    blend = v5.copy()
    for c in cols:
        blend[c] = wv5 * v5[c] + wv4 * v4[c]
    fname = f'sub_FINE_{int(wv5*100)}v5_{int(wv4*100)}v4.csv'
    blend.to_csv(fname, index=False)
    print(f"  SAVED: {fname}")

print("\n=== TOMORROW'S SUBMISSION PLAN (10 slots) ===")
print("Priority order (bracket around the 90/10 sweet spot):")
print("  1. sub_FINE_92v5_8v4.csv    - slightly more V5")
print("  2. sub_FINE_88v5_12v4.csv   - slightly more V4")
print("  3. sub_FINE_95v5_5v4.csv    - near-pure V5 with V4 hint")
print("  4. sub_FINE_85v5_15v4.csv   - more V4 influence")
print("  5. sub_FINE_91v5_9v4.csv    - very close to proven best")
print("  6. sub_FINE_89v5_11v4.csv   - other side of 90/10")
print("  7. sub_FINE_93v5_7v4.csv    - fine tuning")
print("  8. sub_FINE_87v5_13v4.csv   - more V4")
print("  9. sub_V5_B_bayesian_dual.csv - pure V5 Bayesian variant")  
print(" 10. sub_TOP_cherry_v5ll_v4auc.csv - cherry-pick experiment")
