import pandas as pd, numpy as np
from scipy.stats import rankdata
from scipy.special import logit, expit

best = pd.read_csv('sub_ADV_pertarget_progressive.csv')
v4e3 = pd.read_csv('submission_v4_ensemble (3).csv').set_index('ID').reindex(best['ID']).reset_index()
auc_cols = ['Target_07_AUC','Target_90_AUC','Target_120_AUC']
ll_cols = ['Target_07_LogLoss','Target_90_LogLoss','Target_120_LogLoss']
N = len(best)

# BEST COMBO 1: Per-target RKLO with d60
configs = {
    'X': {'07': 82, '90': 78, '120': 78},
    'Y': {'07': 80, '90': 80, '120': 82},
    'Z': {'07': 78, '90': 80, '120': 82},
}
for label, weights in configs.items():
    out = best.copy()
    for col in ll_cols:
        if '07' in col:
            v4w = weights['07'] / 100
        elif '90' in col:
            v4w = weights['90'] / 100
        else:
            v4w = weights['120'] / 100
        bw = 1 - v4w
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = bw * r_best + v4w * r_v4
        sorted_best = np.sort(best[col].values)
        sorted_v4 = np.sort(v4e3[col].values)
        b_logit = logit(np.clip(sorted_best, 1e-6, 1-1e-6))
        v_logit = logit(np.clip(sorted_v4, 1e-6, 1-1e-6))
        sorted_vals = expit(0.4 * b_logit + 0.6 * v_logit)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    out['Target_90_LogLoss'] = np.maximum(out['Target_90_LogLoss'], out['Target_07_LogLoss'])
    out['Target_120_LogLoss'] = np.maximum(out['Target_120_LogLoss'], out['Target_90_LogLoss'])
    fname = f'sub_BEST_RKLO_PT{label}_d60.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'Saved: {fname} | 07d={weights["07"]}% 90d={weights["90"]}% 120d={weights["120"]}%')

# BEST COMBO 2: RKLO r80 d60 + light AUC blending
for auc_pct in [5, 10, 15]:
    out = best.copy()
    for col in ll_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = 0.20 * r_best + 0.80 * r_v4
        sorted_best = np.sort(best[col].values)
        sorted_v4 = np.sort(v4e3[col].values)
        b_logit = logit(np.clip(sorted_best, 1e-6, 1-1e-6))
        v_logit = logit(np.clip(sorted_v4, 1e-6, 1-1e-6))
        sorted_vals = expit(0.4 * b_logit + 0.6 * v_logit)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    aw = auc_pct / 100
    for col in auc_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = (1-aw) * r_best + aw * r_v4
        sorted_vals = np.sort(best[col].values)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    out['Target_90_LogLoss'] = np.maximum(out['Target_90_LogLoss'], out['Target_07_LogLoss'])
    out['Target_120_LogLoss'] = np.maximum(out['Target_120_LogLoss'], out['Target_90_LogLoss'])
    fname = f'sub_BEST_RKLO80d60_auc{auc_pct}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'Saved: {fname} | LL=RKLO(r80,d60) + AUC={auc_pct}% V4')

print('Done!')
