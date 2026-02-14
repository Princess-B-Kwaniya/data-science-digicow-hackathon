import pandas as pd, numpy as np
from scipy.stats import rankdata
from scipy.special import logit, expit

best = pd.read_csv('sub_ADV_pertarget_progressive.csv')
v4e3 = pd.read_csv('submission_v4_ensemble (3).csv').set_index('ID').reindex(best['ID']).reset_index()
auc_cols = ['Target_07_AUC','Target_90_AUC','Target_120_AUC']
ll_cols = ['Target_07_LogLoss','Target_90_LogLoss','Target_120_LogLoss']
N = len(best)

def rklo_ll(out, best, v4e3, N, rank_w=0.80, dist_v4=0.60):
    """Apply RKLO to LL columns: rank_w=V4 rank weight, dist_v4=V4 distribution weight"""
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
    """Rank-avg blend AUC columns with auc_w = V4 weight"""
    for col in auc_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = (1-auc_w) * r_best + auc_w * r_v4
        sorted_vals = np.sort(best[col].values)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    return out

def enforce_monotonic(out):
    out['Target_90_LogLoss'] = np.maximum(out['Target_90_LogLoss'], out['Target_07_LogLoss'])
    out['Target_120_LogLoss'] = np.maximum(out['Target_120_LogLoss'], out['Target_90_LogLoss'])
    return out

# === GRID 1: AUC weight gradient (15-50%) with best LL config ===
for auc_pct in [15, 20, 25, 30, 35, 40, 45, 50]:
    out = best.copy()
    out = rklo_ll(out, best, v4e3, N, rank_w=0.80, dist_v4=0.60)
    out = blend_auc(out, best, v4e3, N, auc_w=auc_pct/100)
    out = enforce_monotonic(out)
    fname = f'sub_BEST2_rklo80d60_auc{auc_pct}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'Saved: {fname}')

# === GRID 2: AUC with RKLO distribution too (not just rank-avg for AUC) ===
for auc_pct in [10, 15, 20, 25, 30]:
    out = best.copy()
    out = rklo_ll(out, best, v4e3, N, rank_w=0.80, dist_v4=0.60)
    # RKLO for AUC too
    for col in auc_cols:
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        aw = auc_pct / 100
        blended_ranks = (1-aw) * r_best + aw * r_v4
        sorted_best = np.sort(best[col].values)
        sorted_v4 = np.sort(v4e3[col].values)
        b_logit = logit(np.clip(sorted_best, 1e-6, 1-1e-6))
        v_logit = logit(np.clip(sorted_v4, 1e-6, 1-1e-6))
        sorted_vals = expit((1-aw) * b_logit + aw * v_logit)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    out = enforce_monotonic(out)
    fname = f'sub_BEST2_rkloFULL_auc{auc_pct}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'Saved: {fname}')

# === GRID 3: Per-target AUC weights ===
auc_configs = {
    'A': {'07': 15, '90': 10, '120': 5},
    'B': {'07': 5, '90': 10, '120': 15},
    'C': {'07': 10, '90': 15, '120': 20},
    'D': {'07': 20, '90': 15, '120': 10},
    'E': {'07': 15, '90': 15, '120': 15},
    'F': {'07': 20, '90': 20, '120': 20},
}
for label, weights in auc_configs.items():
    out = best.copy()
    out = rklo_ll(out, best, v4e3, N, rank_w=0.80, dist_v4=0.60)
    for col in auc_cols:
        if '07' in col: aw = weights['07'] / 100
        elif '90' in col: aw = weights['90'] / 100
        else: aw = weights['120'] / 100
        r_best = rankdata(best[col]) / N
        r_v4 = rankdata(v4e3[col]) / N
        blended_ranks = (1-aw) * r_best + aw * r_v4
        sorted_vals = np.sort(best[col].values)
        rank_order = rankdata(blended_ranks, method='ordinal') - 1
        out[col] = sorted_vals[rank_order.astype(int)]
    out = enforce_monotonic(out)
    fname = f'sub_BEST2_aucPT_{label}_{weights["07"]}_{weights["90"]}_{weights["120"]}.csv'
    out[['ID']+auc_cols+ll_cols].to_csv(fname, index=False)
    print(f'Saved: {fname}')

print('\nAll BEST2 variants generated!')
