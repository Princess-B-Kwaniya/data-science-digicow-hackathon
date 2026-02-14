"""
DEEP EDA ANALYSIS - Understanding Train/Test/Prior deeply
==========================================================
Purpose: Deep data understanding before building improved pipeline
"""
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
prior_df = pd.read_csv('Prior.csv')
ss = pd.read_csv('SampleSubmission.csv')

print("="*80)
print("1. BASIC SHAPES & SCHEMA")
print("="*80)
print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print(f"Prior: {prior_df.shape}")
print(f"SS:    {ss.shape}")

print("\n--- TRAIN COLUMNS ---")
for i, (c, dt) in enumerate(zip(train_df.columns, train_df.dtypes)):
    null_pct = train_df[c].isnull().mean()*100
    print(f"  {i}: {c:30s} {str(dt):10s} nulls={null_pct:.1f}%  nunique={train_df[c].nunique()}")

print("\n--- TEST COLUMNS ---")
for i, (c, dt) in enumerate(zip(test_df.columns, test_df.dtypes)):
    null_pct = test_df[c].isnull().mean()*100
    print(f"  {i}: {c:30s} {str(dt):10s} nulls={null_pct:.1f}%  nunique={test_df[c].nunique()}")

print("\n--- PRIOR COLUMNS ---")
for i, (c, dt) in enumerate(zip(prior_df.columns, prior_df.dtypes)):
    null_pct = prior_df[c].isnull().mean()*100
    print(f"  {i}: {c:30s} {str(dt):10s} nulls={null_pct:.1f}%  nunique={prior_df[c].nunique()}")

print("\n\n" + "="*80)
print("2. TARGET DISTRIBUTIONS")
print("="*80)
targets = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
for t in targets:
    tr_rate = train_df[t].mean()
    pr_rate = prior_df[t].mean()
    print(f"{t}:")
    print(f"  Train: {tr_rate:.4f} ({train_df[t].sum()}/{len(train_df)})")
    print(f"  Prior: {pr_rate:.4f} ({prior_df[t].sum()}/{len(prior_df)})")
    print(f"  Ratio Prior/Train: {pr_rate/tr_rate:.2f}x")

# Target correlations
print("\nTarget correlations (Train):")
print(train_df[targets].corr().round(4))

# Monotonicity check
print("\nMonotonicity check (07 <= 90 <= 120):")
mono_ok = (train_df['adopted_within_07_days'] <= train_df['adopted_within_90_days']).all() and \
          (train_df['adopted_within_90_days'] <= train_df['adopted_within_120_days']).all()
print(f"  Train monotonicity holds: {mono_ok}")
# Check who adopts at 7d but NOT 90d (should be 0)
v1 = ((train_df['adopted_within_07_days']==1) & (train_df['adopted_within_90_days']==0)).sum()
v2 = ((train_df['adopted_within_90_days']==1) & (train_df['adopted_within_120_days']==0)).sum()
print(f"  7d=1 but 90d=0: {v1}")
print(f"  90d=1 but 120d=0: {v2}")

# Incremental adoption
a07 = train_df['adopted_within_07_days'].sum()
a90 = train_df['adopted_within_90_days'].sum()
a120 = train_df['adopted_within_120_days'].sum()
print(f"\n  7d adopters: {a07}")
print(f"  New 90d adopters (not at 7d): {a90 - a07}")
print(f"  New 120d adopters (not at 90d): {a120 - a90}")

print("\n\n" + "="*80)
print("3. CATEGORICAL FEATURE DISTRIBUTIONS")
print("="*80)

for col in ['gender', 'age', 'registration', 'belong_to_cooperative']:
    print(f"\n--- {col} ---")
    for df_name, df_data in [('Train', train_df), ('Test', test_df), ('Prior', prior_df)]:
        vc = df_data[col].value_counts()
        print(f"  {df_name}: {dict(vc.head(10))}")
    
    if col in train_df.columns:
        for t in targets:
            rates = train_df.groupby(col)[t].agg(['mean', 'count'])
            print(f"  Train {t} rates:")
            for idx, row in rates.iterrows():
                print(f"    {idx}: {row['mean']:.4f} (n={int(row['count'])})")

print("\n\n" + "="*80)
print("4. HIGH-CARDINALITY FEATURES")
print("="*80)
for col in ['county', 'subcounty', 'ward', 'group_name']:
    print(f"\n--- {col} ---")
    for df_name, df_data in [('Train', train_df), ('Test', test_df), ('Prior', prior_df)]:
        if col in df_data.columns:
            print(f"  {df_name}: {df_data[col].nunique()} unique values")
    
    # Overlap analysis
    train_vals = set(train_df[col].dropna().unique())
    test_vals = set(test_df[col].dropna().unique())
    prior_vals = set(prior_df[col].dropna().unique()) if col in prior_df.columns else set()
    
    overlap_tt = len(train_vals & test_vals) / max(len(test_vals), 1) * 100
    overlap_pt = len(prior_vals & test_vals) / max(len(test_vals), 1) * 100
    overlap_ptr = len(prior_vals & train_vals) / max(len(train_vals), 1) * 100
    
    print(f"  Test values in Train: {overlap_tt:.1f}%")
    print(f"  Test values in Prior: {overlap_pt:.1f}%")
    print(f"  Train values in Prior: {overlap_ptr:.1f}%")
    
    # Unseen test values
    unseen = test_vals - train_vals - prior_vals
    print(f"  Test values unseen (not in Train or Prior): {len(unseen)} ({len(unseen)/len(test_vals)*100:.1f}%)")

print("\n\n" + "="*80)
print("5. FARMER OVERLAP ANALYSIS (farmer_name)")
print("="*80)
train_farmers = set(train_df['farmer_name'].unique())
test_farmers = set(test_df['farmer_name'].unique())
prior_farmers = set(prior_df['farmer_name'].unique())

print(f"Train unique farmers: {len(train_farmers)}")
print(f"Test unique farmers: {len(test_farmers)}")
print(f"Prior unique farmers: {len(prior_farmers)}")
print(f"Test ∩ Train: {len(test_farmers & train_farmers)} ({len(test_farmers & train_farmers)/len(test_farmers)*100:.1f}%)")
print(f"Test ∩ Prior: {len(test_farmers & prior_farmers)} ({len(test_farmers & prior_farmers)/len(test_farmers)*100:.1f}%)")
print(f"Train ∩ Prior: {len(train_farmers & prior_farmers)} ({len(train_farmers & prior_farmers)/len(train_farmers)*100:.1f}%)")
print(f"Test only (no history): {len(test_farmers - train_farmers - prior_farmers)} ({len(test_farmers - train_farmers - prior_farmers)/len(test_farmers)*100:.1f}%)")

# Farmer duplication
print(f"\nTrain rows per farmer: {train_df.groupby('farmer_name').size().describe()}")
print(f"\nPrior rows per farmer: {prior_df.groupby('farmer_name').size().describe()}")

print("\n\n" + "="*80)
print("6. TRAINER ANALYSIS")
print("="*80)

# Parse trainers
def parse_trainer(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return str(parsed)
    except:
        return str(s)

train_df['trainer_p'] = train_df['trainer'].apply(parse_trainer)
test_df['trainer_p'] = test_df['trainer'].apply(parse_trainer)

train_trainers = set(train_df['trainer_p'].unique())
test_trainers = set(test_df['trainer_p'].unique())
prior_trainers = set(prior_df['trainer'].unique()) if 'trainer' in prior_df.columns else set()

print(f"Train unique trainers: {len(train_trainers)}")
print(f"Test unique trainers: {len(test_trainers)}")
print(f"Prior unique trainers: {len(prior_trainers)}")
print(f"Test trainers ∩ Train: {len(test_trainers & train_trainers)} ({len(test_trainers & train_trainers)/len(test_trainers)*100:.1f}%)")
print(f"Test trainers ∩ Prior: {len(test_trainers & prior_trainers)}")
print(f"Train trainers ∩ Prior: {len(train_trainers & prior_trainers)}")

# Trainer effectiveness (top & bottom)
trainer_stats = train_df.groupby('trainer_p').agg(
    count=('adopted_within_07_days', 'count'),
    rate_07=('adopted_within_07_days', 'mean'),
    rate_90=('adopted_within_90_days', 'mean'),
    rate_120=('adopted_within_120_days', 'mean'),
)
print(f"\nTrainer effectiveness (min 50 farmers, train):")
big_trainers = trainer_stats[trainer_stats['count'] >= 50].sort_values('rate_90', ascending=False)
print(f"  Top 5:")
for idx, row in big_trainers.head(5).iterrows():
    print(f"    {idx}: n={int(row['count'])}, 7d={row['rate_07']:.4f}, 90d={row['rate_90']:.4f}, 120d={row['rate_120']:.4f}")
print(f"  Bottom 5:")
for idx, row in big_trainers.tail(5).iterrows():
    print(f"    {idx}: n={int(row['count'])}, 7d={row['rate_07']:.4f}, 90d={row['rate_90']:.4f}, 120d={row['rate_120']:.4f}")

print("\n\n" + "="*80)
print("7. TOPICS ANALYSIS")
print("="*80)

def parse_topics_nested(s):
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
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return list(set(str(x) for x in parsed))
        return [str(parsed)]
    except:
        return []

train_df['topics_p'] = train_df['topics_list'].apply(parse_topics_nested)
test_df['topics_p'] = test_df['topics_list'].apply(parse_topics_nested)
prior_df['topics_p'] = prior_df['topics_list'].apply(parse_topics_flat)

# Unique topics
all_train_topics = set(t for topics in train_df['topics_p'] for t in topics)
all_test_topics = set(t for topics in test_df['topics_p'] for t in topics)
all_prior_topics = set(t for topics in prior_df['topics_p'] for t in topics)
print(f"Unique topics Train: {len(all_train_topics)}")
print(f"Unique topics Test: {len(all_test_topics)}")
print(f"Unique topics Prior: {len(all_prior_topics)}")

# has_topic_trained_on analysis
print(f"\nhas_topic_trained_on distribution:")
for df_name, df_data in [('Train', train_df), ('Test', test_df), ('Prior', prior_df)]:
    col = 'has_topic_trained_on' if 'has_topic_trained_on' in df_data.columns else 'has_topic_traine_on'
    if col in df_data.columns:
        print(f"  {df_name}: {df_data[col].value_counts().to_dict()}")

# Has_topic vs adoption
if 'has_topic_trained_on' in train_df.columns:
    htcol = 'has_topic_trained_on'
elif 'has_topic_traine_on' in train_df.columns:
    htcol = 'has_topic_traine_on'
else:
    htcol = None

if htcol:
    print(f"\nAdoption rates by {htcol} (Train):")
    for t in targets:
        rates = train_df.groupby(htcol)[t].mean()
        print(f"  {t}: {rates.to_dict()}")

print("\n\n" + "="*80)
print("8. TEMPORAL PATTERNS")
print("="*80)
train_df['dt'] = pd.to_datetime(train_df['training_day'])
test_df['dt'] = pd.to_datetime(test_df['training_day'])
prior_df['dt'] = pd.to_datetime(prior_df['training_day'])

for df_name, df_data in [('Train', train_df), ('Test', test_df), ('Prior', prior_df)]:
    print(f"\n{df_name} date range: {df_data['dt'].min()} to {df_data['dt'].max()}")
    print(f"  Months: {sorted(df_data['dt'].dt.month.unique())}")
    print(f"  Years: {sorted(df_data['dt'].dt.year.unique())}")

# Monthly adoption rates
print("\nMonthly adoption rates (Train):")
train_df['month'] = train_df['dt'].dt.month
monthly = train_df.groupby('month')[targets].mean()
print(monthly.round(4))

# Day of week
train_df['dow'] = train_df['dt'].dt.dayofweek
dow = train_df.groupby('dow')[targets].mean()
print("\nDay-of-week adoption rates (Train):")
print(dow.round(4))

print("\n\n" + "="*80)
print("9. BELONG_TO_COOPERATIVE ANALYSIS")
print("="*80)
coop_rates = train_df.groupby('belong_to_cooperative')[targets].mean()
print("Adoption rates by cooperative membership (Train):")
print(coop_rates.round(4))
coop_counts = train_df['belong_to_cooperative'].value_counts()
print(f"Distribution: {coop_counts.to_dict()}")

print("\n\n" + "="*80)
print("10. PRIOR DATA DEEP DIVE - DISTRIBUTION DIFFERENCES")
print("="*80)
# Compare adoption rate distributions
print("Prior adoption rates by cooperative:")
p_coop = prior_df.groupby('belong_to_cooperative')[targets].mean()
print(p_coop.round(4))

print("\nPrior adoption by gender:")
p_gen = prior_df.groupby('gender')[targets].mean()
print(p_gen.round(4))

print("\nTrain adoption by gender:")
t_gen = train_df.groupby('gender')[targets].mean()
print(t_gen.round(4))

print("\n\nPrior vs Train adoption by age:")
print("Prior:")
print(prior_df.groupby('age')[targets].mean().round(4))
print("\nTrain:")
print(train_df.groupby('age')[targets].mean().round(4))

# Group size distribution comparison
print("\n\nGroup size distributions:")
train_gs = train_df.groupby('group_name').size()
prior_gs = prior_df.groupby('group_name').size()
print(f"Train group sizes: mean={train_gs.mean():.1f}, median={train_gs.median():.0f}, max={train_gs.max()}")
print(f"Prior group sizes: mean={prior_gs.mean():.1f}, median={prior_gs.median():.0f}, max={prior_gs.max()}")

print("\n\n" + "="*80)
print("11. CALIBRATION ANALYSIS - How well are current predictions calibrated?")
print("="*80)
# Check mean prediction vs actual rate
print("For good calibration, mean predicted prob should ≈ actual rate")
print(f"Train target rates: 7d={train_df['adopted_within_07_days'].mean():.4f}, 90d={train_df['adopted_within_90_days'].mean():.4f}, 120d={train_df['adopted_within_120_days'].mean():.4f}")
print(f"Prior target rates: 7d={prior_df['adopted_within_07_days'].mean():.4f}, 90d={prior_df['adopted_within_90_days'].mean():.4f}, 120d={prior_df['adopted_within_120_days'].mean():.4f}")
print("NOTE: If test distribution is closer to train, we want predictions close to train rates")
print("      If test is mixed, Prior rates may bias predictions too high")

print("\n\n" + "="*80)
print("12. FEATURE CROSS-TABULATIONS & INTERACTIONS")
print("="*80)
# Gender x Age x Adoption
print("Gender × Age interaction (Train 90d adoption):")
ga = train_df.groupby(['gender', 'age']).agg(
    n=('adopted_within_90_days', 'count'),
    rate_90=('adopted_within_90_days', 'mean')
)
print(ga.round(4))

# Registration x Adoption
print("\nRegistration method adoption rates (Train):")
reg_rates = train_df.groupby('registration')[targets + ['registration']].agg({
    'adopted_within_07_days': ['mean', 'count'],
    'adopted_within_90_days': 'mean',
    'adopted_within_120_days': 'mean'
})
print(reg_rates.round(4))

print("\n\n" + "="*80)
print("13. METRIC UNDERSTANDING - Score Decomposition")
print("="*80)
print("Competition metric per target: 0.75*(1-LogLoss) + 0.25*AUC")
print("Total score = sum across 3 targets")
print("Max possible total score = 3.0")
print("")
print("CRITICAL: LogLoss weight is 3x AUC weight!")
print("  -> Calibration is 3x more important than ranking")
print("  -> Confident wrong predictions are VERY expensive")
print("  -> With ~1-2% positive rate, predicting near base rate is safe")
print("  -> Predicting too high: LogLoss explodes on 98% negatives")
print("  -> Predicting too low on positives: moderate LogLoss penalty")
print("")
print("Implication for DUAL strategy:")
print("  AUC columns: Only ranking matters → use rank percentiles")
print("  LogLoss columns: Calibration matters → use well-calibrated probabilities")
print("  This is EXACTLY what submission D (our best) does!")

# Simulate LogLoss sensitivity
from sklearn.metrics import log_loss as ll_fn
print("\n\nLogLoss sensitivity analysis (1% positive rate, 1000 samples):")
np.random.seed(42)
y_true = np.zeros(1000)
y_true[:10] = 1  # 1% positive
for p_val in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    preds = np.full(1000, p_val)
    loss = ll_fn(y_true, preds)
    score = 0.75 * (1 - loss)
    print(f"  pred={p_val:.3f} -> LogLoss={loss:.4f}, 0.75*(1-LL)={score:.4f}")

print("\n=> Even small changes in average prediction level dramatically affect LogLoss!")
print("=> Predicting const 0.01 (= true rate) gives best LogLoss for uniform predictions")

print("\n\n" + "="*80)
print("14. PRIOR-BASED BAYESIAN ANALYSIS")
print("="*80)
# For each farmer in test with prior history, compute Bayesian posterior
# Prior: P(adopt) = prior_rate_from_history
# Likelihood: P(features|adopt) ≈ model prediction
# Posterior ∝ Prior × Likelihood

# Analyze how many prior sessions farmers have
test_in_prior = test_df[test_df['farmer_name'].isin(prior_farmers)]
prior_session_counts = prior_df.groupby('farmer_name').size()
test_farmer_sessions = test_in_prior['farmer_name'].map(prior_session_counts)
print(f"Test farmers with prior history: {len(test_in_prior)}")
print(f"Session count distribution for test farmers in prior:")
print(test_farmer_sessions.describe())
print(f"\nBuckets:")
for thresh in [1, 2, 5, 10, 20, 50]:
    n = (test_farmer_sessions >= thresh).sum()
    print(f"  >= {thresh} sessions: {n} farmers ({n/len(test_in_prior)*100:.1f}%)")

# Farmer prior rate reliability
farmer_prior_stats = prior_df.groupby('farmer_name').agg(
    n=('adopted_within_90_days', 'count'),
    rate_07=('adopted_within_07_days', 'mean'),
    rate_90=('adopted_within_90_days', 'mean'),
    rate_120=('adopted_within_120_days', 'mean'),
)
# For farmers with many sessions, prior rate is reliable
reliable = farmer_prior_stats[farmer_prior_stats['n'] >= 5]
print(f"\nFarmers with >=5 prior sessions: {len(reliable)}")
print(f"  Average 90d adoption rate: {reliable['rate_90'].mean():.4f}")
print(f"  Median 90d adoption rate: {reliable['rate_90'].median():.4f}")
print(f"  % with zero 90d adoption: {(reliable['rate_90'] == 0).mean()*100:.1f}%")

print("\n=> Most farmers (even frequent ones) have 0% adoption in prior")
print("=> Bayesian update: strong prior of ~0 for most, model shifts toward global rate")
print("=> For farmers with HIGH prior adoption rates, Bayesian update should boost")

print("\n\n" + "="*80)
print("15. ANALYSIS COMPLETE - KEY INSIGHTS")
print("="*80)
print("""
KEY INSIGHTS FOR PIPELINE IMPROVEMENT:

1. CALIBRATION IS KING (75% of metric):
   - LogLoss punishes overconfident wrong predictions
   - With 1-2% base rates, predictions should stay near base rate
   - Currently best: DUAL strategy (rank for AUC, calibrated for LogLoss)

2. PRIOR DATA - USE AS FEATURES ONLY:
   - Prior rates are HIGHER than train rates
   - Training on Prior shifts predictions too high → kills LogLoss
   - USE: farmer history features, group history, geo history
   - DON'T USE: as training data

3. MULTI-MODEL ENSEMBLE OPPORTUNITY:
   - Current: single LightGBM with 205 features
   - Add: XGBoost, CatBoost with SAME features
   - Average: reduces variance, improves calibration
   - Seed diversity: multiple seeds of same model

4. BAYESIAN UPDATE POTENTIAL:
   - Use model prediction as likelihood
   - Use prior adoption rate as prior (where available)
   - Posterior = weighted combination
   - Already partially done via Prior blend (70/30)

5. OPTUNA TUNING:
   - Current params: num_leaves=63, lr=0.03
   - Could find better: especially regularization params
   - Tune for LogLoss primarily (75% of metric)

6. SAFE IMPROVEMENTS:
   - Do NOT add new features (proven to hurt)
   - Do NOT train on Prior (proven to hurt)
   - DO: multi-seed, multi-algorithm, better calibration
   - DO: optimize blend ratios, post-processing thresholds
""")
