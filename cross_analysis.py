import pandas as pd
import ast

prior = pd.read_csv('Prior.csv')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Date ranges
td = 'training_day'
print("4. DATE RANGES:")
print(f"  Prior: {prior[td].min()} to {prior[td].max()}")
print(f"  Train: {train[td].min()} to {train[td].max()}")
print(f"  Test:  {test[td].min()} to {test[td].max()}")
print()

# Target rates
print("5. TARGET RATES:")
for t in ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']:
    print(f"  Prior {t}: {prior[t].mean():.4f} ({prior[t].sum()}/{len(prior)})")
    print(f"  Train {t}: {train[t].mean():.4f} ({train[t].sum()}/{len(train)})")
    print()

# Topics format
print("6. TOPICS_LIST FORMAT:")
print(f"  Prior sample[0]: {prior['topics_list'].iloc[0]}")
print(f"  Prior sample[2]: {prior['topics_list'].iloc[2]}")
print(f"  Train sample[0]: {train['topics_list'].iloc[0]}")
print(f"  Train sample[1]: {train['topics_list'].iloc[1]}")
print(f"  Test sample[0]:  {test['topics_list'].iloc[0]}")

p0 = ast.literal_eval(prior['topics_list'].iloc[0])
print(f"  Prior[0] parsed: type={type(p0).__name__}, inner_type={type(p0[0]).__name__}, content={p0}")
t0 = ast.literal_eval(train['topics_list'].iloc[0])
print(f"  Train[0] parsed: type={type(t0).__name__}, inner_type={type(t0[0]).__name__}, content={t0}")
t1 = ast.literal_eval(train['topics_list'].iloc[1])
print(f"  Train[1] parsed: type={type(t1).__name__}, inner_type={type(t1[0]).__name__}, content={t1}")
print()

# Trainer format
print("7. TRAINER FORMAT:")
print(f"  Prior[0]: '{prior['trainer'].iloc[0]}' starts_with_[: {str(prior['trainer'].iloc[0]).startswith('[')}")
print(f"  Train[0]: '{train['trainer'].iloc[0]}' starts_with_[: {str(train['trainer'].iloc[0]).startswith('[')}")
print()

# Geographic overlap
print("8. GEOGRAPHIC OVERLAP:")
for col in ['county', 'subcounty', 'ward', 'trainer']:
    p = set(prior[col].unique())
    tr = set(train[col].unique())
    te = set(test[col].unique())
    print(f"  {col}: Prior={len(p)} Train={len(tr)} Test={len(te)} P&Tr={len(p&tr)} P&Te={len(p&te)} Tr&Te={len(tr&te)}")
print()

# Prior has multiple rows per farmer?
print("9. PRIOR FARMER MULTIPLICITY:")
fc = prior.groupby('farmer_name').size()
print(f"  Prior: {len(prior)} rows, {prior['farmer_name'].nunique()} unique farmers")
print(f"  Prior avg rows/farmer: {fc.mean():.1f}, max: {fc.max()}, >1 sessions: {(fc > 1).sum()}")
fc_tr = train.groupby('farmer_name').size()
print(f"  Train: {len(train)} rows, {train['farmer_name'].nunique()} unique farmers")
print(f"  Train avg rows/farmer: {fc_tr.mean():.1f}, max: {fc_tr.max()}, >1: {(fc_tr > 1).sum()}")
print()

# Can we use Prior to build farmer history features for test?
pf = set(prior['farmer_name'])
trf = set(train['farmer_name'])
tef = set(test['farmer_name'])
print("10. FARMER HISTORY FROM PRIOR:")
print(f"  Test farmers in Prior: {len(tef & pf)}/{len(tef)} ({100*len(tef & pf)/len(tef):.1f}%)")
print(f"  Test farmers in Train: {len(tef & trf)}/{len(tef)} ({100*len(tef & trf)/len(tef):.1f}%)")
print(f"  Test farmers in Prior+Train: {len(tef & (pf | trf))}/{len(tef)} ({100*len(tef & (pf | trf))/len(tef):.1f}%)")
print()

# Prior adoption rates for farmers who appear in test
prior_test_farmers = prior[prior['farmer_name'].isin(tef)]
print(f"  Prior rows for test farmers: {len(prior_test_farmers)}")
if len(prior_test_farmers) > 0:
    for t in ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']:
        print(f"    {t}: {prior_test_farmers[t].mean():.4f}")

# How many prior sessions have tests farmers had?
if len(prior_test_farmers) > 0:
    prior_sessions = prior_test_farmers.groupby('farmer_name').size()
    print(f"  Avg prior sessions for test farmers: {prior_sessions.mean():.1f}")
print()

# Topics uniqueness - Prior uses flat list, Train uses list-of-lists?
print("11. TOPIC FORMAT DEEP CHECK:")
# Check a few Prior entries
for i in [0, 5, 100, 500]:
    raw = prior['topics_list'].iloc[i]
    parsed = ast.literal_eval(raw)
    is_nested = isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list)
    print(f"  Prior[{i}]: nested={is_nested}, len={len(parsed)}, first_type={type(parsed[0]).__name__ if parsed else 'empty'}")

# Check a few Train entries
for i in [0, 5, 100, 500]:
    raw = train['topics_list'].iloc[i]
    parsed = ast.literal_eval(raw)
    is_nested = isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list)
    print(f"  Train[{i}]: nested={is_nested}, len={len(parsed)}, first_type={type(parsed[0]).__name__ if parsed else 'empty'}")
print()

# has_topic_trained_on
print("12. HAS_TOPIC_TRAINED_ON:")
print(f"  Prior: 0={( prior['has_topic_trained_on']==0).sum()}, 1={(prior['has_topic_trained_on']==1).sum()}")
print(f"  Train: 0={( train['has_topic_trained_on']==0).sum()}, 1={(train['has_topic_trained_on']==1).sum()}")
print(f"  Test:  0={(test['has_topic_trained_on']==0).sum()}, 1={(test['has_topic_trained_on']==1).sum()}")
print()

# Prior + Train combined potential 
print("13. COMBINED PRIOR + TRAIN SIZE:")
print(f"  If we use Prior as extra training data: {len(prior) + len(train)} rows")
# Check for compatible formats
print(f"  Prior trainer format starts with [: {str(prior['trainer'].iloc[0]).startswith('[')}")
print(f"  Prior topics format nested: ", end='')
p_parsed = ast.literal_eval(prior['topics_list'].iloc[0])
print(f"{isinstance(p_parsed, list) and isinstance(p_parsed[0], str)}")
