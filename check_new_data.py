import pandas as pd
import ast

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

print('topics_list samples:')
for i in range(5):
    print(f'  {repr(train["topics_list"].iloc[i])}')

print('\ntrainer samples:')
for i in range(5):
    print(f'  {repr(train["trainer"].iloc[i])}')

print(f'\nfarmer_name samples: {train["farmer_name"].head(5).tolist()}')
print(f'farmer_name unique train: {train["farmer_name"].nunique()}')
print(f'farmer_name unique test: {test["farmer_name"].nunique()}')
overlap = len(set(train['farmer_name']) & set(test['farmer_name']))
print(f'farmer overlap: {overlap}')

for t in ['adopted_within_07_days','adopted_within_90_days','adopted_within_120_days']:
    print(f'{t}: {train[t].mean():.4f}')

print(f'has_topic_trained_on==0 train: {(train["has_topic_trained_on"]==0).sum()}')
print(f'has_topic_trained_on==0 test: {(test["has_topic_trained_on"]==0).sum()}')

print(f'Training day range train: {train["training_day"].min()} to {train["training_day"].max()}')
print(f'Training day range test: {test["training_day"].min()} to {test["training_day"].max()}')

# Parse topics_list to understand structure
sample = train['topics_list'].iloc[0]
print(f'\nParsed topics_list[0]: {ast.literal_eval(sample)}')
sample2 = train['topics_list'].iloc[1]
print(f'Parsed topics_list[1]: {ast.literal_eval(sample2)}')

# Parse trainer
t0 = train['trainer'].iloc[0]
print(f'\nParsed trainer[0]: {ast.literal_eval(t0)}')

# Check group/county overlap
for col in ['group_name', 'county', 'subcounty', 'ward', 'trainer']:
    overlap = len(set(train[col]) & set(test[col]))
    total_test = test[col].nunique()
    print(f'{col} overlap: {overlap}/{total_test}')

# SS column order
ss = pd.read_csv('SampleSubmission.csv')
print(f'\nSS columns: {list(ss.columns)}')
print(f'SS head:\n{ss.head(2)}')
