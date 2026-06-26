import pandas as pd

df = pd.read_csv('Devcenter/ml/ml_data/ml_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Date range: {df["entry_time"].min()} to {df["entry_time"].max()}')
print(f'Unique dates: {df["entry_time"].str[:10].nunique()}')
print(f'Daily avg: {len(df) / df["entry_time"].str[:10].nunique():.2f}')
