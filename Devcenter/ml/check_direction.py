import pandas as pd

df = pd.read_csv('Devcenter/ml/ml_data/ml_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'\nDirection distribution:')
print(df['direction'].value_counts())
print(f'\nDirection value counts:')
print(df['direction'].value_counts(normalize=True) * 100)
print(f'\nUnique directions: {df["direction"].unique()}')
