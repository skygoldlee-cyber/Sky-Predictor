import pandas as pd

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')

print(f'size_factor 고유값: {df["size_factor"].unique()}')
print(f'\nsize_factor 통계:')
print(df['size_factor'].describe())
print(f'\nsize_factor별 거래 수:')
print(df['size_factor'].value_counts())
