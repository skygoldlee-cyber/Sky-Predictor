import pandas as pd

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')

print('연도별 성과 분석:')
print('=' * 80)
for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]
    print(f'\n{year}년:')
    print(f'  거래 수: {len(df_year)}건')
    print(f'  승률: {df_year["is_win"].mean() * 100:.2f}%')
    print(f'  총 PnL: {df_year["net_krw"].sum():,.0f} 원')
    print(f'  평균 PnL: {df_year["net_krw"].mean():,.0f} 원')

print('\n' + '=' * 80)
print('시간 경과에 따른 승률 추이:')
print('=' * 80)
yearly_winrate = df.groupby('year')['is_win'].mean() * 100
print(yearly_winrate)
