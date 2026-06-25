# -*- coding: utf-8 -*-
"""2024년 데이터 분석"""
import pandas as pd

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')
df_2024 = df[df['year'] == 2024]

print(f'2024년 거래 수: {len(df_2024)}건')
print(f'2024년 승률: {df_2024["is_win"].mean() * 100:.2f}%')
print(f'2024년 총 PnL: {df_2024["net_krw"].sum():,.0f}원')
print(f'2024년 평균 PnL: {df_2024["net_krw"].mean():,.0f}원')

print(f'\n2024년 월별 거래 수:')
print(df_2024.groupby(df_2024['entry_time'].str[:7]).size())

print(f'\n2024년 월별 승률:')
print(df_2024.groupby(df_2024['entry_time'].str[:7])['is_win'].mean() * 100)

print(f'\n2024년 regime별 거래 수:')
print(df_2024.groupby('regime').size())

print(f'\n2024년 regime별 승률:')
print(df_2024.groupby('regime')['is_win'].mean() * 100)

print(f'\n2024년 시간대별 거래 수:')
print(df_2024.groupby('entry_hour').size())

print(f'\n2024년 시간대별 승률:')
print(df_2024.groupby('entry_hour')['is_win'].mean() * 100)
