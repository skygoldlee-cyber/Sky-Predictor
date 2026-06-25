# -*- coding: utf-8 -*-
"""시장 구조 변화 분석"""
import pandas as pd

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')

# 연도별 기본 통계
print("연도별 기본 통계:")
year_stats = df.groupby('year').agg({
    'is_win': ['mean', 'count'],
    'net_krw': ['sum', 'mean'],
    'entry_atr': 'mean',
    'entry_rsi': 'mean',
    'entry_macd': 'mean'
}).round(2)
print(year_stats)

# 연도별 regime 분포
print("\n연도별 regime 분포:")
regime_dist = df.groupby(['year', 'regime']).size().unstack(fill_value=0)
print(regime_dist)

# 연도별 변동성 (ATR) 비교
print("\n연도별 평균 ATR:")
print(df.groupby('year')['entry_atr'].mean())

# 연도별 RSI 비교
print("\n연도별 평균 RSI:")
print(df.groupby('year')['entry_rsi'].mean())

# 연도별 MACD 비교
print("\n연도별 평균 MACD:")
print(df.groupby('year')['entry_macd'].mean())

# 2024년 vs 다른 연도 비교
print("\n2024년 vs 다른 연도 비교:")
df_2024 = df[df['year'] == 2024]
df_other = df[df['year'] != 2024]

print(f"2024년 평균 ATR: {df_2024['entry_atr'].mean():.4f}")
print(f"다른 연도 평균 ATR: {df_other['entry_atr'].mean():.4f}")
print(f"2024년 평균 RSI: {df_2024['entry_rsi'].mean():.4f}")
print(f"다른 연도 평균 RSI: {df_other['entry_rsi'].mean():.4f}")
print(f"2024년 평균 MACD: {df_2024['entry_macd'].mean():.4f}")
print(f"다른 연도 평균 MACD: {df_other['entry_macd'].mean():.4f}")
