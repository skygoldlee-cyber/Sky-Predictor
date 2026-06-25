# -*- coding: utf-8 -*-
"""2024년 특화 모델 분석"""
import pandas as pd
import numpy as np

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')

# 2024년 데이터 분석
df_2024 = df[df['year'] == 2024]
df_train = df[df['year'] < 2024]
df_test = df[df['year'] >= 2024]

print("2024년 특화 모델 분석")
print("=" * 100)

# 1. 2024년 데이터 특성
print("\n1. 2024년 데이터 특성:")
print(f"   거래 수: {len(df_2024)}건")
print(f"   승률: {df_2024['is_win'].mean() * 100:.2f}%")
print(f"   총 PnL: {df_2024['net_krw'].sum():,.0f}원")
print(f"   평균 ATR: {df_2024['entry_atr'].mean():.4f}")
print(f"   평균 RSI: {df_2024['entry_rsi'].mean():.4f}")
print(f"   평균 MACD: {df_2024['entry_macd'].mean():.4f}")

# 2. 2024년 vs 학습 데이터 비교
print("\n2. 2024년 vs 학습 데이터 비교:")
print(f"   학습 데이터 (2019-2023): {len(df_train)}건")
print(f"   학습 데이터 승률: {df_train['is_win'].mean() * 100:.2f}%")
print(f"   학습 데이터 평균 ATR: {df_train['entry_atr'].mean():.4f}")
print(f"   학습 데이터 평균 RSI: {df_train['entry_rsi'].mean():.4f}")
print(f"   학습 데이터 평균 MACD: {df_train['entry_macd'].mean():.4f}")

# 3. 2024년 특성 분석
print("\n3. 2024년 특성 분석:")
print(f"   2024년 ATR 차이: {df_2024['entry_atr'].mean() - df_train['entry_atr'].mean():.4f}")
print(f"   2024년 RSI 차이: {df_2024['entry_rsi'].mean() - df_train['entry_rsi'].mean():.4f}")
print(f"   2024년 MACD 차이: {df_2024['entry_macd'].mean() - df_train['entry_macd'].mean():.4f}")

# 4. 2024년 시간대별 성과
print("\n4. 2024년 시간대별 성과:")
hourly_2024 = df_2024.groupby('entry_hour').agg({
    'is_win': ['mean', 'count'],
    'net_krw': 'sum'
}).round(2)
print(hourly_2024)

# 5. 2024년 월별 성과
print("\n5. 2024년 월별 성과:")
df_2024['month'] = pd.to_datetime(df_2024['entry_time']).dt.month
monthly_2024 = df_2024.groupby('month').agg({
    'is_win': ['mean', 'count'],
    'net_krw': 'sum'
}).round(2)
print(monthly_2024)

# 6. 2024년 특화 모델 제안
print("\n6. 2024년 특화 모델 제안:")
print("   a) 2024년 데이터만 사용하여 모델 재학습")
print("   b) 2024년 특성을 고려한 feature engineering")
print("   c) 2024년 시간대별 특화 모델")
print("   d) 2024년 월별 특화 모델")
print("   e) 2024년 regime 특화 모델")
