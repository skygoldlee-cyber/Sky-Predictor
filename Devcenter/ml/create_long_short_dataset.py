"""
기존 long-only 데이터셋을 long+short 데이터셋으로 변환
"""

import pandas as pd
import numpy as np

# 기존 데이터셋 로드
df = pd.read_csv('Devcenter/ml/ml_data/ml_dataset.csv')

# long 데이터 복사
df_long = df.copy()
df_long['direction'] = 1

# short 데이터 생성 (PnL 반전)
df_short = df.copy()
df_short['direction'] = -1
df_short['net_krw'] = -df_short['net_krw']
df_short['net_pts'] = -df_short['net_pts']
df_short['is_win'] = (df_short['net_krw'] > 0).astype(int)

# 데이터 합치기
df_combined = pd.concat([df_long, df_short], ignore_index=True)

# 섞기
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# 저장
output_path = 'Devcenter/ml/ml_data/ml_dataset_long_short.csv'
df_combined.to_csv(output_path, index=False)

print(f"Long+Short 데이터셋 저장 완료: {output_path}")
print(f"총 거래 수: {len(df_combined)}")
print(f"Long 거래: {len(df_long)}")
print(f"Short 거래: {len(df_short)}")
print(f"Direction 분포:")
print(df_combined['direction'].value_counts())
print(f"승률: {df_combined['is_win'].mean() * 100:.2f}%")
