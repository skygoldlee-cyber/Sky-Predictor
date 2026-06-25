import pandas as pd

df = pd.read_csv('c:/Project/SkyPredictor/Devcenter/ml/ml_models/final_trades.csv')
df_25_26 = df[df['year'] >= 2025]

print(f'2025-2026 거래 수: {len(df_25_26)}건')
print(f'2025-2026 승률: {df_25_26["is_win"].mean() * 100:.2f}%')
print(f'2025-2026 총 PnL: {df_25_26["net_krw"].sum():,.0f} 원')
print(f'2025-2026 평균 PnL: {df_25_26["net_krw"].mean():,.0f} 원')
print(f'2025-2026 승리 거래: {df_25_26["is_win"].sum()}건')
print(f'2025-2026 패배 거래: {(df_25_26["is_win"] == 0).sum()}건')

# 초기자본금 가정 (거래당 평균 포지션 크기 기반)
# 실제 초기자본금은 사용자 정의 필요
initial_capital = 100_000_000  # 1억 원 가정
final_capital = initial_capital + df_25_26["net_krw"].sum()
return_pct = (df_25_26["net_krw"].sum() / initial_capital) * 100

print(f'\n초기자본금 (가정): {initial_capital:,.0f} 원')
print(f'최종 수익금: {df_25_26["net_krw"].sum():,.0f} 원')
print(f'최종 자본금: {final_capital:,.0f} 원')
print(f'수익률: {return_pct:.2f}%')
