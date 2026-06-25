# -*- coding: utf-8 -*-
"""
계약수 증가에 따른 수익성 분석
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def analyze_contract_size():
    """계약수 증가에 따른 수익성 분석"""
    print("=" * 80)
    print("계약수 증가에 따른 수익성 분석")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 현재 1계약 기준 성과
    total_pnl_1 = df['net_krw'].sum()
    total_trades = len(df)
    win_rate = df['is_win'].mean() * 100
    avg_pnl = df['net_krw'].mean()
    max_loss = df['net_krw'].min()
    max_profit = df['net_krw'].max()
    
    print(f"\n현재 1계약 기준 성과:")
    print(f"  총 PnL: {total_pnl_1:,.0f}원")
    print(f"  총 거래 수: {total_trades}건")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  평균 PnL: {avg_pnl:,.0f}원")
    print(f"  최대 손실: {max_loss:,.0f}원")
    print(f"  최대 이익: {max_profit:,.0f}원")
    
    # 계약수 증가 시 수익성 분석
    contract_sizes = [1, 2, 3, 5, 10]
    
    print(f"\n{'='*80}")
    print("계약수 증가에 따른 수익성 분석")
    print(f"{'='*80}")
    
    for size in contract_sizes:
        total_pnl = total_pnl_1 * size
        avg_pnl_size = avg_pnl * size
        max_loss_size = max_loss * size
        max_profit_size = max_profit * size
        
        print(f"\n{size}계약 기준:")
        print(f"  총 PnL: {total_pnl:,.0f}원")
        print(f"  평균 PnL: {avg_pnl_size:,.0f}원")
        print(f"  최대 손실: {max_loss_size:,.0f}원")
        print(f"  최대 이익: {max_profit_size:,.0f}원")
    
    # 리스크 관리 측면 분석
    print(f"\n{'='*80}")
    print("리스크 관리 측면 분석")
    print(f"{'='*80}")
    
    # 손실 거래 분석
    losing_trades = df[df['net_krw'] < 0]
    avg_loss = losing_trades['net_krw'].mean()
    max_drawdown = df['net_krw'].cumsum().min()
    
    print(f"\n손실 거래 분석:")
    print(f"  손실 거래 수: {len(losing_trades)}건")
    print(f"  평균 손실: {avg_loss:,.0f}원")
    print(f"  최대 손실: {max_loss:,.0f}원")
    print(f"  최대 드로우다운: {max_drawdown:,.0f}원")
    
    # 계약수 증가 시 리스크 분석
    print(f"\n계약수 증가 시 리스크 분석:")
    
    for size in contract_sizes:
        avg_loss_size = avg_loss * size
        max_loss_size = max_loss * size
        max_drawdown_size = max_drawdown * size
        
        print(f"\n{size}계약 기준:")
        print(f"  평균 손실: {avg_loss_size:,.0f}원")
        print(f"  최대 손실: {max_loss_size:,.0f}원")
        print(f"  최대 드로우다운: {max_drawdown_size:,.0f}원")
    
    # 계약수 최적화 방안
    print(f"\n{'='*80}")
    print("계약수 최적화 방안")
    print(f"{'='*80}")
    
    # 리스크 허용도 기준 계약수 계산
    risk_tolerance = 10000000  # 1,000만원 리스크 허용도
    max_safe_size = int(risk_tolerance / abs(max_loss))
    
    print(f"\n리스크 허용도 기준 계약수:")
    print(f"  리스크 허용도: {risk_tolerance:,.0f}원")
    print(f"  최대 손실 (1계약): {abs(max_loss):,.0f}원")
    print(f"  최대 안전 계약수: {max_safe_size}계약")
    
    # 자본 기준 계약수 계산
    capital = 100000000  # 1억원 자본
    risk_per_trade = 0.02  # 2% 리스크
    max_loss_per_trade = abs(max_loss)
    size_by_capital = int((capital * risk_per_trade) / max_loss_per_trade)
    
    print(f"\n자본 기준 계약수:")
    print(f"  자본: {capital:,.0f}원")
    print(f"  리스크 비율: {risk_per_trade * 100}%")
    print(f"  최대 손실 (1계약): {abs(max_loss):,.0f}원")
    print(f"  권장 계약수: {size_by_capital}계약")
    
    # 최종 권장사항
    print(f"\n{'='*80}")
    print("최종 권장사항")
    print(f"{'='*80}")
    
    recommended_size = min(max_safe_size, size_by_capital)
    recommended_pnl = total_pnl_1 * recommended_size
    
    print(f"\n권장 계약수: {recommended_size}계약")
    print(f"  예상 총 PnL: {recommended_pnl:,.0f}원")
    print(f"  예상 최대 손실: {max_loss * recommended_size:,.0f}원")
    print(f"  예상 최대 드로우다운: {max_drawdown * recommended_size:,.0f}원")
    
    return total_pnl_1, recommended_size


if __name__ == "__main__":
    total_pnl_1, recommended_size = analyze_contract_size()
