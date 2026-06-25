# -*- coding: utf-8 -*-
"""
리스크 관리 강화안 적용 테스트
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def test_risk_management_strategies():
    """리스크 관리 강화안 적용 테스트"""
    print("=" * 80)
    print("리스크 관리 강화안 적용 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    # 기본 통계
    total_pnl_1 = df['net_krw'].sum()
    max_loss = df['net_krw'].min()
    max_drawdown = df['net_krw'].cumsum().min()
    losing_trades = df[df['net_krw'] < 0]
    avg_loss = losing_trades['net_krw'].mean()
    
    print(f"\n기본 통계 (1계약 기준):")
    print(f"  총 PnL: {total_pnl_1:,.0f}원")
    print(f"  최대 손실: {max_loss:,.0f}원")
    print(f"  최대 드로우다운: {max_drawdown:,.0f}원")
    print(f"  평균 손실: {avg_loss:,.0f}원")
    
    # 계약수별 리스크 분석
    contract_sizes = [1, 2, 3, 5, 10]
    
    print(f"\n{'='*80}")
    print("계약수별 리스크 분석")
    print(f"{'='*80}")
    
    for size in contract_sizes:
        total_pnl = total_pnl_1 * size
        max_loss_size = max_loss * size
        max_drawdown_size = max_drawdown * size
        avg_loss_size = avg_loss * size
        
        print(f"\n{size}계약 기준:")
        print(f"  총 PnL: {total_pnl:,.0f}원")
        print(f"  최대 손실: {max_loss_size:,.0f}원")
        print(f"  최대 드로우다운: {max_drawdown_size:,.0f}원")
        print(f"  평균 손실: {avg_loss_size:,.0f}원")
    
    # 손절매 기준 설정
    print(f"\n{'='*80}")
    print("계약수별 손절매 기준 설정")
    print(f"{'='*80}")
    
    for size in contract_sizes:
        stop_loss_pct = 0.02 / size
        stop_loss_amount = abs(max_loss) * stop_loss_pct / 0.02
        
        # 손절매 기준 미만 손실 거래 수
        df_size = df.copy()
        df_size['net_krw_size'] = df_size['net_krw'] * size
        stopped_count = (df_size['net_krw_size'] < -stop_loss_amount).sum()
        
        print(f"\n{size}계약 기준:")
        print(f"  손절매 비율: {stop_loss_pct * 100:.2f}%")
        print(f"  손절매 금액: {stop_loss_amount:,.0f}원")
        print(f"  손절매 대상 거래 수: {stopped_count}건")
    
    # 최대 드로우다운 한계 설정
    print(f"\n{'='*80}")
    print("계약수별 최대 드로우다운 한계 설정")
    print(f"{'='*80}")
    
    max_drawdown_limit = 0.10  # 10% 최대 드로우다운 한계
    capital = 100000000  # 1억원 자본
    
    for size in contract_sizes:
        max_dd_limit = max_drawdown_limit * capital
        max_dd_per_contract = max_dd_limit / size
        
        # 최대 드로우다운 한계 초과 여부
        df_size = df.copy()
        df_size['net_krw_size'] = df_size['net_krw'] * size
        df_size['cumsum'] = df_size['net_krw_size'].cumsum()
        dd_exceeded = (df_size['cumsum'] < -max_dd_per_contract).any()
        
        print(f"\n{size}계약 기준:")
        print(f"  최대 드로우다운 한계: {max_dd_limit:,.0f}원")
        print(f"  계약당 한계: {max_dd_per_contract:,.0f}원")
        print(f"  최대 드로우다운: {max_drawdown * size:,.0f}원")
        print(f"  한계 초과 여부: {'예' if dd_exceeded else '아니오'}")
    
    # 리스크 관리 권장사항
    print(f"\n{'='*80}")
    print("리스크 관리 권장사항")
    print(f"{'='*80}")
    
    # 자본 기준 계약수
    risk_per_trade = 0.02  # 2% 리스크
    size_by_capital = int((capital * risk_per_trade) / abs(max_loss))
    
    print(f"\n자본 기준 계약수:")
    print(f"  자본: {capital:,.0f}원")
    print(f"  리스크 비율: {risk_per_trade * 100}%")
    print(f"  최대 손실 (1계약): {abs(max_loss):,.0f}원")
    print(f"  권장 계약수: {size_by_capital}계약")
    
    # 리스크 허용도 기준 계약수
    risk_tolerance = 10000000  # 1,000만원 리스크 허용도
    max_safe_size = int(risk_tolerance / abs(max_loss))
    
    print(f"\n리스크 허용도 기준 계약수:")
    print(f"  리스크 허용도: {risk_tolerance:,.0f}원")
    print(f"  최대 손실 (1계약): {abs(max_loss):,.0f}원")
    print(f"  최대 안전 계약수: {max_safe_size}계약")
    
    # 최종 권장사항
    recommended_size = min(size_by_capital, max_safe_size)
    recommended_pnl = total_pnl_1 * recommended_size
    recommended_max_loss = max_loss * recommended_size
    recommended_max_dd = max_drawdown * recommended_size
    
    print(f"\n최종 권장사항:")
    print(f"  권장 계약수: {recommended_size}계약")
    print(f"  예상 총 PnL: {recommended_pnl:,.0f}원")
    print(f"  예상 최대 손실: {recommended_max_loss:,.0f}원")
    print(f"  예상 최대 드로우다운: {recommended_max_dd:,.0f}원")
    
    return total_pnl_1, recommended_size


if __name__ == "__main__":
    total_pnl_1, recommended_size = test_risk_management_strategies()
