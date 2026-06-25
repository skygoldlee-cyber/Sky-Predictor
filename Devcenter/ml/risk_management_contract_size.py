# -*- coding: utf-8 -*-
"""
계약수 증가 시 리스크 관리 강화안
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def risk_management_strategies():
    """계약수 증가 시 리스크 관리 강화안"""
    print("=" * 80)
    print("계약수 증가 시 리스크 관리 강화안")
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
    
    # 리스크 관리 강화안
    print(f"\n{'='*80}")
    print("리스크 관리 강화안")
    print(f"{'='*80}")
    
    # 1. 포지션 사이징 관리
    print(f"\n1. 포지션 사이징 관리")
    print(f"   - 고정 계약수 대신 동적 계약수 사용")
    print(f"   - 자본 비율 기반 계약수 결정")
    print(f"   - 변동성 기반 계약수 조정")
    
    # 자본 비율 기반 계약수 계산
    capital = 100000000  # 1억원
    risk_per_trade = 0.02  # 2% 리스크
    max_loss_per_trade = abs(max_loss)
    size_by_capital = int((capital * risk_per_trade) / max_loss_per_trade)
    
    print(f"\n   자본 비율 기반 계약수:")
    print(f"   - 자본: {capital:,.0f}원")
    print(f"   - 리스크 비율: {risk_per_trade * 100}%")
    print(f"   - 최대 손실 (1계약): {abs(max_loss):,.0f}원")
    print(f"   - 권장 계약수: {size_by_capital}계약")
    
    # 2. 손절매 기준 강화
    print(f"\n2. 손절매 기준 강화")
    print(f"   - 계약수 증가에 따른 손절매 기준 조정")
    print(f"   - 고정 손실 대신 비율 기반 손절매")
    print(f"   - 변동성 기반 동적 손절매")
    
    # 계약수별 손절매 기준
    contract_sizes = [1, 2, 3, 5, 10]
    print(f"\n   계약수별 손절매 기준:")
    for size in contract_sizes:
        stop_loss_pct = 0.02 / size  # 계약수 증가 시 손절매 비율 감소
        stop_loss_amount = abs(max_loss) * stop_loss_pct / 0.02
        print(f"   - {size}계약: 손절매 비율 {stop_loss_pct * 100:.2f}%, 손절매 금액 {stop_loss_amount:,.0f}원")
    
    # 3. 최대 드로우다운 관리
    print(f"\n3. 최대 드로우다운 관리")
    print(f"   - 일일 최대 손실 한계 설정")
    print(f"   - 주간 최대 손실 한계 설정")
    print(f"   - 월간 최대 손실 한계 설정")
    
    # 계약수별 최대 드로우다운 한계
    max_drawdown_limit = 0.10  # 10% 최대 드로우다운 한계
    print(f"\n   계약수별 최대 드로우다운 한계:")
    for size in contract_sizes:
        max_dd_limit = max_drawdown_limit * capital
        max_dd_per_contract = max_dd_limit / size
        print(f"   - {size}계약: 최대 드로우다운 한계 {max_dd_limit:,.0f}원 (계약당 {max_dd_per_contract:,.0f}원)")
    
    # 4. 포트폴리오 분산
    print(f"\n4. 포트폴리오 분산")
    print(f"   - 단일 전략 대신 다중 전략 사용")
    print(f"   - 상관관계가 낮은 전략 조합")
    print(f"   - 시간대별 거래 분산")
    
    # 5. 리스크 헷징
    print(f"\n5. 리스크 헷징")
    print(f"   - 옵션 헷징 전략")
    print(f"   - 역 포지션 헷징")
    print(f"   - 시간차 헷징")
    
    # 6. 모니터링 및 알림
    print(f"\n6. 모니터링 및 알림")
    print(f"   - 실시간 리스크 모니터링")
    print(f"   - 자동 알림 시스템")
    print(f"   - 긴급 정지 기능")
    
    # 7. 백테스팅 및 시뮬레이션
    print(f"\n7. 백테스팅 및 시뮬레이션")
    print(f"   - 다양한 시나리오 테스트")
    print(f"   - 스트레스 테스트")
    print(f"   - 몬테카를로 시뮬레이션")
    
    # 8. 교육 및 훈련
    print(f"\n8. 교육 및 훈련")
    print(f"   - 리스크 관리 교육")
    print(f"   - 시뮬레이션 훈련")
    print(f"   - 정기적 리뷰")
    
    # 최종 권장사항
    print(f"\n{'='*80}")
    print("최종 권장사항")
    print(f"{'='*80}")
    
    print(f"\n1. 기본 설정:")
    print(f"   - 계약수: {size_by_capital}계약")
    print(f"   - 손절매 비율: {risk_per_trade * 100:.2f}%")
    print(f"   - 최대 드로우다운 한계: {max_drawdown_limit * 100:.2f}%")
    
    print(f"\n2. 계약수 증가 시:")
    print(f"   - 단계적 계약수 증가 (1계약 → 2계약 → 3계약)")
    print(f"   - 각 단계별 성과 확인 후 다음 단계 진행")
    print(f"   - 리스크 허용도에 따른 계약수 조정")
    
    print(f"\n3. 긴급 상황 시:")
    print(f"   - 즉시 거래 중단")
    print(f"   - 모든 포지션 청산")
    print(f"   - 원인 분석 및 개선")
    
    return size_by_capital, max_drawdown_limit


if __name__ == "__main__":
    size_by_capital, max_drawdown_limit = risk_management_strategies()
