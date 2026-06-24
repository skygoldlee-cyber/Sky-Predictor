# -*- coding: utf-8 -*-
"""
포지션 사이징 최적화 (Kelly Criterion)

Kelly Criterion을 사용하여 포지션 사이징을 최적화한다.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_final_data() -> pd.DataFrame:
    """최종 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "final_trades.csv")
    
    print(f"최종 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw'].sum():,.0f} 원")
    
    return df


def calculate_kelly_criterion(df: pd.DataFrame) -> float:
    """Kelly Criterion 계산"""
    # 승률
    win_rate = df['is_win'].mean()
    
    # 승리 시 평균 수익 (포인트)
    winning_trades = df[df['is_win'] == 1]
    avg_win = winning_trades['net_pts'].mean()
    
    # 패배 시 평균 손실 (포인트)
    losing_trades = df[df['is_win'] == 0]
    avg_loss = abs(losing_trades['net_pts'].mean())
    
    # Kelly Criterion: f = (bp - q) / b
    # b = avg_win / avg_loss (승리 시 평균 수익 / 패배 시 평균 손실)
    # p = win_rate (승률)
    # q = 1 - p (패배 확률)
    
    if avg_loss == 0:
        return 0.0
    
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Kelly 비율이 음수면 0으로 설정
    kelly_fraction = max(0, kelly_fraction)
    
    # Kelly 비율이 1을 초과하면 1로 설정 (과도한 레버리지 방지)
    kelly_fraction = min(1, kelly_fraction)
    
    print(f"\nKelly Criterion 계산:")
    print(f"  승률 (p): {p:.4f}")
    print(f"  패배 확률 (q): {q:.4f}")
    print(f"  평균 승리 (포인트): {avg_win:.2f}")
    print(f"  평균 패배 (포인트): {avg_loss:.2f}")
    print(f"  승/패 비율 (b): {b:.4f}")
    print(f"  Kelly 비율: {kelly_fraction:.4f}")
    
    return kelly_fraction


def apply_position_sizing(df: pd.DataFrame, kelly_fraction: float, 
                          multiplier: float = 31_500) -> pd.DataFrame:
    """포지션 사이징 적용"""
    df_sized = df.copy()
    
    # 기존 Half Kelly multiplier (31,500)
    base_multiplier = 31_500
    
    # Kelly 비율에 따른 multiplier 조정
    # Full Kelly: kelly_fraction * base_multiplier
    # Half Kelly: 0.5 * kelly_fraction * base_multiplier
    # Quarter Kelly: 0.25 * kelly_fraction * base_multiplier
    
    new_multiplier = kelly_fraction * base_multiplier
    
    # 새로운 PnL 계산
    # 기존 PnL = net_pts * base_multiplier
    # 새로운 PnL = net_pts * new_multiplier
    
    df_sized['net_krw_sized'] = df_sized['net_pts'] * new_multiplier
    
    return df_sized


def compare_position_sizing(df: pd.DataFrame, kelly_fraction: float):
    """다양한 Kelly 비율 테스트"""
    print(f"\n{'='*100}")
    print("다양한 Kelly 비율 테스트")
    print(f"{'='*100}")
    
    # 기존 Half Kelly (고정 multiplier 31,500)
    base_multiplier = 31_500
    df['net_krw_base'] = df['net_pts'] * base_multiplier
    
    # 다양한 Kelly 비율 테스트
    kelly_ratios = [
        ('Full Kelly', 1.0),
        ('Half Kelly', 0.5),
        ('Quarter Kelly', 0.25),
        ('Eighth Kelly', 0.125),
        ('Fixed (Current)', 0.0)  # 현재 고정 multiplier
    ]
    
    results = []
    
    for name, ratio in kelly_ratios:
        if ratio == 0.0:
            # 현재 고정 multiplier 사용
            multiplier = base_multiplier
            df_test = df.copy()
            df_test['net_krw_test'] = df_test['net_pts'] * multiplier
        else:
            # Kelly 비율 적용
            multiplier = kelly_fraction * base_multiplier * ratio
            df_test = df.copy()
            df_test['net_krw_test'] = df_test['net_pts'] * multiplier
        
        total_pnl = df_test['net_krw_test'].sum()
        avg_pnl = df_test['net_krw_test'].mean()
        win_rate = df_test['is_win'].mean() * 100
        
        results.append({
            'name': name,
            'multiplier': multiplier,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate
        })
    
    # 결과 출력
    print(f"\n{'전략':<20}{'Multiplier':>15}{'총 PnL (원)':>20}{'평균 PnL (원)':>20}{'승률 (%)':>15}")
    print(f"{'-'*90}")
    
    for result in results:
        print(f"{result['name']:<20}{result['multiplier']:>15,.0f}"
              f"{result['total_pnl']:>20,.0f}{result['avg_pnl']:>20,.0f}{result['win_rate']:>15.2f}")
    
    return results


def optimize_position_sizing(df: pd.DataFrame, kelly_fraction: float):
    """최적 포지션 사이징 선택"""
    print(f"\n{'='*100}")
    print("최적 포지션 사이징 선택")
    print(f"{'='*100}")
    
    # 기존 Half Kelly (고정 multiplier 31,500)
    base_multiplier = 31_500
    df['net_krw_base'] = df['net_pts'] * base_multiplier
    
    # Half Kelly 사용 (안정성 고려)
    optimal_ratio = 0.5
    optimal_multiplier = kelly_fraction * base_multiplier * optimal_ratio
    
    df_optimal = df.copy()
    df_optimal['net_krw_optimal'] = df_optimal['net_pts'] * optimal_multiplier
    
    print(f"\n선택된 전략: Half Kelly ({optimal_ratio} * Kelly)")
    print(f"Kelly 비율: {kelly_fraction:.4f}")
    print(f"적용된 multiplier: {optimal_multiplier:,.0f}")
    print(f"기존 multiplier: {base_multiplier:,.0f}")
    
    # 성과 비교
    print(f"\n{'='*100}")
    print("포지션 사이징 최적화 전후 성과 비교")
    print(f"{'='*100}")
    
    n_trades = len(df)
    win_rate = df['is_win'].mean() * 100
    
    total_pnl_base = df['net_krw_base'].sum()
    avg_pnl_base = df['net_krw_base'].mean()
    
    total_pnl_optimal = df_optimal['net_krw_optimal'].sum()
    avg_pnl_optimal = df_optimal['net_krw_optimal'].mean()
    
    print(f"\n{'지표':<20}{'기존 Half Kelly':>25}{'최적 Half Kelly':>25}{'변화':>20}")
    print(f"{'-'*90}")
    print(f"{'거래 수':<20}{n_trades:>25}{n_trades:>25}{0:>20}")
    print(f"{'승률 (%)':<20}{win_rate:>25.2f}{win_rate:>25.2f}{0:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_base:>25,.0f}{total_pnl_optimal:>25,.0f}"
          f"{total_pnl_optimal - total_pnl_base:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_base:>25,.0f}{avg_pnl_optimal:>25,.0f}"
          f"{avg_pnl_optimal - avg_pnl_base:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'기존 총 PnL':>20}{'최적 총 PnL':>20}{'변화':>20}")
    print(f"{'-'*70}")
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        base_pnl_year = (df_year['net_pts'] * base_multiplier).sum()
        opt_pnl_year = (df_year['net_pts'] * optimal_multiplier).sum()
        
        print(f"{year:<10}{base_pnl_year:>20,.0f}{opt_pnl_year:>20,.0f}"
              f"{opt_pnl_year - base_pnl_year:>20,.0f}")
    
    return df_optimal, optimal_multiplier


def main():
    """메인 함수"""
    print("=" * 100)
    print("포지션 사이징 최적화 (Kelly Criterion)")
    print("=" * 100)
    
    # 1) 최종 데이터 로드
    df = load_final_data()
    
    # 2) Kelly Criterion 계산
    kelly_fraction = calculate_kelly_criterion(df)
    
    # 3) 다양한 Kelly 비율 테스트
    results = compare_position_sizing(df, kelly_fraction)
    
    # 4) 최적 포지션 사이징 선택
    df_optimal, optimal_multiplier = optimize_position_sizing(df, kelly_fraction)
    
    # 5) 최종 결과 요약
    print(f"\n{'='*100}")
    print("최종 결과 요약")
    print(f"{'='*100}")
    
    print(f"\n최종 거래 수: {len(df_optimal)}건")
    print(f"최종 승률: {df_optimal['is_win'].mean() * 100:.2f}%")
    print(f"최적 multiplier: {optimal_multiplier:,.0f}")
    print(f"최종 총 PnL: {df_optimal['net_krw_optimal'].sum():,.0f} 원")
    print(f"최종 평균 PnL: {df_optimal['net_krw_optimal'].mean():,.0f} 원")
    
    # 최종 데이터셋 저장
    final_path = OUTPUT_DIR / "final_trades_sized.csv"
    df_optimal.to_csv(final_path, index=False)
    print(f"\n최종 데이터셋 저장 완료: {final_path}")


if __name__ == "__main__":
    main()
