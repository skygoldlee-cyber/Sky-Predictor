# -*- coding: utf-8 -*-
"""
포지션 사이징 최적화 (승/패 비율 개선 후 Kelly Criterion)

승/패 비율 개선 후 Kelly Criterion을 재계산하고 다양한 Kelly 비율을 테스트한다.
"""
import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)

# 기본 multiplier
BASE_MULTIPLIER = 31_500


def load_optimized_data() -> pd.DataFrame:
    """최적화된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "exit_ratio_optimized_trades.csv")
    
    print(f"최적화된 데이터 로드 완료: {len(df)}건")
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
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Kelly 비율이 음수면 0으로 설정
    kelly_fraction = max(0, kelly_fraction)
    
    # Kelly 비율이 1을 초과하면 1로 설정
    kelly_fraction = min(1, kelly_fraction)
    
    print(f"\nKelly Criterion 계산:")
    print(f"  승률 (p): {p:.4f}")
    print(f"  패배 확률 (q): {q:.4f}")
    print(f"  승리 시 평균 수익 (포인트): {avg_win:.4f}")
    print(f"  패배 시 평균 손실 (포인트): {avg_loss:.4f}")
    print(f"  승/패 비율 (b): {b:.4f}")
    print(f"  Kelly 비율 (f): {kelly_fraction:.4f}")
    
    return kelly_fraction, b, avg_win, avg_loss


def test_kelly_ratios(df: pd.DataFrame, kelly_fraction: float, b: float):
    """다양한 Kelly 비율 테스트"""
    print(f"\n{'='*100}")
    print("다양한 Kelly 비율 테스트")
    print(f"{'='*100}")
    
    # Kelly 비율에 따른 multiplier 계산
    # Kelly multiplier = Kelly 비율 * 기본 multiplier / (승/패 비율 조정)
    # 승/패 비율이 1보다 크면 더 큰 multiplier 가능
    
    kelly_multipliers = [
        ("Full Kelly", kelly_fraction),
        ("Half Kelly", kelly_fraction * 0.5),
        ("Quarter Kelly", kelly_fraction * 0.25),
        ("Eighth Kelly", kelly_fraction * 0.125),
        ("Fixed (Current)", 1.0),  # 기존 Fixed multiplier 유지
    ]
    
    results = []
    
    for name, fraction in kelly_multipliers:
        # multiplier 계산
        if name == "Fixed (Current)":
            multiplier = BASE_MULTIPLIER
        else:
            # Kelly multiplier = Kelly 비율 * 기본 multiplier
            multiplier = fraction * BASE_MULTIPLIER
        
        # PnL 재계산
        df_copy = df.copy()
        df_copy['net_krw_optimal'] = df_copy['net_pts'] * multiplier
        
        total_pnl = df_copy['net_krw_optimal'].sum()
        avg_pnl = df_copy['net_krw_optimal'].mean()
        win_rate = df_copy['is_win'].mean() * 100
        
        results.append({
            'name': name,
            'fraction': fraction,
            'multiplier': multiplier,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate
        })
    
    # 결과 출력
    print(f"\n{'전략':<20}{'Kelly 비율':<15}{'Multiplier':<15}{'총 PnL (원)':<20}{'평균 PnL (원)':<20}{'승률':<15}")
    print(f"{'-'*100}")
    
    for result in results:
        print(f"{result['name']:<20}{result['fraction']:<15.4f}{result['multiplier']:<15,.0f}"
              f"{result['total_pnl']:<20,.0f}{result['avg_pnl']:<20,.0f}{result['win_rate']:<15.2f}")
    
    return results


def select_optimal_position_sizing(df: pd.DataFrame, results: List[Dict]):
    """최적 포지션 사이징 선택"""
    print(f"\n{'='*100}")
    print("최적 포지션 사이징 선택")
    print(f"{'='*100}")
    
    # 총 PnL이 가장 높은 전략 선택
    optimal = max(results, key=lambda x: x['total_pnl'])
    
    print(f"\n선택된 포지션 사이징:")
    print(f"  전략: {optimal['name']}")
    print(f"  Kelly 비율: {optimal['fraction']:.4f}")
    print(f"  Multiplier: {optimal['multiplier']:,.0f}")
    print(f"  총 PnL: {optimal['total_pnl']:,.0f} 원")
    
    return optimal


def apply_position_sizing(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """포지션 사이징 적용"""
    df_sized = df.copy()
    df_sized['net_krw_optimal'] = df_sized['net_pts'] * multiplier
    
    print(f"\n포지션 사이징 적용 완료:")
    print(f"  Multiplier: {multiplier:,.0f}")
    print(f"  총 PnL: {df_sized['net_krw_optimal'].sum():,.0f} 원")
    
    return df_sized


def compare_performance(df_original: pd.DataFrame, df_sized: pd.DataFrame, 
                        original_multiplier: float, new_multiplier: float):
    """성과 비교"""
    print(f"\n{'='*100}")
    print("성과 비교")
    print(f"{'='*100}")
    
    # 원본
    n_trades_orig = len(df_original)
    win_rate_orig = df_original['is_win'].mean() * 100
    total_pnl_orig = df_original['net_krw'].sum()
    avg_pnl_orig = df_original['net_krw'].mean()
    
    # 최적화
    n_trades_sized = len(df_sized)
    win_rate_sized = df_sized['is_win'].mean() * 100
    total_pnl_sized = df_sized['net_krw_optimal'].sum()
    avg_pnl_sized = df_sized['net_krw_optimal'].mean()
    
    print(f"\n{'지표':<20}{'원본':>20}{'최적화':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_orig:>20}{n_trades_sized:>20}{n_trades_sized - n_trades_orig:>20}")
    print(f"{'승률 (%)':<20}{win_rate_orig:>20.2f}{win_rate_sized:>20.2f}{win_rate_sized - win_rate_orig:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_orig:>20,.0f}{total_pnl_sized:>20,.0f}{total_pnl_sized - total_pnl_orig:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_orig:>20,.0f}{avg_pnl_sized:>20,.0f}{avg_pnl_sized - avg_pnl_orig:>20,.0f}")
    print(f"{'Multiplier':<20}{original_multiplier:>20,.0f}{new_multiplier:>20,.0f}{new_multiplier - original_multiplier:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'원본 거래':>15}{'최적화 거래':>15}{'원본 승률':>15}{'최적화 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        sized_year = df_sized[df_sized['year'] == year]
        
        print(f"{year:<10}{len(orig_year):>15}{len(sized_year):>15}"
              f"{orig_year['is_win'].mean()*100:>15.2f}{sized_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("포지션 사이징 최적화 (승/패 비율 개선 후 Kelly Criterion)")
    print("=" * 100)
    
    # 1) 최적화된 데이터 로드
    df = load_optimized_data()
    
    # 2) Kelly Criterion 계산
    kelly_fraction, b, avg_win, avg_loss = calculate_kelly_criterion(df)
    
    # 3) 다양한 Kelly 비율 테스트
    results = test_kelly_ratios(df, kelly_fraction, b)
    
    # 4) 최적 포지션 사이징 선택
    optimal = select_optimal_position_sizing(df, results)
    
    # 5) 최적 포지션 사이징 적용
    df_sized = apply_position_sizing(df, optimal['multiplier'])
    
    # 6) 성과 비교
    compare_performance(df, df_sized, BASE_MULTIPLIER, optimal['multiplier'])
    
    # 7) 최종 결과 요약
    print(f"\n{'='*100}")
    print("최종 결과 요약")
    print(f"{'='*100}")
    
    print(f"\n선택된 포지션 사이징:")
    print(f"  전략: {optimal['name']}")
    print(f"  Kelly 비율: {optimal['fraction']:.4f}")
    print(f"  Multiplier: {optimal['multiplier']:,.0f}")
    
    print(f"\n최종 거래 수: {len(df_sized)}건")
    print(f"최종 승률: {df_sized['is_win'].mean() * 100:.2f}%")
    print(f"최종 총 PnL: {df_sized['net_krw_optimal'].sum():,.0f} 원")
    print(f"최종 평균 PnL: {df_sized['net_krw_optimal'].mean():,.0f} 원")
    
    # 승/패 비율 재계산
    winning_trades = df_sized[df_sized['is_win'] == 1]
    losing_trades = df_sized[df_sized['is_win'] == 0]
    avg_win = winning_trades['net_pts'].mean()
    avg_loss = abs(losing_trades['net_pts'].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    print(f"최종 승/패 비율: {win_loss_ratio:.4f}")
    
    # 최적화된 데이터셋 저장
    sized_path = OUTPUT_DIR / "final_trades_sized_improved.csv"
    df_sized.to_csv(sized_path, index=False)
    print(f"\n최적화된 데이터셋 저장 완료: {sized_path}")


if __name__ == "__main__":
    main()
