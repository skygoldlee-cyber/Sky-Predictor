# -*- coding: utf-8 -*-
"""
리스크 관리 강화

최대 손실 제한과 연속 손실 제한을 적용하여 리스크를 관리한다.
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


def load_optimized_data() -> pd.DataFrame:
    """최적화된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "final_trades_sized_improved.csv")
    
    print(f"최적화된 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw_optimal'].sum():,.0f} 원")
    
    return df


def calculate_win_loss_ratio(df: pd.DataFrame):
    """승/패 비율 계산"""
    winning_trades = df[df['is_win'] == 1]
    losing_trades = df[df['is_win'] == 0]
    avg_win = winning_trades['net_pts'].mean()
    avg_loss = abs(losing_trades['net_pts'].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    print(f"\n승/패 비율 분석:")
    print(f"  승리 시 평균 수익 (포인트): {avg_win:.2f}")
    print(f"  패배 시 평균 손실 (포인트): {avg_loss:.2f}")
    print(f"  승/패 비율: {win_loss_ratio:.4f}")
    
    return avg_win, avg_loss, win_loss_ratio


def apply_max_loss_limit(df: pd.DataFrame, max_loss_pts: float = 2.0) -> pd.DataFrame:
    """최대 손실 제한 적용"""
    df_filtered = df.copy()
    
    # 최대 손실 제한 적용
    df_filtered = df_filtered[df_filtered['net_pts'] >= -max_loss_pts]
    
    # 승/패 재정의
    df_filtered['is_win'] = (df_filtered['net_pts'] > 0).astype(int)
    
    print(f"\n최대 손실 제한 적용 결과:")
    print(f"  최대 손실 제한: {max_loss_pts} 포인트")
    print(f"  적용 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  적용 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    return df_filtered


def apply_consecutive_loss_limit(df: pd.DataFrame, max_consecutive_losses: int = 3) -> pd.DataFrame:
    """연속 손실 제한 적용"""
    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    
    # 연속 손실 계산
    consecutive_losses = 0
    keep_trades = []
    
    for idx, row in df_sorted.iterrows():
        if row['is_win'] == 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        
        if consecutive_losses <= max_consecutive_losses:
            keep_trades.append(idx)
    
    df_filtered = df_sorted.loc[keep_trades].copy()
    
    print(f"\n연속 손실 제한 적용 결과:")
    print(f"  최대 연속 손실: {max_consecutive_losses}회")
    print(f"  적용 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  적용 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    return df_filtered


def test_multiple_risk_limits(df: pd.DataFrame):
    """다양한 리스크 제한 테스트"""
    print(f"\n{'='*100}")
    print("다양한 리스크 제한 테스트")
    print(f"{'='*100}")
    
    # 최대 손실 제한 테스트
    max_loss_limits = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print(f"\n최대 손실 제한 테스트:")
    print(f"{'최대 손실':<15}{'거래 수':<15}{'승률':<15}{'총 PnL (원)':<20}{'승/패 비율':<15}")
    print(f"{'-'*100}")
    
    max_loss_results = []
    for max_loss in max_loss_limits:
        df_filtered = apply_max_loss_limit(df, max_loss)
        
        # 승/패 비율 계산
        winning_trades = df_filtered[df_filtered['is_win'] == 1]
        losing_trades = df_filtered[df_filtered['is_win'] == 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades['net_pts'].mean()
            avg_loss = abs(losing_trades['net_pts'].mean())
            win_loss_ratio = avg_win / avg_loss
        else:
            avg_win = 0
            avg_loss = 0
            win_loss_ratio = 0
        
        total_pnl = df_filtered['net_krw_optimal'].sum()
        win_rate = df_filtered['is_win'].mean() * 100
        
        max_loss_results.append({
            'max_loss': max_loss,
            'n_trades': len(df_filtered),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'win_loss_ratio': win_loss_ratio
        })
        
        print(f"{max_loss:<15}{len(df_filtered):<15}{win_rate:<15.2f}{total_pnl:<20,.0f}{win_loss_ratio:<15.4f}")
    
    # 연속 손실 제한 테스트
    consecutive_loss_limits = [1, 2, 3, 4, 5]
    
    print(f"\n연속 손실 제한 테스트:")
    print(f"{'최대 연속 손실':<20}{'거래 수':<15}{'승률':<15}{'총 PnL (원)':<20}{'승/패 비율':<15}")
    print(f"{'-'*100}")
    
    consecutive_loss_results = []
    for max_consecutive in consecutive_loss_limits:
        df_filtered = apply_consecutive_loss_limit(df, max_consecutive)
        
        # 승/패 비율 계산
        winning_trades = df_filtered[df_filtered['is_win'] == 1]
        losing_trades = df_filtered[df_filtered['is_win'] == 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades['net_pts'].mean()
            avg_loss = abs(losing_trades['net_pts'].mean())
            win_loss_ratio = avg_win / avg_loss
        else:
            avg_win = 0
            avg_loss = 0
            win_loss_ratio = 0
        
        total_pnl = df_filtered['net_krw_optimal'].sum()
        win_rate = df_filtered['is_win'].mean() * 100
        
        consecutive_loss_results.append({
            'max_consecutive': max_consecutive,
            'n_trades': len(df_filtered),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'win_loss_ratio': win_loss_ratio
        })
        
        print(f"{max_consecutive:<20}{len(df_filtered):<15}{win_rate:<15.2f}{total_pnl:<20,.0f}{win_loss_ratio:<15.4f}")
    
    return max_loss_results, consecutive_loss_results


def select_optimal_risk_limits(df: pd.DataFrame, max_loss_results: List[Dict], 
                                consecutive_loss_results: List[Dict]):
    """최적 리스크 제한 선택"""
    print(f"\n{'='*100}")
    print("최적 리스크 제한 선택")
    print(f"{'='*100}")
    
    # 최대 손실 제한 선택 (총 PnL이 가장 높은)
    optimal_max_loss = max(max_loss_results, key=lambda x: x['total_pnl'])
    
    # 연속 손실 제한 선택 (총 PnL이 가장 높은)
    optimal_consecutive = max(consecutive_loss_results, key=lambda x: x['total_pnl'])
    
    print(f"\n선택된 최대 손실 제한:")
    print(f"  최대 손실: {optimal_max_loss['max_loss']} 포인트")
    print(f"  총 PnL: {optimal_max_loss['total_pnl']:,.0f} 원")
    
    print(f"\n선택된 연속 손실 제한:")
    print(f"  최대 연속 손실: {optimal_consecutive['max_consecutive']}회")
    print(f"  총 PnL: {optimal_consecutive['total_pnl']:,.0f} 원")
    
    return optimal_max_loss, optimal_consecutive


def apply_combined_risk_limits(df: pd.DataFrame, max_loss_pts: float, 
                              max_consecutive_losses: int) -> pd.DataFrame:
    """결합 리스크 제한 적용"""
    # 최대 손실 제한 적용
    df_filtered = apply_max_loss_limit(df, max_loss_pts)
    
    # 연속 손실 제한 적용
    df_filtered = apply_consecutive_loss_limit(df_filtered, max_consecutive_losses)
    
    return df_filtered


def compare_performance(df_original: pd.DataFrame, df_risk_managed: pd.DataFrame,
                        max_loss_pts: float, max_consecutive_losses: int):
    """성과 비교"""
    print(f"\n{'='*100}")
    print("성과 비교")
    print(f"{'='*100}")
    
    # 원본
    n_trades_orig = len(df_original)
    win_rate_orig = df_original['is_win'].mean() * 100
    total_pnl_orig = df_original['net_krw_optimal'].sum()
    avg_pnl_orig = df_original['net_krw_optimal'].mean()
    
    # 리스크 관리
    n_trades_risk = len(df_risk_managed)
    win_rate_risk = df_risk_managed['is_win'].mean() * 100
    total_pnl_risk = df_risk_managed['net_krw_optimal'].sum()
    avg_pnl_risk = df_risk_managed['net_krw_optimal'].mean()
    
    print(f"\n{'지표':<20}{'원본':>20}{'리스크 관리':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_orig:>20}{n_trades_risk:>20}{n_trades_risk - n_trades_orig:>20}")
    print(f"{'승률 (%)':<20}{win_rate_orig:>20.2f}{win_rate_risk:>20.2f}{win_rate_risk - win_rate_orig:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_orig:>20,.0f}{total_pnl_risk:>20,.0f}{total_pnl_risk - total_pnl_orig:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_orig:>20,.0f}{avg_pnl_risk:>20,.0f}{avg_pnl_risk - avg_pnl_orig:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'원본 거래':>15}{'리스크 관리 거래':>15}{'원본 승률':>15}{'리스크 관리 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        risk_year = df_risk_managed[df_risk_managed['year'] == year]
        
        print(f"{year:<10}{len(orig_year):>15}{len(risk_year):>15}"
              f"{orig_year['is_win'].mean()*100:>15.2f}{risk_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("리스크 관리 강화")
    print("=" * 100)
    
    # 1) 최적화된 데이터 로드
    df = load_optimized_data()
    
    # 2) 승/패 비율 계산
    avg_win, avg_loss, win_loss_ratio = calculate_win_loss_ratio(df)
    
    # 3) 다양한 리스크 제한 테스트
    max_loss_results, consecutive_loss_results = test_multiple_risk_limits(df)
    
    # 4) 최적 리스크 제한 선택
    optimal_max_loss, optimal_consecutive = select_optimal_risk_limits(
        df, max_loss_results, consecutive_loss_results
    )
    
    # 5) 결합 리스크 제한 적용
    df_risk_managed = apply_combined_risk_limits(
        df, optimal_max_loss['max_loss'], optimal_consecutive['max_consecutive']
    )
    
    # 6) 성과 비교
    compare_performance(df, df_risk_managed, optimal_max_loss['max_loss'], 
                        optimal_consecutive['max_consecutive'])
    
    # 7) 최종 결과 요약
    print(f"\n{'='*100}")
    print("최종 결과 요약")
    print(f"{'='*100}")
    
    print(f"\n선택된 리스크 제한:")
    print(f"  최대 손실: {optimal_max_loss['max_loss']} 포인트")
    print(f"  최대 연속 손실: {optimal_consecutive['max_consecutive']}회")
    
    print(f"\n최종 거래 수: {len(df_risk_managed)}건")
    print(f"최종 승률: {df_risk_managed['is_win'].mean() * 100:.2f}%")
    print(f"최종 총 PnL: {df_risk_managed['net_krw_optimal'].sum():,.0f} 원")
    print(f"최종 평균 PnL: {df_risk_managed['net_krw_optimal'].mean():,.0f} 원")
    
    # 승/패 비율 재계산
    winning_trades = df_risk_managed[df_risk_managed['is_win'] == 1]
    losing_trades = df_risk_managed[df_risk_managed['is_win'] == 0]
    avg_win = winning_trades['net_pts'].mean()
    avg_loss = abs(losing_trades['net_pts'].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    print(f"최종 승/패 비율: {win_loss_ratio:.4f}")
    
    # 최적화된 데이터셋 저장
    risk_managed_path = OUTPUT_DIR / "final_trades_risk_managed.csv"
    df_risk_managed.to_csv(risk_managed_path, index=False)
    print(f"\n최적화된 데이터셋 저장 완료: {risk_managed_path}")


if __name__ == "__main__":
    main()
