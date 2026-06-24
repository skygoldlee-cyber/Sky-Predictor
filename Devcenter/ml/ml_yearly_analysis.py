# -*- coding: utf-8 -*-
"""
연도별 수익 분석

ML 최적화 전후의 연도별 수익을 분석한다.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"


def load_original_data() -> pd.DataFrame:
    """원본 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    return df


def load_final_data() -> pd.DataFrame:
    """최종 데이터 로드"""
    df = pd.read_csv(OUTPUT_DIR / "final_trades_sized.csv")
    return df


def analyze_yearly_performance(df_original: pd.DataFrame, df_final: pd.DataFrame):
    """연도별 성과 분석"""
    print("=" * 100)
    print("연도별 수익 분석")
    print("=" * 100)
    
    # 원본 데이터 연도별 분석
    print(f"\n{'='*100}")
    print("원본 데이터 연도별 성과")
    print(f"{'='*100}")
    
    orig_yearly = df_original.groupby('year').agg({
        'is_win': ['count', 'mean'],
        'net_krw': ['sum', 'mean'],
        'net_pts': ['mean']
    }).round(2)
    
    orig_yearly.columns = ['거래 수', '승률', '총 PnL (원)', '평균 PnL (원)', '평균 PnL (포인트)']
    orig_yearly['승률'] = orig_yearly['승률'] * 100
    
    print(orig_yearly)
    
    # 최종 데이터 연도별 분석
    print(f"\n{'='*100}")
    print("최종 데이터 연도별 성과")
    print(f"{'='*100}")
    
    final_yearly = df_final.groupby('year').agg({
        'is_win': ['count', 'mean'],
        'net_krw_optimal': ['sum', 'mean'],
        'net_pts': ['mean']
    }).round(2)
    
    final_yearly.columns = ['거래 수', '승률', '총 PnL (원)', '평균 PnL (원)', '평균 PnL (포인트)']
    final_yearly['승률'] = final_yearly['승률'] * 100
    
    print(final_yearly)
    
    # 연도별 비교
    print(f"\n{'='*100}")
    print("연도별 성과 비교 (원본 vs 최종)")
    print(f"{'='*100}")
    
    comparison = pd.DataFrame()
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        final_year = df_final[df_final['year'] == year]
        
        comparison.loc[year, '원본 거래 수'] = len(orig_year)
        comparison.loc[year, '최종 거래 수'] = len(final_year)
        comparison.loc[year, '거래 수 변화'] = len(final_year) - len(orig_year)
        
        comparison.loc[year, '원본 승률 (%)'] = orig_year['is_win'].mean() * 100
        comparison.loc[year, '최종 승률 (%)'] = final_year['is_win'].mean() * 100
        comparison.loc[year, '승률 변화 (%)'] = final_year['is_win'].mean() * 100 - orig_year['is_win'].mean() * 100
        
        comparison.loc[year, '원본 총 PnL (원)'] = orig_year['net_krw'].sum()
        comparison.loc[year, '최종 총 PnL (원)'] = final_year['net_krw_optimal'].sum()
        comparison.loc[year, '총 PnL 변화 (원)'] = final_year['net_krw_optimal'].sum() - orig_year['net_krw'].sum()
        
        comparison.loc[year, '원본 평균 PnL (원)'] = orig_year['net_krw'].mean()
        comparison.loc[year, '최종 평균 PnL (원)'] = final_year['net_krw_optimal'].mean()
        comparison.loc[year, '평균 PnL 변화 (원)'] = final_year['net_krw_optimal'].mean() - orig_year['net_krw'].mean()
    
    comparison = comparison.round(2)
    print(comparison)
    
    # 연도별 PnL 그래프용 데이터
    print(f"\n{'='*100}")
    print("연도별 PnL 변화 요약")
    print(f"{'='*100}")
    
    print(f"\n{'연도':<10}{'원본 총 PnL':>20}{'최종 총 PnL':>20}{'변화':>20}{'변화율 (%)':>15}")
    print(f"{'-'*85}")
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        final_year = df_final[df_final['year'] == year]
        
        orig_pnl = orig_year['net_krw'].sum()
        final_pnl = final_year['net_krw_optimal'].sum()
        change = final_pnl - orig_pnl
        change_pct = (change / abs(orig_pnl) * 100) if orig_pnl != 0 else 0
        
        print(f"{year:<10}{orig_pnl:>20,.0f}{final_pnl:>20,.0f}{change:>20,.0f}{change_pct:>15.2f}")
    
    # 전체 요약
    print(f"\n{'='*100}")
    print("전체 요약")
    print(f"{'='*100}")
    
    total_orig_trades = len(df_original)
    total_final_trades = len(df_final)
    
    total_orig_pnl = df_original['net_krw'].sum()
    total_final_pnl = df_final['net_krw_optimal'].sum()
    
    total_orig_win_rate = df_original['is_win'].mean() * 100
    total_final_win_rate = df_final['is_win'].mean() * 100
    
    print(f"\n{'지표':<20}{'원본':>20}{'최종':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'총 거래 수':<20}{total_orig_trades:>20}{total_final_trades:>20}"
          f"{total_final_trades - total_orig_trades:>20}")
    print(f"{'승률 (%)':<20}{total_orig_win_rate:>20.2f}{total_final_win_rate:>20.2f}"
          f"{total_final_win_rate - total_orig_win_rate:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_orig_pnl:>20,.0f}{total_final_pnl:>20,.0f}"
          f"{total_final_pnl - total_orig_pnl:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{df_original['net_krw'].mean():>20,.0f}"
          f"{df_final['net_krw_optimal'].mean():>20,.0f}"
          f"{df_final['net_krw_optimal'].mean() - df_original['net_krw'].mean():>20,.0f}")
    
    # 수익성이 가장 좋은 연도
    print(f"\n{'='*100}")
    print("수익성이 가장 좋은 연도")
    print(f"{'='*100}")
    
    best_year_orig = df_original.groupby('year')['net_krw'].sum().idxmax()
    best_year_final = df_final.groupby('year')['net_krw_optimal'].sum().idxmax()
    
    print(f"\n원본 데이터: {best_year_orig}년 ({df_original[df_original['year'] == best_year_orig]['net_krw'].sum():,.0f} 원)")
    print(f"최종 데이터: {best_year_final}년 ({df_final[df_final['year'] == best_year_final]['net_krw_optimal'].sum():,.0f} 원)")
    
    # 승률이 가장 좋은 연도
    print(f"\n{'='*100}")
    print("승률이 가장 좋은 연도")
    print(f"{'='*100}")
    
    best_win_rate_orig = df_original.groupby('year')['is_win'].mean().idxmax()
    best_win_rate_final = df_final.groupby('year')['is_win'].mean().idxmax()
    
    print(f"\n원본 데이터: {best_win_rate_orig}년 ({df_original[df_original['year'] == best_win_rate_orig]['is_win'].mean() * 100:.2f}%)")
    print(f"최종 데이터: {best_win_rate_final}년 ({df_final[df_final['year'] == best_win_rate_final]['is_win'].mean() * 100:.2f}%)")


def main():
    """메인 함수"""
    # 데이터 로드
    df_original = load_original_data()
    df_final = load_final_data()
    
    # 연도별 성과 분석
    analyze_yearly_performance(df_original, df_final)


if __name__ == "__main__":
    main()
