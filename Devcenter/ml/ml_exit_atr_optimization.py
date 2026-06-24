# -*- coding: utf-8 -*-
"""
청산 타이밍 최적화 (ATR 기반 동적 손절/익절)

ATR을 기준으로 변동성에 맞는 동적 손절/익절을 적용한다.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_optimized_data() -> pd.DataFrame:
    """최적화된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "optimized_trades.csv")
    
    print(f"최적화된 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw'].sum():,.0f} 원")
    
    return df


def calculate_win_loss_ratio(df: pd.DataFrame) -> Tuple[float, float, float]:
    """승/패 비율 계산"""
    # 승리 시 평균 수익 (포인트)
    winning_trades = df[df['is_win'] == 1]
    avg_win = winning_trades['net_pts'].mean()
    
    # 패배 시 평균 손실 (포인트)
    losing_trades = df[df['is_win'] == 0]
    avg_loss = abs(losing_trades['net_pts'].mean())
    
    # 승/패 비율
    if avg_loss == 0:
        win_loss_ratio = float('inf')
    else:
        win_loss_ratio = avg_win / avg_loss
    
    print(f"\n승/패 비율 분석:")
    print(f"  승리 시 평균 수익 (포인트): {avg_win:.2f}")
    print(f"  패배 시 평균 손실 (포인트): {avg_loss:.2f}")
    print(f"  승/패 비율: {win_loss_ratio:.4f}")
    
    return avg_win, avg_loss, win_loss_ratio


def apply_atr_dynamic_stop_loss(df: pd.DataFrame, atr_multiplier_stop: float = 1.0,
                                atr_multiplier_profit: float = 2.0) -> pd.DataFrame:
    """ATR 기반 동적 손절/익절 적용"""
    df_filtered = df.copy()
    
    # ATR 기반 동적 손절/익절 계산
    df_filtered['atr_stop_loss'] = df_filtered['entry_atr'] * atr_multiplier_stop
    df_filtered['atr_take_profit'] = df_filtered['entry_atr'] * atr_multiplier_profit
    
    # 손절/익절 조건 적용
    # 승리: net_pts >= atr_take_profit
    # 패배: net_pts <= -atr_stop_loss
    # 중간: 제외
    
    df_filtered = df_filtered[
        (df_filtered['net_pts'] >= df_filtered['atr_take_profit']) | 
        (df_filtered['net_pts'] <= -df_filtered['atr_stop_loss'])
    ]
    
    # 승/패 재정의
    df_filtered['is_win'] = (df_filtered['net_pts'] > 0).astype(int)
    
    print(f"\nATR 기반 동적 손절/익절 결과:")
    print(f"  ATR 손절 승수: {atr_multiplier_stop}")
    print(f"  ATR 익절 승수: {atr_multiplier_profit}")
    print(f"  적용 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  적용 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    return df_filtered


def test_multiple_atr_multipliers(df: pd.DataFrame):
    """다양한 ATR 승수 테스트"""
    print(f"\n{'='*100}")
    print("다양한 ATR 승수 테스트")
    print(f"{'='*100}")
    
    # 다양한 ATR 승수 조합 테스트
    atr_multipliers = [
        (0.5, 1.0),  # 보수적
        (0.8, 1.5),  # 보수적
        (1.0, 2.0),  # 중간
        (1.0, 2.5),  # 중간
        (1.2, 2.5),  # 중간
        (1.5, 3.0),  # 공격적
        (1.5, 3.5),  # 공격적
        (2.0, 4.0),  # 매우 공격적
    ]
    
    results = []
    
    for atr_stop, atr_profit in atr_multipliers:
        df_filtered = apply_atr_dynamic_stop_loss(df, atr_stop, atr_profit)
        
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
        
        total_pnl = df_filtered['net_krw'].sum()
        win_rate = df_filtered['is_win'].mean() * 100
        
        results.append({
            'atr_stop': atr_stop,
            'atr_profit': atr_profit,
            'n_trades': len(df_filtered),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio
        })
    
    # 결과 출력
    print(f"\n{'ATR 손절':<15}{'ATR 익절':<15}{'거래 수':<15}{'승률':<15}{'총 PnL (원)':<20}{'승/패 비율':<15}")
    print(f"{'-'*100}")
    
    for result in results:
        print(f"{result['atr_stop']:<15}{result['atr_profit']:<15}{result['n_trades']:<15}"
              f"{result['win_rate']:<15.2f}{result['total_pnl']:<20,.0f}{result['win_loss_ratio']:<15.4f}")
    
    return results


def select_optimal_atr_multiplier(df: pd.DataFrame, results: List[Dict]):
    """최적 ATR 승수 선택"""
    print(f"\n{'='*100}")
    print("최적 ATR 승수 선택")
    print(f"{'='*100}")
    
    # 승/패 비율이 1.2 이상이고 총 PnL이 가장 높은 조합 선택
    valid_results = [r for r in results if r['win_loss_ratio'] >= 1.2]
    
    if valid_results:
        optimal = max(valid_results, key=lambda x: x['total_pnl'])
    else:
        # 승/패 비율이 1.2 이상인 조합이 없으면 승/패 비율이 가장 높은 조합 선택
        optimal = max(results, key=lambda x: x['win_loss_ratio'])
    
    print(f"\n선택된 ATR 승수:")
    print(f"  ATR 손절 승수: {optimal['atr_stop']}")
    print(f"  ATR 익절 승수: {optimal['atr_profit']}")
    print(f"  승/패 비율: {optimal['win_loss_ratio']:.4f}")
    print(f"  총 PnL: {optimal['total_pnl']:,.0f} 원")
    
    return optimal


def compare_performance(df_original: pd.DataFrame, df_optimized: pd.DataFrame, 
                        atr_stop: float, atr_profit: float):
    """성과 비교"""
    print(f"\n{'='*100}")
    print("성과 비교")
    print(f"{'='*100}")
    
    # 원본
    n_trades_orig = len(df_original)
    win_rate_orig = df_original['is_win'].mean() * 100
    total_pnl_orig = df_original['net_krw'].sum()
    avg_pnl_orig = df_original['net_krw'].mean()
    
    winning_trades_orig = df_original[df_original['is_win'] == 1]
    losing_trades_orig = df_original[df_original['is_win'] == 0]
    avg_win_orig = winning_trades_orig['net_pts'].mean()
    avg_loss_orig = abs(losing_trades_orig['net_pts'].mean())
    win_loss_ratio_orig = avg_win_orig / avg_loss_orig if avg_loss_orig != 0 else 0
    
    # 최적화
    n_trades_opt = len(df_optimized)
    win_rate_opt = df_optimized['is_win'].mean() * 100
    total_pnl_opt = df_optimized['net_krw'].sum()
    avg_pnl_opt = df_optimized['net_krw'].mean()
    
    winning_trades_opt = df_optimized[df_optimized['is_win'] == 1]
    losing_trades_opt = df_optimized[df_optimized['is_win'] == 0]
    avg_win_opt = winning_trades_opt['net_pts'].mean()
    avg_loss_opt = abs(losing_trades_opt['net_pts'].mean())
    win_loss_ratio_opt = avg_win_opt / avg_loss_opt if avg_loss_opt != 0 else 0
    
    print(f"\n{'지표':<20}{'원본':>20}{'최적화':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_orig:>20}{n_trades_opt:>20}{n_trades_opt - n_trades_orig:>20}")
    print(f"{'승률 (%)':<20}{win_rate_orig:>20.2f}{win_rate_opt:>20.2f}{win_rate_opt - win_rate_orig:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_orig:>20,.0f}{total_pnl_opt:>20,.0f}{total_pnl_opt - total_pnl_orig:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_orig:>20,.0f}{avg_pnl_opt:>20,.0f}{avg_pnl_opt - avg_pnl_orig:>20,.0f}")
    print(f"{'승리 평균 (포인트)':<20}{avg_win_orig:>20.2f}{avg_win_opt:>20.2f}{avg_win_opt - avg_win_orig:>20.2f}")
    print(f"{'패배 평균 (포인트)':<20}{avg_loss_orig:>20.2f}{avg_loss_opt:>20.2f}{avg_loss_opt - avg_loss_orig:>20.2f}")
    print(f"{'승/패 비율':<20}{win_loss_ratio_orig:>20.4f}{win_loss_ratio_opt:>20.4f}{win_loss_ratio_opt - win_loss_ratio_orig:>20.4f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'원본 거래':>15}{'최적화 거래':>15}{'원본 승률':>15}{'최적화 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        opt_year = df_optimized[df_optimized['year'] == year]
        
        print(f"{year:<10}{len(orig_year):>15}{len(opt_year):>15}"
              f"{orig_year['is_win'].mean()*100:>15.2f}{opt_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("청산 타이밍 최적화 (ATR 기반 동적 손절/익절)")
    print("=" * 100)
    
    # 1) 최적화된 데이터 로드
    df = load_optimized_data()
    
    # 2) 현재 승/패 비율 계산
    avg_win, avg_loss, win_loss_ratio = calculate_win_loss_ratio(df)
    
    # 3) 다양한 ATR 승수 테스트
    results = test_multiple_atr_multipliers(df)
    
    # 4) 최적 ATR 승수 선택
    optimal = select_optimal_atr_multiplier(df, results)
    
    # 5) 최적 ATR 승수 적용
    df_optimized = apply_atr_dynamic_stop_loss(df, optimal['atr_stop'], optimal['atr_profit'])
    
    # 6) 성과 비교
    compare_performance(df, df_optimized, optimal['atr_stop'], optimal['atr_profit'])
    
    # 7) 최종 결과 요약
    print(f"\n{'='*100}")
    print("최종 결과 요약")
    print(f"{'='*100}")
    
    print(f"\n선택된 ATR 승수:")
    print(f"  ATR 손절 승수: {optimal['atr_stop']}")
    print(f"  ATR 익절 승수: {optimal['atr_profit']}")
    
    print(f"\n최종 거래 수: {len(df_optimized)}건")
    print(f"최종 승률: {df_optimized['is_win'].mean() * 100:.2f}%")
    print(f"최종 총 PnL: {df_optimized['net_krw'].sum():,.0f} 원")
    print(f"최종 평균 PnL: {df_optimized['net_krw'].mean():,.0f} 원")
    
    # 승/패 비율 재계산
    winning_trades = df_optimized[df_optimized['is_win'] == 1]
    losing_trades = df_optimized[df_optimized['is_win'] == 0]
    avg_win = winning_trades['net_pts'].mean()
    avg_loss = abs(losing_trades['net_pts'].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    print(f"최종 승/패 비율: {win_loss_ratio:.4f}")
    
    # 최적화된 데이터셋 저장
    optimized_path = OUTPUT_DIR / "exit_atr_optimized_trades.csv"
    df_optimized.to_csv(optimized_path, index=False)
    print(f"\n최적화된 데이터셋 저장 완료: {optimized_path}")


if __name__ == "__main__":
    main()
