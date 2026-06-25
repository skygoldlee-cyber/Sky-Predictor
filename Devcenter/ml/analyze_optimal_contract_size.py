# -*- coding: utf-8 -*-
"""
리스크 대비 최적 계약수 분석
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def analyze_optimal_contract_size():
    """리스크 대비 최적 계약수 분석"""
    print("=" * 80)
    print("리스크 대비 최적 계약수 분석")
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
    std_dev = df['net_krw'].std()
    
    print(f"\n기본 통계 (1계약 기준):")
    print(f"  총 PnL: {total_pnl_1:,.0f}원")
    print(f"  최대 손실: {max_loss:,.0f}원")
    print(f"  최대 드로우다운: {max_drawdown:,.0f}원")
    print(f"  평균 손실: {avg_loss:,.0f}원")
    print(f"  표준 편차: {std_dev:,.0f}원")
    
    # 계약수별 분석
    contract_sizes = [1, 2, 3, 5, 10]
    
    print(f"\n{'='*80}")
    print("계약수별 리스크 대비 수익성 분석")
    print(f"{'='*80}")
    
    results = []
    
    for size in contract_sizes:
        total_pnl = total_pnl_1 * size
        max_loss_size = max_loss * size
        max_drawdown_size = max_drawdown * size
        avg_loss_size = avg_loss * size
        std_dev_size = std_dev * size
        
        # 리스크 대비 수익성 지표
        sharpe_ratio = total_pnl / abs(max_loss_size)  # 샤프 비율 (수익/최대 손실)
        dd_ratio = total_pnl / abs(max_drawdown_size)  # 드로우다운 비율 (수익/최대 드로우다운)
        risk_adjusted_return = total_pnl / abs(avg_loss_size)  # 리스크 조정 수익률
        volatility_ratio = total_pnl / std_dev_size  # 변동성 비율
        
        results.append({
            'size': size,
            'total_pnl': total_pnl,
            'max_loss': max_loss_size,
            'max_drawdown': max_drawdown_size,
            'sharpe_ratio': sharpe_ratio,
            'dd_ratio': dd_ratio,
            'risk_adjusted_return': risk_adjusted_return,
            'volatility_ratio': volatility_ratio
        })
        
        print(f"\n{size}계약 기준:")
        print(f"  총 PnL: {total_pnl:,.0f}원")
        print(f"  최대 손실: {max_loss_size:,.0f}원")
        print(f"  최대 드로우다운: {max_drawdown_size:,.0f}원")
        print(f"  샤프 비율 (수익/최대 손실): {sharpe_ratio:.4f}")
        print(f"  드로우다운 비율 (수익/최대 드로우다운): {dd_ratio:.4f}")
        print(f"  리스크 조정 수익률 (수익/평균 손실): {risk_adjusted_return:.4f}")
        print(f"  변동성 비율 (수익/표준 편차): {volatility_ratio:.4f}")
    
    # 종합 지표 분석
    print(f"\n{'='*80}")
    print("종합 리스크 대비 최적 계약수 분석")
    print(f"{'='*80}")
    
    # 각 지표별 최적 계약수
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    best_dd = max(results, key=lambda x: x['dd_ratio'])
    best_risk_adj = max(results, key=lambda x: x['risk_adjusted_return'])
    best_vol = max(results, key=lambda x: x['volatility_ratio'])
    
    print(f"\n샤프 비율 최적: {best_sharpe['size']}계약 (비율: {best_sharpe['sharpe_ratio']:.4f})")
    print(f"드로우다운 비율 최적: {best_dd['size']}계약 (비율: {best_dd['dd_ratio']:.4f})")
    print(f"리스크 조정 수익률 최적: {best_risk_adj['size']}계약 (비율: {best_risk_adj['risk_adjusted_return']:.4f})")
    print(f"변동성 비율 최적: {best_vol['size']}계약 (비율: {best_vol['volatility_ratio']:.4f})")
    
    # 종합 점수 계산
    for result in results:
        # 각 지표 정규화 (0-1)
        max_sharpe = max(r['sharpe_ratio'] for r in results)
        max_dd = max(r['dd_ratio'] for r in results)
        max_risk_adj = max(r['risk_adjusted_return'] for r in results)
        max_vol = max(r['volatility_ratio'] for r in results)
        
        result['norm_sharpe'] = result['sharpe_ratio'] / max_sharpe
        result['norm_dd'] = result['dd_ratio'] / max_dd
        result['norm_risk_adj'] = result['risk_adjusted_return'] / max_risk_adj
        result['norm_vol'] = result['volatility_ratio'] / max_vol
        
        # 종합 점수 (가중평균)
        result['total_score'] = (
            result['norm_sharpe'] * 0.3 +
            result['norm_dd'] * 0.3 +
            result['norm_risk_adj'] * 0.2 +
            result['norm_vol'] * 0.2
        )
    
    # 종합 점수 순위
    results_sorted = sorted(results, key=lambda x: x['total_score'], reverse=True)
    
    print(f"\n{'='*80}")
    print("종합 점수 순위")
    print(f"{'='*80}")
    
    for i, result in enumerate(results_sorted, 1):
        print(f"\n{i}위: {result['size']}계약")
        print(f"  종합 점수: {result['total_score']:.4f}")
        print(f"  샤프 비율: {result['sharpe_ratio']:.4f} (정규화: {result['norm_sharpe']:.4f})")
        print(f"  드로우다운 비율: {result['dd_ratio']:.4f} (정규화: {result['norm_dd']:.4f})")
        print(f"  리스크 조정 수익률: {result['risk_adjusted_return']:.4f} (정규화: {result['norm_risk_adj']:.4f})")
        print(f"  변동성 비율: {result['volatility_ratio']:.4f} (정규화: {result['norm_vol']:.4f})")
    
    # 최종 권장사항
    best_overall = results_sorted[0]
    
    print(f"\n{'='*80}")
    print("최종 권장사항")
    print(f"{'='*80}")
    
    print(f"\n리스크 대비 최적 계약수: {best_overall['size']}계약")
    print(f"  예상 총 PnL: {best_overall['total_pnl']:,.0f}원")
    print(f"  예상 최대 손실: {best_overall['max_loss']:,.0f}원")
    print(f"  예상 최대 드로우다운: {best_overall['max_drawdown']:,.0f}원")
    print(f"  종합 점수: {best_overall['total_score']:.4f}")
    
    return best_overall['size']


if __name__ == "__main__":
    optimal_size = analyze_optimal_contract_size()
