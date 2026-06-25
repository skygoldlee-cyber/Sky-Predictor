# -*- coding: utf-8 -*-
"""
1억 자본금 기준 수익률 분석
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def analyze_return_on_capital():
    """1억 자본금 기준 수익률 분석"""
    print("=" * 80)
    print("1억 자본금 기준 수익률 분석")
    print("=" * 80)
    
    # 자본금 설정
    capital = 100000000  # 1억원
    
    # 현실적 롤링 윈도우 모델 성과 (1계약 기준)
    total_pnl_1contract = 10714129  # 10,714,129원
    
    # 3계약 기준 성과
    total_pnl_3contract = 32142386  # 32,142,386원
    
    print(f"\n자본금: {capital:,.0f}원")
    
    # 1계약 기준 수익률 분석
    print(f"\n{'='*80}")
    print("1계약 기준 수익률 분석")
    print(f"{'='*80}")
    
    return_1contract = (total_pnl_1contract / capital) * 100
    max_loss_1contract = 1611878  # 최대 손실
    max_dd_1contract = 519422  # 최대 드로우다운
    
    print(f"\n총 PnL: {total_pnl_1contract:,.0f}원")
    print(f"수익률: {return_1contract:.2f}%")
    print(f"최대 손실: {max_loss_1contract:,.0f}원")
    print(f"최대 손실률: {(max_loss_1contract / capital) * 100:.2f}%")
    print(f"최대 드로우다운: {max_dd_1contract:,.0f}원")
    print(f"최대 드로우다운률: {(max_dd_1contract / capital) * 100:.2f}%")
    print(f"샤프 비율: {return_1contract / abs((max_loss_1contract / capital) * 100):.4f}")
    
    # 3계약 기준 수익률 분석
    print(f"\n{'='*80}")
    print("3계약 기준 수익률 분석")
    print(f"{'='*80}")
    
    return_3contract = (total_pnl_3contract / capital) * 100
    max_loss_3contract = 4835634  # 최대 손실
    max_dd_3contract = 1558266  # 최대 드로우다운
    
    print(f"\n총 PnL: {total_pnl_3contract:,.0f}원")
    print(f"수익률: {return_3contract:.2f}%")
    print(f"최대 손실: {max_loss_3contract:,.0f}원")
    print(f"최대 손실률: {(max_loss_3contract / capital) * 100:.2f}%")
    print(f"최대 드로우다운: {max_dd_3contract:,.0f}원")
    print(f"최대 드로우다운률: {(max_dd_3contract / capital) * 100:.2f}%")
    print(f"샤프 비율: {return_3contract / abs((max_loss_3contract / capital) * 100):.4f}")
    
    # 비교 분석
    print(f"\n{'='*80}")
    print("비교 분석")
    print(f"{'='*80}")
    
    print(f"\n{'지표':<20} {'1계약':<15} {'3계약':<15} {'증가율':<15}")
    print(f"{'-'*65}")
    print(f"{'수익률':<20} {return_1contract:>14.2f}% {return_3contract:>14.2f}% {((return_3contract - return_1contract) / return_1contract * 100):>14.2f}%")
    print(f"{'최대 손실률':<20} {abs((max_loss_1contract / capital) * 100):>14.2f}% {abs((max_loss_3contract / capital) * 100):>14.2f}% {((abs((max_loss_3contract / capital) * 100) - abs((max_loss_1contract / capital) * 100)) / abs((max_loss_1contract / capital) * 100) * 100):>14.2f}%")
    print(f"{'최대 드로우다운률':<20} {abs((max_dd_1contract / capital) * 100):>14.2f}% {abs((max_dd_3contract / capital) * 100):>14.2f}% {((abs((max_dd_3contract / capital) * 100) - abs((max_dd_1contract / capital) * 100)) / abs((max_dd_1contract / capital) * 100) * 100):>14.2f}%")
    print(f"{'샤프 비율':<20} {return_1contract / abs((max_loss_1contract / capital) * 100):>14.4f} {return_3contract / abs((max_loss_3contract / capital) * 100):>14.4f} {((return_3contract / abs((max_loss_3contract / capital) * 100)) - (return_1contract / abs((max_loss_1contract / capital) * 100))) / (return_1contract / abs((max_loss_1contract / capital) * 100)) * 100:>14.2f}%")
    
    # 리스크 관리 권장사항
    print(f"\n{'='*80}")
    print("리스크 관리 권장사항")
    print(f"{'='*80}")
    
    print(f"\n1계약 기준:")
    print(f"  수익률: {return_1contract:.2f}%")
    print(f"  리스크: 최대 손실률 {abs((max_loss_1contract / capital) * 100):.2f}%, 최대 드로우다운률 {abs((max_dd_1contract / capital) * 100):.2f}%")
    print(f"  권장: 가장 안전한 옵션, 리스크 관리 용이")
    
    print(f"\n3계약 기준:")
    print(f"  수익률: {return_3contract:.2f}%")
    print(f"  리스크: 최대 손실률 {abs((max_loss_3contract / capital) * 100):.2f}%, 최대 드로우다운률 {abs((max_dd_3contract / capital) * 100):.2f}%")
    print(f"  권장: 리스크 허용도가 높은 경우 고려 가능, 리스크 관리 강화 필요")
    
    # 연평균 수익률 (8년 기간)
    print(f"\n{'='*80}")
    print("연평균 수익률 (8년 기간)")
    print(f"{'='*80}")
    
    annual_return_1 = return_1contract / 8
    annual_return_3 = return_3contract / 8
    
    print(f"\n1계약 기준 연평균 수익률: {annual_return_1:.2f}%")
    print(f"3계약 기준 연평균 수익률: {annual_return_3:.2f}%")
    
    return return_1contract, return_3contract


if __name__ == "__main__":
    return_1contract, return_3contract = analyze_return_on_capital()
