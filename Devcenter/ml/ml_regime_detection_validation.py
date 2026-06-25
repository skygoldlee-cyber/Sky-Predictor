"""
시장 레짐 변화 감지 시스템 실제 데이터 검증
"""

import sys
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"


def chow_test(y: np.ndarray, X: np.ndarray, breakpoint: int) -> tuple:
    """Chow Test로 구조적 변화 감지"""
    n = len(y)
    k = X.shape[1]
    
    # 전체 기간 회귀
    X_full = np.column_stack([np.ones(n), X])
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    rss_full = np.sum((y - X_full @ beta_full) ** 2)
    
    # 분할 기간 회귀
    X1 = X_full[:breakpoint]
    y1 = y[:breakpoint]
    X2 = X_full[breakpoint:]
    y2 = y[breakpoint:]
    
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    
    rss1 = np.sum((y1 - X1 @ beta1) ** 2)
    rss2 = np.sum((y2 - X2 @ beta2) ** 2)
    rss_pooled = rss1 + rss2
    
    # F-statistic 계산
    if n - 2 * k <= 0:
        return 0, 1.0
    
    F = ((rss_full - rss_pooled) / k) / (rss_pooled / (n - 2 * k))
    from scipy.stats import f
    p_value = 1 - f.cdf(F, k, n - 2 * k)
    
    return F, p_value


def detect_regime_changes_by_year(df: pd.DataFrame) -> Dict:
    """연도별 레짐 변화 감지"""
    years = sorted(df['year'].unique())
    results = {}
    
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        hist_data = df[df['year'] == year1]
        curr_data = df[df['year'] == year2]
        
        change_signals = []
        
        # 1. 변동성 급격 변화
        if 'entry_atr' in hist_data.columns and 'entry_atr' in curr_data.columns:
            hist_vol = hist_data['entry_atr'].mean()
            curr_vol = curr_data['entry_atr'].mean()
            
            if hist_vol > 0:
                vol_change = (curr_vol - hist_vol) / hist_vol
                if abs(vol_change) > 0.5:  # 50% 이상 변화
                    change_signals.append(f'변동성 급격 변화 ({vol_change:.1%})')
        
        # 2. 추세 전환
        if 'entry_close' in hist_data.columns and 'entry_close' in curr_data.columns:
            hist_slope = calculate_trend_slope(hist_data['entry_close'].values)
            curr_slope = calculate_trend_slope(curr_data['entry_close'].values)
            
            if hist_slope * curr_slope < 0:
                change_signals.append(f'추세 전환 (이전: {hist_slope:.4f}, 현재: {curr_slope:.4f})')
        
        # 3. 통계적 검정 (Chow Test)
        if 'entry_close' in hist_data.columns and 'entry_close' in curr_data.columns:
            combined_data = pd.concat([hist_data, curr_data])
            breakpoint = len(hist_data)
            
            y = combined_data['entry_close'].values
            X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
            
            F_stat, p_value = chow_test(y, X, breakpoint)
            
            if p_value < 0.05:
                change_signals.append(f'통계적 구조 변화 (F={F_stat:.2f}, p={p_value:.4f})')
        
        # 4. 승률 변화
        hist_winrate = hist_data['is_win'].mean()
        curr_winrate = curr_data['is_win'].mean()
        winrate_change = curr_winrate - hist_winrate
        
        if abs(winrate_change) > 0.2:  # 20% 이상 변화
            change_signals.append(f'승률 급격 변화 ({winrate_change:.1%})')
        
        results[f'{year1}_{year2}'] = {
            'change_signals': change_signals,
            'hist_winrate': hist_winrate,
            'curr_winrate': curr_winrate,
            'winrate_change': winrate_change
        }
    
    return results


def calculate_trend_slope(prices: np.ndarray) -> float:
    """추세 기울기 계산"""
    x = np.arange(len(prices))
    slope, _ = np.polyfit(x, prices, 1)
    return slope


def print_regime_detection_results(results: Dict):
    """레짐 변화 감지 결과 출력"""
    print(f"\n{'='*80}")
    print("시장 레짐 변화 감지 결과 (연도별)")
    print(f"{'='*80}")
    
    total_changes = 0
    significant_changes = []
    
    for period, result in results.items():
        year1, year2 = period.split('_')
        
        print(f"\n{year1} → {year2}:")
        print(f"  승률: {result['hist_winrate']:.2%} → {result['curr_winrate']:.2%} ({result['winrate_change']:+.1%})")
        
        if result['change_signals']:
            print(f"  감지된 변화:")
            for signal in result['change_signals']:
                print(f"    - {signal}")
                total_changes += 1
        else:
            print(f"  감지된 변화: 없음")
        
        # 중요한 변화 기록 (승률 20% 이상 변화 또는 2개 이상 신호)
        if abs(result['winrate_change']) > 0.2 or len(result['change_signals']) >= 2:
            significant_changes.append(period)
    
    print(f"\n{'='*80}")
    print("요약")
    print(f"{'='*80}")
    print(f"총 감지된 변화: {total_changes}건")
    print(f"중요한 변화 기간: {len(significant_changes)}건")
    
    if significant_changes:
        print(f"중요 변화 기간: {', '.join(significant_changes)}")
    
    return significant_changes


def analyze_volatility_by_year(df: pd.DataFrame):
    """연도별 변동성 분석"""
    print(f"\n{'='*80}")
    print("연도별 변동성 분석")
    print(f"{'='*80}")
    
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        
        if 'entry_atr' in year_data.columns:
            atr_mean = year_data['entry_atr'].mean()
            atr_std = year_data['entry_atr'].std()
            
            print(f"\n{year}년:")
            print(f"  ATR 평균: {atr_mean:.4f}")
            print(f"  ATR 표준편차: {atr_std:.4f}")
            print(f"  승률: {year_data['is_win'].mean():.2%}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("시장 레짐 변화 감지 시스템 실제 데이터 검증")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 연도별 변동성 분석
    analyze_volatility_by_year(df)
    
    # 레짐 변화 감지
    results = detect_regime_changes_by_year(df)
    
    # 결과 출력
    significant_changes = print_regime_detection_results(results)
    
    return results, significant_changes


if __name__ == "__main__":
    results, significant_changes = main()
