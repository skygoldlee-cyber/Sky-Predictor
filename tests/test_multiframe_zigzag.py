"""다중 시간프레임 지그재그 백테스트 스크립트."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
from indicators.multi_timeframe_zigzag import MultiTimeframeZigZag


def generate_test_data(n_bars: int = 1000) -> pd.DataFrame:
    """테스트용 OHLC 데이터 생성.
    
    Args:
        n_bars: 생성할 봉 수
        
    Returns:
        OHLC DataFrame
    """
    np.random.seed(42)
    
    # 랜덤 워크 + 추세
    price = 100.0
    prices = []
    
    for i in range(n_bars):
        change = np.random.normal(0, 0.5)
        # 추세 추가 (상승/하락 반복)
        if i % 200 < 100:
            change += 0.1  # 상승 추세
        else:
            change -= 0.1  # 하락 추세
        
        price += change
        prices.append(price)
    
    # OHLC 생성
    data = []
    for i in range(len(prices)):
        if i == 0:
            open_p = prices[i]
        else:
            open_p = prices[i-1]
        
        close_p = prices[i]
        high_p = max(open_p, close_p) + np.random.uniform(0, 0.3)
        low_p = min(open_p, close_p) - np.random.uniform(0, 0.3)
        
        data.append({
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p,
            'Volume': np.random.randint(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    return df


def test_multiframe_basic():
    """기본 다중 시간프레임 테스트."""
    print("=" * 60)
    print("다중 시간프레임 지그재그 기본 테스트")
    print("=" * 60)
    
    # 테스트 데이터 생성
    df = generate_test_data(500)
    print(f"\n테스트 데이터: {len(df)} 봉")
    
    # 1분봉 ZigZag
    config = AdaptiveZigZagConfig(
        atr_multiplier=1.5,
        confirmation_bars=2,
        multi_timeframe_enabled=False,  # [FIX] 테스트에서 비활성화 (상위 TF 데이터 없음)
        multi_timeframe_scales=[1, 5, 15],
        multi_timeframe_consensus_threshold=2,
        multi_timeframe_price_tolerance_pct=1.0,
        multi_timeframe_index_tolerance_multiplier=2.0
    )
    
    zz = AdaptiveZigZag(config)
    
    # 데이터 업데이트
    for idx, row in df.iterrows():
        zz.update(
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            open=row['Open'],
            volume=row['Volume']
        )
    
    # 확정 피봇 확인
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    print(f"\n확정 피봇 수: {len(confirmed_swings)}")
    
    for i, sw in enumerate(confirmed_swings[:5]):
        print(f"  피봇 {i+1}: {sw.swing_type} @ {sw.price:.2f} (idx={sw.index})")
    
    # MultiTimeframeZigZag 테스트
    mtf_zz = MultiTimeframeZigZag(
        scales=[5, 15],
        consensus_threshold=2
    )
    
    # 가짜 상위 시간프레임 피봇 캐시 (실제 피봇 인덱스에 맞게 조정)
    if confirmed_swings:
        test_pivot = confirmed_swings[0]
        pivot_type = 'H' if str(test_pivot.swing_type).upper() == 'HIGH' else 'L'
        
        # 실제 피봇 인덱스 근처에 가짜 피봇 생성 (매칭 테스트용)
        fake_5m_pivots = [
            {'index': test_pivot.index, 'price': test_pivot.price * 1.005, 'pivot_type': pivot_type},  # 0.5% 차이
        ]
        
        fake_15m_pivots = [
            {'index': test_pivot.index, 'price': test_pivot.price * 0.998, 'pivot_type': pivot_type},  # 0.2% 차이
        ]
        
        mtf_zz.update_pivot_cache(5, fake_5m_pivots)
        mtf_zz.update_pivot_cache(15, fake_15m_pivots)
        
        # 합의도 확인
        result = mtf_zz.check_consensus(
            pivot_index=test_pivot.index,
            pivot_price=test_pivot.price,
            pivot_type=pivot_type
        )
        print("\n합의도 확인 결과:")
        print(f"  합의도: {result['consensus']}/{result['total_scales']}")
        print(f"  비율: {result['consensus_ratio']:.1%}")
        print(f"  통과: {result['passed']}")
        
        # 성능 통계
        stats = mtf_zz.get_performance_stats()
        print("\n성능 통계:")
        print(f"  확인 횟수: {stats['check_count']}")
        print(f"  합의도 매칭: {stats['consensus_match_count']}")
        print(f"  합의도 성공률: {stats['consensus_rate']:.1f}%")
    
    print("\n테스트 완료!")


def test_multiframe_config():
    """설정 로드 테스트."""
    print("\n" + "=" * 60)
    print("설정 로드 테스트")
    print("=" * 60)
    
    try:
        import json
        config_path = project_root / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            kospi_zz_config = config.get('adaptive_indicator', {}).get('kospi_zigzag', {})
            futures_zz_config = config.get('adaptive_indicator', {}).get('futures_zigzag', {})
            
            print("\nKOSPI ZigZag 설정:")
            print(f"  multi_timeframe_enabled: {kospi_zz_config.get('multi_timeframe_enabled', False)}")
            print(f"  multi_timeframe_scales: {kospi_zz_config.get('multi_timeframe_scales', [])}")
            print(f"  consensus_threshold: {kospi_zz_config.get('multi_timeframe_consensus_threshold', 0)}")
            
            print("\nFutures ZigZag 설정:")
            print(f"  multi_timeframe_enabled: {futures_zz_config.get('multi_timeframe_enabled', False)}")
            print(f"  multi_timeframe_scales: {futures_zz_config.get('multi_timeframe_scales', [])}")
            print(f"  consensus_threshold: {futures_zz_config.get('multi_timeframe_consensus_threshold', 0)}")
            
            print("\n설정 로드 성공!")
        else:
            print(f"\n설정 파일 없음: {config_path}")
            
    except Exception as e:
        print(f"\n설정 로드 실패: {e}")


if __name__ == "__main__":
    test_multiframe_basic()
    test_multiframe_config()
