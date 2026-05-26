"""기존 ZigZag 피봇 테스트 스크립트.

data/minute_bars/의 CSV 데이터를 사용하여 기존 ZigZag 피봇을 테스트합니다.
KP200 선물과 KOSPI 지수의 피봇 수를 카운트합니다.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_legacy_pivot_features(csv_path: Path, symbol: str = "KP200 선물", zigzag_config_key: str = "futures_zigzag"):
    """기존 ZigZag 피봇 테스트."""
    logger.info("=" * 60)
    logger.info("기존 ZigZag 피봇 테스트 시작")
    logger.info(f"파일: {csv_path}")
    logger.info(f"심볼: {symbol}")
    logger.info(f"ZigZag 설정: {zigzag_config_key}")
    logger.info("=" * 60)

    # CSV 로드
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"데이터 로드 완료: {len(df)} 봉")

    # Config 로드 (파일에서 직접 로드하여 변경사항 반영)
    import json
    with open("config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    ad = cfg_dict.get('adaptive_indicator', {})
    zz_cfg = ad.get(zigzag_config_key, {})
    
    # AdaptiveZigZag 초기화
    zigzag = AdaptiveZigZag(AdaptiveZigZagConfig(
        atr_multiplier=zz_cfg.get('atr_multiplier', 2.0),
        atr_period=zz_cfg.get('atr_period', 14),
        er_period=zz_cfg.get('er_period', 10),
        atr_multiplier_min=zz_cfg.get('atr_multiplier_min', 1.2),
        atr_multiplier_max=zz_cfg.get('atr_multiplier_max', 4.0),
        pivot_threshold_min_pct=zz_cfg.get('pivot_threshold_min_pct', 0.5),
        pivot_threshold_max_pct=zz_cfg.get('pivot_threshold_max_pct', 2.0),
        major_swing_ratio=zz_cfg.get('major_swing_ratio', 1.5),
        max_swings=zz_cfg.get('max_swings', 50),
        confirmation_bars=zz_cfg.get('confirmation_bars', 2),
        freeze_on_confirm=zz_cfg.get('freeze_on_confirm', True),
        min_wave_bars=zz_cfg.get('min_wave_bars', 15),
        min_wave_pct=zz_cfg.get('min_wave_pct', 0.3),
        structure_lookback_swings=zz_cfg.get('structure_lookback_swings', 30),
        structure_points=zz_cfg.get('structure_points', 4),
        early_session_start_time=zz_cfg.get('early_session_start_time', "09:00"),
        early_session_end_time=zz_cfg.get('early_session_end_time', "09:30"),
        early_session_atr_multiplier_max=zz_cfg.get('early_session_atr_multiplier_max', 8.0),
        confirmation_bars_ranging=zz_cfg.get('confirmation_bars_ranging', 3),
        confirmation_bars_unknown=zz_cfg.get('confirmation_bars_unknown', 3),
        structure_majority_threshold=zz_cfg.get('structure_majority_threshold', 0.7),
        decay_start_bars=zz_cfg.get('decay_start_bars', 30),
        decay_rate_per_bar=zz_cfg.get('decay_rate_per_bar', 0.005),
        decay_max_pct=zz_cfg.get('decay_max_pct', 0.3),
        major_wave_ratio=zz_cfg.get('major_wave_ratio', 1.5),
        major_wave_lookback=zz_cfg.get('major_wave_lookback', 3),
        der_mismatch_threshold=zz_cfg.get('der_mismatch_threshold', 0.3),
        der_mismatch_mult_ratio=zz_cfg.get('der_mismatch_mult_ratio', 0.7),
        pivot_lifecycle_log=zz_cfg.get('pivot_lifecycle_log', False),
        pivot_lifecycle_log_prefix=zz_cfg.get('pivot_lifecycle_log_prefix', ""),
        enable_pivot_collector=zz_cfg.get('enable_pivot_collector', False),
        pivot_collector_max_sequence=zz_cfg.get('pivot_collector_max_sequence', 120),
        multi_timeframe_enabled=zz_cfg.get('multi_timeframe_enabled', False),
        multi_timeframe_scales=zz_cfg.get('multi_timeframe_scales', [1, 5, 15]),
        multi_timeframe_consensus_threshold=zz_cfg.get('multi_timeframe_consensus_threshold', 2),
        multi_timeframe_price_tolerance_pct=zz_cfg.get('multi_timeframe_price_tolerance_pct', 1.0),
        multi_timeframe_index_tolerance_multiplier=zz_cfg.get('multi_timeframe_index_tolerance_multiplier', 2.0),
        session_min_wave_bars_table=zz_cfg.get('session_min_wave_bars_table', []),
        session_min_wave_atr_ratio_table=zz_cfg.get('session_min_wave_atr_ratio_table', []),
        use_atr_based_filtering=zz_cfg.get('use_atr_based_filtering', True),
        min_wave_atr_ratio=zz_cfg.get('min_wave_atr_ratio', 1.5),
        cluster_atr_ratio=zz_cfg.get('cluster_atr_ratio', 1.0),
        cluster_tolerance_pct=zz_cfg.get('cluster_tolerance_pct', 0.1),
    ))
    zigzag.set_symbol(symbol)
    logger.info(f"AdaptiveZigZag 초기화 완료 (설정: {zigzag_config_key})")

    # 로그 핸들러 추가하여 확정 피봇 카운트
    class PivotCountHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0
        
        def emit(self, record):
            msg = record.getMessage()
            if "[ZZ][확정]" in msg:
                self.count += 1
    
    pivot_handler = PivotCountHandler()
    pivot_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(pivot_handler)

    # 데이터 순회
    for idx, row in df.iterrows():
        high = float(row['High'])
        low = float(row['Low'])
        close = float(row['Close'])
        ts = row['timestamp']

        try:
            # ZigZag 업데이트
            state = zigzag.update(high=high, low=low, close=close, bar_time=ts)
        except Exception as e:
            logger.debug(f"업데이트 실패 @ {ts}: {e}")
            continue

    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 완료")
    logger.info(f"심볼: {symbol}")
    logger.info(f"ZigZag 설정: {zigzag_config_key}")
    logger.info(f"총 봉 수: {len(df)}")
    logger.info(f"확정된 피봇: {pivot_handler.count}개")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="기존 ZigZag 피봇 테스트")
    parser.add_argument("--file", type=str, help="CSV 파일 경로")
    parser.add_argument("--symbol", type=str, default="KP200 선물", help="심볼 이름")
    parser.add_argument("--zigzag-config", type=str, default="futures_zigzag", 
                        choices=["zigzag", "futures_zigzag", "kospi_zigzag"],
                        help="ZigZag 설정 키 (기본: futures_zigzag)")
    
    args = parser.parse_args()
    
    if args.file:
        csv_path = Path(args.file)
    else:
        # 기본: 가장 최신 kp200 파일
        data_dir = Path("data/minute_bars")
        kp200_files = sorted(data_dir.glob("kp200_*.csv"))
        if kp200_files:
            csv_path = kp200_files[-1]
        else:
            logger.error("CSV 파일을 찾을 수 없습니다")
            sys.exit(1)
    
    test_legacy_pivot_features(csv_path, args.symbol, args.zigzag_config)
