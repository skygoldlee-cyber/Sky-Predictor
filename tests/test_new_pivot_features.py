"""신규 4-Layer 피처 테스트 스크립트.

data/minute_bars/의 CSV 데이터를 사용하여 신규 피처를 테스트합니다.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import (
    ATRAdaptivePivot, ATRAdaptivePivotConfig,
    MarketStructureBreak, MSBConfig,
    KalmanTurningPoint, KalmanConfig,
    PivotScoreIntegrator, IntegratorConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_new_pivot_features(csv_path: Path, symbol: str = "KP200 선물"):
    """신규 피처 테스트."""
    logger.info("=" * 60)
    logger.info("신규 4-Layer 피처 테스트 시작")
    logger.info(f"파일: {csv_path}")
    logger.info(f"심볼: {symbol}")
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
    
    # ATRAdaptivePivot 초기화
    atr_cfg = ad.get('atr_pivot', {})
    aap = ATRAdaptivePivot(ATRAdaptivePivotConfig(
        atr_period=atr_cfg.get('atr_period', 14),
        base_multiplier=atr_cfg.get('base_multiplier', 2.0),
        multiplier_min=atr_cfg.get('multiplier_min', 1.2),
        multiplier_max=atr_cfg.get('multiplier_max', 3.5),
        er_period=atr_cfg.get('er_period', 10),
        confirmation_bars=atr_cfg.get('confirmation_bars', 1),
        min_wave_atr_ratio=atr_cfg.get('min_wave_atr_ratio', 0.5),
        warmup_bars=atr_cfg.get('warmup_bars', 20),
    ))
    aap.set_symbol(symbol)
    logger.info("ATRAdaptivePivot 초기화 완료")

    # MarketStructureBreak 초기화
    msb_cfg = ad.get('msb', {})
    msb = MarketStructureBreak(MSBConfig(
        swing_lookback=msb_cfg.get('swing_lookback', 3),
        bos_buffer_pct=msb_cfg.get('bos_buffer_pct', 0.20),
        structure_lookback_pivots=msb_cfg.get('structure_lookback_pivots', 6),
        choch_enabled=msb_cfg.get('choch_enabled', True),
    ))
    logger.info("MarketStructureBreak 초기화 완료")

    # KalmanTurningPoint 초기화
    kf_cfg = ad.get('kalman', {})
    kf = KalmanTurningPoint(KalmanConfig(
        q=kf_cfg.get('q', 0.01),
        r=kf_cfg.get('r', 2.0),
        warmup_bars=kf_cfg.get('warmup_bars', 15),
        slope_flip_min=kf_cfg.get('slope_flip_min', 0.005),
        adaptive_q=kf_cfg.get('adaptive_q', True),
    ))
    logger.info("KalmanTurningPoint 초기화 완료")

    # PivotScoreIntegrator 초기화
    int_cfg = ad.get('integrator', {})
    integrator = PivotScoreIntegrator(IntegratorConfig(
        w_aap=int_cfg.get('w_aap', 0.30),
        w_msb=int_cfg.get('w_msb', 0.30),
        w_oi=int_cfg.get('w_oi', 0.20),
        w_kf=int_cfg.get('w_kf', 0.20),
        entry_threshold=int_cfg.get('entry_threshold', 0.55),
        strong_threshold=int_cfg.get('strong_threshold', 0.72),
        regime_boost=int_cfg.get('regime_boost', 1.15),
        regime_suppress=int_cfg.get('regime_suppress', 0.85),
    ))
    logger.info("PivotScoreIntegrator 초기화 완료")

    # 피봇 카운터
    aap_pivots = 0
    msb_signals = 0
    kf_turns = 0
    integrated_pivots = 0

    # 데이터 순회
    for idx, row in df.iterrows():
        high = float(row['High'])
        low = float(row['Low'])
        close = float(row['Close'])
        ts = row['timestamp']

        try:
            # ATRAdaptivePivot 업데이트
            aap_state = aap.update(high=high, low=low, close=close, bar_time=ts)
            aap_score = aap_state.pivot_score if aap_state else 0.0
            aap_signal = aap_state.new_pivot_signal if aap_state else "none"
            if aap_state and hasattr(aap_state, 'new_pivot_signal') and aap_state.new_pivot_signal != "none":
                aap_pivots += 1
                logger.info(f"[AAP] {aap_state.new_pivot_signal.upper()} @ {close:.2f} {ts}")

            # MSB 업데이트
            _pivots = list(aap.confirmed_pivots) if aap else None
            msb_state = msb.update(high=high, low=low, close=close, bar_time=ts, pivot_points=_pivots)
            msb_score = msb_state.msb_score if msb_state else 0.0
            msb_signal = msb_state.bos_signal.value if msb_state and msb_state.bos_signal else "none"
            if msb_state and hasattr(msb_state, 'bos_signal') and msb_state.bos_signal.value != "none":
                msb_signals += 1
                logger.info(f"[MSB] {msb_state.bos_signal.value.upper()} @ {close:.2f} {ts}")

            # Kalman 업데이트
            kf_state = kf.update(close=close, high=high, low=low, bar_time=ts)
            kf_score = kf_state.kalman_score if kf_state else 0.0
            kf_signal = kf_state.turning_signal if kf_state else "none"
            if kf_state and hasattr(kf_state, 'turning_signal') and kf_state.turning_signal != "none":
                kf_turns += 1
                logger.info(f"[KALMAN] {kf_state.turning_signal.upper()} @ {close:.2f} {ts}")

            # PivotScoreIntegrator 통합 계산
            int_result = integrator.compute(
                aap_score=aap_score if aap_score > 0 else None,
                msb_score=msb_score if msb_score > 0 else None,
                kalman_score=kf_score if kf_score > 0 else None,
                aap_signal=aap_signal,
                msb_signal=msb_signal,
                kalman_signal=kf_signal,
            )
            if int_result.signal != "none":
                integrated_pivots += 1
                logger.info(f"[INTEGRATOR] {int_result.signal.upper()} ({int_result.signal_strength}) @ {close:.2f} {ts} | score={int_result.adjusted_score:.3f}")

        except Exception as e:
            logger.debug(f"업데이트 실패 @ {ts}: {e}")
            continue

    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 완료")
    logger.info(f"총 봉 수: {len(df)}")
    logger.info(f"ATRAdaptivePivot 피봇: {aap_pivots}개")
    logger.info(f"MarketStructureBreak 신호: {msb_signals}개")
    logger.info(f"KalmanTurningPoint 전환: {kf_turns}개")
    logger.info(f"통합 피봇 (Integrator): {integrated_pivots}개")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="신규 4-Layer 피처 테스트")
    parser.add_argument("--file", type=str, help="CSV 파일 경로")
    parser.add_argument("--symbol", type=str, default="KP200 선물", help="심볼 이름")
    
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
    
    test_new_pivot_features(csv_path, args.symbol)
