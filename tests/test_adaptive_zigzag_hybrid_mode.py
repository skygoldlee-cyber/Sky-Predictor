"""AdaptiveZigZag 하이브리드 모드 테스트."""

from indicators.adaptive_zigzag import (
    AdaptiveZigZag,
    AdaptiveZigZagConfig,
)


class TestHybridModeThreshold:
    """하이브리드 모드 임계값 계산 테스트."""

    def test_hybrid_mode_disabled_uses_atr_only(self):
        """하이브리드 모드 비활성 시 ATR만 사용."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=False,
            use_atr_based_filtering=True,
            atr_multiplier=1.5,
            atr_period=14,
        )
        zz = AdaptiveZigZag(cfg)
        
        # ATR 기반 임계값만 계산되어야 함
        atr = 2.0
        close = 100.0
        threshold = zz._calc_threshold_pct(atr, close, 0)
        
        # ATR 기반 임계값이어야 함 (ATR/close * 100 * multiplier)
        expected_atr_threshold = (atr / close) * 100.0 * 1.5
        assert abs(threshold - expected_atr_threshold) < 0.1

    def test_hybrid_mode_enabled_mixes_atr_and_percent(self):
        """하이브리드 모드 활성 시 ATR과 퍼센트 혼합."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            use_atr_based_filtering=True,
            atr_multiplier=1.5,
            base_pct=0.3,
            atr_weight=0.5,  # 50% ATR, 50% 퍼센트
            atr_period=14,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가 (ER 계산을 위해)
        for i in range(20):
            zz._closes.append(100 + i * 0.1)
            zz._highs.append(100 + i * 0.1 + 0.5)
            zz._lows.append(100 + i * 0.1 - 0.5)
            zz._tr.append(1.0)
        
        atr = 2.0
        close = 100.0
        threshold = zz._calc_threshold_pct(atr, close, 20)
        
        # 하이브리드 임계값이 ATR과 퍼센트 사이여야 함
        atr_threshold = (atr / close) * 100.0 * 1.5
        pct_threshold = 0.3  # base_pct
        
        # 가중 평균이어야 함
        expected_hybrid = 0.5 * pct_threshold + 0.5 * atr_threshold
        assert abs(threshold - expected_hybrid) < 0.5

    def test_atr_weight_0_uses_percent_only(self):
        """atr_weight=0이면 퍼센트만 사용."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            use_atr_based_filtering=True,
            atr_multiplier=1.5,
            base_pct=0.3,
            atr_weight=0.0,  # 퍼센트만
            atr_period=14,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가
        for i in range(20):
            zz._closes.append(100 + i * 0.1)
            zz._highs.append(100 + i * 0.1 + 0.5)
            zz._lows.append(100 + i * 0.1 - 0.5)
            zz._tr.append(1.0)
        
        atr = 2.0
        close = 100.0
        threshold = zz._calc_threshold_pct(atr, close, 20)
        
        # 퍼센트 기반 임계값이어야 함 (ER 배수 포함)
        # base_pct * multiplier (ER에 따라 0.8~2.0 사이)
        assert threshold > 0.3  # ER 배수로 인해 base_pct보다 커짐

    def test_atr_weight_1_uses_atr_only(self):
        """atr_weight=1이면 ATR만 사용."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            use_atr_based_filtering=True,
            atr_multiplier=1.5,
            base_pct=0.3,
            atr_weight=1.0,  # ATR만
            atr_period=14,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가
        for i in range(20):
            zz._closes.append(100 + i * 0.1)
            zz._highs.append(100 + i * 0.1 + 0.5)
            zz._lows.append(100 + i * 0.1 - 0.5)
            zz._tr.append(1.0)
        
        atr = 2.0
        close = 100.0
        threshold = zz._calc_threshold_pct(atr, close, 20)
        
        # ATR 기반 임계값이어야 함
        expected_atr_threshold = (atr / close) * 100.0 * 1.5
        assert abs(threshold - expected_atr_threshold) < 0.1


class TestPercentThreshold:
    """퍼센트 기반 임계값 계산 테스트."""

    def test_percent_threshold_uses_base_pct(self):
        """퍼센트 임계값이 base_pct를 사용."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            base_pct=0.5,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가
        for i in range(20):
            zz._closes.append(100 + i * 0.1)
            zz._highs.append(100 + i * 0.1 + 0.5)
            zz._lows.append(100 + i * 0.1 - 0.5)
            zz._tr.append(1.0)
        
        close = 100.0
        threshold = zz._calc_percent_threshold(close, 20)
        
        # base_pct * multiplier (ER에 따라 0.8~2.0 사이)
        assert threshold > 0.5  # ER 배수로 인해 base_pct보다 커짐

    def test_percent_threshold_uses_er_multiplier(self):
        """퍼센트 임계값이 ER 배수를 사용."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            base_pct=0.3,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가 (추세 강한 데이터)
        for i in range(20):
            zz._closes.append(100 + i)  # 강한 추세
            zz._highs.append(100 + i + 0.5)
            zz._lows.append(100 + i - 0.5)
            zz._tr.append(1.0)
        
        close = 100.0
        threshold = zz._calc_percent_threshold(close, 20)
        
        # 추세 강하면 배수 커져서 임계값 커져야 함
        assert threshold > 0.3


class TestSessionMultiplierScale:
    """시간대별 배율 테스트."""

    def test_session_multiplier_table_empty_returns_1(self):
        """테이블 비어있으면 1.0 반환."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            session_multiplier_table=[],
        )
        zz = AdaptiveZigZag(cfg)
        
        # 시간 데이터 추가
        zz._bar_times = ["09:00", "09:01", "09:02"]
        
        scale = zz._get_session_multiplier_scale(0)
        assert scale == 1.0

    def test_session_multiplier_table_matches_time(self):
        """시간대별 배율이 적용됨."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            session_multiplier_table=[
                ("09:00", "09:30", 1.5),
                ("09:30", "10:30", 1.0),
            ],
        )
        zz = AdaptiveZigZag(cfg)
        
        # 시간 데이터 추가 (_bar_hhmm_map 사용)
        zz._bar_hhmm_map[0] = "09:00"
        zz._bar_hhmm_map[1] = "09:10"
        zz._bar_hhmm_map[2] = "09:20"
        
        scale = zz._get_session_multiplier_scale(1)
        assert scale == 1.5

    def test_session_multiplier_table_no_match_returns_1(self):
        """매칭되는 시간대 없으면 1.0 반환."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            session_multiplier_table=[
                ("09:00", "09:30", 1.5),
            ],
        )
        zz = AdaptiveZigZag(cfg)
        
        # 시간 데이터 추가 (매칭되지 않는 시간)
        zz._bar_times = ["10:00", "10:10", "10:20"]
        
        scale = zz._get_session_multiplier_scale(1)
        assert scale == 1.0


class TestHybridModeIntegration:
    """하이브리드 모드 통합 테스트."""

    def test_hybrid_mode_feed_data(self):
        """하이브리드 모드로 데이터 피드."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            use_atr_based_filtering=True,
            atr_multiplier=1.5,
            base_pct=0.3,
            atr_weight=0.5,
            atr_period=14,
            er_period=10,
            multiplier_min=0.8,
            multiplier_max=2.0,
            min_wave_bars=3,
            min_wave_pct=0.1,
            confirmation_bars=1,
        )
        zz = AdaptiveZigZag(cfg)
        
        # 데이터 피드 (update 메서드 사용)
        highs = [100 + i for i in range(30)]
        lows = [99 + i for i in range(30)]
        closes = [99.5 + i for i in range(30)]
        times = [f"09:{i:02d}" for i in range(30)]
        
        for h, l, c, t in zip(highs, lows, closes, times):
            zz.update(h, l, c, t)
        
        # 피봇이 생성되어야 함
        assert len(zz._all_swings) > 0

    def test_hybrid_mode_atr_weight_clipping(self):
        """atr_weight가 0~1로 클리핑됨."""
        cfg = AdaptiveZigZagConfig(
            use_hybrid_mode=True,
            use_atr_based_filtering=True,
            atr_weight=1.5,  # 1.0 초과
        )
        zz = AdaptiveZigZag(cfg)
        
        # 웜업 데이터 추가
        for i in range(20):
            zz._closes.append(100 + i * 0.1)
            zz._highs.append(100 + i * 0.1 + 0.5)
            zz._lows.append(100 + i * 0.1 - 0.5)
            zz._tr.append(1.0)
        
        atr = 2.0
        close = 100.0
        threshold = zz._calc_threshold_pct(atr, close, 20)
        
        # atr_weight가 1.0으로 클리핑되어 ATR만 사용
        expected_atr_threshold = (atr / close) * 100.0 * 1.5
        assert abs(threshold - expected_atr_threshold) < 0.1
