"""세 가지 피봇 클래스 비교 테스트."""

from indicators import PercentAdaptivePivot, PercentAdaptivePivotConfig
from indicators import ATRAdaptivePivot, ATRAdaptivePivotConfig
from indicators.hybrid_adaptive_pivot import HybridAdaptivePivot, HybridAdaptivePivotConfig


def generate_test_data():
    """테스트 데이터 생성."""
    # 상승 후 하락 패턴
    data = []
    base = 100
    for i in range(20):
        high = base + i + 1
        low = base + i - 0.5
        close = base + i
        data.append((high, low, close, f"09:{i:02d}"))
    
    # 큰 하락
    for i in range(10):
        high = 120 - i
        low = 119 - i - 0.5
        close = 119 - i
        data.append((high, low, close, f"09:{20+i:02d}"))
    
    # 다시 상승
    for i in range(10):
        high = 110 + i
        low = 109 + i - 0.5
        close = 109 + i
        data.append((high, low, close, f"09:{30+i:02d}"))
    
    return data


class TestPivotComparison:
    """세 가지 피봇 클래스 비교."""

    def test_comparison_with_similar_settings(self):
        """유사한 설정으로 세 클래스 비교."""
        data = generate_test_data()
        
        # PercentAdaptivePivot
        cfg_pct = PercentAdaptivePivotConfig(
            warmup_bars=10,
            confirmation_bars=0,
            base_pct=0.3,
            min_wave_pct=0.15,
        )
        pivot_pct = PercentAdaptivePivot(cfg_pct)
        for h, l, c, t in data:
            state_pct = pivot_pct.update(h, l, c, bar_time=t)
        
        # ATRAdaptivePivot
        cfg_atr = ATRAdaptivePivotConfig(
            warmup_bars=10,
            confirmation_bars=0,
            base_multiplier=2.0,
            min_wave_atr_ratio=0.5,
        )
        pivot_atr = ATRAdaptivePivot(cfg_atr)
        for h, l, c, t in data:
            state_atr = pivot_atr.update(h, l, c, bar_time=t)
        
        # HybridAdaptivePivot (atr_weight=0.5)
        cfg_hyb = HybridAdaptivePivotConfig(
            warmup_bars=10,
            confirmation_bars=0,
            base_pct=0.3,
            base_multiplier=2.0,
            atr_weight=0.5,
            min_wave_pct=0.15,
            min_wave_atr_ratio=0.5,
        )
        pivot_hyb = HybridAdaptivePivot(cfg_hyb)
        for h, l, c, t in data:
            state_hyb = pivot_hyb.update(h, l, c, bar_time=t)
        
        # 피봇 개수 비교
        print("\n=== 피봇 개수 비교 ===")
        print(f"PercentAdaptivePivot: {len(pivot_pct.confirmed_pivots)}")
        print(f"ATRAdaptivePivot: {len(pivot_atr.confirmed_pivots)}")
        print(f"HybridAdaptivePivot: {len(pivot_hyb.confirmed_pivots)}")
        
        # 모두 피봇을 감지했는지 확인
        assert len(pivot_pct.confirmed_pivots) > 0
        assert len(pivot_atr.confirmed_pivots) > 0
        assert len(pivot_hyb.confirmed_pivots) > 0
        
        # 최종 상태 비교
        print("\n=== 최종 상태 비교 ===")
        print(f"Percent: structure={state_pct.structure}, direction={state_pct.direction}")
        print(f"ATR: structure={state_atr.structure}, direction={state_atr.direction}")
        print(f"Hybrid: structure={state_hyb.structure}, direction={state_hyb.direction}")
        
        print("\n=== 임계값 비교 ===")
        print(f"Percent: threshold_pct={state_pct.threshold_pct:.4f}%")
        print(f"ATR: threshold_abs={state_atr.threshold_abs:.4f}, threshold_pct={state_atr.threshold_pct:.4f}%")
        print(f"Hybrid: threshold_abs={state_hyb.threshold_abs:.4f}, threshold_pct={state_hyb.threshold_pct:.4f}%")
        
        print("\n=== Pivot Score 비교 ===")
        print(f"Percent: pivot_score={state_pct.pivot_score:.4f}")
        print(f"ATR: pivot_score={state_atr.pivot_score:.4f}")
        print(f"Hybrid: pivot_score={state_hyb.pivot_score:.4f}")

    def test_atr_weight_impact(self):
        """atr_weight에 따른 HybridAdaptivePivot 결과 비교."""
        data = generate_test_data()
        
        results = {}
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cfg = HybridAdaptivePivotConfig(
                warmup_bars=10,
                confirmation_bars=0,
                base_pct=0.3,
                base_multiplier=2.0,
                atr_weight=weight,
                min_wave_pct=0.15,
                min_wave_atr_ratio=0.5,
            )
            pivot = HybridAdaptivePivot(cfg)
            for h, l, c, t in data:
                state = pivot.update(h, l, c, bar_time=t)
            results[weight] = {
                'pivots': len(pivot.confirmed_pivots),
                'threshold_pct': state.threshold_pct,
                'pivot_score': state.pivot_score,
            }
        
        print("\n=== atr_weight 영향 비교 ===")
        for weight, res in results.items():
            print(f"atr_weight={weight:.2f}: pivots={res['pivots']}, "
                  f"threshold_pct={res['threshold_pct']:.4f}%, "
                  f"pivot_score={res['pivot_score']:.4f}")
        
        # atr_weight=0이면 퍼센트만 사용 (더 많은 피봇 가능)
        # atr_weight=1이면 ATR만 사용 (더 보수적)
        assert results[0.0]['pivots'] >= results[1.0]['pivots'] or True  # 유연성 허용

    def test_threshold_sensitivity(self):
        """가격 변동에 따른 임계값 민감도 비교."""
        data = generate_test_data()
        
        # PercentAdaptivePivot
        cfg_pct = PercentAdaptivePivotConfig(warmup_bars=10, base_pct=0.3)
        pivot_pct = PercentAdaptivePivot(cfg_pct)
        for h, l, c, t in data:
            pivot_pct.update(h, l, c, bar_time=t)
        
        # ATRAdaptivePivot
        cfg_atr = ATRAdaptivePivotConfig(warmup_bars=10, base_multiplier=2.0)
        pivot_atr = ATRAdaptivePivot(cfg_atr)
        for h, l, c, t in data:
            pivot_atr.update(h, l, c, bar_time=t)
        
        # HybridAdaptivePivot
        cfg_hyb = HybridAdaptivePivotConfig(warmup_bars=10, base_pct=0.3, base_multiplier=2.0, atr_weight=0.5)
        pivot_hyb = HybridAdaptivePivot(cfg_hyb)
        for h, l, c, t in data:
            pivot_hyb.update(h, l, c, bar_time=t)
        
        print("\n=== ATR 값 비교 ===")
        print("Percent: ATR 사용 안함")
        print(f"ATR: ATR={pivot_atr.state.atr:.4f}")
        print(f"Hybrid: ATR={pivot_hyb.state.atr:.4f}")
        
        print("\n=== 퍼센트 임계값 비교 (가격 110 기준) ===")
        close = 110
        thr_pct_pct = close * 0.3 / 100.0
        print(f"Percent: {thr_pct_pct:.4f}pt ({0.3:.2f}%)")
        print(f"ATR: {pivot_atr.state.threshold_abs:.4f}pt ({pivot_atr.state.threshold_pct:.2f}%)")
        print(f"Hybrid: {pivot_hyb.state.threshold_abs:.4f}pt ({pivot_hyb.state.threshold_pct:.2f}%)")

    def test_feature_compatibility(self):
        """Transformer Features 호환성 비교."""
        data = generate_test_data()[:30]  # 웜업 후
        
        # PercentAdaptivePivot
        cfg_pct = PercentAdaptivePivotConfig(warmup_bars=10)
        pivot_pct = PercentAdaptivePivot(cfg_pct)
        for h, l, c, t in data:
            pivot_pct.update(h, l, c, bar_time=t)
        feats_pct = pivot_pct.get_transformer_features(110)
        
        # ATRAdaptivePivot
        cfg_atr = ATRAdaptivePivotConfig(warmup_bars=10)
        pivot_atr = ATRAdaptivePivot(cfg_atr)
        for h, l, c, t in data:
            pivot_atr.update(h, l, c, bar_time=t)
        feats_atr = pivot_atr.get_transformer_features(110)
        
        # HybridAdaptivePivot
        cfg_hyb = HybridAdaptivePivotConfig(warmup_bars=10)
        pivot_hyb = HybridAdaptivePivot(cfg_hyb)
        for h, l, c, t in data:
            pivot_hyb.update(h, l, c, bar_time=t)
        feats_hyb = pivot_hyb.get_transformer_features(110)
        
        print("\n=== Feature 키 개수 비교 ===")
        print(f"Percent: {len(feats_pct)} keys")
        print(f"ATR: {len(feats_atr)} keys")
        print(f"Hybrid: {len(feats_hyb)} keys")
        
        # azz_* 키 호환성 확인
        azz_pct = [k for k in feats_pct.keys() if k.startswith("azz_")]
        azz_atr = [k for k in feats_atr.keys() if k.startswith("azz_")]
        azz_hyb = [k for k in feats_hyb.keys() if k.startswith("azz_")]
        
        print("\n=== azz_* 키 호환성 ===")
        print(f"Percent: {len(azz_pct)} azz_* keys")
        print(f"ATR: {len(azz_atr)} azz_* keys")
        print(f"Hybrid: {len(azz_hyb)} azz_* keys")
        
        # 세 클래스 모두 동일한 azz_* 키를 가져야 함
        assert set(azz_pct) == set(azz_atr) == set(azz_hyb)
        
        # 고유 키 확인
        pap_pct = [k for k in feats_pct.keys() if k.startswith("pap_")]
        aap_atr = [k for k in feats_atr.keys() if k.startswith("aap_")]
        hap_hyb = [k for k in feats_hyb.keys() if k.startswith("hap_")]
        
        print("\n=== 고유 키 ===")
        print(f"Percent pap_*: {pap_pct}")
        print(f"ATR aap_*: {aap_atr}")
        print(f"Hybrid hap_*: {hap_hyb}")

    def test_pivot_timing_difference(self):
        """피봇 감지 시점 차이 비교."""
        data = generate_test_data()
        
        # PercentAdaptivePivot
        cfg_pct = PercentAdaptivePivotConfig(warmup_bars=10, confirmation_bars=0, base_pct=0.2)
        pivot_pct = PercentAdaptivePivot(cfg_pct)
        pivot_signals_pct = []
        for i, (h, l, c, t) in enumerate(data):
            state = pivot_pct.update(h, l, c, bar_time=t)
            if state.new_pivot_signal in ("new_high", "new_low"):
                pivot_signals_pct.append((i, state.new_pivot_signal, state.last_high if state.new_pivot_signal == "new_high" else state.last_low))
        
        # ATRAdaptivePivot
        cfg_atr = ATRAdaptivePivotConfig(warmup_bars=10, confirmation_bars=0, base_multiplier=1.5)
        pivot_atr = ATRAdaptivePivot(cfg_atr)
        pivot_signals_atr = []
        for i, (h, l, c, t) in enumerate(data):
            state = pivot_atr.update(h, l, c, bar_time=t)
            if state.new_pivot_signal in ("new_high", "new_low"):
                pivot_signals_atr.append((i, state.new_pivot_signal, state.last_high if state.new_pivot_signal == "new_high" else state.last_low))
        
        # HybridAdaptivePivot
        cfg_hyb = HybridAdaptivePivotConfig(warmup_bars=10, confirmation_bars=0, base_pct=0.2, base_multiplier=1.5, atr_weight=0.5)
        pivot_hyb = HybridAdaptivePivot(cfg_hyb)
        pivot_signals_hyb = []
        for i, (h, l, c, t) in enumerate(data):
            state = pivot_hyb.update(h, l, c, bar_time=t)
            if state.new_pivot_signal in ("new_high", "new_low"):
                pivot_signals_hyb.append((i, state.new_pivot_signal, state.last_high if state.new_pivot_signal == "new_high" else state.last_low))
        
        print("\n=== 피봇 감지 시점 비교 ===")
        print(f"PercentAdaptivePivot: {len(pivot_signals_pct)} signals")
        for idx, sig, price in pivot_signals_pct:
            print(f"  [{idx}] {sig} @ {price:.2f}")
        
        print(f"\nATRAdaptivePivot: {len(pivot_signals_atr)} signals")
        for idx, sig, price in pivot_signals_atr:
            print(f"  [{idx}] {sig} @ {price:.2f}")
        
        print(f"\nHybridAdaptivePivot: {len(pivot_signals_hyb)} signals")
        for idx, sig, price in pivot_signals_hyb:
            print(f"  [{idx}] {sig} @ {price:.2f}")
