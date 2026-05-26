def test_adaptive_zigzag_consecutive_swings_smoke() -> None:
    """confirmation_bars=1 설정 시 충분한 단방향 이동 후 스윙이 2개 이상 확정된다.

    수정 이유:
    - 매봉 교번(±5/±6) 패턴은 pending_confirm이 매번 교체되어 확정 불가.
    - _calc_confirmation_bars()가 unknown/ranging 구간에서 base를 올리고
      소파동 보정(wave_size < ATR)으로 +1을 추가하므로 remaining >= 2.
    - 올바른 테스트: 3봉씩 단방향 이동해 remaining을 소진 후 확정 검증.
    - bar_time 필수: _pending_low_idx > _last_confirmed_bar_idx 보장용.
    """
    from indicators import AdaptiveZigZag, AdaptiveZigZagConfig
    import datetime

    zz = AdaptiveZigZag(
        AdaptiveZigZagConfig(
            atr_period=3,
            er_period=3,
            atr_multiplier=1.0,
            atr_multiplier_min=1.0,
            atr_multiplier_max=1.0,
            confirmation_bars=1,
            min_wave_bars=1,
            min_wave_pct=0.0,
            max_swings=50,
            confirmation_bars_ranging=1,   # ranging/unknown 구간도 1봉
            confirmation_bars_unknown=1,
        )
    )

    # 작은 변동폭 warmup → ATR 작게 유지 (소파동 보정 최소화)
    base_dt = datetime.datetime(2026, 4, 25, 9, 1)
    close = 100.0
    bar = 0
    for _ in range(10):
        zz.update(high=close + 0.5, low=close - 0.5, close=close,
                  bar_time=base_dt + datetime.timedelta(minutes=bar))
        bar += 1

    # _pending_low_idx > _last_confirmed_bar_idx 보장용 소폭 하락
    close -= 1.0
    zz.update(high=close + 0.1, low=close - 0.1, close=close,
              bar_time=base_dt + datetime.timedelta(minutes=bar)); bar += 1

    # 3봉씩 단방향 이동: remaining 소진 후 스윙 확정 유도
    new_swings = 0
    pattern = [(+5, 3), (-6, 3), (+5, 3), (-6, 3)]
    for (delta, repeats) in pattern:
        for _ in range(repeats):
            close += delta
            s = zz.update(high=close + 0.3, low=close - 0.3, close=close,
                          bar_time=base_dt + datetime.timedelta(minutes=bar))
            bar += 1
            if getattr(s, "new_swing_signal", "none") != "none":
                new_swings += 1

    assert new_swings >= 2, (
        f"new_swings={new_swings}. 3봉씩 단방향 이동 시 스윙이 2개 이상 확정되어야 함"
    )
