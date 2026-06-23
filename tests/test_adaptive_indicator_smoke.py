import math


def test_adaptive_indicator_manager_smoke() -> None:
    from indicators import AdaptiveIndicatorManager

    mgr = AdaptiveIndicatorManager()

    # Feed a small synthetic minute-bar series.
    # Use a gently rising close so both indicators can initialize.
    close = 400.0
    out = None
    for i in range(60):
        close += 0.1
        high = close + 0.2
        low = close - 0.2
        out = mgr.update(high=high, low=low, close=close)

    assert isinstance(out, dict)
    assert "transformer" in out
    assert "llm_context" in out
    assert "is_ready" in out

    tf = out.get("transformer")
    assert isinstance(tf, dict)

    # AdaptiveIndicatorManager는 ast_*, azz_*, cross_* 피처만 생성합니다.
    # aap_*, msb_*, kf_*, oi_*, ps_* 등은 pipeline의 다른 모듈에서 생성됩니다.
    expected_keys = set(mgr.get_transformer_feature_names())
    assert expected_keys, "AdaptiveIndicatorManager should produce non-empty transformer features"
    assert set(tf.keys()) == expected_keys, (
        f"transformer keys mismatch: got {sorted(tf.keys())}, expected {sorted(expected_keys)}"
    )
    for k, v in tf.items():
        assert isinstance(v, (int, float)), f"{k} is not numeric: {v!r}"
        assert math.isfinite(float(v)), f"{k} is not finite: {v!r}"

    llm_ctx = out.get("llm_context")
    assert isinstance(llm_ctx, str)
    assert llm_ctx.strip() != ""


def test_adaptive_indicator_disabled_path_no_import_errors() -> None:
    # Basic import smoke: the package should be importable without side effects.
    import indicators  # noqa: F401


def test_adaptive_indicator_compute_from_df_matches_sequential_cross_features() -> None:
    import pandas as pd

    from indicators import AdaptiveIndicatorManager, validate_consistency

    rows = []
    close = 400.0
    for _ in range(120):
        close += 0.15
        rows.append(
            {
                "High": close + 0.25,
                "Low": close - 0.25,
                "Close": close,
            }
        )

    df = pd.DataFrame(rows)

    mgr_seq = AdaptiveIndicatorManager()
    seq_cross = []
    n = len(rows)
    for i, row in enumerate(rows):
        out = mgr_seq.update(
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            skip_zigzag=(i == n - 1),
        )
        tf = out["transformer"]
        seq_cross.append(
            {
                "cross_trend_agreement": float(tf["cross_trend_agreement"]),
                "cross_at_support": float(tf["cross_at_support"]),
                "cross_at_resistance": float(tf["cross_at_resistance"]),
                "cross_breakout_potential": float(tf["cross_breakout_potential"]),
            }
        )

    mgr_batch = AdaptiveIndicatorManager()
    df_out = mgr_batch.compute_from_df(df)

    for i in range(len(rows)):
        for k, v in seq_cross[i].items():
            got = float(df_out[k].iloc[i])
            assert math.isfinite(got)
            assert abs(got - float(v)) <= 1e-8

    assert validate_consistency(AdaptiveIndicatorManager(), df, atol=1e-6)


def test_adaptive_supertrend_direction_flip_smoke() -> None:
    from indicators import AdaptiveSuperTrend, AdaptiveSuperTrendConfig

    st = AdaptiveSuperTrend(
        AdaptiveSuperTrendConfig(
            use_bb_correction=False,
            smooth_period=1,
            er_period=3,
            adx_period=3,
            bb_period=5,
            atr_min_period=3,
            atr_max_period=3,
            multiplier_min=2.0,
            multiplier_max=2.0,
        )
    )

    # Phase 1: down move to establish bearish direction
    close = 400.0
    last_dir = None
    for _ in range(10):
        close -= 1.5
        s = st.update(high=close + 0.2, low=close - 0.2, close=close)
        last_dir = s.direction

    assert last_dir in (-1, 1)

    # Phase 2: strong rebound should eventually flip bullish
    flipped = False
    for _ in range(30):
        close += 3.0
        s = st.update(high=close + 0.2, low=close - 0.2, close=close)
        if s.signal == "buy":
            flipped = True
            break

    assert flipped
