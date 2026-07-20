import math

import numpy as np
import pandas as pd


def test_build_llm_context_blocks_smoke() -> None:
    from prediction.context_builder import build_llm_context

    snapshot = {
        "prediction_minutes": 5,
        "transformer": {"prob": 0.5, "signal": "HOLD"},
        "ensemble": {"prob": 0.5, "signal": "HOLD", "confidence": "LOW", "method": "transformer_only", "agreement": True},
        "market": {"current_price": 400.0},
        "orderbook": {"obi": 0.1, "spread": 0.05},
        "options": {"pcr_volume": 1.2, "iv_skew": 0.95},
        "adaptive": {"ast_direction": 1.0},
    }

    ctx = build_llm_context(snapshot=snapshot, ob_records=[], adaptive_context="hello adaptive")

    assert "[PIPELINE_INPUT]" in ctx
    assert "[OPTIONS_SNAPSHOT]" in ctx
    assert "[ADAPTIVE_INDICATORS]" in ctx

    # The options snapshot should be moved out of PIPELINE_INPUT (pop behavior).
    pipe_idx = ctx.find("[PIPELINE_INPUT]")
    opt_idx = ctx.find("[OPTIONS_SNAPSHOT]")
    assert pipe_idx >= 0 and opt_idx > pipe_idx


def test_build_sequence_dim_smoke() -> None:
    from config import FUTURE_KNOWN_DIM
    from prediction.features import ADAPT_KEYS, CD_KEYS, OB_KEYS, build_sequence, get_opt_keys

    seq_len = 10

    # Minimal OB records (only keys used by OB_KEYS are relevant).
    ob_records = [{k: 0.0 for k in OB_KEYS} for _ in range(seq_len)]

    # Minimal candle df with proper columns.
    idx = pd.date_range("2026-01-01 09:00:00", periods=2, freq="min")
    candle_df = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [1.0, 1.0],
            "Low": [1.0, 1.0],
            "Close": [1.0, 1.0],
            "Volume": [1.0, 1.0],
        },
        index=idx,
    )
    # Precompute candle features to match runtime usage.
    from prediction.features import calc_candle_features

    cdf = calc_candle_features(candle_df)

    opt_keys_v1 = list(get_opt_keys("v1"))
    opt_keys_v2 = list(get_opt_keys("v2"))
    opt_features_v1 = {k: 0.0 for k in opt_keys_v1}
    opt_features_v2 = {k: 0.0 for k in opt_keys_v2}
    adapt_features = {k: 0.0 for k in ADAPT_KEYS}

    x_no_adapt = build_sequence(
        ob_records=ob_records,
        candle_df=cdf,
        seq_len=seq_len,
        opt_features=opt_features_v1,
        adaptive_features=None,
        opt_keys_override=opt_keys_v1,
    )
    assert isinstance(x_no_adapt, np.ndarray)
    assert x_no_adapt.shape == (seq_len, len(OB_KEYS) + len(CD_KEYS) + len(opt_keys_v1) + int(FUTURE_KNOWN_DIM))

    x_with_adapt = build_sequence(
        ob_records=ob_records,
        candle_df=cdf,
        seq_len=seq_len,
        opt_features=opt_features_v1,
        adaptive_features=adapt_features,
        opt_keys_override=opt_keys_v1,
    )
    assert isinstance(x_with_adapt, np.ndarray)
    assert x_with_adapt.shape == (
        seq_len,
        len(OB_KEYS) + len(CD_KEYS) + len(opt_keys_v1) + len(ADAPT_KEYS) + int(FUTURE_KNOWN_DIM),
    )

    x_opt_v2 = build_sequence(
        ob_records=ob_records,
        candle_df=cdf,
        seq_len=seq_len,
        opt_features=opt_features_v2,
        adaptive_features=None,
        opt_keys_override=opt_keys_v2,
    )
    assert isinstance(x_opt_v2, np.ndarray)
    assert x_opt_v2.shape == (seq_len, len(OB_KEYS) + len(CD_KEYS) + len(opt_keys_v2) + int(FUTURE_KNOWN_DIM))

    # Ensure the generated arrays contain finite numbers.
    assert np.isfinite(x_no_adapt).all()
    assert np.isfinite(x_with_adapt).all()


def test_option_snapshot_smoke() -> None:
    from prediction.features.option_features import build_option_snapshot

    calls = {
        "C1": {"strike": 400.0, "price": 1.0, "volume": 10, "open_interest": 100, "iv": 0.2},
    }
    puts = {
        "P1": {"strike": 400.0, "price": 1.0, "volume": 12, "open_interest": 120, "iv": 0.22},
    }

    snap = build_option_snapshot(calls, puts, 400.0, option_feature_set="v1")
    assert isinstance(snap, dict)
    assert "pcr_volume" in snap
    assert "iv_skew" in snap

    snap_v2 = build_option_snapshot(calls, puts, 400.0, option_feature_set="v2")
    assert isinstance(snap_v2, dict)
    assert "optm_call_ret" in snap_v2

    for k, v in snap.items():
        if isinstance(v, (int, float)):
            assert math.isfinite(float(v))


def test_prediction_pipeline_init_smoke() -> None:
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(use_llm=False)
    assert int(p.prediction_minutes) > 0
