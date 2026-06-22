import logging

import pytest

from config import TRCode
from data.tick_processor import RealTimeTickProcessor
from prediction import PredictionPipeline
from prediction.features import calc_orderbook_features

logger = logging.getLogger(__name__)


@pytest.mark.unit
def test_tick_processor() -> None:
    processor = RealTimeTickProcessor(default_futures_minutes=120, default_options_minutes=120)

    futures_tick = {
        "trcode": TRCode.FUTURES.value,
        "symbol": "A016XXXX",
        "tick": {
            "price": "430.50",
            "volume": "1000",
            "chetime": "130430",
            "k200jisu": "430.40",
            "openyak": "100",
            "bidho1": "430.45",
            "offerho1": "430.55",
        },
    }
    processor.process_tick(futures_tick)

    assert processor.get_current_price() > 0


def test_predictor_init() -> None:
    predictor = PredictionPipeline(prediction_minutes=5, use_llm=False)
    assert int(predictor.prediction_minutes) == 5


def test_fh0_schema_parsing() -> None:
    fh0_body = {
        "expcode": "101W9000",
        "chetime": "101532",
        "price": 375.45,
        "offerho": 375.50,
        "bidho": 375.45,
        "offerrem": 120,
        "bidrem": 85,
    }

    feat = calc_orderbook_features(fh0_body)
    assert feat.get("_invalid") is None
    assert float(feat.get("spread") or 0.0) >= 0.0


def test_fh0_depth_parsing() -> None:
    fh0_body = {
        "expcode": "101W9000",
        "chetime": "101533",
        "price": 375.45,
        "offerho1": 375.50,
        "bidho1": 375.45,
        "offerrem1": 120,
        "offerrem2": 150,
        "offerrem3": 180,
        "offerrem4": 210,
        "offerrem5": 240,
        "bidrem1": 85,
        "bidrem2": 95,
        "bidrem3": 110,
        "bidrem4": 130,
        "bidrem5": 160,
    }

    feat = calc_orderbook_features(fh0_body)
    assert feat.get("_invalid") is None
    assert float(feat.get("totbidrem") or 0.0) > 0.0
    assert float(feat.get("totofferrem") or 0.0) > 0.0
    assert float(feat.get("bid_slope") or 0.0) != 0.0
    assert float(feat.get("offer_slope") or 0.0) != 0.0


def test_fh0_invalid_no_bid_ask() -> None:
    fh0_body = {"expcode": "101W9000", "price": 375.45}
    feat = calc_orderbook_features(fh0_body)
    assert feat.get("_invalid") is True
    assert float(feat.get("obi") or 0.0) == 0.0


def test_fh0_buffering_prediction_loop_smoke() -> None:
    pipeline = PredictionPipeline(min_minute_bars_required=1, seq_len=5, use_llm=False)

    fc0_tick = {
        "trcode": TRCode.FUTURES.value,
        "symbol": "A016XXXX",
        "tick": {
            "price": "430.50",
            "volume": "1000",
            "chetime": "130430",
            "k200jisu": "430.40",
            "openyak": "100",
            "bidho1": "430.45",
            "offerho1": "430.55",
        },
    }
    pipeline.add_realtime_tick(fc0_tick)

    fh0_tick = {
        "trcode": TRCode.FUTURES_BOOK.value,
        "symbol": "101V3000",
        "tick": {
            "hotime": "130431",
            "totofferrem": "1500",
            "totbidrem": "1200",
            "offerho1": "430.55",
            "bidho1": "430.45",
            **{f"offerrem{i}": str(100 + i) for i in range(1, 6)},
            **{f"bidrem{i}": str(80 + i) for i in range(1, 6)},
        },
    }
    pipeline.add_realtime_tick(fh0_tick)

    assert len(getattr(pipeline, "_ob_records", []) or []) >= 1

    out = pipeline.get_prediction()
    assert isinstance(out, dict)
    assert "error" not in out


def test_full_project_smoke() -> None:
    """Umbrella smoke test.

    This test intentionally reuses the dedicated smoke-test modules so that running
    only this file still validates the core project paths.
    """

    from test_adaptive_indicator_smoke import (
        test_adaptive_indicator_disabled_path_no_import_errors,
        test_adaptive_indicator_manager_smoke,
    )
    from test_prediction_smoke import (
        test_build_llm_context_blocks_smoke,
        test_build_sequence_dim_smoke,
        test_option_snapshot_smoke,
        test_prediction_pipeline_init_smoke,
    )

    test_adaptive_indicator_disabled_path_no_import_errors()
    test_adaptive_indicator_manager_smoke()

    test_prediction_pipeline_init_smoke()
    test_build_sequence_dim_smoke()
    test_option_snapshot_smoke()
    test_build_llm_context_blocks_smoke()
