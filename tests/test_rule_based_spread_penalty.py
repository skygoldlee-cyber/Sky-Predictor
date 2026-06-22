import math

import numpy as np


def _mk_seq(*, spread: float) -> np.ndarray:
    from prediction.features import CD_KEYS, OB_KEYS

    dim = int(len(OB_KEYS) + len(CD_KEYS))
    x = np.zeros((1, dim), dtype=np.float32)

    # OB layout: [obi, spread, level1_ratio, ...]
    x[0, 0] = 0.0
    x[0, 1] = float(spread)
    x[0, 2] = 0.0

    # Candle features (ret3, vol_accel) are read by index; keep defaults.
    return x


def test_rule_based_spread_penalty_monotonic() -> None:
    from prediction.predictor import TransformerPredictor

    p = TransformerPredictor(weights_path=None, confidence_spread_max_for_high=0.05)

    r0 = p._rule_based({"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0}, _mk_seq(spread=0.0))
    r1 = p._rule_based({"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0}, _mk_seq(spread=0.05))
    r2 = p._rule_based({"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0}, _mk_seq(spread=0.10))

    assert math.isfinite(float(r0.prob))
    assert math.isfinite(float(r1.prob))
    assert math.isfinite(float(r2.prob))

    assert float(r0.prob) >= float(r1.prob) >= float(r2.prob)


def test_rule_based_spread_penalty_cap() -> None:
    from prediction.predictor import TransformerPredictor

    p = TransformerPredictor(weights_path=None, confidence_spread_max_for_high=0.05)

    r_hi = p._rule_based({"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0}, _mk_seq(spread=1.0))
    r_hi2 = p._rule_based({"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0}, _mk_seq(spread=2.0))

    # Penalty is capped at 0.25, so probability should not keep decreasing after cap is hit.
    assert float(r_hi.prob) == float(r_hi2.prob)
