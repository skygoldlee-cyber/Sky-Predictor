"""compute_rule_based_probability / adaptive heuristic parse 스모크."""

from __future__ import annotations

from prediction.mixins.adaptive_mixin import _parse_adaptive_heuristic_features
from prediction.predictor import _merge_rule_based_weights, compute_rule_based_probability


def test_merge_rule_based_weights_defaults() -> None:
    m = _merge_rule_based_weights(None)
    assert m["w_obi"] == 0.55


def test_compute_rule_based_neutral() -> None:
    snap = {"obi": 0.0, "spread": 0.0, "level1_ratio": 0.0, "bid_slope": 0.0, "offer_slope": 0.0}
    p, sp = compute_rule_based_probability(
        snap,
        None,
        weights=_merge_rule_based_weights(None),
        mom_multiplier=1.0,
        confidence_spread_max_for_high=1.0,
    )
    assert abs(p - 0.5) < 1e-6
    assert sp == 0.0


def test_parse_adaptive_heuristic_features() -> None:
    ast_dir, sig, azz = _parse_adaptive_heuristic_features(
        {"ast_direction": 1.0, "ast_signal": 0.0, "azz_new_swing": 1.0}
    )
    assert ast_dir == 1
    assert sig == 0.0
    assert azz == 1
