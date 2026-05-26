"""검증 세트(Brier/ECE)로 튜닝할 때 참고하는 `config.json` `prediction` 키 목록.

런타임 로직은 포함하지 않으며, `calibration_report`·문서와 함께 쓴다.
"""

from __future__ import annotations

# (키, 한 줄 설명)
TUNABLE_PREDICTION_KEYS: tuple[tuple[str, str], ...] = (
    ("buy_threshold", "BUY 임계 (앙상블 확률 ≥ 이 값)"),
    ("sell_threshold", "SELL 임계 (확률 ≤ 이 값)"),
    ("confidence_high_margin", "|p−0.5| ≥ 이 값이면 HIGH 후보 (스프레드 조건 병행)"),
    ("confidence_mid_margin", "MEDIUM 최소 마진"),
    ("confidence_spread_max_for_high", "HIGH일 때 허용 최대 스프레드(pt)"),
    ("confidence_conformal_width_max_for_high", "Conformal 구간 폭 ≥ 이 값이면 HIGH 불가"),
    ("confidence_conformal_width_max_for_medium", "Conformal 구간 폭 ≥ 이 값이면 LOW"),
    ("disagreement_hold_prob_diff_max", "모델 간 최대 |Δp| ≥ 이면 HOLD (기본)"),
    ("disagreement_hold_prob_diff_max_by_regime", "레짐별 불일치 임계 (dict, 선택)"),
    ("ensemble_agreement_prob_diff_max", "합의 시 신뢰 상향: |Δp| < 이면 후보"),
    ("pcr_atm_strikes_each_side", "PCR 합산: ATM 기준 위·아래 각 N행사가(0=ATM 한 줄만, 기본 5)"),
)


def format_tunable_keys_reference() -> str:
    """로그/리포트에 붙일 수 있는 텍스트 블록."""
    lines = ["[prediction] 튜닝 후보 키 (검증 세트 Brier/ECE와 함께 조정)"]
    for k, desc in TUNABLE_PREDICTION_KEYS:
        lines.append(f"  - {k}: {desc}")
    return "\n".join(lines)
