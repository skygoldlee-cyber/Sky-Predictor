"""LLM context/prompt builder for the prediction pipeline.

This module turns the pipeline snapshot and recent orderbook history into:
- a compact machine-readable context block
- a strict JSON-output instruction prompt
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import LLM_OUTPUT_SCHEMA

logger = logging.getLogger(__name__)


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Best-effort float conversion used for context summaries."""
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _summarize_orderbook(ob_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize recent orderbook feature records.

    Notes:
    - `ob_records` is expected to be time-ordered (oldest -> newest).
    - `delta` is computed as `last - first`.
    """
    recs = [r for r in (ob_records or []) if isinstance(r, dict)]
    if not recs:
        return {}

    keys = ("obi", "spread", "level1_ratio", "totbidrem", "totofferrem")

    def _series(k: str) -> List[float]:
        """Extract a numeric series for key `k` from records."""
        out: List[float] = []
        for r in recs:
            out.append(_safe_float(r.get(k), 0.0))
        return out

    def _mean(xs: List[float]) -> float:
        """Compute mean with empty-list safety."""
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs: List[float]) -> float:
        """Compute std-dev with empty-list safety."""
        if not xs:
            return 0.0
        m = _mean(xs)
        try:
            v = sum((float(x) - float(m)) ** 2 for x in xs) / float(len(xs))
            return float(v ** 0.5)
        except Exception:
            return 0.0

    last = recs[-1]
    first = recs[0]

    out: Dict[str, Any] = {
        "count": int(len(recs)),
        "last": {k: _safe_float(last.get(k), 0.0) for k in keys},
        "mean": {},
        "std": {},
        "delta": {},
        "trend": {},
    }

    for k in keys:
        xs = _series(k)
        out["mean"][k] = _mean(xs)
        out["std"][k] = _std(xs)
        out["delta"][k] = _safe_float(last.get(k), 0.0) - _safe_float(first.get(k), 0.0)
        try:
            d = float(out["delta"][k])
            if abs(d) <= 1e-9:
                out["trend"][k] = "flat"
            else:
                out["trend"][k] = "up" if d > 0 else "down"
        except Exception:
            out["trend"][k] = "flat"

    return out


def _describe_parity_divergence(opt_snap: Dict[str, Any]) -> str:
    """만기주 콜-풋 패리티 이탈 수치를 LLM이 이해할 수 있는 자연어로 변환한다.

    dte_weight_norm < 0.1 (만기 7일 이상)이면 빈 문자열을 반환한다.
    의미 있는 이탈 신호가 없을 때도 빈 문자열을 반환한다.
    """
    try:
        score   = float(opt_snap.get("parity_divergence_score") or 0.0)
        dte_w   = float(opt_snap.get("dte_weight_norm") or 0.0)
        spread  = float(opt_snap.get("parity_spread_pct") or 0.0)
        delta_p = float(opt_snap.get("call_delta_proxy") or 0.5)
        ret_diff = float(opt_snap.get("call_vs_fut_ret_diff") or 0.0)
    except Exception:
        return ""

    # 만기 7일 이상(dte_w < 0.1)이면 신호 무의미
    if dte_w < 0.1:
        return ""

    lines: list[str] = []

    # 종합 이탈 방향
    if abs(score) >= 0.3:
        direction = "콜 과매도(저평가)" if score < 0 else "콜 과매수(고평가)"
        lines.append(
            f"패리티 이탈 감지: {direction} "
            f"(score={score:.2f}, dte_weight={dte_w:.3f})"
        )

    # 패리티 스프레드
    if abs(spread) >= 0.1:
        lines.append(
            f"  패리티 스프레드: {spread:+.2f}% "
            "(C-P 실제값 - 이론값, 0에 가까울수록 균형)"
        )

    # 델타 비대칭
    if abs(delta_p - 0.5) >= 0.05:
        lines.append(
            f"  콜 델타 비대칭: {delta_p:.3f} "
            "(ATM 이론값=0.50, 이탈 클수록 한쪽 방향 포지션 쏠림)"
        )

    # 수익률 추종 이탈
    if abs(ret_diff) >= 0.002:
        direction2 = "과소추종(콜 저평가 가능)" if ret_diff < 0 else "과다추종(콜 고평가 가능)"
        lines.append(
            f"  콜 수익률 추종 이탈: {ret_diff:+.4f} ({direction2})"
        )

    return "\n".join(lines)


def _describe_premium_bleed(opt_snap: Dict[str, Any]) -> str:
    """만기주 선물 상승 중 옵션 프리미엄 수축 수치를 LLM이 이해할 수 있는 자연어로 변환한다.

    premium_bleed_score가 없거나 dte_weight_norm < 0.1 (만기 7일 이상)이면 빈 문자열을 반환한다.
    선물 수익률 방향과 프리미엄 수축 동시 발생 여부를 명시한다.
    """
    try:
        score      = float(opt_snap.get("premium_bleed_score") or 0.0)
        dte_w      = float(opt_snap.get("dte_weight_norm") or 0.0)
        decay      = float(opt_snap.get("straddle_decay_vs_fut") or 0.0)
        iv_crush   = float(opt_snap.get("iv_crush_proxy") or 0.0)
        fut_ret    = float(opt_snap.get("fut_ret") or 0.0)
        straddle_n = float(opt_snap.get("straddle_now") or 0.0)
        straddle_p = float(opt_snap.get("straddle_prev") or 0.0)
    except Exception:
        return ""

    # 만기 7일 이상이거나 prev 없으면 신호 무의미
    if dte_w < 0.1 or straddle_p <= 0.0:
        return ""

    # 의미 있는 신호 임계값: |score| >= 0.3
    if abs(score) < 0.3:
        return ""

    lines: list[str] = []

    fut_dir = "상승" if fut_ret > 0 else ("하락" if fut_ret < 0 else "횡보")
    bleed_dir = "수축(프리미엄 블리드)" if score < 0 else "팽창(IV 급등)"

    lines.append(
        f"프리미엄 블리드 감지: 선물 {fut_dir} 중 스트래들 {bleed_dir} "
        f"(score={score:.2f}, dte_weight={dte_w:.3f})"
    )

    if abs(fut_ret) >= 0.0003:
        lines.append(f"  선물 수익률: {fut_ret:+.4%}")

    if straddle_p > 0.0 and abs(decay) >= 0.002:
        straddle_ret_pct = (straddle_n - straddle_p) / straddle_p * 100.0
        lines.append(
            f"  스트래들 변화: {straddle_ret_pct:+.2f}% "
            f"({straddle_p:.2f} → {straddle_n:.2f}) "
            f"[decay_vs_fut={decay:+.4f}]"
        )

    if abs(iv_crush) >= 0.01:
        iv_dir = "IV 급락(Crush)" if iv_crush < 0 else "IV 급등"
        lines.append(f"  {iv_dir}: {iv_crush:+.2%} (BS ATM IV 근사)")

    if score < -0.5 and fut_ret > 0:
        lines.append(
            "  → 선물 상승에도 옵션 프리미엄이 수축 중: "
            "Theta 가속/롤오버/IV Crush 가능. 방향성 배팅보다 관망 권장."
        )
    elif score > 0.5:
        lines.append(
            "  → 프리미엄 급팽창: IV 급등 또는 대형 이벤트 예고. "
            "방향성 예측 어려운 구간."
        )

    return "\n".join(lines)


def _describe_otm_premium(opt_snap: Dict[str, Any]) -> str:
    """OTM 프리미엄 변화율을 LLM이 이해할 수 있는 자연어로 변환한다.

    build_option_snapshot()이 모든 fs에서 저장하는 'otm_premium' 키를 사용한다.
    open_price가 주입되지 않은 경우(장 초기 등) call_otm_count=0이므로 섹션이 생략된다.

    Returns:
        자연어 설명 문자열. 데이터 없으면 빈 문자열.
    """
    otm = (opt_snap or {}).get("otm_premium")
    if not isinstance(otm, dict):
        return ""

    c_chg = otm.get("call_otm_prem_chg")   # float or None
    p_chg = otm.get("put_otm_prem_chg")    # float or None
    c_cnt = int(otm.get("call_otm_count") or 0)
    p_cnt = int(otm.get("put_otm_count")  or 0)

    if c_cnt == 0 and p_cnt == 0:
        return ""

    # [OTM-WGT] 단순 평균으로 변경됨 — "가중 평균" 표기 제거
    c_avg_open  = float(otm.get("call_otm_avg_open") or 0.0)
    p_avg_open  = float(otm.get("put_otm_avg_open")  or 0.0)
    c_str_range = otm.get("call_strike_range") or []
    p_str_range = otm.get("put_strike_range")  or []

    def _range_str(rng: list) -> str:
        if len(rng) == 2:
            return f"{rng[0]:.1f}~{rng[1]:.1f}"
        if len(rng) == 1:
            return f"{rng[0]:.1f}"
        return ""

    lines: list = []
    if c_chg is not None and c_cnt > 0:
        c_dir = "상승" if c_chg > 0 else ("하락" if c_chg < 0 else "보합")
        _c_meta = f"{c_cnt}종목"
        if _range_str(c_str_range):
            _c_meta += f" 행사가 {_range_str(c_str_range)}"
        if c_avg_open > 0.0:
            _c_meta += f" 평균시가 {c_avg_open:.2f}pt"
        lines.append(
            f"콜OTM평균: {c_chg:+.1%} ({_c_meta}) — "
            f"콜 프리미엄 {c_dir}. "
            + ("콜 OTM 수요 증가, 상승 베팅 확대 신호." if c_chg > 0.05
               else "콜 프리미엄 수축, 상승 모멘텀 약화 또는 Theta 소멸." if c_chg < -0.05
               else "콜 프리미엄 소폭 변화, 방향성 불명확.")
        )
    if p_chg is not None and p_cnt > 0:
        p_dir = "상승" if p_chg > 0 else ("하락" if p_chg < 0 else "보합")
        _p_meta = f"{p_cnt}종목"
        if _range_str(p_str_range):
            _p_meta += f" 행사가 {_range_str(p_str_range)}"
        if p_avg_open > 0.0:
            _p_meta += f" 평균시가 {p_avg_open:.2f}pt"
        lines.append(
            f"풋OTM평균: {p_chg:+.1%} ({_p_meta}) — "
            f"풋 프리미엄 {p_dir}. "
            + ("풋 OTM 수요 증가, 하락 헤지 확대 또는 하방 베팅 신호." if p_chg > 0.05
               else "풋 프리미엄 수축, 하락 우려 감소 또는 Theta 소멸." if p_chg < -0.05
               else "풋 프리미엄 소폭 변화, 방향성 불명확.")
        )

    # 콜/풋 방향 합산 해석
    if c_chg is not None and p_chg is not None and c_cnt > 0 and p_cnt > 0:
        if c_chg > 0.03 and p_chg < -0.03:
            lines.append("종합: 콜 팽창 + 풋 수축 → 강세 옵션 플로우.")
        elif c_chg < -0.03 and p_chg > 0.03:
            lines.append("종합: 콜 수축 + 풋 팽창 → 약세 옵션 플로우.")
        elif c_chg < -0.03 and p_chg < -0.03:
            lines.append("종합: 콜·풋 모두 수축 → IV Crush 또는 Theta 급가속 가능.")

    return "\n".join(lines)


def _describe_price_level_scan(opt_snap: Dict[str, Any]) -> str:
    """옵션 가격 레벨 터치 스캔 결과를 LLM 컨텍스트용 자연어로 변환한다.

    build_option_snapshot()이 모든 fs에서 저장하는'_price_level_scan' 키를 사용.
    터치 항목이 없으면 빈 문자열을 반환한다.

    출력 예시::
        탐색 레벨: 1.20 / 2.50 / 3.50 / 4.85 / 5.50  (완전 일치)
        콜 고가 터치:
          - 행사가 285.0  고가=2.50  @0911
        풋 저가 터치:
          - 행사가 280.0  저가=1.20  @1023
        시사점: 콜 고가 레벨 정확 일치 → 매도 압력 집중 가능.
    """
    scan = opt_snap.get("_price_level_scan")
    if not isinstance(scan, dict) or not scan.get("has_hit"):
        return ""

    call_hits: list = scan.get("call_hits") or []
    put_hits:  list = scan.get("put_hits")  or []
    levels:    list = scan.get("levels_used") or []

    lines: list = []

    lv_str = " / ".join(f"{lv:.2f}" for lv in sorted(levels))
    lines.append(f"탐색 레벨: {lv_str}  (완전 일치)")

    _FIELD_KR = {"high": "고가", "low": "저가", "price": "현재가"}

    def _render(hits: list, label: str) -> None:
        if not hits:
            return
        by_field: Dict[str, list] = {}
        for h in hits:
            by_field.setdefault(h["field"], []).append(h)
        for fld in ("high", "low", "price"):
            grp = by_field.get(fld)
            if not grp:
                continue
            lines.append(f"{label} {_FIELD_KR.get(fld, fld)} 터치:")
            for h in grp:
                t_str = h.get("time_str") or ""
                time_part = f"  @{t_str}" if t_str else ""
                lines.append(
                    f"  - 행사가 {h['strike']:.1f}  "
                    f"{_FIELD_KR.get(fld, fld)}={h['value']:.2f}"
                    f"{time_part}"
                )

    _render(call_hits, "콜")
    _render(put_hits, "풋")

    implications: list = []
    if any(h["field"] == "high" for h in call_hits):
        implications.append("콜 고가 레벨 정확 일치 → 해당 행사가 부근 매도 압력·저항 집중 가능.")
    if any(h["field"] == "low" for h in put_hits):
        implications.append("풋 저가 레벨 정확 일치 → 해당 행사가 부근 하방 지지선 테스트 신호.")
    if any(h["field"] == "low" for h in call_hits):
        implications.append("콜 저가 레벨 정확 일치 → 프리미엄 수축 진행 중 가능.")
    if any(h["field"] == "high" for h in put_hits):
        implications.append("풋 고가 레벨 정확 일치 → 풋 프리미엄 급등·하방 베팅 강화 신호.")

    hit_lv_c = scan.get("hit_levels_call") or []
    hit_lv_p = scan.get("hit_levels_put")  or []
    if len(hit_lv_c) >= 2 or len(hit_lv_p) >= 2:
        implications.append("다수 레벨 동시 터치 → 장중 변동성 확대 구간 가능성 상승.")
    if hit_lv_c and hit_lv_p and hit_lv_c != hit_lv_p:
        implications.append("콜·풋 서로 다른 레벨 터치 → 방향성 탐색 혹은 양방향 베팅 병존.")

    if implications:
        lines.append("시사점: " + implications[0])
        for imp in implications[1:]:
            lines.append("        " + imp)

    return "\n".join(lines)

def _describe_oi_levels(opt_snap: Dict[str, Any]) -> str:
    """OI 기반 지지저항 레벨 수치를 LLM이 이해할 수 있는 자연어로 변환한다.

    build_option_snapshot()이 모든 fs에서 저장하는 '_oi_levels' 키를 사용하므로
    v1~v5 전 피처셋에서 동작한다. OI 데이터가 없으면 빈 문자열을 반환한다.

    표시 조건:
        - oi_range_pct > 0 (OI 데이터 존재)
        - call_oi_peak > 0 또는 put_oi_peak > 0
    """
    # _oi_levels 키 우선 탐색, 없으면 opt_snap 직접 사용 (v5 호환)
    oi = opt_snap.get("_oi_levels")
    if not isinstance(oi, dict) or not oi:
        # v5: _oi_levels 없이 opt_snap에 직접 노출된 경우 fallback
        oi = opt_snap

    try:
        call_peak  = float(oi.get("call_oi_peak") or 0.0)
        put_peak   = float(oi.get("put_oi_peak") or 0.0)
        call_dist  = float(oi.get("dist_to_call_peak") or 0.0)
        put_dist   = float(oi.get("dist_to_put_peak") or 0.0)
        center_d   = float(oi.get("oi_center_dist_pct") or 0.0)
        range_pct  = float(oi.get("oi_range_pct") or 0.0)
        call_conc  = float(oi.get("call_oi_peak_norm") or 0.0)
        put_conc   = float(oi.get("put_oi_peak_norm") or 0.0)
        above_vt   = float(oi.get("above_vol_trigger") if oi.get("above_vol_trigger") is not None else 1.0)
        zgd        = float(oi.get("zero_gamma_dist_pct") or 0.0)
        vt_strike  = float(oi.get("vol_trigger_strike") or 0.0)
        zg_strike  = float(oi.get("zero_gamma_strike") or 0.0)
        iv_range   = float(oi.get("peak_search_range_used") or 0.0)
    except Exception:
        return ""

    # OI 데이터가 없으면 생략
    if range_pct <= 0.0 and call_peak <= 0.0 and put_peak <= 0.0:
        return ""

    lines: list[str] = []

    # 저항 레벨
    if call_peak > 0.0:
        if call_dist > 0:
            lines.append(
                f"OI 저항(Call Peak): {call_peak:.2f}pt  "
                f"현재가 대비 +{call_dist:.2f}%  집중도 {call_conc:.1%}"
            )
        else:
            lines.append(
                f"OI 저항(Call Peak): {call_peak:.2f}pt  "
                f"현재가 대비 {call_dist:.2f}% [돌파 상태]  집중도 {call_conc:.1%}"
            )

    # 지지 레벨
    if put_peak > 0.0:
        if put_dist > 0:
            lines.append(
                f"OI 지지(Put Peak):  {put_peak:.2f}pt  "
                f"현재가 대비 -{put_dist:.2f}%  집중도 {put_conc:.1%}"
            )
        else:
            lines.append(
                f"OI 지지(Put Peak):  {put_peak:.2f}pt  "
                f"현재가 대비 {put_dist:.2f}% [이탈 상태]  집중도 {put_conc:.1%}"
            )

    # 박스 구조 + IV 기반 탐색 범위
    if range_pct > 0.0:
        iv_range_str = f"  (탐색반경 ±{iv_range:.1f}pt)" if iv_range > 0.0 else ""
        lines.append(
            f"  OI 박스폭: {range_pct:.2f}%  "
            f"중심 대비: {center_d:+.2f}% "
            f"({'현재가가 중심 위' if center_d > 0 else '현재가가 중심 아래' if center_d < 0 else '중심 일치'})"
            f"{iv_range_str}"
        )

    # Volatility Trigger 레짐
    if vt_strike > 0.0:
        if above_vt >= 1.0:
            lines.append(
                f"  레짐: Vol Trigger({vt_strike:.2f}pt) 상방 — "
                "Dealer Long Gamma (레인지 안정권, 역추세 헤지 흐름)"
            )
        else:
            lines.append(
                f"  레짐: Vol Trigger({vt_strike:.2f}pt) 하방 ⚠️ — "
                "Dealer Short Gamma (추세 가속 가능, 방향성 배팅 주의)"
            )
    else:
        # Vol Trigger 미산출 시 net_gamma_proxy 기반 레짐만 표시
        if above_vt >= 1.0:
            lines.append("  레짐: Dealer Long Gamma (안정권)")
        else:
            lines.append("  레짐: Dealer Short Gamma ⚠️ (추세 가속 가능)")

    # Zero Gamma Level 근접 경고
    if zg_strike > 0.0 and abs(zgd) < 0.3:
        lines.append(
            f"  ⚠️ Zero Gamma Level({zg_strike:.2f}pt) 근접 "
            f"(거리 {zgd:+.2f}%) — 딜러 감마 방향 반전 임박"
        )

    return "\n".join(lines)


def _describe_amplitude(amplitude: Dict[str, Any]) -> str:
    """진폭 스냅샷을 LLM이 이해할 수 있는 자연어로 변환한다.

    result["amplitude"] 키(calc_expected_amplitude 반환값)를 받아
    예상 진폭, 실현 진폭, 소진 정도, 남은 여력을 서술한다.

    소진율이 낮으면(0.3 미만) 추가 이동 여력이 충분함을,
    높으면(0.8 초과) 진폭이 대부분 소진되어 추가 추세 둔화 가능성을 시사.
    """
    if not isinstance(amplitude, dict) or not amplitude:
        return ""

    try:
        exp_pt   = float(amplitude.get("expected_amplitude_pt")  or 0.0)
        real_pt  = float(amplitude.get("realized_hl_range_pt")   or 0.0)
        exhaust  = float(amplitude.get("amplitude_exhaustion")    or 0.0)
        remain   = float(amplitude.get("remaining_amplitude_pt")  or 0.0)
        # [FIX-AMP-4] 방향별 잔여 진폭
        up_rem   = float(amplitude.get("upside_remaining_pt")     or 0.0)
        dn_rem   = float(amplitude.get("downside_remaining_pt")   or 0.0)
        open_d   = float(amplitude.get("open_dist_pct")           or 0.0)
        s_open   = float(amplitude.get("session_open")            or 0.0)
        source   = str(amplitude.get("_amplitude_source")         or "")
        oi_w     = float(amplitude.get("_oi_weight")              or 0.0)
        iv_only  = float(amplitude.get("_iv_amplitude_pt")        or 0.0)
        oi_only  = float(amplitude.get("_oi_amplitude_pt")        or 0.0)
        ema_val  = float(amplitude.get("_realized_amplitude_ema") or 0.0)
        ema_blended = bool(amplitude.get("_ema_blended", False))
    except Exception:
        return ""

    if exp_pt <= 0.0 and real_pt <= 0.0:
        return ""

    lines: list = []

    # 예상 진폭 — OI 혼합 + EMA 보정 여부에 따라 서술 분기
    if exp_pt > 0.0:
        if oi_w > 0.0 and oi_only > 0.0 and iv_only > 0.0:
            _base_desc = f"IV {iv_only:.1f}pt × {1-oi_w:.0%} + OI박스 {oi_only:.1f}pt × {oi_w:.0%}"
            if ema_blended and ema_val > 0.0:
                lines.append(
                    f"당일 예상 진폭(H-L): {exp_pt:.1f}pt "
                    f"[{_base_desc} → EMA보정 {ema_val:.1f}pt 혼합]"
                )
            else:
                lines.append(f"당일 예상 진폭(H-L): {exp_pt:.1f}pt [{_base_desc}]")
        elif iv_only > 0.0:
            if ema_blended and ema_val > 0.0:
                lines.append(
                    f"당일 예상 진폭(H-L): {exp_pt:.1f}pt "
                    f"[IV 단독 → EMA보정 {ema_val:.1f}pt 혼합]"
                )
            else:
                lines.append(f"당일 예상 진폭(H-L): {exp_pt:.1f}pt [IV 단독, OI 미반영]")
        else:
            lines.append(f"당일 예상 진폭(H-L): {exp_pt:.1f}pt")
    else:
        lines.append("IV/OI 데이터 없음 — 예상 진폭 산출 불가")

    # 실현 진폭
    # Medium-08: dte_weight_norm 기반 동적 소진율 임계값 적용
    try:
        _dte_w_desc = float(amplitude.get("_dte_weight_norm_ref") or 0.0)
    except Exception:
        _dte_w_desc = 0.0
    if _dte_w_desc >= 0.5:
        _warn_thres, _half_thres = 0.60, 0.35
    elif _dte_w_desc >= 0.2:
        _warn_thres, _half_thres = 0.75, 0.40
    else:
        _warn_thres, _half_thres = 0.80, 0.50

    if real_pt > 0.0:
        lines.append(f"장중 실현 진폭(H-L): {real_pt:.1f}pt")
        if exp_pt > 0.0:
            if exhaust >= _warn_thres:
                lines.append(
                    f"  ⚠️ 진폭 소진율 {exhaust:.0%} — 예상 범위의 {exhaust:.0%} 소진. "
                    f"추가 급등락 여력 제한적 (잔여 {remain:.1f}pt)"
                )
            elif exhaust >= _half_thres:
                lines.append(
                    f"  진폭 소진율 {exhaust:.0%} — 절반 이상 소진. "
                    f"잔여 이동 여력 {remain:.1f}pt"
                )
            else:
                lines.append(
                    f"  진폭 소진율 {exhaust:.0%} — 아직 여력 충분. "
                    f"잔여 이동 여력 {remain:.1f}pt"
                )
            # [FIX-AMP-4] 방향별 잔여 진폭 서술 (시가 기준 대칭 가정)
            # upside/downside 둘 다 0이면 legacy remaining만 사용하므로 표시 생략
            if up_rem > 0.0 or dn_rem > 0.0:
                _dir_parts = []
                if up_rem > 0.0:
                    _dir_parts.append(f"위쪽 {up_rem:.1f}pt")
                if dn_rem > 0.0:
                    _dir_parts.append(f"아래쪽 {dn_rem:.1f}pt")
                lines.append(
                    f"  방향별 잔여: {' / '.join(_dir_parts)} "
                    f"(시가 {s_open:.2f}pt 대칭 기준)"
                )
        if exhaust > 1.0:
            lines.append(
                f"  ⚡ 예상 진폭 초과 달성 ({exhaust:.0%}) — 이례적 변동성 구간"
            )

    # 시가 대비 현재 위치
    if s_open > 0.0:
        direction = "시가 위" if open_d > 0 else ("시가 아래" if open_d < 0 else "시가 동일")
        lines.append(
            f"현재가 vs 시가({s_open:.2f}pt): {open_d:+.2f}% ({direction})"
        )

    # OI 연동 필드 (calc_expected_amplitude에서 oi_levels 전달 시 채워짐)
    try:
        oi_box   = float(amplitude.get("oi_box_pt")       or 0.0)
        oi_ratio = float(amplitude.get("oi_vs_amplitude") or 0.0)
        c_dist   = float(amplitude.get("call_dist_pt")    or 0.0)
        p_dist   = float(amplitude.get("put_dist_pt")     or 0.0)
        if oi_box > 0.0 and oi_ratio > 0.0:
            lines.append(
                f"OI 박스폭 {oi_box:.1f}pt / 예상 진폭 {exp_pt:.1f}pt = {oi_ratio:.2f}x "
                f"({'박스>진폭' if oi_ratio >= 1.0 else '박스<진폭 ⚠️'})"
            )
        if c_dist > 0.0:
            lines.append(f"  Call OI Peak까지 {c_dist:.1f}pt")
        if p_dist > 0.0:
            lines.append(f"  Put OI 지지까지  {p_dist:.1f}pt")
    except Exception:
        pass

    if source:
        lines.append(f"  (진폭 소스: {source})")

    return "\n".join(lines)


def _describe_oi_amplitude_alignment(
    opt_snap: Dict[str, Any],
    amplitude: Dict[str, Any],
) -> str:
    """OI 지지저항 레벨과 당일 진폭의 정합성을 교차 분석한다.

    OI 박스(call_peak ~ put_peak)의 pt 폭과 IV 기반 예상 진폭을 비교하고,
    잔여 진폭이 OI Peak까지의 거리와 어떻게 맞물리는지를 자연어로 서술한다.

    표시 조건: OI 레벨과 진폭 데이터가 모두 유효할 것.
    """
    if not isinstance(opt_snap, dict) or not isinstance(amplitude, dict):
        return ""

    try:
        # ── OI 데이터 ──────────────────────────────────────────────────────
        oi = opt_snap.get("_oi_levels")
        if not isinstance(oi, dict) or not oi:
            oi = opt_snap  # v5 fallback

        call_peak  = float(oi.get("call_oi_peak")  or 0.0)
        put_peak   = float(oi.get("put_oi_peak")   or 0.0)
        call_dist  = float(oi.get("dist_to_call_peak") or 0.0)   # % (양수 = 위)
        put_dist   = float(oi.get("dist_to_put_peak")  or 0.0)   # % (양수 = 아래)
        range_pct  = float(oi.get("oi_range_pct")  or 0.0)

        # PCR(OI): 풋/콜 OI 비율 — 1.0 초과면 풋 편향
        pcr_oi     = float(opt_snap.get("pcr_oi") or oi.get("pcr_oi") or 1.0)

        # ── 진폭 데이터 ────────────────────────────────────────────────────
        exp_pt     = float(amplitude.get("expected_amplitude_pt")  or 0.0)
        real_pt    = float(amplitude.get("realized_hl_range_pt")   or 0.0)
        remain_pt  = float(amplitude.get("remaining_amplitude_pt") or 0.0)
        # [FIX-AMP-4] 방향별 잔여: Call/Put OI Peak 비교에 활용
        up_rem_pt  = float(amplitude.get("upside_remaining_pt")    or 0.0)
        dn_rem_pt  = float(amplitude.get("downside_remaining_pt")  or 0.0)
        exhaust    = float(amplitude.get("amplitude_exhaustion")   or 0.0)
        s_open     = float(amplitude.get("session_open")           or 0.0)
    except Exception:
        return ""

    # 필수 데이터 없으면 생략
    if (call_peak <= 0.0 and put_peak <= 0.0) or exp_pt <= 0.0:
        return ""

    lines: list[str] = []

    # ── 1. OI 박스폭(pt) vs 예상 진폭 비교 ───────────────────────────────
    if call_peak > 0.0 and put_peak > 0.0 and s_open > 0.0:
        oi_box_pt = float(call_peak - put_peak)
        if oi_box_pt > 0.0:
            ratio = oi_box_pt / exp_pt if exp_pt > 0.0 else 0.0
            if ratio >= 1.2:
                assessment = f"OI 박스({oi_box_pt:.1f}pt)가 예상 진폭보다 넓음 — 진폭이 박스 내에서 소화될 가능성 높음"
            elif ratio >= 0.8:
                assessment = f"OI 박스({oi_box_pt:.1f}pt)와 예상 진폭이 유사 — OI Peak가 당일 극값에 근접"
            else:
                assessment = f"OI 박스({oi_box_pt:.1f}pt)가 예상 진폭보다 좁음 ⚠️ — OI Peak 돌파 가능성 있음"
            lines.append(f"OI 박스폭 {oi_box_pt:.1f}pt vs 예상 진폭 {exp_pt:.1f}pt (비율 {ratio:.2f}x): {assessment}")

    # ── 2. 잔여 진폭 vs OI Peak까지 거리 비교 ────────────────────────────
    # amplitude 딕셔너리에 calc_expected_amplitude()가 현재가 기준으로 계산한
    # call_dist_pt / put_dist_pt가 이미 들어 있다.
    # 이전 구현은 dist_to_call_peak(%)와 s_open을 사용해 역산했는데,
    # 역산 기준이 현재가가 아닌 시가여서 오차가 발생했다.
    # FIX: amplitude에서 직접 읽어 재역산 없이 사용한다.
    try:
        call_dist_pt = float(amplitude.get("call_dist_pt") or 0.0)
        put_dist_pt  = float(amplitude.get("put_dist_pt")  or 0.0)

        # [FIX-AMP-4] Call OI Peak 도달 가능성: upside_remaining 우선, fallback remain_pt
        _up_ref = up_rem_pt if up_rem_pt > 0.0 else remain_pt
        # [FIX-AMP-4] Put OI 지지 도달 가능성: downside_remaining 우선, fallback remain_pt
        _dn_ref = dn_rem_pt if dn_rem_pt > 0.0 else remain_pt

        if _up_ref > 0.0 and call_dist_pt > 0.0:
            _up_label = f"위쪽 잔여 {_up_ref:.1f}pt" if up_rem_pt > 0.0 else f"잔여 {_up_ref:.1f}pt"
            if call_dist_pt <= _up_ref * 1.1:
                lines.append(
                    f"{_up_label} ≥ Call OI Peak까지 {call_dist_pt:.1f}pt — "
                    "상승 시 OI 저항에 도달 가능"
                )
            else:
                lines.append(
                    f"{_up_label} < Call OI Peak까지 {call_dist_pt:.1f}pt — "
                    "OI 저항까지 추가 위쪽 여력 부족"
                )

        if _dn_ref > 0.0 and put_dist_pt > 0.0:
            _dn_label = f"아래쪽 잔여 {_dn_ref:.1f}pt" if dn_rem_pt > 0.0 else f"잔여 {_dn_ref:.1f}pt"
            if put_dist_pt <= _dn_ref * 1.1:
                lines.append(
                    f"{_dn_label} ≥ Put OI 지지까지 {put_dist_pt:.1f}pt — "
                    "하락 시 OI 지지에 도달 가능"
                )
            else:
                lines.append(
                    f"{_dn_label} < Put OI 지지까지 {put_dist_pt:.1f}pt — "
                    "OI 지지까지 추가 아래쪽 여력 부족"
                )
    except Exception:
        pass

    # ── 3. 진폭 소진 + OI 레벨 종합 판단 ─────────────────────────────────
    # Medium-08: 소진율 임계값을 만기 근접도(dte_weight_norm)에 따라 동적 조정.
    # 만기 당일(dte_w 높음)에는 0.6도 의미 있는 경고,
    # 만기 1주 이상(dte_w 낮음)에는 0.85 이상이어야 유의미하다.
    try:
        _dte_w_align = float(opt_snap.get("dte_weight_norm") or 0.0)
        if _dte_w_align >= 0.5:        # 만기 당일권 (DTE ≤ 2)
            _exhaust_warn   = 0.60
            _exhaust_plenty = 0.25
        elif _dte_w_align >= 0.2:      # 만기 주 (DTE 3~7)
            _exhaust_warn   = 0.75
            _exhaust_plenty = 0.30
        else:                          # 만기 1주 이상
            _exhaust_warn   = 0.85
            _exhaust_plenty = 0.35
    except Exception:
        _exhaust_warn   = 0.80
        _exhaust_plenty = 0.30

    try:
        if exhaust >= _exhaust_warn and (call_dist <= 0.3 or put_dist <= 0.3):
            lines.append(
                f"⚠️ 진폭 소진율 {exhaust:.0%} + OI Peak 근접 — "
                "추가 방향성 제한적, 반전 또는 횡보 구간 가능성"
            )
        elif exhaust < _exhaust_plenty and range_pct > 0.0:
            lines.append(
                f"진폭 여력 충분({remain_pt:.1f}pt 잔존) + OI 박스 유효 — "
                "OI Peak 방향 추세 지속 가능성"
            )
    except Exception:
        pass

    # ── 4. PCR(OI) × 진폭 복합 해석 (Medium-09) ─────────────────────────
    # PCR > 1.5(풋 편향) + 진폭 소진 → 하방 압력 약화 신호
    # PCR < 0.7(콜 편향) + 진폭 소진 → 상방 압력 약화 신호
    try:
        if exhaust > 0.0 and pcr_oi > 0.0:
            if pcr_oi > 1.5 and exhaust >= _exhaust_warn:
                lines.append(
                    f"📊 PCR(OI) {pcr_oi:.2f} (풋 편향) + 진폭 소진 {exhaust:.0%} — "
                    "하방 포지션 헤지 우세 but 진폭 소진으로 추가 하락 압력 약화 가능"
                )
            elif pcr_oi < 0.7 and exhaust >= _exhaust_warn:
                lines.append(
                    f"📊 PCR(OI) {pcr_oi:.2f} (콜 편향) + 진폭 소진 {exhaust:.0%} — "
                    "상방 포지션 우세 but 진폭 소진으로 추가 상승 여력 제한 가능"
                )
            elif pcr_oi > 1.5 and exhaust < _exhaust_plenty:
                lines.append(
                    f"📊 PCR(OI) {pcr_oi:.2f} (풋 편향) + 진폭 여력 충분 — "
                    "하방 헤지 수요 + 추가 방향성 여력 존재"
                )
    except Exception:
        pass

    return "\n".join(lines)


def build_llm_context(
    *,
    snapshot: Dict[str, Any],
    ob_records: Optional[List[Dict[str, Any]]] = None,
    adaptive_context: Optional[str] = None,
    amplitude: Optional[Dict[str, Any]] = None,
    prob_interval: Optional[tuple] = None,
) -> str:
    """Build a textual context for the LLM.

    The context is primarily JSON (snapshot + optional orderbook summary) so the
    LLM can follow strict-JSON output instructions reliably.

    NW-ARC-03: 각 섹션 구성 실패 시 warning 로그를 남기고 빈 문자열로 fallback한다.
    빈 섹션이 LLM에 전달되면 판단 품질이 저하될 수 있으므로 원인 추적이 필요하다.

    Args:
        amplitude: pipeline._calc_amplitude_snapshot() 반환값.
                   [AMPLITUDE_ANALYSIS] 섹션 생성에 사용된다.
        prob_interval: (lower, upper) float 튜플. Conformal Prediction 구간.
                       None 이면 [PREDICTION_INTERVAL] 섹션을 생략한다.
    """
    snap = dict(snapshot) if isinstance(snapshot, dict) else {}
    opt_snap = None
    try:
        opt_snap = snap.pop("options", None)
    except Exception as e:
        logger.warning("[ContextBuilder] snapshot에서 options 섹션 추출 실패: %s", e)
        opt_snap = None

    # amplitude는 result dict에서 분리 (snap에 포함돼 있을 수 있음)
    amp_data = amplitude
    if amp_data is None:
        try:
            amp_data = snap.pop("amplitude", None)
        except Exception:
            amp_data = None

    ob_sum = _summarize_orderbook(list(ob_records or []))

    lines: List[str] = []
    lines.append("[PIPELINE_INPUT]")
    try:
        lines.append(json.dumps(snap, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("[ContextBuilder] snapshot JSON 직렬화 실패 (str fallback): %s", e)
        lines.append(str(snap))

    if ob_sum:
        lines.append("")
        lines.append("[ORDERBOOK_SUMMARY_LAST_60S]")
        try:
            lines.append(json.dumps(ob_sum, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("[ContextBuilder] orderbook summary JSON 직렬화 실패 (str fallback): %s", e)
            lines.append(str(ob_sum))

    if opt_snap:
        lines.append("")
        lines.append("[OPTIONS_SNAPSHOT]")
        try:
            lines.append(json.dumps(opt_snap, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("[ContextBuilder] options snapshot JSON 직렬화 실패 (str fallback): %s", e)
            lines.append(str(opt_snap))

        # v3/v4 전용: 패리티 이탈 자연어 해설 섹션.
        # dte_weight_norm < 0.1 이거나 이탈 신호가 없으면 섹션 자체가 생략된다.
        try:
            parity_desc = _describe_parity_divergence(opt_snap)
            if parity_desc:
                lines.append("")
                lines.append("[PARITY_ANALYSIS]")
                lines.append(parity_desc)
        except Exception as e:
            logger.debug("[ContextBuilder] parity analysis 생성 실패 (건너뜀): %s", e)

        # v4 전용: 프리미엄 블리드 자연어 해설 섹션.
        # dte_weight_norm < 0.1 이거나 신호가 없으면 섹션 자체가 생략된다.
        try:
            bleed_desc = _describe_premium_bleed(opt_snap)
            if bleed_desc:
                lines.append("")
                lines.append("[PREMIUM_BLEED]")
                lines.append(bleed_desc)
        except Exception as e:
            logger.debug("[ContextBuilder] premium bleed 생성 실패 (건너뜀): %s", e)

        # v1~v5 공통: OI 기반 지지저항 레벨 자연어 해설 섹션.
        # build_option_snapshot()이 모든 fs에서 _oi_levels 키를 저장하므로 항상 시도한다.
        # OI 데이터가 없거나(장 시작 직후) oi_range_pct == 0이면 섹션이 생략된다.
        try:
            oi_desc = _describe_oi_levels(opt_snap)
            if oi_desc:
                lines.append("")
                lines.append("[OI_SUPPORT_RESISTANCE]")
                lines.append(oi_desc)
        except Exception as e:
            logger.debug("[ContextBuilder] OI levels 생성 실패 (건너뜀): %s", e)

        # v1~v5 공통: OTM 프리미엄 변화율 자연어 해설 섹션.
        # open_price 미주입(장 초기) 시 count=0 → 섹션 생략.
        try:
            otm_desc = _describe_otm_premium(opt_snap)
            if otm_desc:
                lines.append("")
                lines.append("[OTM_PREMIUM_FLOW]")
                lines.append(otm_desc)
        except Exception as e:
            logger.debug("[ContextBuilder] OTM premium 생성 실패 (건너뜀): %s", e)

        # v1~v5 공통: 옵션 가격 레벨 터치 스캔 섹션.
        # 고가/저가가 1.20 / 2.50 / 3.50 / 4.85 / 5.50 레벨에 근접한 종목을 탐지한다.
        # 터치 항목이 없으면 섹션이 생략된다.
        try:
            pls_desc = _describe_price_level_scan(opt_snap)
            if pls_desc:
                lines.append("")
                lines.append("[OPTION_PRICE_LEVEL_TOUCH]")
                lines.append(pls_desc)
        except Exception as e:
            logger.debug("[ContextBuilder] price level scan 생성 실패 (건너뜀): %s", e)

    # 진폭 분석 섹션: IV 기반 예상 진폭 vs 장중 실현 진폭.
    # 데이터가 없거나 expected_amplitude_pt == 0이면 섹션이 생략된다.
    try:
        amp_desc = _describe_amplitude(amp_data or {})
        if amp_desc:
            lines.append("")
            lines.append("[AMPLITUDE_ANALYSIS]")
            lines.append(amp_desc)
    except Exception as e:
        logger.debug("[ContextBuilder] amplitude analysis 생성 실패 (건너뜀): %s", e)

    # OI 지지저항 ↔ 진폭 정합성 교차 분석 섹션.
    # OI 레벨과 진폭 데이터가 모두 유효할 때만 생성된다.
    try:
        align_desc = _describe_oi_amplitude_alignment(opt_snap or {}, amp_data or {})
        if align_desc:
            lines.append("")
            lines.append("[OI_AMPLITUDE_ALIGNMENT]")
            lines.append(align_desc)
    except Exception as e:
        logger.debug("[ContextBuilder] OI-amplitude alignment 생성 실패 (건너뜀): %s", e)

    if adaptive_context:
        try:
            txt = str(adaptive_context).strip()
        except Exception as e:
            logger.warning("[ContextBuilder] adaptive_context 변환 실패 (건너뜀): %s", e)
            txt = ""
        if txt:
            lines.append("")
            lines.append("[ADAPTIVE_INDICATORS]")
            lines.append(txt)

    # Conformal Prediction 구간 섹션 — prob_interval 이 제공될 때만 추가.
    # LLM은 이 구간을 참조해 불확실성이 높을 때(구간 폭 넓음) 더 보수적 판단을 내릴 수 있다.
    if prob_interval is not None:
        try:
            lo, hi = float(prob_interval[0]), float(prob_interval[1])
            width = round(hi - lo, 4)
            # 불확실성 수준 레이블: 구간 폭 기준
            if width <= 0.10:
                level = "LOW"       # 좁은 구간 → 모델 확신
            elif width <= 0.20:
                level = "MEDIUM"
            else:
                level = "HIGH"      # 넓은 구간 → 불확실
            interval_info = {
                "prob_lower": round(lo, 4),
                "prob_upper": round(hi, 4),
                "interval_width": width,
                "uncertainty": level,
                "note": (
                    "90% 커버리지 Conformal Prediction 구간. "
                    "구간이 넓을수록 모델 불확실성이 높습니다."
                ),
            }
            lines.append("")
            lines.append("[PREDICTION_INTERVAL]")
            try:
                lines.append(json.dumps(interval_info, ensure_ascii=False, indent=2))
            except Exception:
                lines.append(str(interval_info))
        except Exception as e:
            logger.debug("[ContextBuilder] prediction_interval 섹션 생성 실패 (건너뜀): %s", e)

    return "\n".join(lines).strip()


def build_llm_prompt(*, context: str, prediction_minutes: int) -> Tuple[str, str]:
    """Build `(system, user)` prompt strings for chat-based LLM APIs.

    [LLM-FIX-5] 프롬프트 개선:
    - system 프롬프트에 JSON 출력 강제 지침 구체화
    - 구체적 예시(few-shot) 포함으로 JSON 포맷 준수율 향상
    - context 길이 상한(6000자) 적용으로 토큰 초과 타임아웃 방지
    """
    # [LLM-FIX-5] context 길이 제한: 너무 긴 컨텍스트는 타임아웃을 유발한다.
    # 6000자를 초과하면 앞부분 PIPELINE_INPUT 섹션 위주로 트리밍한다.
    MAX_CONTEXT_CHARS = 6000
    ctx = str(context or "").strip()
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS] + "\n... (truncated)"
        logger.debug("[ContextBuilder] context truncated to %d chars", MAX_CONTEXT_CHARS)

    system = (
        "당신은 KP200 선물 파생상품 트레이딩 전문가입니다.\n"
        f"제공된 시장 데이터를 분석하여 향후 {int(prediction_minutes)}분 관점의 매매 판단을 내리세요.\n\n"
        "피봇후보 정보([후보], [후보해석])가 있으면 후보 확정 가능성을 함께 평가하세요.\n"
        "단, 후보는 미확정이므로 확정 신호(new_swing)와 구분해 보수적으로 해석하세요.\n\n"
        "【필수 출력 규칙】\n"
        "1. 반드시 JSON 단일 객체 하나만 출력하세요.\n"
        "2. 마크다운(```), 설명 텍스트, 줄바꿈 등을 절대 추가하지 마세요.\n"
        "3. 첫 글자는 반드시 '{'이고 마지막 글자는 반드시 '}'이어야 합니다.\n"
        "4. 다른 말은 일절 하지 말고 JSON 객체만 출력하세요."
    )

    schema = dict(LLM_OUTPUT_SCHEMA)

    # 구체적 예시 포함으로 LLM JSON 포맷 준수율 향상
    example = json.dumps(
        {
            "action": "BUY",
            "risk_level": "MEDIUM",
            "rationale": "OBI 상승 추세 + ATM IV 낮음. SuperTrend 상방 신호.",
            "caution": "basis 2.5pt 이상 확대 시 무효화.",
            "pivot_candidate_probability": "MEDIUM",
            "pivot_candidate_reason": "저점 후보가 관찰되며 확정까지 1봉 남아 있으나 아직 미확정 상태.",
        },
        ensure_ascii=False,
    )

    user = (
        f"[입력 데이터 — 예측 {int(prediction_minutes)}분]\n"
        f"{ctx}\n\n"
        "[출력 스키마]\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + f"\n\n[출력 예시]\n{example}\n\n"
        "위 예시처럼 JSON 객체 하나만 출력하세요. 다른 텍스트 없이."
    )

    return system, user
