"""텔레그램 메시지 포매터 모음.

telegram_notifier.py에서 분리된 순수 함수 모음.
TelegramNotifier / PipelineTelegramBridge 등 I/O 클래스에 의존하지 않는다.

외부에서 사용 시:
    from .formatters import format_prediction_message, format_premium_bleed_alert
    from .notifier import TelegramNotifier  # I/O 클래스는 여기서
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────
_TG_MAX_LEN = 4096
_TG_TRUNCATE_SUFFIX = "\n\n⚠️ \\(메시지 일부 생략\\)"

# ──────────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────────

def _esc_mdv2(s: str) -> str:
    """MarkdownV2 특수문자 이스케이프 (모듈 공통).

    텔레그램 MarkdownV2 에서 이스케이프가 필요한 모든 특수문자를 처리한다.
    중복 정의를 방지하기 위해 모든 formatter 가 이 함수를 공유한다.
    """
    for ch in r"\_*[]()~`>#+-=|{}.!":
        s = s.replace(ch, f"\\{ch}")
    return s


# ──────────────────────────────────────────────
# 이모지 매핑
# ──────────────────────────────────────────────
_SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
_CONFIDENCE_EMOJI = {"HIGH": "💪", "MEDIUM": "👌", "LOW": "⚠️"}
_RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
_REGIME_EMOJI = {
    "STRONG_UP": "🚀",
    "WEAK_UP": "📈",
    "RANGE": "↔️",
    "WEAK_DOWN": "📉",
    "STRONG_DOWN": "🆘",
}


# ──────────────────────────────────────────────
# 메시지 포매터
# ──────────────────────────────────────────────

def format_prediction_message(
    result: Dict[str, Any],
    *,
    include_dir_summary: bool = True,
    prev_signal: str = "",
    symbol: str = "",
) -> str:
    """get_prediction() 결과 dict → 텔레그램 MarkdownV2 메시지.

    Args:
        result: PredictionPipeline.get_prediction() 반환값.
        include_dir_summary: True면 model_outputs 기반 [DIR_SUMMARY] 블록을 하단에 추가.
        prev_signal: 이전 신호 (BUY/SELL/HOLD). 비어 있으면 전환 표기 생략.
    """
    esc = _esc_mdv2  # 모듈 공통 함수 사용

    signal = str(result.get("signal", "HOLD")).upper()
    confidence = str(result.get("confidence", "LOW")).upper()
    llm_action = str(result.get("llm_action", "-")).upper()
    risk_level = str(result.get("risk_level", "-")).upper()
    consensus = result.get("consensus", False)
    prob = result.get("prob")
    prob_lower = result.get("prob_lower")
    prob_upper = result.get("prob_upper")
    t_prob = result.get("transformer_prob")
    tft_prob = result.get("tft_prob")
    current_price = result.get("current_price")
    spot_index = result.get("spot_index")
    basis = result.get("basis")
    regime = result.get("regime")
    rationale = str(result.get("rationale", ""))
    caution = str(result.get("caution", ""))
    pivot_cand_prob = str(result.get("pivot_candidate_probability", "") or "").upper()
    pivot_cand_reason = str(result.get("pivot_candidate_reason", "") or "")
    pred_time = result.get("prediction_time", "")
    pred_min = result.get("prediction_minutes", "?")
    target_time = result.get("target_time", "")
    llm_provider = str(result.get("llm_provider", ""))
    ensemble_method = str(result.get("ensemble_method", ""))
    model_agreement = result.get("model_agreement")
    llm_timed_out = result.get("llm_timed_out", False)
    guardrail = result.get("guardrail") or {}

    s_emoji = _SIGNAL_EMOJI.get(signal, "⬜")
    c_emoji = _CONFIDENCE_EMOJI.get(confidence, "")
    r_emoji = _RISK_EMOJI.get(risk_level, "")
    regime_emoji = _REGIME_EMOJI.get(regime or "", "") if regime else ""
    consensus_mark = "✅ 합의" if consensus else "⚡ 불일치"

    # 시간 포매팅
    try:
        pt = datetime.fromisoformat(pred_time)
        pred_time_str = pt.strftime("%H:%M:%S")
    except Exception:
        pred_time_str = pred_time[:19] if pred_time else "?"
    try:
        tt = datetime.fromisoformat(target_time)
        target_time_str = tt.strftime("%H:%M")
    except Exception:
        target_time_str = target_time[:16] if target_time else "?"

    # 이전 신호 → 현재 신호 전환 표기 (ENH-02)
    prev = str(prev_signal or "").strip().upper()
    if prev and prev != signal:
        prev_emoji = _SIGNAL_EMOJI.get(prev, "⬜")
        signal_change_str = f"  \\(← {prev_emoji} `{esc(prev)}`\\)"
    else:
        signal_change_str = ""

    # ── 피봇 확정 여부 확인 ─────────────────────────────────
    model_outputs = result.get("model_outputs")
    _has_pivot = False
    _pivot_signal = ""
    _has_pending = False
    _pending_status = None
    try:
        _h = (model_outputs or {}).get("heuristic") or {}
        _zz = _h.get("zigzag_state")
        if _zz is not None:
            _sig = str(getattr(_zz, "new_swing_signal", "none") or "none")
            if _sig in ("new_high", "new_low"):
                _has_pivot = True
                _is_high = (_sig == "new_high")
                _pivot_signal = "STRONG SELL" if _is_high else "STRONG BUY"
            # 피봇 후보 상태 확인
            _pending_status = getattr(_zz, "pending_candidate_status", None)
            if _pending_status in ("등록", "갱신", "취소"):
                _has_pending = True
    except Exception:
        pass

    # ── 메시지 헤더 (피봇 여부에 따라 다름) ─────────────────────────────────
    # symbol에 따른 제목 결정
    _symbol_norm = str(symbol).strip()
    if "KP200" in _symbol_norm or "선물" in _symbol_norm:
        _pivot_title = "KP200 선물 ZigZag 피봇 신호"
        _pivot_confirm_title = "KP200 선물 ZigZag 피봇 확정"
        _pivot_pending_title = "KP200 선물 ZigZag 피봇 후보"
    else:
        _pivot_title = "KOSPI ZigZag 피봇 신호"
        _pivot_confirm_title = "KOSPI ZigZag 피봇 확정"
        _pivot_pending_title = "KOSPI ZigZag 피봇 후보"

    if _has_pivot:
        # 피봇 확정 시: 피봇 중심 헤더 (다음 예측 시간 제외)
        lines = [
            f"🎯 *{_pivot_title}* \\| {esc(pred_time_str)}",
            "",
        ]
    elif _has_pending:
        # 피봇 후보 시: 후보 중심 헤더
        lines = [
            f"🔍 *{_pivot_pending_title}* \\| {esc(pred_time_str)}",
            "",
        ]
    else:
        # 일반 예측 시: 기존 헤더 유지
        lines = [
            f"📊 *KP200 선물 예측* \\| {esc(pred_time_str)}",
            f"⏭ *다음 예측*: `{esc(target_time_str)}` \\({esc(str(pred_min))}분 후\\)",
            "",
            f"{s_emoji} *신호*: `{esc(signal)}`{signal_change_str}  {c_emoji} *신뢰도*: `{esc(confidence)}`",
            f"🤖 *LLM 판단*: `{esc(llm_action)}`  {r_emoji} *리스크*: `{esc(risk_level)}`",
            f"🤝 *컨센서스*: {consensus_mark}",
        ]

        if regime:
            lines.append(f"🌊 *시장 레짐*: {regime_emoji} `{esc(regime)}`")

        lines.append("")

    # 가격 정보
    if current_price is not None:
        lines.append(f"💰 *현재가*: `{esc(f'{current_price:.2f}')}`")
    if spot_index is not None:
        basis_str = f"  \\(basis: `{esc(f'{basis:+.2f}')}`\\)" if basis is not None else ""
        lines.append(f"📌 *현물지수*: `{esc(f'{spot_index:.2f}')}`{basis_str}")

    # ── 피봇 후보 섹션 ─────────────────────────────────
    if _has_pending:
        try:
            _h = (model_outputs or {}).get("heuristic") or {}
            _zz = _h.get("zigzag_state")
            if _zz is not None:
                _status = str(getattr(_zz, "pending_candidate_status", None) or "")
                _ctype = str(getattr(_zz, "pending_candidate_type", None) or "")
                _ctime = str(getattr(_zz, "pending_candidate_time", None) or "")
                _cprice = float(getattr(_zz, "pending_candidate_price", 0.0) or 0.0)
                _crem = int(getattr(_zz, "pending_candidate_remaining", 0) or 0)

                if _status == "취소":
                    # 취소 메시지
                    lines.append("")
                    lines.append(f"❌ *피봇 후보 취소*")
                    lines.append("")
                    lines.append(f"{'필드':<8}  {'값':}")
                    lines.append(f"```")
                    lines.append(f"상태      취소")
                    lines.append(f"유형      {'고점' if _ctype == 'high' else '저점'}({'H' if _ctype == 'high' else 'L'})")
                    if _ctime:
                        lines.append(f"후보봉    {_ctime}")
                    if _cprice > 0:
                        lines.append(f"후보가    {_cprice:.2f}")
                    lines.append(f"```")
                else:
                    # 등록/갱신 메시지
                    _pt_kor = "고점" if _ctype == "high" else "저점"
                    _pt_emoji = "🔺" if _ctype == "high" else "🔻"
                    _status_emoji = "✨" if _status == "등록" else "🔄"

                    lines.append("")
                    lines.append(f"{_pt_emoji} {_status_emoji} *피봇 후보 {_status}*")
                    lines.append("")
                    lines.append(f"{'필드':<8}  {'값':}")
                    lines.append(f"```")
                    lines.append(f"상태      {_status}")
                    lines.append(f"유형      {_pt_kor}({'H' if _ctype == 'high' else 'L'})")
                    if _ctime:
                        lines.append(f"후보봉    {_ctime}")
                    if _cprice > 0:
                        lines.append(f"후보가    {_cprice:.2f}")
                    lines.append(f"대기봉    {_crem}봉")
                    lines.append(f"```")
        except Exception:
            pass

    # ── 피봇 확정 섹션 (상단 이동) ─────────────────────────────────
    # 피봇이 있을 때만 상단에 표시, 없을 때는 하단에 표시
    if _has_pivot:
        try:
            _h = (model_outputs or {}).get("heuristic") or {}
            _zz = _h.get("zigzag_state")
            if _zz is not None:
                _sig = str(getattr(_zz, "new_swing_signal", "none") or "none")
                if _sig in ("new_high", "new_low"):
                    _is_high = (_sig == "new_high")
                    _pivot_price = float(getattr(_zz, "last_swing_high" if _is_high else "last_swing_low", 0.0) or 0.0)
                    _pivot_time  = getattr(_zz, "last_swing_high_time" if _is_high else "last_swing_low_time", None)
                    _confirm_time = getattr(_zz, "last_swing_high_confirm_time" if _is_high else "last_swing_low_confirm_time", None)
                    _lag_bars    = int(getattr(_zz, "last_swing_high_lag_bars" if _is_high else "last_swing_low_lag_bars", 0) or 0)
                    _bar_open    = float(getattr(_zz, "last_swing_high_open"  if _is_high else "last_swing_low_open",  0.0) or 0.0)
                    _bar_close   = float(getattr(_zz, "last_swing_high_close" if _is_high else "last_swing_low_close", 0.0) or 0.0)
                    _pt_kor      = "고점" if _is_high else "저점"
                    _pt_emoji    = "🔺" if _is_high else "🔻"
                    _dir_txt     = "↓반전 가능" if _is_high else "↑반전 가능"

                    # ── 피봇 확정 테이블 양식 ──────────────────────────────
                    # 순서: 피봇봉 | 유형 | 피봇가 | 확정시각 | 지연 | 확정시가 | 확정종가 | 파동크기
                    _pivot_str  = f"{_pivot_price:.2f}" if _pivot_price > 0 else "?"
                    _pivot_t    = esc(str(_pivot_time))    if _pivot_time    else "?"
                    _confirm_t  = esc(str(_confirm_time))  if _confirm_time  else "?"
                    _lag_str    = f"{_lag_bars}봉" if _lag_bars > 0 else "?"
                    _open_str   = esc(f"{_bar_open:.2f}")  if _bar_open  > 0 else "?"
                    _close_str  = esc(f"{_bar_close:.2f}") if _bar_close > 0 else "?"
                    _wave_size  = float(getattr(_zz, "wave_size",     0.0) or 0.0)
                    _wave_pct   = float(getattr(_zz, "wave_size_pct", 0.0) or 0.0)
                    _wave_sign  = "+" if (_wave_size >= 0 and not _is_high) or (_wave_size < 0 and _is_high) else ""
                    if _is_high:
                        _wave_disp = f"\\-{abs(_wave_size):.2f}pt \\({abs(_wave_pct):.2f}%\\)"
                    else:
                        _wave_disp = f"\\+{abs(_wave_size):.2f}pt \\({abs(_wave_pct):.2f}%\\)"
                    _lag_warn   = " ⚠️" if _lag_bars >= 8 else ""

                    lines.append("")
                    lines.append(f"{_pt_emoji} *{_pivot_confirm_title}*")
                    lines.append("")
                    lines.append(f"{'필드':<8}  {'값':}")
                    lines.append(f"```")
                    lines.append(f"피봇봉      {_pivot_t}")
                    lines.append(f"유형        {_pt_kor}({'H' if _is_high else 'L'})")
                    lines.append(f"피봇가      {_pivot_str}")
                    lines.append(f"확정시각    {_confirm_t}")
                    lines.append(f"지연        {_lag_bars}봉{_lag_warn.strip()}")
                    lines.append(f"확정시가    {_open_str}")
                    lines.append(f"확정종가    {_close_str}")
                    if _wave_size != 0:
                        _sign = '+' if not _is_high else '-'
                        lines.append(f"파동크기    {_sign}{abs(_wave_size):.2f}pt ({abs(_wave_pct):.2f}%)")
                    lines.append(f"```")
                    lines.append(f"   {_dir_txt}{_lag_warn}")

                    # ── 추가 지표 확인 섹션 ─────────────────────────────────
                    try:
                        _mkt = (model_outputs or {}).get("market") or {}
                        _ens = (model_outputs or {}).get("ensemble") or {}
                        _opt = (model_outputs or {}).get("options") or {}
                        _h = (model_outputs or {}).get("heuristic") or {}

                        # Basis
                        _basis = float(_mkt.get("basis", 0.0) or 0.0)
                        _basis_str = f"{_basis:.2f}pt" if _basis != 0 else "N/A"

                        # SuperTrend (heuristic에서 확인)
                        _st_signal = str(_h.get("super_trend_signal", "") or "").upper()
                        _st_str = _st_signal if _st_signal else "N/A"

                        # OBI (heuristic에서 확인)
                        _obi = str(_h.get("obi_signal", "") or "").upper()
                        _obi_str = _obi if _obi else "N/A"

                        # IV Skew (options에서 확인)
                        _iv_call = 0.0
                        _iv_put = 0.0
                        try:
                            _iv_data = _opt.get("iv_snapshot", [])
                            if _iv_data:
                                _iv_call = float(_iv_data[0].get("iv", 0.0) or 0.0)
                                if len(_iv_data) > 1:
                                    _iv_put = float(_iv_data[1].get("iv", 0.0) or 0.0)
                        except Exception:
                            pass
                        _iv_skew_str = f"콜:{_iv_call:.2f} 풋:{_iv_put:.2f}" if _iv_call > 0 and _iv_put > 0 else "N/A"

                        # PCR (options에서 확인)
                        _pcr = 0.0
                        try:
                            _pcr_data = _opt.get("pcr", {})
                            if _pcr_data:
                                _pcr = float(_pcr_data.get("pcr", 0.0) or 0.0)
                        except Exception:
                            pass
                        _pcr_str = f"{_pcr:.2f}" if _pcr > 0 else "N/A"

                        # 앙상블 신호
                        _ens_sig = str(_ens.get("signal", "") or "").upper()
                        _ens_conf = str(_ens.get("confidence", "") or "").upper()

                        # 종합 신호 계산
                        _bullish_signals = 0
                        _bearish_signals = 0

                        # 고점(↓반전) → 매도 조건 확인
                        # 저점(↑반전) → 매수 조건 확인
                        if _is_high:
                            # 고점: 매도 조건 (하락 반전 확인)
                            if _iv_put > _iv_call and _iv_put > 0:
                                _bearish_signals += 1  # 풋IV > 콜IV: 하락 경계
                            if _pcr > 1.0:
                                _bearish_signals += 1  # PCR > 1: 하락 예상
                            if _st_signal == "DOWN":
                                _bearish_signals += 1  # SuperTrend 하방
                            if _basis < 0:
                                _bearish_signals += 1  # 음의 베이시스
                            if _obi == "SELL":
                                _bearish_signals += 1  # OBI 매도 우위
                            if _ens_sig == "SELL":
                                _bearish_signals += 1  # 앙상블 매도
                        else:
                            # 저점: 매수 조건 (상승 반전 확인)
                            if _iv_call > _iv_put and _iv_call > 0:
                                _bullish_signals += 1  # 콜IV > 풋IV: 상승 경계
                            if _pcr < 1.0 and _pcr > 0:
                                _bullish_signals += 1  # PCR < 1: 상승 예상
                            if _st_signal == "UP":
                                _bullish_signals += 1  # SuperTrend 상방
                            if _basis > 0:
                                _bullish_signals += 1  # 양의 베이시스
                            if _obi == "BUY":
                                _bullish_signals += 1  # OBI 매수 우위
                            if _ens_sig == "BUY":
                                _bullish_signals += 1  # 앙상블 매수

                        # 종합 신호 결정 (3개 이상 일치 시 강력 신호)
                        _signal_threshold = 3
                        if _is_high:
                            if _bearish_signals >= _signal_threshold:
                                _final_signal = "🔴 STRONG SELL"
                            elif _bearish_signals >= 2:
                                _final_signal = "🟡 SELL"
                            else:
                                _final_signal = "⚪ NEUTRAL"
                        else:
                            if _bullish_signals >= _signal_threshold:
                                _final_signal = "🟢 STRONG BUY"
                            elif _bullish_signals >= 2:
                                _final_signal = "🟡 BUY"
                            else:
                                _final_signal = "⚪ NEUTRAL"

                        # 추가 지표 표시
                        lines.append("")
                        lines.append("   📊 *추가 지표*")
                        lines.append(f"   Basis: `{_basis_str}`  SuperTrend: `{_st_str}`  OBI: `{_obi_str}`")
                        lines.append(f"   IV Skew: `{_iv_skew_str}`  PCR: `{_pcr_str}`")
                        lines.append(f"   앙상블: `{_ens_sig}` (확신도: {_ens_conf})")
                        lines.append("")
                        lines.append(f"   🎯 *종합 신호*: {_final_signal}")

                    except Exception:
                        pass

                    # 누적 확정 피봇 요약
                    try:
                        _cnt = int(getattr(_zz, "confirmed_pivot_count", 0) or 0)
                        _tail_hhmm = str(getattr(_zz, "confirmed_pivot_tail_hhmm", "") or "").strip()
                        if _cnt > 0:
                            lines.append(f"   누적 확정: `{esc(str(_cnt))}개`")
                            if _tail_hhmm:
                                # tail 을 한 줄로 압축 (| 구분자 그대로 유지)
                                lines.append(f"   최근 피봇: {esc(_tail_hhmm)}")
                    except Exception:
                        pass
        except Exception:
            pass

    # 확률
    if prob is not None:
        lines.append(f"🎯 *상승확률*: `{esc(f'{prob:.1%}')}`")
    # Conformal Prediction 구간 (보정 완료 시에만 출력)
    if prob_lower is not None and prob_upper is not None:
        try:
            width = float(prob_upper) - float(prob_lower)
            if width <= 0.10:
                unc_emoji = "🟢"
            elif width <= 0.20:
                unc_emoji = "🟡"
            else:
                unc_emoji = "🔴"
            lines.append(
                f"  ┗ {unc_emoji} *예측구간*: "
                f"`{esc(f'{prob_lower:.1%}')}` \\~ `{esc(f'{prob_upper:.1%}')}`"
                f"  \\(폭 {esc(f'{width:.1%}')}\\)"
            )
        except Exception:
            pass
    if t_prob is not None or tft_prob is not None:
        detail_parts = []
        if t_prob is not None:
            detail_parts.append(f"Transformer: {t_prob:.1%}")
        if tft_prob is not None:
            detail_parts.append(f"TFT: {tft_prob:.1%}")
        if ensemble_method:
            detail_parts.append(f"앙상블: {ensemble_method}")
        lines.append(f"  ┗ {esc('  |  '.join(detail_parts))}")

    if model_agreement is not None:
        agree_str = "✅ 모델 합의" if model_agreement else "⚠️ 모델 불일치"
        lines.append(f"  ┗ {agree_str}")

    # LLM 근거
    if rationale:
        lines.append("")
        lines.append(f"💡 *근거*: {esc(rationale[:200])}")
    if caution:
        lines.append(f"⚡ *주의*: {esc(caution[:150])}")
    if pivot_cand_prob:
        lines.append(f"🧭 *피봇후보 확정가능성*: `{esc(pivot_cand_prob)}`")
    if pivot_cand_reason:
        lines.append(f"📝 *피봇후보 근거*: {esc(pivot_cand_reason[:150])}")

    # 옵션 요약 (PCR)
    options = result.get("options")
    if isinstance(options, dict) and options:
        pcr_v = options.get("pcr_volume")
        pcr_oi = options.get("pcr_oi")
        if pcr_v is not None or pcr_oi is not None:
            lines.append("")
            parts = []
            if pcr_v is not None:
                parts.append(f"PCR(V): {pcr_v:.2f}")
            if pcr_oi is not None:
                parts.append(f"PCR(OI): {pcr_oi:.2f}")
            lines.append(f"📐 *옵션*: {esc('  |  '.join(parts))}")
        try:
            tf = options.get("_tick_flow")
            if isinstance(tf, dict):
                _t1 = float(tf.get("ticks_1m") or 0.0)
                _a20 = float(tf.get("ticks_avg20m") or 0.0)
                _sr = float(tf.get("surge_ratio") or 0.0)
                _imb = float(tf.get("cp_imbalance") or 0.0)
                _pt = float(tf.get("per_tick_move_pt") or 0.0)
                if _t1 > 0.0 or _a20 > 0.0 or abs(_imb) > 0.0 or _pt > 0.0:
                    lines.append(
                        "🔄 *옵션 틱 유입*: "
                        + esc(f"{_sr:.2f}x (1m={int(round(_t1))}, avg20m={int(round(_a20))})")
                        + " 🔥"
                    )
                    _bias = "중립"
                    if _imb > 0.02:
                        _bias = "콜 우세"
                    elif _imb < -0.02:
                        _bias = "풋 우세"
                    lines.append(
                        "⚖️ *옵션 틱 편향*: "
                        + esc(f"{_imb:+.2f} ({_bias}), 틱당 변동: {_pt:.3f}pt")
                    )
        except Exception:
            pass

    # OI 지지저항 인라인 블록
    # options dict 안의 _oi_levels 키 사용 (모든 fs에서 공통 저장됨)
    # oi_range_pct > 0 일 때만 표시 (장 시작 직후 OI 없는 경우 생략)
    # TG-02: Zero Gamma 경고 임계값을 가드레일(0.2%)보다 넓은 0.3%로 유지하되,
    #        경고 문구에 "(가드레일 미적용)" 여부를 구분하지 않고 동일 임계값 0.3% 사용.
    #        향후 두 임계값을 통일하려면 _OI_ZG_WARN_DIST_PCT 상수를 변경.
    _OI_ZG_WARN_DIST_PCT = 0.3  # 텔레그램 경고 표시 임계값
    try:
        _oi = None
        if isinstance(options, dict):
            _oi = options.get("_oi_levels")
        if isinstance(_oi, dict):
            call_peak  = float(_oi.get("call_oi_peak") or 0.0)
            put_peak   = float(_oi.get("put_oi_peak") or 0.0)
            if not (call_peak > 0.0 or put_peak > 0.0):
                raise ValueError("oi_empty")
            # GR-03/TG-01: None 체크로 읽어 dist=0.0(ATM Peak)도 정상 처리
            _cdist_raw = _oi.get("dist_to_call_peak")
            _pdist_raw = _oi.get("dist_to_put_peak")
            call_dist  = float(_cdist_raw if _cdist_raw is not None else 0.0)
            put_dist   = float(_pdist_raw if _pdist_raw is not None else 0.0)
            call_conc  = float(_oi.get("call_oi_peak_norm") or 0.0)
            put_conc   = float(_oi.get("put_oi_peak_norm") or 0.0)
            above_vt   = float(_oi.get("above_vol_trigger") if _oi.get("above_vol_trigger") is not None else 1.0)
            vt_strike  = float(_oi.get("vol_trigger_strike") or 0.0)
            zgd        = float(_oi.get("zero_gamma_dist_pct") if _oi.get("zero_gamma_dist_pct") is not None else 0.0)
            # IV 기반 동적 탐색 범위 — 표시용
            _iv_range  = float(_oi.get("peak_search_range_used") or 0.0)

            lines.append("")
            lines.append("📊 *OI 지지저항*")

            # 저항 — call_dist: 양수=저항이 위, 0=ATM, 음수=돌파
            if call_peak > 0.0:
                if call_dist > 0:
                    resist_arrow = "▲"
                elif call_dist == 0.0:
                    resist_arrow = "◆"   # ATM Call Peak
                else:
                    resist_arrow = "✂️"  # 돌파 상태
                lines.append(
                    f"  {resist_arrow} 저항: `{esc(f'{call_peak:.2f}')}`  "
                    f"{esc(f'{call_dist:+.2f}%')}  "
                    f"집중도 {esc(f'{call_conc:.0%}')}"
                )
            # 지지 — put_dist: 양수=지지가 아래, 0=ATM, 음수=이탈
            # TG-01 수정: 항상 '-' 접두사 제거 → put_dist 부호를 그대로 반전하여 표시
            # (put_dist 정의: (F-put_peak)/F*100, 지지 아래 = 양수)
            # 텔레그램 표시: 지지까지 거리를 음수(아래 방향)로 표시하므로 -put_dist 사용
            if put_peak > 0.0:
                if put_dist > 0:
                    supp_arrow = "▼"
                elif put_dist == 0.0:
                    supp_arrow = "◆"   # ATM Put Peak
                else:
                    supp_arrow = "✂️"  # 이탈 상태
                # TG-01 수정: -put_dist를 +/- 자동 표시로 출력
                # put_dist > 0 → 지지가 현재가 아래 → 표시: "-1.43%" (지지까지 거리)
                # put_dist < 0 → 이탈 상태 → 표시: "+0.50%" (이탈 정도)
                lines.append(
                    f"  {supp_arrow} 지지: `{esc(f'{put_peak:.2f}')}`  "
                    f"{esc(f'{-put_dist:+.2f}%')}  "
                    f"집중도 {esc(f'{put_conc:.0%}')}"
                )

            # 레짐
            if vt_strike > 0.0:
                if above_vt >= 1.0:
                    lines.append(
                        f"  📗 레짐: Long Gamma "
                        f"\\(VT {esc(f'{vt_strike:.2f}')} 상방\\)"
                    )
                else:
                    lines.append(
                        f"  📕 레짐: Short Gamma ⚠️ "
                        f"\\(VT {esc(f'{vt_strike:.2f}')} 하방\\)"
                    )

            # Zero Gamma 근접 경고 (TG-02: 상수 _OI_ZG_WARN_DIST_PCT = 0.3 사용)
            if abs(zgd) < _OI_ZG_WARN_DIST_PCT:
                zg_strike = float(_oi.get("zero_gamma_strike") or 0.0)
                lines.append(
                    f"  ⚡ ZeroGamma `{esc(f'{zg_strike:.2f}')}` 근접 "
                    f"\\({esc(f'{zgd:+.2f}%')}\\)"
                )

            # IV 기반 탐색 범위 — 디버그/투명성 표시 (0이면 생략)
            if _iv_range > 0.0:
                lines.append(
                    f"  \\(탐색반경 ±{esc(f'{_iv_range:.1f}')}pt\\)"
                )
    except Exception:
        pass

    # 패리티 가드레일 알림 — parity_guardrail이 신호를 변경한 경우에만 표시
    # result["guardrail"]["reason"]에 "parity_divergence" 문자열이 포함된 경우:
    #   - MEDIUM→LOW 강등: 만기 3일 이내 + |score| >= 0.5
    #   - BUY/SELL→HOLD:   만기 당일   + |score| >= 0.8
    try:
        guardrail_reason = str(guardrail.get("reason") or "")
        guardrail_orig_signal = str(guardrail.get("original_signal") or "").upper()
        guardrail_orig_conf   = str(guardrail.get("original_confidence") or "").upper()
        if guardrail.get("applied") and "parity_divergence" in guardrail_reason:
            lines.append("")
            # 원래 신호에서 현재 신호로 바뀐 경우 전환 내용 포함
            change_parts = []
            if guardrail_orig_signal and guardrail_orig_signal != signal:
                orig_e = _SIGNAL_EMOJI.get(guardrail_orig_signal, "⬜")
                cur_e  = _SIGNAL_EMOJI.get(signal, "⬜")
                change_parts.append(
                    f"{orig_e}`{esc(guardrail_orig_signal)}` → {cur_e}`{esc(signal)}`"
                )
            if guardrail_orig_conf and guardrail_orig_conf != confidence:
                change_parts.append(
                    f"신뢰도 {esc(guardrail_orig_conf)} → {esc(confidence)}"
                )
            change_str = "  ".join(change_parts)
            # reason에서 score/dte_w 파싱하여 표시
            reason_disp = esc(guardrail_reason[:80])
            if change_str:
                lines.append(f"🛡 *패리티 가드레일*: {change_str}")
                lines.append(f"   `{reason_disp}`")
            else:
                lines.append(f"🛡 *패리티 가드레일*: `{reason_disp}`")
    except Exception:
        pass

    # 프리미엄 블리드 가드레일 알림 — premium_bleed가 신호를 변경한 경우에만 표시
    # result["guardrail"]["reason"]에 "premium_bleed" 문자열이 포함된 경우:
    #   - MEDIUM→LOW 강등: 만기 3일 이내 + |score| >= 0.5
    #   - BUY/SELL→HOLD:   만기 당일   + score <= -0.75 + 선물 상승 중 수축
    try:
        guardrail_reason_b = str(guardrail.get("reason") or "")
        guardrail_orig_signal_b = str(guardrail.get("original_signal") or "").upper()
        guardrail_orig_conf_b   = str(guardrail.get("original_confidence") or "").upper()
        if guardrail.get("applied") and "premium_bleed" in guardrail_reason_b:
            lines.append("")
            change_parts_b = []
            if guardrail_orig_signal_b and guardrail_orig_signal_b != signal:
                orig_e = _SIGNAL_EMOJI.get(guardrail_orig_signal_b, "⬜")
                cur_e  = _SIGNAL_EMOJI.get(signal, "⬜")
                change_parts_b.append(
                    f"{orig_e}`{esc(guardrail_orig_signal_b)}` → {cur_e}`{esc(signal)}`"
                )
            if guardrail_orig_conf_b and guardrail_orig_conf_b != confidence:
                change_parts_b.append(
                    f"신뢰도 {esc(guardrail_orig_conf_b)} → {esc(confidence)}"
                )
            change_str_b = "  ".join(change_parts_b)
            reason_disp_b = esc(guardrail_reason_b[:80])
            if change_str_b:
                lines.append(f"💧 *블리드 가드레일*: {change_str_b}")
                lines.append(f"   `{reason_disp_b}`")
            else:
                lines.append(f"💧 *블리드 가드레일*: `{reason_disp_b}`")
    except Exception:
        pass

    # OI 지지저항 가드레일 알림 — oi_guardrail이 신호를 변경한 경우에만 표시
    # result["guardrail"]["reason"]에 아래 키워드가 포함된 경우:
    #   - oi_zero_gamma_near   : Zero Gamma Level 근접 → MEDIUM→LOW 강등
    #   - oi_vol_trigger_below : Vol Trigger 하방(Dealer Short Gamma) → BUY MEDIUM→LOW
    #   - oi_call_resistance   : 강한 Call OI Peak 직전 → BUY→HOLD
    #   - oi_put_support       : 강한 Put OI Peak 직전 → SELL→HOLD
    _OI_GUARDRAIL_KEYWORDS = (
        "oi_zero_gamma_near",
        "oi_vol_trigger_below",
        "oi_call_resistance",
        "oi_put_support",
    )
    try:
        guardrail_reason_oi = str(guardrail.get("reason") or "")
        guardrail_orig_signal_oi = str(guardrail.get("original_signal") or "").upper()
        guardrail_orig_conf_oi   = str(guardrail.get("original_confidence") or "").upper()
        if guardrail.get("applied") and any(kw in guardrail_reason_oi for kw in _OI_GUARDRAIL_KEYWORDS):
            lines.append("")
            change_parts_oi = []
            if guardrail_orig_signal_oi and guardrail_orig_signal_oi != signal:
                orig_e = _SIGNAL_EMOJI.get(guardrail_orig_signal_oi, "⬜")
                cur_e  = _SIGNAL_EMOJI.get(signal, "⬜")
                change_parts_oi.append(
                    f"{orig_e}`{esc(guardrail_orig_signal_oi)}` → {cur_e}`{esc(signal)}`"
                )
            if guardrail_orig_conf_oi and guardrail_orig_conf_oi != confidence:
                change_parts_oi.append(
                    f"신뢰도 {esc(guardrail_orig_conf_oi)} → {esc(confidence)}"
                )
            change_str_oi = "  ".join(change_parts_oi)
            reason_disp_oi = esc(guardrail_reason_oi[:80])
            if change_str_oi:
                lines.append(f"🎯 *OI 가드레일*: {change_str_oi}")
                lines.append(f"   `{reason_disp_oi}`")
            else:
                lines.append(f"🎯 *OI 가드레일*: `{reason_disp_oi}`")
    except Exception:
        pass

    # expected_amplitude_pt > 0 일 때만 표시 (IV 없는 경우 생략)
    try:
        _amp = result.get("amplitude")
        if isinstance(_amp, dict) and _amp:
            _exp_pt  = float(_amp.get("expected_amplitude_pt")  or 0.0)
            _real_pt = float(_amp.get("realized_hl_range_pt")   or 0.0)
            _exhaust = float(_amp.get("amplitude_exhaustion")    or 0.0)
            _remain  = float(_amp.get("remaining_amplitude_pt")  or 0.0)
            _open_d  = float(_amp.get("open_dist_pct")           or 0.0)
            _s_open  = float(_amp.get("session_open")            or 0.0)

            # 표시 일관성 보정:
            # 업스트림 amplitude dict가 라운드 전환/스냅샷 섞임으로 불일치할 수 있어
            # 텔레그램 표시값은 expected/realized 기준으로 파생값을 재계산한다.
            if _exp_pt > 0.0:
                _exhaust = max(0.0, float(_real_pt / _exp_pt))
                _remain = max(0.0, float(_exp_pt - _real_pt))

            if _exp_pt > 0.0 or _real_pt > 0.0:
                lines.append("")
                lines.append("📏 *당일 진폭*")

                # 예상 vs 실현
                if _exp_pt > 0.0:
                    lines.append(
                        f"  🔮 예상: `{esc(f'{_exp_pt:.1f}pt')}`"
                    )
                if _real_pt > 0.0:
                    lines.append(
                        f"  📊 실현: `{esc(f'{_real_pt:.1f}pt')}`"
                    )

                # 소진율 표시
                if _exp_pt > 0.0 and _real_pt > 0.0:
                    if _exhaust >= 0.8:
                        exhaust_mark = "⚠️"
                    elif _exhaust >= 0.5:
                        exhaust_mark = "🟡"
                    else:
                        exhaust_mark = "🟢"
                    lines.append(
                        f"  {exhaust_mark} 소진율: `{esc(f'{_exhaust:.0%}')}`  "
                        f"잔여: `{esc(f'{_remain:.1f}pt')}`"
                    )
                    if _exhaust > 1.0:
                        lines.append(
                            f"  ⚡ 예상 진폭 초과 \\({esc(f'{_exhaust:.0%}')}\\)"
                        )

                # 시가 대비 현재 위치
                if _s_open > 0.0:
                    _open_arrow = "▲" if _open_d > 0 else ("▼" if _open_d < 0 else "◆")
                    lines.append(
                        f"  {_open_arrow} 시가\\({esc(f'{_s_open:.2f}')}\\) 대비: "
                        f"`{esc(f'{_open_d:+.2f}%')}`"
                    )
    except Exception:
        pass

    # OTM 프리미엄 변화율 블록
    # result["options"]["otm_premium"] 에 calc_otm_premium_change() 결과가 담겨 있음.
    # call_otm_count / put_otm_count 가 모두 0이면 (open_price 미주입 등) 블록 생략.
    try:
        _otm = None
        if isinstance(options, dict):
            _otm = options.get("otm_premium")
        if isinstance(_otm, dict):
            _c_chg = _otm.get("call_otm_prem_chg")   # float or None
            _p_chg = _otm.get("put_otm_prem_chg")   # float or None
            _c_cnt = int(_otm.get("call_otm_count") or 0)
            _p_cnt = int(_otm.get("put_otm_count")  or 0)
            if _c_cnt > 0 or _p_cnt > 0:
                # 헤더 이모지: 콜/풋 방향 합산으로 동적 결정
                _c_val = float(_c_chg or 0.0)
                _p_val = float(_p_chg or 0.0)
                if _c_val > 0.03 and _p_val < -0.03:
                    _hdr_emoji = "📈"   # 콜 팽창 + 풋 수축 = 강세
                elif _c_val < -0.03 and _p_val > 0.03:
                    _hdr_emoji = "📉"   # 콜 수축 + 풋 팽창 = 약세
                else:
                    _hdr_emoji = "📊"   # 혼재 or 중립
                lines.append("")
                lines.append(f"{_hdr_emoji} *OTM 프리미엄*")
                if _c_chg is not None and _c_cnt > 0:
                    _c_arrow = "▲" if _c_chg > 0 else ("▼" if _c_chg < 0 else "◆")
                    lines.append(
                        f"  {_c_arrow} 콜: `{esc(f'{_c_chg:+.1%}')}` "
                        f"\\({esc(str(_c_cnt))}종목\\)"
                    )
                else:
                    lines.append(f"  ◆ 콜: 집계 중 \\({esc('open_price')} 미주입\\)")
                if _p_chg is not None and _p_cnt > 0:
                    _p_arrow = "▲" if _p_chg > 0 else ("▼" if _p_chg < 0 else "◆")
                    lines.append(
                        f"  {_p_arrow} 풋: `{esc(f'{_p_chg:+.1%}')}` "
                        f"\\({esc(str(_p_cnt))}종목\\)"
                    )
                else:
                    lines.append(f"  ◆ 풋: 집계 중 \\({esc('open_price')} 미주입\\)")
    except Exception:
        pass

    # 실시간 수신 누적량은 텔레그램 본문에서 제외(노이즈 억제).

    if include_dir_summary and isinstance(model_outputs, dict) and model_outputs:
        _ACTION_E = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        rows = []
        _order = [
            ("heuristic", "Heuristic"),
            ("transformer", "Transformer"),
            ("tft", "TFT"),
            ("gpt", "GPT"),
            ("gemini", "Gemini"),
            ("claude", "Claude"),
        ]
        for key, label in _order:
            entry = model_outputs.get(key)
            if not isinstance(entry, dict):
                continue
            action = str(entry.get("action") or entry.get("signal") or "").upper()
            if not action:
                continue
            e = _ACTION_E.get(action, "⬜")
            rows.append(f"{e} {esc(label)}: `{esc(action)}`")
        if rows:
            lines.append("")
            lines.append("📋 *\\[DIR\\_SUMMARY\\]*")
            lines.extend([f"  {r}" for r in rows])

    # LLM 공급자
    footer_parts = []
    if llm_provider:
        footer_parts.append(f"LLM: {llm_provider}")
    if llm_timed_out:
        footer_parts.append("⏰ LLM 타임아웃")
    if footer_parts:
        lines.append("")
        lines.append(f"_{esc('  |  '.join(footer_parts))}_")

    text = "\n".join(lines)

    # 텔레그램 4096자 제한 (DS-03)
    limit = _TG_MAX_LEN - len(_TG_TRUNCATE_SUFFIX)
    if len(text) > limit:
        text = text[:limit] + _TG_TRUNCATE_SUFFIX

    return text


def format_premium_bleed_alert(
    opt_snap: Dict[str, Any],
    current_price: float,
    *,
    dte_days: Optional[float] = None,
) -> str:
    """선물 상승 중 옵션 프리미엄 수축 감지 → 텔레그램 MarkdownV2 독립 알림 메시지.

    send_premium_bleed_alert()에서 호출된다.
    예측 사이클과 무관하게 opt_snap 기반으로 즉시 전송하는 알림이다.

    Args:
        opt_snap:      build_option_snapshot() 반환 dict (v4 기준).
        current_price: 현재 선물 가격.
        dte_days:      만기 잔존일 (없으면 dte_weight_norm 역산).
    """
    esc = _esc_mdv2

    try:
        score      = float(opt_snap.get("premium_bleed_score") or 0.0)
        dte_w      = float(opt_snap.get("dte_weight_norm") or 0.0)
        decay      = float(opt_snap.get("straddle_decay_vs_fut") or 0.0)
        iv_crush   = float(opt_snap.get("iv_crush_proxy") or 0.0)
        fut_ret    = float(opt_snap.get("fut_ret") or 0.0)
        straddle_n = float(opt_snap.get("straddle_now") or 0.0)
        straddle_p = float(opt_snap.get("straddle_prev") or 0.0)
    except Exception:
        score = dte_w = decay = iv_crush = fut_ret = straddle_n = straddle_p = 0.0

    now_str = datetime.now().strftime("%H:%M:%S")

    # 강도 이모지
    if abs(score) >= 0.7:
        intensity_emoji = "🔥"
        intensity_str   = "강한"
    elif abs(score) >= 0.5:
        intensity_emoji = "⚠️"
        intensity_str   = "중간"
    else:
        intensity_emoji = "📎"
        intensity_str   = "약한"

    fut_dir   = "상승 📈" if fut_ret > 0 else ("하락 📉" if fut_ret < 0 else "횡보 ↔️")
    bleed_dir = "수축 💧" if score < 0 else "팽창 🔥"

    # 만기 잔존일 표시 (직접 DTE 우선, 표시는 반올림 대신 내림)
    try:
        if dte_days is not None:
            _d = max(0.0, float(dte_days))
            dte_str = f"{int(math.floor(_d))}일"
        elif dte_w > 0.0:
            _d = max(0.0, float(1.0 / (dte_w * 10.0)))
            dte_str = f"{int(math.floor(_d))}일"
        else:
            dte_str = "?"
    except Exception:
        dte_str = "?"

    lines = [
        f"{intensity_emoji} *프리미엄 블리드 알림* \\| {esc(now_str)}",
        "",
        f"💰 *선물가*: `{esc(f'{current_price:.2f}')}`  \\|  *만기*: `{esc(dte_str)} 전`",
        f"📊 *선물 방향*: {fut_dir}  \\|  *스트래들*: {bleed_dir}",
        f"🎯 *블리드 스코어*: `{esc(f'{score:+.2f}')}`  \\({intensity_str} 수축\\)",
        "",
    ]

    # 스트래들 변화
    if straddle_p > 0.0:
        straddle_ret_pct = (straddle_n - straddle_p) / straddle_p * 100.0
        lines.append(
            f"📉 *스트래들*: `{esc(f'{straddle_p:.2f}')}` → `{esc(f'{straddle_n:.2f}')}` "
            f"\\({esc(f'{straddle_ret_pct:+.2f}')}%\\)"
        )

    # 선물 수익률
    if abs(fut_ret) >= 0.0003:
        lines.append(f"📈 *선물 수익률*: `{esc(f'{fut_ret:+.4%}')}`")

    # 수축 지표
    if abs(decay) >= 0.002:
        lines.append(f"💧 *Decay vs Fut*: `{esc(f'{decay:+.4f}')}`")

    # IV Crush
    if abs(iv_crush) >= 0.01:
        iv_label = "IV Crush ⬇️" if iv_crush < 0 else "IV 급등 ⬆️"
        lines.append(f"🌊 *{iv_label}*: `{esc(f'{iv_crush:+.2%}')}`")

    lines.append("")

    # 해석
    if score < -0.5 and fut_ret > 0:
        lines.append(
            "💡 *해석*: 선물 상승 중 옵션 프리미엄 비정상 수축\\."
        )
        lines.append(
            "   Theta 급가속 / IV Crush / MM 롤오버 가능\\."
        )
        lines.append(
            "   → 방향성 신규 진입 자제, 기존 포지션 청산 국면 가능\\."
        )
    elif score > 0.5:
        lines.append("💡 *해석*: 프리미엄 급팽창\\. 방향성 예측 어려운 구간\\.")

    text = "\n".join(lines)

    # 4096자 제한
    limit = _TG_MAX_LEN - len(_TG_TRUNCATE_SUFFIX)
    if len(text) > limit:
        text = text[:limit] + _TG_TRUNCATE_SUFFIX

    return text


def format_price_level_touch_alert(
    opt_snap: Dict[str, Any],
    current_price: float,
) -> str:
    """옵션 가격 레벨 터치 감지 → 텔레그램 MarkdownV2 독립 알림 메시지.

    build_option_snapshot()의 '_price_level_scan' 키를 소비한다.
    터치 항목이 없으면 빈 문자열을 반환한다.

    각 터치 항목은 "행사가  가격=X.XX  @HHMM" 형식으로 한 줄씩 표시된다.

    Args:
        opt_snap:      build_option_snapshot() 반환 dict.
        current_price: 현재 선물 가격(pt).
    """
    scan = opt_snap.get("_price_level_scan")
    if not isinstance(scan, dict) or not scan.get("has_hit"):
        return ""

    esc = _esc_mdv2

    call_hits: list = scan.get("call_hits") or []
    put_hits:  list = scan.get("put_hits")  or []
    levels:    list = scan.get("levels_used") or []

    now_str = datetime.now().strftime("%H:%M:%S")
    lv_str  = " \\| ".join(esc(f"{lv:.2f}") for lv in sorted(levels))

    lines = [
        f"🎯 *옵션 레벨 터치* \\| {esc(now_str)}",
        "",
        f"💰 *선물가*: `{esc(f'{current_price:.2f}')}`",
        f"📌 *탐색 레벨*: {lv_str}",
        "",
    ]

    _FIELD_KR    = {"high": "고가", "low": "저가", "price": "현재가"}
    _FIELD_EMOJI = {"high": "⬆️", "low": "⬇️", "price": "⏺"}

    def _render_hits(hits: list, header: str) -> None:
        """hits 목록을 레벨별로 그룹핑하여 lines에 추가한다.

        각 항목 형식:
            ⬆️ 행사가 285.0  고가=2.50  @0911
        """
        if not hits:
            return
        lines.append(header)
        by_lv: Dict[float, list] = {}
        for h in hits:
            by_lv.setdefault(float(h["level"]), []).append(h)
        for lv in sorted(by_lv):
            grp = by_lv[lv]
            fld_emoji = _FIELD_EMOJI.get(grp[0]["field"], "⏺")
            fld_kr    = _FIELD_KR.get(grp[0]["field"], grp[0]["field"])
            lv_esc    = esc(f"{lv:.2f}")
            lines.append(f"  {fld_emoji} 레벨 `{lv_esc}`")
            for h in grp:
                strike_esc = esc(f"{h['strike']:.1f}")
                val_esc    = esc(f"{h['value']:.2f}")
                t_str = h.get("time_str") or ""
                time_part  = f"  `@{esc(t_str)}`" if t_str else ""
                lines.append(
                    f"    행사가 `{strike_esc}`  {fld_kr}\\={val_esc}{time_part}"
                )

    _render_hits(call_hits, "📞 *콜 터치*")
    if call_hits and put_hits:
        lines.append("")
    _render_hits(put_hits, "🔵 *풋 터치*")

    # ── 시사점 ─────────────────────────────────────────────────────────────
    implications: list = []
    if any(h["field"] == "high" for h in call_hits):
        implications.append("콜 고가 레벨 → 저항·매도 집중 가능")
    if any(h["field"] == "low" for h in put_hits):
        implications.append("풋 저가 레벨 → 하방 지지선 테스트")
    if any(h["field"] == "low" for h in call_hits):
        implications.append("콜 저가 레벨 → 프리미엄 수축 진행")
    if any(h["field"] == "high" for h in put_hits):
        implications.append("풋 고가 레벨 → 하방 베팅 강화 신호")
    hit_lv_c = scan.get("hit_levels_call") or set()
    hit_lv_p = scan.get("hit_levels_put")  or set()
    if len(hit_lv_c) >= 2 or len(hit_lv_p) >= 2:
        implications.append("다수 레벨 동시 터치 → 변동성 확대 경계")

    if implications:
        lines.append("")
        lines.append("💡 " + esc(implications[0]))
        for imp in implications[1:]:
            lines.append("   " + esc(imp))

    text = "\n".join(lines)
    limit = _TG_MAX_LEN - len(_TG_TRUNCATE_SUFFIX)
    if len(text) > limit:
        text = text[:limit] + _TG_TRUNCATE_SUFFIX

    return text


def format_futures_call_divergence_alert(
    cds_result: Dict[str, Any],
    current_price: float,
    atm_strike: float,
    *,
    dte_days: Optional[float] = None,
) -> str:
    """선물-ATM 콜 추적 이탈(CDS) 감지 → 텔레그램 MarkdownV2 독립 알림 메시지.

    send_futures_call_divergence_alert()에서 호출된다.

    Args:
        cds_result:    FuturesCallSimilarity.composite_divergence_score() 반환 dict.
        current_price: 현재 선물가.
        atm_strike:    ATM 행사가.
        dte_days:      만기 잔존일 (None 가능).
    """
    esc = _esc_mdv2

    try:
        cds    = float(cds_result.get("cds")  or 0.0)
        corr   = float(cds_result.get("corr") or 0.0)
        dtw    = float(cds_result.get("dtw")  or 0.0)
        r2     = float(cds_result.get("r2")   or 0.0)
        beta   = float(cds_result.get("beta") or 0.5)
        n_samp = int(cds_result.get("n_samples") or 0)
    except Exception:
        cds = corr = dtw = r2 = 0.0
        beta = 0.5
        n_samp = 0

    now_str = datetime.now().strftime("%H:%M:%S")

    # 강도 등급
    if cds >= 0.7:
        emoji, grade = "🔴", "위험"
    elif cds >= 0.5:
        emoji, grade = "🟠", "경보"
    elif cds >= 0.3:
        emoji, grade = "🟡", "주의"
    else:
        emoji, grade = "🟢", "정상"

    # R² 감마 진단
    if r2 < 0.3:
        r2_diag = "감마 완전 지배 ⚡"
    elif r2 < 0.5:
        r2_diag = "감마 지배 시작 ⚠️"
    elif r2 < 0.7:
        r2_diag = "감마 영향 증가 📈"
    else:
        r2_diag = "선형 추적 양호 ✅"

    # DTE 표시
    try:
        if dte_days is not None:
            _d = max(0.0, float(dte_days))
            dte_str = f"{int(math.floor(_d))}일"
        else:
            dte_str = "?"
    except Exception:
        dte_str = "?"

    lines = [
        f"{emoji} *선물\\-콜 이탈 알림* \\| {esc(now_str)}",
        "",
        f"💰 *선물가*: `{esc(f'{current_price:.2f}')}`  \\|  "
        f"*ATM*: `{esc(f'{atm_strike:.1f}')}`  \\|  *만기*: `{esc(dte_str)} 전`",
        "",
        f"🎯 *CDS\\(이탈 스코어\\)*: `{esc(f'{cds:.3f}')}` — *{esc(grade)}*",
        "",
        f"📐 *상관계수*: `{esc(f'{corr:+.3f}')}` \\(1\\=완전추적\\)",
        f"📏 *DTW 거리*:  `{esc(f'{dtw:.3f}')}` \\(0\\=형태일치\\)",
        f"📉 *R²\\(선형도\\)*: `{esc(f'{r2:.3f}')}` — {esc(r2_diag)}",
        f"📊 *Beta\\(콜/선물\\)*: `{esc(f'{beta:.3f}')}`  \\|  *샘플*: `{esc(str(n_samp))}`",
        "",
    ]

    # 매매 조언 메시지
    if cds >= 0.5:
        lines.append("⛔ *방향성 콜 매매 주의* — 감마 비선형 구간")
    elif cds >= 0.3:
        lines.append("⚠️ *콜 포지션 리스크 증가* — 모니터링 강화")
    else:
        lines.append("✅ *선물\\-콜 추적 정상*")

    return "\n".join(lines)


def format_error_message(result: Dict[str, Any]) -> str:
    """에러 결과 dict → 텔레그램 메시지."""
    error = str(result.get("error", "unknown"))
    message = str(result.get("message", ""))
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"🚨 *예측 오류* \\| {now}\n"
        f"`{_esc_mdv2(error)}`\n"
        f"{_esc_mdv2(message[:300])}"
    )


# ──────────────────────────────────────────────
# TelegramNotifier
# ──────────────────────────────────────────────

