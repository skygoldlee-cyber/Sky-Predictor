"""실시간 상태·리플레이 다이얼로그 헬퍼 (`gui_controller` 10·11단계 분리)."""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "open_replay_ticks_file_dialog",
    "fc0_is_stale",
    "predictor_metrics_summary_strings",
    "format_rt_status_line",
]


def open_replay_ticks_file_dialog(parent: Any, cwd: str) -> str:
    """리플레이 틱 파일 선택. 취소 시 빈 문자열."""
    try:
        from PySide6.QtWidgets import QFileDialog

        p, _ = QFileDialog.getOpenFileName(
            parent,
            "Select replay ticks file",
            cwd,
            "Tick logs (*.jsonl *.gz);;JSONL (*.jsonl);;GZip (*.gz);;All (*.*)",
        )
        return str(p or "").strip()
    except Exception:
        return ""


def fc0_is_stale(
    *,
    stale_thr_edit_text: str,
    effective_pred_defaults: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
) -> bool:
    """FC0 나이가 UI/기본 임계값보다 크면 True."""
    stale_thr_eff = None
    try:
        s2 = str(stale_thr_edit_text or "").strip()
        if s2:
            stale_thr_eff = float(s2)
        else:
            stale_thr_eff = float(effective_pred_defaults.get("fc0_stale_threshold_sec") or 10.0)
    except Exception:
        stale_thr_eff = None

    fc0_age_val = None
    try:
        if metrics is not None and metrics.get("fc0_age_sec") is not None:
            fc0_age_val = float(metrics.get("fc0_age_sec"))
    except Exception:
        fc0_age_val = None

    if stale_thr_eff is not None and fc0_age_val is not None:
        return float(fc0_age_val) > float(stale_thr_eff)
    return False


def predictor_metrics_summary_strings(metrics: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """요약 패널용 FC0 문자열 (미수신 시 '-')."""
    out = {
        "fc0_age": "-",
    }
    if not metrics:
        return out
    m = metrics
    try:
        v = m.get("fc0_age_sec")
        if v is not None:
            out["fc0_age"] = f"{float(v):.1f}s"
    except Exception:
        pass
    return out


def format_rt_status_line(
    st: Optional[Dict[str, Any]],
    *,
    predictor_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """상단 ``status_lbl`` 한 줄 (TG 라벨은 호출부에서 별도 처리)."""
    st = st if isinstance(st, dict) else {}
    c = (st or {}).get("counts") or {}
    eval_dir_hits = (st or {}).get("eval_dir_hits")
    eval_dir_total = (st or {}).get("eval_dir_total")
    eval_dir_rate = (st or {}).get("eval_dir_rate")

    fut_now = (st or {}).get("fut_now")
    fut_5m = (st or {}).get("fut_5m")
    call_now = (st or {}).get("call_now")
    put_now = (st or {}).get("put_now")
    oc0_call_count = (st or {}).get("oc0_call_count")
    oc0_put_count = (st or {}).get("oc0_put_count")
    oh0_call_count = (st or {}).get("oh0_call_count")
    oh0_put_count = (st or {}).get("oh0_put_count")
    spot_idx = (st or {}).get("spot_index")
    spot_time = (st or {}).get("spot_time")

    s = (
        f"RT "
        f"OC0(C/P)={int(oc0_call_count or 0)}/{int(oc0_put_count or 0)} "
        f"OH0(C/P)={int(oh0_call_count or 0)}/{int(oh0_put_count or 0)} "
    )
    
    # KOSPI 실시간 지수 표시 (spot_idx)
    try:
        if spot_idx is not None and float(spot_idx or 0.0) > 0.0:
            s += f" KOSPI={float(spot_idx):.2f}"
    except Exception:
        pass
    try:
        s += f" | fut_5m_ago={float(fut_5m or 0.0):.2f}"
        s += f" | fut_now={float(fut_now or 0.0):.2f}"
        s += f" | call_now={float(call_now or 0.0):.2f}"
        s += f" | put_now={float(put_now or 0.0):.2f}"
    except Exception:
        pass

    try:
        if spot_idx is not None and float(spot_idx or 0.0) > 0.0:
            s += f" | spot={float(spot_idx):.2f}"
            try:
                if spot_time:
                    s += f"({str(spot_time)})"
            except Exception:
                pass
            try:
                if fut_now is not None and float(fut_now or 0.0) > 0.0:
                    b = float(fut_now) - float(spot_idx)
                    s += f" | basis={float(b):+.2f}"
            except Exception:
                pass
    except Exception:
        pass

    try:
        if int(eval_dir_total or 0) > 0:
            s += f" | DIR={float(eval_dir_rate or 0.0):.1f}% ({int(eval_dir_hits or 0)}/{int(eval_dir_total or 0)})"
    except Exception:
        pass

    return s


def sr_label_text_from_tick_stats(st: Dict[str, Any]) -> str:
    """평가 방향 통계용 ``sr_lbl`` 텍스트."""
    if not isinstance(st, dict):
        st = {}
    eval_dir_hits = st.get("eval_dir_hits")
    eval_dir_total = st.get("eval_dir_total")
    eval_dir_rate = st.get("eval_dir_rate")
    eval_hold_count = st.get("eval_hold_count")
    try:
        if int(eval_dir_total or 0) > 0:
            hold_s = ""
            try:
                if int(eval_hold_count or 0) > 0:
                    hold_s = f" | HOLD={int(eval_hold_count or 0)}"
            except Exception:
                hold_s = ""
            return (
                f"SR: {float(eval_dir_rate or 0.0):.1f}% "
                f"({int(eval_dir_hits or 0)}/{int(eval_dir_total or 0)}){hold_s}"
            )
        try:
            if int(eval_hold_count or 0) > 0:
                return f"SR: - | HOLD={int(eval_hold_count or 0)}"
            return "SR: -"
        except Exception:
            return "SR: -"
    except Exception:
        return "SR: -"
