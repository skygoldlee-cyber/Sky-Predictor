"""gui_controller_rt_helpers 순수 로직 (PySide6 불필요)."""

from __future__ import annotations

from gui.controller_rt_helpers import (
    fc0_is_stale,
    format_rt_status_line,
    predictor_metrics_summary_strings,
    sr_label_text_from_tick_stats,
)


def test_predictor_metrics_summary_strings_empty() -> None:
    d = predictor_metrics_summary_strings(None)
    assert d["fc0_age"] == "-"
    assert d["fb_snap"] == "-"


def test_predictor_metrics_summary_strings_fc0() -> None:
    d = predictor_metrics_summary_strings({"fc0_age_sec": 2.0})
    assert d["fc0_age"] == "2.0s"


def test_fc0_is_stale_not_stale() -> None:
    eff = {"fc0_stale_threshold_sec": 10.0}
    assert (
        fc0_is_stale(
            stale_thr_edit_text="",
            effective_pred_defaults=eff,
            metrics={"fc0_age_sec": 5.0},
        )
        is False
    )


def test_fc0_is_stale_over_threshold() -> None:
    eff = {"fc0_stale_threshold_sec": 10.0}
    assert (
        fc0_is_stale(
            stale_thr_edit_text="",
            effective_pred_defaults=eff,
            metrics={"fc0_age_sec": 15.0},
        )
        is True
    )


def test_format_rt_status_line_minimal() -> None:
    s = format_rt_status_line({"counts": {"FC0": 1, "FH0": 2, "JIF": 0}})
    assert "RT FC0=1" in s
    assert "FH0=2" in s


def test_sr_label_text_from_tick_stats_with_total() -> None:
    t = sr_label_text_from_tick_stats(
        {"eval_dir_total": 10, "eval_dir_hits": 7, "eval_dir_rate": 70.0, "eval_hold_count": 1}
    )
    assert "70.0%" in t
    assert "7/10" in t
    assert "HOLD=1" in t


def test_sr_label_text_from_tick_stats_empty() -> None:
    assert "SR: -" in sr_label_text_from_tick_stats({})


def test_fc0_is_stale_edit_overrides() -> None:
    eff = {"fc0_stale_threshold_sec": 10.0}
    assert (
        fc0_is_stale(
            stale_thr_edit_text="3",
            effective_pred_defaults=eff,
            metrics={"fc0_age_sec": 5.0},
        )
        is True
    )
