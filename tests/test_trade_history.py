"""
tests/test_trade_history.py
===========================
trade_history_viewer.py 단위 테스트 (43개).

외부 의존성 없이 tmp_path 픽스처만 사용한다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading.history_viewer import (
    _Stats,
    _grp_update,
    _group_by_confidence,
    _group_by_iv,
    _group_by_reason,
    _group_by_side,
    _group_by_slot,
    _iv_bucket,
    _iter_jsonl,
    _load_all,
    _load_date,
    _load_range,
    best_worst_days,
    cumulative_pnl,
    daily_pnl_series,
    load_history,
    main,
    summary_stats,
)


# ════════════════════════════════════════════════════════════════════════════
# 테스트 데이터 픽스처
# ════════════════════════════════════════════════════════════════════════════

def _make_record(
    *,
    date_str: str = "2026-03-22",
    side: str = "LONG",
    slot: str = "A",
    pnl: float = 2.0,
    close_reason: str = "목표수익",
    confidence: str = "HIGH",
    prob: float = 0.75,
    atm_iv: float = 0.20,
    hold_minutes: float = 30.0,
) -> dict:
    entry_time = f"{date_str}T10:00:00.000000"
    close_time = f"{date_str}T10:30:00.000000"
    entry_price = 382.0
    close_price = (entry_price + pnl) if side == "LONG" else (entry_price - pnl)
    tid = datetime.now().strftime("%Y%m%d_%H%M%S_") + "000000"
    return {
        "trade_id":              tid,
        "slot":                  slot,
        "side":                  side,
        "entry_price":           entry_price,
        "entry_time":            entry_time,
        "entry_signal":          "BUY" if side == "LONG" else "SELL",
        "entry_confidence":      confidence,
        "entry_prob":            prob,
        "entry_atm_iv":          atm_iv,
        "entry_atm_delta":       0.5,
        "entry_net_gamma":       1.0,
        "entry_above_vol_trigger": 1.0,
        "entry_target_pt":       2.0,
        "entry_stop_pt":         1.0,
        "close_price":           close_price,
        "close_time":            close_time,
        "close_reason":          close_reason,
        "pnl_pt":                pnl,
        "hold_minutes":          hold_minutes,
    }


def _write_jsonl(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_history(tmp_path: Path, days_records: dict) -> Path:
    """days_records = {"YYYY-MM-DD": [record, ...]}"""
    for ds, recs in days_records.items():
        _write_jsonl(tmp_path / f"{ds}.jsonl", recs)
    return tmp_path


# ════════════════════════════════════════════════════════════════════════════
# _iter_jsonl
# ════════════════════════════════════════════════════════════════════════════

class TestIterJsonl:

    def test_reads_all_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, [_make_record(), _make_record(pnl=-1.0)])
        result = list(_iter_jsonl(path))
        assert len(result) == 2

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        result = list(_iter_jsonl(path))
        assert len(result) == 2

    def test_missing_file_yields_nothing(self, tmp_path):
        result = list(_iter_jsonl(tmp_path / "missing.jsonl"))
        assert result == []

    def test_malformed_line_skipped(self, tmp_path, capsys):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"ok":1}\nNOT_JSON\n{"ok":2}\n', encoding="utf-8")
        result = list(_iter_jsonl(path))
        assert len(result) == 2


# ════════════════════════════════════════════════════════════════════════════
# _load_date / _load_range / _load_all
# ════════════════════════════════════════════════════════════════════════════

class TestLoaders:

    def test_load_date_found(self, tmp_path):
        _write_jsonl(tmp_path / "2026-03-22.jsonl", [_make_record()])
        result = _load_date(tmp_path, "2026-03-22")
        assert len(result) == 1

    def test_load_date_missing(self, tmp_path):
        result = _load_date(tmp_path, "2026-03-22")
        assert result == []

    def test_load_range_includes_endpoints(self, tmp_path):
        from datetime import date
        _write_jsonl(tmp_path / "2026-03-20.jsonl", [_make_record(date_str="2026-03-20")])
        _write_jsonl(tmp_path / "2026-03-22.jsonl", [_make_record(date_str="2026-03-22")])
        result = _load_range(tmp_path, date(2026, 3, 20), date(2026, 3, 22))
        assert "2026-03-20" in result
        assert "2026-03-22" in result

    def test_load_range_excludes_outside(self, tmp_path):
        from datetime import date
        _write_jsonl(tmp_path / "2026-03-19.jsonl", [_make_record(date_str="2026-03-19")])
        result = _load_range(tmp_path, date(2026, 3, 20), date(2026, 3, 22))
        assert "2026-03-19" not in result

    def test_load_all_multiple_files(self, tmp_path):
        for ds in ["2026-03-20", "2026-03-21", "2026-03-22"]:
            _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds)])
        result = _load_all(tmp_path)
        assert len(result) == 3

    def test_load_all_empty_dir(self, tmp_path):
        result = _load_all(tmp_path)
        assert result == {}

    def test_load_all_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "README.txt").write_text("ignore me")
        (tmp_path / "2026-03-22.jsonl").write_text(
            json.dumps(_make_record()) + "\n"
        )
        result = _load_all(tmp_path)
        assert len(result) == 1


# ════════════════════════════════════════════════════════════════════════════
# _Stats
# ════════════════════════════════════════════════════════════════════════════

class TestStats:

    def test_empty(self):
        st = _Stats([])
        assert st.count == 0
        assert st.wins == 0
        assert st.win_rate == 0.0
        assert st.total_pnl == 0.0
        assert st.profit_factor is None

    def test_all_wins(self):
        records = [_make_record(pnl=2.0) for _ in range(3)]
        st = _Stats(records)
        assert st.wins == 3
        assert st.losses == 0
        assert st.win_rate == pytest.approx(1.0)
        assert st.total_pnl == pytest.approx(6.0)
        assert st.profit_factor is None   # 손실=0 → None

    def test_all_losses(self):
        records = [_make_record(pnl=-1.0) for _ in range(2)]
        st = _Stats(records)
        assert st.losses == 2
        assert st.win_rate == pytest.approx(0.0)
        assert st.total_pnl == pytest.approx(-2.0)
        assert st.profit_factor == pytest.approx(0.0)

    def test_mixed(self):
        records = [
            _make_record(pnl=2.0),
            _make_record(pnl=-1.0),
            _make_record(pnl=3.0),
        ]
        st = _Stats(records)
        assert st.wins == 2
        assert st.losses == 1
        assert st.win_rate == pytest.approx(2 / 3)
        assert st.total_pnl == pytest.approx(4.0)
        assert st.avg_pnl == pytest.approx(4.0 / 3)
        assert st.profit_factor == pytest.approx(5.0 / 1.0)

    def test_avg_win_avg_loss(self):
        records = [
            _make_record(pnl=4.0),
            _make_record(pnl=2.0),
            _make_record(pnl=-1.0),
            _make_record(pnl=-3.0),
        ]
        st = _Stats(records)
        assert st.avg_win  == pytest.approx(3.0)
        assert st.avg_loss == pytest.approx(-2.0)

    def test_max_win_max_loss(self):
        records = [_make_record(pnl=v) for v in [1.0, 5.0, -2.0, -0.5]]
        st = _Stats(records)
        assert st.max_win  == pytest.approx(5.0)
        assert st.max_loss == pytest.approx(-2.0)

    def test_avg_hold(self):
        records = [_make_record(hold_minutes=m) for m in [10.0, 20.0, 30.0]]
        st = _Stats(records)
        assert st.avg_hold == pytest.approx(20.0)

    def test_to_dict_keys(self):
        st = _Stats([_make_record(pnl=2.0)])
        d = st.to_dict()
        for key in ("count", "wins", "losses", "draws", "win_rate",
                    "total_pnl_pt", "avg_pnl_pt", "avg_win_pt", "avg_loss_pt",
                    "max_win_pt", "max_loss_pt", "profit_factor", "avg_hold_min"):
            assert key in d, f"누락 키: {key}"


# ════════════════════════════════════════════════════════════════════════════
# 그룹 함수
# ════════════════════════════════════════════════════════════════════════════

class TestGroupFunctions:

    def test_grp_update(self):
        grps: dict = {}
        grps["A"] = []
        _grp_update(grps, "A", {"x": 1})
        _grp_update(grps, "A", {"x": 2})
        assert len(grps["A"]) == 2

    def test_group_by_slot(self):
        records = [
            _make_record(slot="A"),
            _make_record(slot="B"),
            _make_record(slot="A"),
        ]
        grps = _group_by_slot(records)
        assert "A" in grps and "B" in grps
        assert grps["A"].count == 2
        assert grps["B"].count == 1

    def test_group_by_reason(self):
        records = [
            _make_record(close_reason="목표수익"),
            _make_record(close_reason="손절"),
            _make_record(close_reason="목표수익"),
        ]
        grps = _group_by_reason(records)
        assert grps["목표수익"].count == 2
        assert grps["손절"].count == 1

    def test_group_by_side(self):
        records = [
            _make_record(side="LONG"),
            _make_record(side="SHORT", pnl=-1.0),
        ]
        grps = _group_by_side(records)
        assert "LONG" in grps and "SHORT" in grps

    def test_group_by_confidence_order(self):
        records = [
            _make_record(confidence="LOW"),
            _make_record(confidence="HIGH"),
            _make_record(confidence="MEDIUM"),
        ]
        grps = _group_by_confidence(records)
        keys = list(grps.keys())
        assert keys.index("HIGH") < keys.index("MEDIUM") < keys.index("LOW")

    def test_iv_bucket_labels(self):
        assert _iv_bucket(0.0)    == "데이터없음"
        assert _iv_bucket(0.05)   == "IV<10%"
        assert _iv_bucket(0.12)   == "10%≤IV<15%"
        assert _iv_bucket(0.17)   == "15%≤IV<20%"
        assert _iv_bucket(0.22)   == "20%≤IV<25%"
        assert _iv_bucket(0.30)   == "IV≥25%"

    def test_group_by_iv(self):
        records = [
            _make_record(atm_iv=0.05),
            _make_record(atm_iv=0.20),
            _make_record(atm_iv=0.05),
        ]
        grps = _group_by_iv(records)
        assert grps["IV<10%"].count == 2
        assert grps["20%≤IV<25%"].count == 1


# ════════════════════════════════════════════════════════════════════════════
# 편의 API
# ════════════════════════════════════════════════════════════════════════════

class TestConvenienceAPI:

    def test_load_history_all(self, tmp_path):
        for ds in ["2026-03-20", "2026-03-21"]:
            _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds)])
        data = load_history(str(tmp_path))
        assert len(data) == 2

    def test_load_history_days(self, tmp_path):
        from datetime import date
        today = date.today()
        ds = today.isoformat()
        _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds)])
        data = load_history(str(tmp_path), days=1)
        assert ds in data

    def test_summary_stats_empty(self, tmp_path):
        st = summary_stats(str(tmp_path))
        assert st["count"] == 0
        assert st["win_rate"] == 0.0

    def test_summary_stats_non_empty(self, tmp_path):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(pnl=2.0), _make_record(pnl=-1.0)],
        )
        st = summary_stats(str(tmp_path))
        assert st["count"] == 2
        assert st["total_pnl_pt"] == pytest.approx(1.0)

    def test_daily_pnl_series_sorted(self, tmp_path):
        for ds, pnl in [("2026-03-21", -1.0), ("2026-03-22", 2.0)]:
            _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds, pnl=pnl)])
        series = daily_pnl_series(str(tmp_path))
        assert series[0][0] < series[1][0]  # 날짜 오름차순
        assert series[0][1] == pytest.approx(-1.0)
        assert series[1][1] == pytest.approx(2.0)

    def test_cumulative_pnl(self, tmp_path):
        for ds, pnl in [("2026-03-20", 1.0), ("2026-03-21", 2.0), ("2026-03-22", -1.0)]:
            _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds, pnl=pnl)])
        series = cumulative_pnl(str(tmp_path))
        assert series[0][1] == pytest.approx(1.0)
        assert series[1][1] == pytest.approx(3.0)
        assert series[2][1] == pytest.approx(2.0)

    def test_best_worst_days(self, tmp_path):
        for ds, pnl in [
            ("2026-03-18", 5.0),
            ("2026-03-19", 3.0),
            ("2026-03-20", -2.0),
            ("2026-03-21", -4.0),
            ("2026-03-22", 1.0),
        ]:
            _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds, pnl=pnl)])
        best, worst = best_worst_days(str(tmp_path), n=2)
        assert best[0][0]  == "2026-03-18"   # 최고 +5.0
        assert worst[0][0] == "2026-03-21"   # 최저 -4.0


# ════════════════════════════════════════════════════════════════════════════
# CLI main()
# ════════════════════════════════════════════════════════════════════════════

class TestCLIMain:

    def test_main_no_data_returns_zero(self, tmp_path, capsys):
        ret = main(["--dir", str(tmp_path), "--days", "1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "이력이 없습니다" in out

    def test_main_with_data(self, tmp_path, capsys):
        from datetime import date
        ds = date.today().isoformat()
        _write_jsonl(tmp_path / f"{ds}.jsonl", [_make_record(date_str=ds, pnl=2.0)])
        ret = main(["--dir", str(tmp_path), "--days", "1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert ds in out

    def test_main_json_output(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(pnl=2.0), _make_record(pnl=-1.0)],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--json"])
        assert ret == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "overall" in parsed
        assert "daily" in parsed
        assert parsed["overall"]["count"] == 2

    def test_main_invalid_dir_returns_one(self, capsys):
        ret = main(["--dir", "/nonexistent_path_xyz", "--all"])
        assert ret == 1

    def test_main_invalid_date_format(self, tmp_path, capsys):
        ret = main(["--dir", str(tmp_path), "--date", "not-a-date"])
        assert ret == 1

    def test_main_slot_flag(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(slot="A"), _make_record(slot="B")],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--slot"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "슬롯" in out

    def test_main_json_with_slot(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(slot="A"), _make_record(slot="B")],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--json", "--slot"])
        assert ret == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "overall_by_slot" in parsed

    def test_main_verbose(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(pnl=2.0)],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--verbose"])
        assert ret == 0
        out = capsys.readouterr().out
        # verbose 모드에서는 진입가(382.00) 가 출력돼야 함
        assert "382" in out

    def test_main_reason_flag(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(close_reason="목표수익"), _make_record(close_reason="손절", pnl=-1.0)],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--reason"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "청산 사유" in out

    def test_main_json_with_iv(self, tmp_path, capsys):
        _write_jsonl(
            tmp_path / "2026-03-22.jsonl",
            [_make_record(atm_iv=0.20), _make_record(atm_iv=0.05)],
        )
        ret = main(["--dir", str(tmp_path), "--all", "--json", "--iv"])
        assert ret == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "overall_by_iv" in parsed
