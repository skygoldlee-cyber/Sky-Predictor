"""
trade_history_viewer.py
=======================
trade_history/ 디렉토리의 JSONL 거래 이력을 읽어
CLI에서 분석·조회하는 도구.

사용법 예시
-----------
    # 최근 7일 요약
    python trade_history_viewer.py --days 7

    # 특정 날짜 상세 조회
    python trade_history_viewer.py --date 2026-03-22

    # 전체 통계 (모든 날짜)
    python trade_history_viewer.py --all

    # 슬롯별 성과 분석
    python trade_history_viewer.py --slot

    # 청산 사유별 분석
    python trade_history_viewer.py --reason

    # JSON 출력 (파이프라인 연동용)
    python trade_history_viewer.py --all --json

    # 디렉토리 지정
    python trade_history_viewer.py --dir /data/trade_history --all

옵션
----
    --dir DIR           JSONL 파일이 저장된 디렉토리 (기본: trade_history)
    --date DATE         특정 날짜 조회 (YYYY-MM-DD)
    --days N            최근 N일 조회 (기본: 7)
    --all               전체 기간 조회
    --slot              슬롯(A/B/C)별 통계 출력
    --reason            청산 사유별 통계 출력
    --side              방향(Long/Short)별 통계 출력
    --conf              confidence 등급별 통계 출력
    --iv                ATM IV 구간별 통계 출력 (Phase 3)
    --json              결과를 JSON 형식으로 출력
    --verbose / -v      각 거래 상세 출력
    --help / -h         도움말
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# ── 의존성: trade_state (같은 프로젝트 내) ───────────────────────────────────
try:
    from .state import CloseReason, PositionSide, TradeRecord, TradeSlot
except ImportError:
    # 독립 실행 시 폴백 — trade_state 없이도 기본 기능 동작
    TradeRecord = None  # type: ignore[assignment,misc]


# ════════════════════════════════════════════════════════════════════════════
# JSONL 로더
# ════════════════════════════════════════════════════════════════════════════

def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    """JSONL 파일에서 한 줄씩 dict 를 yield 한다."""
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"  [WARN] {path.name}:{lineno} JSON 파싱 오류 — {e}",
                        file=sys.stderr,
                    )
    except OSError as e:
        print(f"  [WARN] {path} 읽기 실패 — {e}", file=sys.stderr)


def _load_date(history_dir: Path, date_str: str) -> List[dict]:
    """특정 날짜의 JSONL 파일을 로드한다."""
    path = history_dir / f"{date_str}.jsonl"
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


def _load_range(
    history_dir: Path,
    start: date,
    end: date,
) -> Dict[str, List[dict]]:
    """[start, end] 날짜 범위의 모든 JSONL 파일을 로드한다.

    Returns:
        ``{"YYYY-MM-DD": [record_dict, ...], ...}``
    """
    result: Dict[str, List[dict]] = {}
    current = start
    while current <= end:
        ds = current.isoformat()
        records = _load_date(history_dir, ds)
        if records:
            result[ds] = records
        current += timedelta(days=1)
    return result


def _load_all(history_dir: Path) -> Dict[str, List[dict]]:
    """디렉토리의 모든 JSONL 파일을 로드한다."""
    result: Dict[str, List[dict]] = {}
    if not history_dir.exists():
        return result
    for path in sorted(history_dir.glob("????-??-??.jsonl")):
        ds = path.stem
        records = list(_iter_jsonl(path))
        if records:
            result[ds] = records
    return result


# ════════════════════════════════════════════════════════════════════════════
# 통계 계산
# ════════════════════════════════════════════════════════════════════════════

class _Stats:
    """거래 기록 집합의 통계 계산 헬퍼."""

    __slots__ = (
        "records", "_wins", "_losses", "_draws",
        "_pnl_pts", "_hold_mins",
    )

    def __init__(self, records: List[dict]) -> None:
        self.records  = records
        wins, losses, draws = 0, 0, 0
        pnl_pts: List[float] = []
        hold_mins: List[float] = []
        for r in records:
            p = float(r.get("pnl_pt", 0.0))
            if p > 0:
                wins += 1
            elif p < 0:
                losses += 1
            else:
                draws += 1
            pnl_pts.append(p)
            hold_mins.append(float(r.get("hold_minutes", 0.0)))
        self._wins     = wins
        self._losses   = losses
        self._draws    = draws
        self._pnl_pts  = pnl_pts
        self._hold_mins = hold_mins

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def wins(self) -> int:
        return self._wins

    @property
    def losses(self) -> int:
        return self._losses

    @property
    def draws(self) -> int:
        return self._draws

    @property
    def win_rate(self) -> float:
        """승률 (0~1). 거래 없으면 0.0."""
        denom = self._wins + self._losses
        return self._wins / denom if denom > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(self._pnl_pts)

    @property
    def avg_pnl(self) -> float:
        return sum(self._pnl_pts) / len(self._pnl_pts) if self._pnl_pts else 0.0

    @property
    def avg_win(self) -> float:
        wins = [p for p in self._pnl_pts if p > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [p for p in self._pnl_pts if p < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def max_win(self) -> float:
        return max(self._pnl_pts) if self._pnl_pts else 0.0

    @property
    def max_loss(self) -> float:
        return min(self._pnl_pts) if self._pnl_pts else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        """Profit Factor = 총수익 / |총손실|.  손실 없으면 None."""
        gross_profit = sum(p for p in self._pnl_pts if p > 0)
        gross_loss   = abs(sum(p for p in self._pnl_pts if p < 0))
        if gross_loss == 0:
            return None
        return gross_profit / gross_loss

    @property
    def avg_hold(self) -> float:
        return sum(self._hold_mins) / len(self._hold_mins) if self._hold_mins else 0.0

    def to_dict(self) -> dict:
        return {
            "count":         self.count,
            "wins":          self.wins,
            "losses":        self.losses,
            "draws":         self.draws,
            "win_rate":      round(self.win_rate, 4),
            "total_pnl_pt":  round(self.total_pnl, 2),
            "avg_pnl_pt":    round(self.avg_pnl, 2),
            "avg_win_pt":    round(self.avg_win, 2),
            "avg_loss_pt":   round(self.avg_loss, 2),
            "max_win_pt":    round(self.max_win, 2),
            "max_loss_pt":   round(self.max_loss, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor is not None else None,
            "avg_hold_min":  round(self.avg_hold, 1),
        }


# ════════════════════════════════════════════════════════════════════════════
# 그룹별 분석 유틸
# ════════════════════════════════════════════════════════════════════════════

def _grp_update(
    groups: Dict[str, List[dict]],
    key: str,
    record: dict,
) -> None:
    """groups[key] 리스트에 record 를 추가한다."""
    groups[key].append(record)


def _group_by_slot(records: List[dict]) -> Dict[str, _Stats]:
    grps: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        _grp_update(grps, r.get("slot", "?"), r)
    return {k: _Stats(v) for k, v in sorted(grps.items())}


def _group_by_reason(records: List[dict]) -> Dict[str, _Stats]:
    grps: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        _grp_update(grps, r.get("close_reason") or "미청산", r)
    return {k: _Stats(v) for k, v in sorted(grps.items())}


def _group_by_side(records: List[dict]) -> Dict[str, _Stats]:
    grps: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        _grp_update(grps, r.get("side", "?"), r)
    return {k: _Stats(v) for k, v in sorted(grps.items())}


def _group_by_confidence(records: List[dict]) -> Dict[str, _Stats]:
    grps: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        _grp_update(grps, r.get("entry_confidence", "?"), r)
    # HIGH > MEDIUM > LOW 순서로 정렬
    _rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return {
        k: _Stats(v)
        for k, v in sorted(grps.items(), key=lambda x: _rank.get(x[0], 9))
    }


def _iv_bucket(iv: float) -> str:
    """ATM IV 값을 구간 레이블로 변환한다."""
    if iv <= 0.0:
        return "데이터없음"
    pct = iv * 100.0
    if pct < 10.0:
        return "IV<10%"
    if pct < 15.0:
        return "10%≤IV<15%"
    if pct < 20.0:
        return "15%≤IV<20%"
    if pct < 25.0:
        return "20%≤IV<25%"
    return "IV≥25%"


def _group_by_iv(records: List[dict]) -> Dict[str, _Stats]:
    grps: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        bkt = _iv_bucket(float(r.get("entry_atm_iv", 0.0)))
        _grp_update(grps, bkt, r)
    _order = ["데이터없음", "IV<10%", "10%≤IV<15%", "15%≤IV<20%", "20%≤IV<25%", "IV≥25%"]
    ordered = {k: _Stats(v) for k, v in grps.items()}
    return {k: ordered[k] for k in _order if k in ordered}


# ════════════════════════════════════════════════════════════════════════════
# 출력 포맷터
# ════════════════════════════════════════════════════════════════════════════

_SEP  = "━" * 50
_SEP2 = "─" * 50


def _fmt_stats_line(label: str, st: _Stats) -> str:
    pf = st.profit_factor
    pf_str = f"PF={pf:.2f}" if pf is not None else "PF=∞"
    return (
        f"  {label:<14}  {st.count:>3}건  "
        f"승률 {st.win_rate:>6.1%}  "
        f"합계 {st.total_pnl:>+7.2f}pt  "
        f"평균 {st.avg_pnl:>+6.2f}pt  {pf_str}"
    )


def _print_stats_header() -> None:
    print(f"  {'구분':<14}  {'건수':>4}  {'승률':>7}  {'합계':>9}  {'평균':>8}  PF")
    print(f"  {_SEP2}")


def _print_record_verbose(r: dict, idx: int) -> None:
    side_icon = "▲" if r.get("side") == "LONG" else "▼"
    entry_t   = r.get("entry_time", "")[:16].replace("T", " ")
    close_t   = (r.get("close_time") or "")[:16].replace("T", "")
    pnl       = float(r.get("pnl_pt", 0.0))
    pnl_str   = f"{pnl:+.2f}pt"
    rsn       = r.get("close_reason") or "미청산"
    slot      = r.get("slot", "?")
    conf      = r.get("entry_confidence", "?")[0]     # H/M/L
    prob      = float(r.get("entry_prob", 0.0))
    iv        = float(r.get("entry_atm_iv", 0.0))
    iv_str    = f"IV={iv:.1%}" if iv > 0 else ""
    tid       = r.get("trade_id", "")[-10:]           # 끝 10자리만 표시

    pnl_icon = "✓" if pnl > 0 else ("✗" if pnl < 0 else "·")
    print(
        f"  {idx:>2}. {pnl_icon} {side_icon} "
        f"{entry_t}  슬롯{slot}  {conf} prob={prob:.2f}  "
        f"진입={r.get('entry_price',0):.2f}→청산={r.get('close_price',0):.2f}  "
        f"{pnl_str}  {rsn}  hold={r.get('hold_minutes',0):.0f}분  "
        + (f"{iv_str}  " if iv_str else "")
        + f"[{tid}]"
    )


def _print_daily(date_str: str, records: List[dict], *, verbose: bool = False) -> None:
    st = _Stats(records)
    pnl_arrow = "📈" if st.total_pnl > 0 else ("📉" if st.total_pnl < 0 else "➖")
    wr_str = f"{st.win_rate:.0%}" if st.count > 0 else "—"
    print(f"{_SEP}")
    print(
        f"  {date_str}  {pnl_arrow}  "
        f"{st.count}건  승률 {wr_str}  "
        f"손익 {st.total_pnl:+.2f}pt  "
        f"(최대 {st.max_win:+.2f} / 최저 {st.max_loss:+.2f})"
    )
    if verbose:
        for i, r in enumerate(records, 1):
            _print_record_verbose(r, i)


def _print_group_stats(
    title: str,
    groups: Dict[str, _Stats],
) -> None:
    print(f"\n  ── {title} ──")
    _print_stats_header()
    for label, st in groups.items():
        print(_fmt_stats_line(label, st))


def _print_overall(all_records: List[dict]) -> None:
    st = _Stats(all_records)
    pf = st.profit_factor
    pf_str = f"{pf:.2f}" if pf is not None else "∞"
    print(f"\n{_SEP}")
    print("  📊 전체 통계 요약")
    print(f"  {_SEP2}")
    print(f"  총 거래:        {st.count}건")
    print(f"  승 / 패 / 무:   {st.wins} / {st.losses} / {st.draws}")
    print(f"  승률:           {st.win_rate:.1%}")
    print(f"  총 손익:        {st.total_pnl:+.2f}pt")
    print(f"  평균 손익:      {st.avg_pnl:+.2f}pt")
    print(f"  평균 수익(승):  {st.avg_win:+.2f}pt")
    print(f"  평균 손실(패):  {st.avg_loss:+.2f}pt")
    print(f"  최대 수익:      {st.max_win:+.2f}pt")
    print(f"  최대 손실:      {st.max_loss:+.2f}pt")
    print(f"  Profit Factor:  {pf_str}")
    print(f"  평균 보유 시간: {st.avg_hold:.1f}분")


# ════════════════════════════════════════════════════════════════════════════
# JSON 출력
# ════════════════════════════════════════════════════════════════════════════

def _build_json_output(
    data: Dict[str, List[dict]],
    *,
    show_slot:   bool = False,
    show_reason: bool = False,
    show_side:   bool = False,
    show_conf:   bool = False,
    show_iv:     bool = False,
) -> dict:
    all_records = [r for recs in data.values() for r in recs]
    overall = _Stats(all_records).to_dict()

    daily = {}
    for ds, recs in sorted(data.items()):
        entry: dict = _Stats(recs).to_dict()
        if show_slot:
            entry["by_slot"]   = {k: v.to_dict() for k, v in _group_by_slot(recs).items()}
        if show_reason:
            entry["by_reason"] = {k: v.to_dict() for k, v in _group_by_reason(recs).items()}
        if show_side:
            entry["by_side"]   = {k: v.to_dict() for k, v in _group_by_side(recs).items()}
        if show_conf:
            entry["by_conf"]   = {k: v.to_dict() for k, v in _group_by_confidence(recs).items()}
        if show_iv:
            entry["by_iv"]     = {k: v.to_dict() for k, v in _group_by_iv(recs).items()}
        entry["trades"] = recs
        daily[ds] = entry

    out: dict = {"overall": overall, "daily": daily}

    if all_records:
        if show_slot:
            out["overall_by_slot"]   = {k: v.to_dict() for k, v in _group_by_slot(all_records).items()}
        if show_reason:
            out["overall_by_reason"] = {k: v.to_dict() for k, v in _group_by_reason(all_records).items()}
        if show_side:
            out["overall_by_side"]   = {k: v.to_dict() for k, v in _group_by_side(all_records).items()}
        if show_conf:
            out["overall_by_conf"]   = {k: v.to_dict() for k, v in _group_by_confidence(all_records).items()}
        if show_iv:
            out["overall_by_iv"]     = {k: v.to_dict() for k, v in _group_by_iv(all_records).items()}

    return out


# ════════════════════════════════════════════════════════════════════════════
# 메인 진입점
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trade_history_viewer",
        description="trade_history/ 디렉토리의 JSONL 거래 이력 분석 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("옵션")[0].strip(),  # 모듈 docstring에서 사용법 예시 추출
    )
    # 기간 선택 (상호 배타)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--date", metavar="YYYY-MM-DD", help="특정 날짜 조회")
    grp.add_argument("--days", type=int, default=7,   help="최근 N일 조회 (기본: 7)")
    grp.add_argument("--all",  action="store_true",   help="전체 기간 조회")

    # 디렉토리
    p.add_argument("--dir", default="trade_history", metavar="DIR", help="JSONL 디렉토리")

    # 분석 옵션
    p.add_argument("--slot",    action="store_true", help="슬롯(A/B/C)별 통계")
    p.add_argument("--reason",  action="store_true", help="청산 사유별 통계")
    p.add_argument("--side",    action="store_true", help="방향(Long/Short)별 통계")
    p.add_argument("--conf",    action="store_true", help="confidence 등급별 통계")
    p.add_argument("--iv",      action="store_true", help="ATM IV 구간별 통계")

    # 출력 제어
    p.add_argument("--json",    action="store_true", help="JSON 형식으로 출력")
    p.add_argument("--verbose", "-v", action="store_true", help="각 거래 상세 출력")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    history_dir = Path(args.dir)
    if not history_dir.exists():
        print(f"[ERROR] 디렉토리 없음: {history_dir}", file=sys.stderr)
        return 1

    # ── 데이터 로드 ──
    today = date.today()

    if args.all:
        data = _load_all(history_dir)
    elif args.date:
        try:
            date.fromisoformat(args.date)
        except ValueError:
            print(f"[ERROR] 날짜 형식 오류: {args.date} (YYYY-MM-DD 필요)", file=sys.stderr)
            return 1
        records = _load_date(history_dir, args.date)
        data = {args.date: records} if records else {}
    else:
        start = today - timedelta(days=args.days - 1)
        data  = _load_range(history_dir, start, today)

    if not data:
        print("조회 기간에 거래 이력이 없습니다.")
        return 0

    all_records = [r for recs in data.values() for r in recs]

    # ── JSON 출력 ──
    if args.json:
        out = _build_json_output(
            data,
            show_slot=args.slot,
            show_reason=args.reason,
            show_side=args.side,
            show_conf=args.conf,
            show_iv=args.iv,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ── 텍스트 출력 ──
    print(f"\n{'━'*50}")
    print(f"  📁 {history_dir}  ({len(data)}일, {len(all_records)}건)")
    print(f"{'━'*50}")

    for ds, recs in sorted(data.items()):
        _print_daily(ds, recs, verbose=args.verbose)

    _print_overall(all_records)

    # ── 그룹별 분석 ──
    if args.slot:
        _print_group_stats("슬롯(A/B/C)별", _group_by_slot(all_records))

    if args.reason:
        _print_group_stats("청산 사유별", _group_by_reason(all_records))

    if args.side:
        _print_group_stats("방향(Long/Short)별", _group_by_side(all_records))

    if args.conf:
        _print_group_stats("Confidence 등급별", _group_by_confidence(all_records))

    if args.iv:
        _print_group_stats("ATM IV 구간별", _group_by_iv(all_records))

    print(f"\n{_SEP}\n")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# 편의 API (다른 모듈에서 import 용)
# ════════════════════════════════════════════════════════════════════════════

def load_history(
    history_dir: str = "trade_history",
    *,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """거래 이력을 로드해 날짜별 dict 로 반환한다.

    Args:
        history_dir:  JSONL 파일 디렉토리 (기본: ``"trade_history"``).
        days:         최근 N일만 로드. ``None`` 이면 전체 로드.
        start_date:   ``"YYYY-MM-DD"`` 형식. ``days`` 와 함께 사용 불가.
        end_date:     ``"YYYY-MM-DD"`` 형식. 기본: 오늘.

    Returns:
        ``{"YYYY-MM-DD": [record_dict, ...], ...}``

    Examples::

        # 최근 30일
        data = load_history(days=30)

        # 특정 기간
        data = load_history(start_date="2026-03-01", end_date="2026-03-22")

        # 전체
        data = load_history()
    """
    hp = Path(history_dir)
    today = date.today()

    if days is not None:
        start = today - timedelta(days=days - 1)
        return _load_range(hp, start, today)

    if start_date is not None:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else today
        return _load_range(hp, s, e)

    return _load_all(hp)


def summary_stats(
    history_dir: str = "trade_history",
    *,
    days: Optional[int] = None,
) -> dict:
    """거래 이력의 전체 통계를 dict 로 반환한다.

    Args:
        history_dir: JSONL 파일 디렉토리.
        days:        최근 N일만 집계. ``None`` 이면 전체.

    Returns:
        :meth:`_Stats.to_dict` 형식의 dict.

    Example::

        stats = summary_stats(days=30)
        print(f"승률: {stats['win_rate']:.1%}")
    """
    data = load_history(history_dir, days=days)
    all_records = [r for recs in data.values() for r in recs]
    return _Stats(all_records).to_dict()


def daily_pnl_series(
    history_dir: str = "trade_history",
    *,
    days: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """날짜별 일일 손익 시계열을 반환한다.

    Args:
        history_dir: JSONL 파일 디렉토리.
        days:        최근 N일만. ``None`` 이면 전체.

    Returns:
        ``[("YYYY-MM-DD", pnl_pt), ...]`` — 날짜 오름차순 정렬.

    Example::

        series = daily_pnl_series(days=10)
        for date_str, pnl in series:
            print(f"{date_str}: {pnl:+.2f}pt")
    """
    data = load_history(history_dir, days=days)
    return [
        (ds, round(sum(float(r.get("pnl_pt", 0.0)) for r in recs), 2))
        for ds, recs in sorted(data.items())
    ]


def cumulative_pnl(
    history_dir: str = "trade_history",
    *,
    days: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """날짜별 누적 손익 시계열을 반환한다.

    Returns:
        ``[("YYYY-MM-DD", cumulative_pnl_pt), ...]`` — 날짜 오름차순.
    """
    series = daily_pnl_series(history_dir, days=days)
    cum    = 0.0
    result = []
    for ds, pnl in series:
        cum += pnl
        result.append((ds, round(cum, 2)))
    return result


def best_worst_days(
    history_dir: str = "trade_history",
    *,
    n: int = 5,
    days: Optional[int] = None,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """최고/최저 일일 손익 날짜 상위 N개를 반환한다.

    Returns:
        ``(best_n, worst_n)`` — 각각 ``[("YYYY-MM-DD", pnl), ...]``.
    """
    series = daily_pnl_series(history_dir, days=days)
    sorted_asc  = sorted(series, key=lambda x: x[1])
    sorted_desc = sorted(series, key=lambda x: x[1], reverse=True)
    return sorted_desc[:n], sorted_asc[:n]


# ════════════════════════════════════════════════════════════════════════════
# 공개 심볼 목록
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    # 로더
    "load_history",
    # 통계
    "summary_stats",
    "daily_pnl_series",
    "cumulative_pnl",
    "best_worst_days",
    # 내부 클래스 (테스트 등에서 직접 사용 가능)
    "_Stats",
    "_grp_update",
    "_group_by_slot",
    "_group_by_reason",
    "_group_by_side",
    "_group_by_confidence",
    "_group_by_iv",
    "_iv_bucket",
]


if __name__ == "__main__":
    sys.exit(main())
