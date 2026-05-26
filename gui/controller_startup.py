"""GUI 시작 시 인터넷 시간 동기화 (`gui_controller` 9단계 분리)."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

__all__ = ["run_startup_internet_time_sync"]


def run_startup_internet_time_sync(
    append_log: Callable[[str], None],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """``INTERNET_TIME_SYNC`` 환경변수에 따라 NTP 측정/동기화 후 GUI·파일 로그에 요약 기록."""
    lg = log or logging.getLogger(__name__)
    try:
        from utils.internet_time_sync import sync_best_effort

        env_v = str(os.environ.get("INTERNET_TIME_SYNC") or "").strip().lower()
        do_sync = env_v not in ("0", "false", "no", "n", "off")
        try:
            append_log(f"[TIME_SYNC] startup (enabled={str(bool(do_sync)).lower()} env={env_v!r})")
        except Exception:
            pass
        r = sync_best_effort(
            sync=bool(do_sync),
            samples=2,
            timeout=1.0,
            min_abs_offset_ms_to_sync=500.0,
        )
        parts: list[str] = []
        try:
            best = r.get("best") if isinstance(r, dict) else None
            best = best if isinstance(best, dict) else {}

            ok = bool(r.get("ok")) if isinstance(r, dict) else False
            reason = str(r.get("reason")) if isinstance(r, dict) else "unknown"
            sync_attempted = bool(r.get("sync_attempted")) if isinstance(r, dict) else False

            server = str(best.get("server") or "")
            samples = best.get("samples")
            offset_ms_avg = best.get("offset_ms_avg")
            offset_ms_p95 = best.get("offset_ms_p95")

            if ok and sync_attempted:
                parts.append("인터넷 시간 동기화 성공")
            elif ok and not sync_attempted:
                if reason == "measured_only":
                    parts.append("인터넷 시간 오프셋 측정 완료(동기화 미수행)")
                elif reason == "skip_small_offset":
                    parts.append("인터넷 시간 오프셋이 작아 동기화를 건너뜀")
                else:
                    parts.append("인터넷 시간 처리 완료(동기화 미수행)")
            else:
                if sync_attempted:
                    parts.append("인터넷 시간 동기화 실패")
                else:
                    parts.append("인터넷 시간 오프셋 측정 실패")

            if server:
                parts.append(f"서버={server}")
            if samples is not None:
                parts.append(f"샘플={samples}")
            if offset_ms_avg is not None:
                try:
                    parts.append(f"평균오프셋={float(offset_ms_avg):.2f}ms(+면 시스템이 느림)")
                except Exception:
                    parts.append(f"평균오프셋={offset_ms_avg}ms")
            if offset_ms_p95 is not None:
                try:
                    parts.append(f"P95오프셋={float(offset_ms_p95):.2f}ms")
                except Exception:
                    parts.append(f"P95오프셋={offset_ms_p95}ms")
            if (not ok) or (
                reason and reason not in ("measured_only", "skip_small_offset", "SetSystemTime_ok", "w32tm_resync_ok")
            ):
                parts.append(f"사유={reason}")

            append_log(f"[TIME_SYNC] {' / '.join(parts)}")
        except Exception:
            try:
                append_log(f"[TIME_SYNC] {str(r)}")
            except Exception:
                pass
        try:
            if parts:
                lg.info("[TIME_SYNC] %s", " / ".join(parts))
        except Exception:
            pass
    except Exception as e:
        try:
            append_log(f"[TIME_SYNC] failed: {e}")
        except Exception:
            pass
        try:
            lg.warning("[TIME_SYNC] failed: %s", e)
        except Exception:
            pass
