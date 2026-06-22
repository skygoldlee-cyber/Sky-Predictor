import logging
import socket
import struct
import argparse
import subprocess
import sys
import ctypes
import time
from datetime import datetime, timezone, timedelta

NTP_DELTA = 2208988800  # NTP epoch(1900) -> Unix epoch(1970) 초 보정

logger = logging.getLogger(__name__)

# sync_best_effort / CLI 기본 순회 순서(응답이 잘 나오는 서버를 앞에 두면 불필요한 타임아웃 대기 감소)
_DEFAULT_NTP_HOSTS = (
    "time.google.com",
    "time.windows.com",
    "pool.ntp.org",
)

def query_ntp(server="time.windows.com", timeout=2.0):
    """
    NTP 서버에서 UTC 시간을 받아 datetime(UTC)로 반환.
    외부 라이브러리 없이 UDP 123 직접 사용.
    """
    msg = b'\x1b' + 47 * b'\0'  # LI=0, VN=3, Mode=3 (client)
    addr = (server, 123)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        s.sendto(msg, addr)
        data, _ = s.recvfrom(48)

    if len(data) < 48:
        raise RuntimeError("NTP 응답 길이 오류")

    # Transmit Timestamp (bytes 40~47)
    transmit_timestamp = struct.unpack("!I", data[40:44])[0] + \
                        struct.unpack("!I", data[44:48])[0] / 2**32
    # NTP epoch -> Unix epoch
    unix_time = transmit_timestamp - NTP_DELTA
    return datetime.fromtimestamp(unix_time, tz=timezone.utc)

def measure_offset(server="time.windows.com", samples=5, timeout=2.0):
    """
    NTP 시간과 로컬 시간의 오프셋(지연 보정 포함)을 평균으로 계산.
    각 샘플에서 '응답 도착 직후 로컬UTC'와 '응답 중간시점' 보정 사용.
    """
    offsets = []

    for _ in range(samples):
        # 요청 전/후 로컬 UTC를 재서 중간 시점을 추정
        t_before = datetime.now(tz=timezone.utc)
        try:
            t_ntp = query_ntp(server=server, timeout=timeout)
        except Exception as e:
            # 앱 시작 시 sync_best_effort가 여러 호스트를 순회하므로, 여기서 WARN 출력하면
            # 다음 호스트로 성공해도 로그만 스팸이 됨 → DEBUG로 남김
            logger.debug(f"NTP sample failed server={server}: {e}")
            continue
        t_after = datetime.now(tz=timezone.utc)

        # 중간시점 보정(왕복지연의 절반을 보정)
        t_mid = t_before + (t_after - t_before) / 2
        offset = (t_ntp - t_mid).total_seconds()  # +면 시스템이 느림, -면 빠름
        offsets.append(offset)

    if not offsets:
        raise RuntimeError("오프셋 샘플을 얻지 못했습니다.")

    avg = sum(offsets) / len(offsets)
    p95 = sorted(offsets)[int(len(offsets)*0.95) - 1] if len(offsets) >= 2 else offsets[0]

    return {
        "server": server,
        "samples": len(offsets),
        "offset_seconds_avg": avg,
        "offset_ms_avg": avg * 1000.0,
        "offset_seconds_p95": p95,
        "offset_ms_p95": p95 * 1000.0,
        "offsets_each_ms": [o*1000.0 for o in offsets],
    }


def sync_best_effort(
    *,
    server: str = "",
    samples: int = 5,
    timeout: float = 2.0,
    sync: bool = True,
    min_abs_offset_ms_to_sync: float = 1000.0,
) -> dict:
    """Measure NTP offset and best-effort sync Windows system time.

    Returns a dict with keys:
      ok(bool), reason(str), best(dict|None), sync_attempted(bool)
    """

    hosts = [server] if str(server).strip() else list(_DEFAULT_NTP_HOSTS)

    best = None
    best_abs_ms = None
    last_err = None

    for host in hosts:
        try:
            r = measure_offset(server=host, samples=int(samples), timeout=float(timeout))
            abs_ms = abs(float(r.get("offset_ms_avg", 0.0)))
            if best is None or (best_abs_ms is not None and abs_ms < best_abs_ms):
                best = r
                best_abs_ms = abs_ms
        except Exception as e:
            last_err = str(e)

    if best is None:
        return {
            "ok": False,
            "reason": last_err or "no_ntp_result",
            "best": None,
            "sync_attempted": False,
        }

    if not bool(sync):
        return {
            "ok": True,
            "reason": "measured_only",
            "best": best,
            "sync_attempted": False,
        }

    try:
        abs_ms = float(best_abs_ms) if best_abs_ms is not None else abs(float(best.get("offset_ms_avg", 0.0)))
    except Exception:
        abs_ms = abs(float(best.get("offset_ms_avg", 0.0)))

    if abs_ms < float(min_abs_offset_ms_to_sync):
        return {
            "ok": True,
            "reason": "skip_small_offset",
            "best": best,
            "sync_attempted": False,
        }

    try:
        offset_s = float(best.get("offset_seconds_avg", 0.0))
        now_utc = datetime.now(tz=timezone.utc)
        target_utc = now_utc + timedelta(seconds=offset_s)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"compute_target_failed:{e}",
            "best": best,
            "sync_attempted": True,
        }

    ok, msg = _set_windows_system_time_utc(target_utc)
    if ok:
        return {
            "ok": True,
            "reason": "SetSystemTime_ok",
            "best": best,
            "sync_attempted": True,
        }

    ok2, msg2 = _try_w32tm_resync()
    if ok2:
        return {
            "ok": True,
            "reason": "w32tm_resync_ok",
            "best": best,
            "sync_attempted": True,
        }

    return {
        "ok": False,
        "reason": f"SetSystemTime_failed:{msg};w32tm_failed:{msg2}",
        "best": best,
        "sync_attempted": True,
    }


def _fmt_dt(dt: datetime) -> str:
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return str(dt)


def _kst(dt: datetime) -> datetime:
    try:
        return dt.astimezone(timezone(timedelta(hours=9)))
    except Exception:
        return dt


class _SYSTEMTIME(ctypes.Structure):
    _fields_ = [
        ("wYear", ctypes.c_ushort),
        ("wMonth", ctypes.c_ushort),
        ("wDayOfWeek", ctypes.c_ushort),
        ("wDay", ctypes.c_ushort),
        ("wHour", ctypes.c_ushort),
        ("wMinute", ctypes.c_ushort),
        ("wSecond", ctypes.c_ushort),
        ("wMilliseconds", ctypes.c_ushort),
    ]


def _set_windows_system_time_utc(dt_utc: datetime) -> tuple[bool, str | None]:
    try:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_utc = dt_utc.astimezone(timezone.utc)

        st = _SYSTEMTIME()
        st.wYear = dt_utc.year
        st.wMonth = dt_utc.month
        st.wDay = dt_utc.day
        st.wHour = dt_utc.hour
        st.wMinute = dt_utc.minute
        st.wSecond = dt_utc.second
        st.wMilliseconds = int(dt_utc.microsecond / 1000)

        ok = bool(ctypes.windll.kernel32.SetSystemTime(ctypes.byref(st)))
        if not ok:
            err = ctypes.GetLastError()
            return False, f"SetSystemTime 실패(GetLastError={err})"
        return True, None
    except Exception as e:
        return False, str(e)


def _try_w32tm_resync() -> tuple[bool, str | None]:
    try:
        proc = subprocess.run(
            ["w32tm", "/resync"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode == 0:
            return True, out or None
        return False, (err or out or f"w32tm /resync 실패(returncode={proc.returncode})")
    except FileNotFoundError:
        return False, "w32tm 명령을 찾지 못했습니다."
    except Exception as e:
        return False, str(e)


def _print_report(server: str, samples: int, timeout: float) -> tuple[dict | None, str | None]:
    try:
        r = measure_offset(server=server, samples=samples, timeout=timeout)
        offset_s = float(r["offset_seconds_avg"])
        now_utc = datetime.now(tz=timezone.utc)
        ntp_utc_est = now_utc + timedelta(seconds=offset_s)

        print(f"[{r['server']}] 샘플 {r['samples']}개")
        print(f"  평균 오프셋: {r['offset_ms_avg']:.2f} ms (+면 시스템이 느림, -면 빠름)")
        print(f"  P95 오프셋:  {r['offset_ms_p95']:.2f} ms")
        print(f"  개별(ms): {', '.join(f'{x:.1f}' for x in r['offsets_each_ms'])}")
        print(f"  시스템(UTC): {_fmt_dt(now_utc)}")
        print(f"  NTP추정(UTC): {_fmt_dt(ntp_utc_est)}")
        print(f"  시스템(KST): {_fmt_dt(_kst(now_utc))}")
        print(f"  NTP추정(KST): {_fmt_dt(_kst(ntp_utc_est))}")
        return r, None
    except Exception as e:
        return None, str(e)


def _print_sync_troubleshooting(*, setsystemtime_error: str | None, w32tm_error: str | None) -> None:
    try:
        print("\n" + "=" * 70)
        print("[SYNC][HINT] 동기화 실패 원인/해결")
        if setsystemtime_error and ("GetLastError=1314" in setsystemtime_error):
            print("  - SetSystemTime GetLastError=1314: 권한 부족(SeSystemtimePrivilege).")
            print("    해결: 터미널/IDE를 '관리자 권한으로 실행' 후 다시 시도")
        elif setsystemtime_error:
            print(f"  - SetSystemTime 실패: {setsystemtime_error}")

        if w32tm_error and ("0x80070426" in w32tm_error):
            print("  - w32tm 0x80070426: Windows Time(w32time) 서비스가 시작되지 않음")
            print("    해결(관리자 PowerShell):")
            print("      sc config w32time start= auto")
            print("      sc start w32time")
            print("      w32tm /resync")
        elif w32tm_error:
            print(f"  - w32tm 실패: {w32tm_error}")
        print("=" * 70)
    except Exception:
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="", help="단일 NTP 서버(미지정 시 기본 3개 순회)")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=2.0)
    sync_group = ap.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync",
        dest="sync",
        action="store_true",
        help="NTP 시간으로 시스템 시간을 동기화(Windows)",
    )
    sync_group.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        help="동기화 수행하지 않음(측정만)",
    )
    ap.set_defaults(sync=True)
    args = ap.parse_args()

    hosts = [args.server] if str(args.server).strip() else list(_DEFAULT_NTP_HOSTS)

    best = None
    best_abs_ms = None
    any_ok = False

    for host in hosts:
        r, err = _print_report(host, args.samples, args.timeout)
        if err is not None:
            print(f"[ERROR] {host}: {err}")
            continue
        any_ok = True
        try:
            abs_ms = abs(float(r["offset_ms_avg"]))
            if best is None or (best_abs_ms is not None and abs_ms < best_abs_ms):
                best = r
                best_abs_ms = abs_ms
        except Exception as e:
            print(f"[DEBUG] offset_ms_avg parsing error: {e}")

    if not any_ok:
        sys.exit(1)

    if not args.sync:
        try:
            if best_abs_ms is not None and float(best_abs_ms) >= 1000.0:
                print("\n" + "=" * 70)
                print("[HINT] 시스템 시간이 인터넷시간과 크게 다릅니다.")
                print("  동기화 결과([SYNC]/[AFTER SYNC])를 보려면 아래처럼 실행하세요:")
                print("  python internet_time_sync.py --sync")
                print("=" * 70)
        except Exception as e:
            print(f"[DEBUG] hint display error: {e}")

    if args.sync:
        if best is None:
            print("[ERROR] 동기화할 오프셋을 얻지 못했습니다.")
            sys.exit(1)
        offset_s = float(best["offset_seconds_avg"])
        now_utc = datetime.now(tz=timezone.utc)
        target_utc = now_utc + timedelta(seconds=offset_s)

        try:
            print("\n" + "=" * 70)
            print("[SYNC] 인터넷시간 동기화 시도")
            print(f"  기준 서버: {best.get('server')}")
            print(f"  적용 오프셋(ms): {float(best.get('offset_ms_avg')):.2f} (+면 시스템이 느림)")
            print(f"  목표 UTC: {_fmt_dt(target_utc)}")
            print("=" * 70)
        except Exception as e:
            print(f"[DEBUG] sync info display error: {e}")

        ok, msg = _set_windows_system_time_utc(target_utc)
        if ok:
            print(f"[SYNC] SetSystemTime 성공: {_fmt_dt(target_utc)} UTC")

            # 동기화 후 재측정
            time.sleep(0.8)
            try:
                print("\n" + "-" * 70)
                print("[AFTER SYNC] 재측정 결과")
                print("-" * 70)
            except Exception as e:
                print(f"[DEBUG] after-sync header display error: {e}")
            _, err3 = _print_report(str(best.get("server")), max(3, int(args.samples)), float(args.timeout))
            if err3 is not None:
                print(f"[AFTER SYNC][ERROR] {best.get('server')}: {err3}")
            sys.exit(0)

        print(f"[SYNC] SetSystemTime 실패: {msg}")
        ok2, msg2 = _try_w32tm_resync()
        if ok2:
            print("[SYNC] w32tm /resync 성공")
            if msg2:
                print(f"  {msg2}")

            # 동기화 후 재측정
            time.sleep(0.8)
            try:
                print("\n" + "-" * 70)
                print("[AFTER SYNC] 재측정 결과")
                print("-" * 70)
            except Exception as e:
                print(f"[DEBUG] after-sync header display error: {e}")
            _, err3 = _print_report(str(best.get("server")), max(3, int(args.samples)), float(args.timeout))
            if err3 is not None:
                print(f"[AFTER SYNC][ERROR] {best.get('server')}: {err3}")
            sys.exit(0)

        print(f"[SYNC] w32tm /resync 실패: {msg2}")
        _print_sync_troubleshooting(setsystemtime_error=msg, w32tm_error=msg2)
        sys.exit(2)
