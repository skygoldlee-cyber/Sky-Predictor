"""실행 모드 함수 모음.

main.py에서 분리된 각 실행 모드 진입점:
- run_test_mode()
- run_replay_mode()
- run_replay_mode_with_predictor()
- run_live_mode()
- run_simple_prediction()
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import threading
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from config import AppConfig
from core.logging_utils import get_logger
from .pipeline_builder import _build_pipeline
from ebestapi.live import run_ebest_live_mode

logger = get_logger()


def run_test_mode() -> int:
    """
    테스트 모드 실행
    
    Returns:
        Exit code
    """
    logger.info("Running tests...")

    try:
        import pytest

        rc = int(pytest.main([]))
        if rc == 0:
            logger.info("✓ All tests passed")
            return 0
        logger.error("✗ Tests failed")
        return 1
    except Exception as e:
        logger.error("Failed to run tests: %s", e, exc_info=True)
        return 1


def run_replay_mode(replay_file: str) -> int:
    """
    리플레이 모드 실행
    
    Args:
        replay_file: 리플레이 파일 경로
        
    Returns:
        Exit code
    """
    logger.info("Replay mode: %s", replay_file)

    logger.error("Replay mode requires predictor instance; use main() CLI entry")
    return 1


def run_replay_mode_with_predictor(
    replay_file: str,
    predictor: Any,
    *,
    speed: float = 0.0,
    max_lines: Optional[int] = None,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> int:
    p = Path(str(replay_file))
    if not p.exists() or (not p.is_file()):
        logger.error("Replay file not found: %s", replay_file)
        return 1

    def _open_text(path: Path):
        if str(path).lower().endswith(".gz"):
            return io.TextIOWrapper(gzip.open(str(path), "rb"), encoding="utf-8", errors="replace")
        return open(str(path), "rt", encoding="utf-8")

    def _restore_tick_for_predictor(tick: Any) -> Any:
        if not isinstance(tick, dict):
            return tick
        out: dict[str, Any] = {}
        for k, v in tick.items():
            ks = str(k)
            if (ks.startswith("offerho") or ks.startswith("bidho")) and isinstance(v, int):
                out[ks] = float(v) / 100.0
                continue
            out[ks] = v
        return out

    logger.info("[REPLAY] file=%s speed=%.3f max_lines=%s", str(p), float(speed), str(max_lines))

    processed = 0
    last_ts_ms: Optional[int] = None
    t0 = time.time()
    text = ""
    try:
        if str(p).lower().endswith(".gz"):
            raw = b""
            try:
                with open(str(p), "rb") as fb:
                    raw = fb.read()
            except Exception:
                raw = b""
            if not raw:
                try:
                    with gzip.open(str(p), "rt", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                    raw = None  # type: ignore
                except Exception:
                    raw = b""
            try:
                if isinstance(raw, (bytes, bytearray)) and raw:
                    text = gzip.decompress(raw).decode("utf-8", errors="replace")
            except Exception:
                text = ""
        else:
            with open(str(p), "rt", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
    except Exception:
        text = ""

    norm_text = str(text or "")
    try:
        norm_text = norm_text.replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        norm_text = str(text or "")

    for line in norm_text.split("\n"):
        if stop_event is not None and stop_event.is_set():
            break
        while pause_event is not None and pause_event.is_set():
            time.sleep(0.1)

        if max_lines is not None and processed >= int(max_lines):
            break

        s = str(line or "").strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue

        trcode = str(rec.get("trcode") or "").strip()
        symbol = str(rec.get("symbol") or "").strip()
        tick = rec.get("tick")

        processed += 1

        ts_ms = None
        try:
            if "ts_ms" in rec:
                ts_ms = int(rec.get("ts_ms") or 0)
        except Exception:
            ts_ms = None

        tick_for_pred = _restore_tick_for_predictor(tick)
        try:
            from prediction.pipeline import PredictionPipeline

            PredictionPipeline.add_realtime_tick(
                predictor,
                {"trcode": trcode, "symbol": symbol, "tick": tick_for_pred},
            )
        except Exception as e2:
            logger.warning("[REPLAY] predictor.add_realtime_tick failed: %s", e2)

        if speed and speed > 0 and ts_ms is not None and last_ts_ms is not None:
            try:
                dt = max(0.0, (ts_ms - last_ts_ms) / 1000.0)
                sleep_sec = dt / float(speed)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)
        last_ts_ms = ts_ms if ts_ms is not None else last_ts_ms

    elapsed = max(1e-6, time.time() - t0)
    logger.info("[REPLAY] processed=%d elapsed=%.3fs (%.1f lines/s)", processed, elapsed, processed / elapsed)
    try:
        metrics = getattr(predictor, "get_metrics", None)
        if callable(metrics):
            logger.info("[REPLAY] predictor metrics: %s", metrics())
    except Exception as _e:
        logger.debug("오류 무시: %s", _e)
    return 0


async def run_live_mode(
    config: AppConfig,
    args: argparse.Namespace,
    predictor,  # KP200HybridPredictor
) -> dict:
    """
    실시간 모드 실행
    
    Args:
        config: 설정
        args: 인자
        predictor: 예측기
        
    Returns:
        실행 결과 딕셔너리
    """
    # 옵션 포함 여부
    include_options = bool(args.include_options)
    
    # 틱 저장 설정
    save_ticks = not args.no_save_ticks
    out_ticks = args.out_ticks
    
    if save_ticks and not out_ticks:
        from core.utils import get_default_ticks_output_path
        out_ticks = get_default_ticks_output_path()

    compress_ticks_enabled = bool(getattr(args, "compress_ticks", False))
    if save_ticks and compress_ticks_enabled and out_ticks:
        try:
            p = Path(str(out_ticks))
            if p.suffix.lower() == ".jsonl":
                out_ticks = str(p.with_suffix(p.suffix + ".gz"))
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)

    def _compress_ticks_jsonl(path: str) -> Optional[str]:
        try:
            p = Path(str(path))
            if not p.exists() or not p.is_file():
                return None
            if p.suffix.lower() != ".jsonl":
                return None
            zip_path = p.with_suffix(p.suffix + ".zip")
            with zipfile.ZipFile(str(zip_path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(str(p), arcname=p.name)
            try:
                p.unlink()
            except Exception as _e:
                logger.debug("[_compress_ticks_jsonl] 오류 무시: %s", _e)
            return str(zip_path)
        except Exception:
            return None
    
    logger.info("Live mode starting...")
    logger.info("  Duration: %ss", args.duration_sec)
    logger.info("  Include options: %s", include_options)
    logger.info("  Save ticks: %s", save_ticks)
    if save_ticks:
        logger.info("  Output file: %s", out_ticks)
    
    result = await run_ebest_live_mode(
        predictor=predictor,
        duration_sec=args.duration_sec,
        include_options=include_options,
        option_month_info=args.option_month,
        config_path=args.config,
        opt_itm=config.options_subscription.itm,
        opt_wait_sec=config.options_subscription.wait_sec,
        out_ticks=out_ticks,
        save_ticks_enabled=save_ticks,
    )
    
    if save_ticks and compress_ticks_enabled and out_ticks:
        # If we are already writing a compressed stream (.jsonl.gz), do not perform end-of-run zip.
        try:
            if str(out_ticks).lower().endswith(".jsonl.gz"):
                return result
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        zipped = _compress_ticks_jsonl(str(out_ticks))
        if zipped:
            logger.info("Tick file compressed: %s", zipped)

    return result


def run_simple_prediction(
    predictor,  # KP200HybridPredictor
    args: argparse.Namespace,
) -> dict:
    """
    단순 예측 실행
    
    Args:
        predictor: 예측기
        args: 인자
        
    Returns:
        예측 결과 딕셔너리
    """
    logger.info("Running prediction...")
    
    # 예측 실행
    result = predictor.get_prediction(
        auto_mode=True,
    )
    
    logger.info("Prediction completed")
    
    return result


