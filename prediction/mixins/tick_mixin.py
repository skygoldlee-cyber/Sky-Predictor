"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

from config import TRCode
from core.utils import adaptive_uses_kospi_spot_index_minute_bars, parse_chetime
from ..features.features import calc_orderbook_features


class TickMixin:
    """Mixin: TickMixin methods extracted from PredictionPipeline."""

    def add_realtime_tick(self, tick_data: Dict[str, Any]) -> None:
        """Ingest a realtime tick.

        This method is called by `ebest_live.py` for FC0/OC0/FO0 ticks.
        - FC0/OC0 are forwarded to `tick_processor.process_tick()`.
        - FO0 is also buffered into `self._ob_records` after parsing and 1Hz downsampling.
        """
        self._metrics_inc("ticks_processed")
        try:
            self.tick_processor.process_tick(tick_data)
        except Exception:
            # Tick processor should not block orderbook buffering (especially in replay).
            pass

        trcode = str((tick_data or {}).get("trcode") or "").strip().upper()
        if trcode == str(TRCode.FUTURES.value).strip().upper():
            try:
                self._last_fc0_seen_epoch = float(time.time())
            except Exception:
                self._last_fc0_seen_epoch = self._last_fc0_seen_epoch
        if trcode == str(TRCode.FUTURES_BOOK.value).strip().upper():
            tick_raw = (tick_data or {}).get("tick") or {}
            tick_norm = (tick_data or {}).get("tick_norm") or {}

            # calc_orderbook_features expects FH0 raw-key schema:
            # offerho1~5, bidho1~5, offerrem1~5, bidrem1~5, totofferrem, totbidrem, ...
            # tick_norm uses list-based schema (offerhos/bidhos/offerrems/bidrems).
            # When tick_norm is present, unpack its arrays into the raw dict (non-destructive).
            tick = tick_raw if isinstance(tick_raw, dict) else {}
            if isinstance(tick_norm, dict) and tick_norm:
                try:
                    mapping = (
                        ("offerhos", "offerho"),
                        ("bidhos", "bidho"),
                        ("offerrems", "offerrem"),
                        ("bidrems", "bidrem"),
                        ("offercnts", "offercnt"),
                        ("bidcnts", "bidcnt"),
                    )
                    for src_key, dst_prefix in mapping:
                        arr = tick_norm.get(src_key)
                        if not isinstance(arr, list):
                            continue
                        for i, v in enumerate(arr, 1):
                            if i > 5:
                                break
                            tick.setdefault(f"{dst_prefix}{i}", v)

                    for k in ("totofferrem", "totbidrem", "totoffercnt", "totbidcnt", "hotime", "danhochk", "alloc_gubun", "futcode"):
                        if k in tick_norm and tick_norm.get(k) is not None:
                            tick.setdefault(k, tick_norm.get(k))
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

            if isinstance(tick, dict):
                # FH0 may carry multiple schemas depending on environment.
                # Buffer only when the payload looks like an orderbook/quote snapshot.
                try:
                    looks_like_orderbook = any(
                        k in tick
                        for k in (
                            "offerho",
                            "bidho",
                            "offerrem",
                            "bidrem",
                            "offerho1",
                            "bidho1",
                            "offerrem1",
                            "bidrem1",
                            "offerho5",
                            "bidho5",
                            "offerrem5",
                            "bidrem5",
                            "totofferrem",
                            "totbidrem",
                            "offerhos",
                            "bidhos",
                            "offerrems",
                            "bidrems",
                        )
                    )
                except Exception:
                    looks_like_orderbook = False

                if not looks_like_orderbook:
                    return

                try:
                    restored: Dict[str, Any] = {}
                    for k, v in tick.items():
                        ks = str(k)
                        if (ks.startswith("offerho") or ks.startswith("bidho")) and isinstance(v, int):
                            try:
                                iv = int(v)
                                if 1000 <= abs(iv) <= 100000000:
                                    restored[ks] = float(iv) / 100.0
                                    continue
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                        restored[ks] = v
                    tick = restored
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

                # Some replay logs include only L1 prices; fill missing price levels so
                # downstream feature extractors can operate.
                try:
                    o1 = tick.get("offerho1")
                    b1 = tick.get("bidho1")
                    if o1 is not None:
                        for i in range(2, 6):
                            tick.setdefault(f"offerho{i}", o1)
                    if b1 is not None:
                        for i in range(2, 6):
                            tick.setdefault(f"bidho{i}", b1)
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

                try:
                    # 1Hz downsample key
                    sec_key = None
                    try:
                        che = tick.get("hotime") or tick.get("chetime")
                        if che is not None and str(che).strip() != "":
                            dt = parse_chetime(che)
                            sec_key = int(dt.replace(microsecond=0).timestamp())
                    except Exception:
                        sec_key = None
                    if sec_key is None:
                        sec_key = int(time.time())

                    self._last_fo0_seen_epoch = float(time.time())
                    self._fo0_stale_warned = False

                    ob = calc_orderbook_features(tick)

                    ob.setdefault("_ts_epoch", int(sec_key))

                    try:
                        # OBI delta/EMA features (computed from buffered history).
                        cur_obi = float(ob.get("obi") or 0.0)
                        prev_obi = None
                        prev_ema = None
                        obi_5s_ago = None
                        try:
                            with self._ob_lock:
                                hist = list(self._ob_records)
                        except Exception:
                            hist = []

                        if hist:
                            try:
                                prev_obi = float(hist[-1].get("obi") or 0.0)
                            except Exception:
                                prev_obi = None
                            try:
                                prev_ema = float(hist[-1].get("obi_ema5") or hist[-1].get("obi") or 0.0)
                            except Exception:
                                prev_ema = None

                            tgt_ts = int(sec_key) - 5
                            for r in reversed(hist):
                                try:
                                    ts = int((r or {}).get("_ts_epoch") or 0)
                                except Exception:
                                    ts = 0
                                if ts <= int(tgt_ts) and ts > 0:
                                    try:
                                        obi_5s_ago = float((r or {}).get("obi") or 0.0)
                                    except Exception:
                                        obi_5s_ago = None
                                    break

                        if prev_obi is None:
                            prev_obi = float(cur_obi)
                        if obi_5s_ago is None:
                            obi_5s_ago = float(prev_obi)
                        if prev_ema is None:
                            prev_ema = float(prev_obi)

                        alpha = 2.0 / (5.0 + 1.0)
                        ob["obi_delta1"] = float(cur_obi) - float(prev_obi)
                        ob["obi_delta5"] = float(cur_obi) - float(obi_5s_ago)
                        ob["obi_ema5"] = float(alpha) * float(cur_obi) + (1.0 - float(alpha)) * float(prev_ema)
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)

                    try:
                        # Attach per-second option snapshot so build_sequence can build
                        # a true time series without tiling.
                        #
                        # v3: _build_option_snapshot_safe()를 재사용한다.
                        # 이 메서드는 _prev_* 상태를 읽어 calc_parity_divergence에 전달하고,
                        # 계산 후 _prev_* 상태를 갱신한다. 따라서 OB 버퍼 경로에서도
                        # call_vs_fut_ret_diff가 직전 초 대비로 정확하게 계산된다.
                        #
                        # 같은 sec_key 내 중복 FO0 틱은 캐시를 재사용하여 불필요한
                        # calc_parity_divergence 재계산을 방지한다.
                        cached = None
                        try:
                            if self._last_opt_sec_key is not None and int(self._last_opt_sec_key) == int(sec_key):
                                cached = self._last_opt_features
                        except Exception:
                            cached = None

                        if isinstance(cached, dict) and cached:
                            ob["_opt_features"] = dict(cached)
                        else:
                            cur_px = 0.0
                            try:
                                cur_px = float(self.tick_processor.get_current_price() or 0.0)
                            except Exception:
                                cur_px = 0.0

                            if cur_px > 0.0:
                                # _build_option_snapshot_safe: prev 상태 주입 + 갱신을 모두 처리
                                opt_snap = self._build_option_snapshot_safe(current_price=float(cur_px))
                                if isinstance(opt_snap, dict) and opt_snap:
                                    feat = {k: opt_snap.get(k) for k in list(self._opt_keys)}
                                    ob["_opt_features"] = dict(feat)
                                    self._last_opt_sec_key = int(sec_key)
                                    self._last_opt_features = dict(feat)
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)

                    try:
                        if self._adaptive_mgr is not None and isinstance(self._adaptive_last_features, dict) and self._adaptive_last_features:
                            ob["_adaptive_features"] = dict(self._adaptive_last_features)
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)

                    if bool(ob.get("_invalid")):
                        try:
                            # Best-effort: some replay/feeds omit enough keys for calc_orderbook_features.
                            # Rebuild a minimal quote and retry once.
                            mini: Dict[str, Any] = {}
                            for k in (
                                "offerho",
                                "bidho",
                                "offerho1",
                                "bidho1",
                                "totofferrem",
                                "totbidrem",
                                "offerrem",
                                "bidrem",
                                "offerrem1",
                                "bidrem1",
                            ):
                                if k in tick and tick.get(k) is not None:
                                    mini[k] = tick.get(k)
                            ob2 = calc_orderbook_features(mini)
                            ob2.setdefault("_ts_epoch", int(sec_key))
                            if not bool(ob2.get("_invalid")):
                                ob = ob2
                            else:
                                # Allow buffering when bid/ask exist and there is any qty/total,
                                # even if the feature extractor flagged it invalid.
                                try:
                                    bid_ok = float(mini.get("bidho1") or mini.get("bidho") or 0.0) > 0.0
                                    ask_ok = float(mini.get("offerho1") or mini.get("offerho") or 0.0) > 0.0
                                    qty_ok = (
                                        float(mini.get("totbidrem") or 0.0) > 0.0
                                        or float(mini.get("totofferrem") or 0.0) > 0.0
                                        or float(mini.get("bidrem1") or mini.get("bidrem") or 0.0) > 0.0
                                        or float(mini.get("offerrem1") or mini.get("offerrem") or 0.0) > 0.0
                                    )
                                    if bid_ok and ask_ok and qty_ok:
                                        ob = ob2
                                except Exception as _e:
                                    logger.debug("오류 무시: %s", _e)
                        except Exception as _e:
                            logger.debug("오류 무시: %s", _e)

                    if bool(ob.get("_invalid")):
                        return  # invalid OB snapshot — do not append/update last snapshot

                    # signature-based duplicate skip
                    sig = (
                        round(float(ob.get("obi", 0.0) or 0.0), 6),
                        round(float(ob.get("spread", 0.0) or 0.0), 6),
                        round(float(ob.get("level1_ratio", 0.0) or 0.0), 6),
                        round(float(ob.get("totbidrem", 0.0) or 0.0), 3),
                        round(float(ob.get("totofferrem", 0.0) or 0.0), 3),
                    )

                    if self._last_fo0_second == sec_key and self._last_fo0_sig == sig:
                        return

                    # enforce 1 record per second (keep the latest within the second)
                    with self._ob_lock:
                        if self._last_fo0_second == sec_key and self._ob_records:
                            self._ob_records[-1] = ob
                        else:
                            self._ob_records.append(ob)

                        self._last_fo0_second = sec_key
                        self._last_fo0_sig = sig
                        self._last_ob_snapshot = dict(ob)
                    if self._fo0_log_schema and (not self._fo0_schema_logged):
                        try:
                            if all(float(ob.get(k, 0.0) or 0.0) == 0.0 for k in ("totbidrem", "totofferrem", "spread")):
                                logger.info(
                                    "[FO0_SCHEMA] keys=%s",
                                    sorted([str(k) for k in ob.keys()]),
                                )
                                self._fo0_schema_logged = True
                        except Exception as _e:
                            logger.debug("오류 무시: %s", _e)
                except Exception as e:
                    logger.debug("[FO0] processing error: %s", e, exc_info=True)

    def _get_now_dt(self, *, now_override: Any = None) -> datetime:
        now_dt = now_override
        if not isinstance(now_dt, datetime):
            now_dt = datetime.now()
        return now_dt

    def _update_fc0_stale_detection(self) -> None:
        try:
            # Best-effort FC0 stale detection (timestamp quality / missing ticks).
            fc0_age = None
            try:
                tp = getattr(self, "tick_processor", None)
                if tp is not None and bool(getattr(tp, "market_closed", False)):
                    # 장종료(JIF close) 이후에는 FC0 미수신이 정상 동작이므로 stale 경고 비활성.
                    self._metrics_set("fc0_age_sec", None)
                    return
            except Exception as _e:
                logger.debug("[_update_fc0_stale_detection] 오류 무시: %s", _e)
            try:
                if self._last_fc0_seen_epoch is not None:
                    fc0_age = float(time.time()) - float(self._last_fc0_seen_epoch)
            except Exception:
                fc0_age = None
            if fc0_age is not None:
                self._metrics_set("fc0_age_sec", float(fc0_age))
            stale_thr = float(getattr(self, "_fc0_stale_threshold_sec", 10.0) or 10.0)
            cooldown_sec = float(getattr(self, "_fc0_stale_cooldown_sec", 60.0) or 60.0)
            # FC0를 한 번도 받지 못한 초기 상태(fc0_age is None)는 stale 경고로 보지 않는다.
            if (fc0_age is not None) and (float(fc0_age) > float(stale_thr)):
                warned_recently = (
                    self._last_fc0_stale_warn_epoch is not None
                    and (float(time.time()) - float(self._last_fc0_stale_warn_epoch)) < float(cooldown_sec)
                )
                if not warned_recently:
                    try:
                        tp = getattr(self, "tick_processor", None)
                        if tp is not None:
                            ft = len(getattr(tp, "futures_ticks", []) or [])
                            ct = int(getattr(tp, "call_option_ticks", 0) or 0)
                            pt = int(getattr(tp, "put_option_ticks", 0) or 0)
                            if int(ft) <= 0 and int(ct) <= 0 and int(pt) <= 0:
                                return
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    self._last_fc0_stale_warn_epoch = float(time.time())
                    self._metrics_inc("fc0_stale_warnings")
                    logger.warning(
                        "[FC0_STALE] age_sec=%s (no FC0 futures ticks recently; check subscription/feed)",
                        fc0_age,
                    )
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)

    def _compute_flow_features(self, *, now_dt: datetime, current_price: float) -> Dict[str, float]:
        out = {"ofi_1s": 0.0, "ofi_5s": 0.0, "vwap_dev": 0.0}
        try:
            ticks = getattr(self.tick_processor, "futures_ticks", None)
            if ticks is None:
                return out
            if not ticks:
                return out

            try:
                cutoff_1s = now_dt - timedelta(seconds=1.0)
            except Exception:
                cutoff_1s = None
            try:
                cutoff_5s = now_dt - timedelta(seconds=5.0)
            except Exception:
                cutoff_5s = None

            signed_1s = 0.0
            signed_5s = 0.0
            v_sum = 0.0
            pv_sum = 0.0

            for rec in reversed(ticks):
                try:
                    ts = rec.get("timestamp")
                except Exception:
                    ts = None
                if cutoff_5s is not None and isinstance(ts, datetime) and ts < cutoff_5s:
                    break

                px = float(rec.get("price") or 0.0)
                if px <= 0.0:
                    continue
                vol = float(rec.get("cvolume") or 0.0)
                if vol <= 0.0:
                    vol = float(rec.get("volume") or 0.0)
                if vol <= 0.0:
                    continue

                bid = float(rec.get("bid") or 0.0)
                ask = float(rec.get("ask") or 0.0)
                sign = 0.0
                if bid > 0.0 and ask > 0.0:
                    mid = 0.5 * (bid + ask)
                    if px >= mid:
                        sign = 1.0
                    elif px < mid:
                        sign = -1.0
                else:
                    sign = 1.0 if px >= float(current_price) else -1.0

                signed_5s += sign * float(vol)
                try:
                    if cutoff_1s is None or (isinstance(ts, datetime) and ts >= cutoff_1s):
                        signed_1s += sign * float(vol)
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

                v_sum += float(vol)
                pv_sum += float(px) * float(vol)

            out["ofi_1s"] = float(signed_1s)
            out["ofi_5s"] = float(signed_5s)
            if v_sum > 0.0 and float(current_price) > 0.0:
                vwap = pv_sum / v_sum
                out["vwap_dev"] = float((float(current_price) - float(vwap)) / float(current_price))
        except Exception:
            return out
        return out

    def _get_current_price_or_error(self) -> float:
        current_price = self.tick_processor.get_current_price()
        if current_price is None:
            raise RuntimeError("no_price")
        return float(current_price)

    def _get_minute_df_or_error(self, *, warmup_bars: int) -> "pd.DataFrame":
        w = int(warmup_bars)
        if adaptive_uses_kospi_spot_index_minute_bars(self):
            df = self.tick_processor.get_kospi_minute_df(w)
        else:
            df = self.tick_processor.get_futures_minute_df(w)
        if df is None or df.empty or len(df) < int(self.min_minute_bars_required):
            raise RuntimeError(f"insufficient_minutes:{0 if df is None else len(df)}")
        return df

