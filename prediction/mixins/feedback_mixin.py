"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeedbackMixin:
    """Mixin: FeedbackMixin methods extracted from PredictionPipeline."""

    def _prob_to_dir(self, prob: Optional[float]) -> Optional[str]:
        """확률값을 BUY/SELL/None으로 변환한다.

        None을 반환하면 해당 예측기의 방향성을 알 수 없음을 의미하며,
        피드백 업데이트를 건너뛴다.
        """
        try:
            if prob is None:
                return None
            pf = float(prob)
            if pf >= float(self._buy_threshold):
                return "BUY"
            if pf <= float(self._sell_threshold):
                return "SELL"
            return "HOLD"
        except Exception:
            return None

    def _maybe_process_feedback(self, *, now_dt: datetime, current_price: float) -> None:
        """Evaluate matured predictions and update ensemble weights (best-effort).

        Uses threshold-based labeling:
        - BUY if price_change >= +threshold
        - SELL if price_change <= -threshold
        - else HOLD

        threshold = tick_size * feedback_threshold_ticks
        """
        if not self._feedback_queue:
            return

        horizon_sec = float(max(1, int(self.prediction_minutes))) * 60.0

        # [IMP-FB-01] 피드백 임계값 동적화: ATM IV 기반으로 조정.
        # 고정된 tick_size × threshold_ticks 대신 당일 변동성을 반영한 임계값을 사용한다.
        # 공식: thr = max(base_thr, F × ATM_IV × sqrt(dt/252) × 0.5)
        #   dt = prediction_minutes / 1440 (하루 대비 예측 구간 비율)
        _base_thr = float(self._tick_size) * float(self._feedback_threshold_ticks)
        try:
            _atm_iv_fb = float(
                (getattr(self, "_last_opt_snap", None) or {}).get("atm_iv") or 0.0
            )
            _cur_px_fb = float(current_price or 0.0)
            _dt_ratio = float(max(1, int(self.prediction_minutes))) / 1440.0
            if _atm_iv_fb > 0.0 and _cur_px_fb > 0.0:
                _iv_thr = _cur_px_fb * _atm_iv_fb * math.sqrt(_dt_ratio / 252.0) * 0.5
                thr = max(_base_thr, float(_iv_thr))
            else:
                thr = _base_thr
        except Exception:
            thr = _base_thr

        skip_thr = float(self._tick_size) * float(self._feedback_skip_hold_ticks)
        if thr <= 0.0:
            return

        try:
            now_epoch = float(now_dt.timestamp())
        except Exception:
            now_epoch = float(time.time())

        matured: List[Dict[str, Any]] = []
        try:
            while self._feedback_queue:
                head = self._feedback_queue[0]
                try:
                    ts = float(head.get("ts_epoch") or 0.0)
                except Exception:
                    ts = 0.0
                try:
                    tgt = float(head.get("target_ts_epoch") or 0.0)
                except Exception:
                    tgt = 0.0
                if ts <= 0.0:
                    self._feedback_queue.popleft()
                    continue

                is_mature = False
                if tgt > 0.0:
                    is_mature = bool(now_epoch >= float(tgt))
                else:
                    is_mature = bool((now_epoch - ts) >= float(horizon_sec))

                if not is_mature:
                    break
                matured.append(self._feedback_queue.popleft())
        except Exception:
            matured = []

        if not matured:
            return

        upd = getattr(self.numeric_predictor, "update_adaptive_weights", None)
        if not callable(upd):
            return

        def _weight_from_prob(p: Optional[float]) -> float:
            try:
                if p is None:
                    return 0.0
                pf = float(p)
                margin = abs(float(pf) - 0.5)
                if margin >= float(self._confidence_high_margin):
                    return float(self._feedback_weight_high)
                if margin >= float(self._confidence_mid_margin):
                    return float(self._feedback_weight_mid)
                return float(self._feedback_weight_low)
            except Exception:
                return 0.0

        for rec in matured:
            try:
                entry_px = float(rec.get("price") or 0.0)
            except Exception:
                entry_px = 0.0
            if entry_px <= 0.0:
                continue

            eval_price = float(current_price)
            if bool(self._feedback_use_price_snapshot):
                snapshot_ok = False
                try:
                    tgt = float(rec.get("target_ts_epoch") or 0.0)
                except Exception:
                    tgt = 0.0
                if tgt > 0.0:
                    try:
                        tgt_dt = datetime.fromtimestamp(float(tgt))
                        px = self.tick_processor.get_price_at(tgt_dt)
                        if px is None and float(self._feedback_snapshot_tolerance_sec) > 0.0:
                            fn_near = getattr(self.tick_processor, "get_price_near", None)
                            if callable(fn_near):
                                px = fn_near(tgt_dt, tolerance_sec=float(self._feedback_snapshot_tolerance_sec))
                        if px is not None and float(px) > 0.0:
                            eval_price = float(px)
                            snapshot_ok = True
                            self._metrics_inc("feedback_snapshot_used")
                    except Exception:
                        pass
                if not snapshot_ok:
                    try:
                        self._metrics_inc("feedback_snapshot_miss")
                    except Exception:
                        pass
                    if bool(self._feedback_snapshot_required):
                        try:
                            self._metrics_inc("feedback_snapshot_skipped_required")
                        except Exception:
                            pass
                        continue

            delta = float(eval_price) - float(entry_px)
            if delta >= float(thr):
                actual = "BUY"
            elif delta <= -float(thr):
                actual = "SELL"
            else:
                actual = "HOLD"

            if actual == "HOLD":
                try:
                    if float(skip_thr) > 0.0 and abs(float(delta)) < float(skip_thr):
                        try:
                            self._metrics_inc("feedback_skipped_hold_small_move")
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

            t_dir = self._prob_to_dir(rec.get("transformer_prob"))
            f_dir = self._prob_to_dir(rec.get("tft_prob"))
            if t_dir is None:
                continue

            transformer_correct = bool(t_dir == actual)
            _tft_available = f_dir is not None

            if _tft_available:
                tft_correct = bool(f_dir == actual)
                f_w = _weight_from_prob(rec.get("tft_prob"))
            else:
                # TFT stub 모드: weight update 대상에서 TFT 제외하고 메트릭에 기록
                tft_correct = bool(transformer_correct)
                f_w = 0.0
                try:
                    self._metrics_inc("feedback_tft_stub_skipped")
                except Exception:
                    pass

            t_w = _weight_from_prob(rec.get("transformer_prob"))
            try:
                try:
                    # [IMP-ENS-01] transformer_prob/tft_prob를 함께 전달해 Brier Score 점수화
                    upd(
                        transformer_correct=transformer_correct,
                        tft_correct=tft_correct,
                        transformer_weight=float(t_w),
                        tft_weight=float(f_w),
                        transformer_prob=rec.get("transformer_prob"),
                        tft_prob=rec.get("tft_prob") if _tft_available else None,
                    )
                except TypeError:
                    upd(transformer_correct=transformer_correct, tft_correct=tft_correct)
                self._metrics_inc("feedback_weight_updates")
            except Exception:
                pass
            try:
                self._metrics_inc("feedback_evaluations")
            except Exception:
                pass

