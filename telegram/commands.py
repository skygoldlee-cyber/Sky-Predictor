"""PipelineTelegramBridge 분리 모듈.

이 파일은 telegram_notifier.py에서 분리된 Mixin 클래스입니다.
직접 인스턴스화하지 마세요.
"""
from __future__ import annotations
import logging
import threading
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CommandsMixin:
    """Telegram 커맨드 핸들러 Mixin.

    지원 커맨드:
        /predict /status /pause /resume /interval /regime
        /reset /json /trade_status /trade_gate /help
    """

    def _handle_command(self, command: str, chat_id: int) -> None:
        """텔레그램 명령어 처리."""
        try:
            tp = getattr(self._pipeline, "tick_processor", None)
            if tp is not None and bool(tp.market_closed):
                # CQ-04: 장 종료 중 명령 묵살 → 안내 응답
                self._notifier.send_text("🔒 <b>장 종료 중</b> — 예측이 비활성 상태입니다.")
                return
        except Exception:
            pass

        # 명령어와 인수 분리 ("/interval 30" → cmd="/interval", args="30")
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/predict", "/@"): 
            self._notifier.send_text("⏳ 예측 중...", parse_mode="HTML")
            self.predict_now(force=True, include_dir_summary=True)

        elif cmd == "/status":
            # ENH-01: 상태 정보 보강
            if self._user_pause_event.is_set():
                state_str = "⏸ 사용자 일시정지"
            elif self._market_paused:
                state_str = "🔒 장 종료 대기"
            else:
                state_str = "▶️ 실행 중"
            result = self._last_result
            with self._notifier._signal_lock:
                last_signal = self._notifier._last_signal or "-"
            send_count = self._notifier.get_send_count_total()
            if result and "error" not in result:
                signal = result.get("signal", "?")
                prob = result.get("prob", 0)
                pred_time = str(result.get("prediction_time", ""))[:19]
                msg = (
                    f"📊 <b>현재 상태</b>: {state_str}\n"
                    f"마지막 예측: {pred_time}\n"
                    f"신호: <b>{signal}</b> | 확률: {prob:.1%}\n"
                    f"마지막 전송 신호: <b>{last_signal}</b>\n"
                    f"총 전송 횟수: {send_count}회\n"
                    f"예측 주기: {self._interval:.0f}초"
                )
            else:
                msg = (
                    f"📊 <b>현재 상태</b>: {state_str}\n"
                    f"마지막 예측 없음\n"
                    f"총 전송 횟수: {send_count}회\n"
                    f"예측 주기: {self._interval:.0f}초"
                )
            self._notifier.send_text(msg)

        elif cmd == "/pause":
            self._user_pause_event.set()
            self._notifier.send_text("⏸ <b>알림 일시정지</b>")

        elif cmd == "/resume":
            self._user_pause_event.clear()
            self._notifier.send_text("▶️ <b>알림 재개</b>")
            self.predict_now(force=True)

        elif cmd == "/json":
            if self._last_result:
                self._notifier.send_json_result(self._last_result)
            else:
                self._notifier.send_text("아직 예측 결과가 없습니다.")

        elif cmd == "/interval":
            # /interval <초> — 예측 주기 변경
            # 인수 없이 "/interval" 만 입력하면 현재 주기를 조회
            if not args:
                self._notifier.send_text(
                    f"⏱ 현재 예측 주기: <b>{self._interval:.0f}초</b>\n"
                    f"변경: /interval &lt;초&gt;  "
                    f"(범위: {self._INTERVAL_MIN:.0f}–{self._INTERVAL_MAX:.0f}초)"
                )
                return
            try:
                new_sec = float(args)
            except ValueError:
                self._notifier.send_text(
                    f"❌ 잘못된 값: <code>{args}</code>\n"
                    f"숫자(초)를 입력하세요. 예: /interval 30"
                )
                return
            if not (self._INTERVAL_MIN <= new_sec <= self._INTERVAL_MAX):
                self._notifier.send_text(
                    f"❌ 범위 초과: {new_sec:.0f}초\n"
                    f"허용 범위: {self._INTERVAL_MIN:.0f}–{self._INTERVAL_MAX:.0f}초"
                )
                return
            old_sec = self._interval
            self._interval = new_sec
            logger.info(
                "[TG][INTERVAL] 예측 주기 변경: %.0f초 → %.0f초",
                old_sec, new_sec,
            )
            self._notifier.send_text(
                f"⏱ 예측 주기 변경: <b>{old_sec:.0f}초</b> → <b>{new_sec:.0f}초</b>"
            )

        elif cmd == "/regime":
            # /regime — 현재 시장 레짐 조회
            # _last_result에서 읽음. 없으면 즉시 예측을 1회 시도하여 획득.
            result = self._last_result
            regime: Optional[str] = None
            prob: Optional[float] = None
            signal: Optional[str] = None
            pred_time: str = ""

            if result and "error" not in result:
                regime = result.get("regime")
                prob = result.get("prob")
                signal = str(result.get("signal") or "")
                pred_time = str(result.get("prediction_time", ""))[:19]

            if regime is None:
                # 캐시된 결과 없음 → 즉시 예측 1회 시도
                try:
                    fresh = self._pipeline.get_prediction()
                    if fresh and "error" not in fresh:
                        self._last_result = fresh
                        regime = fresh.get("regime")
                        prob = fresh.get("prob")
                        signal = str(fresh.get("signal") or "")
                        pred_time = str(fresh.get("prediction_time", ""))[:19]
                except Exception as exc:
                    logger.warning("[TG][REGIME] 즉시 예측 실패: %s", exc)

            if regime:
                emoji = _REGIME_EMOJI.get(regime, "")
                signal_emoji = _SIGNAL_EMOJI.get(str(signal).upper(), "")
                prob_str = f"{prob:.1%}" if prob is not None else "-"
                msg = (
                    f"🌊 <b>시장 레짐</b>: {emoji} <b>{regime}</b>\n"
                    f"신호: {signal_emoji} {signal}  |  상승확률: {prob_str}\n"
                    f"기준: {pred_time}"
                )
            else:
                msg = "🌊 <b>시장 레짐</b>: 아직 레짐 정보가 없습니다.\n/predict 로 먼저 예측을 실행하세요."
            self._notifier.send_text(msg)

        elif cmd == "/reset":
            # /reset — 신호 중복 억제 상태 초기화
            # _last_signal을 지워 다음 예측이 신호 변경 없이도 전송되도록 함.
            # _last_boundary_minute도 초기화하여 DIR_SUMMARY가 다음 틱에 포함되도록 함.
            # _first_prediction_sent는 건드리지 않음 (재시작 의미가 아님).
            with self._notifier._signal_lock:
                old_signal = self._notifier._last_signal or "(없음)"
                self._notifier._last_signal = ""
            try:
                self._last_boundary_minute = -1
            except Exception:
                pass
            logger.info(
                "[TG][RESET] 신호 억제 상태 초기화 (이전 last_signal=%s)",
                old_signal,
            )
            self._notifier.send_text(
                f"🔄 <b>신호 억제 상태 초기화</b>\n"
                f"이전 신호: <code>{old_signal}</code> → 클리어\n"
                f"다음 예측은 신호 종류에 관계없이 전송됩니다."
            )

        elif cmd == "/trade_status":
            # /trade_status — 현재 포지션 및 오늘 거래 결과 조회
            if self._trade_gate is None or not getattr(self._trade_gate, "enabled", False):
                self._notifier.send_text(
                    "⚠️ <b>TradeGate 비활성</b>\n"
                    "config.json의 <code>trade_gate.enabled</code>를 <code>true</code>로 설정하세요."
                )
                return
            try:
                summary = self._trade_gate.get_daily_summary_dict()
                state   = self._trade_gate._state.get_state()
                cfg     = self._trade_gate._cfg

                if state.has_position and state.active is not None:
                    pos = state.active
                    tp = getattr(self._pipeline, "tick_processor", None)
                    cur_price = float(tp.get_current_price() if tp is not None else 0.0)
                    pnl_now = (
                        cur_price - pos.entry_price if pos.side.value == "LONG"
                        else pos.entry_price - cur_price
                    ) if cur_price > 0 else 0.0
                    pnl_emoji = "📈" if pnl_now > 0 else ("📉" if pnl_now < 0 else "➖")
                    side_str = "매수" if pos.side.value == "LONG" else "매도"
                    eff_target = pos.record.entry_target_pt if pos.record.entry_target_pt > 0 else cfg.target_profit_pt
                    eff_stop   = pos.record.entry_stop_pt   if pos.record.entry_stop_pt   > 0 else cfg.stop_loss_pt
                    hold_min   = (self._now_fn() - pos.entry_time).total_seconds() / 60
                    iv_line = (
                        f"\nIV: {pos.record.entry_atm_iv:.1%}  목표: {eff_target:.2f}pt  손절: {eff_stop:.2f}pt"
                        if pos.record.entry_atm_iv > 0.0 else ""
                    )
                    pos_block = (
                        f"📌 <b>보유 중 ({side_str})</b>\n"
                        f"진입가: <code>{pos.entry_price:.2f}</code>  "
                        f"({pos.entry_time.strftime('%H:%M')})\n"
                        f"{pnl_emoji} 현재 손익: <b>{pnl_now:+.2f}pt</b>  "
                        f"({hold_min:.0f}분 경과)"
                        f"{iv_line}\n"
                        f"━━━━━━━━━━━━━━━━━━━\n"
                    )
                else:
                    pos_block = "📌 현재 포지션: 없음\n━━━━━━━━━━━━━━━━━━━\n"

                total = summary.get("count", 0)
                wins  = summary.get("wins", 0)
                losses= summary.get("losses", 0)
                pnl   = summary.get("total_pnl_pt", 0.0)
                pnl_e = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
                used  = [s.value for s in state.used_slots]
                slots_str = " ".join(
                    f"[{x}✓]" if x in used else f"[{x}]"
                    for x in ("A", "B", "C")
                )
                trades_block = ""
                for i, t in enumerate(summary.get("trades", []), 1):
                    s_arrow = "▲" if t["side"] == "LONG" else "▼"
                    t_str   = t["entry_time"][11:16]
                    p_str   = f"{t['pnl_pt']:+.2f}pt"
                    r_str   = t.get("close_reason", "-")
                    iv_str  = f"  IV {t.get('entry_atm_iv', 0)*100:.0f}%" if t.get("entry_atm_iv") else ""
                    trades_block += f"  {i}. {s_arrow} {t_str}  {p_str}  ({r_str}){iv_str}\n"

                msg = (
                    f"📊 <b>TradeGate 상태</b>  ({summary.get('date', '-')})\n"
                    f"{pos_block}"
                    f"슬롯: {slots_str}  진입: {total}/{cfg.max_daily_trades}회\n"
                    f"승: {wins}  패: {losses}  "
                    f"{pnl_e} 합계: <b>{pnl:+.2f}pt</b>\n"
                    + (f"\n{trades_block}" if trades_block else "")
                )
                self._notifier.send_text(msg)
            except Exception as exc:
                logger.warning("[TG][TRADE_STATUS] 조회 실패: %s", exc)
                self._notifier.send_text(f"❌ trade_status 조회 실패: {exc}")

        elif cmd == "/trade_gate":
            # /trade_gate on|off — 런타임 활성화/비활성화
            if not args:
                enabled = (
                    self._trade_gate is not None
                    and getattr(self._trade_gate, "enabled", False)
                )
                state_str = "✅ 활성" if enabled else "⏹ 비활성"
                self._notifier.send_text(
                    f"🎛 <b>TradeGate</b>: {state_str}\n"
                    f"변경: /trade_gate on  또는  /trade_gate off"
                )
                return

            subcmd = args.strip().lower()
            if subcmd not in ("on", "off"):
                self._notifier.send_text(
                    f"❌ 잘못된 인수: <code>{args}</code>\n"
                    f"사용법: /trade_gate on  또는  /trade_gate off"
                )
                return

            if not _TRADE_GATE_AVAILABLE or TradeExecutionGate is None:
                self._notifier.send_text("❌ trade_gate 모듈을 불러올 수 없습니다.")
                return

            try:
                if self._trade_gate is None:
                    from trading.gate import TradeGateConfig as _TGC
                    self._trade_gate = TradeExecutionGate(
                        self._notifier, _TGC(enabled=(subcmd == "on"))
                    )
                else:
                    old_cfg = self._trade_gate._cfg
                    from trading.gate import TradeGateConfig as _TGC
                    new_cfg = _TGC.from_dict({
                        slot: getattr(old_cfg, slot)
                        for slot in old_cfg.__slots__
                    })
                    object.__setattr__(new_cfg, "enabled", subcmd == "on")
                    self._trade_gate = TradeExecutionGate(self._notifier, new_cfg)

                state_str = "✅ 활성화" if subcmd == "on" else "⏹ 비활성화"
                logger.info("[TG][TRADE_GATE] /trade_gate %s", subcmd)
                self._notifier.send_text(f"🎛 <b>TradeGate {state_str}</b>")

                if subcmd == "on" and (
                    self._trade_monitor_thread is None
                    or not self._trade_monitor_thread.is_alive()
                ):
                    self._trade_monitor_thread = threading.Thread(
                        target=self._trade_monitor_loop,
                        daemon=True,
                        name="TradeMonitor",
                    )
                    self._trade_monitor_thread.start()
                    logger.info("[TG][TRADE] 진입/청산 감시 모니터 재시작")
            except Exception as exc:
                logger.exception("[TG][TRADE_GATE] 전환 실패")
                self._notifier.send_text(f"❌ trade_gate 전환 실패: {exc}")

        elif cmd == "/help":
            help_text = (
                "📖 <b>SkyEbest 텔레그램 봇 명령어</b>\n\n"
                "/predict              — 즉시 예측 실행\n"
                "/status               — 현재 상태 조회\n"
                "/pause                — 알림 일시정지\n"
                "/resume               — 알림 재개\n"
                "/interval &lt;초&gt;       — 예측 주기 변경 (조회: 인수 생략)\n"
                "/regime               — 현재 시장 레짐 조회\n"
                "/reset                — 신호 억제 상태 초기화\n"
                "/json                 — 마지막 예측 JSON 출력\n"
                "/trade_status         — 포지션 및 오늘 거래 현황\n"
                "/trade_gate on|off    — 진입/청산 게이트 활성화 전환\n"
                "/help                 — 이 도움말"
            )
            self._notifier.send_text(help_text)

        else:
            self._notifier.send_text(f"❓ 알 수 없는 명령: {command}\n/help 를 입력하세요.")

