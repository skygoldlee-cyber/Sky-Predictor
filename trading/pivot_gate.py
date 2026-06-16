"""
pivot_gate.py
=============
피봇 기반 진입/청산 게이트.

피봇 확정 시점 기반으로 진입/청산 결정을 내린다.

진입 조건:
  - 매수(LONG): 피봇이 저가(L)로 확정되는 시점
  - 매도(SHORT): 피봇이 고가(H)로 확정되는 시점

청산 조건:
  - 매수(LONG): 다음 피봇 고가(H) 확정 시점 OR 가격이 피봇 저가 아래로 내려가는 경우
  - 매도(SHORT): 다음 피봇 저가(L) 확정 시점 OR 가격이 피봇 고가 위로 올라가는 경우

사용법:
    from trading.pivot_gate import PivotExecutionGate, PivotGateConfig
    from trading.state import TradeStateManager
    
    config = PivotGateConfig()
    gate = PivotExecutionGate(notifier, config)
    
    # 피봇 확정 시 호출
    gate.on_pivot_confirmed(pivot_type="LOW", pivot_price=380.0, current_price=380.5)
    
    # 주기적으로 가격 돌파 체크
    gate.check_price_breakout(current_price=381.0)
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, Optional

from .state import (
    ActivePosition,
    CloseReason,
    DailyState,
    PositionSide,
    TradeRecord,
    TradeSlot,
    TradeStateManager,
)
from .position_sizing import PositionSizer, SizingConfig

logger = logging.getLogger(__name__)

# ── 설정 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class PivotGateConfig:
    """피봇 게이트 설정."""
    enabled: bool = True
    initial_capital: float = 10000000.0
    tick_size: float = 0.05
    commission_rate: float = 0.00015
    slippage_ticks: int = 1
    force_close_time: str = "15:30"
    market_open_time: str = "08:45"
    
    # 포지션 사이징 (SizingConfig로 분리)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    
    # 이력 저장
    history_save_enabled: bool = True
    history_dir: str = "trade_history"
    
    # 리스크 관리
    max_daily_loss: float = 0.02  # 일일 최대 손실 한도 (자본 대비)
    stop_loss_buffer_ticks: int = 2  # 손절 버퍼 (틱)


# ── 피봇 게이트 클래스 ───────────────────────────────────────────────────────

class PivotExecutionGate:
    """피봇 기반 진입/청산 게이트."""
    
    def __init__(
        self,
        notifier,
        config: PivotGateConfig,
    ):
        """초기화.
        
        Args:
            notifier: 텔레그램 알림 객체
            config: 게이트 설정
        """
        self._notifier = notifier
        self._cfg = config
        self._lock = threading.RLock()
        self._state = TradeStateManager(lock=self._lock)  # 동일 락 공유 (데드락 방지)
        self._futures_index_ratio: float = 1.0  # 선물/지수 비율 (기본값: 1.0)
        
        # 시간 파싱 캐싱 (__init__에서 한 번만 파싱)
        self._force_close_time_cache: Optional[time] = self._parse_time(config.force_close_time)
        self._market_open_time_cache: Optional[time] = self._parse_time(config.market_open_time)
        
        # 실시간 백테스트 상태
        self._backtest_state = {
            "pivots": [],  # 확정된 피봇 목록
            "trades": [],  # 시뮬레이션 거래
            "total_profit": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
            "stop_loss_trades": 0,
        }
        
        # 포지션 사이저 초기화
        self._sizer = PositionSizer(self._cfg.sizing)
        
        # 이력 저장 디렉토리 초기화
        if self._cfg.history_save_enabled:
            try:
                os.makedirs(self._cfg.history_dir, exist_ok=True)
            except Exception as e:
                logger.warning("[PIVOT_GATE] history_dir 생성 실패: %s", e)
    
    @property
    def enabled(self) -> bool:
        return self._cfg.enabled
    
    @staticmethod
    def _parse_time(time_str: str):
        """시간 문자열을 datetime.time 객체로 파싱.
        
        Args:
            time_str: "HH:MM" 형식의 문자열
        
        Returns:
            datetime.time 객체 또는 파싱 실패 시 None
        """
        try:
            h, m = time_str.split(":")
            return time(int(h), int(m))
        except Exception as e:
            logger.warning("[PIVOT_GATE] time_str 파싱 오류: %s, None 반환", e)
            return None
    
    def on_pivot_confirmed(
        self,
        pivot_type: str,
        pivot_price: float,
        current_price: float,
    ) -> None:
        """피봇 확정 시 호출되어 진입/청산 판단.
        
        Args:
            pivot_type: "HIGH" 또는 "LOW" (KOSPI 지수 피봇 타입)
            pivot_price: 피봇 가격 (KOSPI 지수)
            current_price: 현재 가격 (KP200 선물 가격)
        
        Note:
            - 피봇 확정은 KOSPI 지수에서 감지
            - 실제 매매는 KP200 선물 가격으로 진행
            - pivot_price는 KOSPI 지수 가격이며, 가격 돌파 체크용으로 사용
        """
        if not self._cfg.enabled:
            return
        
        try:
            # 선물/지수 비율 업데이트 (현재 선물 가격 / 피봇 지수 가격)
            if pivot_price > 0 and current_price > 0:
                self._futures_index_ratio = current_price / pivot_price
            
            # 진입/청산 판단: current_price(KP200 선물 가격) 사용
            # pivot_price는 KOSPI 지수 가격으로 저장 (가격 돌파 체크용)
            self._on_pivot_confirmed_inner(pivot_type, pivot_price, current_price)
            self._update_simulation_metrics(pivot_type, pivot_price, current_price)
        except Exception:
            logger.exception("[PIVOT_GATE] on_pivot_confirmed 오류")
    
    def set_futures_index_ratio(self, ratio: float) -> None:
        """선물/지수 비율 설정.
        
        Args:
            ratio: 선물 가격 / 지수 가격 비율
        """
        if ratio > 0:
            self._futures_index_ratio = ratio
            logger.info("[PIVOT_GATE] 선물/지수 비율 설정: %.4f", ratio)
    
    def _update_simulation_metrics(self, pivot_type: str, pivot_price: float, current_price: float) -> None:
        """실시간 시뮬레이션 지표 업데이트.
        
        Args:
            pivot_type: 피봇 타입 ("HIGH" 또는 "LOW")
            pivot_price: 피봇 가격 (KOSPI 지수)
            current_price: 현재 가격 (KP200 선물)
        """
        with self._lock:
            try:
                # 새 피봇 추가
                self._backtest_state["pivots"].append({
                    "type": pivot_type,
                    "price": pivot_price,
                    "futures_price": current_price,
                    "time": datetime.now().strftime("%H:%M"),
                })
                
                pivots = self._backtest_state["pivots"]
                
                # 피봇이 2개 이상이어야 시뮬레이션 가능
                if len(pivots) < 2:
                    return
                
                # 마지막 두 피봇으로 시뮬레이션
                cur = pivots[-2]
                nxt = pivots[-1]
                
                # 진입가: 이전 피봇 확정 시 선물 가격
                # 청산가: 현재 피봇 확정 시 선물 가격
                c_entry = cur["futures_price"]
                n_exit = nxt["futures_price"]
                
                # 시뮬레이션
                trade = None
                if cur["type"] == "LOW" and nxt["type"] == "HIGH":
                    # 저점 매수 → 고점 청산
                    profit = n_exit - c_entry
                    result = "WIN" if profit > 0 else "LOSS"
                    self._backtest_state["total_profit"] += profit
                    if profit > 0:
                        self._backtest_state["win_trades"] += 1
                    else:
                        self._backtest_state["loss_trades"] += 1
                    trade = {
                        "type": "BUY",
                        "entry_time": cur["time"],
                        "entry_price": c_entry,
                        "exit_time": nxt["time"],
                        "exit_price": n_exit,
                        "profit": profit,
                        "result": result,
                        "exit_reason": "TARGET",  # 피봇 기반은 목표 청산만
                    }
                elif cur["type"] == "HIGH" and nxt["type"] == "LOW":
                    # 고점 매도 → 저점 청산
                    profit = c_entry - n_exit
                    result = "WIN" if profit > 0 else "LOSS"
                    self._backtest_state["total_profit"] += profit
                    if profit > 0:
                        self._backtest_state["win_trades"] += 1
                    else:
                        self._backtest_state["loss_trades"] += 1
                    trade = {
                        "type": "SELL",
                        "entry_time": cur["time"],
                        "entry_price": c_entry,
                        "exit_time": nxt["time"],
                        "exit_price": n_exit,
                        "profit": profit,
                        "result": result,
                        "exit_reason": "TARGET",  # 피봇 기반은 목표 청산만
                    }
                
                if trade:
                    self._backtest_state["trades"].append(trade)
                    
                    # 로그 출력
                    total_trades = len(self._backtest_state["trades"])
                    win_rate = self._backtest_state["win_trades"] / total_trades * 100 if total_trades > 0 else 0.0
                    logger.info(
                        "[PIVOT_SIMULATION] 실시간 시뮬레이션: %s 거래완료 | 총거래=%d 승률=%.1f%% 총수익=%.2fpt",
                        trade["type"], total_trades, win_rate, self._backtest_state["total_profit"],
                    )
                    
                    # 텔레그램 알림 (선택적) - 별도 스레드로 분리하여 락 블로킹 방지
                    if self._notifier:
                        msg = (
                            f"📊 *실시간 시뮬레이션*\n"
                            f"거래: {trade['type']} {trade['entry_time']}→{trade['exit_time']}\n"
                            f"진입: {trade['entry_price']:.2f} | 청산: {trade['exit_price']:.2f}\n"
                            f"수익: {trade['profit']:+.2f}pt ({trade['result']})\n"
                            f"총거래: {total_trades} | 승률: {win_rate:.1f}% | 총수익: {self._backtest_state['total_profit']:+.2f}pt\n"
                            f"⚠️ 샘플이 적을 때 지표는 참고용으로만 해석하세요"
                        )
                        threading.Thread(
                            target=self._send_telegram_notification,
                            args=(msg,),
                            daemon=True,
                        ).start()
            except Exception as e:
                logger.warning("[PIVOT_SIMULATION] 지표 업데이트 실패: %s", e)
    
    def check_price_breakout(self, current_price: float) -> None:
        """가격 돌파 체크 (주기적으로 호출).
        
        Args:
            current_price: 현재 가격
        """
        if not self._cfg.enabled:
            return
        
        try:
            self._check_price_breakout_inner(current_price)
        except Exception:
            logger.exception("[PIVOT_GATE] check_price_breakout 오류")
    
    def get_simulation_metrics(self) -> Dict[str, Any]:
        """실시간 시뮬레이션 지표 반환.
        
        Returns:
            시뮬레이션 지표 딕셔너리
        """
        with self._lock:
            total_trades = len(self._backtest_state["trades"])
            win_rate = self._backtest_state["win_trades"] / total_trades * 100 if total_trades > 0 else 0.0
            avg_profit = self._backtest_state["total_profit"] / total_trades if total_trades > 0 else 0.0
            
            # 샤프 비율은 샘플이 적을 때 의미 없으므로 실시간에서는 제외
            sharpe_ratio = 0.0
            if total_trades >= 3:
                profits = [t.get("profit", 0.0) for t in self._backtest_state["trades"]]
                if len(profits) > 1:
                    avg_profit_val = sum(profits) / len(profits)
                    # 표본 표준편차 사용 (/ (n-1))
                    std_profit = math.sqrt(sum((p - avg_profit_val) ** 2 for p in profits) / (len(profits) - 1))
                    sharpe_ratio = avg_profit_val / std_profit if std_profit > 0 else 0.0
            
            return {
                "total_pivots": len(self._backtest_state["pivots"]),
                "total_trades": total_trades,
                "win_trades": self._backtest_state["win_trades"],
                "loss_trades": self._backtest_state["loss_trades"],
                "stop_loss_trades": self._backtest_state["stop_loss_trades"],
                "win_rate": win_rate,
                "total_profit": self._backtest_state["total_profit"],
                "avg_profit": avg_profit,
                "sharpe_ratio": sharpe_ratio,  # 3건 이상일 때만 계산
                "trades": list(self._backtest_state["trades"]),  # 복사본 반환
            }
    
    def check_close(self, *, current_price: float = 0.0) -> None:
        """강제청산 조건을 점검.
        
        별도 감시 루프에서 주기적으로 호출한다.
        
        Args:
            current_price: 현재 가격
        """
        if not self._cfg.enabled:
            return
        
        try:
            self._check_close_inner(current_price=current_price)
        except Exception:
            logger.exception("[PIVOT_GATE] check_close 오류")
    
    # ── 내부 로직 ─────────────────────────────────────────────────────────────
    
    def _on_pivot_confirmed_inner(
        self,
        pivot_type: str,
        pivot_price: float,
        current_price: float,
    ) -> None:
        """피봇 확정 내부 로직.
        
        Args:
            pivot_type: 피봇 타입 (KOSPI 지수)
            pivot_price: 피봇 가격 (KOSPI 지수)
            current_price: 현재 가격 (KP200 선물)
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        with self._lock:
            state = self._state.get_state()
            
            # ── 청산 판단: 다음 피봇 확정 시 ──
            if state.has_position and state.active is not None:
                pos = state.active
                
                # 매수 포지션: 다음 피봇 고가(H) 확정 시 청산
                if pos.side == PositionSide.LONG and pivot_type == "HIGH":
                    logger.info(
                        "[PIVOT_GATE] 매수 포지션 청산 (다음 피봇 고가 확정) 피봇지수=%.2f @ 선물=%.2f",
                        pivot_price, current_price,
                    )
                    self._execute_close(
                        price=current_price,
                        now=now,
                        today_str=today_str,
                        reason=CloseReason.TARGET_PROFIT,
                    )
                    return
                
                # 매도 포지션: 다음 피봇 저가(L) 확정 시 청산
                if pos.side == PositionSide.SHORT and pivot_type == "LOW":
                    logger.info(
                        "[PIVOT_GATE] 매도 포지션 청산 (다음 피봇 저가 확정) 피봇지수=%.2f @ 선물=%.2f",
                        pivot_price, current_price,
                    )
                    self._execute_close(
                        price=current_price,
                        now=now,
                        today_str=today_str,
                        reason=CloseReason.TARGET_PROFIT,
                    )
                    return
            
            # ── 진입 판단 ──
            if not state.has_position:
                # 매수 진입: 피봇 저가(L) 확정
                if pivot_type == "LOW":
                    self._try_enter(
                        signal="BUY",
                        pivot_price=pivot_price,
                        current_price=current_price,
                        now=now,
                        today_str=today_str,
                        state=state,
                    )
                
                # 매도 진입: 피봇 고가(H) 확정
                elif pivot_type == "HIGH":
                    self._try_enter(
                        signal="SELL",
                        pivot_price=pivot_price,
                        current_price=current_price,
                        now=now,
                        today_str=today_str,
                        state=state,
                    )
    
    def _check_price_breakout_inner(self, current_price: float) -> None:
        """가격 돌파 체크 내부 로직.
        
        Args:
            current_price: 현재 KP200 선물 가격
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        with self._lock:
            state = self._state.get_state()
            
            if not state.has_position or state.active is None:
                return
            
            pos = state.active
            
            # pivot_price는 KOSPI 지수 가격, current_price는 KP200 선물 가격
            # 비교를 위해 pivot_price를 선물 가격으로 변환
            pivot_price_futures = pos.pivot_price * self._futures_index_ratio if pos.pivot_price else 0.0
            
            # 손절 버퍼 적용
            buffer_amount = self._cfg.stop_loss_buffer_ticks * self._cfg.tick_size
            
            # 매수 포지션: 가격이 피봇 저가(선물 변환값) 아래로 내려가면 청산
            if pos.side == PositionSide.LONG:
                stop_loss_price = pivot_price_futures - buffer_amount if pivot_price_futures > 0 else 0.0
                if stop_loss_price > 0 and current_price < stop_loss_price:
                    logger.info(
                        "[PIVOT_GATE] 매수 포지션 청산 (가격 돌파) current=%.2f < stop_loss=%.2f (pivot_futures=%.2f, pivot_index=%.2f, buffer=%.2f)",
                        current_price, stop_loss_price, pivot_price_futures, pos.pivot_price, buffer_amount,
                    )
                    self._execute_close(
                        price=current_price,
                        now=now,
                        today_str=today_str,
                        reason=CloseReason.STOP_LOSS,
                    )
            
            # 매도 포지션: 가격이 피봇 고가(선물 변환값) 위로 올라가면 청산
            elif pos.side == PositionSide.SHORT:
                stop_loss_price = pivot_price_futures + buffer_amount if pivot_price_futures > 0 else 0.0
                if stop_loss_price > 0 and current_price > stop_loss_price:
                    logger.info(
                        "[PIVOT_GATE] 매도 포지션 청산 (가격 돌파) current=%.2f > stop_loss=%.2f (pivot_futures=%.2f, pivot_index=%.2f, buffer=%.2f)",
                        current_price, stop_loss_price, pivot_price_futures, pos.pivot_price, buffer_amount,
                    )
                    self._execute_close(
                        price=current_price,
                        now=now,
                        today_str=today_str,
                        reason=CloseReason.STOP_LOSS,
                    )
    
    def _check_close_inner(self, *, current_price: float) -> None:
        """강제청산 체크 내부 로직."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        with self._lock:
            state = self._state.get_state()
            
            if not state.has_position or state.active is None:
                return
            
            # 강제청산 시각 도달 여부
            if current_price > 0.0 and self._is_after_force_close(now):
                logger.info("[PIVOT_GATE] 강제청산 시각 도달 price=%.2f", current_price)
                self._execute_close(
                    price=current_price,
                    now=now,
                    today_str=today_str,
                    reason=CloseReason.FORCE_CLOSE,
                )
    
    def _try_enter(
        self,
        signal: str,
        pivot_price: float,
        current_price: float,
        now: datetime,
        today_str: str,
        state: DailyState,
    ) -> None:
        """진입 시도.
        
        Args:
            signal: "BUY" 또는 "SELL"
            pivot_price: 피봇 가격 (KOSPI 지수)
            current_price: 현재 가격 (KP200 선물 가격)
        
        Note:
            호출자가 이미 락을 보유하고 있어야 합니다.
        """
        # 현재가 확인
        if current_price <= 0.0:
            logger.warning("[PIVOT_GATE] 진입 차단 — 현재가 미확인")
            return
        
        # 일일 최대 손실 한도 체크
        # max_daily_loss는 비율(0.02 = 2%), total_pnl_pct는 퍼센트 단위
        if self._cfg.max_daily_loss > 0:
            daily_loss_pct = state.total_pnl_pct
            if daily_loss_pct < -self._cfg.max_daily_loss * 100:
                logger.warning(
                    "[PIVOT_GATE] 진입 차단 — 일일 손실 한도 초과: %.2f%% < %.2f%%",
                    daily_loss_pct, -self._cfg.max_daily_loss * 100,
                )
                return
        
        # 슬리피지 적용 (KP200 선물 가격 기준)
        slippage = self._cfg.slippage_ticks * self._cfg.tick_size
        if signal == "BUY":
            entry_price = current_price + slippage
        else:  # SELL
            entry_price = current_price - slippage
        
        # 포지션 사이징
        # total_pnl_pct는 퍼센트 단위, 자본 계산에 반영
        current_equity = self._cfg.initial_capital * (1 + state.total_pnl_pct / 100)
        size_info = self._sizer.calculate_size(
            equity=current_equity,
            entry_price=entry_price,
            stop_loss_pt=0.0,  # 피봇 기반은 고정 손절 없음
        )
        size = size_info["size"]
        
        # 포지션 사이드 결정
        side = PositionSide.LONG if signal == "BUY" else PositionSide.SHORT
        
        # 거래 기록 생성 (진입 시점)
        # entry_price는 KP200 선물 가격, pivot_price는 KOSPI 지수 가격
        trade_record = TradeRecord(
            slot=TradeSlot.A,  # 피봇 기반은 슬롯 구분 안 함
            side=side,
            entry_price=entry_price,
            entry_time=now,
            entry_signal=signal,
            entry_confidence="HIGH",  # 피봇 확정은 HIGH로 간주
            entry_prob=0.8 if signal == "BUY" else 0.2,
            pivot_price=pivot_price,  # 피봇 가격 저장 (KOSPI 지수)
            position_size=size,  # 포지션 사이즈
            capital_used=entry_price * size,  # 사용 자본
            sizing_method=self._cfg.sizing.method.value,  # 사이징 방법
        )
        
        # 포지션 생성
        pos = ActivePosition(
            record=trade_record,
            pivot_price=pivot_price,  # 피봇 가격 저장
        )
        
        # 상태 업데이트
        def _enter(s: DailyState) -> None:
            s.active = pos
        
        self._state.update(today_str, _enter)
        
        logger.info(
            "[PIVOT_GATE] 진입 완료 %s @ 선물=%.2f (피봇지수=%.2f, size=%.2f, capital=%.0f, ratio=%.4f)",
            signal, entry_price, pivot_price, size, entry_price * size, self._futures_index_ratio,
        )
        
        # 텔레그램 알림 (별도 스레드로 분리)
        if self._notifier:
            threading.Thread(
                target=self._send_trade_entry_alert,
                args=(signal, entry_price, size, pivot_price),
                daemon=True,
            ).start()
    
    def _execute_close(
        self,
        price: float,
        now: datetime,
        today_str: str,
        reason: CloseReason,
    ) -> None:
        """청산 실행.
        
        Note:
            호출자가 이미 락을 보유하고 있어야 합니다.
        """
        state = self._state.get_state()
        
        if not state.has_position or state.active is None:
            return
        
        pos = state.active
        
        # 슬리피지 적용
        slippage = self._cfg.slippage_ticks * self._cfg.tick_size
        if pos.side == PositionSide.LONG:
            exit_price = price - slippage
            profit = (exit_price - pos.record.entry_price) * pos.record.position_size
        else:  # SHORT
            exit_price = price + slippage
            profit = (pos.record.entry_price - exit_price) * pos.record.position_size
        
        # 수수료
        commission = (pos.record.entry_price + exit_price) * pos.record.position_size * self._cfg.commission_rate
        profit -= commission
        
        # 보유 기간
        bars_held = int((now - pos.record.entry_time).total_seconds() / 60)
        
        # 기존 거래 기록 업데이트
        pos.record.close_price = exit_price
        pos.record.close_time = now
        pos.record.close_reason = reason
        
        # 수익 계산
        profit_pct = profit / pos.record.capital_used * 100 if pos.record.capital_used > 0 else 0.0
        
        # 상태 업데이트
        def _close(s: DailyState) -> None:
            s.active = None
            s.trade_log.append(pos.record)
            s.total_pnl_pct += profit_pct
        
        self._state.update(today_str, _close)
        
        logger.info(
            "[PIVOT_GATE] 청산 완료 %s @ %.2f (reason=%s, profit=%.2fpt, bars=%d)",
            pos.side.value, exit_price, reason.value, profit_pct, bars_held,
        )
        
        # 텔레그램 알림 (별도 스레드로 분리)
        if self._notifier:
            threading.Thread(
                target=self._send_trade_exit_alert,
                args=(pos.side.value, exit_price, profit, reason.value),
                daemon=True,
            ).start()
        
        # 이력 저장 (별도 스레드에서 수행하여 락 블로킹 방지)
        if self._cfg.history_save_enabled:
            # 락 보유 중에 to_dict()로 직렬화하여 데이터 오염 방지
            record_dict = pos.record.to_dict()
            # 별도 스레드에서 저장 (락 보유 시간 최소화)
            threading.Thread(
                target=self._save_record_dict,
                args=(record_dict, today_str),
                daemon=True,
            ).start()
    
    def _send_telegram_notification(self, msg: str) -> None:
        """텔레그램 알림 전송 (별도 스레드에서 호출).
        
        Args:
            msg: 전송할 메시지
        """
        try:
            self._notifier.send_text(msg, parse_mode="Markdown")
        except Exception as e:
            logger.warning("[PIVOT_GATE] 텔레그램 알림 실패: %s", e)
    
    def _send_trade_exit_alert(self, action: str, price: float, profit: float, reason: str) -> None:
        """청산 알림 전송 (별도 스레드에서 호출).
        
        Args:
            action: 포지션 사이드 ("LONG" 또는 "SHORT")
            price: 청산 가격
            profit: 수익
            reason: 청산 사유
        """
        try:
            self._notifier.send_trade_exit_alert(
                action=action,
                price=price,
                profit=profit,
                reason=reason,
            )
        except Exception as e:
            logger.warning("[PIVOT_GATE] 텔레그램 알림 실패: %s", e)
    
    def _send_trade_entry_alert(self, signal: str, entry_price: float, size: float, pivot_price: float) -> None:
        """진입 알림 전송 (별도 스레드에서 호출).
        
        Args:
            signal: "BUY" 또는 "SELL"
            entry_price: 진입 가격
            size: 포지션 사이즈
            pivot_price: 피봇 가격
        """
        try:
            self._notifier.send_trade_entry_alert(
                action=signal,
                price=entry_price,
                size=size,
                pivot_price=pivot_price,
            )
        except Exception as e:
            logger.warning("[PIVOT_GATE] 텔레그램 알림 실패: %s", e)
    
    def _is_after_force_close(self, now: datetime) -> bool:
        """현재 시각이 force_close_time 이후인지 반환."""
        if self._force_close_time_cache is None:
            return False
        return now.time() >= self._force_close_time_cache
    
    def _save_record_dict(self, record_dict: dict, today_str: str) -> None:
        """거래 기록 저장 (딕셔너리 직렬화 버전).
        
        Args:
            record_dict: TradeRecord.to_dict()로 직렬화된 딕셔너리
            today_str: 오늘 날짜 문자열
        """
        import json
        from pathlib import Path
        
        try:
            # 1. 기본 history_dir에 저장
            history_dir = Path(self._cfg.history_dir)
            history_file = history_dir / f"{today_str}.jsonl"
            
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
            
            logger.debug("[PIVOT_GATE] 거래 기록 저장: %s", history_file)
            
            # 2. logs/trades 디렉토리에도 저장 (GUI 뷰어용)
            logs_dir = Path("logs/trades")
            logs_dir.mkdir(parents=True, exist_ok=True)
            logs_file = logs_dir / f"trades_{today_str}.jsonl"
            
            with open(logs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
            
            logger.debug("[PIVOT_GATE] 거래 기록 저장 (GUI용): %s", logs_file)
        except Exception as e:
            logger.warning("[PIVOT_GATE] 거래 기록 저장 실패: %s", e)
    
    def get_daily_summary_dict(self) -> dict:
        """오늘의 거래 요약 dict 반환."""
        with self._lock:
            state = self._state.get_state()
            return state.summary_dict()
    
    def save_daily_summary(self, file_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """오늘의 거래 요약을 JSON 파일로 저장.
        
        Args:
            file_path: 저장할 파일 경로 (None이면 기본 경로 사용)
        
        Returns:
            (저장 성공 여부, 텔레그램 전송용 요약 텍스트)
        """
        if not self._cfg.history_save_enabled:
            logger.debug("[PIVOT_GATE] history_save_enabled=False, 요약 저장 건너뜀")
            return False, None
        
        try:
            summary = self.get_daily_summary_dict()
            sim_metrics = self.get_simulation_metrics()
            
            # 요약 데이터 구성
            summary_data = {
                "date": summary.get("date"),
                "daily_summary": summary,
                "simulation_metrics": {
                    "total_pivots": sim_metrics.get("total_pivots"),
                    "total_trades": sim_metrics.get("total_trades"),
                    "win_trades": sim_metrics.get("win_trades"),
                    "loss_trades": sim_metrics.get("loss_trades"),
                    "stop_loss_trades": sim_metrics.get("stop_loss_trades"),
                    "win_rate": sim_metrics.get("win_rate"),
                    "total_profit": sim_metrics.get("total_profit"),
                    "avg_profit": sim_metrics.get("avg_profit"),
                    "sharpe_ratio": sim_metrics.get("sharpe_ratio"),
                },
                "config": {
                    "initial_capital": self._cfg.initial_capital,
                    "tick_size": self._cfg.tick_size,
                    "commission_rate": self._cfg.commission_rate,
                    "sizing_method": self._cfg.sizing.method.value,
                },
                "saved_at": datetime.now().isoformat(),
            }
            
            # 파일 경로 결정
            if file_path is None:
                from pathlib import Path
                history_dir = Path(self._cfg.history_dir)
                today_str = datetime.now().strftime("%Y-%m-%d")
                file_path = str(history_dir / f"{today_str}_summary.json")
            
            # JSON 파일로 저장
            import json
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info("[PIVOT_GATE] 일일 요약 저장 완료: %s", file_path)
            
            # 텔레그램 전송용 요약 텍스트 생성
            telegram_text = self._format_summary_for_telegram(summary, sim_metrics)
            
            return True, telegram_text
        except Exception as e:
            logger.warning("[PIVOT_GATE] 일일 요약 저장 실패: %s", e)
            return False, None
    
    def _format_summary_for_telegram(self, summary: dict, sim_metrics: dict) -> str:
        """텔레그램 전송용 요약 텍스트 포맷팅.
        
        Args:
            summary: 일일 요약 딕셔너리
            sim_metrics: 시뮬레이션 지표 딕셔너리
        
        Returns:
            포맷팅된 텔레그램 텍스트
        """
        try:
            total_pnl_pct = summary.get("total_pnl_pct", 0.0)
            total_pnl_pt = summary.get("total_pnl_pt", 0.0)
            daily_count = summary.get("daily_count", 0)
            
            total_trades = sim_metrics.get("total_trades", 0)
            win_trades = sim_metrics.get("win_trades", 0)
            loss_trades = sim_metrics.get("loss_trades", 0)
            win_rate = sim_metrics.get("win_rate", 0.0)
            total_profit = sim_metrics.get("total_profit", 0.0)
            sharpe_ratio = sim_metrics.get("sharpe_ratio", 0.0)
            
            # 이모지 추가
            pnl_emoji = "📈" if total_pnl_pct >= 0 else "📉"
            
            text = f"""📊 <b>피봇 기반 매매 일일 요약</b>

📅 날짜: {summary.get('date', 'N/A')}

{pnl_emoji} <b>수익률:</b> {total_pnl_pct:+.2f}%
💰 <b>수익(Pt):</b> {total_pnl_pt:+.2f}
🔄 <b>거래 횟수:</b> {daily_count}

📈 <b>총 거래:</b> {total_trades}
✅ <b>승리:</b> {win_trades}
❌ <b>패배:</b> {loss_trades}
📊 <b>승률:</b> {win_rate:.1f}%

💵 <b>총 수익:</b> {total_profit:,.0f}원
📐 <b>샤프 비율:</b> {sharpe_ratio:.2f}
"""
            return text
        except Exception as e:
            logger.warning("[PIVOT_GATE] 텔레그램 요약 포맷팅 실패: %s", e)
            return "📊 피봇 기반 매매 요약 생성 실패"
    
    def shutdown(self) -> Optional[str]:
        """프로그램 종료 시 정리 작업.
        
        Returns:
            텔레그램 전송용 요약 텍스트 (저장 실패 시 None)
        """
        logger.info("[PIVOT_GATE] 종료 정리 시작")
        
        # 일일 요약 저장 및 텔레그램 텍스트 생성
        success, telegram_text = self.save_daily_summary()
        
        logger.info("[PIVOT_GATE] 종료 정리 완료")
        return telegram_text if success else None
