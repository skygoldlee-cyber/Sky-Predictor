# 실시간 거래 이벤트 로깅 가이드

**버전:** 2026-04-25 (대시보드 구현 완료)  
**파일:** `docs/TRADE_LOGGING_GUIDE.md`

---

## 개요

실시간 매매 진입/청산 이벤트를 기록하고, 백테스팅이 이를 기반으로 실행할 수 있도록 지원합니다. 포지션 상태 추적, 동적 리스크 관리, OHLCV 데이터 통합, 리스크 메트릭 로깅, 알림 통합, 성능 분석, 청산 사유 세분화, 데이터베이스 저장, 에러 핸들링 강화, 실시간 모니터링 대시보드 등 완전한 실거래 시뮬레이션을 제공합니다.

## 구조

```
┌─────────────────────────────────────────────────────────────┐
│                  실시간 거래 이벤트 로깅                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  실시간 거래                                                 │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────┐                                          │
│  │ TradeLogger  │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ trades_YYYY- │                                          │
│  │ MM-DD.jsonl  │                                          │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ 백테스팅     │                                          │
│  │ (로그 기반)   │                                          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 설정

### config.json

```json
{
  "adaptive_indicator": {
    "trade_logging": {
      "enabled": true,
      "log_dir": "logs/trades"
    }
  }
}
```

### 설정 항목

| 항목 | 설명 | 기본값 |
|---|---|---|
| enabled | 로거 활성화 여부 | false |
| log_dir | 로그 디렉토리 | logs/trades |

## 사용법

### 1. 실시간 거래 이벤트 기록

#### 진입 이벤트

```python
from prediction.trade_logger import get_trade_logger

logger = get_trade_logger()

logger.log_entry(
    action="BUY",
    price=325.50,
    size=1.0,
    confidence="HIGH",
    signal_reason="zigzag_pivot_low,ADX_strong",
    stop_loss=324.00,
    take_profit=327.50,
    atr=1.5
)
```

#### 청산 이벤트

```python
logger.log_exit(
    action="BUY",
    price=327.50,
    size=1.0,
    confidence="HIGH",
    reason="take_profit",
    signal_reason=""
)
```

### 2. 로그 파일 형식

#### 파일명

```
trades_YYYY-MM-DD.jsonl
```

예: `trades_2026-04-25.jsonl`

#### 로그 엔트리

```json
{
  "event_type": "ENTRY",
  "timestamp": "2026-04-25T10:30:00.123456",
  "action": "BUY",
  "price": 325.50,
  "size": 1.0,
  "confidence": "HIGH",
  "reason": null,
  "signal_reason": "zigzag_pivot_low,ADX_strong",
  "stop_loss": 324.00,
  "take_profit": 327.50,
  "atr": 1.5
}
```

### 3. 로그 기반 백테스팅

#### 백테스팅 실행

```python
from prediction.backtest_pivot_signals import PivotSignalBacktester
from pathlib import Path

backtester = PivotSignalBacktester()
log_file = Path("logs/trades/trades_2026-04-25.jsonl")

results = backtester.run_backtest_from_logs(log_file)
backtester.print_results(results)
```

#### 백테스팅 결과

```
============================================================
백테스팅 결과
============================================================
총 거래: 5
승리: 3 | 패배: 2
승률: 60.00%
총 수익: 150,000원 (1.50%)
평균 수익/거래: 30,000원 (0.30%)
최대 낙폭: -2.0% (-200,000원)
Sharpe Ratio: 1.2
```

### 3. 포지션 상태 추적

#### PositionTracker 사용

```python
from prediction.trade_logger import get_position_tracker

tracker = get_position_tracker()

# 포지션 생성
position_id = tracker.create_position(
    action="BUY",
    entry_price=325.50,
    size=1.0,
    confidence="HIGH",
    signal_reason="zigzag_pivot_low",
    stop_loss=324.00,
    take_profit=327.50,
    atr=1.5
)

# 포지션 업데이트 (트레일링 스탑)
new_stop = tracker.update_position(
    position_id=position_id,
    current_price=326.50,
    atr=1.5,
    trailing_stop_multiplier=1.5
)

# 청산 여부 판단
should_exit, reason = tracker.should_exit(position_id, 327.50)
if should_exit:
    tracker.close_position(position_id, 327.50, reason)
```

#### 포지션 상태 조회

```python
# 활성 포지션 리스트
active_positions = tracker.get_active_positions()

# 특정 포지션 조회
position = tracker.get_position(position_id)

# 부분 청산
tracker.add_partial_exit(position_id, 327.00, 0.5, "take_profit")
```

### 4. OHLCV 데이터 통합 백테스팅

#### 로그 + OHLCV 기반 백테스팅

```python
from prediction.backtest_pivot_signals import PivotSignalBacktester
from pathlib import Path
import pandas as pd

backtester = PivotSignalBacktester()
log_file = Path("logs/trades/trades_2026-04-25.jsonl")

# OHLCV 데이터 로드
df = pd.read_csv("data/kp200_ohlcv_2026-04-25.csv", index_col="timestamp", parse_dates=True)

# 로그 + OHLCV 기반 백테스팅
results = backtester.run_backtest_from_logs_with_ohlcv(log_file, df)
backtester.print_results(results)
```

#### OHLCV 통합의 장점

1. **실제 가격 검증**: 로그의 청산 가격 vs OHLCV의 실제 청산 시점 가격
2. **보유 기간 정확도**: 실제 봉 수 기반 보유 기간 계산
3. **변동성 반영**: 해당 기간의 실제 가격 변동 반영
4. **슬리피지 정확도**: 실제 시장 상황 기반 슬리피지 계산

### 5. 리스크 메트릭 로깅

#### 리스크 메트릭 이벤트 구조

```python
@dataclass
class RiskMetricsEvent:
    timestamp: datetime
    position_id: str
    current_price: float
    atr: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    distance_to_stop: float
    distance_to_take_profit: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    risk_reward_ratio: float
    position_size_pct: float
    confidence: str
```

#### 리스크 메트릭 로깅

```python
from prediction.trade_logger import get_trade_logger, get_position_tracker

logger = get_trade_logger()
tracker = get_position_tracker()

# 리스크 메트릭 계산
metrics = tracker.calculate_risk_metrics(
    position_id=position_id,
    current_price=326.50,
    atr=1.5,
    position_size_pct=0.95,
    capital=10000000
)

# 리스크 메트릭 로그 기록
if metrics:
    logger.log_risk_metrics(
        position_id=position_id,
        current_price=326.50,
        atr=1.5,
        unrealized_pnl=metrics["unrealized_pnl"],
        unrealized_pnl_pct=metrics["unrealized_pnl_pct"],
        distance_to_stop=metrics["distance_to_stop"],
        distance_to_take_profit=metrics["distance_to_take_profit"],
        max_favorable_excursion=metrics["max_favorable_excursion"],
        max_adverse_excursion=metrics["max_adverse_excursion"],
        risk_reward_ratio=metrics["risk_reward_ratio"],
        position_size_pct=metrics["position_size_pct"],
        confidence=metrics["confidence"]
    )
```

#### 리스크 메트릭 로그 파일

```
logs/trades/risk_metrics_2026-04-25.jsonl
```

### 6. 알림 통합

#### 알림 설정

```python
from prediction.trade_notifier import TradeNotifier, NotificationConfig, get_trade_notifier

config = NotificationConfig(
    enabled=True,
    telegram_enabled=True,
    telegram_bot_token="YOUR_BOT_TOKEN",
    telegram_chat_id="YOUR_CHAT_ID",
    notify_on_entry=True,
    notify_on_exit=True,
    notify_on_risk_alert=True,
    risk_alert_threshold_pct=2.0
)

notifier = get_trade_notifier(config)
```

#### 진입 알림

```python
event = {
    "action": "BUY",
    "price": 325.50,
    "size": 1.0,
    "confidence": "HIGH",
    "signal_reason": "zigzag_pivot_low",
    "stop_loss": 324.00,
    "take_profit": 327.50
}

notifier.notify_entry(event)
```

#### 청산 알림

```python
event = {
    "action": "BUY",
    "price": 327.50,
    "reason": "take_profit"
}

pnl = 20000  # 20,000원 수익
notifier.notify_exit(event, pnl)
```

#### 리스크 알림

```python
notifier.notify_risk_alert(
    position_id="pos_1",
    current_price=324.50,
    unrealized_pnl_pct=-1.5,
    reason="손절 근접"
)
```

### 7. 실시간 거래와 백테스팅 로직 일치화

#### BacktestPositionManager 사용

```python
from prediction.backtest_pivot_signals import BacktestPositionManager, BacktestConfig

config = BacktestConfig(
    initial_capital=10000000,
    trailing_stop_atr_multiplier=1.5
)

pos_mgr = BacktestPositionManager(config)

# 포지션 생성 (실제 PositionTracker와 동일한 로직)
position_id = pos_mgr.create_position(
    action="BUY",
    entry_price=325.50,
    size=1.0,
    confidence="HIGH",
    signal_reason="zigzag_pivot_low",
    stop_loss=324.00,
    take_profit=327.50,
    atr=1.5,
    entry_time=datetime.now()
)

# 포지션 업데이트 (트레일링 스탑)
new_stop = pos_mgr.update_position(position_id, 326.50, 1.5)

# 청산 여부 판단
should_exit, reason = pos_mgr.should_exit(position_id, 327.50)
if should_exit:
    pos_mgr.close_position(position_id, 327.50, reason)
```

#### 로직 일치화의 장점

1. **시뮬레이션 정확도**: 실제 포지션 관리 로직과 동일
2. **트레일링 스탑**: 실제와 동일한 트레일링 스탑 로직
3. **청산 판단**: 실제와 동일한 손절/이익실현 판단
4. **호재/악재 추적**: 최대 호재/악재 기록

### 8. 일일 요약

#### 요약 조회

```python
from prediction.trade_logger import get_trade_logger

logger = get_trade_logger()
summary = logger.get_daily_summary()

print(summary)
```

#### 요약 출력

```python
{
  "total_entries": 10,
  "total_exits": 8,
  "completed_trades": 8,
  "win_trades": 5,
  "loss_trades": 3,
  "win_rate": 0.625,
  "total_profit": 120000.0,
  "avg_profit_per_trade": 15000.0
}
```

### 9. 성능 분석 리포트

#### PerformanceAnalyzer 사용

```python
from prediction.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
report = analyzer.generate_report(trades)
analyzer.print_report(report)
```

#### 분석 항목

- **기본 통계**: 총 거래, 승률, 수익, 보유 기간
- **시간대별 분석**: 오전/오후/장마감별 성과
- **요일별 분석**: 월~금별 성과
- **신뢰도별 분석**: HIGH/MEDIUM/LOW별 성과
- **청산 사유별 분석**: stop_loss/take_profit/signal_reversal별 성과
- **보유 기간 분석**: 단기/중기/장기별 성과
- **리스크 메트릭**: MDD, Sharpe Ratio, Sortino Ratio, Profit Factor
- **호재/악재 분석**: 최대 호재/악재 분석

### 10. 청산 사유 세분화

#### 세분화된 청산 사유

```python
from prediction.trade_logger import ExitReason

# stop_loss 관련
ExitReason.STOP_LOSS_INITIAL  # 초기 손절
ExitReason.STOP_LOSS_TRAILING  # 트레일링 스탑
ExitReason.STOP_LOSS_ATR_SPIKE  # ATR 급증으로 인한 손절

# take_profit 관련
ExitReason.TAKE_PROFIT_INITIAL  # 초기 이익실현
ExitReason.TAKE_PROFIT_PARTIAL  # 부분 이익실현
ExitReason.TAKE_PROFIT_TRAILING  # 트레일링 이익실현

# 시장 관련
ExitReason.MARKET_CLOSE  # 장 마감 강제 청산
ExitReason.LIQUIDITY_ISSUE  # 유동성 부족
ExitReason.VOLATILITY_SPIKE  # 변동성 급증

# 시스템 관련
ExitReason.MANUAL_OVERRIDE  # 수동 개입
ExitReason.SYSTEM_ERROR  # 시스템 오류
ExitReason.TIMEOUT  # 타임아웃

# 기존 호환
ExitReason.SIGNAL_REVERSAL  # 반대 신호
ExitReason.STOP_LOSS  # 손절 (간단)
ExitReason.TAKE_PROFIT  # 이익실현 (간단)
```

#### 사용 예시

```python
logger.log_exit(
    action="BUY",
    price=324.00,
    size=1.0,
    confidence="HIGH",
    reason=ExitReason.STOP_LOSS_TRAILING
)
```

### 11. 데이터베이스 저장

#### TradeDatabase 사용

```python
from prediction.trade_database import TradeDatabase
from datetime import datetime

db = TradeDatabase("trades.db")

# 이벤트 저장
db.save_event(event.to_dict())

# 포지션 저장
db.save_position(position.__dict__)

# 리스크 메트릭 저장
db.save_risk_metrics(metrics)

# 기간별 거래 조회
start_date = datetime(2026, 4, 1)
end_date = datetime(2026, 4, 30)
trades = db.query_trades(start_date, end_date)

# 활성 포지션 조회
active_positions = db.query_active_positions()

# DB 기반 성능 분석
performance = db.analyze_performance(start_date, end_date)
```

#### DB 테이블 구조

- **events**: 거래 이벤트 (ENTRY, EXIT, TRAILING_STOP, ATR_SNAPSHOT)
- **positions**: 포지션 상태
- **risk_metrics**: 리스크 메트릭

#### 인덱스

- events: timestamp, position_id
- positions: position_id
- risk_metrics: timestamp, position_id

### 12. 에러 핸들링 강화

#### 에러 핸들링 기능

```python
from prediction.trade_logger import get_trade_logger
from prediction.trade_notifier import get_trade_notifier

logger = get_trade_logger()
notifier = get_trade_notifier()

# 알림 인스턴스 설정
logger.set_notifier(notifier)

# 에러 발생 시 자동 처리
# 1. 최대 3회 재시도
# 2. 백업 경로 시도
# 3. 에러 로그 별도 기록
# 4. 실패 시 알림 전송
logger.log_event(event)
```

#### 에러 핸들링 로직

1. **재시도**: 최대 3회 재시도 (0.1초 대기)
2. **백업 경로**: 주 파일 실패 시 백업 디렉토리 시도
3. **에러 로그**: 별도 에러 파일에 기록 (`errors_YYYY-MM-DD.log`)
4. **알림**: 최종 실패 시 알림 전송

#### 백업 파일 구조

```
logs/trades/
├── trades_2026-04-25.jsonl          # 주 로그 파일
├── backup/                           # 백업 디렉토리
│   ├── backup_20260425_103012.jsonl  # 백업 로그
│   └── ...
└── errors_2026-04-25.log            # 에러 로그
```

### 13. 실시간 모니터링 대시보드

#### TradeDashboard 사용 (CLI)

```python
from prediction.trade_dashboard import TradeDashboard

dashboard = TradeDashboard()
dashboard.print_summary()
```

#### TradeDashboard 사용 (Web API)

```python
from prediction.trade_dashboard import TradeDashboard

dashboard = TradeDashboard()
dashboard.run_api(host="0.0.0.0", port=8000)
```

#### API 엔드포인트

- `GET /api/summary`: 대시보드 요약
- `GET /api/active-positions`: 활성 포지션
- `GET /api/daily-pnl?days=7`: 일일 손익
- `GET /api/risk-metrics`: 리스크 메트릭
- `GET /api/recent-trades?limit=20`: 최근 거래

#### 대시보드 기능

- **활성 포지션**: 실시간 포지션 상태 표시
- **일일 손익**: 기간별 손익 그래프
- **리스크 메트릭**: 총 노출, 손절/이익실현 거리
- **최근 거래**: 최근 거래 이력

#### 설치 필요

```bash
pip install fastapi uvicorn
```

## TradeEvent 데이터 구조

### 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| event_type | str | 이벤트 타입 ("ENTRY", "EXIT", "TRAILING_STOP", "ATR_SNAPSHOT") |
| timestamp | datetime | 타임스탬프 |
| action | str | 액션 ("BUY" or "SELL") |
| price | float | 가격 |
| size | float | 사이즈 |
| confidence | str | 신뢰도 ("HIGH", "MEDIUM", "LOW") |
| reason | Optional[str] | 청산 사유 (EXIT만) |
| signal_reason | str | 신호 이유 |
| stop_loss | Optional[float] | 손절 가격 |
| take_profit | Optional[float] | 이익실현 가격 |
| atr | Optional[float] | ATR 값 |
| position_id | Optional[str] | 포지션 ID |
| trailing_stops | List[dict] | 트레일링 스탑 기록 |
| atr_snapshots | List[float] | ATR 스냅샷 |
| partial_exits | List[dict] | 부분 청산 기록 |

### 청산 사유 (reason)

| 사유 | 설명 |
|---|---|
| stop_loss | 손절 청산 |
| take_profit | 이익실현 청산 |
| signal_reversal | 반대 신호 청산 |

## 백테스팅 로직

### 진입/청산 매칭

1. ENTRY 이벤트와 EXIT 이벤트를 타임스탬프 순으로 정렬
2. 각 ENTRY에 대해 타임스탬프가 큰 EXIT를 매칭
3. 가장 빠른 EXIT를 사용하여 거래 완료

### 수익 계산

```python
# BUY 포지션
profit = (exit_price - entry_price) * size

# SELL 포지션
profit = (entry_price - exit_price) * size

# 수수료 차감
commission = (entry_price + exit_price) * size * commission_rate
profit -= commission
```

### 결과 지표

| 지표 | 설명 |
|---|---|
| total_trades | 총 거래 수 |
| win_trades | 승리 거래 수 |
| loss_trades | 패배 거래 수 |
| win_rate | 승률 |
| total_profit | 총 수익 |
| total_profit_pct | 총 수익률 |
| avg_profit_per_trade | 평균 수익/거래 |
| max_drawdown | 최대 낙폭 |
| sharpe_ratio | Sharpe Ratio |

## Pipeline 통합

### PredictionPipeline에서 자동 로깅

```python
# config.json에서 enabled=true 시 자동 로깅
self._trade_logger = get_trade_logger()

# 진입 시
if self._trade_logger and signal in ("BUY", "SELL"):
    self._trade_logger.log_entry(
        action=signal,
        price=current_price,
        size=size,
        confidence=confidence,
        signal_reason=reason,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr
    )

# 청산 시
if self._trade_logger and should_exit:
    self._trade_logger.log_exit(
        action=position_action,
        price=exit_price,
        size=size,
        confidence=confidence,
        reason=exit_reason
    )
```

## 로그 관리

### 로그 파일 위치

```
logs/trades/
├── trades_2026-04-25.jsonl
├── trades_2026-04-26.jsonl
├── trades_2026-04-27.jsonl
└── ...
```

### 로그 파일 회전

- 매일 새로운 파일 생성
- 파일명: `trades_YYYY-MM-DD.jsonl`
- 자동 삭제 기능 없음 (수동 관리 필요)

### 로그 파일 크기

- 1거래당 약 300 bytes
- 100거래/일 → 약 30 KB/일
- 1년 → 약 11 MB

## 예시

### 실제 거래 로그 예시

```jsonl
{"event_type":"ENTRY","timestamp":"2026-04-25T10:30:00.123456","action":"BUY","price":325.50,"size":1.0,"confidence":"HIGH","reason":null,"signal_reason":"zigzag_pivot_low,ADX_strong","stop_loss":324.00,"take_profit":327.50,"atr":1.5}
{"event_type":"EXIT","timestamp":"2026-04-25T11:45:00.789012","action":"BUY","price":327.50,"size":1.0,"confidence":"HIGH","reason":"take_profit","signal_reason":""}
{"event_type":"ENTRY","timestamp":"2026-04-25T13:00:00.345678","action":"SELL","price":330.00,"size":1.0,"confidence":"MEDIUM","reason":null,"signal_reason":"zigzag_pivot_high","stop_loss":331.50,"take_profit":327.00,"atr":1.5}
{"event_type":"EXIT","timestamp":"2026-04-25T14:15:00.901234","action":"SELL","price":327.00,"size":1.0,"confidence":"MEDIUM","reason":"take_profit","signal_reason":""}
```

### 백테스팅 결과 예시

```
============================================================
백테스팅 결과
============================================================
총 거래: 2
승리: 2 | 패배: 0
승률: 100.00%
총 수익: 200,000원 (2.00%)
평균 수익/거래: 100,000원 (1.00%)
최대 낙폭: 0.0% (0원)
Sharpe Ratio: 0.0
```

## 주의사항

1. **타임스탬프**: UTC 또는 로컬 시간 일관성 유지 필요
2. **진입/청산 쌍**: 매칭되지 않는 진입은 백테스팅에서 제외
3. **수수료**: 백테스팅 설정과 실제 수수료 일치 필요
4. **슬리피지**: 백테스팅 설정과 실제 슬리피지 일치 필요
5. **로그 보관**: 로그 파일 주기적 백업 권장

---

**문서 버전**: 2026-04-25 (대시보드 구현 완료)  
**마지막 업데이트**: 실시간 모니터링 대시보드 (CLI + Web API) 구현 완료
