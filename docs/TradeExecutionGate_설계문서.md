# TradeExecutionGate — 장중 진입/청산 로직 설계 및 구현 문서

> **Transformer Prediction Pipeline — Phase 2 + Phase 3 구현 완료**
> 최초 작성: 2026-03-22 | 최종 갱신: 2026-04-26 (Trailing Stop-loss 추가)

---

## 목차

1. 개요
2. 아키텍처
3. trade_state.py — 상태 데이터 모델
4. trade_gate.py — 진입/청산 로직
5. trade_history_viewer.py — 거래 이력 분석 도구
6. telegram_notifier.py 수정 내용
7. 텔레그램 메시지 포맷
8. 활성화 방법
9. 단위 테스트
10. Phase 3 구현 상세

---

## 1. 개요

기존 5분 주기 방향예측 파이프라인(Transformer + LLM 앙상블)에 **장중 진입/청산 판단 레이어**를 추가한다.
예측 신호(BUY/SELL/HOLD)를 직접 매매로 연결하지 않고, 독립 게이트를 모두 통과한 신호만 진입으로 처리한다.

### 1.1 핵심 설계 원칙

- **하루 최대 3회 진입** — 오전·중반·후반 시간대(슬롯 A/B/C) 각 1회씩 배분
- **기존 예측 흐름에 영향 없음** — enabled: false(기본값) 시 완전 비활성, 기존 텔레그램 전송 그대로 동작
- **독립 감시 루프** — 청산 조건(목표/손절/강제)은 30초 주기 별도 스레드에서 점검
- **단일 진입/청산 원칙** — 포지션 보유 중 추가 진입 불가, 청산 후 재진입 가능
- **런타임 토글** — 텔레그램 명령으로 재시작 없이 활성화/비활성화 가능 (Phase 3)

### 1.2 관련 파일

| 파일 | 구분 | 역할 |
|---|---|---|
| trade_state.py | 신규 | 상태 데이터 모델 (Enum, dataclass) |
| trade_gate.py | 신규 | 진입/청산 판단 로직 + Phase 3 확장 |
| trade_history_viewer.py | 신규 (Phase 3) | JSONL 이력 분석 CLI 도구 |
| telegram_notifier.py | 수정 | Bridge에 게이트 주입 및 루프 연결 |
| config.py | 수정 | AppConfig에 trade_gate 필드 추가 |
| config.json | 수정 | trade_gate 섹션 추가 (Phase 3 키 포함) |
| main.py | 수정 | bridge 생성 후 set_trade_gate_config() 호출 |
| tests/test_trade_gate.py | 신규/확장 | 단위 테스트 70개 |
| tests/test_trade_history.py | 신규 (Phase 3) | 단위 테스트 43개 |

---

## 2. 아키텍처

### 2.1 레이어 구조

기존 파이프라인은 관찰 레이어(예측 신호 생성)이고, TradeExecutionGate는 판단 레이어(진입/청산 결정)다.
두 레이어는 독립적으로 동작한다.

```
PipelineTelegramBridge (기존)
│
│  5분 예측 루프 ──► 신호 변경 필터 ──► 텔레그램 전송(기존)
│                         │
│                         ▼  on_signal() 호출 (매 틱)
│  TradeExecutionGate  (Phase 2~3)
│  │
│  │  게이트1: 신호 연속성
│  │  게이트2: 신호 품질 (confidence / prob / consensus)
│  │  게이트3: 시간대 슬롯 / 일일 횟수
│  │  게이트4: Dealer Gamma 방향 일치 [Phase 3, 옵션]
│  │         │
│  │         ▼  ATM IV 기반 동적 목표/손절 계산 [Phase 3]
│  │         ▼  신뢰도 기반 배수 적용 [Phase 3]
│  │         ▼  진입/청산 텔레그램 알림
│  │         ▼  JSONL 이력 저장 [Phase 3]
│  │
│  텔레그램 명령 핸들러 [Phase 3]
│    /trade_status  /trade_gate on|off
│
_trade_monitor_loop (30초 주기) ──► check_close()
```

### 2.2 스레드 구성

| 스레드명 | 주기 | 역할 |
|---|---|---|
| PipelineBridge | 5분 | 예측 생성 + on_signal() 호출 |
| TradeMonitor | 30초 | check_close() — 목표/손절/강제청산 감시 |
| BleedMonitor / OIMonitor 등 | 기존 | 기존 v4/v5 모니터 — 변경 없음 |

### 2.3 일일 흐름 요약

```
09:05  장 시작 — 슬롯 A 열림
  │
  ├─ BUY 신호 2틱 연속 + HIGH + consensus + [Gamma 게이트 통과]
  │       └─► ATM IV 기반 동적 목표/손절 계산
  │       └─► 슬롯 A 진입 (daily_count=1)
  │               └─► 목표 도달 → 청산 (TARGET_PROFIT)
  │               └─► JSONL 이력 저장
  │
10:30  슬롯 B 열림
  │
13:00  슬롯 C 열림
  │
14:50  강제청산 시각
  │
15:05  일일 결산 텔레그램 자동 전송
```

---

## 3. trade_state.py — 상태 데이터 모델

순수 데이터 모델 파일. 비즈니스 로직은 포함하지 않으며 trade_gate.py가 이 모델을 사용한다.

### 3.1 Enum 타입

| Enum 클래스 | 값 | 설명 |
|---|---|---|
| PositionSide | NONE / LONG / SHORT | 현재 포지션 방향 |
| CloseReason | 목표수익 / 손절 / 반대신호 / 강제청산 | 청산 사유 |
| TradeSlot | A / B / C | 하루 3개 시간대 슬롯 |

### 3.2 trade_id — 마이크로초 기반 고유 거래 ID (Phase 3 추가)

거래 이력의 중복 없는 식별, JSONL 저장, 역직렬화 검증에 사용된다.

```python
def _make_trade_id(dt: Optional[datetime] = None) -> str:
    ts = dt if dt is not None else datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"
# 예: "20260322_100512_834291"
```

TradeRecord.__post_init__에서 trade_id=""이면 entry_time 기준으로 자동 생성된다.
명시적으로 지정한 ID는 덮어씌워지지 않는다.

### 3.3 TradeRecord — 단일 거래 기록

| 필드 | 타입 | 설명 |
|---|---|---|
| trade_id | str | 마이크로초 기반 고유 ID (Phase 3) |
| slot | TradeSlot | 진입한 시간대 슬롯 |
| side | PositionSide | LONG(매수) / SHORT(매도) |
| entry_price / close_price | float | 진입가 / 청산가 |
| entry_time / close_time | datetime | 진입 / 청산 시각 |
| entry_signal / entry_confidence | str | 진입 시 신호 및 신뢰도 |
| entry_prob | float | 진입 시 상승확률 |
| entry_atm_iv | float | ATM 내재변동성 (Phase 3) |
| entry_atm_delta | float | ATM Delta 절대값 (Phase 3) |
| entry_net_gamma | float | net_gamma_proxy (Phase 3) |
| entry_above_vol_trigger | float | Vol Trigger 위/아래 (Phase 3) |
| entry_target_pt | float | 실제 적용된 목표수익 pt (Phase 3) |
| entry_stop_pt | float | 실제 적용된 손절 pt (Phase 3) |
| close_reason | CloseReason | 청산 사유 |
| pnl_pt (프로퍼티) | float | 손익 포인트 |
| hold_minutes (프로퍼티) | float | 보유 시간 (분) |

Phase 3 추가: `from_dict()` 클래스메서드로 JSONL 역직렬화 가능.

### 3.4 DailyState — 주요 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| date | str | "YYYY-MM-DD" |
| used_slots | List[TradeSlot] | 이미 사용한 슬롯 목록 |
| trade_log | List[TradeRecord] | 완결 거래 목록 |
| active | Optional[ActivePosition] | 현재 포지션 |
| consecutive_signal / count | str / int | 연속 신호 추적 |
| **daily_open_price** | **float** | **당일 선물 시가. ATM IV 동적 계산의 기준가. on_signal() 첫 유효 틱에서 기록, 이후 불변. (Phase 3)** |
| **confidence_stats** | **Dict[str, Dict[str, int]]** | **신뢰도별 통계 {"HIGH": {"total": N, "wins": M}, ...} (Phase 3)** |
| **consecutive_losses** | **int** | **연속 손실 횟수 (리스크 관리)** |
| **slot_performance** | **Dict[str, Dict[str, Any]]** | **슬롯별 성과 {"A": {"total": N, "wins": M, "pnl": P}, ...} (리스크 관리)** |

### 3.5 DailyState.summary_dict() — win_rate 필드 추가 (Phase 3)

```python
def summary_dict(self) -> dict:
    wins   = self.wins
    losses = self.losses
    return {
        ...
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0,  # Phase 3 추가
        ...
    }
```

승률 계산 기준: `wins / (wins + losses)`. 무승부(pnl=0)는 분모에서 제외된다.

### 3.6 슬롯 매핑

| 시각 | 슬롯 | 비고 |
|---|---|---|
| ~ 09:04 | None | 장 시작 전 — 진입 금지 |
| 09:05 ~ 10:29 | A | 장 초반 |
| 10:30 ~ 12:59 | B | 장 중반 |
| 13:00 ~ 14:49 | C | 장 후반 |
| 14:50 ~ | None | 강제청산 시각 이후 — 신규 진입 금지 |

모든 경계 시각은 config.json에서 설정 가능하다.

---

## 4. trade_gate.py — 진입/청산 로직

### 4.1 TradeGateConfig

config.json의 trade_gate 섹션을 파싱한다. Phase 3 키가 추가됐다.

#### Phase 2 기존 설정

| 설정 키 | 기본값 | 설명 |
|---|---|---|
| enabled | false | 활성화 스위치 |
| max_daily_trades | 3 | 하루 최대 진입 횟수 |
| min_consecutive_signals | 2 | 연속 신호 필요 횟수 |
| min_confidence | MEDIUM | 최소 신뢰도 |
| min_prob_buy | 0.62 | BUY 최소 상승확률 |
| max_prob_sell | 0.38 | SELL 최대 상승확률 |
| require_consensus | true | 합의 필수 여부 |
| target_profit_pt | 2.0 | 목표 수익 (pt) |
| stop_loss_pt | 1.0 | 손절 (pt) |
| market_open_time | 09:05 | 진입 허용 시작 |
| slot_a_end | 10:30 | 슬롯 A 종료 |
| slot_b_end | 13:00 | 슬롯 B 종료 |
| force_close_time | 14:50 | 강제청산 시각 |
| reverse_close_count | 2 | 반대신호 N회 → 청산 |

#### Phase 3 추가 설정

| 설정 키 | 기본값 | 설명 |
|---|---|---|
| iv_dynamic_enabled | true | ATM IV 기반 동적 목표/손절 활성화 |
| iv_target_mult | 0.5 | target = ATM_IV × daily_open × mult |
| iv_stop_mult | 0.25 | stop = ATM_IV × daily_open × mult |
| iv_target_min | 1.5 | 동적 목표 하한 (pt) |
| iv_target_max | 5.0 | 동적 목표 상한 (pt) |
| iv_stop_min | 0.75 | 동적 손절 하한 (pt) |
| iv_stop_max | 2.5 | 동적 손절 상한 (pt) |
| gamma_gate_enabled | false | Dealer Gamma 방향 게이트 활성화 |
| confidence_dynamic_enabled | true | 신뢰도 기반 동적 목표/손절 활성화 |
| confidence_high_target_mult | 1.5 | HIGH confidence: 목표 × 1.5 (공격적) |
| confidence_high_stop_mult | 0.8 | HIGH confidence: 손절 × 0.8 (타이트) |
| confidence_medium_target_mult | 1.0 | MEDIUM confidence: 목표 × 1.0 (기본) |
| confidence_medium_stop_mult | 1.0 | MEDIUM confidence: 손절 × 1.0 (기본) |
| confidence_low_target_mult | 0.7 | LOW confidence 목표 배수 | 0.5 ~ 0.9 |
| confidence_low_stop_mult | 1.3 | LOW confidence 손절 배수 | 1.0 ~ 1.5 |
| max_consecutive_losses | int | 3 | 최대 연속 손실 횟수 (0=비활성) | 0 ~ 5 |
| max_daily_loss_pt | float | 5.0 | 일일 최대 손실 (pt, 0=비활성) | 0 ~ 10 |
| slot_performance_enabled | bool | false | 슬롯별 성과 기반 할당 활성화 | true/false |
| trailing_stop_enabled | bool | false | Trailing Stop-loss 활성화 | true/false |
| trailing_stop_activation_pt | float | 1.0 | Trailing 시작 이익 (pt) | 0.5 ~ 2.0 |
| trailing_stop_distance_pt | float | 0.5 | Trailing 거리 (pt) | 0.3 ~ 1.0 |
| history_save_enabled | bool | true | 일별 JSONL 저장 활성화 |
| history_dir | str | trade_history | JSONL 저장 디렉토리 |

### 4.2 TradeExecutionGate — 공개 API

```python
gate = TradeExecutionGate(notifier, config)

# Phase 2 API
gate.on_signal(result, current_price=price)    # 예측 루프에서 매 틱 호출
gate.check_close(current_price=price)          # 감시 루프에서 30초 주기 호출
gate.send_daily_summary(force=False)           # 일일 결산 수동 전송
gate.get_daily_summary_dict()                  # 오늘 거래 요약 dict 반환

# Phase 3 추가 API
gate.handle_telegram_command(text)             # /trade_status, /trade_gate on|off 처리
```

### 4.3 진입 게이트 (Phase 2: 3단계 → Phase 3: 4단계)

#### 게이트 1 — 신호 연속성

같은 방향 신호가 min_consecutive_signals 틱 이상 연속 유지되어야 한다. HOLD 또는 방향 전환 시 카운터 리셋.

```
예시 (min_consecutive_signals=2):
  틱1: BUY  → count=1 → 진입 불가
  틱2: BUY  → count=2 → 게이트 1 통과 ✓
  틱3: HOLD → count=0 (리셋)
```

#### 게이트 2 — 신호 품질

confidence / prob / consensus 세 조건을 동시에 검사한다. 하나라도 미달이면 차단.

| 조건 | 차단 기준 |
|---|---|
| confidence | min_confidence 미달 |
| prob (BUY) | prob < min_prob_buy |
| prob (SELL) | prob > max_prob_sell |
| consensus | False (require_consensus=true 시) |

#### 게이트 3 — 시간대 슬롯 / 일일 횟수

| 차단 조건 | 설명 |
|---|---|
| slot = None | 장 외 시간대 |
| daily_count >= max_daily_trades | 일일 한도 초과 |
| slot in used_slots | 슬롯 재사용 시도 |
| current_price <= 0 | 현재가 미확인 |

#### 게이트 4 — Dealer Gamma 방향 (Phase 3, gamma_gate_enabled=True 시 활성)

net_gamma_proxy와 above_vol_trigger 값을 기반으로 진입 방향의 적합성을 판단한다.

| 딜러 상태 | Vol Trigger | BUY | SELL |
|---|---|---|---|
| Long Gamma (net_gamma > 0) | 위 (≥0.5) | 허용 | 차단 |
| Short Gamma (net_gamma < 0) | 아래 (<0.5) | 차단 | 허용 |
| net_gamma = 0 (데이터 없음) | 무관 | 통과 | 통과 |

net_gamma=0이면 데이터 없음으로 간주하여 안전하게 통과시킨다.

### 4.4 Phase 3: ATM IV 기반 동적 목표/손절 + 신뢰도 기반 배수

iv_dynamic_enabled=True이고 ATM IV > 0인 경우, 진입 시점에 목표/손절 폭을 자동 계산한다.
기준가로 **당일 선물 시가(daily_open_price)** 를 사용한다. 진입 시각의 현재가 대신 시가를 기준으로 삼아, 장중 가격 변동에 따른 목표/손절폭 왜곡을 방지한다.

```
공식:
  # 1단계: ATM IV 기반 기본값 계산
  base_target = clamp(ATM_IV × daily_open × iv_target_mult, iv_target_min, iv_target_max)
  base_stop   = clamp(ATM_IV × daily_open × iv_stop_mult,   iv_stop_min,   iv_stop_max)
  
  # 2단계: 신뢰도 기반 배수 적용 (활성화 시)
  if confidence_dynamic_enabled:
      target_mult, stop_mult = get_confidence_multiplier(confidence)
      target = base_target × target_mult
      stop   = base_stop × stop_mult
  else:
      target = base_target
      stop   = base_stop

예시 (ATM_IV=20%, daily_open=820, confidence=HIGH):
  base_target = clamp(0.20 × 820 × 0.5, 1.5, 5.0) = 5.0pt  ← 상한
  base_stop   = clamp(0.20 × 820 × 0.25, 0.75, 2.5) = 2.5pt  ← 상한
  target = 5.0 × 1.5 = 7.5pt  # HIGH confidence: 공격적 목표
  stop   = 2.5 × 0.8 = 2.0pt  # HIGH confidence: 타이트 손절
```

**신뢰도별 배수 기본값**:
- **HIGH**: 목표 × 1.5, 손절 × 0.8 (공격적: 높은 목표, 타이트 손절)
- **MEDIUM**: 목표 × 1.0, 손절 × 1.0 (기본)
- **LOW**: 목표 × 0.7, 손절 × 1.3 (보수적: 낮은 목표, 넓은 손절)

신뢰도 기반 배수는 config.json에서 조절 가능하며, confidence_dynamic_enabled=false로 비활성화할 수 있다.

당일 시가 기록 규칙: `on_signal()`이 호출될 때마다 장 시작 시각(market_open_time, 기본 09:05) 이후 첫 번째 유효 가격(> 0)을 시가로 기록한다. 진입 여부와 무관하게 매 틱 확인하며, 한 번 기록된 시가는 당일 갱신되지 않는다. 날짜가 바뀌면 `DailyState` 리셋과 함께 자동 초기화된다. 시가가 미기록(0.0) 상태에서 진입이 발생하면 현재가로 fallback하여 계산한다.

### 4.5 리스크 관리 (Phase 3 추가)

#### 4.5.1 최대 연속 손실 제한

N회 연속 손실 시 일일 진입을 정지합니다.

```
설정: max_consecutive_losses = 3 (0=비활성)
동작:
  - 청산 시 손실이면 consecutive_losses += 1
  - 청산 시 이익이면 consecutive_losses = 0
  - 진입 시 consecutive_losses >= max_consecutive_losses 이면 차단
```

#### 4.5.2 일일 최대 손실 제한

일일 손실이 특정 pt 초과 시 진입을 정지합니다.

```
설정: max_daily_loss_pt = 5.0 (0=비활성)
동작:
  - 진입 시 total_pnl_pt <= -max_daily_loss_pt 이면 차단
  - 손실 한도 대비 현재 손실 비율을 일일 결산에 표시
```

#### 4.5.3 슬롯별 성과 기반 할당

성과가 좋은 슬롯에 더 많은 기회를 부여합니다.

```
설정: slot_performance_enabled = false
동작:
  - 청산 시 슬롯별 성과 추적 (total, wins, pnl)
  - 진입 시 승률 높은 슬롯 우선 선택
  - 승률 같으면 총 손익 높은 슬롯 우선
  - 데이터 없으면 중립(0.5) 처리
```

#### 4.5.4 Trailing Stop-loss

이익 발생 시 손절선을 동적으로 이동하여 이익을 보호합니다.

```
설정: trailing_stop_enabled = false
동작:
  - 이익이 activation_pt 이상 도달 시 Trailing 시작
  - Trailing Stop 가격 = 현재가 ± distance_pt
  - 가격이 Trailing Stop 가격 이하/이상 도달 시 청산
  - LONG: trailing_stop_price = price - distance_pt (가격 상승 시 상향 이동)
  - SHORT: trailing_stop_price = price + distance_pt (가격 하락 시 하향 이동)

예시 (LONG, activation=1.0pt, distance=0.5pt):
  진입가: 380.0
  가격 381.5 → pnl=1.5pt → trailing_stop=381.0 (활성화)
  가격 382.5 → pnl=2.5pt → trailing_stop=382.0 (업데이트)
  가격 381.8 → pnl=1.8pt → 381.8 ≤ 382.0 → 청산 (이익 1.8pt 보호)
```

### 4.6 Phase 3: JSONL 이력 저장

모든 청산 완료 거래는 _save_history()에서 일별 JSONL 파일에 자동 저장된다.

```
파일 경로: {history_dir}/YYYY-MM-DD.jsonl
형식: 한 줄 = 거래 1건의 JSON (append 모드)
예: trade_history/2026-03-22.jsonl
```

history_save_enabled=False이면 저장하지 않는다. 저장 실패는 WARNING 로그만 출력하고 청산 로직에는 영향을 주지 않는다.

### 4.6 Phase 3: 텔레그램 명령 핸들러

PipelineTelegramBridge의 메시지 수신 루프에서 handle_telegram_command(text)를 호출한다.

| 명령 | 동작 |
|---|---|
| /trade_status | 현재 포지션 정보 + 오늘 거래 건수/승률/손익 전송 |
| /trade_gate on | TradeExecutionGate 런타임 활성화 |
| /trade_gate off | TradeExecutionGate 런타임 비활성화 |

런타임 토글은 _cfg.enabled를 직접 변경하므로 재시작 없이 즉시 적용된다. 이미 같은 상태인 경우 "이미 활성/비활성" 메시지를 전송한다.

### 4.7 청산 조건 요약

| 경로 | 트리거 | CloseReason |
|---|---|---|
| on_signal (반대신호) | 반대 방향 N회 연속 | REVERSE_SIGNAL |
| check_close (목표) | pnl >= entry_target_pt | TARGET_PROFIT |
| check_close (손절) | pnl <= -entry_stop_pt | STOP_LOSS |
| check_close (강제) | 현재 시각 >= force_close_time | FORCE_CLOSE |

강제청산 판단은 _is_after_force_close()로 직접 시각 비교한다. get_trade_slot() is None을 사용하지 않는다 (장 시작 전 오발동 방지).

---

## 5. trade_history_viewer.py — 거래 이력 분석 도구 (Phase 3 신규)

### 5.1 개요

trade_history/ 디렉토리의 JSONL 파일을 읽어 CLI에서 통계를 조회하는 독립 스크립트.
다른 모듈에서 import하여 프로그래밍 방식으로 사용할 수도 있다.

### 5.2 CLI 사용법

```bash
# 최근 7일 요약
python trade_history_viewer.py --days 7

# 특정 날짜 상세 조회
python trade_history_viewer.py --date 2026-03-22

# 전체 통계 + 슬롯/청산사유/IV 분석
python trade_history_viewer.py --all --slot --reason --iv

# JSON 출력 (파이프라인 연동용)
python trade_history_viewer.py --all --json

# 거래 상세 출력
python trade_history_viewer.py --days 3 --verbose
```

### 5.3 _Stats 클래스 — 통계 계산

| 프로퍼티 | 설명 |
|---|---|
| count | 총 거래 수 |
| wins / losses / draws | 승/패/무 |
| win_rate | 승률 = wins / (wins + losses) |
| total_pnl / avg_pnl | 총/평균 손익 |
| avg_win / avg_loss | 평균 수익(승) / 평균 손실(패) |
| max_win / max_loss | 최대 수익 / 최대 손실 |
| profit_factor | 총수익 / |총손실|. 손실=0이면 None |
| avg_hold | 평균 보유 시간 (분) |

### 5.4 그룹 분석

| 함수 | 기준 |
|---|---|
| _group_by_slot | 슬롯 A/B/C |
| _group_by_reason | 청산 사유 |
| _group_by_side | 방향 Long/Short |
| _group_by_confidence | 신뢰도 HIGH/MEDIUM/LOW |
| _group_by_iv | ATM IV 구간 |

_grp_update(groups, key, record): groups[key] 리스트에 record를 append하는 내부 유틸.

ATM IV 구간 레이블: 데이터없음 / IV<10% / 10%≤IV<15% / 15%≤IV<20% / 20%≤IV<25% / IV≥25%

### 5.5 편의 API (import 사용)

```python
from trade_history_viewer import (
    load_history,        # 날짜별 dict 로드
    summary_stats,       # 전체 통계 dict
    daily_pnl_series,    # 일별 손익 시계열
    cumulative_pnl,      # 누적 손익 시계열
    best_worst_days,     # 최고/최저 일자 N개
)

# 최근 30일 통계
stats = summary_stats("trade_history", days=30)
print(f"승률: {stats['win_rate']:.1%}  PF: {stats['profit_factor']}")

# 누적 손익 시계열
for date_str, cum in cumulative_pnl("trade_history", days=10):
    print(f"{date_str}: {cum:+.2f}pt")
```

---

## 6. telegram_notifier.py 수정 내용

### 6.1 PipelineTelegramBridge 수정 항목

| 수정 항목 | 내용 |
|---|---|
| __init__ 필드 추가 | self._trade_gate, self._trade_monitor_thread |
| set_trade_gate_config() 추가 | TradeGateConfig 주입 및 게이트 재초기화 |
| start() 수정 | enabled=True 시 TradeMonitor 스레드 시작 |
| 예측 루프 수정 | send_prediction() 성공 직후 gate.on_signal() 호출 |
| _trade_monitor_loop() 추가 | 30초 주기로 check_close() 호출 |
| 메시지 수신 루프 수정 (Phase 3) | handle_telegram_command() 연결 |

### 6.2 on_signal 연결 위치

send_prediction()이 성공(ok=True)한 직후에만 on_signal()을 호출한다. 전송 실패한 틱은 카운트하지 않아 연속성 계산의 신뢰도를 높인다.

### 6.3 텔레그램 명령 연결 (Phase 3)

`/trade_status`와 `/trade_gate on|off`는 `PipelineTelegramBridge._handle_command()` 안에 직접 구현되어 있다.
`trade_gate.py`에도 `handle_telegram_command()` 메서드가 존재하지만, 현재 Bridge에서는 호출하지 않는다.

> **참고**: `trade_gate.handle_telegram_command()`는 현재 사용되지 않는 메서드다.
> 향후 Bridge에서 게이트 명령을 위임 방식으로 리팩토링할 때 활용할 수 있다.

```python
# telegram_notifier.py — _handle_command() 내 직접 처리 방식
elif cmd == "/trade_status":
    # 현재 포지션 정보 + 오늘 거래 건수/승률/손익 조회
    if self._trade_gate is None or not getattr(self._trade_gate, "enabled", False):
        self._notifier.send_text("⚠️ TradeGate 비활성")
        return
    # ... 상태 조회 및 전송

elif cmd == "/trade_gate":
    # on/off 토글 처리
    # self._trade_gate._cfg.enabled를 갱신하여 즉시 적용
    ...
```

---

## 7. 텔레그램 메시지 포맷

기존 5분 예측 메시지와 완전히 분리된 별도 메시지 타입. HTML parse_mode 사용.

### 7.1 진입 알림 (Phase 3: ATM IV 정보 추가)

```
🟢 진입 알림 (매수)
━━━━━━━━━━━━━━━━━━━
진입가:    382.50
시각:      10:42:15  슬롯 A
신호 근거: BUY HIGH  (prob 0.74)
목표:      385.50  (+3.00pt)  손절: 381.75  (-0.75pt)
ATM IV:   18.5%  Gamma: Long Gamma  VT: ↑
오늘 진입: 1 / 3회
```

ATM IV 정보는 entry_atm_iv > 0일 때만 표시된다.

### 7.2 청산 알림 (Phase 3: 신뢰도 표시 추가)

```
🚪 청산 알림 (매수)
━━━━━━━━━━━━━━━━━━━
청산가:    385.75  (11:05:30)
진입가:    382.50
✅ 손익:   +3.25pt
보유 시간: 23분
진입 신호: BUY HIGH (prob 0.74)
청산 사유: TARGET_PROFIT
```

### 7.3 일일 결산 (15:05 이후 자동 전송, Phase 3: 신뢰도별 승률/리스크 관리 추가)

```
📊 일일 결산  (2026-03-22)
━━━━━━━━━━━━━━━━━━━
총 진입: 2회 / 5회
승: 1  패: 1
📈 손익 합계: +2.25pt

🛡️ 리스크 관리:
연속 손실: 0/3회
손실 한도: 0.00/5.00pt (0%)

📈 신뢰도별 승률:
HIGH: 1/1 (100%)
MEDIUM: 0/1 (0%)
LOW: 0/0 (0%)
```

### 7.4 /trade_status 응답 (Phase 3)

```
📋 거래 현황
━━━━━━━━━━━━━━━━━━━
게이트: 🟢 활성

📌 보유 포지션: 매수(Long)
  진입가: 382.50  (10:42:15)
  슬롯: A
  목표: +3.00pt  손절: -0.75pt
  연속 반대신호: 0/2
  ATM IV: 18.5%  Gamma: Long Gamma

📊 오늘 결과  (2026-03-22)
  진입: 1/3회  승률: —  (0승 0패)
  ➖ 손익: +0.00pt
```

---

## 8. 활성화 방법

### 8.1 config.json 수정 (Phase 3 키 포함)

```json
"trade_gate": {
    "enabled": true,
    "max_daily_trades": 3,
    "min_consecutive_signals": 2,
    "min_confidence": "MEDIUM",
    "min_prob_buy": 0.62,
    "max_prob_sell": 0.38,
    "require_consensus": true,
    "target_profit_pt": 2.0,
    "stop_loss_pt": 1.0,
    "market_open_time": "09:05",
    "slot_a_end": "10:30",
    "slot_b_end": "13:00",
    "force_close_time": "14:50",
    "reverse_close_count": 2,
    "iv_dynamic_enabled": true,
    "iv_target_mult": 0.5,
    "iv_stop_mult": 0.25,
    "iv_target_min": 1.5,
    "iv_target_max": 5.0,
    "iv_stop_min": 0.75,
    "iv_stop_max": 2.5,
    "gamma_gate_enabled": false,
    "history_save_enabled": true,
    "history_dir": "trade_history"
}
```

### 8.2 런타임 토글 (텔레그램 명령, Phase 3)

재시작 없이 즉시 적용된다.

```
/trade_gate on    — 활성화
/trade_gate off   — 비활성화
/trade_status     — 현재 상태 조회
```

### 8.3 주의 사항

set_trade_gate_config()는 bridge.start() 이후에 호출해야 한다. start() 이전에 호출하면 TradeMonitor 스레드가 시작되지 않는다.

---

## 9. 단위 테스트

### 9.1 tests/test_trade_gate.py — 73개 테스트

```bash
python -m pytest tests/test_trade_gate.py -v
```

| 테스트 클래스 | 수 | 검증 항목 |
|---|---|---|
| TestGetTradeSlot | 5 | 시각별 슬롯 매핑, 경계값 |
| TestTradeGateConfig | 4 | 기본값, confidence 비교, from_dict |
| TestTradeStateManager | 2 | 날짜 변경 리셋, 같은 날 유지 |
| TestTradeRecord | 4 | LONG/SHORT pnl 계산, 미청산 상태 |
| TestEntryGate | 9 | 3단계 게이트 각 차단 조건 |
| TestCloseLogic | 5 | 목표/손절/범위 내/강제/반대신호 청산 |
| TestDailySummary | 3 | 결산 dict, 중복 방지, force 재전송 |
| TestDisabledGate | 1 | enabled=False 완전 no-op |
| TestDynamicTargets | 10 | IV 기반 동적 계산, 클램핑, fallback, daily_open 기준가, 첫 틱 시가 기록 |
| TestGammaGate | 5 | 딜러 방향별 허용/차단, 게이트 비활성 |
| TestHistorySave | 3 | JSONL 생성, append, 비활성 시 미생성 |
| TestPhase3Config | 2 | Phase 3 키 from_dict 파싱, 기본값 |
| TestTradeId | 6 | 자동 생성, 포맷, 유일성, 오버라이드, to_dict, from_dict 왕복 |
| TestTelegramCommands | 10 | /trade_status, /trade_gate on/off, 미지정 명령 |
| TestSummaryDictWinRate | 4 | 거래 없음, 전승, 절반, 무승부 제외 |

### 9.2 tests/test_trade_history.py — 43개 테스트

```bash
python -m pytest tests/test_trade_history.py -v
```

| 테스트 클래스 | 수 | 검증 항목 |
|---|---|---|
| TestIterJsonl | 4 | 전체 읽기, 빈 줄 스킵, 파일 없음, 오류 줄 스킵 |
| TestLoaders | 7 | 날짜 로드, 범위 포함/제외, 전체 로드, 비매칭 파일 무시 |
| TestStats | 8 | 빈 레코드, 전승, 전패, 혼합, avg, max, hold, to_dict 키 |
| TestGroupFunctions | 7 | _grp_update, 슬롯/사유/방향/신뢰도 그룹, iv_bucket 레이블 |
| TestConvenienceAPI | 8 | load_history, summary_stats, daily_pnl_series, cumulative_pnl, best_worst_days |
| TestCLIMain | 9 | 데이터 없음, 데이터 있음, JSON 출력, 잘못된 경로, 날짜 형식, --slot, --json --slot, --verbose, --reason, --json --iv |

---

## 10. Phase 3 구현 상세

### 10.1 구현 완료 항목

| 항목 | 파일 | 상태 |
|---|---|---|
| ATM IV 기반 동적 목표/손절 | trade_gate.py | 완료 |
| 신뢰도 기반 동적 목표/손절 배수 | trade_gate.py | 완료 |
| 신뢰도별 통계 추적 | trade_state.py | 완료 |
| 청산 알림 신뢰도 표시 | trade_gate.py | 완료 |
| 일일 결산 신뢰도별 승률 | trade_gate.py | 완료 |
| 최대 연속 손실 제한 | trade_gate.py | 완료 |
| 일일 최대 손실 제한 | trade_gate.py | 완료 |
| 슬롯별 성과 기반 할당 | trade_gate.py | 완료 |
| Trailing Stop-loss | trade_gate.py | 완료 |
| /trade_status 텔레그램 명령 | trade_gate.py | 완료 |
| /trade_gate on/off 명령 | trade_gate.py | 완료 |
| Dealer Gamma 방향 게이트 | trade_gate.py | 완료 (기본 비활성) |
| 거래 이력 JSONL 저장 | trade_gate.py | 완료 |
| trade_id 마이크로초 고유 ID | trade_state.py | 완료 |
| from_dict 역직렬화 | trade_state.py | 완료 |
| win_rate 필드 | trade_state.py | 완료 |
| trade_history_viewer.py | trade_history_viewer.py | 완료 |

### 10.2 IV 기반 동적 목표/손절 파라미터 튜닝 가이드

기본값(iv_target_mult=0.5, iv_stop_mult=0.25)은 ATM IV 기준으로 상한(5.0pt, 2.5pt)이 사실상 항상 작동하도록 설계됐다. daily_open=820 기준으로 상한이 발동하는 IV 임계값은 약 1.2%(target), 0.6%(stop)에 불과하므로, 통상 장중 IV에서는 클램프 상한이 적용된다. 실질 유효 범위는 하한(iv_target_min=1.5pt, iv_stop_min=0.75pt)에서 상한(5.0pt, 2.5pt) 사이이며, 하한이 적용되는 초저변동성 구간(IV<0.4%)에서는 Risk:Reward = 1:2 비율이 유지된다.

조정 권장 시나리오:
- 변동성 확대 국면(IV>25%): iv_target_max를 7.0~8.0으로 확대
- 손절 보수화: iv_stop_mult를 0.3으로 증가
- 보수적 목표: iv_target_mult를 0.3으로 감소

### 10.3 Gamma 게이트 운영 지침

gamma_gate_enabled=false(기본)로 시작하여 충분한 데이터 축적 후 활성화를 권장한다.

활성화 전 확인 사항:
- net_gamma_proxy 데이터가 일관되게 수신되는지 확인 (net_gamma=0인 경우 데이터 없음으로 처리)
- trade_history_viewer.py --all --iv 로 IV 구간별 성과 사전 분석
- 최소 20~30건 이상의 거래 이력이 확보된 후 활성화

### 10.4 JSONL 이력 활용

```bash
# 지난 주 성과 요약
python trade_history_viewer.py --days 7

# 슬롯별 최적 진입 시간대 분석
python trade_history_viewer.py --all --slot

# IV 구간별 성과 — 동적 파라미터 튜닝 근거
python trade_history_viewer.py --all --iv

# 외부 분석 도구 연동용 JSON
python trade_history_viewer.py --all --json > history_report.json
```

---

*TradeExecutionGate 설계 문서 — Transformer Prediction Pipeline Phase 2+3*
*최종 갱신: 2026-04-26*
