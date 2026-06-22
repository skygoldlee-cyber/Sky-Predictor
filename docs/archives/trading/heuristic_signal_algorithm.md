# 휴리스틱 신호 결정 알고리즘

**버전:** 2026-04-24  
**파일:** `prediction/adaptive_mixin.py`, `prediction/prediction_mixin.py`

---

## 1. 개요

휴리스틱 신호는 **ZigZag 피봇 확정**을 기본 트리거로, **Adaptive SuperTrend 방향**을 필터로 사용하여 `BUY / SELL / HOLD` 신호와 `HIGH / MEDIUM / LOW` 신뢰도를 결정한다.

Transformer 수치 모델과 달리 확률 기반이 아닌 **구조적 전환점 기반** 신호이므로, 피봇이 확정된 봉에서만 발생하고 나머지 구간은 항상 `HOLD`이다.

---

## 2. 전체 흐름

```
[AdaptiveIndicatorManager]
  AdaptiveZigZag.update() + AdaptiveSuperTrend.update()
          ↓
  get_transformer_features() → adaptive_features dict
          ↓
[adaptive_mixin._compute_adaptive_bundle()]
  _parse_adaptive_heuristic_features()
          ↓
  ┌─────────────────────────────────┐
  │  1단계: 피봇 + ST 방향 판단     │  → BUY / SELL / HOLD
  │  2단계: 보완 규칙 4개           │  → HOLD 억제 또는 MEDIUM 강등
  └─────────────────────────────────┘
          ↓
  model_outputs["heuristic"] = {
      "action": BUY/SELL/HOLD,
      "confidence": HIGH/MEDIUM/LOW,
      "reason": ...,
  }
          ↓
[prediction_mixin.get_prediction()]
  [PIVOT-OVERRIDE]
  heuristic.action이 피봇 신호이면
  → Transformer 기반 signal을 피봇 신호로 교체
  → heuristic.confidence를 최종 confidence에 반영
          ↓
  LLM 호출 (교체된 signal 기준으로 판단)
          ↓
  최종 출력 dict: signal / confidence / llm_action / consensus
```

---

## 3. 입력 feature 정의

### 3-1. ZigZag 관련

| feature | 타입 | 설명 |
|---|---|---|
| `azz_new_swing` | float | 피봇 확정 여부. `+1`=고점(H), `-1`=저점(L), `0`=없음 |
| `azz_higher_highs` | float(0/1) | 최근 고점이 직전 고점보다 높은지 (HH 구조) |
| `azz_lower_lows` | float(0/1) | 최근 저점이 직전 저점보다 낮은지 (LL 구조) |
| `azz_structure_up` | float(0/1) | ZigZag 구조가 `uptrend` (HH+HL) |
| `azz_structure_down` | float(0/1) | ZigZag 구조가 `downtrend` (LH+LL) |
| `azz_structure_ranging` | float(0/1) | ZigZag 구조가 `ranging` (혼재) |
| `azz_wave_size_pct` | float | 직전 파동 크기 (정규화, ×100 = %) |
| `azz_bars_since_swing` | float | 마지막 피봇 이후 경과 봉 수 (정규화) |
| `azz_swing_recency` | float | 피봇 신선도 (`exp(-age/5)`, 1에 가까울수록 최신) |

`structure=unknown`: `azz_structure_up / down / ranging` 모두 0 — ZigZag 피봇이 충분히 쌓이지 않은 장 초반 워밍업 상태.

### 3-2. SuperTrend 관련

| feature | 타입 | 설명 |
|---|---|---|
| `ast_direction` | float | ST 방향. `+1`=상승(가격이 ST 위), `-1`=하락(가격이 ST 아래) |
| `ast_signal` | float | ST 신호. `+1`=buy, `-1`=sell, `0`=hold |
| `ast_efficiency_ratio` | float | Efficiency Ratio (0~1). 높을수록 추세 강함 |
| `ast_adx_norm` | float | ADX ÷ 100. 0.25 이상이면 추세 구간 |
| `ast_trend_duration` | float | 현 방향 유지 봉 수 (정규화, ×78 = 봉 수) |

---

## 4. 1단계: 피봇 + SuperTrend 방향 판단

피봇 확정 여부(`azz_new_swing`)와 ST 방향(`ast_direction`)의 **일치 여부**로 신호를 결정한다.

```
azz_swing = round(azz_new_swing)  →  +1 / -1 / 0
ast_dir   = round(ast_direction)  →  +1 / -1 / 0
```

### 결정 테이블

| azz_swing | ast_dir | 신호 | reason 태그 |
|---|---|---|---|
| `-1` (L확정) | `+1` (ST상승) | **BUY** | `zigzag_pivot_low(L)->BUY+ST_UP` |
| `-1` (L확정) | `-1` 또는 `0` | **HOLD** | `zigzag_pivot_low(L)->HOLD_ST_NOT_UP` |
| `+1` (H확정) | `-1` (ST하락) | **SELL** | `zigzag_pivot_high(H)->SELL+ST_DOWN` |
| `+1` (H확정) | `+1` 또는 `0` | **HOLD** | `zigzag_pivot_high(H)->HOLD_ST_NOT_DOWN` |
| `0` / `None` | — | **HOLD** | `zigzag_no_pivot->HOLD` |

### 설계 의도

ZigZag 피봇은 **전환점**을 나타내지만, ST 방향과 불일치하면 추세 흐름을 역행하는 허위 전환일 가능성이 있다. ST 방향을 필터로 사용함으로써 ranging 구간에서 H→L→H→L 반복 시 연속적인 SELL/BUY 발생을 1차로 억제한다.

```
예시:  09:20 H확정, ST방향=↑(상승)
       → HOLD (피봇은 고점이지만 ST는 아직 상승 중 → 신호 불일치)

예시:  10:20 L확정, ST방향=↑(상승)
       → BUY  (저점 확정 + ST 상승 방향 일치 → 반전 신호)
```

---

## 5. 2단계: 보완 규칙 4개

1단계에서 `BUY` 또는 `SELL`이 결정된 경우에만 적용한다. 우선순위 순서로 평가하며, 신호를 **HOLD로 억제**하거나 **confidence를 MEDIUM으로 강등**한다.

기본 confidence: BUY/SELL → `HIGH`, HOLD → `LOW`

---

### [보완-1] `structure=unknown` → HOLD 억제

**조건:** `azz_structure_up = 0`, `azz_structure_down = 0`, `azz_structure_ranging = 0`

**처리:** `신호 → HOLD`, `confidence → LOW`

**이유:**  
장 시작 후 ZigZag 피봇이 충분히 쌓이기 전(약 3개 미만)에는 `structure`가 확정되지 않는다. 이 상태에서 발생하는 피봇은 구조적 맥락 없이 단순 가격 변동일 가능성이 높으므로 신호 발행 자체를 차단한다.

```python
_s_unk = not (_rng or _s_up or _s_dn)
if _s_unk:
    a = "HOLD"
    _conf = "LOW"
    reason += " HOLD:structure=unknown(초기구조미확정)"
```

**실제 사례 (2026-04-24):**
```
09:21  L확정 + ST↑ → 1단계에서 BUY
       structure=unknown (09:01 H 하나만 확정된 상태)
       → 보완-1 적용 → HOLD/LOW
```

---

### [보완-2] LL / HH 구조 미확인 → confidence MEDIUM

**조건 (우선순위 2위, unknown 억제 후 생존한 경우):**
- BUY 신호이나 `azz_lower_lows = False` (LL 미확인)
- SELL 신호이나 `azz_higher_highs = False` (HH 미확인)

**처리:** `confidence → MEDIUM`, 신호는 유지

**이유:**  
ZigZag에서 저점(L)이 확정됐더라도 그 저점이 이전 저점보다 낮지 않으면(`LL=False`) 하락 추세 구조가 확인되지 않은 반등이다. 고점(H)도 마찬가지로 `HH=False`이면 상승 추세 구조가 불명확하다. 신호 자체는 유효하지만 신뢰도를 낮춘다.

```python
elif (a == "BUY" and not _azz_ll) or (a == "SELL" and not _azz_hh):
    _conf = "MEDIUM"
    reason += f" MEDIUM:{'LL=False' if a=='BUY' else 'HH=False'}(구조미확인)"
```

**실제 사례 (2026-04-24):**
```
09:26  H확정 + ST↓ → 1단계에서 SELL
       HH=False (고점이 09:01 H 6516 보다 낮은 6478)
       → 보완-2 적용 → SELL/MEDIUM

10:34  H확정 + ST↓ → 1단계에서 SELL
       HH=False (ranging 구간, 고점 갱신 안됨)
       → 보완-2 적용 → SELL/MEDIUM
```

---

### [보완-3] `ranging` 구간 → confidence MEDIUM

**조건:** `azz_structure_ranging = True`

**처리:** `confidence → MEDIUM` (이미 MEDIUM이면 재강등하지 않음), 신호는 유지

**이유:**  
ranging 구간은 고점과 저점이 혼재하여 ZigZag 피봇이 빈번하게 발생한다. ST 방향 일치로 1단계를 통과하더라도 추세 강도가 약하므로 신뢰도를 낮춘다. 완전 억제는 하지 않는데, ranging이라도 피봇+ST 일치 시 단기 반전 거래 기회가 실제로 존재하기 때문이다.

```python
if _rng:
    if _conf == "HIGH":
        _conf = "MEDIUM"
    reason += " MEDIUM:ranging구간"
```

**실제 사례 (2026-04-24):**
```
10:01  H확정 + ST↓, ranging → SELL/MEDIUM
10:30  L확정 + ST↑, ranging → BUY/MEDIUM
11:12  H확정 + ST↓, ranging → SELL/MEDIUM
```

---

### [보완-4] 추세 역방향 반등 → confidence MEDIUM

**조건:**
- BUY 신호이나 `azz_structure_down = True` (downtrend 내 저점 반등 매수)
- SELL 신호이나 `azz_structure_up = True` (uptrend 내 고점 하락 매도)

**처리:** `confidence → MEDIUM` (이미 MEDIUM이면 재강등하지 않음), 신호는 유지

**이유:**  
추세 방향과 반대로 진입하는 것은 단기 반등/조정을 노리는 전략이다. 성공 가능성이 없지는 않지만 주추세에 역행하므로 신뢰도를 낮춰 LLM과 TradeGate가 추가 판단할 수 있게 한다.

```python
if (a == "BUY" and _s_dn) or (a == "SELL" and _s_up):
    if _conf == "HIGH":
        _conf = "MEDIUM"
    reason += f" MEDIUM:{'downtrend내반등매수' if a=='BUY' else 'uptrend내하락매도'}"
```

**실제 사례 (2026-04-24):**
```
09:30  L확정 + ST↑, structure=downtrend → BUY/MEDIUM
       (downtrend 내 저점 반등 — 단기 반등 가능하나 주추세 역행)
```

---

## 6. 보완 규칙 우선순위 및 상호작용

```
BUY/SELL 신호 진입
      │
      ▼
[보완-1] structure=unknown?
      ├─ YES → HOLD/LOW (이후 규칙 모두 스킵)
      └─ NO
          │
          ▼
[보완-2] LL=False(BUY) / HH=False(SELL)?
          ├─ YES → MEDIUM (보완-3/4는 추가 적용 가능)
          └─ NO (HIGH 유지)
              │
              ▼
          [보완-3] ranging?
              ├─ YES → MEDIUM (이미 MEDIUM이면 그대로)
              └─ NO
          [보완-4] 추세 역방향?
              ├─ YES → MEDIUM (이미 MEDIUM이면 그대로)
              └─ NO
```

보완-2~4는 독립적으로 적용되므로 복수 조건 동시 충족 가능 (모두 MEDIUM 강등, 중복 적용은 없음).

### confidence 결정 매트릭스

| 피봇 | ST | structure | HH/LL | ranging | 추세역방향 | 신호 | conf |
|---|---|---|---|---|---|---|---|
| L | ↑ | uptrend | LL=T | N | N | BUY | **HIGH** |
| L | ↑ | uptrend | LL=F | N | N | BUY | MEDIUM |
| L | ↑ | ranging | LL=T | Y | N | BUY | MEDIUM |
| L | ↑ | downtrend | LL=T | N | Y | BUY | MEDIUM |
| L | ↑ | unknown | — | — | — | **HOLD** | LOW |
| L | ↓ | — | — | — | — | **HOLD** | LOW |
| H | ↓ | downtrend | HH=T | N | N | SELL | **HIGH** |
| H | ↓ | downtrend | HH=F | N | N | SELL | MEDIUM |
| H | ↓ | ranging | HH=T | Y | N | SELL | MEDIUM |
| H | ↓ | uptrend | HH=T | N | Y | SELL | MEDIUM |
| H | ↑ | — | — | — | — | **HOLD** | LOW |
| — | — | — | — | — | — | **HOLD** | LOW |

---

## 7. PIVOT-OVERRIDE (prediction_mixin.py)

휴리스틱 신호는 `model_outputs["heuristic"]`에 저장되지만, 기존 아키텍처에서는 이 값이 Transformer 기반 `signal`을 직접 교체하지 않았다. `PIVOT-OVERRIDE`는 피봇 신호 발생 시 `signal`을 직접 교체하는 브리지이다.

### 동작 조건

```python
_is_pivot_override = (
    heuristic.action in ("BUY", "SELL")
    and (
        "zigzag_pivot_low(L)->BUY"  in heuristic.reason
        or "zigzag_pivot_high(H)->SELL" in heuristic.reason
    )
)
```

HOLD로 억제된 경우(`HOLD:structure=unknown`, `HOLD_ST_NOT_UP` 등)는 override 조건에 해당하지 않아 Transformer 신호가 그대로 유지된다.

### 실행 순서

```
_run_numeric_prediction_and_guardrails()   ← Transformer + 기존 가드레일
    ↓
[PIVOT-OVERRIDE]                           ← 피봇 신호로 signal 교체
    signal     = heuristic.action          (BUY 또는 SELL)
    confidence = heuristic.confidence      (HIGH 또는 MEDIUM)
    guardrail_reason += "pivot_override:PREV->NEW(conf=X)"
    ↓
_build_llm_snapshot()                      ← LLM이 교체된 signal 기준으로 판단
    ↓
LLM 호출
    ↓
consensus = (signal == llm_action)         ← 자동으로 합의 판단
```

### before / after 비교

```
[피봇 없는 구간]
  Transformer signal: HOLD  → PIVOT-OVERRIDE 미적용 → 최종: HOLD

[L확정 + ST↑, structure=downtrend (보완-4)]
  Transformer signal: HOLD
  heuristic.action  : BUY / MEDIUM
  → PIVOT-OVERRIDE 적용
  → 최종: signal=BUY / confidence=MEDIUM

[L확정 + ST↑, structure=unknown (보완-1)]
  Transformer signal: BUY
  heuristic.action  : HOLD / LOW   (억제됨)
  → PIVOT-OVERRIDE 미적용 (reason에 BUY 태그 없음)
  → 최종: signal=BUY / confidence=LOW  (Transformer 그대로)
```

---

## 8. reason 문자열 포맷

`heuristic["reason"]`에 단계별 태그가 순차적으로 누적된다.

### 기본 포맷

```
zigzag_pivot_low(L)->BUY+ST_UP (symbol=KOSPI200 선물 azz_new_swing=-1 ast_dir=1 ast_signal=+0)
```

### 보완 규칙 적용 후

```
zigzag_pivot_low(L)->BUY+ST_UP (symbol=... azz_new_swing=-1 ast_dir=1 ast_signal=+0) MEDIUM:LL=False(구조미확인) MEDIUM:ranging구간
```

### 억제 시

```
zigzag_pivot_low(L)->BUY+ST_UP (symbol=... ) HOLD:structure=unknown(초기구조미확정)
```

---

## 9. 로그 출력

| 이벤트 | 레벨 | 메시지 형식 |
|---|---|---|
| structure=unknown 억제 | INFO | `[HEURISTIC] structure=unknown → HOLD 억제 (symbol=X)` |
| LL/HH 미확인 강등 | INFO | `[HEURISTIC] BUY → MEDIUM confidence (LL=False) symbol=X` |
| ranging 강등 | INFO | `[HEURISTIC] BUY ranging → MEDIUM confidence (symbol=X)` |
| 추세 역방향 강등 | INFO | `[HEURISTIC] BUY trend-counter → MEDIUM confidence (downtrend내반등매수) symbol=X` |
| PIVOT-OVERRIDE 발동 | INFO | `[PIVOT-OVERRIDE] signal HOLD→BUY confidence=MEDIUM reason=...` |

---

## 10. 2026-04-24 실제 적용 결과

| 시각 | 피봇 | ST | structure | HH | LL | 신호 | conf | 비고 |
|---|---|---|---|---|---|---|---|---|
| 09:21 | L확정 | ↑ | unknown | F | T | **HOLD** | LOW | 보완-1: 초기구조미확정 |
| 09:26 | H확정 | ↓ | downtrend | F | T | SELL | MEDIUM | 보완-2: HH=False |
| 09:30 | L확정 | ↑ | downtrend | F | T | BUY | MEDIUM | 보완-4: downtrend반등 |
| 10:01 | H확정 | ↓ | ranging | T | T | SELL | MEDIUM | 보완-3: ranging |
| 10:30 | L확정 | ↑ | ranging | T | T | BUY | MEDIUM | 보완-3: ranging |
| 10:34 | H확정 | ↓ | ranging | F | T | SELL | MEDIUM | 보완-2+3: HH=F+ranging |
| 10:50 | L확정 | ↑ | ranging | F | F | BUY | MEDIUM | 보완-2+3: LL=F+ranging |
| 11:12 | H확정 | ↓ | ranging | T | F | SELL | MEDIUM | 보완-3: ranging |
| 13:10 | L확정 | ↑ | ranging | T | T | BUY | MEDIUM | 보완-3: ranging |

`HIGH` confidence가 발생하려면 `structure=uptrend/downtrend` + `HH/LL 확인` + `비ranging` 조건이 동시에 충족돼야 한다. 오늘 장 전체가 ranging 구조여서 모든 피봇 신호가 MEDIUM으로 발행됐다.

---

## 11. 향후 보완 가능 항목 (미구현)

| 항목 | 조건 | 처리 |
|---|---|---|
| wave_size_pct 하한 | `azz_wave_size_pct * 100 < 0.3%` | 잡음 피봇 차단 → HOLD |
| ST trend_duration 최소 봉수 | `ast_trend_duration * 78 < 3봉` | ST 방향 전환 직후 whipsaw → MEDIUM |
| bars_since_swing 최소 | `azz_bars_since_swing * 50 < 5봉` | 연속 피봇 중복 억제 → HOLD |
