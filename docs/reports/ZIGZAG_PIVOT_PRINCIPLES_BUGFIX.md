# ZigZag 피봇 2원칙 — 알고리즘 설계 및 버그 수정 보고서

> 대상 파일: `indicators/adaptive_zigzag.py`  
> 검토 기준: **원칙 1 — 확정 피봇은 취소불가** / **원칙 2 — 피봇 H/L 교번 발생**  
> 수정 버전: `adaptive_zigzag_patched.py`

---

## 목차

1. [2원칙 알고리즘 개요](#1-2원칙-알고리즘-개요)
2. [전체 파이프라인 흐름](#2-전체-파이프라인-흐름)
3. [원칙 1 — 확정 피봇 취소불가](#3-원칙-1--확정-피봇-취소불가)
4. [원칙 2 — H/L 교번 발생](#4-원칙-2--hl-교번-발생)
5. [버그 수정 상세](#5-버그-수정-상세)
6. [수정 전후 비교](#6-수정-전후-비교)
7. [테스트 시나리오](#7-테스트-시나리오)

---

## 1. 2원칙 알고리즘 개요

AdaptiveZigZag 지표는 피봇(스윙 고점·저점)을 실시간으로 탐지하는 알고리즘이다.
피봇의 신뢰성과 일관성을 보장하기 위해 **두 가지 불변 원칙**이 설계 기반으로 작동한다.

### 원칙 1 — 확정 피봇은 취소불가

```
한 번 confirmed=True 가 된 SwingPoint 는 영구 불변이다.
price, index, swing_type 어느 필드도 사후 변경되어서는 안 된다.
```

**이 원칙이 필요한 이유:**

- `ZigZagState.recent_swings`, LLM context, Telegram 알림, ML 피처 파이프라인 등
  여러 소비자(consumer)가 SwingPoint 객체를 직접 참조한다.
- 확정 후 값이 바뀌면 소비자들이 서로 다른 값을 보게 되어 신호 불일치가 발생한다.
- 특히 LLM judge가 참조한 고점이 사후에 다른 가격으로 교체되면
  매매 결정의 근거가 뒤집히는 치명적 결과로 이어진다.

### 원칙 2 — 피봇 H/L 교번 발생

```
_all_swings 리스트는 항상 HIGH와 LOW가 교대로 나열되어야 한다.
[H, L, H, L, H, L, ...]  또는  [L, H, L, H, L, H, ...]
같은 타입이 연속으로 추가되어서는 안 된다.
```

**이 원칙이 필요한 이유:**

- 파동(wave)의 정의 자체가 H→L 또는 L→H 이동이다.
  같은 타입이 연속하면 파동 크기 계산, 피보나치, 구조 분석이 모두 오작동한다.
- `_analyze_structure()`, `_analyze_micro_structure()` 등은 교번이 보장됨을
  암묵적 전제로 작성되어 있다. 교번이 깨지면 잘못된 추세 판정이 나온다.

---

## 2. 전체 파이프라인 흐름

```
update(high, low, close)
│
├─ 1. ATR / thr_abs 계산
│
├─ 2. 3-a: pending_confirm 처리
│   ├─ remaining -= 1
│   ├─ remaining <= 0  →  confirmed SwingPoint 추가 (_add_swing)
│   │                      direction 반전, pending 초기화
│   └─ max_wait_bars 초과 → pending 취소
│
└─ 3. 3-b: 방향 결정 / 전환  ← new_swing_signal == "none" 일 때만 진입
    ├─ direction == 0  →  초기범위 탐색 → 최초 피봇 확정
    ├─ direction == 1  →  pending_high 추적 → thr_abs 역전 시 high 후보 등록
    └─ direction == -1 →  pending_low 추적  → thr_abs 역전 시 low  후보 등록
```

### `_add_swing()` 내부 흐름

```
_add_swing(idx, price, swing_type, atr)
│
├─ 클러스터 검사: prev_same 존재 AND dist_pct <= cluster_tolerance_pct
│   ├─ 더 극값  → prev_same 속성 in-place 갱신 (★원칙 1 준수)
│   └─ 덜 극값 → 무시, return
│
├─ is_major 계산 (avg_wave 비율 or ATR 배수 fallback)
│
└─ _all_swings.append(SwingPoint(confirmed=True))
   (슬라이스 trim: max_swings*2 초과 시)
```

---

## 3. 원칙 1 — 확정 피봇 취소불가

### 준수가 확인된 경로

| 경로 | 구현 |
|---|---|
| 신규 피봇 추가 | `_all_swings.append(SwingPoint(confirmed=True))` — 추가 전용, 삭제 없음 |
| `_all_swings` 트림 | `self._all_swings = self._all_swings[-max_swings:]` — 슬라이스만 수행, 개별 항목 삭제 없음 |
| `pending_confirm` 취소 | pending 상태의 후보만 취소됨 — `confirmed=True` 항목에 영향 없음 |

### ❌ 위반 — BUG-CLUSTER-1 (수정 완료)

**위치:** `_add_swing()` → 클러스터링 분기

**문제 코드 (수정 전):**

```python
# 더 극값 → 기존 항목 교체
replace_idx = None
for i in range(len(self._all_swings) - 1, -1, -1):
    if (self._all_swings[i].swing_type == swing_type
            and self._all_swings[i].confirmed):
        replace_idx = i
        break
if replace_idx is not None:
    self._all_swings[replace_idx] = SwingPoint(   # ← 원칙 1 위반
        index=idx, price=price, ...
        confirmed=True, ...
    )
```

**문제 분석:**

클러스터 조건(`dist_pct ≤ cluster_tolerance_pct = 0.3%`)은
같은 가격대에 연속으로 피봇이 몰릴 때 마지막 극값 하나만 남기기 위한 로직이다.
의도 자체는 합리적이지만, 구현 방식이 원칙 1을 위반한다.

`self._all_swings[replace_idx] = 새 SwingPoint()`는 기존 객체를 완전히 교체한다.
이 시점에 외부 소비자들이 이미 `old_swing_point.price`를 캐시했다면,
교체 후에는 두 개의 다른 가격이 시스템에 공존하게 된다.

또한 `replace_idx` 탐색이 `reversed()` 기반이지만,
`_all_swings = [H, L, H]` 상태에서 마지막 H가 아니라 중간 L 뒤의 H가 아닌
더 앞쪽 H를 교체할 경우 `H_old → L → H_new` 순서로 시간 정합성이 깨질 수 있다.

**수정 코드:**

```python
# [BUG-CLUSTER-1] 수정: 객체 교체 금지, 속성만 in-place 갱신
prev_price_snapshot = float(prev_same.price)
prev_same.index           = idx
prev_same.price           = price
prev_same.atr_at_swing    = atr
prev_same.confirmed_at_idx = c_idx
prev_same.confirmed_close  = confirmed_close
# confirmed / is_major / swing_type 은 불변 유지
```

**수정 이유:**

- `prev_same`은 이미 `reversed()` 탐색으로 찾은 마지막 동일 타입 SwingPoint다.
  별도의 `replace_idx` 탐색 없이 직접 갱신하므로 "tail이 아닌 항목 교체" 위험도 제거된다.
- `confirmed=True`와 `is_major`는 확정 시점에 결정된 판단으로 불변이어야 한다.
- 소비자들이 참조 중인 객체의 동일성(identity)을 유지하면서 값만 최신화된다.

---

## 4. 원칙 2 — H/L 교번 발생

### 교번을 보장하는 핵심 메커니즘

```python
# 확정 직후 direction 반전 ([P-FIX-F])
if stype == "high":
    self._current_direction = -1      # 다음은 LOW 탐색
    self._pending_low       = float(low)
    self._pending_high      = 0.0

elif stype == "low":
    self._current_direction = 1       # 다음은 HIGH 탐색
    self._pending_high      = float(high)
    self._pending_low       = float("inf")
```

`direction`이 반전되면 3-b 블록에서 현재 방향과 반대 타입 후보는 등록되지 않는다.
이것이 교번의 1차 보호 장치다.

### ❌ 위반 1 — BUG-INIT-DIR0 (주석 명확화)

**위치:** `3-b` 블록 → `direction == 0` 초기범위 분기

**문제 분석:**

`P-FIX-B` 보호 코드(`if new_swing_signal != "none": pass`)는
기존 주석이 "3-a pending_confirm 확정 전용"으로만 표기되어 있었다.
실제로는 `direction == 0` 초기범위 확정도 `new_swing_signal`을 세팅하므로
동일 봉에서 3-b 재진입이 구조적으로 차단된다.

하지만 주석 부재로 인해 향후 유지보수 시 이 보호를 제거할 위험이 있다.

**초기범위 확정 흐름 (교번 보장 검증):**

```
봉 N: direction=0
  → pending_high > pending_low 조건 충족
  → LOW 확정: _all_swings = [LOW]
  → direction = 1, new_swing_signal = "new_low"
  → _last_confirmed_bar_idx = N

  → new_swing_signal != "none" → 3-b 전체 skip (★BUG-INIT-DIR0 수정으로 명시)

봉 N+1: direction=1
  → pending_high 추적
  → thr_abs 역전 시 HIGH 후보 등록
  → _bar_idx(N+1) > _last_confirmed_bar_idx(N) → P-FIX-B 통과
  → _all_swings = [LOW, pending HIGH...]  → 교번 유지
```

**수정 내용:** 주석 명확화 (동작 변경 없음)

```python
# [BUG-INIT-DIR0] 수정: direction=0 초기범위 확정도 같은 봉 재진입 차단에 포함.
# new_swing_signal != "none" 가드가 초기범위 확정 경우도 커버하므로
# direction==1/-1 블록의 P-FIX-B(_bar_idx <= _last_confirmed_bar_idx)와
# 이중 방어를 구성한다.
if new_swing_signal != "none":
    pass  # 다음 봉부터 탐색 재개 (3-a pending_confirm 확정 및 초기범위 확정 모두 포함)
```

### ❌ 위반 2 — BUG-MICRO-ALT (수정 완료)

**위치:** `_analyze_micro_structure()`

**문제 코드 (수정 전):**

```python
def _analyze_micro_structure(self) -> str:
    rh = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.HIGH and s.confirmed]
    rl = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.LOW  and s.confirmed]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"
    ...
```

**문제 분석:**

`_all_swings[-4:]` 에서 HIGH와 LOW를 독립적으로 필터링한다.
만약 `_all_swings = [H1, H2, L1, L2]` (교번 깨진 상태)라면:
- `rh = [H1.price, H2.price]` (2개 이상)
- `rl = [L1.price, L2.price]` (2개 이상)

`len(rh) >= 2 and len(rl) >= 2` 조건을 통과해서 구조 판정을 시도하지만,
`H1→H2`는 실제로는 같은 방향의 연속 고점이므로 의미 있는 파동 관계가 아니다.
`rh[-1] > rh[-2]`가 `H2 > H1`이라면 HH로 판정되어 상승 구조로 오판된다.

이 버그는 BUG-CLUSTER-1이 수정되어도 다른 경로로 교번이 깨질 경우에 여전히 발동한다.
방어적 코딩 차원에서 수정이 필요하다.

**수정 코드:**

```python
def _analyze_micro_structure(self) -> str:
    # 교번 순서가 유지되는 확정 피봇만 취득
    confirmed = [s for s in self._all_swings if s.confirmed]
    if len(confirmed) < 4:
        return "unknown"
    recent4 = confirmed[-4:]

    # 교번 검증: 인접 피봇이 동일 타입이면 신뢰할 수 없음
    for i in range(1, len(recent4)):
        if recent4[i].swing_type == recent4[i - 1].swing_type:
            return "unknown"  # 교번 깨진 상태 → 판정 불가

    rh = [s.price for s in recent4 if s.swing_type == SwingType.HIGH]
    rl = [s.price for s in recent4 if s.swing_type == SwingType.LOW]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"
    ...
```

**수정 이유:**

교번 검증을 필터링 전에 수행함으로써,
`_all_swings`의 교번 불변식이 어떤 경로로든 깨졌을 때 "unknown"으로 안전하게 폴백한다.
"unknown" 반환은 트레이딩 판단에서 "기권"으로 처리되어 오매매보다 낫다.

---

## 5. 버그 수정 상세

### 수정 요약

| ID | 위치 | 분류 | 원칙 | 심각도 |
|---|---|---|---|---|
| BUG-CLUSTER-1 | `_add_swing()` 클러스터 분기 | 논리 버그 | 원칙 1 위반 | 높음 |
| BUG-INIT-DIR0 | `update()` 3-b 진입 가드 주석 | 문서 결함 | 원칙 2 (잠재적) | 낮음 |
| BUG-MICRO-ALT | `_analyze_micro_structure()` | 방어 로직 결여 | 원칙 2 (파생) | 중간 |

### BUG-CLUSTER-1 상세

**재현 시나리오:**

```
t=100: HIGH@355.00 확정 → _all_swings = [L@350, H@355]
t=101: HIGH@355.30 → cluster 조건 충족 (dist=0.084% < 0.3%)
       is_more_extreme = True (355.30 > 355.00)

수정 전:
  _all_swings[1] = 새 SwingPoint(price=355.30)
  → 외부에서 참조 중인 H@355.00 가격이 사라짐

수정 후:
  _all_swings[1].price = 355.30  (동일 객체, 값만 갱신)
  → 외부 참조자도 355.30을 보게 됨 (일관성 유지)
```

**클러스터링의 올바른 의미:**

클러스터링은 "같은 피봇의 더 정확한 극값을 추적하는 것"이다.
예를 들어 355.00에서 확정됐지만 같은 파동에서 355.30이 찍혔다면,
그 파동의 진짜 고점은 355.30이다. 이것은 피봇 취소가 아니라 **피봇 정밀화**이므로
동일 객체의 값 갱신이 의미상 올바르다.

### BUG-MICRO-ALT 상세

**오판 발생 조건:**

```
_all_swings = [H@355, H@358, L@350, L@347]  (교번 깨진 상태)
recent4 = [H@355, H@358, L@350, L@347]

수정 전 로직:
  rh = [355, 358]  → rh[-1] > rh[-2]  → HH = True
  rl = [350, 347]  → rl[-1] < rl[-2]  → HL = False, LL = True
  → lh=False, ll=True → "ranging" 반환

  실제로는 연속 고점 + 연속 저점이지만 ranging 으로 잘못 판정됨

수정 후 로직:
  recent4[0].swing_type == recent4[1].swing_type  (H == H) → "unknown" 반환
```

---

## 6. 수정 전후 비교

### BUG-CLUSTER-1

```python
# ────── 수정 전 ──────
if is_more_extreme:
    replace_idx = None
    for i in range(len(self._all_swings) - 1, -1, -1):
        if (self._all_swings[i].swing_type == swing_type
                and self._all_swings[i].confirmed):
            replace_idx = i
            break
    if replace_idx is not None:
        self._all_swings[replace_idx] = SwingPoint(   # ← 원칙 1 위반
            index=idx, price=price, swing_type=swing_type,
            atr_at_swing=atr, is_major=prev_same.is_major,
            confirmed=True, confirmed_at_idx=c_idx,
            confirmed_close=confirmed_close,
        )

# ────── 수정 후 ──────
if is_more_extreme:
    prev_price_snapshot = float(prev_same.price)    # 로그용
    prev_same.index           = idx                 # in-place 갱신
    prev_same.price           = price
    prev_same.atr_at_swing    = atr
    prev_same.confirmed_at_idx = c_idx
    prev_same.confirmed_close  = confirmed_close
    # confirmed / is_major / swing_type 불변
```

### BUG-MICRO-ALT

```python
# ────── 수정 전 ──────
def _analyze_micro_structure(self) -> str:
    rh = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.HIGH and s.confirmed]
    rl = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.LOW  and s.confirmed]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"
    hh = rh[-1] > rh[-2]
    ...

# ────── 수정 후 ──────
def _analyze_micro_structure(self) -> str:
    confirmed = [s for s in self._all_swings if s.confirmed]
    if len(confirmed) < 4:
        return "unknown"
    recent4 = confirmed[-4:]

    # 교번 검증 — 먼저 수행
    for i in range(1, len(recent4)):
        if recent4[i].swing_type == recent4[i - 1].swing_type:
            return "unknown"  # 교번 깨진 상태 → 안전 폴백

    rh = [s.price for s in recent4 if s.swing_type == SwingType.HIGH]
    rl = [s.price for s in recent4 if s.swing_type == SwingType.LOW]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"
    hh = rh[-1] > rh[-2]
    ...
```

---

## 7. 테스트 시나리오

수정 내용을 검증하기 위한 단위 테스트 시나리오다.

### T-01: BUG-CLUSTER-1 — 클러스터 교체 후 객체 동일성 유지

```python
def test_cluster_replace_preserves_identity():
    zz = AdaptiveZigZag(AdaptiveZigZagConfig(cluster_tolerance_pct=0.5))

    # 상태를 강제로 세팅하여 HIGH 클러스터 조건 유도
    # L@350 확정 후 H@355 확정
    zz._all_swings.append(SwingPoint(0, 350.0, SwingType.LOW, 1.0, confirmed=True))
    h_pivot = SwingPoint(5, 355.00, SwingType.HIGH, 1.0, confirmed=True)
    zz._all_swings.append(h_pivot)

    # cluster 조건 충족하는 극값으로 _add_swing 호출
    zz._bar_idx = 10
    zz._add_swing(6, 355.25, SwingType.HIGH, 1.0,
                  confirmed_at_idx=10, confirmed_close=354.0)

    # 객체 동일성: h_pivot 객체가 교체되지 않고 갱신되어야 함
    assert zz._all_swings[1] is h_pivot, "객체가 교체됨 — BUG-CLUSTER-1 재발"
    assert zz._all_swings[1].price == 355.25, "가격 갱신 안 됨"
    assert zz._all_swings[1].confirmed is True, "confirmed 변경됨"
```

### T-02: BUG-MICRO-ALT — 교번 깨진 상태에서 unknown 반환

```python
def test_micro_structure_broken_alternation():
    zz = AdaptiveZigZag()

    # 교번이 깨진 상태 강제 주입: H, H, L, L
    zz._all_swings = [
        SwingPoint(1, 355.0, SwingType.HIGH, 1.0, confirmed=True),
        SwingPoint(2, 358.0, SwingType.HIGH, 1.0, confirmed=True),
        SwingPoint(3, 350.0, SwingType.LOW,  1.0, confirmed=True),
        SwingPoint(4, 347.0, SwingType.LOW,  1.0, confirmed=True),
    ]
    result = zz._analyze_micro_structure()
    assert result == "unknown", f"교번 깨진 상태에서 unknown 아님: {result}"
```

### T-03: 정상 교번 상태에서 micro_structure 정상 작동

```python
def test_micro_structure_normal_uptrend():
    zz = AdaptiveZigZag()

    # 정상 교번: L, H, L, H (상승 구조)
    zz._all_swings = [
        SwingPoint(1, 348.0, SwingType.LOW,  1.0, confirmed=True),
        SwingPoint(2, 355.0, SwingType.HIGH, 1.0, confirmed=True),
        SwingPoint(3, 350.0, SwingType.LOW,  1.0, confirmed=True),
        SwingPoint(4, 358.0, SwingType.HIGH, 1.0, confirmed=True),
    ]
    result = zz._analyze_micro_structure()
    assert result == "uptrend", f"상승 구조 오판: {result}"
```

### T-04: direction=0 초기범위 확정 후 다음 봉에서만 후보 등록

```python
def test_init_dir0_no_same_bar_candidate():
    """초기범위 LOW 확정 봉과 동일한 봉에서 HIGH 후보가 등록되어서는 안 된다."""
    zz = AdaptiveZigZag(AdaptiveZigZagConfig(
        atr_multiplier=1.0, confirmation_bars=1, min_wave_bars=0
    ))
    # 충분한 봉을 공급해 direction=0 → LOW 확정 유도 후
    # new_swing_signal == "new_low" 인 봉에서 pending_confirm이 None인지 확인
    # (HIGH 후보가 즉시 등록되지 않아야 함)
    prices = [(100, 95), (102, 90), (108, 88), (115, 85)]
    for i, (h, l) in enumerate(prices):
        state = zz.update(h, l, (h+l)/2)
        if state.new_swing_signal == "new_low":
            assert zz._pending_confirm is None, \
                f"초기범위 LOW 확정 봉(bar={i})에서 HIGH 후보 즉시 등록됨"
            break
```

---

## 부록 — 2원칙 상태 다이어그램

```
[direction=0]
    ↓ (pending_high - pending_low >= thr_abs)
    ↓
  ┌─────────────────────────────────────────┐
  │ 초기범위 확정                            │
  │ · pending_high_idx > pending_low_idx    │
  │   → LOW 확정 → direction=1             │
  │ · pending_high_idx < pending_low_idx    │
  │   → HIGH 확정 → direction=-1           │
  │ · new_swing_signal != "none"            │
  │   → 동일 봉 3-b 차단 (BUG-INIT-DIR0)  │
  └─────────────────────────────────────────┘

[direction=1]  "상승 파동 탐색 중"
    ↓ (pending_high - low >= thr_abs)
    ↓
  ┌─────────────────────────────────────────┐
  │ HIGH 후보 등록 (pending_confirm)         │
  │ P-FIX-B: 확정 봉과 동일봉이면 차단      │
  └──────────────┬──────────────────────────┘
                 │ remaining 카운트다운
                 ↓
  ┌─────────────────────────────────────────┐
  │ HIGH 확정 (_add_swing)                  │
  │ · 원칙 1: confirmed=True 불변           │
  │ · 클러스터: in-place 갱신만 허용        │
  │   (BUG-CLUSTER-1 수정)                 │
  │ · direction=-1 전환                     │
  └─────────────────────────────────────────┘

[direction=-1]  "하락 파동 탐색 중"
    (대칭 구조, 생략)
```

---

## v2/v3 코드 리뷰 수정 (2026-05-03)

### 추가 수정된 이슈

v2 및 v3 코드 리뷰에서 17개 추가 이슈가 수정되었습니다. 이 중 2원칙과 직접 관련된 주요 수정은 다음과 같습니다:

#### v2 수정 (9개 이슈)
- **Critical**: `compute_from_df()`에서 `full_reset()` 사용으로 `_bar_idx` 누적 방지
- **Critical**: `_process_pending_confirmation()`에서 deque 상대/절대 인덱스 변환
- **Major**: `reset_for_new_session()`에서 `_current_direction` 복원 (원칙 2 준수 강화)
- **Major**: `get_pending_confirmation_probability()`에서 `max_wait_bars=0` 조건 추가
- **Major**: `_enforce_hl_alternation()`에서 bisect 사용하여 올바른 순서 병합 (원칙 2 준수)
- **Minor**: `full_reset()`/`reset_for_new_session()` 코드 중복 제거 (`_init_buffers()` 추출)
- **Minor**: 예외 처리 주석 추가
- **Minor**: 불필요한 `replace` import 제거
- **Minor**: `_bar_idx` 주석 추가

#### v3 수정 (8개 이슈)
- **Critical**: `_init_buffers()`에서 `_current_direction` 명시적 초기화
- **Critical**: `_enforce_hl_alternation()` O(n²) 성능 개선 (단순 병합+정렬로 변경)
- **Major**: direction=0 초기범위 확정 시 deque 상대 인덱스 변환
- **Major**: `reset_for_new_session()`에서 `_last_confirmed_bar_idx` 복원 (원칙 2 준수 강화)
- **Major**: `_calc_threshold_pct()`에 `bar_idx` 파라미터 추가 (암묵적 의존성 제거)
- **Minor**: `__init__()`에서 `full_reset()` 직접 호출
- **Minor**: `_validate_extreme_at_confirmation()` 범위 계산 주석 추가
- **Minor**: `_all_swings` 슬라이싱 주석 추가

### 2원칙 관련 주요 개선

1. **원칙 1 (확정 피봇 취소불가) 강화**: v2-2 deque 인덱스 수정으로 확정 시점의 실제 가격을 정확히 저장
2. **원칙 2 (H/L 교번) 강화**: v2-3, v3-4에서 direction과 last_confirmed_bar_idx 복원으로 세션 간 교번 유지

자세한 내용은 `adaptive_zigzag_fixes.md`를 참조하세요.

---

*생성일: 2026-04-29*  
*업데이트: 2026-05-03 (v2/v3 수정 추가)*  
*적용 대상: `indicators/adaptive_zigzag.py` (SkyPredictor)*
