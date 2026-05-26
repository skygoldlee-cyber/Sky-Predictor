# AdaptiveZigZag 피봇 확정 알고리즘 보완 설계서

**기준 버전:** 2026-04-24 (rev.2 — SESSION-MW 추가)  
**분석 기반:** 실측 로그 (logs/prediction.log, 10건 피봇)  
**대상 파일:** `kospi_indicators/kospi_indicators/adaptive_zigzag.py`, `config.py`, `prediction/pipeline.py`, `prediction/data_builder.py`

---

## 목차

1. [현행 알고리즘 요약](#1-현행-알고리즘-요약)
2. [실측 데이터 분석](#2-실측-데이터-분석)
3. [보완 항목 상세](#3-보완-항목-상세)
   - [3-1. confirmation_bars 동적 조절](#3-1-confirmation_bars-동적-조절--즉시)
   - [3-2. min_wave_pct 하한 설정](#3-2-min_wave_pct-하한-설정--즉시)
   - [3-3. structure 다수결 판정](#3-3-structure-다수결-판정--즉시)
   - [3-4. 피봇 클러스터링 실제 적용](#3-4-피봇-클러스터링-실제-적용--단기)
   - [3-5. bars_since 기반 임계값 decay](#3-5-bars_since-기반-임계값-decay--단기)
   - [3-6. pending 피봇을 잠정 SR에 포함](#3-6-pending-피봇을-잠정-sr에-포함--단기)
   - [3-7. is_major 파동 비율 기준](#3-7-is_major-파동-비율-기준--중기)
   - [3-8. 방향 ER (Directional ER)](#3-8-방향-er-directional-er--중기)
   - [3-9. 시간대별 min_wave_bars 테이블 (SESSION-MW)](#3-9-시간대별-min_wave_bars-테이블-session-mw--즉시)
4. [구현 우선순위 요약](#4-구현-우선순위-요약)
5. [Config 변경 목록](#5-config-변경-목록)

---

## 1. 현행 알고리즘 요약

```
FC0 틱 수신
    ↓
update(high, low, close, bar_time)
    ↓
[1] True Range → ATR (RMA)
    ↓
[2] _calc_threshold_pct()
      ATR/close × mult(ER 기반) → thr_pct [0.3% ~ 3.0%]
      장초반(09:00~09:30): atr_multiplier_max=8.0 적용
    ↓
[3-a] pending_confirm 처리
      remaining 카운트다운 → 0이 되면 확정
      freeze_on_confirm=True: 대기 중 추가 갱신 차단
      max_wait_bars > 0: 초과 시 자동 취소
    ↓
[3-b] 방향 결정 / 전환
      current_direction: 0(초기) / +1(상승) / -1(하락)
      임계값 돌파 시 반대 방향으로 전환 + pending 등록
    ↓
[4] 파동 크기, Fibonacci, S/R, 구조 분석
    ↓
ZigZagState 반환
```

### 주요 파라미터 (현행)

| 파라미터 | 현행값 | 역할 |
|---|---|---|
| `confirmation_bars` | 1 | 후보 등록 후 확정까지 대기 봉 수 |
| `min_wave_bars` | 5 | 피봇 간 최소 봉 수 (단일값) |
| `min_wave_pct` | 0.0 | 파동 최소 크기 (비활성) |
| `atr_multiplier_min` | 1.0 | ER=0일 때 ATR 배수 |
| `atr_multiplier_max` | 4.0 | ER=1일 때 ATR 배수 |
| `early_session_start_time` | "09:00" | 장초반 시작 시각 (ATR mmax 전용) |
| `early_session_end_time` | "09:30" | 장초반 종료 시각 (ATR mmax 전용) |
| `early_session_atr_multiplier_max` | 8.0 | 장초반 ATR 최대 배수 |
| `cluster_tolerance_pct` | 0.3 | 클러스터 허용 범위 (미사용) |
| `structure_points` | 3 | 구조 판정에 사용할 피봇 수 |
| `freeze_on_confirm` | True | 대기 중 후보 갱신 차단 |
| `session_min_wave_bars_table` | [] | 시간대별 min_wave_bars 테이블 (3-9) |

---

## 2. 실측 데이터 분석

**2026-04-24 KP200 선물 피봇 10건**

| # | 시각 | 유형 | 가격 | thr_pct | dist_pct | 확정지연 | 파동크기 | 피봇 간격 |
|---|---|---|---|---|---|---|---|---|
| 1 | 09:20 | H | 6516.54 | 0.7632% | +0.72% | 1봉 | — | — |
| 2 | 09:21 | L | 6456.97 | 0.3000% | -0.25% | 1봉 | 59.57pt | 1분 |
| 3 | 09:26 | H | 6478.40 | 0.3000% | +0.33% | 1봉 | 21.43pt | 5분 |
| 4 | 09:30 | L | 6456.33 | 0.3000% | -0.33% | 1봉 | 22.07pt | 4분 |
| 5 | 10:01 | H | 6486.21 | 0.3000% | +0.29% | 1봉 | 29.88pt | 31분 |
| 6 | 10:30 | L | 6448.64 | 0.3000% | -0.35% | 1봉 | 37.57pt | 29분 |
| 7 | 10:34 | H | 6476.72 | 0.3000% | +0.30% | 1봉 | 28.08pt | 4분 |
| 8 | 10:50 | L | 6450.73 | 0.3000% | -0.37% | 1봉 | 25.99pt | 16분 |
| 9 | 11:12 | H | 6492.64 | 0.3000% | +0.37% | 1봉 | 41.91pt | 22분 |
| 10 | 13:10 | L | 6403.74 | 0.3000% | -0.51% | 1봉 | 88.90pt | 118분 |

**관찰된 문제점:**

- 전 피봇 `waited=1봉` — `confirmation_bars=1` 고정으로 지연 없이 확정되나, ranging 구간에서 허위 전환 위험
- `thr_pct=0.30%` 최솟값에 고착 (09:26~10:50 구간 8건) — ER이 낮아 multiplier가 min에 도달
- 09:26 H(6478)→10:34 H(6477): 동일 저항대 두 번 등록 (클러스터 미처리)
- 10:01~11:12 전 구간 `structure=ranging` — 조건이 너무 엄격해 추세 구분 불가
- 11:12→13:10: 118분 공백 후 88.90pt 대파동 — 장시간 무피봇 구간에서 감도 유지 실패

---

## 3. 보완 항목 상세

---

### 3-1. confirmation_bars 동적 조절 🔴 즉시

#### 문제

`confirmation_bars=1` 고정으로 ranging 구간에서 **1봉 후 즉시 확정**됩니다. ranging 구간은 가격이 좁은 범위를 오가므로 허위 전환이 빈번하고, 확정 직후 반전 취소가 반복됩니다.

#### 설계

```
구조         확정지연 봉수   근거
──────────────────────────────────────────────────
uptrend      1봉           추세 편승 즉시성 유지
downtrend    1봉           추세 편승 즉시성 유지
ranging      2봉           허위 전환 2차 검증
unknown      3봉           구조 미확정 추가 보수
파동 < ATR   +1봉          소파동 추가 검증
```

#### 구현

```python
# AdaptiveZigZag에 추가
def _calc_confirmation_bars(self) -> int:
    """구조/파동 크기에 따른 동적 confirmation_bars 계산."""
    base = int(getattr(self.config, "confirmation_bars", 1))
    structure = str(self._state.structure)

    if structure == "ranging":
        base = max(base, 2)
    elif structure == "unknown":
        base = max(base, 3)
    # 소파동 추가 검증
    if self._prev_atr > 0:
        wave = abs(self._state.wave_size)
        if wave < self._prev_atr:
            base += 1
    return base

# pending 등록 시 (기존 cfg.confirmation_bars 대신)
self._pending_confirm = dict(
    type="high", idx=self._pending_high_idx,
    price=self._pending_high, atr=atr,
    remaining=self._calc_confirmation_bars(),   # ← 동적
)
```

#### Config 추가

```python
confirmation_bars_ranging: int = 2     # ranging 구간 확정 대기 봉 수
confirmation_bars_unknown: int = 3     # unknown 구간 확정 대기 봉 수
```

#### 기대 효과

- ranging 구간 허위 피봇 30~40% 감소
- uptrend/downtrend에서는 즉시성 유지

---

### 3-2. min_wave_pct 하한 설정 🔴 즉시

#### 문제

`min_wave_pct=0.0` 기본값으로 경계 잡음 피봇이 허용됩니다. `thr_pct=0.30%`에 맞닿은 dist_pct 0.25~0.33% 피봇들이 실제 추세 전환보다 ATR 임계값 경계의 노이즈일 가능성이 높습니다.

#### 설계

```
min_wave_pct = 0.25%   →  파동 크기가 현재가의 0.25% 미만이면 후보 등록 차단
```

`_is_wave_length_ok()` 에서 **현재 임계값 dist_pct** 기반으로 추가 체크합니다.

#### 구현

```python
# AdaptiveZigZagConfig
min_wave_pct: float = 0.25    # 변경 (기존 0.0)

# _is_wave_length_ok() 보완 (기존 코드에 추가)
def _is_wave_length_ok(self, thr_abs: float, close: float) -> bool:
    cfg = self.config
    # 기존 min_wave_bars 체크
    min_bars = int(getattr(cfg, "min_wave_bars", 0) or 0)
    if min_bars > 0 and self._last_confirmed_bar_idx >= 0:
        if (self._bar_idx - self._last_confirmed_bar_idx) < min_bars:
            return False
    # 파동 크기 % 체크 (기존)
    min_pct = float(getattr(cfg, "min_wave_pct", 0.0) or 0.0)
    if min_pct > 0 and close > 0:
        if (thr_abs / close * 100.0) < min_pct:
            return False
    # [신규] dist_pct 체크: 실제 이동 거리가 min_wave_pct 미만이면 차단
    # pending_high/low와 현재 close 간 거리 계산
    if min_pct > 0 and close > 0:
        if self._current_direction == 1 and self._pending_high > 0:
            actual_dist_pct = (self._pending_high - close) / close * 100.0
            if abs(actual_dist_pct) < min_pct:
                return False
        elif self._current_direction == -1 and self._pending_low < float("inf"):
            actual_dist_pct = (close - self._pending_low) / close * 100.0
            if abs(actual_dist_pct) < min_pct:
                return False
    return True
```

#### 기대 효과

- 오늘 로그 기준: 09:21 L(dist=-0.25%) → `min_wave_pct=0.25`로 차단
- 경계 잡음 피봇 3~4건 → 1~2건으로 감소 예상

---

### 3-3. structure 다수결 판정 🔴 즉시

#### 문제

`_analyze_structure()`가 **모든** 포인트가 일관돼야 `uptrend`/`downtrend`를 반환합니다. 피봇 1개라도 역방향이면 `ranging`으로 판정됩니다. KP200처럼 변동성이 큰 시장에서는 추세 중 단기 조정이 불가피하므로 항상 `ranging`에 갇히는 경향이 있습니다.

#### 설계

```
현재: 모든 고점 상승 AND 모든 저점 상승 → uptrend
보완: 70% 이상 일관 → uptrend (다수결)
```

또한 **단기 micro_structure** 필드(최근 2피봇 기준)를 추가해 장기 구조와 단기 흐름을 구분합니다.

#### 구현

```python
def _analyze_structure(self) -> str:
    if len(self._all_swings) < 4:
        return "unknown"
    lookback = int(getattr(self.config, "structure_lookback_swings", 8) or 8)
    points   = int(getattr(self.config, "structure_points", 3) or 3)

    rh = [s.price for s in self._all_swings[-lookback:]
          if s.swing_type == SwingType.HIGH][-points:]
    rl = [s.price for s in self._all_swings[-lookback:]
          if s.swing_type == SwingType.LOW][-points:]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"

    n = len(rh) - 1
    # [보완] 다수결: 70% 이상 일관
    hh_score = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i-1])
    hl_score = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i-1])
    lh_score = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i-1])
    ll_score = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i-1])

    threshold = 0.7   # 70% 다수결
    if hh_score >= n * threshold and hl_score >= n * threshold:
        return "uptrend"
    if lh_score >= n * threshold and ll_score >= n * threshold:
        return "downtrend"
    return "ranging"

# ZigZagState에 micro_structure 추가 (최근 2피봇 기준)
def _analyze_micro_structure(self) -> str:
    """최근 2피봇 기준 단기 구조 (빠른 방향 판단용)."""
    rh = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.HIGH]
    rl = [s.price for s in self._all_swings[-4:]
          if s.swing_type == SwingType.LOW]
    if len(rh) < 2 or len(rl) < 2:
        return "unknown"
    hh = rh[-1] > rh[-2]
    hl = rl[-1] > rl[-2]
    lh = rh[-1] < rh[-2]
    ll = rl[-1] < rl[-2]
    if hh and hl: return "uptrend"
    if lh and ll: return "downtrend"
    return "ranging"
```

#### ZigZagState 추가 필드

```python
micro_structure: str = "unknown"   # 최근 2피봇 기반 단기 구조
structure_confidence: float = 0.0  # 구조 판정 신뢰도 (0~1)
```

#### 기대 효과

- 오늘 로그 기준: 09:30~10:01 구간에서 `downtrend` 조기 판정 가능
- `ranging` 과잉 판정 50~60% 감소 예상

---

### 3-4. 피봇 클러스터링 실제 적용 🟡 단기

#### 문제

`cluster_tolerance_pct=0.3` 설정이 Config에 있지만 `_add_swing()`에서 **실제로 사용되지 않습니다.** 동일 가격대 피봇이 반복 등록되어 S/R 레벨에 노이즈가 포함됩니다.

오늘 로그: 09:26 H(6478.40)와 10:34 H(6476.72) — 차이 0.026%, 사실상 동일 저항대.

#### 설계

```
신규 피봇과 직전 동일 유형 피봇의 가격 차이가
cluster_tolerance_pct(0.3%) 이내이면:
  - 신규가 더 극값 → 기존 피봇을 신규로 교체(갱신)
  - 신규가 덜 극값 → 신규 피봇 무시 (기존 유지)
```

#### 구현

```python
def _add_swing(
    self, idx, price, swing_type, atr, *,
    confirmed_at_idx=None, confirmed_close=0.0
) -> None:
    cfg = self.config
    cluster_tol = float(getattr(cfg, "cluster_tolerance_pct", 0.3) or 0.0)

    # 직전 동일 유형 확정 피봇 찾기
    prev = next((s for s in reversed(self._all_swings)
                 if s.swing_type == swing_type and s.confirmed), None)

    if prev is not None and cluster_tol > 0:
        dist_pct = abs(price - prev.price) / prev.price * 100.0
        if dist_pct <= cluster_tol:
            # 클러스터 범위 내
            is_more_extreme = (
                (swing_type == SwingType.HIGH and price > prev.price) or
                (swing_type == SwingType.LOW  and price < prev.price)
            )
            if is_more_extreme:
                # 더 극값 → 기존 교체
                self._all_swings[-1] = SwingPoint(
                    index=idx, price=price,
                    swing_type=swing_type,
                    atr_at_swing=atr,
                    is_major=prev.is_major,
                    confirmed=True,
                    confirmed_at_idx=int(self._bar_idx if confirmed_at_idx is None
                                        else confirmed_at_idx),
                    confirmed_close=confirmed_close,
                )
                self._pivot_event_emit(
                    "클러스터교체",
                    close=float(confirmed_close),
                    prev_price=round(prev.price, 4),
                    new_price=round(price, 4),
                    dist_pct=round(dist_pct, 4),
                )
            else:
                # 덜 극값 → 무시
                self._pivot_event_emit(
                    "클러스터무시",
                    close=float(confirmed_close),
                    prev_price=round(prev.price, 4),
                    new_price=round(price, 4),
                    dist_pct=round(dist_pct, 4),
                )
            return  # 새 항목 추가하지 않음

    # 클러스터 아닌 경우 기존 로직
    prev_any = next((s for s in reversed(self._all_swings)
                     if s.swing_type == swing_type), None)
    is_major = (True if prev_any is None
                else abs(price - prev_any.price) >= atr * cfg.major_swing_ratio)
    c_idx = int(self._bar_idx if confirmed_at_idx is None else confirmed_at_idx)
    self._all_swings.append(SwingPoint(
        index=idx, price=price, swing_type=swing_type,
        atr_at_swing=atr, is_major=is_major,
        confirmed=True, confirmed_at_idx=c_idx,
        confirmed_close=confirmed_close,
    ))
    if len(self._all_swings) > cfg.max_swings * 2:
        self._all_swings = self._all_swings[-cfg.max_swings:]
```

#### 기대 효과

- S/R 레벨 중복 제거 → `_find_nearest_sr()` 정확도 향상
- 누적 피봇 수 감소 → Fibonacci 계산 기준 안정화

---

### 3-5. bars_since 기반 임계값 decay 🟡 단기

#### 문제

마지막 피봇 이후 봉 수(`bars_since_last_swing`)가 누적되어도 임계값이 변하지 않습니다. 오늘 11:12→13:10 118분(118봉) 공백처럼 장시간 무피봇 구간에서 변동성이 실제로 줄어들어 기존 ATR이 과대 추정되면 피봇 감지 자체가 억제됩니다.

#### 설계

```
bars_since_last_swing > decay_start_bars(30봉) 이상 → 임계값 점진 완화
decay_rate = 0.005% / 봉  (30봉 초과분)
최대 완화폭 = 0.3%
```

#### 구현

```python
# _calc_threshold_pct() 말미에 추가
def _calc_threshold_pct(self, atr: float, close: float) -> float:
    # ... 기존 로직 ...
    base = float(np.clip(base, cfg.pivot_threshold_min_pct, cfg.pivot_threshold_max_pct))

    # [보완] bars_since decay: 장시간 무피봇 구간 감도 향상
    decay_start = int(getattr(cfg, "decay_start_bars", 30) or 30)
    decay_rate  = float(getattr(cfg, "decay_rate_per_bar", 0.005) or 0.005)
    decay_max   = float(getattr(cfg, "decay_max_pct", 0.3) or 0.3)

    bars_since = max(0, self._bar_idx - self._last_confirmed_bar_idx) \
                 if self._last_confirmed_bar_idx >= 0 else 0
    if bars_since > decay_start:
        excess = bars_since - decay_start
        decay  = min(decay_max, excess * decay_rate)
        base   = max(cfg.pivot_threshold_min_pct, base - decay)

    return float(base)
```

#### Config 추가

```python
decay_start_bars: int   = 30     # decay 시작 봉 수
decay_rate_per_bar: float = 0.005  # 봉당 감소율 (%)
decay_max_pct: float  = 0.3      # 최대 감소폭 (%)
```

#### 기대 효과

- 118봉 공백 시 decay = min(0.3, (118-30)×0.005) = 0.3% → 임계값 0.30% - 0.30% = 0.0% → `pivot_threshold_min_pct`에 의해 하한 유지
- 장시간 횡보 후 대파동 피봇 조기 감지

---

### 3-6. pending 피봇을 잠정 S/R에 포함 🟡 단기

#### 문제

`_find_nearest_sr()`이 `confirmed=True` 피봇만 사용합니다. pending 중인 후보는 아직 확정되지 않았지만, 이미 가격이 해당 수준에 도달했다는 증거입니다. LLM 컨텍스트와 Transformer feature에서 S/R이 한 봉 늦게 반영됩니다.

#### 구현

```python
def _find_nearest_sr(self, close: float) -> Tuple[float, float]:
    swings = [s for s in self._all_swings if s.confirmed]

    # [보완] pending 후보를 잠정 S/R로 포함 (낮은 가중치)
    if isinstance(self._pending_confirm, dict) and self._pending_confirm:
        pc_type  = self._pending_confirm.get("type")
        pc_price = float(self._pending_confirm.get("price") or 0.0)
        if pc_type == "high" and pc_price > 0:
            highs = [s.price for s in swings
                     if s.swing_type == SwingType.HIGH and s.price > close]
            highs.append(pc_price)   # 잠정 저항 추가
        elif pc_type == "low" and pc_price > 0:
            lows = [s.price for s in swings
                    if s.swing_type == SwingType.LOW and s.price < close]
            lows.append(pc_price)    # 잠정 지지 추가

    highs = [s.price for s in swings
             if s.swing_type == SwingType.HIGH and s.price > close]
    lows  = [s.price for s in swings
             if s.swing_type == SwingType.LOW  and s.price < close]
    return (max(lows) if lows else 0.0), (min(highs) if highs else 0.0)
```

#### ZigZagState 추가 필드

```python
pending_resistance: float = 0.0   # 잠정 저항 (pending high 후보)
pending_support: float = 0.0      # 잠정 지지 (pending low 후보)
```

#### 기대 효과

- S/R 레벨 선행성 1봉 개선
- LLM 컨텍스트에서 "피봇 후보 진행 중" 정보 활용 가능

---

### 3-7. is_major 파동 비율 기준 🟢 중기

#### 문제

`is_major = abs(price - prev.price) >= atr * major_swing_ratio(2.0)` — ATR이 장세에 따라 변동하므로 동일한 파동도 ATR이 낮을 때는 major, 높을 때는 minor로 불일치하게 판정됩니다.

#### 설계

```
ATR 배수 방식 → 직전 N파동 평균 대비 비율 방식
is_major: 직전 3개 파동 평균의 1.5배 이상 → major
```

#### 구현

```python
def _calc_avg_wave_size(self, n: int = 3) -> float:
    """최근 N개 파동 평균 크기 계산."""
    confirmed = [s for s in self._all_swings if s.confirmed]
    if len(confirmed) < 2:
        return 0.0
    sizes = []
    for i in range(1, min(n + 1, len(confirmed))):
        sizes.append(abs(confirmed[-i].price - confirmed[-i-1].price))
    return sum(sizes) / len(sizes) if sizes else 0.0

# _add_swing() 에서
avg_wave = self._calc_avg_wave_size(n=3)
if avg_wave > 0:
    is_major = abs(price - (prev.price if prev else price)) >= avg_wave * 1.5
else:
    # fallback: 기존 ATR 기반
    is_major = (True if prev is None
                else abs(price - prev.price) >= atr * cfg.major_swing_ratio)
```

#### Config 추가

```python
major_wave_ratio: float = 1.5     # 평균 파동 대비 major 판정 배수
major_wave_lookback: int = 3      # 평균 계산에 사용할 이전 파동 수
```

#### 기대 효과

- 장세 변동에 무관하게 일관된 major/minor 판정
- 장시간 횡보 후 대파동을 major로 정확히 인식

---

### 3-8. 방향 ER (Directional ER) 🟢 중기

#### 문제

현재 ER(Efficiency Ratio)은 방향 강도만 측정하고 상승/하락 방향을 반영하지 않습니다. 결과적으로 상승 추세에서 고점 후보(H)를 더 엄격하게 걸러야 하는 상황에서도 동일한 임계값이 적용됩니다.

#### 설계

```
방향 ER (DER) = ER × direction_sign

direction_sign: +1(상승), -1(하락)
DER 범위: -1.0 ~ +1.0

적용:
  current_direction = +1 (상승) 중 high 후보 등록:
    DER > 0 → 상승 추세 확인 → 임계값 유지 (정방향)
    DER < 0 → 추세 약화 → 임계값 완화 (조기 전환 감지)

  current_direction = -1 (하락) 중 low 후보 등록:
    DER < 0 → 하락 추세 확인 → 임계값 유지
    DER > 0 → 추세 약화 → 임계값 완화
```

#### 구현

```python
def _calc_der(self) -> float:
    """방향 Efficiency Ratio (-1.0 ~ +1.0)."""
    er = self._calc_er()
    n = len(self._closes)
    if n < 2:
        return 0.0
    direction_sign = 1.0 if float(self._closes[-1]) >= float(self._closes[0]) else -1.0
    return float(er * direction_sign)

# _calc_threshold_pct() 에서 추가 활용
def _calc_threshold_pct(self, atr: float, close: float) -> float:
    er  = self._calc_er()
    der = self._calc_der()

    # 방향 불일치 시 임계값 완화 (조기 전환 감지)
    direction_mismatch = (
        (self._current_direction == 1  and der < -0.3) or
        (self._current_direction == -1 and der >  0.3)
    )
    if direction_mismatch:
        mmax = mmax * 0.7   # 임계값 30% 완화
    # ... 기존 로직 ...
```

#### 기대 효과

- 추세 전환 조기 감지 (방향 불일치 구간에서 임계값 자동 완화)
- 오늘 로그 10:01~10:30 구간: 하락 DER 감지 → 저점 피봇 조기 등록 가능

---

### 3-9. 시간대별 min_wave_bars 테이블 (SESSION-MW) 🔴 즉시

#### 문제

`min_wave_bars` 단일 전역값으로는 **장중 시간대별로 다른 변동성 특성**에 대응할 수 없습니다.

- 장초반(09:00~09:30): 갭·급등락으로 노이즈 피봇이 빈발 → **엄격하게** (봉 수 많이)
- 중반(10:30~14:30): 완만한 흐름 → **느슨하게** (봉 수 적게)
- 장마감(14:30~15:20): 포지션 정리로 변동성 재상승 → **다시 조임**
- 동시호가(15:20~15:30): 방향성 없는 급변 → **매우 엄격하게**

또한 기존 `early_session_*` 필드는 `_calc_threshold_pct()` 내부의 **ATR multiplier 상향 전용**이므로, 봉 수 기반 필터링은 완전히 별개 레이어가 필요합니다.

```
레이어 구분
─────────────────────────────────────────────────────────
early_session_*            _calc_threshold_pct()
  → ATR mmax 상향          → 임계값 자체를 크게 (후보 등록 억제)

session_min_wave_bars_table  _is_wave_length_ok()
  → 확정 간 봉 수 강제      → 등록된 후보라도 간격 짧으면 차단
─────────────────────────────────────────────────────────
```

두 레이어는 **독립적으로 동시에 작동**하며 상호 대체 관계가 아닙니다.

#### 설계

```
시간대           min_wave_bars   근거
──────────────────────────────────────────────────────────
09:00 ~ 09:30       10          장초반 노이즈 강 억제
09:30 ~ 10:30        7          오전장 중간 억제
10:30 ~ 14:30        4          중반 느슨 (추세 포착 우선)
14:30 ~ 15:20        7          장마감 전 재조임
15:20 ~ 15:31       10          동시호가 강 억제
테이블 미해당         min_wave_bars(전역값) 폴백
```

처리 규칙:
- 테이블 항목을 **순서대로** 평가, 첫 번째 일치 구간 적용
- 최종 min_bars = `max(전역 min_wave_bars, 테이블 값)` (테이블이 전역값보다 낮아지는 것 방지)
- `session_min_wave_bars_table = []` (빈 리스트) → 기존 `min_wave_bars` 단일값 동작 유지 (하위 호환)

#### 구현

**`adaptive_zigzag.py` — `AdaptiveZigZagConfig` 필드 추가:**

```python
# [SESSION-MW] 시간대별 min_wave_bars 테이블
# 빈 리스트(기본값) → min_wave_bars 단일값 폴백 (하위 호환)
# JSON 형식: [["HH:MM", "HH:MM", bars], ...]
session_min_wave_bars_table: List[Tuple[str, str, int]] = field(
    default_factory=list
)
```

**`adaptive_zigzag.py` — 헬퍼 메서드 신규 추가:**

```python
def _get_session_min_wave_bars(self) -> int:
    """[SESSION-MW] 현재 봉 시각에 맞는 min_wave_bars 반환.

    session_min_wave_bars_table이 설정된 경우 테이블을 순서대로 평가해
    처음 일치하는 구간의 값을 반환한다.
    테이블이 비어 있거나 해당 구간이 없으면 config.min_wave_bars 폴백.
    """
    cfg = self.config
    base = int(getattr(cfg, "min_wave_bars", 0) or 0)
    table = getattr(cfg, "session_min_wave_bars_table", None) or []
    if not table:
        return base
    current_time = self._bar_hhmm(self._bar_idx - 1)
    if not current_time:
        return base
    for start, end, bars in table:
        try:
            if str(start) <= current_time < str(end):
                return max(base, int(bars))
        except Exception:
            continue
    return base
```

**`adaptive_zigzag.py` — `_is_wave_length_ok()` 수정:**

```python
def _is_wave_length_ok(self, thr_abs: float, close: float) -> bool:
    cfg = self.config
    # [SESSION-MW] 시간대별 min_wave_bars 테이블 적용 (기존 단일값 대체)
    min_bars = self._get_session_min_wave_bars()
    if min_bars > 0 and self._last_confirmed_bar_idx >= 0:
        if (self._bar_idx - self._last_confirmed_bar_idx) < min_bars:
            return False
    # 이하 min_wave_pct 등 기존 로직 유지 ...
```

**`config.py` — `AdaptiveZigZagSettings` 필드 추가:**

```python
# [SESSION-MW] 시간대별 min_wave_bars 테이블
# 빈 리스트 → 단일 min_wave_bars 폴백 (하위 호환)
# JSON 형식: [["09:00", "09:30", 10], ...]
session_min_wave_bars_table: list = field(default_factory=list)
```

**`config.py` — `AppConfig._from_dict()` 파싱 추가:**

```python
zigzag=AdaptiveZigZagSettings(
    # ... 기존 파라미터 ...
    session_min_wave_bars_table=cls._parse_session_min_wave_bars_table(
        zz_data.get("session_min_wave_bars_table")
    ),
),
```

**`config.py` — 정적 파싱 메서드 신규 추가:**

```python
@staticmethod
def _parse_session_min_wave_bars_table(raw: Any) -> list:
    """[SESSION-MW] JSON → List[Tuple[str, str, int]] 변환.
    항목 형식 오류 / 음수 bars → 해당 항목 skip + WARNING 로깅."""
    if not raw:
        return []
    result = []
    for i, item in enumerate(raw):
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                raise ValueError(f"항목 형식 오류")
            start, end, bars = str(item[0]).strip(), str(item[1]).strip(), int(item[2])
            if bars < 0:
                raise ValueError(f"min_wave_bars 음수 불가: {bars}")
            result.append((start, end, bars))
        except Exception as e:
            logger.warning("[SESSION-MW] table[%d] 파싱 실패: %s", i, e)
    return result
```

**`config.py` — 모듈 레벨 공개 함수 추가 (pipeline.py / data_builder.py용):**

```python
def parse_session_min_wave_bars_table(raw: Any) -> list:
    """AppConfig._parse_session_min_wave_bars_table()의 모듈 레벨 래퍼."""
    return AppConfig._parse_session_min_wave_bars_table(raw)
```

**`prediction/pipeline.py` / `data_builder.py` — import 및 생성 블록 수정:**

```python
# import 추가
from config import parse_session_min_wave_bars_table as _parse_session_mwb_table

# AdaptiveZigZagConfig(...) 생성 블록 내
session_min_wave_bars_table=_parse_session_mwb_table(
    zz.get("session_min_wave_bars_table")
),
```

#### config.json 설정

```json
"adaptive_indicator": {
  "zigzag": {
    "min_wave_bars": 3,
    "session_min_wave_bars_table": [
      ["09:00", "09:30", 10],
      ["09:30", "10:30",  7],
      ["10:30", "14:30",  4],
      ["14:30", "15:20",  7],
      ["15:20", "15:31", 10]
    ]
  }
}
```

> **`early_session_*` 제거 여부:** `early_session_start_time`, `early_session_end_time`, `early_session_atr_multiplier_max`는 `_calc_threshold_pct()`의 **ATR mmax 상향 전용**으로 `session_min_wave_bars_table`과 레이어가 다릅니다. **제거하지 않습니다.**

#### 기대 효과

- 실측 로그 기준 09:21 L·09:26 H·09:30 L (1~5분 간격 장초반 피봇 3건): `min_wave_bars=10` 적용 시 간격 미달로 차단
- 시간대별 조정으로 장중 전체 피봇 밀도 균형화
- 기존 `min_wave_bars` 단일값 설정과 완전 하위 호환

---

## 4. 구현 우선순위 요약

| 순위 | 항목 | 복잡도 | 기대 효과 | 파일 위치 |
|---|---|---|---|---|
| 🔴 즉시 | 3-1. confirmation_bars 동적 조절 | 중 | ranging 허위 피봇 30~40% 감소 | `_calc_confirmation_bars()` 신규 |
| 🔴 즉시 | 3-2. min_wave_pct = 0.25% | 낮 | 경계 잡음 피봇 차단 | `_is_wave_length_ok()` 수정 |
| 🔴 즉시 | 3-3. structure 다수결 70% | 낮 | uptrend/downtrend 과소 판정 해소 | `_analyze_structure()` 수정 |
| 🔴 즉시 | 3-9. 시간대별 min_wave_bars 테이블 | 낮 | 장중 피봇 밀도 시간대별 균형화 | `_get_session_min_wave_bars()` 신규, `config.py` 파싱 추가 |
| 🟡 단기 | 3-4. 클러스터링 실제 적용 | 중 | S/R 정확도 향상 | `_add_swing()` 수정 |
| 🟡 단기 | 3-5. bars_since decay | 낮 | 장시간 공백 감도 향상 | `_calc_threshold_pct()` 수정 |
| 🟡 단기 | 3-6. pending S/R 포함 | 낮 | S/R 선행성 1봉 개선 | `_find_nearest_sr()` 수정 |
| 🟢 중기 | 3-7. is_major 파동 비율 | 중 | 장세 무관 일관 판정 | `_add_swing()` + `_calc_avg_wave_size()` |
| 🟢 중기 | 3-8. 방향 ER (DER) | 중 | 추세 전환 조기 감지 | `_calc_der()` 신규 |

---

## 5. Config 변경 목록

기존 `AdaptiveZigZagConfig`에 추가되는 파라미터 목록입니다.

```python
@dataclass
class AdaptiveZigZagConfig:
    # ... 기존 파라미터 ...

    # [3-1] confirmation_bars 동적 조절
    confirmation_bars_ranging: int   = 2      # ranging 구간
    confirmation_bars_unknown: int   = 3      # unknown 구간

    # [3-2] min_wave_pct 기본값 변경
    min_wave_pct: float = 0.25               # 0.0 → 0.25

    # [3-3] structure 다수결 임계값
    structure_majority_threshold: float = 0.7  # 70% 이상 일관 → 추세 판정

    # [3-5] bars_since decay
    decay_start_bars: int    = 30
    decay_rate_per_bar: float = 0.005
    decay_max_pct: float     = 0.3

    # [3-7] is_major 파동 비율
    major_wave_ratio: float  = 1.5
    major_wave_lookback: int = 3

    # [3-9] 시간대별 min_wave_bars 테이블 (SESSION-MW)
    # 빈 리스트(기본값) → min_wave_bars 단일값 폴백 (하위 호환)
    session_min_wave_bars_table: List[Tuple[str, str, int]] = field(
        default_factory=list
    )
```

### config.json 키 전체 목록

`adaptive_indicator.zigzag` 섹션에 추가되는 키 목록입니다.

```json
{
  "adaptive_indicator": {
    "zigzag": {
      "confirmation_bars_ranging": 2,
      "confirmation_bars_unknown": 3,
      "min_wave_pct": 0.25,
      "structure_majority_threshold": 0.7,
      "decay_start_bars": 30,
      "decay_rate_per_bar": 0.005,
      "decay_max_pct": 0.3,
      "major_wave_ratio": 1.5,
      "major_wave_lookback": 3,
      "session_min_wave_bars_table": [
        ["09:00", "09:30", 10],
        ["09:30", "10:30",  7],
        ["10:30", "14:30",  4],
        ["14:30", "15:20",  7],
        ["15:20", "15:31", 10]
      ]
    }
  }
}
```

### 기존 파라미터 기본값 변경

| 파라미터 | 기존 | 변경 | 근거 |
|---|---|---|---|
| `min_wave_pct` | `0.0` | `0.25` | 경계 잡음 피봇 차단 |

### 관련 파일 변경 목록 (3-9 SESSION-MW)

| 파일 | 변경 내용 |
|---|---|
| `kospi_indicators/adaptive_zigzag.py` | `AdaptiveZigZagConfig`에 `session_min_wave_bars_table` 필드 추가, `_get_session_min_wave_bars()` 신규, `_is_wave_length_ok()` 수정 |
| `config.py` | `AdaptiveZigZagSettings`에 필드 추가, `_from_dict()` 파싱 추가, `_parse_session_min_wave_bars_table()` 정적 메서드 신규, `parse_session_min_wave_bars_table()` 모듈 레벨 공개 함수 신규 |
| `prediction/pipeline.py` | `_parse_session_mwb_table` import 추가, `AdaptiveZigZagConfig(...)` 생성 블록에 파라미터 추가 |
| `prediction/data_builder.py` | 동일 |
| `config.json` | `adaptive_indicator.zigzag.session_min_wave_bars_table` 키 추가 |

> **하위호환:** 모든 신규 파라미터는 기존 동작과 동일한 기본값 또는 비활성(0, False, 빈 리스트) 기본값으로 설정합니다. 기존 설정 파일 변경 없이 기존 동작을 유지할 수 있습니다.

---

*문서 끝 — rev.2 (2026-04-24): 3-9 SESSION-MW 추가, 목차·파라미터표·우선순위표 보완*
