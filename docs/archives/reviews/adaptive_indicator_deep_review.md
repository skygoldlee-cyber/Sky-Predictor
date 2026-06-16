# Adaptive Indicator 심층 코드 리뷰

**Transformer vs SkyEbest — 2차 정밀 분석**  
`AdaptiveSuperTrend` · `AdaptiveZigZag` · `WilderRMA` · Integration Layer

---

## 0. 분석 범위 및 구조

이 리뷰는 1차 분석 이후 두 프로젝트의 전체 코드를 재검토하여 새로운 버그, 알고리즘 논리 오류, 아키텍처 차이를 추가로 발굴한 결과입니다. 특히 WilderRMA 공유 컴포넌트, ZigZag ER 방향 로직, `calculate()` 메서드 버그, SkyEbest 전용 래퍼 불일치를 집중 검토했습니다.

| 항목 | Transformer | SkyEbest |
|---|---|---|
| WilderRMA 위치 | `adaptive_indicator/wilder_smooth.py` (전용 모듈) | `UnifiedTA.py` 내부 인라인 정의 |
| 통합 레이어 | `AdaptiveIndicatorManager` (cross_features 포함) | 없음 (각 지표 독립 사용) |
| 배치 처리 | `compute_from_df()` (소문자 컬럼) | `compute_from_df()` + `calculate()` (대문자 컬럼) |
| 래퍼 메서드 | 없음 | `get_super_trend()` / `get_super_trend_enhanced()` 분리 |
| 테스트 코드 | `tests/` 디렉토리 존재 | 없음 |

---

## 1. WilderRMA — `ready` 속성 1봉 지연 버그

**발견 위치: SkyEbest `UnifiedTA.py` line 173**

두 구현의 `WilderRMA.ready` 속성 조건이 다릅니다. 이 차이는 SuperTrend의 ADX 워밍업 기간에 직접 영향을 줍니다.

| 구현 | ready 조건 | 의미 |
|---|---|---|
| Transformer | `return count >= period` | 14봉째부터 `ready=True` |
| SkyEbest | `return count > period` | 15봉째부터 `ready=True` (**1봉 지연**) |

### 영향 경로

- SkyEbest `SuperTrend._calc_adx()`: `rma_tr.ready` 체크 → `False`이면 `return 25.0`
- 결과: SkyEbest에서 ADX 워밍업이 14봉이 아닌 **15봉** 동안 지속됨
- 이 기간 동안 `adaptive_mult` 계산이 고정값(`adx=25.0` 기준)으로 수행됨
- Transformer는 14봉째에 정확히 RMA 스위칭 → 1봉 빠른 적응 시작

> 🔴 **즉시 수정 필요 (SkyEbest)**
>
> `UnifiedTA.py` line 173:
> ```python
> # 현재 (오류)
> return self.count > self.period
>
> # 수정
> return self.count >= self.period
> ```
> 영향: ADX 초기화가 1봉 빨라지고, Transformer와 ADX 동작이 동기화됨

---

## 2. SuperTrend — `bars_in_trend` 플립 봉 누적 버그

**발견 위치: Transformer `adaptive_supertrend.py`**

SuperTrend 방향 전환(플립) 시 `bars_in_trend` 카운터 처리 방식이 두 구현에서 다릅니다. SkyEbest가 올바른 구현입니다.

### Transformer (버그 있음)

```python
if prev_dir == -1 and direction == 1:
    signal = "buy"
    self._state.bars_in_trend = 0   # ① 리셋
...
self._state.bars_in_trend += 1      # ② 무조건 +1 → 플립 봉에서 1이 됨
```

### SkyEbest (올바름)

```python
just_flipped = False
if pd_ == -1 and direction == 1:
    signal = "buy"
    self._state.bars_in_trend = 0
    just_flipped = True

if not just_flipped:
    self._state.bars_in_trend += 1  # 플립 봉에서는 스킵 → 0 유지
```

플립 봉에서 `bars_in_trend=0`을 외부에 정확히 노출해야 Transformer의 `ast_trend_duration` feature가 올바른 값(0)을 가집니다. Transformer에서는 플립 봉에서 이미 1로 증가하므로 feature가 항상 최소 1 이상입니다.

> 🔴 **즉시 수정 필요 (Transformer)**
>
> `adaptive_supertrend.py` `update()` 메서드에서 `just_flipped` 패턴 적용:
> ```python
> just_flipped = False  # 신호 생성 직전에 초기화
>
> if prev_dir is not None:
>     if int(prev_dir) == -1 and int(direction) == 1:
>         signal = "buy"
>         self._state.last_flip_price = close
>         self._state.bars_in_trend = 0
>         just_flipped = True
>     elif int(prev_dir) == 1 and int(direction) == -1:
>         signal = "sell"
>         self._state.last_flip_price = close
>         self._state.bars_in_trend = 0
>         just_flipped = True
>
> if not just_flipped:
>     self._state.bars_in_trend += 1
> ```

---

## 3. ZigZag — ER-adaptive Threshold 방향 논리 오류

**발견 위치: 양쪽 모두 동일한 논리 오류**

`AdaptiveZigZag._calc_threshold_pct()`에서 Efficiency Ratio(ER)에 따른 ATR 배수(multiplier) 결정 방향이 의도와 **반대**로 되어 있습니다. 이 버그는 Transformer와 SkyEbest 양쪽에 동일하게 존재합니다.

### 현재 코드

```python
mult = float(mmax - er * (mmax - mmin))
# mmax=4.0, mmin=1.0 기본값 기준:
# ER=1.0 (완전 추세) → mult = 4.0 - 1.0*3.0 = 1.0  (threshold 최소 → 스윙 많음)
# ER=0.0 (완전 횡보) → mult = 4.0 - 0.0*3.0 = 4.0  (threshold 최대 → 스윙 적음)
```

| ER 값 | 현재 mult | 현재 threshold 결과 | 올바른 동작이어야 할 것 |
|---|---|---|---|
| 1.0 (강한 추세) | 1.0 | 최소 → 민감한 스윙 감지 | 추세 노이즈 필터 → 큰 임계값 필요 |
| 0.0 (횡보) | 4.0 | 최대 → 둔감한 스윙 감지 | 잦은 스윙 감지 → 작은 임계값 필요 |

현재 로직은 **강한 추세일 때 스윙 포인트를 더 자주 찍고, 횡보일 때는 거의 찍지 않는** 역설적 동작을 합니다.

SuperTrend에서 ER 활용 방향(`ER↑ → ATR 기간↓ → 빠른 반응`)은 올바릅니다. ZigZag에서는 `ER↑`일 때 **큰 움직임만** 스윙으로 인식해 추세 중 노이즈를 필터링해야 합니다.

### 올바른 수정 코드

```python
# 현재 (잘못됨)
mult = float(mmax - er * (mmax - mmin))  # ER↑ → mult↓ → threshold↓

# 수정 후 (올바름)
mult = float(mmin + er * (mmax - mmin))  # ER↑ → mult↑ → threshold↑
```

수정 후 동작:

| ER | mult | threshold | 해석 |
|---|---|---|---|
| 1.0 (강한 추세) | 4.0 | 큼 | 큰 되돌림만 스윙 → 추세 방향성 보존 |
| 0.5 (중립) | 2.5 | 중간 | 중간 수준 스윙 감지 |
| 0.0 (횡보) | 1.0 | 작음 | 작은 움직임도 스윙 → 지지/저항 세밀 감지 |

> 🔴 **즉시 수정 필요 (양쪽 모두)**
>
> **Transformer** `adaptive_zigzag.py` line 807:
> ```python
> mult = float(mmax - er * (mmax - mmin))
> # →
> mult = float(mmin + er * (mmax - mmin))
> ```
>
> **SkyEbest** `UnifiedTA.py` line 1477 (동일 패턴):
> ```python
> mult = float(mmax - er * (mmax - mmin))
> # →
> mult = float(mmin + er * (mmax - mmin))
> ```
>
> ⚠️ 이 변경은 스윙 포인트 빈도와 지지/저항 정확도에 상당한 영향을 줌. **변경 후 백테스트 검증 필수**

---

## 4. ZigZag — `pending_confirm` 교체 로직 불일치

**발견 위치: SkyEbest `UnifiedTA.py` ZigZag direction 전환 블록**

빠른 연속 방향 전환이 발생할 때의 처리 방식이 다릅니다.

| 구현 | `pending_confirm` 이미 존재 시 | 결과 |
|---|---|---|
| Transformer | 반대 타입(high↔low)이면 교체 / 같은 타입이면 유지 | 연속 반전 시에도 최적의 피크 포착 |
| SkyEbest | 이미 존재하면 무조건 스킵 | 두 번째 이후 전환 신호 누락 가능 |

### 시나리오: 상승→하락→상승 빠른 전환

- 봉 A: 상승 중 하락 전환 → `pending_confirm={"type":"high"}` 등록
- 봉 B: `confirmation_bars` 만료 전에 다시 상승 전환 조건 충족
  - **Transformer**: type이 `"high"→"low"` 반대 → 교체 → 저점 스윙 등록
  - **SkyEbest**: `pending_confirm` 이미 존재 → 스킵 → **저점 스윙 누락**

> 🟠 **권고 (SkyEbest)**
>
> `UnifiedTA.py` ZigZag `update()` direction==1 블록:
> ```python
> # 현재
> if self._pending_confirm is None:
>     self._pending_confirm = dict(type="high", ...)
>
> # 수정
> if self._pending_confirm is None or self._pending_confirm.get("type") != "high":
>     self._pending_confirm = dict(type="high", ...)
> ```
>
> direction==-1 블록도 동일하게 `"low"` 기준으로 수정

---

## 5. ZigZag `calculate()` — `_all_swings` 크기 감소 분기 논리 오류

**발견 위치: SkyEbest `UnifiedTA.py` line 1195–1199**

`calculate()` 메서드에서 `_add_swing()` 이후 `_all_swings` 크기가 줄어드는 경우를 처리하는 분기가 있으나, 동작이 모호합니다.

```python
before_len = len(self._all_swings)
self.update(...)               # _add_swing() 내부에서 del 발생 가능
after_len = len(self._all_swings)

if after_len > before_len:
    new_swings = self._all_swings[before_len:]   # 새로 추가된 것 (정상)
elif after_len < before_len:
    new_swings = self._all_swings[-1:]           # ← 마지막 1개만 처리 (모호)
```

SkyEbest `_add_swing()`은 `del self._all_swings[...]`로 앞부분을 삭제합니다. 이때 `after_len < before_len`이 발생하며, 실제 추가된 스윙(마지막 원소)과 삭제된 스윙이 동시에 존재합니다.

반면 Transformer는 `self._all_swings = self._all_swings[-n:]`로 새 리스트를 할당하므로 `before_len`이 old 리스트를 참조하여 `after_len < before_len` 자체가 발생하지 않습니다. (Transformer에는 `calculate()` 메서드가 없으므로 이 문제를 애초에 피합니다.)

> 🟠 **권고 (SkyEbest)**
>
> `_add_swing()` 정리 방식을 `del` → 슬라이싱 재할당으로 변경:
> ```python
> # 현재
> del self._all_swings[:len(self._all_swings) - cfg.max_swings]
>
> # 수정
> self._all_swings = self._all_swings[-cfg.max_swings:]
> ```
>
> 또는 `calculate()`의 `elif` 분기를 단순화:
> ```python
> if after_len != before_len:
>     new_swings = [self._all_swings[-1]] if self._all_swings else []
> ```

---

## 6. SuperTrend — LLM `advice` 딕셔너리 키 오류

**발견 위치: SkyEbest `UnifiedTA.py` `get_llm_context()`**

```python
advice = {
    "uptrend":   "상승 구조 유지 중 — 매도 신호에 신중하세요.",
    "downtrend": "하락 구조 유지 중 — 매수 신호에 신중하세요.",
}.get(s.trend_strength, "횡보 구조 — ...")   # ← 여기가 문제
```

`s.trend_strength`는 `"weak"` / `"neutral"` / `"strong"` 중 하나입니다. 딕셔너리 키 `"uptrend"` / `"downtrend"`와 절대 일치하지 않으므로, **항상 기본값(횡보 문구)** 이 반환됩니다.

> 🔴 **즉시 수정 필요 (SkyEbest)**
>
> ```python
> # 방법 1: s.direction 기반으로 변경
> advice = {
>     1:  "상승 추세 유지 중 — 매도 신호에 신중하세요.",
>     -1: "하락 추세 유지 중 — 매수 신호에 신중하세요.",
> }.get(s.direction, "횡보 구조 — 지지/저항 범위 매매가 유리할 수 있습니다.")
>
> # 방법 2: s.trend_strength 기반 조언으로 교체
> advice = {
>     "strong":  "강한 추세 — 추세 추종 전략이 유효합니다.",
>     "neutral": "중간 추세 — 신호 확인 후 진입하세요.",
>     "weak":    "약한 추세 — 역추세 주의, 횡보 가능성.",
> }.get(s.trend_strength, "분석 불충분")
> ```

---

## 7. SuperTrend — ATR 재초기화 임계값 불일치

**발견 위치: 두 프로젝트 `update()` ATR 계산 블록**

| 구현 | 재초기화 조건 | `atr_max=21` 기준 예시 | 문제점 |
|---|---|---|---|
| Transformer | `period_change > atr_max * 0.3` | 변화량 > 6.3봉이면 재초기화 | `atr_max` 설정 변경 시 민감도 달라짐 |
| SkyEbest | `period_change_ratio > 0.5` | 이전 대비 50% 이상 변화 시 | 기간 범위와 무관하게 일관적 |

예시: ATR 기간이 21→7로 급변할 경우, 두 구현 모두 재초기화를 실행합니다. 그러나 `atr_max`를 14로 변경하면 Transformer 임계가 4.2로 줄어 더 민감해집니다.

> 🟡 **권고 (Transformer)**
>
> `adaptive_supertrend.py` line 175:
> ```python
> # 현재
> if (not self._atr_initialized) or (period_change > int(cfg.atr_max_period * 0.3)):
>
> # 수정
> period_change_ratio = abs(adaptive_period - self._prev_adaptive_period) / max(float(self._prev_adaptive_period), 1.0)
> if (not self._atr_initialized) or (period_change_ratio > 0.5):
> ```

---

## 8. SkyEbest — `get_super_trend` / `get_super_trend_enhanced` 불일치

**발견 위치: SkyEbest `UnifiedTA.py` `MyTechnicalAnalysis` 클래스**

SkyEbest는 동일한 SuperTrend를 두 개의 메서드로 래핑하고 있으며, 파라미터가 일관되지 않습니다.

| 항목 | `get_super_trend()` | `get_super_trend_enhanced()` | 문제 |
|---|---|---|---|
| `smooth_period` | `1` (스무딩 없음) | `3` (EMA 스무딩) | 동일 데이터, 다른 출력 |
| `use_bb_correction` | `False` (항상) | `enable_ranging_filter` 파라미터 따름 | 기본 동작 다름 |
| 출력 배열 수 | 3개 (`st, ub, lb`) | 5개 (`st, ub, lb, regime, strength`) | 용도 분리 |
| warmup 처리 | `i >= lb` 이후만 저장 | `i >= lb` 이후만 저장 | 동일: 초기 `lb`봉 NaN |

같은 `lookback`/`multiplier`를 전달해도 `smooth_period` 차이로 인해 두 메서드의 `st` 값이 다릅니다. 시각화와 신호 생성 경로가 서로 다른 지표를 사용하게 될 수 있습니다.

> 🟠 **권고 (SkyEbest)**
>
> 두 메서드의 `smooth_period` 기본값을 통일하거나, 파라미터로 노출:
> ```python
> def get_super_trend(self, df, lookback, multiplier, smooth_period=3):
>     ast = AdaptiveSuperTrend(AdaptiveSuperTrendConfig(
>         ..., smooth_period=smooth_period))
> ```
>
> warmup NaN 처리: `i >= lb` 조건을 `i >= lb // 2` 또는 제거 검토  
> → 초기 봉에서도 부분적 결과가 나오는 것이 더 유용할 수 있음

---

## 9. Transformer 전용 — `AdaptiveIndicatorManager` & Cross Features

SkyEbest에는 없는 통합 기능입니다.

| Cross Feature | 계산 방식 | 활용 가치 |
|---|---|---|
| `cross_trend_agreement` | ST direction == ZZ direction → +1/-1 | 양 지표 방향 일치 시 신호 강화 |
| `cross_at_support` | ZZ 지지선 0.5% 이내 → 0~1 스케일 | 지지선 근접 매수 타이밍 |
| `cross_at_resistance` | ZZ 저항선 0.5% 이내 → 0~1 스케일 | 저항선 근접 매도 타이밍 |
| `cross_breakout_potential` | ER × 저항 근접도 (방향 포함) | 돌파/붕괴 가능성 수치화 |

또한 `AdaptiveIndicatorManager.is_ready()` 메서드가 있어 ZigZag 스윙 4개 이상 확정 시까지 feature 사용을 억제합니다. SkyEbest에는 이 준비 상태 판정 로직이 없습니다.

> 🟢 **권고 (SkyEbest)**
>
> 실시간 파이프라인에서 두 지표를 동시 업데이트한다면 `AdaptiveIndicatorManager` 패턴을 도입하거나 최소한 `cross_trend_agreement`를 계산하세요.
>
> LLM 컨텍스트 빌드 시 두 지표의 방향 일치 여부를 텍스트로 포함하면 판단 품질이 향상됩니다.

---

## 10. `compute_from_df` — 컬럼명 컨벤션 불일치

| 메서드 | Transformer 기본값 | SkyEbest 기본값 | 위험 |
|---|---|---|---|
| `SuperTrend.compute_from_df` | `high_col='high'` | `high_col='High'` | `AttributeError` |
| `ZigZag.compute_from_df` | `high_col='high'` | `high_col='High'` | `AttributeError` |
| `ZigZag.calculate()` | 없음 (`df["High"]` 하드코딩) | 있음 | 교차 사용 불가 |
| Transformer `azz_` 컬럼 | `azz_fib_0382`, `azz_fib_0618` 별칭 포함 | 별칭 없음 | `KeyError` |

> 🟡 **권고 (공통)**
>
> 공통 전처리 함수 또는 case-insensitive 컬럼 탐지 추가:
> ```python
> def _resolve_col(df, name):
>     for col in df.columns:
>         if col.lower() == name.lower():
>             return col
>     return name  # fallback
>
> high_col = _resolve_col(df, 'high')
> ```
>
> Transformer `features.py`가 `azz_fib_0382`를 사용한다면 SkyEbest에도 해당 별칭 추가 필요

---

## 11. 종합 우선순위 및 수정 가이드

| 우선순위 | 대상 | 항목 | 핵심 수정 내용 |
|---|---|---|---|
| 🔴 즉시 | Transformer | `bars_in_trend` 플립봉 누적 | `just_flipped` 패턴 적용 |
| 🔴 즉시 | SkyEbest | WilderRMA `ready >= ` 수정 | `count > ` → `count >= ` |
| 🔴 즉시 | SkyEbest | LLM `advice` 키 오류 | `s.direction` 기반 조건 사용 |
| 🔴 즉시 | 양쪽 | ZigZag ER 방향 반전 | `mmax - er*(range)` → `mmin + er*(range)` |
| 🟠 권장 | SkyEbest | `pending_confirm` 교체 조건 | 타입 불일치 시 교체 허용 |
| 🟠 권장 | SkyEbest | `calculate()` `_all_swings` 관리 | `del` → 슬라이싱 재할당 |
| 🟠 권장 | SkyEbest | ST 래퍼 `smooth_period` 불일치 | 두 메서드 파라미터 통일 |
| 🟡 중기 | Transformer | ATR 재초기화 임계 방식 | 비율 기반으로 통일 |
| 🟡 중기 | 양쪽 | `compute_from_df` 컬럼명 | 대소문자 정규화 또는 통일 |
| 🟢 개선 | SkyEbest | `AdaptiveIndicatorManager` 도입 | cross_features 생성 |
| 🟢 개선 | 양쪽 | 통합 테스트 추가 | 배치/스트리밍 일관성 검증 |

---

## 12. 결론

이번 2차 심층 리뷰에서 1차 분석 대비 **4개의 추가 버그**를 발견했습니다.

특히 **ZigZag ER 방향 논리 오류**는 양쪽 프로젝트에 공통으로 존재하며 스윙 포인트 품질에 실질적 영향을 줍니다. **WilderRMA `ready` 1봉 지연 버그**는 ADX 기반 multiplier 적응을 15봉 동안 비활성화시킵니다.

장기적으로는 두 프로젝트가 하나의 공유 라이브러리에서 `adaptive_indicator`를 가져오는 **Single Source of Truth** 구조로 전환하는 것이 가장 안전합니다. 현재처럼 별도 유지하면 같은 버그가 다시 발산할 위험이 있습니다.

---

*분석 기준: Transformer/adaptive_indicator/ · SkyEbest/views/charts/UnifiedTA.py*
