# indicators/ (Runtime)

## 역할

- 분봉 OHLCV(High/Low/Close)를 입력으로 받아 **적응형 지표 피처(ADAPT 40개)** 와 **LLM 컨텍스트 텍스트**를 생성
- 런타임에서는 `prediction/pipeline.py`의 `PredictionPipeline.get_prediction()`에서:
  - `indicators` 패키지의 `AdaptiveIndicatorManager`를 사용
  - `AdaptiveIndicatorManager.config`의 기간 파라미터(`atr_max_period`, `bb_period`, `adx_period`) 이상의 분봉으로 지표 상태를 warmup
  - 마지막 분봉으로 `AdaptiveIndicatorManager.update(...)`를 호출해 최신 피처/컨텍스트를 생성

## 파일 구성

| 파일 | 주요 클래스/요소 | 설명 |
|---|---|---|
| `__init__.py` | re-export | 외부에서 `from indicators import ...`로 사용 가능하도록 공개 API 제공 |
| `adaptive_supertrend.py` | `AdaptiveSuperTrend`, `AdaptiveSuperTrendConfig`, `SuperTrendState` | ER/ADX/BB 기반으로 ATR period 및 multiplier를 적응형으로 조정하는 SuperTrend |
| `adaptive_zigzag.py` | `AdaptiveZigZag`, `AdaptiveZigZagConfig`, `ZigZagState`, `SwingPoint`, `SwingType` | ATR 기반 동적 임계값으로 스윙/피보/지지저항/구조를 추적하는 ZigZag |
| `indicator_integration.py` | `AdaptiveIndicatorManager`, `IndicatorManagerConfig` | SuperTrend + ZigZag를 묶어 Transformer 피처 + LLM 컨텍스트를 단일 출력 |

---

## 핵심 클래스/함수

### 1) `indicator_integration.py`

| 이름 | 종류 | 설명 | 주요 I/O |
|---|---|---|---|
| `IndicatorManagerConfig` | dataclass | 통합 매니저 설정(supertrend/zigzag/symbol) | in: config |
| `AdaptiveIndicatorManager` | class | 런타임 통합 관리자 | stateful |
| `AdaptiveIndicatorManager.update(high, low, close)` | method | 1개 분봉을 반영해 피처/컨텍스트/상태를 반환 | out: dict |

반환 dict 주요 키(요약):

- `transformer`: `Dict[str, float]` (28개)
- `llm_context`: `str`
- `supertrend_state`: `SuperTrendState`
- `zigzag_state`: `ZigZagState`
- `bar_count`: `int`

### 2) `adaptive_supertrend.py`

| 이름 | 종류 | 설명 |
|---|---|---|
| `AdaptiveSuperTrendConfig` | dataclass | atr_min/max, multiplier_min/max, er/adx/bb 관련 파라미터 |
| `SuperTrendState` | dataclass | value/direction/bands/atr/er/adx/signal 등 상태 |
| `AdaptiveSuperTrend` | class | 적응형 SuperTrend 계산기 |
| `AdaptiveSuperTrend.update(high, low, close)` | method | 상태 업데이트 |
| `AdaptiveSuperTrend.reset()` | method | 내부 버퍼와 상태 초기화 (데이터 소스 전환 시 호출) |
| `AdaptiveSuperTrend.get_transformer_features(close)` | method | `ast_*` 9개 피처 dict 생성 |
| `AdaptiveSuperTrend.get_llm_context(close, symbol)` | method | SuperTrend 자연어 요약 생성 |

### 3) `adaptive_zigzag.py`

| 이름 | 종류 | 설명 |
|---|---|---|
| `SwingType` | enum | HIGH/LOW |
| `SwingPoint` | dataclass | 스윙 포인트 정보 |
| `ZigZagState` | dataclass | 방향/스윙/피보/SR/구조 등 상태 |
| `AdaptiveZigZagConfig` | dataclass | ATR/ER 기반 임계값/스윙 설정 |
| `AdaptiveZigZag` | class | 적응형 ZigZag 계산기 |
| `AdaptiveZigZag.update(high, low, close)` | method | 상태 업데이트 |
| `AdaptiveZigZag.get_transformer_features(close)` | method | `azz_*` 15개 피처 dict 생성 |
| `AdaptiveZigZag.get_llm_context(close, symbol)` | method | ZigZag 자연어 요약 생성 |

ZigZag 임계값은 고정 %가 아니라 다음 파라미터로 동적으로 결정됩니다.

- `er_period`: Efficiency Ratio(ER) 계산 기간
- `atr_multiplier_min/max`: ER에 따라 ATR multiplier를 `[min,max]` 범위에서 자동 선택
  - ER이 높을수록(추세) `max`에 가까워져 threshold가 커지고(노이즈 필터),
    ER이 낮을수록(횡보) `min`에 가까워져 threshold가 작아집니다(스윙 더 자주)
- `min_wave_bars`, `min_wave_pct`: 전환 감지 후 스윙 확정 전 최소 파동 길이 필터(노이즈 억제)

추가 설정(운영 튜닝용):

- `freeze_on_confirm`: confirmation window 동안 후보 스윙(price/idx) 고정 여부. `True` 권장(소급 변경(repainting) 완화)
- `structure_lookback_swings`: 구조 분석(up/down/ranging)에 사용할 최근 스윙 개수
- `structure_points`: 구조 판정에 사용할 HIGH/LOW 샘플 개수

---

## 피처 출력(ADAPT 40)

- 피처 키/순서는 런타임에서는 `prediction/features.py`의 `ADAPT_KEYS`가 정본입니다.
- 구성:
  - `ast_*` 9개
  - `azz_*` 27개
  - `cross_*` 4개

참고:
- 배치 처리(`AdaptiveIndicatorManager.compute_from_df`)는 cross 피처(`cross_*`)도 행별 state를 따라가도록 계산됨
- 회귀 방지 테스트:
  - `tests/test_adaptive_indicator_smoke.py`

---

## warmup_bars 권장 규칙

런타임에서 `indicators` 패키지의 warmup은 지표 내부 윈도우가 최소 1회 이상 안정적으로 채워진 뒤 피처를 사용하기 위한 값입니다.

```
base = max(supertrend.atr_max_period, supertrend.bb_period, supertrend.adx_period)

warmup_min       = base + 5
warmup_recommend = base + 10
warmup_stable    = 2 * base
```

---

## 런타임 연결 지점

- `prediction/pipeline.py`
  - warmup 구간에서 `AdaptiveIndicatorManager.update()`를 반복 호출
  - 마지막 분봉으로 최신 `adaptive_features` / `adaptive_context`를 생성
  - `build_sequence(..., adaptive_features=adaptive_features)`로 ADAPT 블록을 모델 입력에 포함
  - `build_llm_context(..., adaptive_context=...)`로 `[ADAPTIVE_INDICATORS]` 블록을 프롬프트 컨텍스트에 포함

## 예제 실행

- `adaptive_indicator/usage_example.py`
  - 현재 프로젝트에서는 `adaptive_indicator` 패키지를 import 하도록 정리되어 직접 실행 가능

---

## 데이터 소스 전환 시 SuperTrend 전체 재계산

GUI 차트 뷰어에서 데이터 소스(KOSPI ↔ KP200 선물)를 전환할 때 SuperTrend가 올바르게 계산되도록 하기 위한 메커니즘입니다.

### 문제
데이터 소스 전환 시 SuperTrend가 이전 데이터 소스의 캐시를 사용하여 증분 업데이트를 수행하면, 잘못된 SuperTrend 값이 계산됩니다.

### 해결 방법
`gui/engines/chart_engine.py`의 `compute()` 메서드에서 `force_recompute=True` 또는 데이터 소스 변경 시 다음을 수행합니다:

1. **캐시 초기화**:
   - `_st_cache_sig = None` (캐시 서명 초기화)
   - `_st_cache_values = None` (값 캐시 초기화)
   - `_st_cache_dirs = None` (방향 캐시 초기화)
   - `_st_fed_bars = 0` (feed한 완결봉 수 초기화)

2. **인스턴스 리셋**:
   - `AdaptiveSuperTrend.reset()` 호출하여 내부 버퍼와 상태 초기화

3. **전체 재계산 강제**:
   - `_st_fed_bars = 0` 설정으로 무조건 전체 재계산 수행

### 동작 흐름
```
데이터 소스 변경 (KOSPI → KP200)
  ↓
force_recompute = True 설정
  ↓
ChartEngine.compute() 호출
  ↓
SuperTrend 캐시 초기화
  ↓
AdaptiveSuperTrend.reset() 호출
  ↓
전체 재계산 수행 (411 완결봉)
  ↓
올바른 KP200 SuperTrend 값 계산 (~1172.70)
```

### 로그 확인
데이터 소스 전환 시 다음 로그가 나타나야 합니다:
```
[ChartEngine] SuperTrend 전체 재계산: 411 완결봉
```

### 관련 파일
- `gui/engines/chart_engine.py`: `compute()` 메서드
- `gui/chart_viewer.py`: 데이터 소스 변경 감지 및 `force_recompute` 설정
- `indicators/adaptive_supertrend.py`: `AdaptiveSuperTrend.reset()` 메서드

---

# 부록 — 통합 분석/튜닝 가이드

> 원본 보고서는 리팩토링 과정에서 삭제되었습니다.
> - `adaptive_indicator_improvements.md` → 부록 A
> - `adaptive_indicator_parameters.md` → 부록 B

---

## 부록 A. 개선점 및 권고사항

**원본**: `adaptive_indicator_improvements.md`

### 해결된 항목

| # | 문제 | 해결 방법 | 테스트 |
|---|------|-----------|--------|
| 1 | `compute_from_df()`의 cross 피처가 마지막 state만 사용 | 행별 state 기반으로 재계산 | `tests/test_adaptive_indicator_smoke.py` 배치 vs 순차 비교 |
| 2 | 피처 개수/설명 불일치 (22개/26개/28개) | 코드/문서를 실제 피처 기준(`ast_9` + `azz_27` + `cross_4` = 40개)으로 통일 | `indicator_integration.py` docstring, 본문 섹션 업데이트 |
| 3 | `AdaptiveSuperTrend.update()` 방향 결정 로직 중복 | 표준 SuperTrend final band + flip 규칙으로 리팩터링 | `tests/test_adaptive_indicator_smoke.py` flip smoke 테스트 |
| 4 | `usage_example.py` import 경로 불일치 | `indicators` 패키지 기준으로 수정 (`usage_example.py`는 현재 미존재) | - |

### 반영된 권고 (P0 → P3)

| 우선순위 | 항목 | 상태 | 구현 내용 |
|----------|------|------|-----------|
| P0 | 피처 개수/설명 정리 | ✅ 완료 | `indicator_integration.py` 모듈/클래스 docstring을 40개 피처 기준으로 정리, `adaptive_indicator.md` 본문 `azz_*` 개수 15개 → 27개 수정 |
| P1 | 준비 상태 플래그 | ✅ 완료 | `AdaptiveIndicatorManager.is_ready()`가 이미 구현되어 있으며, `update()` 반환 딕셔너리에 `is_ready` 포함 |
| P1 | 초기 구간 NaN/상수 처리 | ✅ 문서화 완료 | `adaptive_supertrend.py` 모듈 docstring에 ER/ADX/BB 초기 구간 처리 정책 추가, `SuperTrendState.adx` 초기값 25.0 유지 |
| P2 | 범위 validation | ✅ 완료 | `AdaptiveSuperTrendConfig` 및 `AdaptiveZigZagConfig`에 `__post_init__` validation 추가 (`period >= 2`, `min <= max`, 양수값 보장) |
| P3 | 성능 최적화 | ✅ 완료 | `compute_from_df()`가 이미 `itertuples()`를 사용하며, `deque` 기반 버퍼는 `adaptive_supertrend.py`/`adaptive_zigzag.py`에서 이미 사용 중 |

### 결정이 필요한 사항

- `compute_from_df()`는 학습 데이터 생성에 실제 사용 중이며, 현재 `itertuples()` 기반으로 최적화되어 있음.
- 추가 벡터화가 필요한 경우 `update()` 루프 자체를 NumPy/Numba로 재작성하는 별도 작업이 필요.

---

## 부록 B. 파라미터 가이드

**원본**: `adaptive_indicator_parameters.md`

### 공통 튜닝 방향

| 목표 | AST 조정 | AZZ 조정 |
|------|----------|----------|
| 민감도↑ (빠른 반응) | `multiplier_*` ↓, `atr_*_period` ↓, `smooth_period` ↓ | `pivot_threshold_min_pct` ↓, `atr_multiplier_min/max` ↓, `confirmation_bars` ↓, `min_wave_bars` ↓ |
| 안정성↑ (노이즈 감소) | `multiplier_*` ↑, `smooth_period` ↑, `atr_max_period` ↑ | `pivot_threshold_min_pct` ↑, `confirmation_bars` ↑, `min_wave_bars` ↑, `freeze_on_confirm=true` |

### Adaptive SuperTrend (AST) 파라미터

| 파라미터 | 의미 | 권장 범위 | 튜닝 팁 |
|----------|------|-----------|---------|
| `atr_min_period` | 적응형 ATR 기간 하한 | 5 ~ 10 | 휩쏘 많으면 `atr_max_period` ↑ |
| `atr_max_period` | 적응형 ATR 기간 상한 | 14 ~ 30 | - |
| `multiplier_min` | SuperTrend multiplier 하한 | 1.5 ~ 2.5 | 자주 flip이면 상향 |
| `multiplier_max` | SuperTrend multiplier 상한 | 3.5 ~ 5.5 | 신호 늦으면 하향 |
| `er_period` | ER 계산 기간 | 8 ~ 20 | 값 ↓ = 민감도 ↑, 노이즈 ↑ |
| `adx_period` | ADX 계산 기간 | 10 ~ 20 | - |
| `smooth_period` | SuperTrend EMA 스무딩 기간 | 1 ~ 7 | 휩쏘 많으면 3~5 |
| `bb_period` | 볼린저 밴드 기간 | 14 ~ 30 | `use_bb_correction=true`일 때 사용 |
| `bb_std` | 볼린저 밴드 표준편차 | 1.5 ~ 2.5 | - |
| `adx_mult_norm_cap` | ADX 정규화 cap | 40 ~ 80 | - |
| `bb_correction_floor` | BB 보정 floor | 0.5 ~ 0.9 | - |
| `bb_correction_ref_pct` | BB 보정 기준 % | 0.03 ~ 0.10 | - |

### Adaptive ZigZag (AZZ) 파라미터

| 파라미터 | 의미 | 권장 범위 | 튜닝 팁 |
|----------|------|-----------|---------|
| `atr_period` | ATR 계산 기간 | 10 ~ 20 | - |
| `er_period` | ER 계산 기간 | 8 ~ 20 | - |
| `atr_multiplier_min` | 동적 multiplier 하한 | 0.8 ~ 1.8 | 스윙 안 잡히면 ↓ |
| `atr_multiplier_max` | 동적 multiplier 상한 | 3.0 ~ 6.0 | 스윙 잦으면 ↓ |
| `atr_multiplier` | ER 워밍업 기본값 | 1.0 ~ 3.0 | 기본값으로 두는 것이 안전 |
| `pivot_threshold_min_pct` | threshold 하한 % | 0.2 ~ 1.0 | - |
| `pivot_threshold_max_pct` | threshold 상한 % | 2.0 ~ 6.0 | - |
| `confirmation_bars` | reversal 후 확정 지연 | 0 ~ 5 | 만기주/노이즈: 2~4 |
| `freeze_on_confirm` | 후보 스윙 고정 여부 | `true` | `true` 권장 (repainting 완화) |
| `min_wave_bars` | 최소 파동 봉 수 | 1 ~ 15 | 스윙 잦으면 ↑ |
| `min_wave_pct` | 최소 파동 % | 0.0 ~ 3.0 | 0 권장 시작점 |
| `major_swing_ratio` | major 스윙 분류 기준 | 1.5 ~ 3.5 | `atr × ratio` 기준 |
| `max_swings` | 유지할 최근 스윙 개수 | 10 ~ 60 | - |
| `cluster_tolerance_pct` | 지지/저항 클러스터 허용 오차 | 0.2 ~ 0.8 | - |
| `structure_lookback_swings` | 구조 판정 스윙 개수 | 6 ~ 20 | - |
| `structure_points` | 구조 판정 샘플 개수 | 2 ~ 4 | - |

### warmup_bars 권장 규칙

```text
base = max(supertrend.atr_max_period, supertrend.bb_period, supertrend.adx_period)
warmup_min       = base + 5
warmup_recommend = base + 10
warmup_stable    = 2 * base
```

---

## 변경 이력

- 2026-06-24: 부록 A의 P0~P3 권고 실제 코드 반영
  - P0: 피처 개수/설명 정리 (`indicator_integration.py` docstring 40개 기준, `adaptive_indicator.md` 본문 정리)
  - P1: `is_ready` 구현 상태 문서화 및 초기 구간 NaN/상수 처리 정책 추가 (`adaptive_supertrend.py`)
  - P2: `AdaptiveSuperTrendConfig`/`AdaptiveZigZagConfig`에 `__post_init__` validation 추가
  - P3: `compute_from_df()`가 이미 `itertuples()`를 사용함을 문서/주석에 반영
- 2026-06-24: `tests/test_adaptive_indicator_smoke.py` 정리
  - `ADAPT_KEYS` 전체 기대 → `AdaptiveIndicatorManager`가 실제 생성하는 피처 검증
  - 잘못된 `adaptive_indicator`/`kospi_indicators` import를 `indicators`로 수정
  - `is_ready` 반환 확인 추가
- 2026-06-24: `IndicatorManagerConfig`에 누락된 `supertrend_pivot_filter` 필드 추가
