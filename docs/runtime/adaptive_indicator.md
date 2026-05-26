# adaptive_indicator/ (Runtime)

## 역할

- 분봉 OHLCV(High/Low/Close)를 입력으로 받아 **적응형 지표 피처(ADAPT 28개)** 와 **LLM 컨텍스트 텍스트**를 생성
- 런타임에서는 `prediction/pipeline.py`의 `PredictionPipeline.get_prediction()`에서:
  - `adaptive_indicator.warmup_bars` 만큼의 분봉으로 지표 상태를 warmup
  - 마지막 분봉으로 `AdaptiveIndicatorManager.update(...)`를 호출해 최신 피처/컨텍스트를 생성

## 파일 구성

| 파일 | 주요 클래스/요소 | 설명 |
|---|---|---|
| `__init__.py` | re-export | 외부에서 `adaptive_indicator import ...`로 사용 가능하도록 공개 API 제공 |
| `adaptive_supertrend.py` | `AdaptiveSuperTrend`, `AdaptiveSuperTrendConfig`, `SuperTrendState` | ER/ADX/BB 기반으로 ATR period 및 multiplier를 적응형으로 조정하는 SuperTrend |
| `adaptive_zigzag.py` | `AdaptiveZigZag`, `AdaptiveZigZagConfig`, `ZigZagState`, `SwingPoint`, `SwingType` | ATR 기반 동적 임계값으로 스윙/피보/지지저항/구조를 추적하는 ZigZag |
| `indicator_integration.py` | `AdaptiveIndicatorManager`, `IndicatorManagerConfig` | SuperTrend + ZigZag를 묶어 Transformer 피처 + LLM 컨텍스트를 단일 출력 |
| `usage_example.py` | 예시 코드 | 실시간 업데이트/배치 처리/피처 목록 확인 예시 |

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

## 피처 출력(ADAPT 28)

- 피처 키/순서는 런타임에서는 `prediction/features.py`의 `ADAPT_KEYS`가 정본입니다.
- 구성:
  - `ast_*` 9개
  - `azz_*` 15개
  - `cross_*` 4개

참고:
- 배치 처리(`AdaptiveIndicatorManager.compute_from_df`)는 cross 피처(`cross_*`)도 행별 state를 따라가도록 계산됨
- 회귀 방지 테스트:
  - `tests/test_adaptive_indicator_smoke.py`

---

## warmup_bars 권장 규칙

런타임에서 `adaptive_indicator.warmup_bars`는 지표 내부 윈도우가 최소 1회 이상 안정적으로 채워진 뒤 피처를 사용하기 위한 값입니다.

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
