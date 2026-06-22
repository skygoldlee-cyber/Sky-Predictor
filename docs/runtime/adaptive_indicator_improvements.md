# adaptive_indicator 개선점(제안)

본 문서는 `adaptive_indicator/` 모듈(Adaptive SuperTrend / Adaptive ZigZag / Integration)의 **정확성·일관성·성능·유지보수성** 관점에서 확인된 개선 포인트를 정리한 것입니다.

---

## 1) 현재 구조 요약

- `adaptive_indicator/adaptive_supertrend.py`
  - `AdaptiveSuperTrend.update(high, low, close)`로 상태 갱신
  - `get_transformer_features(close)` → `ast_*` 9개
- `adaptive_indicator/adaptive_zigzag.py`
  - `AdaptiveZigZag.update(high, low, close)`로 상태 갱신
  - `get_transformer_features(close)` → `azz_*` 14개
- `adaptive_indicator/indicator_integration.py`
  - `AdaptiveIndicatorManager.update(high, low, close)`에서
    - SuperTrend/ZigZag update → 피처 결합
    - cross 피처 4개 추가(`cross_*`)
    - LLM 컨텍스트 조립

피처 정본(런타임 입력 스키마)은 `prediction/features.py`의 `ADAPT_KEYS`(총 28개)입니다.

---

## 2) 확인된 문제/리스크

### 2.1 배치 계산(`compute_from_df`)의 cross 피처 계산이 상태를 따라가지 않음(정확성)

`AdaptiveIndicatorManager.compute_from_df()`는 아래 순서로 동작합니다.

1. `self.supertrend.compute_from_df(df, ...)`
2. `self.zigzag.compute_from_df(df, ...)`
3. 이후 `df.iterrows()`를 돌며 `_calc_cross_features(...)`를 추가

하지만 3번 단계의 cross 계산에서 사용하는 `st_state = self.supertrend.state`, `zz_state = self.zigzag.state`는 **루프가 진행되어도 갱신되지 않습니다.**
- `compute_from_df()` 호출이 끝난 시점의 state(마지막 봉 state)만을 계속 사용하게 되어,
- 결과적으로 cross 컬럼이 **행별로 변화하지 않거나 왜곡될 가능성**이 큽니다.

권장 방향:
- 배치 모드에서도 `update()`를 행 단위로 호출하여 state가 시간에 따라 진행되게 하거나,
- `compute_from_df`가 내부적으로 행별 state 시퀀스를 함께 산출하도록 구조 변경.

반영됨:
- `AdaptiveIndicatorManager.compute_from_df()`에서 cross 피처가 **행별 state 기반**으로 계산되도록 수정 완료
- 회귀 방지용 테스트 추가: `tests/test_adaptive_indicator_smoke.py`의 배치 vs 순차(update) cross 비교

### 2.2 feature count/설명 불일치(일관성)

`indicator_integration.py` docstring/주석에는 “약 22개 피처” 같은 표현이 남아 있으나,
실제 런타임 정본(`prediction/features.py`)은 **ADAPT 28개**입니다.

권장 방향:
- 문서/주석/리턴 구조 설명을 `ADAPT_KEYS(28개)` 기준으로 통일.

### 2.3 `AdaptiveSuperTrend.update()`의 방향 결정 로직이 중복/혼재(정확성·유지보수)

`AdaptiveSuperTrend.update()`의 “방향 결정” 부분은 같은 변수를 여러 번 덮어쓰는 형태가 있어,
의도 파악/수정이 어렵고 작은 변경에도 회귀가 생길 수 있습니다.

권장 방향:
- SuperTrend 표준 구현(밴드 연속성 + flip 규칙)을 기준으로 로직을
  - 단계별로 분리
  - 최종 `direction` 결정 경로를 1개로 수렴
  - 최소 단위 테스트(플립 케이스, 횡보 케이스) 추가

반영됨:
- `AdaptiveSuperTrend.update()`의 방향 결정 구간을 표준 SuperTrend의 final band + flip 규칙 형태로 리팩터링하여 중복 덮어쓰기 제거
- 최소 플립 테스트 추가: `tests/test_adaptive_indicator_smoke.py`

### 2.4 NaN/초기 구간 처리의 일관성 부족(안정성)

- ADX 초기화 전에는 상수에 가까운 값(예: 25.0)을 반환하는데,
  warmup 직후 수치가 급격히 튀는 구간이 생길 수 있습니다.
- BB 폭 계산에서 표준편차 `ddof=1` 사용 시 표본 수가 작으면 NaN 위험이 있습니다.

권장 방향:
- “지표 유효성” 플래그(예: `is_ready`)를 상태에 포함하고,
  `warmup_bars`와 연동하여 피처 사용 시점을 명시.

### 2.5 `usage_example.py`의 import 경로가 현재 프로젝트와 다름(실행성)

`usage_example.py`는 `from indicators import ...`로 import 하고 있는데,
현재 패키지명은 `adaptive_indicator` 입니다.

권장 방향:
- 예제 파일이 실제로 실행 가능한지 확인하고, import 경로를 프로젝트에 맞게 정리.

반영됨:
- `adaptive_indicator/usage_example.py`의 import를 `adaptive_indicator` 기준으로 수정

### 2.6 성능: 순수 파이썬 list 버퍼 + iterrows(성능)

- 실시간에서는 문제 없을 수 있으나, 배치 처리에서는 `iterrows()`는 느립니다.
- 버퍼가 list로 관리되어 슬라이싱/재할당이 자주 발생할 수 있습니다.

권장 방향:
- 배치 모드는 가능한 벡터화/NumPy 연산으로 처리하거나,
- 최소한 `itertuples()`로 전환.
- 버퍼는 `collections.deque(maxlen=...)`로 단순화하는 것도 후보.

---

## 3) 우선순위 개선안(추천)

### P0 (정확성)

- `AdaptiveIndicatorManager.compute_from_df()`의 cross 피처 계산을 **행별 state 기반**으로 재구현 (완료)
- `indicator_integration.py`의 “피처 개수/설명”을 `ADAPT_KEYS(27)`에 맞게 정리

### P1 (안정성)

- 지표 “준비 상태”(`is_ready`)를 추가하여 warmup 구간의 피처 품질을 명확히 함
- BB/ADX/ER 등 초기 구간 NaN/상수 처리 정책을 문서화

추가 반영됨(런타임 관측/디버깅):

- adaptive indicator가 활성화된 경우, 예측 결과의 `model_outputs.heuristic`에 **adaptive 기반 HEURISTIC action**을 포함하여
  로그의 `[HEURISTIC]` 블록 및 `[DIR_SUMMARY]` 요약에 반영되도록 연결(ready 전에도 placeholder 출력 가능).

### P2 (유지보수)

- SuperTrend 방향 결정/밴드 연속성 로직을 정리(중복 제거) (완료)
- config dataclass에 대한 범위 validation(예: period >= 2, min<=max)을 추가

### P3 (성능)

- 배치 계산 시 `iterrows()` 제거(가능하면 벡터화, 최소 `itertuples()`로)
- 내부 버퍼 자료구조 최적화(deque 도입 등)

---

## 4) 테스트 보강 제안

- `tests/test_adaptive_indicator_smoke.py` 외에 다음 레벨의 테스트가 있으면 회귀를 줄일 수 있습니다.

- **배치 vs 실시간 일치성 테스트**
  - 동일 OHLCV를 입력했을 때
    - (A) `update()`를 순차 호출하며 얻은 피처 시퀀스
    - (B) `compute_from_df()`로 얻은 컬럼
  - 두 결과가 근사하게 일치하는지 비교

반영됨:
- 배치 vs 순차(update) cross 피처 비교 테스트 추가

- **SuperTrend flip 케이스**
  - 상승→하락, 하락→상승 전환이 의도대로 발생하는지

반영됨:
- 최소 flip smoke 테스트 추가(상승 플립 신호 발생 여부)

---

## 5) 결정이 필요한 사항(질문)

- `compute_from_df()`는 학습 데이터 생성에 실제로 사용 중인가, 아니면 예제/분석용인가?
  - 학습 파이프라인에서 사용 중이면 P0로 즉시 수정 권장
  - 미사용이면 문서/예제 수준으로 유지하고, 런타임 경로만 보강해도 됨
