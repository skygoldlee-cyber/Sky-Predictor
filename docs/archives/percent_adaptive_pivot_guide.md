# PercentAdaptivePivot 가이드

## 개요

`PercentAdaptivePivot`은 ATR / WilderRMA 의존성 없이 **퍼센트 기반 동적 threshold** 로 동작하는 적응형 피봇 탐지기입니다.  
`ATRAdaptivePivot` 과 동등한 탐지 품질을 유지하면서 외부 패키지 없이 `indicators` 내부만 사용합니다.

### 주요 특징

- **ATR 무의존성** — WilderRMA 불필요, 패키지 추가 없이 즉시 사용
- **동적 threshold** — 표준 Kaufman ER 기반 multiplier 자동 조정
- **direction 상태머신** — `+1 / −1 / 0` 으로 HIGH↔LOW 교번 강제, 연속 동일 타입 피봇 방지
- **후보 취소 로직** — 되돌림이 `thr × cancel_ratio` 미만이면 후보 취소 + 방향 복귀
- **이중 소파동 필터** — 퍼센트 크기 + 봉 간격(`min_bar_gap`) 동시 적용
- **시장 구조 분석** — HH·HL / LH·LL 기준으로 uptrend / downtrend / ranging / unknown 분류
- **pipeline 호환** — `get_transformer_features()` 가 `ATRAdaptivePivot` `azz_*` 키 25개 완전 호환
- **LLM 컨텍스트** — `get_llm_context()` 로 프롬프트 삽입용 텍스트 즉시 생성

---

## ATRAdaptivePivot 대비 차이점

| 항목 | ATRAdaptivePivot | PercentAdaptivePivot |
|---|---|---|
| threshold 기준 | `ATR × multiplier` (절대 pt) | `close × base_pct/100 × multiplier` (%) |
| 외부 의존성 | `WilderRMA` 필요 | 없음 |
| 변동성 추적 | ATR 이력 버퍼 | threshold_pct 이력 버퍼 |
| features 고유 키 | `aap_atr`, `aap_threshold_pct`, `aap_pivot_score` | `pap_threshold_pct`, `pap_pivot_score` |
| 나머지 설계 | 동일 (상태머신·취소로직·구조분석·Score) | ← |

---

## 핵심 알고리즘

### 1. 동적 threshold 계산

```
thr_pct = base_pct × multiplier × session_scale   (% 단위)
thr_abs = close × thr_pct / 100                   (절대 pt)
```

`multiplier` 는 ER 과 세션 테이블로 결정됩니다.

```
multiplier = multiplier_min + ER × (multiplier_max − multiplier_min)
           → 세션 scale 적용 후 clip(multiplier_min×0.5, multiplier_max×2.0)
```

### 2. Efficiency Ratio (표준 Kaufman)

```
ER = │close[t] − close[t−n]│ / Σ│close[i] − close[i−1]│
```

| ER | 시장 상태 | 효과 |
|---|---|---|
| ↑ (≈1.0) | 강한 추세 | multiplier ↑ → threshold 확대, 노이즈 차단 |
| ↓ (≈0.0) | 횡보 | multiplier ↓ → threshold 축소, 민감도 회복 |

> **구 버전 대비**: 방향변경 횟수 기반 비표준 ER → 표준 Kaufman ER 로 교체

### 3. direction 상태머신

```
direction = 0  →  _init_direction(): HIGH/LOW 인덱스 시간순으로 첫 앵커 결정
direction = +1 →  상승 탐색: 고점 갱신 추적, low 반전 시 HIGH 후보 등록
direction = -1 →  하락 탐색: 저점 갱신 추적, high 반전 시 LOW 후보 등록
```

피봇 확정 시 방향이 반드시 전환되어 연속 동일 타입 피봇이 발생하지 않습니다.

### 4. 후보 등록 → 확정 → 취소 흐름

```
반전 감지
  │
  ├─ _wave_ok() 통과?  ─No─→ 무시
  │
  Yes
  │
  ↓
_register_candidate()   ← pending_confirm dict 생성, remaining = confirmation_bars
  │
  ↓ (매봉 _process_pending 호출)
  ├─ 극단값 갱신 (freeze=False): 더 극단적이면 price/idx 갱신
  ├─ 되돌림 < thr × cancel_ratio?  →  취소 + 방향 복귀
  └─ remaining 감소 → 0 도달 시 _confirm_pivot() → direction 전환
```

### 5. 소파동 이중 필터 (`_wave_ok`)

```python
# 조건 1: 직전 확정 피봇 이후 최소 봉 간격
if candidate_idx - last_confirmed_bar < min_bar_gap:
    return False

# 조건 2: 현재 파동 크기 퍼센트
wave_pct = wave_abs / close × 100
if wave_pct < min_wave_pct:
    return False
```

### 6. is_major 판별

```
avg_wave = 최근 3파동 pct 평균
is_major = (직전 동일 타입 대비 │price 차이│ / 직전 가격 × 100) ≥ avg_wave × 1.5
```

### 7. 시장 구조 분석

최근 피봇 8개에서 HIGH 3개, LOW 3개를 추출 후:

```
HH 비율 ≥ 0.7 AND HL 비율 ≥ 0.7  →  uptrend
LH 비율 ≥ 0.7 AND LL 비율 ≥ 0.7  →  downtrend
그 외                              →  ranging
피봇 4개 미만                      →  unknown
```

### 8. Pivot Score

```
score = 변동성 변화(40%) + ER 강도(30%) + 후보 임박도(30%)
```

| 구성 요소 | 계산 | 가중치 |
|---|---|---|
| 변동성 변화 | `│recent_thr − MA20_thr│ / MA20_thr × 2` clip(0,1) | 0.4 |
| ER 강도 | Kaufman ER 값 그대로 | 0.3 |
| 후보 임박도 | `1 − remaining / confirmation_bars` clip(0,1) | 0.3 |

---

## 파라미터

### PercentAdaptivePivotConfig

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `base_pct` | float | 0.3 | 기본 퍼센트 임계값 (%) |
| `multiplier_min` | float | 0.8 | ER 기반 배수 하한 |
| `multiplier_max` | float | 2.0 | ER 기반 배수 상한 |
| `er_period` | int | 10 | ER 계산 기간 |
| `confirmation_bars` | int | 1 | 확정 확인 봉 수 |
| `min_wave_pct` | float | 0.15 | 최소 파동 크기 (%) |
| `min_bar_gap` | int | 3 | 직전 확정 피봇 이후 최소 봉 간격 |
| `max_pivots` | int | 30 | 보관할 최대 피봇 수 |
| `session_multiplier_table` | List[Tuple] | [] | 시간대별 배율 테이블 |
| `warmup_bars` | int | 20 | ER 안정화 웜업 기간 |
| `cancel_ratio` | float | 0.3 | 후보 취소 판단 비율 (되돌림 < thr × ratio 시 취소) |

### 파라미터 튜닝 가이드

#### `base_pct`

KOSPI200 선물 1분봉 기준 권장 범위: **0.2% ~ 0.5%**

- `0.2%` 이하: 피봇 과다 생성, 노이즈 증가
- `0.3%`: 일반적 추천값 (주요 파동 중심)
- `0.5%` 이상: 큰 파동만 탐지, 신호 희소

#### `multiplier_min / multiplier_max`

범위가 넓을수록 ER 적응력 증가, 좁을수록 일관된 민감도 유지.

- 보수적: `1.0 ~ 2.5` (추세장 threshold 크게 확대)
- 균형: `0.8 ~ 2.0` (기본 추천)
- 공격적: `0.6 ~ 1.4` (적응폭 제한)

#### `confirmation_bars` (확정 지연)

- **0**: 즉시 확정, 실시간 반응
- **1**: 1봉 확인 후 확정 (추천, 안정성 확보)
- **2-3**: 더 높은 안정성, 약간의 지연
- **4 이상**: 매우 보수적, 많은 지연

#### `cancel_ratio`

- `0.3`: 기본값 (ATRAdaptivePivot 동일)
- 낮출수록 (0.1~0.2): 후보 취소 어려워짐 → 더 많은 확정
- 높일수록 (0.5): 빠른 취소 → 방향 전환 신속

#### `min_bar_gap`

- `3`: 기본값, 직전 확정 후 최소 3봉 대기
- `1~2`: 빠른 전환 허용 (노이즈 증가 가능)
- `5~10`: 보수적, 큰 파동 위주

---

## 사용 예시

### 기본 사용

```python
from indicators import PercentAdaptivePivot, PercentAdaptivePivotConfig

cfg = PercentAdaptivePivotConfig(
    base_pct=0.3,
    multiplier_min=0.8,
    multiplier_max=2.0,
    er_period=10,
    confirmation_bars=1,
    min_wave_pct=0.15,
    min_bar_gap=3,
    warmup_bars=20,
)

pivot = PercentAdaptivePivot(cfg)
pivot.set_symbol("KP200 선물")

for high, low, close, bar_time in bars:
    state = pivot.update(high, low, close, bar_time=bar_time)

    if state.new_pivot_signal != "none":
        print(f"신호: {state.new_pivot_signal}")
        print(f"구조: {state.structure}")
        print(f"threshold: {state.threshold_pct:.3f}%")
        print(f"ER: {state.efficiency_ratio:.3f}")
        print(f"Pivot Score: {state.pivot_score:.3f}")
```

### 세션별 배율 적용

```python
cfg = PercentAdaptivePivotConfig(
    base_pct=0.3,
    session_multiplier_table=[
        ("09:00", "09:30", 1.5),   # 장초반: threshold 확대 (급등락 노이즈 차단)
        ("09:30", "14:30", 1.0),   # 정규 세션: 기본
        ("14:30", "15:20", 0.9),   # 마감: 약간 민감하게
    ],
)
```

### Transformer pipeline 주입

```python
for high, low, close, bar_time in bars:
    state = pivot.update(high, low, close, bar_time)

    # azz_* 25키 + pap_* 2키 = 27키
    # ATRAdaptivePivot 교체 시 features.py 수정 불필요
    features = pivot.get_transformer_features(close)
```

### LLM 프롬프트 삽입

```python
context = pivot.get_llm_context(close)
# → [PercentAdaptivePivot - KP200 선물]
#   현재가: 362.50  방향: 상승  구조: 상승 구조
#   신호: new_high
#   후보: 피봇후보 없음
#   최근 고점: 363.10  저점: 359.40  파동: 1.02%
#   ...
```

### reset 후 재사용

```python
pivot.reset()   # 완전 초기화 (웜업 포함)
```

---

## PAPState 필드

`update()` 의 반환값인 `PAPState` 의 주요 필드입니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `new_pivot_signal` | str | `"new_high"` / `"new_low"` / `"none"` |
| `direction` | int | `+1` 상승탐색 / `-1` 하락탐색 / `0` 미결정 |
| `structure` | str | `"uptrend"` / `"downtrend"` / `"ranging"` / `"unknown"` |
| `threshold_pct` | float | 현재 봉 reversal threshold (%) |
| `efficiency_ratio` | float | 현재 Kaufman ER (0~1) |
| `last_high` | float | 최근 확정 고점 (미확정 시 `nan`) |
| `last_low` | float | 최근 확정 저점 (미확정 시 `nan`) |
| `pending_type` | str\|None | 현재 후보 타입 (`"high"` / `"low"`) |
| `pending_price` | float | 현재 후보 가격 |
| `pending_remaining` | int | 확정까지 남은 봉 수 |
| `pivot_score` | float | 현재 Pivot Score (0~1) |
| `wave_size_pct` | float | 최근 파동 크기 (%) |
| `bars_since_pivot` | int | 마지막 확정 피봇 이후 경과 봉 수 |
| `recent_pivots` | List[PivotPoint] | 최근 확정 피봇 목록 |

---

## get_transformer_features() 키 목록

### azz_* 키 (ATRAdaptivePivot 완전 호환, 25개)

| 키 | 범위 | 설명 |
|---|---|---|
| `azz_direction` | −1 / 0 / 1 | 탐색 방향 |
| `azz_last_high` | −1 ~ 1 | 최근 확정 고점까지 거리 (±5% 정규화) |
| `azz_last_low` | −1 ~ 1 | 최근 확정 저점까지 거리 (±5% 정규화) |
| `azz_wave_size_pct` | 0 ~ 1 | 파동 크기 (10% 기준 정규화) |
| `azz_support_dist_pct` | 0 ~ 1 | 직하 지지 거리 |
| `azz_res_dist_pct` | 0 ~ 1 | 직상 저항 거리 |
| `azz_bars_since_swing` | 0 ~ 1 | 경과 봉 (50봉 기준 정규화) |
| `azz_higher_highs` | 0 / 1 | uptrend 여부 |
| `azz_lower_lows` | 0 / 1 | downtrend 여부 |
| `azz_new_swing` | −1 / 0 / 1 | 이번 봉 신호 |
| `azz_swing_recency` | 0 ~ 1 | 피봇 최신성 (exp decay) |
| `azz_threshold_pct` | 0 ~ 1 | threshold (3% 기준 정규화) |
| `azz_structure_up` | 0 / 1 | uptrend 구조 |
| `azz_structure_down` | 0 / 1 | downtrend 구조 |
| `azz_structure_ranging` | 0 / 1 | ranging 구조 |
| `azz_micro_up` | 0 / 1 | (structure_up 동일) |
| `azz_micro_down` | 0 / 1 | (structure_down 동일) |
| `azz_micro_ranging` | 0 / 1 | (structure_ranging 동일) |
| `azz_structure_conf` | 0 / 0.7 | 구조 확정 여부 |
| `azz_pend_sr_dist` | −1 ~ 1 | 후보 가격까지 거리 |
| `azz_pending_type` | −1 / 0 / 1 | 후보 타입 |
| `azz_pending_dist` | −1 ~ 1 | 후보 가격까지 거리 (동일) |
| `azz_pending_urgency` | 0 ~ 1 | 후보 확정 임박도 |
| `azz_pending_age` | 0 ~ 1 | 후보 등록 후 경과 (exp decay) |
| `azz_pending_prob` | 0 ~ 1 | Pivot Score |

### pap_* 키 (고유, 2개)

| 키 | 범위 | 설명 |
|---|---|---|
| `pap_threshold_pct` | 0 ~ 1 | threshold % (3% 기준 정규화) |
| `pap_pivot_score` | 0 ~ 1 | Pivot Score |

---

## 튜닝 프리셋

### 보수적 트레이딩 (장기 추세 추종)

```python
cfg = PercentAdaptivePivotConfig(
    base_pct=0.5,           # 높은 임계값
    multiplier_min=1.0,     # 높은 배수
    multiplier_max=2.5,
    min_wave_pct=0.3,       # 큰 파동만
    confirmation_bars=2     # 확인 지연
)
```

### 공격적 트레이딩 (단기 스캘핑)

```python
cfg = PercentAdaptivePivotConfig(
    base_pct=0.2,           # 낮은 임계값
    multiplier_min=0.5,     # 낮은 배수
    multiplier_max=1.4,
    min_wave_pct=0.1,       # 작은 파동도
    confirmation_bars=0     # 즉시 확정
)
```

### 밸런스드 트레이딩 (일반적)

```python
cfg = PercentAdaptivePivotConfig(
    base_pct=0.3,           # 중간 임계값
    multiplier_min=0.8,     # 중간 배수
    multiplier_max=2.0,
    min_wave_pct=0.15,      # 중간 파동
    confirmation_bars=1     # 1봉 확인
)
```

---

## 주의사항

- `warmup_bars` 미만 구간에서는 신호가 출력되지 않습니다 (ER 미안정).
- `confirmation_bars=0` 설정 시 후보 취소 로직이 작동하지 않으므로 fake pivot 위험이 증가합니다. 기본값인 `confirmation_bars=1`을 권장합니다.
- `reset()` 호출 시 확정 피봇 이력, 방향 상태, 후보 모두 초기화됩니다. 장 시작 시 호출하세요.
- pandas 의존성이 제거되었습니다. 시간 포맷팅은 표준 datetime 모듈을 사용합니다.
- `get_transformer_features()` 의 모든 값은 `math.isfinite()` 보장 (nan/inf 자동 0.0 대체).

---

## 변경 로그

- **2026-05-15 v1.0**: 초기 버전 (구 PercentAdaptivePivot)
  - ATR 의존성 제거, 퍼센트 기반 동적 임계값, 비표준 ER
- **2026-05-15 v2.0**: 전면 재설계 (ATRAdaptivePivot 기능 동등)
  - ER → 표준 Kaufman ER 교체
  - direction 상태머신 도입, 후보 취소 로직 개선
  - Transformer/LLM 호환성 추가
  - 소파동 이중 필터 (pct + 봉 간격)
  - 시장 구조 분석 (uptrend/downtrend/ranging/unknown)
  - Pivot Score 3요소 가중합
  - `get_transformer_features()` azz_* 호환
  - `get_llm_context()` 추가
- **2026-05-15 v2.1**: 코드 리팩토링 및 최적화
  - 상수 정의 (DEFAULT_WARMUP_BARS, DEFAULT_CONFIRMATION_BARS 등)
  - 마법 숫자 상수화 (MULTIPLIER_CLIP_MIN_FACTOR, MULTIPLIER_CLIP_MAX_FACTOR)
  - pandas 의존성 제거, datetime 모듈 사용
  - 중복 계산 제거 (_calc_pivot_score에 er 인자 전달)
  - config.json 기본값과 일치 (multiplier_max: 2.0, confirmation_bars: 1, min_wave_pct: 0.15, warmup_bars: 20)
- **2026-05-15 v2.2**: CRIT/WARN 리뷰 수정
  - _init_direction() 단위 오류 수정: 절대 포인트를 퍼센트로 변환 후 전달
  - _register_candidate() confirmation_bars=0 무력화 수정: max(0, ...) 변경
  - _process_pending() 방향 복귀 추가: 취소 시 self._direction 명시 복귀
  - _add_pivot() max_pivots * 2 보관 패턴 복원: SR 탐색용 피봇 풀 확보
  - _run_logic() thr_pct 주석 처리: 향후 확장용 보존
