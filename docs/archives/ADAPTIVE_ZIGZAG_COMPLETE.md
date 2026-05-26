# Adaptive ZigZag 알고리즘 상세 설명 및 코드 리뷰

## 개요

Adaptive ZigZag는 ATR(Average True Range) 기반 동적 임계값과 ER(Efficiency Ratio) 적응형 필터링을 사용하여 스윙 고점/저점을 탐지하는 기술적 지표입니다. 시장 변동성에 따라 임계값을 자동 조정하여 추세 반전을 정확하게 감지합니다.

본 문서는 알고리즘의 상세 설명과 코드 리뷰를 통합하여 제공합니다.

---

# Part 1: 알고리즘 상세 설명

## 1. 핵심 알고리즘 구조

### 1.1 메인 처리 흐름 (`update` 메서드)

```
1. OHLC 데이터 수집
2. True Range & ATR 계산
3. ATR 변화 모니터링 (급격 변동 감지)
4. 적응형 임계값 계산 (ATR × 배율)
5. Pending Confirmation 처리
6. 방향 결정/전환
7. 스윙 추가 및 클러스터링
8. 상태 업데이트
```

### 1.2 데이터 구조

```python
class AdaptiveZigZag:
    _highs: deque[float]      # 고가 시계열
    _lows: deque[float]       # 저가 시계열
    _closes: deque[float]     # 종가 시계열
    _tr: deque[float]         # True Range 시계열
    _atr_values: deque[float] # ATR 시계열
    _current_direction: int   # 현재 방향 (1: 상승, -1: 하락, 0: 미정)
    _pending_high: float      # 후보 고점
    _pending_low: float       # 후보 저점
    _all_swings: List[SwingPoint]  # 모든 스윙 포인트
    _swing_version: int       # 스윙 버전 카운터 (렌더링 캐시 무효화용)
```

## 2. ATR 기반 동적 임계값 계산

### 2.1 True Range 계산

```python
if n >= 2:
    pc = self._closes[-2]  # 이전 종가
    tr = max(high - low, abs(high - pc), abs(low - pc))
else:
    tr = high - low
```

**True Range**: 현재 봉의 실제 변동폭
- 고가 - 저가
- 고가 - 이전 종가 (절대값)
- 저가 - 이전 종가 (절대값)
- 셋 중 최대값

### 2.2 ATR 계산 (Wilder's RMA)

```python
atr = self._atr_rma.update(tr)
```

**Wilder's RMA (Relative Moving Average)**: 지수 이동평균의 변형
- α = 1 / N (N: 기간)
- ATR[t] = ATR[t-1] + α × (TR[t] - ATR[t-1])
- 최근 데이터에 더 높은 가중치 부여

### 2.3 ATR 급격 변동 감지

```python
if spike_detected:
    if change_pct > 0:
        self._dynamic_atr_ratio = 1.3  # ATR 급증 시
    else:
        self._dynamic_atr_ratio = 0.7  # ATR 급락 시
```

**ATR 급변 시 동적 비율 조정** ([REVIEW-FIX-2] 논리적 일관성 복구):
- ATR 급증(>임계값): 배율 1.3 → 임계값 높임 → 노이즈 억제
- ATR 급락(<-임계값): 배율 0.7 → 임계값 낮춤 → 민감도 회복

## 3. 적응형 임계값 계산 (`_calc_threshold_pct`)

### 3.1 ER (Efficiency Ratio) 계산

```python
def _calc_er(self) -> float:
    # [REVIEW-FIX-4] look-ahead 편향 방지: 현재 봉 제외하고 완결봉만 사용
    cs = list(self._closes)[-(period + 1):-1]  # 현재 봉 제외
    if len(cs) < period:
        return 0.5
    direction = abs(cs[-1] - cs[0])
    volatility = sum(abs(cs[i] - cs[i-1]) for i in range(1, len(cs)))
    er = direction / volatility if volatility > 0 else 0.5
    return clip(er, 0.0, 1.0)
```

**Efficiency Ratio (ER)**: 추세의 효율성 측정
- 방향성 변화 / 총 변동성
- 1에 가까울수록 강한 추세 (노이즈 적음)
- 0에 가까울수록 횡보 (노이즈 많음)
- [REVIEW-FIX-4] 현재 봉(미완결) 제외하여 look-ahead 편향 방지

### 3.2 DER (Directional Efficiency Ratio) 계산

```python
def _calc_der(self) -> float:
    # 현재 방향과 ER 기반 방향의 일치도 계산
    # 방향 불일치 시 음수 반환
```

**DER**: 현재 추세 방향과 ER 기반 방향의 일치도
- 방향 불일치 시 임계값 완화하여 전환 조기 감지

### 3.3 동적 배율 계산

```python
# [FIX-1] ER 방향 역전 수정 (기존 버그 수정)
# 기존: mult = mmax - er*(mmax-mmin)  → ER 높을수록 threshold 작음 (역설)
# 수정: mult = mmin + er*(mmax-mmin)  → ER 높을수록 threshold 큼 (추세 노이즈 필터)

mult = mmin + er * (mmax - mmin)

# 방향 불일치 시 완화
if direction_mismatch:
    mult = mult * der_ratio  # 0.7배로 감소
```

**동적 배율 계산 로직**:
- `mmin`: 최소 배율 (기본 1.0)
- `mmax`: 최대 배율 (기본 4.0, 장초반 8.0)
- `er`: Efficiency Ratio (0~1)
- `mult`: ER에 따라 mmin~mmax 사이 선형 보간
- ER 높을수록 배율 높음 → 임계값 높음 → 강한 추세에서 잡음 필터링

### 3.4 시간대 기반 조절

```python
if early_start <= current_time <= early_end:
    mmax = early_session_atr_multiplier_max  # 장초반 8.0
else:
    mmax = atr_multiplier_max  # 일반 4.0
```

**장초반 변동성 대응**:
- 장초반 (09:00-09:30): 최대 배율 8.0 → 임계값 높임 → 잦은 피봇 변경 방지
- 일반 시간: 최대 배율 4.0

### 3.5 세션별 파라미터 테이블

```python
session_min_wave_atr_ratio_table = [
    ("09:00", "09:30", 0.8),  # 장초반 - 변동성 큼
    ("09:30", "11:30", 0.5),  # 장중반
    ("11:30", "13:20", 0.4),  # 점심시간 - 변동성 작음
    ("13:20", "15:35", 0.6),  # 장마감
]
```

**세션별 최소 파동 크기 (ATR 배수)**:
- 각 시간대별 변동성 특성에 맞춘 파라미터
- ATR 급변 시 배율 적용 (0.7× 또는 1.3×)

### 3.6 최종 임계값 계산

```python
thr_pct = mult * atr / close * 100  # 백분율 임계값
thr_abs = close * thr_pct / 100    # 절대 임계값
```

## 4. 스윙 탐지 로직

### 4.1 초기 방향 결정 (direction=0)

```python
if (pending_high - pending_low) >= thr_abs:
    if pending_high_idx > pending_low_idx:
        # 저점 먼저 → 상승 추세 시작
        current_direction = 1
        add_swing(low_idx, low, LOW)
    else:
        # 고점 먼저 → 하락 추세 시작
        current_direction = -1
        add_swing(high_idx, high, HIGH)
```

**초기 범위 확정**:
- 고점-저점 차이가 임계값 이상이면 방향 확정
- 먼저 발생한 극값을 기준 스윙으로 등록

### 4.2 상승 추세 (direction=1)

```python
# 고점 갱신
if high > pending_high:
    pending_high = high
    pending_high_idx = bar_idx

# 반전 조건 확인
if (pending_high - close) >= thr_abs:
    # 고점 확정
    add_swing(pending_high_idx, pending_high, HIGH)
    current_direction = -1  # 방향 전환
```

**상승 추세 로직**:
- 고점을 계속 갱신 (pending_high)
- 현재 가격이 고점에서 임계값 이상 하락하면 고점 확정
- 방향 전환 (상승 → 하락)

### 4.3 하락 추세 (direction=-1)

```python
# 저점 갱신
if low < pending_low:
    pending_low = low
    pending_low_idx = bar_idx

# 반전 조건 확인
if (close - pending_low) >= thr_abs:
    # 저점 확정
    add_swing(pending_low_idx, pending_low, LOW)
    current_direction = 1  # 방향 전환
```

**하락 추세 로직**:
- 저점을 계속 갱신 (pending_low)
- 현재 가격이 저점에서 임계값 이상 상승하면 저점 확정
- 방향 전환 (하락 → 상승)

### 4.4 Pending Confirmation 윈도우

```python
def _process_pending_confirmation(self, high, low, close, atr, thr_pct):
    # 확정 봉 수 (confirmation_bars) 동안 반전 조건 확인
    # 조건 충족 시 피봇 확정
    # 조건 미충족 시 취소
```

**Pending Confirmation**:
- 피봇 후보 등록 후 `confirmation_bars` 봉 동안 대기
- 대기 중 반전 조건 충족 → 피봇 확정
- 대기 중 조건 미충족 → 피봇 취소
- `freeze_on_confirm=True`: 확정 시 등록 가격 고정

### 4.5 Replace Opposite (반대 방향 교체)

```python
# [FIX-2] 반대 타입인 경우 교체 허용
if swing_type != prev_same.swing_type:
    # 기존 후보 취소 후 새 후보 등록
```

**Replace Opposite**:
- 기존 후보와 반대 타입이면 교체 허용
- 빠른 반전 시 스윙 누락 방지

## 5. 피봇 클러스터링

### 5.1 클러스터링 로직

```python
def _add_swing(self, ...):
    # 직전 동일 타입 피봇 확인
    prev_same = next((s for s in reversed(_all_swings)
                      if s.swing_type == swing_type and s.confirmed), None)

    if prev_same and dist_pct <= cluster_tolerance_pct:
        if is_more_extreme:
            # [BUG-CLUSTER-1] confirmed 객체 완전 교체 버그 수정
            # 기존 객체의 가변 속성만 in-place 갱신
            prev_same.index = new_index
            prev_same.price = new_price
            # [REVIEW-FIX-5] in-place 갱신 시 버전 카운터 증가
            self._swing_version += 1
            # confirmed, is_major, swing_type은 불변 유지
```

**클러스터링**:
- 직전 동일 타입 피봇과 허용 범위 이내이면 병합
- 더 극단적인 가격이면 기존 피봇 갱신
- [BUG-CLUSTER-1] 수정: confirmed 객체 완전 교체 방지 (불변 속성 유지)
- [REVIEW-FIX-5] swing_version 카운터로 렌더링 캐시 무효화

### 5.2 ATR 기반 클러스터링

```python
if use_atr_based_filtering and atr > 0:
    cluster_tol_atr = atr * cluster_atr_ratio / price * 100
    cluster_tol = max(cluster_tol, cluster_tol_atr)
```

**ATR 기반 클러스터링**:
- ATR 배수 기반 클러스터링 거리 계산
- 기존 백분율 기반과 ATR 기반 중 더 큰 값 사용

## 6. ATR 기반 필터링

### 6.1 파동 크기 필터링

```python
def _check_wave_size(self, close, pending_high, pending_low, atr):
    if use_atr_based_filtering and atr > 0:
        min_wave_abs = atr * min_wave_atr_ratio
        wave_size = abs(pending_high - pending_low)
        return wave_size >= min_wave_abs
    return True
```

**파동 크기 필터링**:
- 파동 크기가 ATR × min_wave_atr_ratio 미만이면 필터링
- 작은 파동 무시하여 잡음 제거

## 7. 피보나치 레벨 계산

```python
fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
for ratio in fib_ratios:
    fib_level = swing_price + (swing_price - opposite_swing_price) * ratio
```

**피보나치 리트레이스먼트**:
- 스윙 간 피보나치 레벨 계산
- 지지/저항선 참조용

## 8. 미시 구조 분석

```python
def _analyze_micro_structure(self):
    # 최근 4개 confirmed 피봇의 교번 확인
    # HIGH-LOW-HIGH-LOW 패턴 검증
    # 구조 판정 (상승/하락/횡보/unknown)
```

**미시 구조 분석**:
- 최근 피봇의 교번 패턴 확인
- 시장 구조 판정 (추세/횡보)

## 9. 파라미터 튜닝 가이드

### 9.1 기본 파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| atr_multiplier | 1.5 | ATR 기본 배율 | 1.0 ~ 3.0 |
| atr_period | 14 | ATR 계산 기간 | 10 ~ 20 |
| er_period | 10 | ER 계산 기간 | 5 ~ 15 |
| atr_multiplier_min | 1.0 | 최소 배율 | 0.5 ~ 1.5 |
| atr_multiplier_max | 4.0 | 최대 배율 | 2.0 ~ 6.0 |
| confirmation_bars | 2 | 확정 대기 봉 수 | 1 ~ 5 |
| freeze_on_confirm | True | 확정 시 가격 고정 | True/False |
| min_wave_bars | 5 | 최소 파동 봉 수 | 3 ~ 10 |
| cluster_tolerance_pct | 0.3 | 클러스터링 허용 범위(%) | 0.1 ~ 0.5 |

### 9.2 ATR 필터링 파라미터

| 파라미터 | 기본값 | 설명 | 튜닝 범위 |
|---------|--------|------|-----------|
| use_atr_based_filtering | False | ATR 필터링 활성화 | True/False |
| min_wave_atr_ratio | 0.5 | 최소 파동 크기(ATR 배수) | 0.3 ~ 1.0 |
| cluster_atr_ratio | 0.5 | 클러스터링 ATR 배수 | 0.3 ~ 1.0 |

### 9.3 시간대 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| early_session_start_time | "09:00" | 장초반 시작 시간 |
| early_session_end_time | "09:30" | 장초반 종료 시간 |
| early_session_atr_multiplier_max | 8.0 | 장초반 최대 배율 |

### 9.4 튜닝 전략

**변동성 높은 시장**:
- `atr_multiplier`: 증가 (2.0 ~ 3.0)
- `confirmation_bars`: 증가 (3 ~ 5)
- `cluster_tolerance_pct`: 증가 (0.4 ~ 0.5)

**변동성 낮은 시장**:
- `atr_multiplier`: 감소 (1.0 ~ 1.5)
- `confirmation_bars`: 감소 (1 ~ 2)
- `cluster_tolerance_pct`: 감소 (0.1 ~ 0.3)

**빠른 전환 감지**:
- `atr_multiplier_min`: 감소 (0.5 ~ 0.8)
- `freeze_on_confirm`: False
- `min_wave_bars`: 감소 (3 ~ 5)

**안정적 추세**:
- `atr_multiplier`: 증가 (2.0 ~ 2.5)
- `confirmation_bars`: 증가 (3 ~ 4)
- `freeze_on_confirm`: True

## 10. 알고리즘 특징 및 장단점

### 10.1 장점

1. **적응형 임계값**: 변동성에 따라 자동 조정
2. **ER 필터링**: 추세/횡보 구분으로 노이즈 필터링
3. **클러스터링 기능**: 근접 피봇 병합으로 신호 정리
4. **시간대 조절**: 장초반/세션별 파라미터 지원
5. **ATR 필터링**: 파동 크기 기반 필터링
6. **버그 수정**: 다양한 엣지 케이스 대응
7. **[REVIEW-FIX-2] ATR 급변 논리적 일관성**: 변동성 증가 시 임계값 높여 노이즈 억제
8. **[REVIEW-FIX-4] look-ahead 편향 방지**: 현재 봉 제외하여 실시간 환경 보정
9. **[REVIEW-FIX-5] 렌더링 캐시 무효화**: swing_version으로 in-place 갱신 감지

### 10.2 단점

1. **파라미터 민감도**: 파라미터 튜닝 필요
2. **지연**: 확정 대기 봉 수로 인한 지연 발생
3. **복잡성**: 로직 복잡하여 이해 어려움
4. **계산 비용**: ATR, ER 계산으로 인한 오버헤드

---

# Part 2: 코드 리뷰 및 개선 사항

## 🔴 심각 (수학적 오류 / 로직 버그)

### 1. ER 방향 수정의 의도와 실제 효과 불일치 ⏳ 검토 필요

문서에 "[FIX-1] ER 방향 역전 수정"이라고 기재되어 있지만, 수정 결과가 직관에 반합니다.

```python
# 수정된 공식
mult = mmin + er * (mmax - mmin)
# ER=1 (강한 추세) → mult = mmax = 4.0 → threshold 높음 → 전환 신호 어려움
# ER=0 (횡보)      → mult = mmin = 1.0 → threshold 낮음 → 전환 신호 쉬움
```

**문제**: 강한 추세에서는 임계값이 높아야 잡음을 걸러내는 것은 맞지만, 동시에 추세 전환 신호 자체도 억제됩니다. 횡보 구간에서 임계값이 낮으면 가짜 전환 신호가 과다 발생합니다. Perry Kaufman의 KAMA 원리와 반대 방향입니다.

**권장 검토**:

| 시장 상태 | ER | 권장 threshold | 현재 동작 |
|-----------|----|----------------|------------|
| 강한 추세 | 높음 | 낮게 (추세 따라가기) | 높게 ❌ |
| 횡보 | 낮음 | 높게 (잡음 무시) | 낮게 ❌ |

원래 공식(`mmax - er*(mmax-mmin)`)이 Kaufman의 의도에 더 부합할 수 있습니다. 설계 의도를 명문화하고 백테스트로 검증이 필요합니다.

---

### 2. ATR 급변 감지의 비대칭 처리 ✅ **해결 완료**

```python
if change_pct > 0:
    self._dynamic_atr_ratio = 0.7  # ATR 급증 시 → 임계값 낮춤
else:
    self._dynamic_atr_ratio = 1.3  # ATR 급락 시 → 임계값 높임
```

**문제**: ATR이 급증하면 실제 시장 변동성이 커진 것인데, 임계값을 낮추면 오히려 노이즈 피봇이 더 많이 생성됩니다. 반대로 ATR 급락 시 임계값을 높이면 진짜 전환 신호를 놓칩니다.

**수정 완료** ([REVIEW-FIX-2]):
```python
# 논리적으로 일관된 방향으로 수정 완료
if change_pct > 0:
    self._dynamic_atr_ratio = 1.3  # ATR 급증 시 → 임계값 높여 노이즈 억제 ✓
else:
    self._dynamic_atr_ratio = 0.7  # ATR 급락 시 → 임계값 낮춰 민감도 회복 ✓
```

---

### 3. _process_pending_confirmation의 취소 조건 미명시 ⏳ 대기

문서에서 "조건 미충족 시 취소"라고만 기술되어 있으나, `max_wait_bars` 초과 후 취소 시 pending 가격 처리 방법이 불명확합니다.

취소 시 `_pending_high/_pending_low`를 현재 봉 기준으로 리셋하지 않으면, 이전 극값이 다음 사이클에서 잘못된 기준점으로 작용합니다.

---

## 🟠 성능 / 정확도 이슈

### 4. ER 계산의 Look-ahead 편향 가능성 ✅ **해결 완료**

```python
def _calc_er(self) -> float:
    direction = abs(close - self._closes[-period])
    volatility = sum(abs(self._closes[i] - self._closes[i-1]) for i in range(-period, 0))
```

`self._closes[-period]`가 deque의 현재 봉 포함 여부에 따라 현재 봉 데이터가 ER 계산에 포함될 수 있습니다. 실시간 환경에서는 현재 봉이 미완결 상태이므로 `[-period-1:-1]` 범위를 명시적으로 사용해야 합니다.

**수정 완료** ([REVIEW-FIX-4]):
```python
# look-ahead 편향 방지: 현재 봉 제외하고 완결봉만 사용
cs = list(self._closes)[-(period + 1):-1]  # 현재 봉 제외 ✓
if len(cs) < period:
    return 0.5
```

---

### 5. 클러스터링 로직의 in-place 갱신 부작용 ✅ **해결 완료**

```python
# [BUG-CLUSTER-1] 수정: 기존 객체의 가변 속성만 in-place 갱신
prev_same.index = new_index
prev_same.price = new_price
```

**문제**: `_all_swings` 리스트를 외부에서 참조하는 코드(예: `chart_viewer.py`의 `_build_pivot_markers`)가 동일 객체를 가리키고 있으면, in-place 갱신이 렌더러의 캐시 무효화를 우회합니다. `chart_viewer.py`의 `_pm_hash`는 마지막 피봇의 idx/y만 비교하므로 중간 피봇이 갱신되어도 해시가 변하지 않아 화면이 갱신되지 않습니다.

**수정 완료** ([REVIEW-FIX-5]):
```python
# swing_version 카운터 추가 및 in-place 갱신 시 증가
self._swing_version: int = 0  # 초기화
self._swing_version += 1      # 클러스터링 갱신 시 증가 ✓

# ZigZagState에 swing_version 추가
s.swing_version = self._swing_version

# _pm_hash에 swing_version 포함
key = (..., pm.get("swing_version", 0))  # 캐시 무효화 ✓
```

---

### 6. deque maxlen 미설정 시 메모리 누수 ✅ **확인 완료 (문제 없음)**

```python
self._highs: deque[float]
self._lows: deque[float]
```

`maxlen`이 설정되지 않으면 장중 누적 데이터가 무한 증가합니다. 해외선물은 24시간 거래이므로 특히 위험합니다.

**확인 완료**: 이미 `maxlen=max_buf`로 설정되어 있어 문제 없음
```python
max_buf = int(max(cfg.atr_period * 5, 100))
self._highs: deque = deque(maxlen=max_buf)  # ✓ 이미 설정됨
```

---

## 🟡 해외선물 적용 관점 검토

### 7. 세션 파라미터가 KST 고정 ⏳ 대기

```python
session_min_wave_atr_ratio_table = [
    ("09:00", "09:30", 0.8),  # 장초반
    ("13:20", "15:35", 0.6),  # 장마감
]
```

**문제**: 해외선물(CME, EUREX 등)은 KST 기준 세션이 전혀 다릅니다.

| 상품 | 주요 변동성 시간 (KST) |
|------|------------------------|
| E-mini S&P | 22:30 (미국 개장), 03:00 (선물 마감) |
| 유로/달러 | 16:00 (유럽 개장), 22:30 (미국 개장) |
| 원유(CL) | 22:30, 장중 EIA 발표 시 |

세션 테이블을 상품별로 분리 설정할 수 있는 구조가 필요합니다.

---

### 8. ATR 기간(14봉)이 해외선물 분봉에 부적합할 수 있음 ⏳ 대기

국내 KP200은 1분봉 기준 하루 약 370봉이지만, 해외선물 1분봉은 24시간 × 60 = 1,440봉입니다. ATR 14봉은 14분 변동성만 반영하여 지나치게 단기적입니다.

**해외선물 권장 ATR 기간 (분봉 기준)**:
```python
# 1분봉: 60~120 (1~2시간)
# 5분봉: 24~48 (2~4시간)
# 15분봉: 14~28 (3.5~7시간)
```

---

### 9. confirmation_bars의 시간 의미가 봉 수에 종속 ⏳ 대기

`confirmation_bars=2`는 1분봉에서 2분, 15분봉에서 30분을 의미합니다. 해외선물에서 타임프레임을 바꾸면 확정 지연 시간이 크게 달라져 전략 성격이 변합니다.

**권장**: 봉 수 대신 시간(초) 기반 확정 조건
```python
confirmation_seconds: int = 120  # 2분 고정
confirmation_bars = max(1, confirmation_seconds // bar_interval_seconds)
```

---

## 🔵 설계 / 유지보수성

### 10. _calc_der 구현이 문서에 불완전하게 기술됨 ⏳ 대기

```python
def _calc_der(self) -> float:
    # 현재 방향과 ER 기반 방향의 일치도 계산
    # 방향 불일치 시 음수 반환
```

음수를 반환한다고 기술되어 있으나, 이후 `mult = mult * der_ratio`에서 음수 × 양수 = 음수 threshold가 될 수 있습니다. 실제 구현에서 `abs()` 처리가 있는지 확인 필요합니다.

---

## 우선 순위 권장사항

### 즉시 수정 (심각) ✅ **완료**

1. **ATR 급변 감지 비대칭 수정**: 논리적 일관성 복구 ✅ 완료 ([REVIEW-FIX-2])
2. **deque maxlen 설정**: 메모리 누수 방지 ✅ 확인 완료 (이미 설정됨)
3. **ER 계산 look-ahead 수정**: 실시간 환경 보정 ✅ 완료 ([REVIEW-FIX-4])

### 단기 수정 (성능) ✅ **완료**

4. **클러스터링 버전 카운터**: 렌더링 캐시 무효화 ✅ 완료 ([REVIEW-FIX-5])
5. **pending 취소 조건 명시화**: 로직 명확화 ⏳ 대기

### 중기 검토 (설계)

6. **ER 공식 재검토**: Kaufman 원리와 일치성 검증 ⏳ 대기
7. **세션 파라미터 상품별 분리**: 해외선물 대응 ⏳ 대기
8. **ATR 기간 타임프레임별 튜닝**: 시장 적합성 확보 ⏳ 대기

### 장기 개선 (아키텍처)

9. **시간 기반 confirmation**: 타임프레임 독립성 확보 ⏳ 대기
10. **DER 구현 문서화**: 설계 의도 명확화 ⏳ 대기

---

## 백테스트 검증 항목

### ER 공식 비교

| 공식 | ER=1 시 동작 | ER=0 시 동작 | 백테스트 지표 |
|------|--------------|--------------|---------------|
| 현재 (`mmin + er*(mmax-mmin)`) | threshold 높음 | threshold 낮음 | 신호 빈도, 수익률 |
| 원래 (`mmax - er*(mmax-mmin)`) | threshold 낮음 | threshold 높음 | 신호 빈도, 수익률 |

### ATR 급변 처리 비교

| 상황 | 현재 동작 | 권장 동작 | 백테스트 지표 |
|------|-----------|------------|---------------|
| ATR 급증 | 임계값 낮춤 | 임계값 높임 | 노이즈 피봇 수 |
| ATR 급락 | 임계값 높임 | 임계값 낮춤 | 전환 신호 누락 수 |

---

## 결론

Adaptive ZigZag는 실무 적용을 고려한 잘 설계된 알고리즘이나, 수학적 일관성과 해외선물 적용 관점에서 개선이 필요합니다.

### 수정 완료 항목
- [REVIEW-FIX-2] ATR 급변 감지 비대칭 수정: 논리적 일관성 복구 완료
- [REVIEW-FIX-4] ER 계산 look-ahead 편향 수정: 실시간 환경 보정 완료
- [REVIEW-FIX-5] 클러스터링 버전 카운터: 렌더링 캐시 무효화 완료

### 검토 필요 항목
- ER 공식과 ATR 급변 처리의 논리적 오류 검토 필요
- 해외선물 적용을 위해서는 세션 파라미터와 ATR 기간의 상품별 튜닝 필수
- pending 취소 조건 명시화 및 DER 구현 문서화 필요

---

## 참고 문헌

- Wilder's RMA: J. Welles Wilder Jr., "New Concepts in Technical Trading Systems"
- Efficiency Ratio: Perry Kaufman, "Smarter Trading"
- ZigZag: 기술적 분석 표준 지표
