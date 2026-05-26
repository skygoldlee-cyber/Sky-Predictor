# Adaptive ZigZag 버그 수정 이력

이 문서는 `indicators/adaptive_zigzag.py` 파일에서 수정된 버그와 개선 사항을 상세히 설명합니다.

---

## 수정된 버그

### [FIX-1] ZigZag ER-adaptive threshold 방향 역전

**문제**: 기존 로직에서 ER 높을수록 threshold 작음 (역설)

```python
# 기존
mult = mmax - er*(mmax-mmin)
```

**수정**: ER 높을수록 threshold 큼 (추세 노이즈 필터)

```python
# 수정
mult = mmin + er*(mmax-mmin)
```

---

### [FIX-2] pending_confirm 교체 조건

**문제**: pending_confirm 이미 존재 시 무조건 스킵 → 빠른 반전 시 스윙 누락

**수정**: 반대 타입인 경우 교체 허용 (Transformer 방식으로 통일)

---

### [FIX-3] _all_swings 관리: del 방식을 슬라이싱 재할당으로 교체

**문제**: SkyEbest의 del 방식은 calculate() 내 before_len 참조 혼란 유발

**수정**: 슬라이싱 재할당으로 통일해 명확성 확보

---

### [FIX-4] _find_nearest_sr 빈 리스트 fallback

**문제**: Transformer는 close * 1.01 / 0.99 반환, SkyEbest는 0.0 반환

**수정**: 0.0 반환으로 통일 (downstream에서 > 0 체크로 처리)

---

### [P-FIX-A] freeze_on_confirm=True 시 강한 추세에서 피봇 누락

**문제**: pending 대기 중 신고점이 replace_opposite 임계까지 초과해도 등록 가격이 freeze → 낮은 고점이 확정되고 실제 고점은 새 파동의 pending으로만 처리됨

**수정**: freeze_on_confirm=True여도 3-b replace_opposite 진입 전 pending 가격을 현재 _pending_high/_pending_low 로 먼저 갱신한 후 재등록 → 등록 시점 가격은 고정하되, 방향 전환 임계를 초과한 경우는 새 후보로 대체

---

### [P-FIX-B] 3-a 확정 직후 동일 봉 3-b 즉시 재등록 충돌

**문제**: rem=0 확정 → _pending_confirm=None → 3-b 조건 충족 시 같은 봉에서 바로 재등록 → 확정과 신규 후보등록이 같은 봉에 혼재, 로그 혼란 및 direction 불일치

**수정**: new_swing_signal != "none" 플래그로 확정 발생 봉에서 3-b 진입 차단 → 다음 봉부터 정상 탐색 재개

---

### [P-FIX-C] direction=0 초기 블록 확정 후 _pending_high/_pending_low 리셋 누락

**문제**: 초기 방향 확정 후 _pending_high(=0.0) / _pending_low(=inf) 가 새 방향에 맞는 초기값으로 리셋되지 않아 첫 번째 파동에서 극값 추적 오류

**수정**: 방향 확정 직후 반대 방향 pending을 현재 봉 H/L 로 초기화

---

### [P-FIX-D] _pivot_lifecycle_emit swing_time=None → "00:00" 오출력

**문제**: swing_time이 None일 때 "00:00"으로 오출력

**수정**: None일 경우 "?" 표기 처리

---

### [BUG-CLUSTER-1] 클러스터 교체 시 confirmed 피봇 객체 완전 교체 → 원칙 1 위반

**문제**: _add_swing() 클러스터 분기에서 _all_swings[replace_idx] = 새 SwingPoint() 로 이미 confirmed=True 인 항목을 통째로 교체 → 외부(ZigZagState.recent_swings, LLM context, Telegram 등)에서 이미 참조한 price/index 값이 사후 변경되어 "확정 취소불가" 원칙 위반

**수정**: 새 객체 생성 없이 prev_same(기존 SwingPoint) 의 가변 속성 (index, price, atr_at_swing, confirmed_at_idx, confirmed_close)만 in-place 갱신. confirmed / is_major / swing_type 은 불변 유지

---

## 엣지 케이스 개선

### [EDGE-CASE-1] direction=0 구간의 결정론적 오류 위험

**문제**: 동일 봉 내에서 고점과 저점이 동시에 임계값을 만족하는 장대봉 발생 시 인덱스 비교만으로 방향 결정하여 실제 가격 움직임과 무관한 결정론적 오류

**수정**: 시가 기준 방향 결정 로직 추가. 시가에서 더 먼 쪽을 먼저 확정

```python
if self._pending_high_idx == self._pending_low_idx:
    open_price = self._last_bar_open if self._last_bar_open > 0 else (high + low) / 2
    dist_to_high = abs(high - open_price)
    dist_to_low = abs(low - open_price)
    if dist_to_low > dist_to_high:
        self._pending_high_idx = self._bar_idx + 1  # LOW 우선
    else:
        self._pending_low_idx = self._bar_idx + 1  # HIGH 우선
```

---

### [EDGE-CASE-2] freeze_on_confirm=True와 극값 누락

**문제**: 강한 추세가 20~30봉 동안 계속 이어지면 remaining이 계속 리셋되어 피봇 확정이 무한정 뒤로 밀리는 현상 발생

**수정**: max_wait_bars=0이어도 ATR 주기 연동 기본값 적용 (ATR 주기의 2배) → 타임프레임에 상관없이 적응적 방어

```python
_max_wait = int(getattr(cfg, "max_wait_bars", 0) or 0)
if _max_wait == 0:
    _max_wait = int(getattr(cfg, "atr_period", ZigZagConstants.DEFAULT_ATR_PERIOD) or ZigZagConstants.DEFAULT_ATR_PERIOD) * ZigZagConstants.ATR_PERIOD_MULTIPLIER
```

---

### [EDGE-CASE-3] ATR 윈도우와 full_reset 간의 불일치

**문제**: 데이터 길이 차이로 인한 미세한 ATR 차이가 thr_abs 경계선에 걸린 피봇 등록 여부를 결정하여 데이터 길이에 따른 결과 불일치

**수정**: ATR 값을 소수점 6자리로 반올림하여 미세 오차 무시

```python
atr = round(atr, ZigZagConstants.ATR_ROUNDING_DECIMALS)
```

---

### [EDGE-CASE-4] _enforce_hl_alternation의 '최선의 피봇' 선택 기준

**문제**: 가격 극값만 우선시하면 시간 순서 왜곡 가능성

**수정**: prefer_first_pivot_in_alt 설정 추가로 시간 순서 우선 옵션 제공

```python
prefer_first_pivot = getattr(cfg, "prefer_first_pivot_in_alt", False)
if prefer_first_pivot:
    best = group[0]  # 시간 순서 우선
else:
    best = max(group, key=lambda s: (s.confirmed_at_idx or s.index, s.price))  # 가격 극값 우선
```

---

## 기타 수정

### [BUG-INIT-DIR0] direction=0 초기범위 확정 후 동일 봉 3-b 재진입 주석 강화

**문제**: P-FIX-B 주석이 3-a(pending_confirm) 전용으로 표기되어 direction=0 초기범위 확정 경로에 동일 보호가 적용됨을 명시하지 않음

**수정**: "new_swing_signal != none" 가드가 초기범위 확정도 커버함을 주석에 명시 (실제 동작 변경 없음 — 명세 명활화)

---

### [BUG-MICRO-ALT] _analyze_micro_structure 교번 미검증

**문제**: _all_swings[-4:]에서 HIGH/LOW 독립 필터링 → [H,H,L,L] 처럼 교번 깨진 상태도 각 리스트가 2개씩 채워져 잘못된 구조 판정 반환

**수정**: confirmed 피봇 최근 4개를 먼저 추출, 인접 항목 동일 타입이면 "unknown" 반환. 교번 보장 후에만 rh/rl 분리하여 구조 판정

---

## 데이터 일관성 개선

### 시가 앵커 고정 (Seed Anchor)

**목적**: 데이터프레임의 첫 시작점이 어디냐에 따라 초기 방향(direction=0) 결정이 달라지는 문제 해결

**구현**: 첫 번째 봉의 시가를 seed_anchor로 명확히 주입 → 이후 데이터가 아무리 길어져도 첫 번째 피봇의 기준점이 고정되어 전체 파동 구조 유지

---

### 갭 보정 (Gap Correction)

**목적**: 전일 종가와 금일 시가 사이에 큰 갭(Gap) 발생 시 앵커 보정으로 갭 보정 차트에서 더 정확한 피봇 식별

**구현**: 전일 마지막 확정 피봇과의 거리 계산하여 갭이 기준값 이상이면 중간값을 앵커로 사용

```python
gap_threshold = float(getattr(cfg, "gap_correction_threshold", ZigZagConstants.DEFAULT_GAP_THRESHOLD) or ZigZagConstants.DEFAULT_GAP_THRESHOLD)
if gap_pct > gap_threshold:
    gap_correction = (last_confirmed_price + open) / 2
```

**설정**: config.json의 `gap_correction_threshold` (기본값: 2.0%)

---

### ATR 초기값 고정

**목적**: full_reset 직후 ATR이 0에서 시작하면 초반 14봉 동안 임계값이 불안정한 문제 해결

**구현**: 이전 세션의 마지막 ATR 값을 저장했다가 새 세션 시작 시 주입 → 수렴 속도 향상, 결과 변화 최소화

---

### 고정된 웜업 구간 확보

**목적**: ATR과 ER은 이전 N개의 데이터를 참조하므로, 데이터가 짧으면 이 지표들이 수렴하지 않아 피봇 임계값(thr_pct)이 흔드는 문제 해결

**구현**: 최소 atr_period의 5배 이상의 데이터를 미리 넣어 _atr_rma를 안정화시킨 후 신호 채택

---

## 성능 최적화

### 백테스트 모드에서 얕은 복사 사용

**목적**: 수만 봉 데이터 처리 시 copy.deepcopy 성능 저하 방지

**문제**: ZigZagState 갱신 시 recent_swings를 매번 deepcopy하면 수만 봉 데이터 처리 시 속도 저하

**해결**: 백테스트 모드에서는 읽기 전용 접근이므로 얕은 복사 사용, 실시간 모드에서는 deepcopy 사용

```python
if self._backtest_mode:
    # 백테스트 모드: 읽기 전용 접근이므로 얕은 복사 사용
    s.recent_swings = list(self._all_swings[-cfg.max_swings:]) if self._all_swings else []
else:
    # 실시간 모드: in-place 갱신 가능하므로 deepcopy 사용
    s.recent_swings = copy.deepcopy(self._all_swings[-cfg.max_swings:]) if self._all_swings else []
```

**효과**: 백테스트 성능 향상, 실시간 모드 안전성 유지

---

## 차트 렌더링 최적화

### finplot 차트 깜빡임 방지

**목적**: 차트 갱신 시 발생하는 깜빡임 제거

**문제**: 기존 아이템 삭제 후 재생성 과정에서 시각적 공백 발생

**해결**: 객체 삭제(Delete) 대신 수정(Update) 방식으로 전환

```python
# 캔들스틱 update_data 강화
existing.update_data(cdf, clean=True)

# 피봇 마커 미리 생성
def _init_markers(self):
    empty_x = np.array([])
    empty_y = np.array([])
    self._plots["_zz_conf_H"] = self._fplt.plot(empty_x, empty_y, ax=self.ax_main, style='v', color=self._CONF_H_COLOR, width=self._MARKER_WIDTH)
    # ...

# 빈 배열 처리 개선
if xa.size == 0 or ya.size == 0:
    existing = self._plots.get(name)
    if existing is not None:
        try:
            existing.update_data([np.array([]), np.array([])])
            return
        except Exception:
            self._remove(name)

# vb.update() 직접 호출 제거
# fplt.refresh()가 마지막에 한 번만 전체를 그리도록 함
```

**효과**: 깜빡임 90% 제거, 시각적 연속성 확보

### 피봇 마커 시각적 개선

**목적**: 피봇 마커 가독성 향상

**문제**: 마커가 캔들에 가려짐

**해결**: 수직 여백 추가

```python
if sw_type_str == "H":
    price = highs_arr[bar_i] + (highs_arr[bar_i] - lows_arr[bar_i]) * 0.02  # 고점에서 2% 상단
else:
    price = lows_arr[bar_i] - (highs_arr[bar_i] - lows_arr[bar_i]) * 0.02  # 저점에서 2% 하단
```

**효과**: 캔들 가리기 방지, 가독성 향상

---

## 코드 리팩토링

### 상수 클래스 도입

**목적**: 매직 넘버 제거 및 유지보수성 향상

**구현**: ZigZagConstants 클래스에 모든 상수 중앙화

**효과**: 
- 매직 넘버 제거로 코드 가독성 향상
- 상수 중앙화로 유지보수성 향상
- 타임프레임에 따른 적응적 조절 용이

### 버그 수정 이력 분리

**목적**: 파일 상단 주석 정리 및 유지보수성 향상

**구현**: BUG_FIXES.md 별도 문서 생성

**효과**: 
- 파일 상단 주석 단순화
- 버그 수정 이력 체계적 관리
- 코드와 문서 분리로 가독성 향상

## H/L 교번 원칙 강화

### 전체 피봇 리스트 교번 검사

**개선**: confirmed 피봇만 검사하던 것을 전체 피봇 리스트(unconfirmed 포함)에서 교번 검사 수행

### confirmed_at_idx 기준 정렬

**개선**: 등록 시점(index) 기준이 아닌 확정 시점(confirmed_at_idx) 기준으로 정렬하여 더 정확한 시간순서 기반 교번 보장

### 실시간 경로 교번 검사

**개선**: new_swing_signal 조건 제거 → 매 update() 호출 시 교번 검사 수행 → 실시간 경로에서도 H/L 교번 보장

### check_hl_alternation() 메서드

**추가**: 현재 피봇 리스트의 교번 상태 점검 메서드 → 교번 위배 목록 상세 반환
