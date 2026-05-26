# Adaptive ZigZag 피봇 확정 로직

## 목적

`prediction` 런타임에서 사용하는 Adaptive ZigZag의 피봇 확정 흐름을 코드 기준으로 정리한다.

- 후보 등록/변경/취소/확정 조건
- `confirmation_bars` 의미와 실제 지연 해석
- 로그(`ZZ_PIVOT`, `ZZ_CONFIRM`)를 통한 운영 점검 방법

---

## 적용 코드 경로

- 핵심 엔진: `kospi_indicators/kospi_indicators/adaptive_zigzag.py`
- 통합 매니저: `kospi_indicators/kospi_indicators/indicator_integration.py`
- 파이프라인 연결: `prediction/adaptive_mixin.py`, `prediction/pipeline.py`
- 텔레그램 포맷: `telegram_formatters.py`

---

## 상태/용어

- `pending_confirm`: 미확정 피봇 후보(dict)
  - `type`: `"high"` 또는 `"low"`
  - `idx`: 후보 피봇 봉 인덱스
  - `price`: 후보 가격
  - `remaining`: 확정까지 남은 봉 수
- `confirmation_bars`: 후보 등록 후 확정까지 필요한 봉 수
- `new_swing_signal`: 확정 이벤트
  - `new_high`: 고점 피봇 확정
  - `new_low`: 저점 피봇 확정
- `_all_swings`: 누적 확정 스윙 목록

---

## 피봇 확정 흐름

## 1) 후보 등록 (`후보등록`)

방향 전환 조건이 임계값(`thr_abs`)을 넘으면 후보를 등록한다.

- 상승 방향 탐색 중(`current_direction == 1`) 하락 반전 조건 충족 시 `high` 후보 등록
- 하락 방향 탐색 중(`current_direction == -1`) 상승 반전 조건 충족 시 `low` 후보 등록
- 등록 시 `remaining = confirmation_bars`

추가 필터:

- `min_wave_bars`: 직전 확정 이후 최소 봉 수
- `min_wave_pct`: 최소 파동 퍼센트

---

## 2) 후보 변경 (`후보갱신`)

두 케이스가 있다.

1. **pending window 갱신**
   - `freeze_on_confirm=False`일 때만 후보 가격/인덱스가 최신 극값으로 갱신될 수 있다.
2. **같은 타입 재트리거**
   - 동일 타입 반전 조건이 다시 들어오면 `reason=same_type_retrigger`로 로그만 남긴다.

참고:

- 반대 타입 후보가 새로 들어오면 기존 후보는 `취소(reason=반대후보교체)` 후 새 후보로 교체된다.

---

## 3) 후보 취소 (`취소`)

대표 취소 사유:

- `반대후보교체`: opposite type 후보가 들어와 교체
- `pending_confirm_exception`: pending 처리 중 예외 발생

---

## 4) 피봇 확정 (`확정`)

`pending_confirm.remaining`이 0 이하가 되면 확정된다.

- `high` 후보 확정 시 `new_swing_signal = "new_high"`
- `low` 후보 확정 시 `new_swing_signal = "new_low"`
- 확정 후 `pending_confirm = None`

확정 시 기록되는 정보:

- 피봇봉 시각(`last_swing_*_time`)
- 확정봉 시각(`last_swing_*_confirm_time`)
- 피봇봉→확정봉 지연 봉수(`last_swing_*_lag_bars`)
- 확정봉 O/C(`last_swing_*_open`, `last_swing_*_close`)

---

## `confirmation_bars` 해석

`confirmation_bars`는 "후보 등록 후 몇 봉을 더 확인할지"를 뜻한다.

- `confirmation_bars = 1`: 다음 봉에서 바로 확정될 수 있음
- `confirmation_bars = 5~20`: 후보 등록 후 최소 수 봉 경과 뒤 확정

주의:

- 텔레그램 송출 시각과 확정봉 시각 차이(예: 1분)는 전송 타이밍 영향이며,
  피봇봉→확정봉 지연(`lag_bars`)과는 별개다.
- 실제 후행성은 `피봇봉 시각`과 `확정봉 시각` 또는 `lag_bars`로 판단해야 한다.

---

## 마지막 봉 제외 정책

배치/일부 경로에서 마지막(미완결) 봉은 ZigZag 업데이트에서 제외한다.

- 목적: 미완결 봉 변동으로 확정이 흔들리는 현상 완화
- 구현:
  - `AdaptiveIndicatorManager.compute_from_df()`: 마지막 행은 `zz_tmp.state` 사용
  - `AdaptiveZigZag.compute_from_df()`: 마지막 행에서 `update` 생략
  - `AdaptiveIndicatorManager.update(skip_zigzag=True)`: 실시간에서도 선택적으로 동일 정책 적용 가능

---

## 로그 체계

### 1) `ZZ_PIVOT`

피봇 라이프사이클 이벤트 로그.

- 현재 표준 로그는 `ZZ_ENGINE_*` 계열이며, `ZZ_PIVOT`는 비교 호환을 위해 비활성화 가능하다.
- 이벤트: `후보등록`, `후보갱신`, `취소`, `확정`, `후보상태`
- 공통으로 포함되는 디버그 필드:
  - pending 상태 요약(`dist`, `urgency`, `age`, `waited`, `note`)
  - 누적 확정 요약(`confirmed_count`, `confirmed_tail`)

`confirmed_tail` 예:

- `H@13:15:6390.47 | L@13:59:6362.20`

### 2) `ZZ_CONFIRM`

확정 완료 요약 로그(운영 친화형).

- 피봇 종류(고점/저점), 피봇 가격
- 최근 파동 퍼센트, 구조, 바 타임스탬프

---

## 운영 점검 체크리스트

1. `후보등록` 이후 `후보상태`에서 `remaining`/`urgency`가 합리적으로 진행되는지 확인
2. `취소(reason=반대후보교체)`가 과도하게 빈번한지 확인 (노이즈/임계값 점검)
3. `확정` 시 `confirmed_count` 증가 및 `confirmed_tail` 갱신 확인
4. 텔레그램 본문의 `피봇봉/확정봉/후행지연`이 `ZZ_PIVOT` 로그와 일치하는지 확인
5. `minute_df_rewind`가 자주 발생하면 state reset으로 연속성이 깨질 수 있으므로 우선 데이터 경로 점검

---

## 권장 파라미터 가이드(초안)

- 빠른 반응(단기): `confirmation_bars=1~2`, `min_wave_bars` 낮게
- 안정성 중시(중기): `confirmation_bars=5~10`, `min_wave_bars=3~5`
- 보수적(노이즈 억제): `confirmation_bars=10~20` + `min_wave_pct` 소폭 상향

실운영에서는 종목 변동성/분봉 품질에 따라 함께 튜닝해야 한다.

