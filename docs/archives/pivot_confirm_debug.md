# KOSPI/KP200 피봇 확정 디버그 가이드

## 목적

인덱스 차트의 적응형 ZigZag에서 아래 이슈를 빠르게 진단하기 위한 운영 문서.

- `ZZ_CONFIRMED_PIVOTS` 로그와 차트 마커가 다르게 보임
- `ZZ_CANDIDATE` 로그가 안 보이거나 드물게만 보임
- 장중에 확정 피봇이 흔들려 보임

---

## 핵심 경로

- 계산/로그: `views/charts/technical_analysis.py`
- 차트 마커/폴리라인 렌더: `views/charts/plot_manager.py`
- 헤드리스 확정 처리: `services/kospi_adaptive_zz_confirm.py`
- 세션 슬라이스: `services/kospi_zz_rth_slice.py`
- 적응형 엔진: `views/charts/UnifiedTA.py`

---

## 로그별 의미

### `ZZ_CANDIDATE`

- 의미: 현재 미확정(unconfirmed) 후보 버킷 요약
- 포맷: `pending=<개수> ... cand=[HH:MM,가격,타입; ...]`
- 출력 조건(현재):
  - 후보 문자열이 바뀌면 출력
  - 또는 `df_last_dt`(새 분봉) 변경 시 출력
  - 최소 간격 스로틀 적용 (`ZZ_KOSPI_CONFIRMED_PIVOTS_LOG_MIN_SEC`)

### `ZZ_CONFIRMED_PIVOTS`

- 의미: 확정(confirmed) 버킷 요약
- 포맷: `confirmed=<개수> ... conf=[1) HH:MM 타입 가격 | ...]`
- 주의: 시가 anchor는 **차트 마커와 동일하게 제외**한 개수/목록으로 출력하도록 보정됨

### `ZZ Points changed`

- 의미: 차트 업데이트 관점에서 피봇 타임스탬프/버킷 변화
- 현재는 confirmed 상세에서 anchor를 제외해 `ZZ_CONFIRMED_PIVOTS`/마커와 정합 유지

---

## 자주 발생한 원인과 조치

### 1) 로그의 확정 목록 vs 차트 마커 불일치

원인:
- `_inject_open_anchor_pivot()`가 넣는 `anchor_idx`는 폴리라인 시작점 용도
- 차트 마커는 anchor를 표시하지 않음
- 과거에는 로그 요약에 anchor가 포함되어 숫자/목록이 어긋날 수 있었음

조치:
- 로그 포맷터에서 anchor 제외
- `confirmed=` 개수도 anchor 제외 기준으로 계산

### 2) `ZZ_CANDIDATE`가 안 보임

원인:
- 후보 문자열이 동일하면 출력 안 되는 구조였음

조치:
- 분봉(`df_last_dt`)이 바뀌면 후보가 같아도 로그 재출력하도록 보강
- 스팸 방지를 위한 최소 간격은 유지

### 3) 확정 피봇이 흔들림(09:07 ↔ 09:01 등)

가능 원인:
- 장중 데이터 재수신/보정으로 과거 OHLC가 변경되면 adaptive ZZ 재계산 결과가 변동
- `bfill()`로 앞쪽 NaN이 뒤 데이터로 채워져 초반 ATR/스윙 왜곡

조치:
- `UnifiedTA` 적응형 전처리에서 `bfill()` 제거, `ffill()`만 유지

---

## 운영 체크리스트

1. 같은 시각에 아래 3개를 같이 본다.
   - `ZZ_CANDIDATE`
   - `ZZ_CONFIRMED_PIVOTS`
   - 차트의 confirmed/unconfirmed 마커
2. `reason=`으로 경로를 구분한다.
   - `headless:ij` (헤드리스)
   - 차트 갱신 경로 (`unknown` 또는 차트 reason)
3. 불일치 시 먼저 anchor 여부 확인
   - `anchor_idx`가 있는 프레임인지 확인
4. 후보 로그 누락 시
   - 새 분봉(`df_last_dt`) 진입 후에도 후보 로그가 없는지 확인
   - 로그 최소 간격 설정값 확인
5. 흔들림 재현 시
   - 직전 `KOSPI refetched`/재구독 이벤트 유무 확인
   - 같은 분봉에서 OHLC 재작성 여부 확인

---

## 관련 설정

- `ADAPTIVE_ZZ_CONFIRMATION_BARS`
  - 크면 확정이 늦어지고 흔들림 체감은 줄 수 있음
- `ADAPTIVE_ZZ_EXCLUDE_LAST_BAR_LIVE`
- `ADAPTIVE_ZZ_EXCLUDE_LAST_OPEN_BARS`
- `ZZ_KOSPI_CONFIRMED_PIVOTS_LOG_MIN_SEC`

---

## 빠른 확인 예시

- 기대 상황:
  - `ZZ_CONFIRMED_PIVOTS`의 `confirmed=N`과 차트 confirmed 마커 개수가 동일
  - `ZZ_CANDIDATE`는 후보 변화가 없더라도 새 분봉 시 간헐 재출력
- 비정상 의심:
  - confirmed 로그는 늘었는데 마커가 그대로
  - 분봉이 여러 개 지났는데 candidate 로그가 전혀 없음

