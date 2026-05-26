# Adaptive ZigZag 피봇 판정 로직 통합 문서

본 문서는 아래 두 문서를 통합한 비교 가이드다.

- 기준 문서 A: `pivot_confirmation_logic.md` (타 프로젝트 기준)
- 기준 문서 B: `docs/pivot_confirmation_logic_skyebest.md` (현재 SkyEbest 기준)

---

## 공통 로직

두 프로젝트 모두 핵심 엔진 개념은 동일하다.

- 엔진: `pending_confirm` 기반 확정 모델
- 후보 필드: `type`, `idx`, `price`, `remaining`
- 등록: 반전 임계값(`thr_abs`) + 파동 필터(`min_wave_bars`, `min_wave_pct`) 통과 시
- 확정: `remaining <= 0`
- 취소/교체: 반대 후보 진입, 예외, timeout(`max_wait_bars`) 경로
- 후행성 평가지표: `lag_bars`(피봇봉→확정봉 경과 봉수)

요약하면, 두 프로젝트 모두 "후보 등록 -> 대기 -> 확정/취소" 상태 머신은 같다.

---

## 프로젝트별 차이

### 1) 런타임 설정 주입

- 타 프로젝트 문서(A)
  - `confirmation_bars` 일반론 중심(1~20 가능)
- SkyEbest(B)
  - 실제 런타임에서 `UnifiedTA`가 `settings.ADAPTIVE_ZZ_CONFIRMATION_BARS`를 주입
  - 기본 운영값은 `1`에 맞춰져 있어 체감 지연이 짧음

### 2) 차트/헤드리스 동기화 보호 로직

- 타 프로젝트 문서(A)
  - 엔진·파이프라인 중심 설명
- SkyEbest(B)
  - 차트 표시 정합을 위한 보호층 포함:
    - `PlotManager._stabilize_index_adaptive_confirmed()`
    - `services/kospi_adaptive_zz_confirm.py`의 스냅샷 유지/회귀 보정
    - `sync_kospi_headless_zz_snapshot_from_chart(...)`

### 3) 로그 체계

- 타 프로젝트 문서(A)
  - `ZZ_PIVOT`, `ZZ_CONFIRM` 중심
- SkyEbest(B)
  - 운영 로그가 다음으로 확장:
    - `ZZ_CANDIDATE`, `ZZ_CONFIRMED_PIVOTS`, `ZZ Points changed`
    - `ZZ_ENGINE_REGISTER/UPDATE/CANCEL/CHAIN`
  - `event_id` 및 dedupe로 중복 최소화
  - `ZZ_ENGINE_CHAIN`에서 final(confirm/cancel) + `lag_bars` 요약

### 4) 실무 해석 포인트

- 타 프로젝트 문서(A)
  - `confirmation_bars` 가중치 설명이 중심
- SkyEbest(B)
  - 실제 운영 체감은 아래 3요인 영향이 큼:
    - 후보 등록 시점(임계/파동 조건)
    - 후보 취소·교체 빈도
    - 차트/헤드리스 동기화 보호 로직

---

## 결론

- **알고리즘 본체는 동일 계열**이다.
- **운영 체감 차이는 SkyEbest의 런타임 설정(`confirmation_bars=1` 계열)과 동기화/로그 보호층**에서 발생한다.
- 따라서 "어느 쪽이 맞다"보다, 동일 엔진 위에 **운영 정책을 어떻게 얹었는지**가 핵심 차이이다.

