# ZigZag 피봇 파라미터 통일 작업 보고서

**대상 프로젝트**: SkyEbest / PatchTST  
**대상 파일**: `kospi_indicators/kospi_indicators/adaptive_zigzag.py` 외 6개  
**작업 일시**: 2026-04-24

---

## 1. 작업 배경

두 프로젝트가 공유하는 `AdaptiveZigZag` 알고리즘의 파라미터 설정 방식과 필드명이 달라, 동일한 로직임에도 동작 값이 달라지는 문제가 있었음. 단일 원칙으로 통일하여 양쪽 프로젝트에서 동일한 피봇 탐지 결과를 보장하는 것이 목표.

---

## 2. 수정 전 주요 차이점

| 파라미터 | SkyEbest (수정 전) | PatchTST (수정 전) | 문제 |
|---|---|---|---|
| 임계값 하한 필드명 | `min_threshold_pct` | `pivot_threshold_min_pct` | 필드명 불일치 |
| 임계값 상한 필드명 | `max_threshold_pct` | `pivot_threshold_max_pct` | 필드명 불일치 |
| `atr_period` | `10` (UnifiedTA 하드코딩) | `14` | 값 불일치 — 과도하게 민감 |
| `unifiedta_fallback` 기본값 | `0.2 / 5.0` | `0.3 / 3.0` | 값 불일치 |
| 장초반 ATR 설정 주입 방식 | `config.ini` 전역 읽기 | `Config` 필드 캡슐화 | 구조 불일치 |
| `confirmation_bars` fallback | dataclass=`1`, config fallback=`2` | — | config 로드 시 불일치 |
| `max_wait_bars` | SkyEbest에만 존재 | 없음 | 기능 비대칭 |

---

## 3. 적용된 수정 내용 (7개 파일)

### 3-1. SkyEbest — `kospi_indicators/kospi_indicators/adaptive_zigzag.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| Config 필드명 | `min_threshold_pct: 0.3` | `pivot_threshold_min_pct: 0.3` | 필드명 변경 |
| Config 필드명 | `max_threshold_pct: 3.0` | `pivot_threshold_max_pct: 3.0` | 필드명 변경 |
| Config 신규 필드 | — | `early_session_start_time: "09:00"` | 신규 추가 |
| Config 신규 필드 | — | `early_session_end_time: "09:30"` | 신규 추가 |
| Config 신규 필드 | — | `early_session_atr_multiplier_max: 8.0` | 신규 추가 |
| `_get_time_based_multiplier_max()` | 독립 메서드 (config.ini 읽기) | `_calc_threshold_pct()` 내부로 인라인 통합 | 구조 변경 |
| `_calc_threshold_pct()` 내 참조 | `cfg.min_threshold_pct` / `cfg.max_threshold_pct` | `cfg.pivot_threshold_min_pct` / `cfg.pivot_threshold_max_pct` | 필드명 변경 |

**핵심 변경 이유**  
- `_get_time_based_multiplier_max()`는 외부 `config` 모듈을 직접 임포트하여 전역 `config.ini`에 의존했음. PatchTST처럼 `Config` 객체 필드를 직접 참조하는 방식으로 교체하여 외부 의존 제거 및 다중 심볼 인스턴스 대응.

---

### 3-2. SkyEbest — `views/charts/UnifiedTA.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| `AdaptiveZigZagConfig` 로컬 dataclass 필드명 | `min_threshold_pct: 0.3` / `max_threshold_pct: 3.0` | `pivot_threshold_min_pct: 0.3` / `pivot_threshold_max_pct: 3.0` | 필드명 변경 |
| 인스턴스화 `atr_period` | `atr_period=10` | `atr_period=14` | 값 변경 |
| 인스턴스화 키워드 인자 | `min_threshold_pct=min_thr` / `max_threshold_pct=max_thr` | `pivot_threshold_min_pct=min_thr` / `pivot_threshold_max_pct=max_thr` | 필드명 변경 |
| dict 리터럴 내 `atr_period` | `"atr_period": int(10)` | `"atr_period": int(14)` | 값 변경 |

---

### 3-3. SkyEbest — `views/charts/unifiedta_fallback.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| Config 필드명 + 기본값 | `min_threshold_pct: 0.2` / `max_threshold_pct: 5.0` | `pivot_threshold_min_pct: 0.3` / `pivot_threshold_max_pct: 3.0` | 필드명 변경 + 값 변경 |
| `_calc_threshold_pct()` 내 참조 | `cfg.min_threshold_pct` / `cfg.max_threshold_pct` | `cfg.pivot_threshold_min_pct` / `cfg.pivot_threshold_max_pct` | 필드명 변경 |

---

### 3-4. PatchTST — `kospi_indicators/kospi_indicators/adaptive_zigzag.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| Config 신규 필드 | — | `max_wait_bars: int = 0` | 신규 추가 |
| 내부 예외 클래스 | — | `class _MaxWaitCancel(Exception)` | 신규 추가 |
| `update()` 3-a 블록 | 타임아웃 로직 없음 | pending 진입 직후 대기 봉수 체크 → 초과 시 `_pivot_event_emit("취소")` 후 `raise _MaxWaitCancel()` | 신규 추가 |
| `update()` except 절 | `except Exception as exc:` | `except _MaxWaitCancel: pass` → 기존 `except Exception` | 신규 추가 |

**핵심 변경 이유**  
- SkyEbest `[FIX-7]` 이식. PatchTST는 `azz_pending_age` 등 pending 상태를 ML 피처로 활용하므로, 장기 미확정 후보가 누적될 경우 피처 신뢰도 저하 문제가 더 직접적으로 발생함.
- `_MaxWaitCancel` 내부 예외를 별도로 두어 정상 취소 경로와 오류 취소 경로를 분리, 불필요한 경고 로그 방지.

---

### 3-5. PatchTST — `config.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| `AdaptiveZigZagSettings` 신규 필드 | — | `max_wait_bars: int = 0` | 신규 추가 |
| `confirmation_bars` config 로드 fallback | `fallback=2` | `fallback=1` | 값 변경 |
| `confirmation_bars` validation 참조값 | `getattr(zz, "confirmation_bars", 2)` | `getattr(zz, "confirmation_bars", 1)` | 값 변경 |

---

### 3-6. PatchTST — `prediction/pipeline.py` / `prediction/data_builder.py`

| 대상 | 변경 전 | 변경 후 | 유형 |
|---|---|---|---|
| `AdaptiveZigZagConfig` 인스턴스화 인자 | `max_wait_bars` 인자 없음 | `max_wait_bars=int(zz.get("max_wait_bars", 0) or 0)` | 신규 추가 |

---

## 4. 수정 후 파라미터 현황

### AdaptiveZigZagConfig — 22개 필드 전부 일치

| 필드 | 값 | 양쪽 일치 여부 |
|---|---|---|
| `atr_period` | `14` | ✅ |
| `er_period` | `10` | ✅ |
| `atr_multiplier` | `1.5` | ✅ |
| `atr_multiplier_min` | `1.0` | ✅ |
| `atr_multiplier_max` | `4.0` | ✅ |
| `pivot_threshold_min_pct` | `0.3` | ✅ |
| `pivot_threshold_max_pct` | `3.0` | ✅ |
| `confirmation_bars` | `2` (dataclass 기본) | ✅ |
| `freeze_on_confirm` | `True` | ✅ |
| `max_wait_bars` | `0` (무제한) | ✅ |
| `min_wave_bars` | `5` | ✅ |
| `min_wave_pct` | `0.0` | ✅ |
| `major_swing_ratio` | `2.0` | ✅ |
| `max_swings` | `20` | ✅ |
| `structure_lookback_swings` | `8` | ✅ |
| `structure_points` | `3` | ✅ |
| `cluster_tolerance_pct` | `0.3` | ✅ |
| `early_session_start_time` | `"09:00"` | ✅ |
| `early_session_end_time` | `"09:30"` | ✅ |
| `early_session_atr_multiplier_max` | `8.0` | ✅ |
| `fib_ratios` | `[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]` | ✅ |
| `atr_multiplier` (기본) | `1.5` | ✅ |

---

## 5. 설계 의도에 따른 잔존 차이 (통일 대상 아님)

아래 차이는 각 프로젝트의 아키텍처 요구사항에 의한 것으로, 피봇 탐지 계산 결과에 영향을 주지 않으며 통일할 필요가 없다.

### 5-1. 로그 체계

| 항목 | SkyEbest | PatchTST |
|---|---|---|
| 방식 | 콜백 주입 (`zz_log_fn`, `zz_log_chart_key`, `zz_idx_to_time_fn`, `zz_status_fn`) | `bool` 플래그 + Python `logging` (`pivot_lifecycle_log`, `pivot_lifecycle_log_prefix`) |
| 기능 | 피봇 생애주기 로그 (등가) | 피봇 생애주기 로그 (등가) |
| 통일 불필요 이유 | SkyEbest는 PySide6 차트 UI에 직접 콜백을 주입하는 구조가 필수. logging 방식으로 대체 불가 | |

### 5-2. `update()` 메서드 시그니처

| 항목 | SkyEbest | PatchTST |
|---|---|---|
| 시그니처 | `update(high, low, close)` | `update(high, low, close, bar_time=None, open=0.0)` |
| 추가 인자 목적 | — | `bar_time`: HH:MM 매핑 → 텔레그램·로그 시각 표기 / `open`: 확정봉 OHLC 기록 |
| 피봇 계산 영향 | 없음 | 없음 |

### 5-3. `SwingPoint` 확장 필드

| 필드 | SkyEbest | PatchTST |
|---|---|---|
| `confirmed_at_idx` | 없음 | 있음 (`int = -1`) |
| `confirmed_close` | 없음 | 있음 (`float = 0.0`) |
| 용도 | — | 확정봉 인덱스·종가 기록 (텔레그램 송출·리포트) |

### 5-4. `ZigZagState` 확장 필드

PatchTST는 피봇 확정 시각(`last_swing_high_time` 등), 확정봉 OHLC, `cancelled_candidates`, `confirmed_pivot_count` 등을 추가로 보유. 모두 텔레그램·LLM 컨텍스트 전용 메타 정보이며 피봇 계산 출력값과 무관.

### 5-5. `get_transformer_features()` 반환 피처 수

| 항목 | SkyEbest | PatchTST |
|---|---|---|
| 반환 키 수 | 15개 | 21개 |
| 추가 피처 | — | `azz_last_high`, `azz_last_low` (현재가 대비 정규화), `azz_pending_type`, `azz_pending_dist`, `azz_pending_urgency`, `azz_pending_age` |
| 추가 이유 | — | ZigZag 후행성 보완 — 확정 전 pending 상태를 ML이 조기 인식하도록 노출 |
| 통일 불필요 이유 | SkyEbest에는 해당 피처를 소비하는 ML 파이프라인 없음 | |

### 5-6. `seed_anchor()` 메서드

| 항목 | SkyEbest | PatchTST |
|---|---|---|
| 존재 여부 | 없음 | 있음 |
| 용도 | — | 장 시작 09:00 시가를 앵커 피봇으로 주입, 초기 수렴 가속 |
| 통일 불필요 이유 | SkyEbest는 차트 기록 데이터로 초기 피봇이 충분히 생성되어 앵커 주입이 불필요 | |

---

## 6. 수정 파일 목록

```
zigzag_param_unified.zip
├── SkyEbest/
│   ├── kospi_indicators/kospi_indicators/adaptive_zigzag.py
│   └── views/charts/
│       ├── UnifiedTA.py
│       └── unifiedta_fallback.py
└── PatchTST/
    ├── config.py
    ├── kospi_indicators/kospi_indicators/adaptive_zigzag.py
    └── prediction/
        ├── pipeline.py
        └── data_builder.py
```

---

## 7. 검증

수정 후 7개 파일 전부 Python 문법 검사(`py_compile`) 통과 확인.

```
SkyEbest kospi_indicators/adaptive_zigzag.py  OK
SkyEbest views/charts/UnifiedTA.py            OK
SkyEbest views/charts/unifiedta_fallback.py   OK
PatchTST kospi_indicators/adaptive_zigzag.py  OK
PatchTST config.py                            OK
PatchTST prediction/pipeline.py               OK
PatchTST prediction/data_builder.py           OK
```
