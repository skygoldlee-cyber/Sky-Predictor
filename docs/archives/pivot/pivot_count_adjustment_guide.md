---
description: 피봇 수 조정 가이드
---

# KOSPI / KP200 피봇 수 조정 가이드

## 현재 설정 요약 (2026-05-10 갱신)

### futures_zigzag
- `pivot_threshold_min_pct`: 0.4
- `pivot_threshold_max_pct`: 0.5
- `atr_multiplier`: 0.01
- `min_wave_bars`: 1
- `min_wave_pct`: 0.1
- `confirmation_bars`: 1
- `min_wave_atr_ratio`: 0.1
- `use_atr_based_filtering`: false
- `session_min_wave_bars_table`: [[09:00, 09:30, 1], [09:30, 10:30, 2], [10:30, 15:30, 2]]
- **현재 피벗 수**: 약 12개 (원래 28개에서 절반으로 조정됨)

### kospi_zigzag
- `pivot_threshold_min_pct`: 0.3
- `pivot_threshold_max_pct`: 0.5
- `atr_multiplier`: 0.01
- `min_wave_bars`: 1
- `confirmation_bars`: 1
- `min_wave_atr_ratio`: 0.3
- `use_atr_based_filtering`: false

## 피봇 수 조정 가이드

### 피봇 수 늘리기 (현재 futures_zigzag 적용)

현재 `futures_zigzag`는 이미 매우 느슨한 설정입니다:
- `pivot_threshold_min_pct=0.05` (0.05% 반전만 허용)
- `confirmation_bars=0` (즉시 확정)
- `min_wave_bars=0` (봉 간격 제한 없음)
- `use_atr_based_filtering=false` (ATR 필터 비활성화)

**더 늘리려면:**
```json
"futures_zigzag": {
    "pivot_threshold_min_pct": 0.01,  // 0.05 → 0.01 (더 낮게)
    "confirmation_bars": 0,            // 이미 0
    "min_wave_bars": 0,               // 이미 0
    "use_atr_based_filtering": false  // 이미 false
}
```

### 피봇 수 줄이기 (현재 futures_zigzag 적용)

**2026-05-10 수정 내용 (28개 → 12개, 약 절반으로 감소):**
```json
"futures_zigzag": {
    "min_wave_pct": 0.1,              // 0.0 → 0.1 (최소 파동 비율 증가)
    "pivot_threshold_min_pct": 0.4,  // 0.3 → 0.4 (임계값 상향)
    "pivot_threshold_max_pct": 0.5,  // 유지
    "min_wave_bars": 1,              // 유지
    "confirmation_bars": 1,          // 유지
    "use_atr_based_filtering": false, // 유지
    "session_min_wave_bars_table": [
        ["09:00", "09:30", 1],        // 유지
        ["09:30", "10:30", 2],        // 1 → 2 (증가)
        ["10:30", "15:30", 2]         // 1 → 2 (증가)
    ]
}
```

**더 줄이려면:**
```json
"futures_zigzag": {
    "pivot_threshold_min_pct": 0.5,   // 0.4 → 0.5 (추가 상향)
    "pivot_threshold_max_pct": 1.0,   // 0.5 → 1.0
    "confirmation_bars": 2,            // 1 → 2 (확정 지연)
    "min_wave_bars": 2,               // 1 → 2 (봉 간격 제한)
    "use_atr_based_filtering": true,  // false → true (ATR 필터 활성화)
    "session_min_wave_atr_ratio_table": [
        ["08:45", "09:30", 2.0],
        ["09:30", "10:30", 1.5],
        ["10:30", "13:00", 1.2],
        ["13:00", "14:30", 1.0],
        ["14:30", "15:20", 1.2],
        ["15:20", "15:31", 1.5]
    ]
}
```

### KOSPI 피봇 수 조정

**현재 KOSPI 설정 (상대적으로 보수적):**
```json
"kospi_zigzag": {
    "pivot_threshold_min_pct": 0.3,  // 0.3%
    "confirmation_bars": 1,
    "min_wave_bars": 1,
    "use_atr_based_filtering": false
}
```

**KOSPI 피봇 늘리기:**
```json
"kospi_zigzag": {
    "pivot_threshold_min_pct": 0.1,   // 0.3 → 0.1
    "pivot_threshold_max_pct": 0.5,   // 유지
    "confirmation_bars": 0,            // 1 → 0
    "min_wave_bars": 0,               // 1 → 0
    "use_atr_based_filtering": false  // 유지
}
```

**KOSPI 피봇 줄이기:**
```json
"kospi_zigzag": {
    "pivot_threshold_min_pct": 0.5,   // 0.3 → 0.5
    "pivot_threshold_max_pct": 1.5,   // 0.5 → 1.5
    "confirmation_bars": 3,            // 1 → 3
    "min_wave_bars": 5,               // 1 → 5
    "use_atr_based_filtering": true,   // false → true
    "session_min_wave_atr_ratio_table": [
        ["09:00", "09:30", 2.0],
        ["09:30", "10:30", 1.5],
        ["10:30", "13:00", 1.2],
        ["13:00", "14:30", 1.0],
        ["14:30", "15:20", 1.2],
        ["15:20", "15:31", 1.5]
    ]
}
```

## 파라미터 우선순위

피봇 수 조정 시 다음 순서로 조정하는 것을 권장합니다:

1. **`pivot_threshold_min_pct`** - 가장 직접적인 영향 (0.01 ~ 1.0 범위)
2. **`min_wave_pct`** - 최소 파동 비율 (0.0 ~ 1.0 범위)
3. **`confirmation_bars`** - 확정 속도 조절 (0 ~ 5 범위)
4. **`min_wave_bars`** - 봉 간격 제한 (0 ~ 10 범위)
5. **`session_min_wave_bars_table`** - 시간대별 봉 간격 (1 ~ 20 범위)
6. **`use_atr_based_filtering` + `session_min_wave_atr_ratio_table`** - 변동성 적응 필터 (권장)

## 주의사항

현재 `futures_zigzag` 설정은 2026-05-10에 보수적으로 조정되어 피벗 수가 절반으로 줄어들었습니다. 더 줄이거나 늘리려면 위의 가이드를 참조하여 파라미터를 조정하세요. 데이터 자체에 충분한 움직임이 없는 경우 파라미터 조정만으로는 피벗 수를 조절하기 어렵습니다.

## 파라미터 상세 설명

### pivot_threshold_min_pct / pivot_threshold_max_pct
- 방향 전환 임계값 (%)
- 낮을수록 작은 반전도 피봇으로 인정
- 범위: 0.01 ~ 3.0

### confirmation_bars
- 피봇 후보 확정까지 필요한 반전 봉 수
- 0이면 즉시 확정, 높을수록 더 많은 확인 필요
- 범위: 0 ~ 5

### min_wave_bars
- 직전 확정 피봇으로부터 최소 봉 간격
- 높을수록 피봇 간격이 넓어짐
- 범위: 0 ~ 10

### use_atr_based_filtering + session_min_wave_atr_ratio_table
- ATR 기반 변동성 적응 필터
- 시간대별로 다른 ratio 적용 가능
- 변동성이 높을 때 자동으로 threshold 상향
- 권장 설정으로 노이즈 필터링 효과 우수

### min_wave_pct
- 최소 파동 비율 (%)
- 피봇으로 인정되기 위한 최소 가격 변동폭
- 범위: 0.0 ~ 1.0

### session_min_wave_bars_table
- 시간대별 최소 파동 봉 수
- 장 시작 시간대에는 더 낮게, 장 중후반에는 더 높게 설정 가능
- 형식: [시작시간, 종료시간, 봉 수]
- 범위: 1 ~ 20

## 수정 이력

### 2026-05-10
- **목표**: KP200 피벗 수를 절반으로 줄이기 (28개 → 약 14개)
- **수정 내용**:
  - `min_wave_pct`: 0.0 → 0.1
  - `pivot_threshold_min_pct`: 0.3 → 0.4
  - `session_min_wave_bars_table`: [1, 1, 1] → [1, 2, 2]
- **결과**: 피벗 수가 12개로 줄어듦 (약 57% 감소, 목표 달성)
- **참고**: 초기 시도에서 파라미터를 너무 강하게 조정하여 피벗 수가 1개로 줄어든 후, 보수적으로 재조정하여 성공

## 관련 수정 사항

### 데이터 소스 변경 시 ZigZag 설정 업데이트 (gui/engines/chart_engine.py)
- **문제**: 데이터 소스 변경 시 (kospi → futures) `_zz_cfg`가 업데이트되지 않아 올바른 ZigZag 설정이 적용되지 않음
- **해결**: `chart_engine.py`의 데이터 소스 변경 로직에 `_init_zigzag(cfg, data_source)` 호출 추가
- **수정 위치**: `gui/engines/chart_engine.py` (데이터 소스 변경 시 캐시 초기화 및 ZigZag 재초기화 로직)
- **효과**: 데이터 소스 변경 시 해당 소스에 맞는 ZigZag 설정(`futures_zigzag`, `kospi_zigzag`)이 올바르게 적용됨

### config.json의 adaptive_mode와 GUI Adaptive 체크박스 연동 (2026-05-10)
- **문제**: config.json의 adaptive_mode 설정과 GUI의 Adaptive 체크박스가 연동되지 않음
- **해결**:
  - `gui/components/control_bar.py`: `adaptive_enabled` 파라미터 추가 및 체크박스 초기값 설정
  - `gui/chart_viewer.py`: `_build_control_bar`에서 config.adaptive_mode 전달, `_compute_data`에서 동기화
- **수정 파일**:
  - `gui/components/control_bar.py`: adaptive_enabled 파라미터 추가
  - `gui/chart_viewer.py`: config.adaptive_mode와 체크박스 상태 동기화 로직 추가
- **효과**: config.json의 adaptive_mode 값에 따라 애플리케이션 시작 시 체크박스 상태 자동 설정

### 레짐 표시 개선 (2026-05-10)
- **수정 내용**:
  - adaptive_mode와 무관하게 레짐 항상 표시하도록 수정
  - 레짐 정보 라벨을 control_bar에서 summary 그룹박스로 이동
  - 레짐 표시 글자 크기를 10px에서 12px로 증가
- **수정 파일**:
  - `gui/controller.py`: summary_layout에 레짐 정보 라벨 추가, regime_label_callback 추가
  - `gui/chart_viewer.py`: regime_label_callback 사용으로 레짐 라벨 업데이트 로직 수정
  - `gui/components/control_bar.py`: regime_info_label 제거
- **효과**: 레짐 정보가 summary 그룹박스 안에 표시되며, adaptive_mode 설정과 무관하게 항상 표시됨

### optimize_zigzag_lag.py 갱신 완료 (2026-05-10)
- **수정 내용**:
  - `min_wave_pct` 파라미터를 최적화 그리드에 추가 (현재 우선순위 2위)
  - `session_min_wave_bars_table` 파라미터 지원 추가 (현재 우선순위 5위)
  - 시간대별 최적화 결과를 `session_min_wave_bars_table` 형식으로 변환 기능 추가
  - 현재 config.json 설정(`min_wave_pct: 0.1`)을 최적화 범위 기본값으로 반영
  - 파라미터 우선순위에 따라 최적화 순서 조정
  - 데이터 요약 출력 기능 추가 (ATR 포함 주요 지표 요약)
- **수정 파일**:
  - `indicators/optimize_zigzag_lag.py`: evaluate_single, optimize_parameters_on_df, optimize_parameters_with_regime, train_test_split_optimize, optimize_parameters, extract_and_save_best_params, update_config_json 함수 수정
  - 새로운 함수 추가: convert_time_based_to_session_table, update_config_with_session_table, print_data_summary
- **효과**: 최적화 스크립트가 가이드라인의 새로운 파라미터를 지원하며, 시간대별 최적화 결과를 config에 직접 적용할 수 있음. 최적화 전 데이터 요약을 통해 ATR 등 주요 지표를 확인 가능
- **데이터 요약 출력 상세**:
  - 기본 정보: 데이터 크기, 기간, 거래일 수
  - 가격 통계: 종가 범위, 평균, 표준편차, 총 변동률
  - ATR 통계: ATR 값, ATR 백분율, 평균 ATR, ATR 표준편차 (기본 기간: 14)
  - 거래량 통계: 평균, 최대, 최소 거래량
  - 결측치 확인

### 실시간 차트 갱신 장 시간 수정 (2026-05-11)
- **문제**: 장 시간이 KP200 선물(08:45~15:45), KOSPI 지수(09:00~15:30)이지만, 코드에서 일괄적으로 09:00~15:30으로 설정되어 KP200 선물에서 장 전으로 인식되어 자동 갱신이 안됨
- **해결**: `gui/chart_viewer.py`의 `_is_market_closed()` 함수에서 선택된 플롯에 따라 장 시간을 다르게 적용하도록 수정
  - KP200 선물: 08:45 ~ 15:45
  - KOSPI 지수: 09:00 ~ 15:30
- **수정 파일**: `gui/chart_viewer.py`: _is_market_closed() 함수 수정
- **효과**: 선택된 플롯에 따라 올바른 장 시간이 적용되어 실시간 차트 자동 갱신 정상 작동

### 실시간 고가/저가 캔들 업데이트 수정 (2026-05-11)
- **문제**: 현재 분봉의 고가/저가가 실시간으로 업데이트되지 않고 분봉이 지나야 표시됨. 분봉 병합 캐시 로직에서 현재 분봉의 틱이 추가되어도 캐시 키가 변하지 않아 업데이트되지 않음
- **해결**: `data/tick_processor.py`의 캐시 키에 현재 분봉 틱 수를 포함하도록 수정
  - get_futures_minute_df: 캐시 키에 현재 분봉 틱 수 추가
  - get_spot_index_minute_df: 캐시 키에 현재 분봉 틱 수 추가
- **수정 파일**: `data/tick_processor.py`: get_futures_minute_df, get_spot_index_minute_df 함수 수정
- **효과**: 현재 분봉의 틱이 추가될 때마다 캐시 키가 변하여 실시간 고가/저가 업데이트 정상 작동

### 데이터 부족 시 피봇 캐시 저장 방지 (2026-05-11)
- **문제**: KOSPI 장 시작 후 데이터가 21봉 이상 쌓이기 전에 피봇 캐시가 생성되어, 데이터 부족으로 Low 피봇만 인식하고 캐시되는 문제 발생. 이로 인해 "캐시된 피봇이 모두 동일한 타입(L)입니다" 경고 발생
- **해결**: `gui/engines/chart_engine.py`에서 데이터가 최소 20봉 이상일 때만 피봇 캐시와 ZigZag 상태 캐시를 저장하도록 수정
- **수정 파일**: `gui/engines/chart_engine.py`: 피봇 캐시 저장 로직에 데이터 최소 봉 수 검증 추가
- **효과**: 데이터 부족 시 잘못된 피봇 캐시가 생성되지 않아 정상적인 피봇 확정 가능

### 피봇 후보 등록 없이 바로 확정되는 경우 로그 추가 (2026-05-11)
- **문제**: 피봇 후보 등록 과정 없이 바로 확정되는 피봇을 감지할 수 없어 디버깅 어려움
- **해결**: `indicators/adaptive_zigzag.py`의 `_emit_cross_project_debug_logs` 메서드에서 action이 "확정"이고 mode가 "초기범위"인 경우 별도 경고 로그 출력
- **수정 파일**: `indicators/adaptive_zigzag.py`: _emit_cross_project_debug_logs 메서드에 피봇 후보 등록 없이 확정되는 경우 로그 추가
- **효과**: 피봇 후보 등록 없이 바로 확정되는 경우를 `[ZZ][확정-무후보]` 로그로 출력 가능

### 피봇 이벤트 로그 레벨 조정 (2026-05-11)
- **문제**: `main.py`의 `ZZLogFilter`가 `[ZZ]`로 시작하는 INFO 레벨 로그를 필터링하여 피봇 이벤트 로그가 보이지 않음
- **해결**: `indicators/adaptive_zigzag.py`의 `_emit_cross_project_debug_logs` 메서드에서 모든 피봇 이벤트 로그를 INFO에서 WARNING 레벨로 변경
- **수정 파일**: `indicators/adaptive_zigzag.py`: 후보등록, 후보갱신, 취소, 확정, 피봇목록 로그 레벨을 WARNING으로 변경
- **효과**: ZZLogFilter 필터링을 우회하여 모든 피봇 이벤트 로그가 정상적으로 출력됨

### QtLogHandler 필터 우회 추가 (2026-05-11)
- **문제**: ZZLogFilter가 로거 레벨에서 로그를 필터링하면 QtLogHandler에 도달하기 전에 차단되어 GUI 로그창에 피봇 이벤트 로그가 보이지 않음
- **해결**: `gui/qt_logging.py`의 QtLogHandler에 항상 True를 반환하는 필터를 추가하여 핸들러 레벨에서 ZZ 필터를 우회
- **수정 파일**: `gui/qt_logging.py`: QtLogHandler.__init__에 _zz_bypass_filter 필터 추가
- **효과**: 로거 레벨에서 ZZLogFilter가 필터링하더라도 QtLogHandler를 통해 GUI 로그창에 모든 피봇 이벤트 로그 표시

### 현재가 라인 중첩 문제 해결 (2026-05-11)
- **문제**: 현재가 라인이 업데이트될 때 과거 라인들이 제거되지 않고 중첩되어 표시됨
- **해결**: `gui/renderers/fplt_renderer.py`의 _render_current_price_line 메서드에서 기존 라인 업데이트 실패 시 강제 제거 로직 추가
- **수정 파일**: `gui/renderers/fplt_renderer.py`: 기존 라인 업데이트 실패 시 ViewBox에서 직접 제거하고 scene에서도 제거하는 로직 추가
- **효과**: 현재가 라인이 중첩되지 않고 하나만 표시됨

### 피봇 마커 인덱스 중첩 문제 해결 (2026-05-11)
- **문제**: 피봇 마커가 차트의 앞쪽으로 몰려서 표시됨 (인덱스 매핑 오류). 피봇 체크박스 해제 후 다시 켜면 정상 인덱스에 보임
- **해결**: `gui/renderers/fplt_renderer.py`의 _render_pivots 메서드에서 기존 피봇 마커 제거 후 새로 렌더링, 인덱스 범위 검증 강화
- **수정 파일**: `gui/renderers/fplt_renderer.py`: _render_pivots 시작 시 기존 _zz_ 마커 모두 제거, 인덱스 매핑 전 유효성 검증 추가
- **효과**: 피봇 마커가 올바른 인덱스에 표시되고 중첩되지 않음

### 불필요한 피벗 렌더링 로그 제거 (2026-05-11)
- **문제**: 피벗 렌더링 조건 로그가 매번 출력되어 차트 깜빡임 유발
- **해결**: `gui/renderers/fplt_renderer.py`에서 불필요한 INFO 로그를 DEBUG로 변경, 실제 렌더링 호출 시에만 INFO 로그 출력
- **수정 파일**: `gui/renderers/fplt_renderer.py`: 피벗 렌더링 조건 로그 제거, _render_pivots 내부 로그 DEBUG로 변경
- **효과**: 불필요한 로그 출력 제거로 차트 깜빡임 감소

### 틱 데이터 업데이트 시 깜빡임 해결 (2026-05-11)
- **문제**: 동일 분봉 갱신 시 차트 전체가 깜빡거림 (마지막 캔들만 갱신하더라도 전체 렌더링 발생)
- **해결**: `gui/renderers/fplt_renderer.py`에서 마지막 캔들 직접 업데이트 로직 추가, 캔들 객체 내부 데이터 직접 수정
- **수정 파일**: `gui/renderers/fplt_renderer.py`: _render_candles에 last_only 파라미터 추가, 캔들 객체 data 속성 직접 수정
- **효과**: 동일 분봉 내 틱 업데이트 시 마지막 캔들만 직접 수정하여 전체 깜빡임 제거

### 백테스트 모듈 import 경로 수정 (2026-05-11)
- **문제**: 백테스트 실행 시 ModuleNotFoundError: No module named 'prediction.backtest_pivot_signals'
- **해결**: `scripts/run_daily_backtest.py`에서 import 경로를 prediction.backtest.backtest_pivot_signals로 수정
- **수정 파일**: `scripts/run_daily_backtest.py`: import 경로 수정
- **효과**: 백테스트 모듈 정상 import
