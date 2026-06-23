# Runtime Documentation Index

이 폴더는 프로젝트의 **실시간 동작(runtime)** 경로에서 실제로 호출되는(또는 호출될 수 있는) 핵심 모듈들의 **클래스/함수**를 모듈군별로 요약합니다.

- 대상 범위: `main.py`, `config.py`, `tick_processor.py`, `tick_normalizer.py`, `ebest_*.py`, `prediction/`(파이프라인/피처/컨텍스트/LLM)
- 목적: “어디가 엔트리포인트이고, 어떤 데이터가 어디로 흐르며, 무엇을 수정해야 하는지” 빠르게 찾기

## 문서 목록

- `config.md`
  - `config.json` 구조/로드/검증(AppConfig)
- `ticks.md`
  - `RealTimeTickProcessor` 및 tick 정규화
- `ebest.md`
  - eBest live orchestration, REST helpers, callbacks, 옵션 심볼 선택
- `prediction.md`
  - 예측 파이프라인(`PredictionPipeline`)과 하위 모듈(피처/컨텍스트/LLM)
- `adaptive_indicator.md`
  - Adaptive SuperTrend/ZigZag 통합 모듈(ADAPT 피처 + LLM 컨텍스트)
  - **SuperTrend 데이터 소스 전환 시 전체 재계산**: 데이터 소스(KOSPI ↔ KP200) 변경 시 SuperTrend 캐시와 상태를 초기화하여 전체 재계산 수행. `gui/engines/chart_engine.py`의 `compute()` 메서드에서 `force_recompute=True` 시 `_st_cache_sig`, `_st_cache_values`, `_st_cache_dirs`, `_st_fed_bars`를 초기화하고 `AdaptiveSuperTrend.reset()` 호출.
  - **부록**: `adaptive_indicator_improvements.md`, `adaptive_indicator_parameters.md` 통합
- `volume_imbalance.md`
  - FH0 오더북 기반 Volume Imbalance(OBI) 계산/버퍼링/사용처 정리
- `telegram.md`
  - 텔레그램 알림/명령 수신(`telegram_notifier.py`), CLI/GUI 연동
  - v4 전용 프리미엄 블리드 독립 알림(BleedMonitor 스레드)
  - **부록**: `runtime_telegram.md` 통합
- `realtime_trading.md`
  - 실시간 데이터 수집, 신호 생성, 자동 주문 연동
  - **부록**: `main.md`, `Market_Open_Subscription_Flow.md`, `live_run_troubleshooting.md` 통합
- `RECOMMENDED_STRATEGY.md`
  - 최종 전략 가이드 및 피봇/롱-또는-플랫 분석 부록
- `pivot_refactor_plan.md`
  - 피봇 전략 파일 통합 완료 보고서

### 통합/제거된 문서

| 원본 문서 | 통합 대상 | 비고 |
|-----------|-----------|------|
| `adaptive_indicator_improvements.md` | `adaptive_indicator.md` 부록 A | ✅ 제거됨 |
| `adaptive_indicator_parameters.md` | `adaptive_indicator.md` 부록 B | ✅ 제거됨 |
| `runtime_telegram.md` | `telegram.md` 상세 명령/공통 규칙 | ✅ 제거됨 |
| `main.md` | `realtime_trading.md` 부록 A | ✅ 제거됨 |
| `Market_Open_Subscription_Flow.md` | `realtime_trading.md` 부록 B | ✅ 제거됨 |
| `live_run_troubleshooting.md` | `realtime_trading.md` 부록 C | ✅ 제거됨 |
| `LONG_OR_FLAT_IMPROVEMENTS.md` | `RECOMMENDED_STRATEGY.md` 부록 A | ✅ `archives/legacy/docs/`로 이동 |
| `LONG_OR_FLAT_IMPROVEMENTS_BACKTEST.md` | `RECOMMENDED_STRATEGY.md` 부록 B | ✅ `archives/legacy/docs/`로 이동 |
| `PIVOT_REVERSAL_VIABILITY.md` | `RECOMMENDED_STRATEGY.md` 부록 C | ✅ `archives/legacy/docs/`로 이동 |
| `PIVOT_REVERSAL_VERIFICATION.md` | `RECOMMENDED_STRATEGY.md` 부록 D | ✅ `archives/legacy/docs/`로 이동 |

## 전체 흐름(요약)

1. `main.py`가 `config.py`로 설정을 로드
2. `PredictionPipeline` 생성 (`option_feature_set` v1~v4 선택)
3. `ebest_live.py`가 실시간 구독을 등록
4. `ebest_callbacks.py` 콜백이 tick을 정규화(`tick_normalizer.py`) 후 pipeline으로 전달
5. `tick_processor.py`가 분봉/옵션 스냅샷을 축적
6. `adaptive_indicator/`가 (설정 시) 분봉 기반 ADAPT(28) 피처 + LLM 컨텍스트를 생성
7. `prediction/`이 수치 예측 + LLM 판단 + 컨텍스트를 생성
   - v3/v4: `[PARITY_ANALYSIS]` 섹션 LLM 컨텍스트 포함
   - v4: `[PREMIUM_BLEED]` 섹션 LLM 컨텍스트 포함, 가드레일 4단계 적용
8. `PipelineTelegramBridge`가 결과를 텔레그램으로 전송
   - v4: BleedMonitor 스레드가 5초 주기로 프리미엄 블리드 독립 알림 병행 전송

## 옵션 피처 버전별 feature_dim

`prediction.option_feature_set`과 `adaptive_indicator.enabled` 조합에 따라 모델 입력 차원이 달라집니다.

| option_feature_set | adaptive=false | adaptive=true |
|---|---|---|
| v1 (OPT 7) | 19 | 47 |
| v2 (OPT 16) | 28 | 56 |
| v3 (OPT 23) | 35 | 63 |
| v4 (OPT 29) | 41 | 69 |

> ⚠️ feature_dim이 바뀌면 dataset 재생성 및 모델 재학습이 필요합니다.

## 테스트(스모크)

- 전체 스모크(우산) 테스트: `python -m pytest -q tests/test_smoke.py`
- 세부 스모크:
  - `tests/test_adaptive_indicator_smoke.py`
  - `tests/test_prediction_smoke.py`

## 옵션 피처 설계 문서

| 문서 | 내용 |
|---|---|
| [`../call_put_parity_divergence_design.md`](../call_put_parity_divergence_design.md) | v3 콜-풋 패리티 이탈 지표 설계 |
| [`../premium_bleed_design.md`](../premium_bleed_design.md) | v4 프리미엄 블리드 지표 설계 |
