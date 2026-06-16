# SkyPredictor 문서 인덱스

전체 문서 목록 및 설명입니다.

## 메인 문서 (docs/)

### 시스템 아키텍처
- **ARCHITECTURE.md** - 시스템 아키텍처 상세
- **docs_README.md** - 프로젝트 개요 및 상세 사용 가이드

### 기술 가이드
- **ADAPTIVE_INDICATOR_GUIDE.md** - 적응형 지표 가이드
- **ML_PREDICTION_GUIDE.md** - 머신러닝 예측 시스템 가이드
- **TFT_DUAL_MODEL_DESIGN_GUIDE.md** - TFT 설계 가이드
- **PIVOT_BASED_TRADING_STRATEGY.md** - 피봇 기반 트레이딩 전략
- **PIVOT_ML_ALGORITHM_GUIDE.md** - 피봇 예측 ML 알고리즘 가이드 (v2.0)
- **CONFIG_REFERENCE_GUIDE.md** - config.json 전체 설정 가이드
- **DUAL_MODE_GUIDE.md** - 듀얼 모드 구조 가이드
- **ZIGZAG_PIVOT_COMPREHENSIVE_GUIDE.md** - ZigZag 피봇 종합 가이드 (병합)
- **PIVOT_DETECTION_COMPARISON_GUIDE.md** - 피봇 탐지 시스템 비교 가이드 (병합)
- **PIVOT_INFO_PANEL_GUIDE.md** - 피봇 정보 패널 구현 및 배치 가이드 (병합)

### 해외 선물
- **OVERSEAS_FUTURES_ADAPTIVE_ZIGZAG_APPLICABILITY.md** - 해외 선물 적응형 ZigZag 적용성
- **OVERSEAS_FUTURES_RECOMMENDATIONS.md** - 해외 선물 권장사항

### 튜닝 가이드
- **regime_zigzag_tuning.md** - 레짐 기반 ZigZag 튜닝
- **zigzag_tuning.md** - ZigZag 파라미터 튜닝
- **multi_timeframe_zigzag.md** - 멀티타임프레임 ZigZag

### 진단 문서
- **CHART_FLICKERING_DIAGNOSIS.md** - 차트 플리커링 진단
- **RENDERING_ISSUES_DIAGNOSIS.md** - 렌더링 이슈 진단

---

## 아카이브 문서 (docs/archives/)

### 시스템 설계
- **Architecture.md** - 시스템 아키텍처 (레거시)
- **EVENT_ARCHITECTURE_GUIDE.md** - 이벤트 아키텍처 가이드
- **SYSTEM_ALGORITHM_OVERVIEW.md** - 시스템 알고리즘 개요

### 설정
- **hardcoded_values.md** - 하드코딩된 값 목록

### 머신러닝
- **MODELS_GUIDE.md** - 모델 가이드
- **MODEL_TRAINING_GUIDE.md** - 모델 학습 가이드
- **dataset_training_guide.md** - 데이터셋 학습 가이드
- **CONFORMAL_PREDICTION_GUIDE.md** - Conformal Prediction 가이드
- **transformer_quality_measurement.md** - Transformer 품질 측정

### 피처
- **MULTISCALE_FEATURES_GUIDE.md** - 멀티스케일 피처 가이드
- **MULTITIMEFRAME_FEATURES.md** - 멀티타임프레임 피처

### LLM
- **LLM_JUDGE_SYSTEM_GUIDE.md** - LLM 판단 시스템 가이드
- **LLM_INPUT_TABLE.md** - LLM 입력 테이블
- **LLM_RateLimit_Changes.md** - LLM Rate Limit 변경사항

### 옵션
- **OPTION_FLOW_ANALYSIS_GUIDE.md** - 옵션 흐름 분석 가이드
- **OPTION_SENTIMENT_INTEGRATION_GUIDE.md** - 옵션 센티먼트 통합 가이드
- **call_put_parity_divergence_design.md** - 콜-풋 패리티 다이버전스 설계
- **premium_bleed_design.md** - 프리미엄 블리드 설계

### 피봇 (중복/레거시)
- **PIVOT_COLLECTOR_GUIDE.md** - 피봇 수집기 가이드
- **hybrid_pivot_evaluation.md** - 하이브리드 피봇 평가
- **pivot_confirmation_logic_merged.md** - 피봇 확정 로직 (병합)
- **pivot_count_adjustment_guide.md** - 피봇 수 조정 가이드

### ZigZag (중복/레거시)
- **zigzag_pivot_confirmation_lag.md** - ZigZag 피봇 확정 래그
- **zigzag_param_unification_report.md** - ZigZag 파라미터 통합 리포트

### 레짐
- **market_regime_classifier.md** - 시장 레짐 분류기
- **regime_based_trading.md** - 레짐 기반 트레이딩

### 트레이딩
- **TRADING_SIGNAL_GENERATION_GUIDE.md** - 트레이딩 시그널 생성 가이드
- **TRADE_LOGGING_GUIDE.md** - 트레이드 로깅 가이드
- **POSITION_SIZING_GUIDE.md** - 포지션 사이징 가이드
- **TradeExecutionGate_설계문서.md** - TradeExecutionGate 설계문서
- **heuristic_signal_algorithm.md** - 휴리스틱 시그널 알고리즘

### 피드백 & 가드레일
- **FEEDBACK_SYSTEM_GUIDE.md** - 피드백 시스템 가이드
- **GUARDRAIL_SYSTEM_GUIDE.md** - 가드레일 시스템 가이드

### 파라미터 튜닝
- **PARAMETER_TUNING_GUIDE.md** - 파라미터 튜닝 가이드

### 성능
- **PERFORMANCE_ANALYSIS_GUIDE.md** - 성능 분석 가이드
- **Rendering_Performance_Optimization.md** - 렌더링 성능 최적화

### 런타임
- **RUNTIME_API_REFERENCE.md** - 런타임 API 레퍼런스
- **RUNTIME_VS_BACKTEST_GUIDE.md** - 실시간 vs 백테스트 파이프라인

### eBest API
- **eBest_OpenAPI_Schema.md** - eBest OpenAPI 스키마
- **t8415_minute_data_logic.md** - t8415 분봉 데이터 로직

### 운영
- **GITHUB_ACTIONS_GUIDE.md** - GitHub Actions 가이드
- **INCIDENT_HANDLING_GUIDE.md** - 인시던트 핸들링 가이드

### 리팩토링
- **REFACTORING_SRC_LAYOUT.md** - 소스 레이아웃 리팩토링

### 기타
- **DOCUMENTATION_STATUS.md** - 문서화 현황
- **TODO.md** - TODO 목록
- **prediction.md** - 예측 관련
- **daily_train_patch_tst.md** - 일일 학습 패치 TST

---

## 디자인 문서 (docs/design/)

- **HybridAdaptivePivot_Design_v1.1.md** - 하이브리드 적응형 피봇 설계 v1.1
- **HybridAdaptivePivot_Design_v1.1.html** - 하이브리드 적응형 피봇 설계 v1.1 (HTML)

---

## 운영 문서 (docs/operations/)

- **DAILY_TICK_TRAINING_RUNBOOK.md** - 일일 틱 학습 런북
- **telegram.md** - 텔레그램 연동 가이드

---

## 리포트 (docs/reports/)

- **PIVOT_SIGNAL_IMPROVEMENTS.md** - 피봇 시그널 개선사항
- **ZigZag_Pivot_Principles_BugFix.md** - ZigZag 피봇 원칙 버그픽스

---

## 리뷰 (docs/reviews/)

- **ZigZag_Pivot_Analysis.md** - ZigZag 피봇 분석

---

## 런타임 문서 (docs/runtime/)

(14개 항목 - 런타임 API 레퍼런스)

---

## 트레이닝 문서 (docs/tring/)

(5개 항목 - 학습 레퍼런스)

---

## 정리 완료

### 삭제 완료 (오래된/중복)
1. ✅ **pivot_confirmation_logic.md** - pivot_confirmation_logic_merged.md로 병합됨
2. ✅ **pivot_confirm_debug.md** - 디버깅용 임시 파일
3. ✅ **percent_adaptive_pivot_guide.md** - 레거시 가이드
4. ✅ **daily_train_patch_tst.md** - 임시 패치 파일
5. ✅ **TODO.md** - 내용이 거의 없음

### 메인으로 승격 완료
1. ✅ **PIVOT_ML_ALGORITHM_GUIDE.md** - archives/에서 docs/로 이동
2. ✅ **CONFIG_REFERENCE_GUIDE.md** - archives/에서 docs/로 이동
3. ✅ **DUAL_MODE_GUIDE.md** - archives/에서 docs/로 이동

### 병합 완료
1. ✅ **ZigZag 관련 파일** - zigzag_pivot_improvement.md, zigzag_pivot_logic.md, zigzag_pivot_configuration_guide.md → ZIGZAG_PIVOT_COMPREHENSIVE_GUIDE.md
2. ✅ **피봇 검출 비교** - pivot_detection_comparison.md, pivot_detector_comparison.md → PIVOT_DETECTION_COMPARISON_GUIDE.md
3. ✅ **피봇 정보 패널** - pivot_info_panel_chart_internal_placement.md, pivot_info_panel_implementation.md → PIVOT_INFO_PANEL_GUIDE.md

---

**문서 버전**: 1.0  
**작성일**: 2026-06-16  
**마지막 수정**: 2026-06-16
