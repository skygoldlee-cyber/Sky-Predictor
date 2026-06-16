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

### 시스템 가이드
- **LLM_JUDGE_SYSTEM_GUIDE.md** - LLM 판단 시스템 가이드
- **OPTION_FLOW_ANALYSIS_GUIDE.md** - 옵션 흐름 분석 가이드
- **FEEDBACK_SYSTEM_GUIDE.md** - 피드백 시스템 가이드
- **GUARDRAIL_SYSTEM_GUIDE.md** - 가드레일 시스템 가이드
- **TRADING_SIGNAL_GENERATION_GUIDE.md** - 트레이딩 시그널 생성 가이드
- **CONFORMAL_PREDICTION_GUIDE.md** - Conformal Prediction 가이드
- **MULTISCALE_FEATURES_GUIDE.md** - 멀티스케일 피처 가이드

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

카테고리별로 정리된 아카이브 문서입니다.

### architecture/ (시스템 설계)
- **Architecture.md** - 시스템 아키텍처 (레거시)
- **EVENT_ARCHITECTURE_GUIDE.md** - 이벤트 아키텍처 가이드
- **SYSTEM_ALGORITHM_OVERVIEW.md** - 시스템 알고리즘 개요

### config/ (설정 및 운영)
- **hardcoded_values.md** - 하드코딩된 값 목록
- **PARAMETER_TUNING_GUIDE.md** - 파라미터 튜닝 가이드
- **eBest_OpenAPI_Schema.md** - eBest OpenAPI 스키마
- **RUNTIME_API_REFERENCE.md** - 런타임 API 레퍼런스
- **RUNTIME_VS_BACKTEST_GUIDE.md** - 실시간 vs 백테스트 파이프라인
- **t8415_minute_data_logic.md** - t8415 분봉 데이터 로직
- **GITHUB_ACTIONS_GUIDE.md** - GitHub Actions 가이드
- **INCIDENT_HANDLING_GUIDE.md** - 인시던트 핸들링 가이드
- **REFACTORING_SRC_LAYOUT.md** - 소스 레이아웃 리팩토링

### ml/ (머신러닝)
- **MODELS_GUIDE.md** - 모델 가이드
- **MODEL_TRAINING_GUIDE.md** - 모델 학습 가이드
- **dataset_training_guide.md** - 데이터셋 학습 가이드
- **transformer_quality_measurement.md** - Transformer 품질 측정
- **MULTITIMEFRAME_FEATURES.md** - 멀티타임프레임 피처
- **prediction.md** - 예측 관련

### llm/ (LLM)
- **LLM_INPUT_TABLE.md** - LLM 입력 테이블
- **LLM_RateLimit_Changes.md** - LLM Rate Limit 변경사항

### options/ (옵션)
- **OPTION_SENTIMENT_INTEGRATION_GUIDE.md** - 옵션 센티먼트 통합 가이드
- **call_put_parity_divergence_design.md** - 콜-풋 패리티 다이버전스 설계
- **premium_bleed_design.md** - 프리미엄 블리드 설계

### pivot/ (피봇)
- **PIVOT_COLLECTOR_GUIDE.md** - 피봇 수집기 가이드
- **hybrid_pivot_evaluation.md** - 하이브리드 피봇 평가
- **pivot_confirmation_logic_merged.md** - 피봇 확정 로직 (병합)
- **pivot_count_adjustment_guide.md** - 피봇 수 조정 가이드

### zigzag/ (ZigZag)
- **zigzag_pivot_confirmation_lag.md** - ZigZag 피봇 확정 래그
- **zigzag_param_unification_report.md** - ZigZag 파라미터 통합 리포트

### regime/ (레짐)
- **market_regime_classifier.md** - 시장 레짐 분류기
- **regime_based_trading.md** - 레짐 기반 트레이딩

### trading/ (트레이딩)
- **TRADE_LOGGING_GUIDE.md** - 트레이드 로깅 가이드
- **POSITION_SIZING_GUIDE.md** - 포지션 사이징 가이드
- **TradeExecutionGate_설계문서.md** - TradeExecutionGate 설계문서
- **heuristic_signal_algorithm.md** - 휴리스틱 시그널 알고리즘

### performance/ (성능)
- **PERFORMANCE_ANALYSIS_GUIDE.md** - 성능 분석 가이드
- **Rendering_Performance_Optimization.md** - 렌더링 성능 최적화

### operations/ (운영)
- **operations_telegram.md** - 텔레그램 연동 (중복)

### 기타
- **DOCUMENTATION_STATUS.md** - 문서화 현황

---

## 디자인 문서 (docs/archives/design/)

- **HybridAdaptivePivot_Design_v1.1.md** - 하이브리드 적응형 피봇 설계 v1.1
- **HybridAdaptivePivot_Design_v1.1.html** - 하이브리드 적응형 피봇 설계 v1.1 (HTML)

---

## 운영 문서 (docs/operations/)

- **DAILY_TICK_TRAINING_RUNBOOK.md** - 일일 틱 학습 런북

---

## 리포트 (docs/reports/)

- **PIVOT_SIGNAL_IMPROVEMENTS.md** - 피봇 시그널 개선사항
- **ZigZag_Pivot_Principles_BugFix.md** - ZigZag 피봇 원칙 버그픽스
- **Transformer_보완_개선_리포트.md** - Transformer 보완 개선 리포트
- **optimize_zigzag_lag_improvements.md** - ZigZag 래그 최적화
- **transformer_improvement_report.md** - Transformer 개선 리포트

---

## 리뷰 (docs/archives/reviews/)

- **ZigZag_Pivot_Analysis.md** - ZigZag 피봇 분석
- **Transformer_코드리뷰.md** - Transformer 코드 리뷰
- **adaptive_indicator_analysis.md** - 적응형 지표 분석
- **adaptive_indicator_deep_review.md** - 적응형 지표 심층 리뷰
- **code_review_report.md** - 코드 리뷰 리포트
- **prediction_algorithm_review.md** - 예측 알고리즘 리뷰
- **prediction_code_review.md** - 예측 코드 리뷰
- **source_review_v2.md** - 소스 리뷰 v2
- **telegram_review.md** - 텔레그램 리뷰

---

## 아카이브 리포트 (docs/archives/reports/)

- **BUG_FIXES.md** - 버그 수정 리포트
- **SkyPredictor_개선사항.md** - SkyPredictor 개선사항
- **adaptive_zigzag_fixes.md** - 적응형 ZigZag 수정
- **improvement_report.md** - 개선 리포트

---

## 런타임 문서 (docs/runtime/)

런타임 API 레퍼런스 및 트러블슈팅 가이드입니다.

### 핵심 모듈
- **adaptive_indicator.md** - 적응형 지표 런타임 가이드
- **config.md** - 설정 로드 및 검증
- **prediction.md** - 예측 파이프라인 런타임
- **main.md** - 메인 진입점
- **ebest.md** - eBest API 런타임

### 트러블슈팅
- **live_run_troubleshooting.md** - 실시간 실행 문제 해결
- **runtime_README.md** - 런타임 개요

### 기타
- **Market_Open_Subscription_Flow.md** - 장 시작 구독 흐름
- **adaptive_indicator_improvements.md** - 적응형 지표 개선사항
- **adaptive_indicator_parameters.md** - 적응형 지표 파라미터
- **runtime_telegram.md** - 런타임 텔레그램
- **telegram.md** - 텔레그램 연동
- **ticks.md** - 틱 데이터 처리
- **volume_imbalance.md** - 거래량 불균형

---

## 트레이닝 문서 (docs/training/)

오프라인 데이터셋 생성 및 학습 가이드입니다.

- **README.md** - 트레이닝 문서 인덱스
- **data_builder.md** - 데이터셋 생성 (ticks JSONL → dataset NPZ)
- **train_transformer.md** - Transformer 학습
- **train_tft.md** - TFT 학습
- **datasets.md** - 데이터셋 병합 및 검증

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

### 아카이브 이동 완료
1. ✅ **design/** → archives/design/ 이동 (HybridAdaptivePivot_Design_v1.1.md/html)
2. ✅ **reviews/** → archives/reviews/ 이동 (9개 코드 리뷰 파일)
3. ✅ **reports/** 오래된 파일 → archives/reports/ 이동 (BUG_FIXES.md, SkyPredictor_개선사항.md, adaptive_zigzag_fixes.md, improvement_report.md)
4. ✅ **7개 시스템 가이드** → docs/로 승격 (LLM_JUDGE_SYSTEM_GUIDE.md, OPTION_FLOW_ANALYSIS_GUIDE.md, FEEDBACK_SYSTEM_GUIDE.md, GUARDRAIL_SYSTEM_GUIDE.md, TRADING_SIGNAL_GENERATION_GUIDE.md, CONFORMAL_PREDICTION_GUIDE.md, MULTISCALE_FEATURES_GUIDE.md)
5. ✅ **operations/telegram.md** → archives/operations_telegram.md 이동 (runtime/telegram.md와 중복)

### 인덱스 업데이트 완료
1. ✅ **runtime/** 디렉토리 상세 목록 추가
2. ✅ **training/** 디렉토리 상세 목록 추가
3. ✅ **시스템 가이드** 섹션 추가 및 승격된 가이드 반영
4. ✅ **archives/** 카테고리별 재분류 (architecture, config, ml, llm, options, pivot, zigzag, regime, trading, performance, operations)
5. ✅ **reviews/** 섹션 상세 목록 추가
6. ✅ **reports/** 섹션 상세 목록 추가

---

**문서 버전**: 1.0  
**작성일**: 2026-06-16  
**마지막 수정**: 2026-06-16
