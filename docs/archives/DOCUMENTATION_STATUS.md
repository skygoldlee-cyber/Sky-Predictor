# 문서화 현황 및 추가 정리 필요 항목

## 현재 문서화 현황

### ✅ 완료된 문서 (최신)

| 문서 | 상태 | 설명 |
|------|------|------|
| ML_ENGINE_OVERVIEW.md | ✅ 최신 | 머신러닝 엔진 개요 (2026-04-25 작성) |
| PIVOT_ML_ALGORITHM_GUIDE.md | ✅ 최신 | 피봇 예측 ML 알고리즘 상세 가이드 |
| PIVOT_COLLECTOR_GUIDE.md | ✅ 최신 | 피봇 수집기 사용 가이드 |
| DUAL_MODE_GUIDE.md | ✅ 최신 | 듀얼 모드 구조 가이드 (2026-04-25 작성) |
| LLM_JUDGE_SYSTEM_GUIDE.md | ✅ 최신 | LLM 판단 시스템 가이드 (2026-04-25 작성) |
| TRADING_SIGNAL_GENERATION_GUIDE.md | ✅ 최신 | 트레이딩 시그널 생성 가이드 (2026-04-25 작성) |
| CONFIG_REFERENCE_GUIDE.md | ✅ 최신 | config.json 전체 설정 가이드 (2026-04-25 작성) |
| OPTION_FLOW_ANALYSIS_GUIDE.md | ✅ 최신 | 옵션 흐름 분석 시스템 (2026-04-25 작성) |
| FEEDBACK_SYSTEM_GUIDE.md | ✅ 최신 | 피드백 시스템 가이드 (2026-04-25 작성) |
| GUARDRAIL_SYSTEM_GUIDE.md | ✅ 최신 | 가드레일 시스템 가이드 (2026-04-25 작성) |
| RUNTIME_VS_BACKTEST_GUIDE.md | ✅ 최신 | 실시간 vs 백테스트 파이프라인 (2026-04-25 작성) |
| CONFORMAL_PREDICTION_GUIDE.md | ✅ 최신 | Conformal Prediction 가이드 (2026-04-25 작성) |
| MULTISCALE_FEATURES_GUIDE.md | ✅ 최신 | 멀티스케일 피처 가이드 (2026-04-25 작성) |
| TRANSFORMER_GUIDE.md | ✅ 완료 | Transformer 가이드 |
| TFT_DUAL_MODEL_DESIGN_GUIDE.md | ✅ 완료 | TFT 설계 가이드 |
| ADAPTIVE_INDICATOR_GUIDE.md | ✅ 완료 | 적응형 지표 가이드 |
| DAILY_TICK_TRAINING_RUNBOOK.md | ✅ 완료 | 일일 틱 학습 런북 |
| Architecture.md | ✅ 완료 | 시스템 아키텍처 |
| Prediction_Algorithm.md | ✅ 완료 | 예측 알고리즘 |
| telegram.md | ✅ 완료 | 텔레그램 알림 시스템 |
| runtime/README.md | ✅ 완료 | 런타임 레퍼런스 |
| training/README.md | ✅ 완료 | 학습 레퍼런스 |

## 📋 추가 정리 필요 항목

모든 항목이 완료되었습니다.

### 1. 듀얼 모드 (KOSPI/KP200) 구조 가이드

**우선순위**: 높음  
**이유**: 최근 구현된 기능으로 문서화 부족

**포함 내용**:
- 듀얼 모드 개요 및 목적
- config.json 설정 (`dual_mode`, `kospi_symbol`, `futures_symbol`)
- AdaptiveIndicatorManager 듀얼 모드 구조
- 각 ZigZag 인스턴스의 역할
- GUI 플롯 선택 기능
- 피봇 로그 구분 (`[KOSPI]`, `[KP200]` 접두사)
- 사용 예시

**파일명**: `DUAL_MODE_GUIDE.md`

---

### 2. LLM 판단 시스템 가이드

**우선순위**: 높음  
**이유**: 복잡한 LLM 통합 로직

**포함 내용**:
- LLM 판단 시스템 개요
- 지원되는 LLM 제공자 (Claude, GPT, Gemini)
- dual_llm 모드 구조
- LLM 입력 컨텍스트 구조 (LLM_INPUT_TABLE.md 참조)
- 판단 로직 (BUY/SELL/HOLD)
- provider fallback 메커니즘
- 캐싱 및 rate limiting
- 사용 예시

**파일명**: `LLM_JUDGE_SYSTEM_GUIDE.md`

---

### 3. 옵션 흐름 (Option Flow) 분석 시스템

**우선순위**: 중간  
**이유**: 옵션 데이터 분석 핵심 기능

**포함 내용**:
- 옵션 흐름 개요
- ITM/OTM 구독 로직
- 옵션 피처 계산 (OI, PCR, Volume 등)
- 프리미엄 블리드 지표
- 콜-풋 패리티 이탈 지표
- 알림 조건 및 메시지 포맷
- 사용 예시

**파일명**: `OPTION_FLOW_ANALYSIS_GUIDE.md`

---

### 4. 피드백 시스템 가이드

**우선순위**: 중간  
**이유**: 동적 가중치 조절 핵심

**포함 내용**:
- 피드백 시스템 개요
- 피드백 수집 방식
- 가중치 조절 로직 (transformer_weight)
- 레짐별 가중치 조절
- confidence_high_margin 역할
- 사용 예시

**파일명**: `FEEDBACK_SYSTEM_GUIDE.md`

---

### 5. 가드레일 시스템 가이드

**우선순위**: 중간  
**이유**: 리스크 관리 핵심

**포함 내용**:
- 가드레일 시스템 개요
- guard_basis_hold_thr
- guard_atm_spread_pct_thr
- guard_atm_liq_log_thr
- IV 동적 가드레일
- 가드레일 트리거 조건
- 사용 예시

**파일명**: `GUARDRAIL_SYSTEM_GUIDE.md`

---

### 6. 트레이딩 시그널 생성 가이드

**우선순위**: 높음  
**이유**: 최종 트레이딩 의사결정 로직

**포함 내용**:
- 시그널 생성 개요
- buy_threshold, sell_threshold
- confidence 레벨 (HIGH/MEDIUM/LOW)
- HOLD 조건 (disagreement_hold)
- 앙상블 일치 시 confidence boost
- LLM action과 수치 예측 통합
- trade_gate와의 연동
- 사용 예시

**파일명**: `TRADING_SIGNAL_GENERATION_GUIDE.md`

---

### 7. Conformal Prediction 가이드

**우선순위**: 낮음  
**이유**: 선택적 기능

**포함 내용**:
- Conformal Prediction 개요
- conformal_alpha 설정
- 예측 구간 계산
- 신뢰도 해석
- 사용 예시

**파일명**: `CONFORMAL_PREDICTION_GUIDE.md`

---

### 8. config.json 전체 설정 가이드

**우선순위**: 높음  
**이유**: 모든 설정을 한 곳에 정리

**포함 내용**:
- config.json 전체 구조
- 각 섹션별 설정 설명
  - ai_providers
  - ebest
  - options_subscription
  - telegram
  - prediction
  - adaptive_indicator
  - trade_gate
- 설정값별 권장 범위
- 사용 예시

**파일명**: `CONFIG_REFERENCE_GUIDE.md`

---

### 9. 실시간 vs 백테스트 파이프라인 차이

**우선순위**: 중간  
**이유**: 두 파이프라인의 차이점 명확화

**포함 내용**:
- 실시간 파이프라인 구조
- 백테스트 파이프라인 구조
- 주요 차이점
  - 데이터 소스
  - 타이밍
  - LLM 사용 여부
  - 피드백 적용 여부
- 사용 시나리오

**파일명**: `RUNTIME_VS_BACKTEST_GUIDE.md`

---

### 10. 멀티스케일 피처 가이드

**우선순위**: 낮음  
**이유**: 선택적 기능

**포함 내용**:
- 멀티스케일 피처 개요
- 지원되는 타임스케일 (1분, 5분, 15분)
- multiscale_enabled 설정
- 피처 계산 로직
- 사용 예시

**파일명**: `MULTISCALE_FEATURES_GUIDE.md`

---

## 우선순위 정리

### 높음 (즉시 필요)
1. 듀얼 모드 구조 가이드
2. LLM 판단 시스템 가이드
3. 트레이딩 시그널 생성 가이드
4. config.json 전체 설정 가이드

### 중간 (단기 필요)
5. 옵션 흐름 분석 시스템
6. 피드백 시스템 가이드
7. 가드레일 시스템 가이드
8. 실시간 vs 백테스트 파이프라인 차이

### 낮음 (장계획)
9. Conformal Prediction 가이드
10. 멀티스케일 피처 가이드

---

## 문서화 템플릿

각 가이드 문서는 다음 구조를 따르는 것을 권장:

```markdown
# [가이드 이름]

## 개요
- 목적
- 대상 독자

## 핵심 개념
- 주요 용어 정의
- 아키텍처 다이어그램

## 설정
- 관련 config.json 설정
- 파라미터 설명

## 사용 방법
- 기본 사용법
- 고급 사용법
- 예제 코드

## 주의사항
- 일반적인 주의사항
- 에러 처리

## 관련 문서
- 참조할 다른 문서
```

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
