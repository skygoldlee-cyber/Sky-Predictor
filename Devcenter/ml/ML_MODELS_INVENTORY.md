# Devcenter/ml 예측 모델 정리

> 작성일: 2026-07-19
> 범위: Devcenter/ml 폴더 내 ML 관련 스크립트, 모델, 보고서, 데이터 일괄 정리

---

## 1. 개요

Devcenter/ml은 기존 피봇 기반 인트라데이 전략에 머신러닝을 적용해
**승률/수익성을 최적화**한 실험 산출물을 담고 있습니다.

- **대상 데이터**: KOSPI200 5분봉, 2019–2026년
- **최종 목표**: 거래 필터링 → 진입 타이밍 → 청산 타이밍 → 포지션 사이징 → 리스크 관리의 ML 파이프라인 구축
- **핵심 성과**: 승률 51.08% → 95.47%, 총 PnL 996만 원 → 2,008만 원 (Half Kelly 기준)

---

## 2. 파이프라인 구조

```
ml_data/ml_dataset.csv
    ↓
[거래 필터링]  XGBoost  →  trade_filter_xgboost*.json
    ↓
[진입 타이밍]  Random Forest  →  entry_timing_rf.pkl
    ↓
[청산 타이밍]  LSTM  →  exit_timing_lstm.keras
    ↓
[포지션 사이징]  Half Kelly / Fixed
    ↓
[리스크 관리]  ATR 기반 손절/동적 계약수
    ↓
final_trades*.csv
```

---

## 3. 주요 스크립트 분류

### 3.1 데이터 준비

| 파일 | 역할 |
|------|------|
| `ml_data_preparation.py` | 원본 백테스트 데이터에서 16개 피처(RSI, MACD, ATR, SuperTrend, BB, 시간, 레짐 등)로 학습용 CSV 생성 |
| `create_long_short_dataset.py` | 롱/숏 구분 데이터셋 추가 생성 |

### 3.2 거래 필터링

| 파일 | 역할 |
|------|------|
| `ml_trade_filter.py` | XGBoost 기반 거래 필터링 모델 학습/평가 |
| `ml_trade_filter_all_years.py` | 연도별 특화 필터 모델 학습 및 threshold 최적화 |
| `ml_trade_filter_2019.py` ~ `ml_trade_filter_2026.py` | 개별 연도 필터 실험 |
| `ml_dynamic_model_selection.py` | 연도별/롤링 윈도우 모델을 자동 선택하는 동적 모델 선택기 |

### 3.3 진입/청산 타이밍

| 파일 | 역할 |
|------|------|
| `ml_entry_timing.py` | Random Forest 진입 타이밍 최적화 |
| `ml_exit_timing.py` | LSTM 청산 타이밍 최적화 |
| `ml_exit_atr_optimization.py` | ATR 기반 청산 최적화 |
| `ml_exit_ratio_optimization.py` | 손익 비율 기반 청산 최적화 |

### 3.4 포지션 사이징 & 리스크

| 파일 | 역할 |
|------|------|
| `ml_position_sizing.py` | Kelly Criterion / Fixed 포지션 사이징 |
| `ml_position_sizing_improved.py` | 개선된 포지션 사이징 |
| `ml_risk_management.py` | 동적 계약수, 드로우다운 제한 |
| `analyze_optimal_contract_size.py` | 최적 계약수 분석 |

### 3.5 검증 & 워크포워드

| 파일 | 역할 |
|------|------|
| `ml_walk_forward_validation.py` | 워크포워드 검증 (시간 기반 분할) |
| `ml_cross_validation.py` | 교차 검증 |
| `ml_rolling_window_model.py` | 롤링 윈도우 재학습 모델 |
| `train_optimized_rolling_window.py` | 최적 롤링 윈도우 학습 |
| `train_realistic_rolling_window.py` | 현실적 롤링 윈도우 학습 |
| `ml_quarterly_retraining_pipeline.py` | 분기별 재학습 파이프라인 |
| `test_realistic_model_selection.py` | 현실적 모델 선택 테스트 |
| `test_walk_forward_validation.py` | 워크포워드 검증 테스트 |

### 3.6 시장 구조 & 레짐

| 파일 | 역할 |
|------|------|
| `ml_regime_detection.py` | 시장 레짐 분류 |
| `ml_regime_models.py` | 레짐별 모델 학습 |
| `analyze_market_structure.py` | 시장 구조 분석 |
| `ml_volatility_adaptive.py` | 변동성 적응형 모델 |

### 3.7 실시간/페이퍼 트레이딩

| 파일 | 역할 |
|------|------|
| `ml_live_trading.py` | 실시간 트레이딩 연동 스크립트 |
| `ml_live_trading_simulation.py` | 실시간 트레이딩 시뮬레이션 |
| `ml_live_trading_test.py` | 실시간 트레이딩 테스트 |
| `ml_paper_trading.py` | 페이퍼 트레이딩 |

### 3.8 분석 스크립트

| 파일 | 역할 |
|------|------|
| `analyze_2023.py`, `analyze_2024.py`, `analyze_25_26.py` | 연도별 성과 분석 |
| `analyze_2024_model.py` | 2024년 모델 상세 분석 |
| `analyze_contract_size.py` | 계약수별 성과 분석 |
| `analyze_yearly_performance.py` | 연도별 성과 추적 |
| `analyze_performance_improvement.py` | 개선 효과 분석 |
| `analyze_return_on_capital.py` | 자본 대비 수익률 분석 |
| `pivot_frequency_analysis.py` | 피봇 빈도 분석 |

### 3.9 하이퍼파라미터 & 앙상블

| 파일 | 역할 |
|------|------|
| `ml_hyperparameter_optimization.py` | 하이퍼파라미터 최적화 |
| `ml_hyperparameter_tuning.py` | 추가 하이퍼파라미터 튜닝 |
| `ml_ensemble.py` | 앙상블 모델 |
| `ml_transformer_model.py` | Transformer 실험 모델 |
| `ml_regression_target.py` | 회귀 타겟 실험 |

### 3.10 기타

| 파일 | 역할 |
|------|------|
| `ml_sample_weighting.py` | 샘플 가중치 실험 |
| `ml_slippage_modeling.py` | 슬리피지 모델링 |
| `check_*.py` | 데이터/계약/방향성 등 간단 검증 유틸 |
| `test_*.py` | 각종 테스트 스크립트 |

---

## 4. 핵심 보고서

| 파일 | 핵심 내용 |
|------|-----------|
| `ml_optimization_report.md` | XGBoost→RF→LSTM→Kelly 5단계 최적화 과정 및 최종 성과 |
| `ml_comparison_table.md` | 기존 방식 vs ML 최적화 성과 비교 (연도별, 단계별) |
| `ml_dynamic_model_selection_guide.md` | 연도별/롤링 윈도우 동적 모델 선택 가이드 및 실전 적용 방법 |
| `ml_architecture_design.md` | 전체 ML 아키텍처 설계 문서 (대용량) |
| `entry_timing_optimization_report.md` | 진입 타이밍 최적화 결과 |
| `exit_timing_optimization_report.md` | 청산 타이밍 최적화 결과 |
| `position_sizing_optimization_report.md` | 포지션 사이징 최적화 결과 |
| `risk_management_optimization_report.md` | 리스크 관리 최적화 결과 |
| `win_loss_ratio_improvement_proposal.md` | 손익 비율 개선 제안 |
| `win_loss_ratio_optimization_final_report.md` | 손익 비율 최적화 최종 보고서 |

---

## 5. 모델 산출물 (ml_models/)

### 5.1 거래 필터링

- `trade_filter_xgboost.json` — 기본 XGBoost 필터
- `trade_filter_xgboost_2019.json` ~ `trade_filter_xgboost_2026.json` — 연도별 특화 모델
- `trade_filter_xgboost_rolling.json` — 롤링 윈도우 모델
- `trade_filter_xgboost_rolling_optimized.json` — 최적 롤링 윈도우
- `trade_filter_xgboost_rolling_realistic.json` — 현실적 롤링 윈도우 (실전 권장)
- `trade_filter_v_20260622.pkl` — 2026-06-22 버전 필터
- `xgboost_optimized.pkl`, `xgboost_optimized_final.pkl` — 최적화된 XGBoost
- `xgboost_weighted_*.pkl` — 난이도/성과/시간/변동성 가중치 모델

### 5.2 진입/청산 타이밍

- `entry_timing_rf.pkl` — Random Forest 진입 타이밍 모델
- `entry_timing_v_20260622.pkl` — 2026-06-22 버전 진입 모델
- `exit_timing_lstm.keras` — LSTM 청산 타이밍 모델
- `exit_timing_v_20260622.keras` — 2026-06-22 버전 청산 모델
- `random_forest_optimized.pkl`, `random_forest_optimized_final.pkl` — 최적화 RF

### 5.3 레짐/변동성

- `regime_classifier.pkl` — 시장 레짐 분류기
- `volatility_model_cluster_*.pkl` — 변동성 클러스터별 모델
- `volatility_scaler.pkl` — 변동성 스케일러

### 5.4 결과 데이터

- `filtered_trades.csv` — 필터링된 거래
- `optimized_trades.csv` — 최적화된 거래
- `final_trades.csv` — 최종 거래
- `final_trades_sized_improved.csv` — 개선된 사이징 적용 최종 거래
- `final_trades_risk_managed.csv` — 리스크 관리 적용 최종 거래
- `exit_atr_optimized_trades.csv` — ATR 청산 최적화 결과
- `exit_ratio_optimized_trades.csv` — 손익비 청산 최적화 결과
- `frequency_analysis_result.json` — 피봇 빈도 분석 결과
- `walk_forward_validation_result.json` — 워크포워드 검증 결과
- `hyperparameter_optimization_result.json` — 하이퍼파라미터 최적화 결과
- `live_trading_simulation_result.json` — 실시간 시뮬레이션 결과

---

## 6. 데이터 산출물 (ml_data/)

| 파일 | 설명 |
|------|------|
| `ml_dataset.csv` | 기본 학습/평가용 데이터 (1,521건) |
| `ml_dataset_long_short.csv` | 롱/숏 구분 데이터 |
| `ml_dataset_with_regime.csv` | 레짐 정보 추가 데이터 |
| `ml_dataset_with_slippage.csv` | 슬리피지 반영 데이터 |

---

## 7. 최근(2026-06) 주요 업데이트

- `*_v_20260622.pkl/keras` 버전 모델 추가
- 현실적 롤링 윈도우 모델(`trade_filter_xgboost_rolling_realistic.json`) 도입
- 동적 모델 선택 시스템(`ml_dynamic_model_selection.py`) 완성 및 가이드 작성
- 계약수 3계약 적용 시뮬레이션, 총 PnL 3,214만 원 달성 보고

---

## 8. 참고

- 본 폴더의 ML 모델은 `prediction/` 폴더의 딥러닝 방향 예측 모델(PriceTransformer/PatchTST/Mamba/TFT)과 **별개 계층**입니다.
- `prediction/` 모델은 실시간 방향성 예측용, `Devcenter/ml` 모델은 백테스트 기반 거래 필터/타이밍/사이징 최적화용입니다.
