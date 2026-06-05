# SkyPredictor Architecture

## 시스템 개요

SkyPredictor는 KOSPI200 선물/옵션 틱 데이터를 기반으로 실시간 방향성 예측을 제공하는 시스템입니다. 오더북 분석, 기술적 지표, 머신러닝 모델, LLM 판단을 통합하여 BUY/SELL/HOLD 신호를 생성합니다.

### 핵심 기능

- **실시간 틱 처리**: eBest API로부터 선물/옵션 틱 데이터 수신 및 분봉 집계
- **오더북 분석**: FH0 호가 스냅샷 기반 오더북 불균형 피처 추출
- **적응형 지표**: Adaptive ZigZag, Adaptive SuperTrend로 시장 구조/추세 판단
- **ML 예측**: Transformer, TFT, Ensemble 모델로 수치적 확률 예측
- **LLM 판단**: Claude/GPT/Gemini로 전략적 해석 및 판단
- **가드레일**: 베이시스, 옵션 유동성, 패리티 다이버전스 등으로 신호 필터링

---

## 시스템 아키텍처

### 전체 데이터 흐름

```
eBest API (FC0/FH0/OC0/OH0/IJ_)
    ↓
TickProcessor (틱 처리, 분봉 집계)
    ↓
PredictionPipeline (예측 파이프라인)
    ├─ TickMixin (FO0 버퍼, 분봉 DF)
    ├─ OptionMixin (옵션 스냅샷, OI, IV)
    ├─ AdaptiveMixin (레짐·지표)
    └─ PredictionMixin.get_prediction()
        ├─ 수치: build_sequence + ModelInput → numeric_predictor.predict()
        ├─ 가드레일: 옵션/베이시스/패리티/블리드/OI·진폭
        ├─ LLM: 스냅샷·프롬프트 → 판단
        └─ 피드백 큐 (앙상블 가중 갱신용)
    ↓
출력: signal, confidence, llm_action, rationale 등
```

### 모듈 구조

```
project/
├── core/                   # 핵심 유틸리티
│   ├── cli_args.py         # CLI 인자 파싱
│   ├── logging_utils.py    # 로깅 설정
│   └── utils.py            # 공통 유틸
├── config/                 # 설정 관리
│   ├── config.py           # config.json 로드
│   └── constants.py        # 상수/Enum
├── data/                   # 데이터 처리
│   └── tick_processor.py   # 틱 처리, 분봉 집계
├── indicators/            # 기술적 지표
│   ├── adaptive_zigzag.py  # Adaptive ZigZag
│   ├── adaptive_supertrend.py # Adaptive SuperTrend
│   └── indicator_integration.py # 지표 통합
├── prediction/             # 예측 시스템
│   ├── pipeline.py         # PredictionPipeline (오케스트레이션)
│   ├── predictor.py        # NumericPredictor (Transformer/TFT/Ensemble)
│   ├── model.py            # PriceTransformer
│   ├── tft_model.py        # TemporalFusionTransformer
│   ├── features.py         # 피처 엔지니어링
│   ├── option_features.py  # 옵션 피처
│   ├── context_builder.py  # LLM 컨텍스트
│   ├── llm_judge.py        # LLM 호출
│   └── data_builder.py     # 학습 데이터셋 생성
├── ebestapi/               # eBest 연동
│   ├── live.py             # LiveState, 라이브 루프
│   └── *.py                # API 헬퍼
├── telegram/               # 텔레그램 연동
│   ├── bridge.py           # PipelineTelegramBridge
│   ├── notifier.py         # 알림 전송
│   └── commands.py         # 명령 처리
├── training/               # 학습 스크립트
│   └── train.py            # Transformer 학습
└── app/                    # 애플리케이션
    ├── app_setup.py        # 앱 설정
    └── run_modes.py        # 실행 모드
```

---

## 핵심 컴포넌트

### 1. TickProcessor

**역할**: 실시간 틱 데이터 처리 및 분봉 집계

**주요 기능**:
- FC0 (선물 체결), FH0 (선물 호가), OC0 (옵션 체결), OH0 (옵션 호가) 처리
- 분봉 OHLCV 집계
- 옵션 심볼 필터링 (ATM 기준)
- 메모리 관리 (options_minute_data cleanup)

**데이터 흐름**:
```
eBest Tick → TickProcessor.update() → 분봉 DF → FO0 버퍼
```

### 2. PredictionPipeline

**역할**: 예측 파이프라인 오케스트레이션

**Mixin 구조**:
- `TickMixin`: FO0 버퍼, 분봉 DF 관리
- `OptionMixin`: 옵션 스냅샷, OI, IV 계산
- `AdaptiveMixin`: 레짐·지표 (Adaptive ZigZag/SuperTrend)
- `PredictionMixin`: 예측 로직

**주요 메서드**:
- `get_prediction()`: 메인 예측 메서드
- `_build_and_predict_numeric()`: 수치 예측
- `_judge_with_llm()`: LLM 판단
- `_apply_guards()`: 가드레일 적용

### 3. NumericPredictor

**역할**: 수치적 확률 예측

**모델 유형**:
- `transformer`: PriceTransformer 단독
- `tft`: TemporalFusionTransformer 단독
- `ensemble`: Transformer + TFT 앙상블
- `rule_based`: 휴리스틱 fallback

**입력**: `ModelInput(sequence, past_known, future_known, seq_len)`
**출력**: `prob` (상승 확률 0~1), `signal`, `confidence`

### 4. AdaptiveIndicators

**Adaptive ZigZag**:
- ATR 기반 동적 임계값
- ER(Efficiency Ratio) 적응형 필터링
- 피봇 확정 알고리즘 (confirmation_bars)
- 구조 분석 (상승/하락/횡보)

**Adaptive SuperTrend**:
- ATR 기반 추세 추종
- 변동성에 따른 상승/하락선 조정
- ADX 기반 추세 강도 판정

### 5. LLM Judge

**지원 Provider**: Claude, GPT, Gemini

**기능**:
- 스냅샷 기반 판단
- Provider fallback
- Dual LLM 모드 (GPT + Gemini 동시 호출)
- Timeout/Rate-limit 처리

---

## 데이터 흐름 상세

### 1. 틱 → 분봉

```
FC0 Tick (price, chetime, _ts_epoch)
    ↓
TickProcessor.update()
    ↓
분봉 DF (Open, High, Low, Close, Volume)
    ↓
FO0 버퍼 (최근 N분)
```

### 2. 오더북 피처

```
FH0 호가 스냅샷 (1Hz)
    ↓
features.build_sequence()
    ↓
OB_KEYS (7): obi, spread, level1_ratio, level2_ratio, level3_ratio, level4_ratio, level5_ratio
```

### 3. 옵션 피처

```
OC0 옵션 체결
    ↓
OptionMixin.update()
    ↓
OPT_KEYS (v1: 7, v2: 16, v3: 23, v4: 29)
    ├─ PCR (Put/Call Ratio)
    ├─ IV Skew
    ├─ Max Pain
    └─ Microstructure features
```

### 4. 적응형 지표 피처

```
분봉 OHLCV
    ↓
AdaptiveIndicatorManager.update()
    ↓
ADAPT_KEYS (28):
    ├─ ast_* (SuperTrend: direction, dist_pct, atr_pct, efficiency_ratio)
    ├─ azz_* (ZigZag: current_direction, nearest_support, nearest_resistance)
    └─ cross_* (크로스 피처: agree, at_sup, at_res, bkout)
```

### 5. 시퀀스 빌드

```
분봉 DF + FO0 버퍼 + 옵션 스냅샷 + 적응형 지표
    ↓
features.build_sequence()
    ↓
ModelInput:
    ├─ sequence: (seq_len, feature_dim)
    ├─ past_known: (seq_len, TIME_KEYS) - TFT용
    ├─ future_known: (horizon, TIME_KEYS) - TFT용
    └─ seq_len: 시퀀스 길이 (초)
```

---

## 설정 구조

### config.json

```json
{
  "prediction": {
    "numeric_predictor": "ensemble",
    "dual_llm": false,
    "buy_threshold": 0.62,
    "sell_threshold": 0.38,
    "transformer_weight": 0.5,
    "disagreement_hold": true
  },
  "adaptive_indicator": {
    "enabled": true,
    "supertrend": { ... },
    "zigzag": { ... }
  },
  "ai_providers": {
    "anthropic": { "api_key": "" },
    "openai": { "api_key": "" },
    "gemini": { "api_key": "" }
  }
}
```

### feature_dim 조합

| option_feature_set | adaptive_indicator.enabled | feature_dim |
|---------------------|---------------------------|-------------|
| v1 | false | 19 |
| v1 | true | 47 |
| v2 | false | 28 |
| v2 | true | 56 |
| v3 | false | 35 |
| v3 | true | 63 |
| v4 | false | 41 |
| v4 | true | 69 |

---

## 학습 파이프라인

### 데이터셋 생성

```
ticks_replay_*.jsonl.gz
    ↓
prediction.data_builder
    ↓
dataset_*.npz (X, y, metadata)
```

### 학습

```
dataset_*.npz
    ↓
train.py (Transformer) / train_tft.py (TFT)
    ↓
weights/*.pt (transformer_5m.pt, tft_5m.pt)
```

### Rolling Merge

```
dataset_YYYYMMDD.npz (최근 N일)
    ↓
merge_datasets.py
    ↓
merged_dataset.npz
    ↓
재학습
```

---

## 가드레일 시스템

### 1. 베이시스 가드레일
- 선물-현물 괴리 모니터링
- 과대/과소 평가 판단
- `guard_basis_hold_thr`, `guard_basis_downgrade_thr`

### 2. 옵션 유동성 가드레일
- OI 집중도 기반 필터링
- IV 기반 피봇 확인
- ATM 스프레드 모니터링

### 3. 패리티 다이버전스
- 콜-풋 스큐 모니터링
- 페널티 기반 조정

### 4. Disagreement Hold
- 앙상블 모델 간 불일치 시 HOLD 강제
- `disagreement_hold_prob_diff_max`

---

## 실행 모드

### 1. GUI 모드 (기본)
```bash
python main.py
```

### 2. CLI 모드
```bash
python main.py --cli
```

### 3. 테스트 모드
```bash
python main.py --test
```

### 4. Heuristic-only 모드
```bash
python main.py --heuristic-only --no-ebest-live
```

### 5. eBest Live 모드
```bash
python main.py  # 기본값
```

---

## 관련 문서

### 메인 문서
- [README.md](../README.md) - 프로젝트 개요 및 빠른 시작
- [docs_README.md](docs_README.md) - 프로젝트 개요 및 상세 사용 가이드

### 기술 가이드
- [ADAPTIVE_INDICATOR_GUIDE.md](ADAPTIVE_INDICATOR_GUIDE.md) - 적응형 지표 가이드
- [ML_PREDICTION_GUIDE.md](ML_PREDICTION_GUIDE.md) - 머신러닝 예측 시스템 가이드
- [TFT_DUAL_MODEL_DESIGN_GUIDE.md](TFT_DUAL_MODEL_DESIGN_GUIDE.md) - TFT 설계 가이드

### 운영 가이드
- [DAILY_TICK_TRAINING_RUNBOOK.md](DAILY_TICK_TRAINING_RUNBOOK.md) - 운영 런북
