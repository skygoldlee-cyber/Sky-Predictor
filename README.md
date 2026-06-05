# SkyPredictor

KOSPI200 선물/옵션 틱 데이터를 기반으로 실시간 방향성 예측을 제공하는 시스템입니다. 오더북 분석, 기술적 지표, 머신러닝 모델, LLM 판단을 통합하여 BUY/SELL/HOLD 신호를 생성합니다.

## 주요 기능

- **실시간 틱 처리**: eBest API로부터 선물/옵션 틱 데이터 수신 및 분봉 집계
- **오더북 분석**: FH0 호가 스냅샷 기반 오더북 불균형 피처 추출
- **적응형 지표**: Adaptive ZigZag, Adaptive SuperTrend로 시장 구조/추세 판단
- **ML 예측**: Transformer, TFT, Ensemble 모델로 수치적 확률 예측
- **LLM 판단**: Claude/GPT/Gemini로 전략적 해석 및 판단
- **가드레일**: 베이시스, 옵션 유동성, 패리티 다이버전스 등으로 신호 필터링
- **텔레그램 연동**: 실시간 알림 및 명령 처리

## 설치

### 전제 조건

- Python 3.10 이상
- Windows OS (eBest API 호환)

### 단계별 설치

1. 저장소 클론:
```bash
git clone https://github.com/skygoldlee-cyber/Sky-Predictor.git
cd SkyPredictor
```

2. 가상 환경 생성:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 설정

### config.json

프로젝트 루트에 `config.json`을 생성합니다:

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
    "supertrend": {
      "atr_period": 10,
      "atr_multiplier": 3.0
    },
    "zigzag": {
      "atr_period": 14,
      "atr_multiplier": 1.5,
      "confirmation_bars": 2
    }
  }
}
```

### config.secrets.json

민감 정보는 `config.secrets.json`에 저장합니다 (gitignore 대상):

```json
{
  "ai_providers": {
    "anthropic": { "api_key": "" },
    "openai": { "api_key": "" },
    "gemini": { "api_key": "" }
  },
  "ebest": {
    "appkey": "",
    "appsecretkey": ""
  }
}
```

또는 환경변수로 설정:
```powershell
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:EBEST_APPKEY="..."
$env:EBEST_APPSECRET="..."
```

## 사용법

### GUI 모드 (기본)
```bash
python main.py
```

### CLI 모드
```bash
python main.py --cli
```

### 테스트
```bash
python main.py --test
# 또는
python -m pytest -q
```

### Heuristic-only 모드 (실시간 연결 없이)
```bash
python main.py --heuristic-only --no-ebest-live
```

### 오프라인 학습

데이터셋 생성:
```bash
python -m prediction.data_builder --config config.json \
  --files ticks_replay_YYYYMMDD.jsonl.gz \
  --out dataset_YYYYMMDD.npz --seq-len 60 --horizon 5
```

Transformer 학습:
```bash
python train.py --config config.json --data dataset_YYYYMMDD.npz \
  --out prediction/weights/transformer_5m.pt
```

TFT 학습:
```bash
python train_tft.py --config config.json --data dataset_tft_YYYYMMDD.npz \
  --out prediction/weights/tft_5m.pt
```

## 프로젝트 구조

```
SkyPredictor/
├── core/              # 핵심 유틸리티 (CLI, 로깅, 유틸)
├── config/            # 설정 관리
├── data/              # 틱 처리, 분봉 집계
├── indicators/        # 기술적 지표 (Adaptive ZigZag, SuperTrend)
├── prediction/        # 예측 시스템 (Pipeline, Predictor, Model, TFT)
├── ebestapi/          # eBest API 연동
├── telegram/          # 텔레그램 연동
├── training/          # 학습 스크립트
├── app/               # 애플리케이션 설정
├── docs/              # 문서
├── tests/             # 테스트
├── main.py            # 진입점
├── config.json        # 설정 파일
└── requirements.txt   # 의존성
```

## 문서

### 메인 문서
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 시스템 아키텍처 상세
- [docs_README.md](docs/docs_README.md) - 프로젝트 개요 및 상세 사용 가이드

### 기술 가이드
- [ADAPTIVE_INDICATOR_GUIDE.md](docs/ADAPTIVE_INDICATOR_GUIDE.md) - 적응형 지표 가이드
- [ML_PREDICTION_GUIDE.md](docs/ML_PREDICTION_GUIDE.md) - 머신러닝 예측 시스템 가이드
- [TFT_DUAL_MODEL_DESIGN_GUIDE.md](docs/TFT_DUAL_MODEL_DESIGN_GUIDE.md) - TFT 설계 가이드

### 운영 가이드
- [DAILY_TICK_TRAINING_RUNBOOK.md](docs/operations/DAILY_TICK_TRAINING_RUNBOOK.md) - 운영 런북
- [telegram.md](docs/operations/telegram.md) - 텔레그램 연동 가이드

## 라이선스

MIT License

## 기여

기여를 환영합니다! Pull Request를 제출하기 전에 테스트를 실행해 주세요.
