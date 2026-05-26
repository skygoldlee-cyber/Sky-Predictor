# SkyPredictor

SkyPredictor는 한국 주식 시장을 위한 고급 기술적 분석 및 예측 시스템입니다. 하이브리드 적응형 피봇(Hybrid Adaptive Pivot) 탐지, AI 기반 예측, 실시간 트레이딩 시스템을 통합하여 트레이더에게 포괄적인 도구를 제공합니다.

## 주요 기능

- **하이브리드 적응형 피봇 (HAP)**: ATR과 퍼센트 기반 피봇 탐지를 결합한 고급 기술적 지표
- **AI 기반 예측**: Transformer, TFT, Mamba 등 다양한 딥러닝 모델을 활용한 피봇 예측
- **실시간 트레이딩**: E-Best API를 통한 실시간 데이터 수집 및 자동 트레이딩
- **백테스트 시스템**: 파라미터 최적화 및 성과 분석
- **GUI 대시보드**: PyQt 기반 직관적인 트레이딩 인터페이스
- **시장 레짐 분류**: 시장 상태를 자동으로 분류하여 파라미터 추천

## 설치

### 전제 조건

- Python 3.10 이상
- Windows OS (E-Best API 호환)

### 단계별 설치

1. 저장소 클론:
```bash
git clone <repository-url>
cd SkyPredictor
```

2. 가상 환경 생성:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. 의존성 설치:
```bash
pip install -e ".[all]"
```

또는 개발용 의존성:
```bash
pip install -e ".[dev,all]"
```

참고: 이 프로젝트는 `pyproject.toml`를 단일 의존성 소스로 사용합니다. `requirements.txt`는 더 이상 사용되지 않습니다.

## 설정

1. `config/config.json` 복사 및 수정:
```bash
cp config/config.example.json config/config.json
```

2. `config/config.secrets.json` 생성 (민감 정보):
```json
{
  "ebest": {
    "api_key": "your-api-key",
    "api_secret": "your-api-secret"
  }
}
```

## 사용법

### GUI 실행
```bash
python main.py
```

### 백테스트 실행
```bash
python scripts/run_daily_backtest.py
```

### 데이터 수집
```bash
python scripts/fetch_daily_data.py
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 단위 테스트만 실행
pytest tests/ -m unit

# 통합 테스트만 실행
pytest tests/ -m integration

# 커버리지 리포트 생성
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

### CI/CD
이 프로젝트는 GitHub Actions를 사용하여 CI/CD 파이프라인을 구성했습니다. `.github/workflows/ci.yml`에서 설정을 확인할 수 있습니다.

자세한 사용법은 [GitHub Actions 가이드](docs/GITHUB_ACTIONS_GUIDE.md)를 참조하세요.

### 코드 커버리지
로컬에서 커버리지 리포트를 생성하려면:
```bash
# Linux/Mac
bash scripts/run_coverage.sh

# Windows
scripts\run_coverage.bat
```

## 프로젝트 구조

```
SkyPredictor/
├── app/              # 애플리케이션 설정 및 실행 모드
├── config/           # 설정 파일
├── core/             # 핵심 유틸리티
├── data/             # 데이터 처리
├── docs/             # 문서
├── ebestapi/         # E-Best API 연동
├── events/           # 이벤트 버스 및 핸들러
├── gui/              # PyQt GUI
├── indicators/       # 기술적 지표
├── prediction/       # 예측 및 ML 모델
├── scripts/          # 유틸리티 스크립트
├── services/         # 백그라운드 서비스
├── telegram/         # Telegram 알림
├── tests/            # 테스트
├── trading/          # 트레이딩 로직
└── main.py           # 진입점
```

## 문서

- [하이브리드 적응형 피봇 설계](docs/design/HybridAdaptivePivot_Design_v1.1.md)
- [트레이드 로깅 가이드](docs/guides/TRADE_LOGGING_GUIDE.md)
- [추가 문서](docs/)

## 라이선스

MIT License

## 기여

기여를 환영합니다! Pull Request를 제출하기 전에 테스트를 실행해 주세요.

## 연락처

프로젝트 관련 문의는 이슈 트래커를 통해 제출해 주세요.
