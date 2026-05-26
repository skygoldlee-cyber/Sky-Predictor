# eBest API Integration

이 디렉토리는 eBest OpenAPI와의 통합을 위한 서비스를 포함합니다.

## 구성 요소

### fetch_market_data_service.py

eBest OpenAPI의 t8415/t8418 쿼리를 사용하여 분봉 데이터를 가져오는 서비스입니다.

#### 기능

- **t8415**: 선물/옵션 분봉 OHLCV 데이터 조회
- **t8418**: KOSPI 지수 분봉 데이터 조회
- 전일 데이터 추출 및 Pivot 포인트 계산
- 리플레이 모드 및 장 시작 전 옵션 데이터 스킵 처리

#### 사용 방법

```python
from services.fetch_market_data_service import FetchMarketDataService
from data.tick_processor import RealTimeTickProcessor

# 1. 서비스 인스턴스 생성
fetch_service = FetchMarketDataService(api_client=your_ebest_client)

# 2. TickProcessor에 서비스 주입
processor = RealTimeTickProcessor(
    fetch_market_service=fetch_service
)

# 3. ChartViewerWidget에 API 사용 파라미터 전달
from gui.chart_viewer import ChartViewerWidget

# 방법 1: config.json에서 자동 가져오기 (권장)
viewer = ChartViewerWidget(
    predictor=predictor,
    config=config,
    use_api=True  # 종목 코드는 config.json에서 자동 가져옴
)

# 방법 2: 직접 종목 코드 전달
viewer = ChartViewerWidget(
    predictor=predictor,
    config=config,
    use_api=True,
    kp200_upcode="101V3000",  # KP200 선물 코드
    kospi_upcode="001"        # KOSPI 지수 코드
)
```

#### 종목 코드 가져오기 순서

`ChartViewerWidget`은 다음 순서로 종목 코드를 가져옵니다:

1. **생성자 파라미터**: `kp200_upcode`, `kospi_upcode`가 직접 전달된 경우
2. **config.json**: `ebest.kp200_upcode`, `ebest.kospi_upcode` 설정
3. **Predictor 객체**: `predictor.kp200_symbol`, `predictor.kospi_symbol` 속성
4. **Pipeline 객체**: `predictor.pipeline.kp200_symbol`, `predictor.pipeline.kospi_symbol` 속성

#### config.json 설정

```json
{
  "ebest": {
    "kp200_upcode": "101V3000",
    "kospi_upcode": "001"
  }
}
```

#### API 사용 시 주의사항

1. **이벤트 루프 제약**: 현재 구현에서는 이벤트 루프가 실행 중인 경우 API 호출이 자동으로 틱 집계 방식으로 폴백됩니다. 비동기 환경에서의 완전한 지원은 추가 구현이 필요합니다.

2. **API 클라이언트**: 실제 eBest API 클라이언트를 `FetchMarketDataService` 생성자에 전달해야 합니다. 현재는 더미 구현으로 빈 DataFrame을 반환합니다.

3. **종목 코드 자동 가져오기**: 코드 내에서 종목 코드를 자동으로 가져올 수 있으므로 별도로 전달할 필요가 없습니다.

## 구현 상태

- [x] `fetch_market_data_service.py` 기본 구조 완료
- [x] `tick_processor.py`에 API 통합 완료
- [x] `chart_viewer.py`에 API 파라미터 전달 완료
- [x] 종목 코드 자동 가져오기 기능 구현 (config/predictor에서)
- [x] `config.json`에 종목 코드 설정 추가
- [ ] 실제 eBest API 클라이언트 연결 필요
- [ ] 비동기 이벤트 루프 환경에서의 완전한 지원 필요

## 다음 단계

1. eBest API 클라이언트 구현
2. `_mock_api_request` 메서드를 실제 API 호출로 교체
3. 비동기 환경에서의 API 호출 최적화
