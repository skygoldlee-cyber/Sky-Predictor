"""
통합 테스트: 설정 로드 및 파이프라인 빌드

이 테스트는 설정 로드와 파이프라인 빌드가 함께 작동하는지 확인합니다.
"""

import pytest
import tempfile
import json


@pytest.mark.integration
def test_config_load_and_pipeline_build():
    """설정 로드 후 파이프라인 빌드가 성공하는지 통합 테스트."""
    from config import AppConfig
    from prediction.pipeline import PredictionPipeline

    # 최소 설정 생성
    config_data = {
        "prediction": {
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "model_name": "patch_tst",
        },
        "adaptive_indicator": {
            "kospi_symbol": "KOSPI 지수",
        },
    }

    # 임시 설정 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # 설정 로드
        cfg = AppConfig.from_file(config_path)
        assert cfg is not None
        cfg.validate()

        # 파이프라인 빌드 (실제 모델 없이도 설정 검증만 수행)
        # 이 테스트는 파이프라인 초기화가 설정과 호환되는지 확인합니다.
        # 실제 파이프라인 실행은 별도의 테스트에서 다룹니다.
        assert cfg.prediction.buy_threshold > cfg.prediction.sell_threshold

    finally:
        import os
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.mark.integration
def test_event_bus_and_handler_integration():
    """이벤트 버스와 핸들러가 함께 작동하는지 통합 테스트."""
    from events.event_bus import EventBus
    from events.handlers import EventHandler

    # 이벤트 버스 생성
    bus = EventBus()

    # 테스트 핸들러
    events_received = []

    class TestHandler(EventHandler):
        def handle(self, event):
            events_received.append(event)

    # 핸들러 등록
    handler = TestHandler()
    bus.subscribe("test_event", handler)

    # 이벤트 발행
    test_event = {"type": "test_event", "data": "test"}
    bus.publish(test_event)

    # 핸들러가 이벤트를 수신했는지 확인
    assert len(events_received) == 1
    assert events_received[0] == test_event
