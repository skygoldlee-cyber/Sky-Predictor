"""Pivot Lifespan Prediction Inference

학습된 시계열 모델을 사용하여 후보 수명 예측.

Usage:
    python prediction/pivot_lifespan_inference.py \
        --model_path prediction/weights/pivot_lifespan_best.pt \
        --sequence [...]
"""

import argparse
import logging
from typing import Dict, List

import torch
import numpy as np

from pivot_models import PivotLifespanPredictor
from features import ADAPT_KEYS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class LifespanPredictor:
    """후보 수명 예측기."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """초기화.
        
        Args:
            model_path: 모델 가중치 경로
            device: 장치
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        input_dim = checkpoint.get("input_dim", len(ADAPT_KEYS))
        hidden_dim = checkpoint.get("hidden_dim", 64)
        num_layers = checkpoint.get("num_layers", 2)
        max_seq_len = checkpoint.get("max_seq_len", 120)
        
        self.model = PivotLifespanPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.max_seq_len = max_seq_len
        self.test_metrics = checkpoint.get("test_metrics", {})
        
        _logger.info(f"모델 로드 완료: {model_path}")
        _logger.info(f"테스트 메트릭: {self.test_metrics}")
    
    def predict(
        self,
        sequence: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """후보 '잔여' 수명 예측.

        Args:
            sequence: 등록~현재까지 관측된 부분 시퀀스 (각 요소는 피처 딕셔너리)

        Returns:
            예측 결과 딕셔너리.
            - predicted_remaining_bars: 지금부터 확정/취소까지 남은 봉수(모델 출력)
            - predicted_lifespan_bars: 전체 수명 추정 = 경과 봉수 + 잔여 봉수
              (경과 ≈ 관측 스냅샷 수 - 1; 하위 호환용)
        """
        # 시계열 피처 추출 (시간 순서 유지)
        seq_features = []
        for snapshot in sequence:
            feature_vector = []
            for key in ADAPT_KEYS:
                val = snapshot.get(key, 0.0)
                if isinstance(val, (int, float, np.number)):
                    feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            seq_features.append(feature_vector)

        if not seq_features:
            return {
                "predicted_remaining_bars": 0.0,
                "predicted_lifespan_bars": 0.0,
                "sequence_length": 0,
                "confidence": 0.0,
            }

        # 트리밍 (최근 max_seq_len) + 실제 길이 기록 → packing 으로 패딩 무시
        if len(seq_features) > self.max_seq_len:
            seq_features = seq_features[-self.max_seq_len:]
        true_len = len(seq_features)
        while len(seq_features) < self.max_seq_len:
            seq_features.append([0.0] * len(ADAPT_KEYS))

        # 텐서 변환
        x = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([true_len], dtype=torch.long)

        # 예측 (학습과 동일하게 lengths 전달)
        with torch.no_grad():
            remaining_log = self.model.predict(x, lengths)
            remaining_log = remaining_log.cpu().item()

        # 역정규화 (잔여 수명)
        remaining = float(max(0.0, np.expm1(remaining_log)))

        # 전체 수명 추정 = 경과(관측 봉수-1) + 잔여
        elapsed = max(0, len(sequence) - 1)
        total_lifespan = float(elapsed + remaining)

        return {
            "predicted_remaining_bars": remaining,
            "predicted_lifespan_bars": total_lifespan,
            "sequence_length": len(sequence),
            "confidence": min(1.0, len(sequence) / 10.0),  # 시퀀스 길이 기반 신뢰도
        }
    
    def predict_from_collector(
        self,
        collector_record: Dict,
    ) -> Dict[str, float]:
        """Collector 레코드로부터 예측.

        Args:
            collector_record: CandidateRecord 딕셔너리

        Returns:
            예측 결과 딕셔너리
        """
        sequence = collector_record.get("sequence", [])

        # 각 스냅샷의 features 딕셔너리만 추출하여 predict 에 전달
        # (predict 가 ADAPT_KEYS 순서로 벡터화하므로 여기서는 dict 그대로 넘긴다)
        feature_dicts = [
            snap.get("features", {})
            for snap in sequence
            if isinstance(snap.get("features"), dict) and len(snap["features"]) > 0
        ]

        return self.predict(feature_dicts)


def main():
    parser = argparse.ArgumentParser(description="Pivot Lifespan Prediction Inference")
    parser.add_argument("--model_path", type=str, required=True, help="모델 경로")
    parser.add_argument("--device", type=str, default="cuda", help="장치")
    
    args = parser.parse_args()
    
    # 예측기 초기화
    predictor = LifespanPredictor(args.model_path, args.device)
    
    # 테스트 예측
    test_sequence = [{key: 0.0 for key in ADAPT_KEYS} for _ in range(10)]
    test_sequence[0]["azz_direction"] = 1.0
    
    result = predictor.predict(test_sequence)
    _logger.info(f"테스트 예측: {result}")


if __name__ == "__main__":
    main()
