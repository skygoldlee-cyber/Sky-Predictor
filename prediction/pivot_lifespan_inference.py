"""Pivot Lifespan Prediction Inference

학습된 시계열 모델을 사용하여 후보 수명 예측.

Usage:
    python prediction/pivot_lifespan_inference.py \
        --model_path prediction/weights/pivot_lifespan_best.pt \
        --sequence [...]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
        """후보 수명 예측.
        
        Args:
            sequence: 시계열 피처 리스트 (각 요소는 피처 딕셔너리)
        
        Returns:
            예측 결과 딕셔너리
        """
        # 시계열 피처 추출
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
        
        # 패딩/트리밍
        if len(seq_features) > self.max_seq_len:
            seq_features = seq_features[-self.max_seq_len:]
        else:
            while len(seq_features) < self.max_seq_len:
                seq_features.append([0.0] * len(ADAPT_KEYS))
        
        # 텐서 변환
        x = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            lifespan_log = self.model.predict(x)
            lifespan_log = lifespan_log.cpu().item()
        
        # 역정규화
        lifespan = np.expm1(lifespan_log)
        
        return {
            "predicted_lifespan_bars": float(lifespan),
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
        
        # 스냅샷에서 피처 추출
        seq_features = []
        for snapshot in sequence:
            features = snapshot.get("features", {})
            feature_vector = []
            for key in ADAPT_KEYS:
                val = features.get(key, 0.0)
                if isinstance(val, (int, float, np.number)):
                    feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            seq_features.append(features)
        
        return self.predict(seq_features)


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
