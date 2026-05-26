"""Pivot Prediction Inference

학습된 피봇 예측 모델을 사용하여 후보 확정 확률 예측.

Usage:
    python prediction/pivot_inference.py \
        --model_path prediction/weights/pivot_classifier_best.pt \
        --features {...}
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

from pivot_models import PivotConfirmationClassifier
from features import ADAPT_KEYS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class PivotPredictor:
    """피봇 예측기."""
    
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
        hidden_dim = checkpoint.get("hidden_dim", 128)
        
        self.model = PivotConfirmationClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.test_metrics = checkpoint.get("test_metrics", {})
        
        _logger.info(f"모델 로드 완료: {model_path}")
        _logger.info(f"테스트 메트릭: {self.test_metrics}")
    
    def predict(
        self,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """확정 확률 예측.
        
        Args:
            features: 피처 딕셔너리 (ADAPT_KEYS 기준)
        
        Returns:
            예측 결과 딕셔너리
        """
        # 피처 벡터 생성
        feature_vector = []
        for key in ADAPT_KEYS:
            val = features.get(key, 0.0)
            if isinstance(val, (int, float, np.number)):
                feature_vector.append(float(val))
            else:
                feature_vector.append(0.0)
        
        # 텐서 변환
        x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            prob = self.model.predict(x)
            prob = prob.cpu().item()
        
        return {
            "confirmation_probability": prob,
            "prediction": 1 if prob >= 0.5 else 0,
            "confidence": abs(prob - 0.5) * 2,  # 0~1
        }
    
    def predict_batch(
        self,
        features_list: list[Dict[str, float]],
    ) -> list[Dict[str, float]]:
        """배치 예측.
        
        Args:
            features_list: 피처 딕셔너리 리스트
        
        Returns:
            예측 결과 리스트
        """
        # 피처 행렬 생성
        feature_matrix = []
        for features in features_list:
            feature_vector = []
            for key in ADAPT_KEYS:
                val = features.get(key, 0.0)
                if isinstance(val, (int, float, np.number)):
                    feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            feature_matrix.append(feature_vector)
        
        # 텐서 변환
        x = torch.tensor(feature_matrix, dtype=torch.float32).to(self.device)
        
        # 예측
        with torch.no_grad():
            probs = self.model.predict(x)
            probs = probs.cpu().numpy()
        
        results = []
        for prob in probs:
            results.append({
                "confirmation_probability": float(prob),
                "prediction": 1 if prob >= 0.5 else 0,
                "confidence": abs(prob - 0.5) * 2,
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Pivot Prediction Inference")
    parser.add_argument("--model_path", type=str, required=True, help="모델 경로")
    parser.add_argument("--device", type=str, default="cuda", help="장치")
    
    args = parser.parse_args()
    
    # 예측기 초기화
    predictor = PivotPredictor(args.model_path, args.device)
    
    # 테스트 예측
    test_features = {key: 0.0 for key in ADAPT_KEYS}
    test_features["azz_direction"] = 1.0
    test_features["azz_structure_up"] = 1.0
    test_features["azz_pending_prob"] = 0.8
    
    result = predictor.predict(test_features)
    _logger.info(f"테스트 예측: {result}")


if __name__ == "__main__":
    main()
