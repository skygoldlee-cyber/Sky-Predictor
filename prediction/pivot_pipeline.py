"""Pivot Prediction Pipeline

피봇 예측 통합 파이프라인.

분류/회귀/시계열 모델을 통합하여 후보 확정 확률 및 수명 예측.

Usage:
    from prediction.pivot_pipeline import PivotPredictionPipeline
    
    pipeline = PivotPredictionPipeline(
        classifier_path="prediction/weights/pivot_classifier_best.pt",
        regressor_path="prediction/weights/pivot_regressor_best.pt",
        lifespan_path="prediction/weights/pivot_lifespan_best.pt",
        zigzag=zigzag,
    )
    
    result = pipeline.predict(close)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from pivot_inference import PivotPredictor
from pivot_lifespan_inference import LifespanPredictor as LifespanInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class PivotPredictionPipeline:
    """피봇 예측 통합 파이프라인.
    
    분류 모델, 회귀 모델, 시계열 모델을 통합하여
    후보 확정 확률 및 수명을 예측합니다.
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        regressor_path: Optional[str] = None,
        lifespan_path: Optional[str] = None,
        zigzag=None,
        device: str = "cuda",
        ensemble_weight: float = 0.5,
    ):
        """초기화.
        
        Args:
            classifier_path: 분류 모델 경로
            regressor_path: 회귀 모델 경로
            lifespan_path: 시계열 모델 경로
            zigzag: AdaptiveZigZag 인스턴스
            device: 장치
            ensemble_weight: 앙상블 가중치 (분류 모델)
        """
        self.zigzag = zigzag
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ensemble_weight = ensemble_weight
        
        # 분류 모델
        self.classifier = None
        if classifier_path and Path(classifier_path).exists():
            try:
                self.classifier = PivotPredictor(classifier_path, device)
                _logger.info(f"분류 모델 로드 완료: {classifier_path}")
            except Exception as e:
                _logger.error(f"분류 모델 로드 실패: {e}")
        
        # 회귀 모델
        self.regressor = None
        if regressor_path and Path(regressor_path).exists():
            try:
                self.regressor = PivotPredictor(regressor_path, device)
                _logger.info(f"회귀 모델 로드 완료: {regressor_path}")
            except Exception as e:
                _logger.error(f"회귀 모델 로드 실패: {e}")
        
        # 시계열 모델
        self.lifespan_predictor = None
        if lifespan_path and Path(lifespan_path).exists():
            try:
                self.lifespan_predictor = LifespanInference(lifespan_path, device)
                _logger.info(f"시계열 모델 로드 완료: {lifespan_path}")
            except Exception as e:
                _logger.error(f"시계열 모델 로드 실패: {e}")
        
        _logger.info(f"파이프라인 초기화 완료 (device={self.device})")
    
    def predict(self, close: float) -> Dict[str, Any]:
        """현재 후보의 확정 확률 및 수명 예측.

        Args:
            close: 현재 종가

        Returns:
            예측 결과 딕셔너리
        """
        if self.zigzag is None:
            return {"has_candidate": False, "error": "zigzag not set"}

        # 후보 확인
        if not self.zigzag._pending_confirm:
            return {
                "has_candidate": False,
                "candidate_type": None,
                "candidate_price": None,
            }

        # 후보 정보
        pc = self.zigzag._pending_confirm
        candidate_type = pc.get("type")
        candidate_price = pc.get("price")

        # 피처 추출
        try:
            features = self.zigzag.get_transformer_features(close)
        except Exception as e:
            _logger.error(f"피처 추출 실패: {e}")
            return {
                "has_candidate": True,
                "candidate_type": candidate_type,
                "candidate_price": candidate_price,
                "error": str(e),
            }

        result = {
            "has_candidate": True,
            "candidate_type": candidate_type,
            "candidate_price": candidate_price,
        }

        # 분류 모델 예측
        cls_prob = None
        if self.classifier is not None:
            try:
                cls_result = self.classifier.predict(features)
                cls_prob = cls_result["confirmation_probability"]
                result["classification_prob"] = cls_prob
                result["classification_prediction"] = cls_result["prediction"]
                result["classification_confidence"] = cls_result["confidence"]
            except Exception as e:
                _logger.error(f"분류 모델 예측 실패: {e}")

        # 회귀 모델 예측
        reg_prob = None
        if self.regressor is not None:
            try:
                reg_result = self.regressor.predict(features)
                reg_prob = reg_result["confirmation_probability"]
                result["regression_prob"] = reg_prob
                result["regression_prediction"] = reg_result["prediction"]
                result["regression_confidence"] = reg_result["confidence"]
            except Exception as e:
                _logger.error(f"회귀 모델 예측 실패: {e}")

        # 앙상블
        if cls_prob is not None and reg_prob is not None:
            ensemble_prob = self.ensemble_weight * cls_prob + (1 - self.ensemble_weight) * reg_prob
            result["ensemble_prob"] = ensemble_prob
            result["ensemble_prediction"] = 1 if ensemble_prob >= 0.5 else 0
            result["ensemble_confidence"] = abs(ensemble_prob - 0.5) * 2
        elif cls_prob is not None:
            result["ensemble_prob"] = cls_prob
            result["ensemble_prediction"] = 1 if cls_prob >= 0.5 else 0
            result["ensemble_confidence"] = abs(cls_prob - 0.5) * 2
        elif reg_prob is not None:
            result["ensemble_prob"] = reg_prob
            result["ensemble_prediction"] = 1 if reg_prob >= 0.5 else 0
            result["ensemble_confidence"] = abs(reg_prob - 0.5) * 2

        # Heuristic 확률
        try:
            heuristic_prob = self.zigzag.get_pending_confirmation_probability(close)
            result["heuristic_prob"] = heuristic_prob
        except Exception as e:
            _logger.error(f"Heuristic 확률 계산 실패: {e}")

        # 시계열 모델 예측 (수명)
        if self.lifespan_predictor is not None:
            try:
                # Collector에서 시퀀스 가져오기
                collector = self.zigzag.pivot_collector
                if collector is not None:
                    candidate_id = getattr(self.zigzag, "_current_candidate_id", None)
                    if candidate_id in collector.active_candidates:
                        record = collector.active_candidates[candidate_id]
                        sequence = record.sequence
                        # 스냅샷을 피처 딕셔너리로 변환
                        seq_features = []
                        for snapshot in sequence:
                            seq_features.append(snapshot.features)

                        lifespan_result = self.lifespan_predictor.predict(seq_features)
                        result["predicted_lifespan_bars"] = lifespan_result["predicted_lifespan_bars"]
                        result["predicted_remaining_bars"] = lifespan_result.get("predicted_remaining_bars")
                        result["lifespan_confidence"] = lifespan_result["confidence"]
            except Exception as e:
                _logger.error(f"시계열 모델 예측 실패: {e}")

        # [P3] 사전 신호 발생 (확정 확률 70% 이상)
        ensemble_prob = result.get("ensemble_prob", 0.0)
        if ensemble_prob >= 0.7:
            if candidate_type == "low":
                result["early_signal"] = "BUY"
            elif candidate_type == "high":
                result["early_signal"] = "SELL"
            else:
                result["early_signal"] = None
            result["early_confidence"] = "MEDIUM"  # 사전 신호는 MEDIUM
            result["early_prob"] = ensemble_prob
            _logger.info(
                f"[EARLY_SIGNAL] 사전 신호 발생: type={candidate_type}, prob={ensemble_prob:.2f}, signal={result['early_signal']}"
            )
        else:
            result["early_signal"] = None

        return result
    
    def predict_batch(self, features_list: list[Dict[str, float]]) -> list[Dict[str, Any]]:
        """배치 예측 (분류/회귀만).
        
        Args:
            features_list: 피처 딕셔너리 리스트
        
        Returns:
            예측 결과 리스트
        """
        results = []
        
        # 분류 모델 배치 예측
        cls_probs = None
        if self.classifier is not None:
            try:
                cls_results = self.classifier.predict_batch(features_list)
                cls_probs = [r["confirmation_probability"] for r in cls_results]
            except Exception as e:
                _logger.error(f"분류 모델 배치 예측 실패: {e}")
        
        # 회귀 모델 배치 예측
        reg_probs = None
        if self.regressor is not None:
            try:
                reg_results = self.regressor.predict_batch(features_list)
                reg_probs = [r["confirmation_probability"] for r in reg_results]
            except Exception as e:
                _logger.error(f"회귀 모델 배치 예측 실패: {e}")
        
        # 결과 조합
        for i in range(len(features_list)):
            result = {}
            
            if cls_probs is not None:
                result["classification_prob"] = cls_probs[i]
                result["classification_prediction"] = 1 if cls_probs[i] >= 0.5 else 0
            
            if reg_probs is not None:
                result["regression_prob"] = reg_probs[i]
                result["regression_prediction"] = 1 if reg_probs[i] >= 0.5 else 0
            
            # 앙상블
            if cls_probs is not None and reg_probs is not None:
                ensemble_prob = self.ensemble_weight * cls_probs[i] + (1 - self.ensemble_weight) * reg_probs[i]
                result["ensemble_prob"] = ensemble_prob
                result["ensemble_prediction"] = 1 if ensemble_prob >= 0.5 else 0
            elif cls_probs is not None:
                result["ensemble_prob"] = cls_probs[i]
                result["ensemble_prediction"] = 1 if cls_probs[i] >= 0.5 else 0
            elif reg_probs is not None:
                result["ensemble_prob"] = reg_probs[i]
                result["ensemble_prediction"] = 1 if reg_probs[i] >= 0.5 else 0
            
            results.append(result)
        
        return results
    
    def get_model_status(self) -> Dict[str, bool]:
        """모델 로드 상태 확인."""
        return {
            "classifier_loaded": self.classifier is not None,
            "regressor_loaded": self.regressor is not None,
            "lifespan_loaded": self.lifespan_predictor is not None,
            "zigzag_connected": self.zigzag is not None,
        }
