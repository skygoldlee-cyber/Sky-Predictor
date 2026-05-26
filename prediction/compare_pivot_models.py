"""Pivot Model Comparison Script

분류 모델과 회귀 모델의 성능 비교.

Usage:
    python prediction/compare_pivot_models.py \
        --classifier_path prediction/weights/pivot_classifier_best.pt \
        --regressor_path prediction/weights/pivot_regressor_best.pt \
        --data_path data/pivot_candidates.pkl
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pivot_models import PivotConfirmationClassifier, PivotProbabilityRegressor
from features import ADAPT_KEYS
from train_pivot_classifier import PivotDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


def load_model(model_path: str, model_class, device: str = "cuda"):
    """모델 로드."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = checkpoint.get("input_dim", len(ADAPT_KEYS))
    hidden_dim = checkpoint.get("hidden_dim", 128)
    
    model = model_class(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    test_metrics = checkpoint.get("test_metrics", {})
    
    return model, device, test_metrics


def compare_models(
    classifier_path: str,
    regressor_path: str,
    data_path: str,
    output_dir: str = "prediction/results",
):
    """모델 비교."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    _logger.info("모델 로드 중...")
    classifier, device, cls_metrics = load_model(
        classifier_path, PivotConfirmationClassifier
    )
    regressor, device, reg_metrics = load_model(
        regressor_path, PivotProbabilityRegressor
    )
    
    # 데이터셋 로드
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    records = data["completed_candidates"]
    
    dataset = PivotDataset(records, ADAPT_KEYS)
    
    # 예측
    _logger.info("예측 중...")
    classifier_probs = []
    regressor_probs = []
    labels = []
    
    with torch.no_grad():
        for X, y in dataset:
            X = X.unsqueeze(0).to(device)
            cls_prob = classifier(X).squeeze(-1).cpu().item()
            reg_prob = regressor(X).squeeze(-1).cpu().item()
            
            classifier_probs.append(cls_prob)
            regressor_probs.append(reg_prob)
            labels.append(y.item())
    
    classifier_probs = np.array(classifier_probs)
    regressor_probs = np.array(regressor_probs)
    labels = np.array(labels)
    
    # 메트릭 계산
    cls_preds = (classifier_probs >= 0.5).astype(int)
    reg_preds = (regressor_probs >= 0.5).astype(int)
    
    cls_accuracy = accuracy_score(labels, cls_preds)
    reg_accuracy = accuracy_score(labels, reg_preds)
    
    cls_precision = precision_score(labels, cls_preds, zero_division=0)
    reg_precision = precision_score(labels, reg_preds, zero_division=0)
    
    cls_recall = recall_score(labels, cls_preds, zero_division=0)
    reg_recall = recall_score(labels, reg_preds, zero_division=0)
    
    cls_f1 = f1_score(labels, cls_preds, zero_division=0)
    reg_f1 = f1_score(labels, reg_preds, zero_division=0)
    
    cls_auc = roc_auc_score(labels, classifier_probs)
    reg_auc = roc_auc_score(labels, regressor_probs)
    
    # 결과 출력
    _logger.info("=" * 60)
    _logger.info("분류 모델 vs 회귀 모델 비교")
    _logger.info("=" * 60)
    _logger.info(f"분류 모델 테스트 메트릭: {cls_metrics}")
    _logger.info(f"회귀 모델 테스트 메트릭: {reg_metrics}")
    _logger.info("-" * 60)
    _logger.info(f"Accuracy:  분류={cls_accuracy:.4f}, 회귀={reg_accuracy:.4f}")
    _logger.info(f"Precision: 분류={cls_precision:.4f}, 회귀={reg_precision:.4f}")
    _logger.info(f"Recall:    분류={cls_recall:.4f}, 회귀={reg_recall:.4f}")
    _logger.info(f"F1 Score:  분류={cls_f1:.4f}, 회귀={reg_f1:.4f}")
    _logger.info(f"AUC:       분류={cls_auc:.4f}, 회귀={reg_auc:.4f}")
    _logger.info("=" * 60)
    
    # ROC 곡선 그리기
    try:
        cls_fpr, cls_tpr, _ = roc_curve(labels, classifier_probs)
        reg_fpr, reg_tpr, _ = roc_curve(labels, regressor_probs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(cls_fpr, cls_tpr, label=f'Classifier (AUC = {cls_auc:.4f})')
        plt.plot(reg_fpr, reg_tpr, label=f'Regressor (AUC = {reg_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        roc_path = output_path / "roc_comparison.png"
        plt.savefig(roc_path)
        _logger.info(f"ROC 곡선 저장: {roc_path}")
    except Exception as e:
        _logger.warning(f"ROC 곡선 그리기 실패: {e}")
    
    # 결과 저장
    comparison_result = {
        "classifier_metrics": {
            "accuracy": cls_accuracy,
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "auc": cls_auc,
        },
        "regressor_metrics": {
            "accuracy": reg_accuracy,
            "precision": reg_precision,
            "recall": reg_recall,
            "f1": reg_f1,
            "auc": reg_auc,
        },
    }
    
    result_path = output_path / "comparison_result.pkl"
    with open(result_path, 'wb') as f:
        pickle.dump(comparison_result, f)
    _logger.info(f"비교 결과 저장: {result_path}")
    
    return comparison_result


def main():
    parser = argparse.ArgumentParser(description="Pivot Model Comparison")
    parser.add_argument("--classifier_path", type=str, required=True, help="분류 모델 경로")
    parser.add_argument("--regressor_path", type=str, required=True, help="회귀 모델 경로")
    parser.add_argument("--data_path", type=str, required=True, help="데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="prediction/results", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    compare_models(
        classifier_path=args.classifier_path,
        regressor_path=args.regressor_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
