"""Pivot Prediction Models

피봇 후보 확정/취소 예측 모델들.

Models:
- PivotConfirmationClassifier: 이진 분류 (확정/취소)
- PivotProbabilityRegressor: 회귀 (확정 확률)
- PivotLifespanPredictor: 시계열 (후보 수명)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

_logger = logging.getLogger(__name__)


class PivotConfirmationClassifier(nn.Module):
    """후보 피봇 확정/취소 분류 모델.
    
    입력: 후보 등록 시점의 피처 벡터 (ADAPT_KEYS 기준 32차원)
    출력: 확정 확률 (0~1)
    
    Architecture:
        Input (32) → Linear(128) → ReLU → Dropout(0.2)
               → Linear(64) → ReLU → Dropout(0.2)
               → Linear(1) → Sigmoid
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, input_dim) or (input_dim,)
        
        Returns:
            (batch, 1) or (1,) 확정 확률
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.encoder(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """확정 확률 예측.
        
        Args:
            x: (batch, input_dim) or (input_dim,)
        
        Returns:
            (batch,) or () 확정 확률
        """
        with torch.no_grad():
            prob = self.forward(x)
            return prob.squeeze(-1)


class PivotProbabilityRegressor(nn.Module):
    """확정 확률 직접 예측 회귀 모델.
    
    분류 모델과 동일 구조, MSE 손실 사용.
    
    입력: 후보 등록 시점의 피처 벡터
    출력: 확정 확률 (0~1)
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 확률로 제한
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prob = self.forward(x)
            return prob.squeeze(-1)


class PivotLifespanPredictor(nn.Module):
    """후보 수명(봉수) 시계열 예측 모델.
    
    입력: 후보 등록 후 매 봉마다의 피처 변화 (seq_len, input_dim)
    출력: 예상 수명(봉수)
    
    Architecture:
        Input (seq_len, 32) → LSTM(64) → Linear(1)
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) or (seq_len, input_dim)
        
        Returns:
            (batch, 1) or (1,) 예상 수명(봉수)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 출력
        last_output = lstm_out[:, -1, :]
        
        # FC
        output = self.fc(last_output)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lifespan = self.forward(x)
            return lifespan.squeeze(-1)


class PivotEnsemble(nn.Module):
    """피봇 예측 앙상블 모델.
    
    분류 모델과 회귀 모델의 앙상블.
    """
    
    def __init__(
        self,
        classifier: PivotConfirmationClassifier,
        regressor: PivotProbabilityRegressor,
        classifier_weight: float = 0.5,
    ):
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.classifier_weight = classifier_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            (cls_prob, reg_prob, ensemble_prob)
        """
        cls_prob = self.classifier(x)
        reg_prob = self.regressor(x)
        ensemble_prob = self.classifier_weight * cls_prob + (1 - self.classifier_weight) * reg_prob
        
        return cls_prob, reg_prob, ensemble_prob
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            cls_prob, reg_prob, ensemble_prob = self.forward(x)
            return {
                "classification": cls_prob.squeeze(-1),
                "regression": reg_prob.squeeze(-1),
                "ensemble": ensemble_prob.squeeze(-1),
            }


def create_classifier(input_dim: int = 32, hidden_dim: int = 128) -> PivotConfirmationClassifier:
    """분류 모델 생성 헬퍼."""
    return PivotConfirmationClassifier(input_dim=input_dim, hidden_dim=hidden_dim)


def create_regressor(input_dim: int = 32, hidden_dim: int = 128) -> PivotProbabilityRegressor:
    """회귀 모델 생성 헬퍼."""
    return PivotProbabilityRegressor(input_dim=input_dim, hidden_dim=hidden_dim)


def create_lifespan_model(
    input_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
) -> PivotLifespanPredictor:
    """시계열 모델 생성 헬퍼."""
    return PivotLifespanPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


def create_ensemble(
    classifier: PivotConfirmationClassifier,
    regressor: PivotProbabilityRegressor,
    classifier_weight: float = 0.5,
) -> PivotEnsemble:
    """앙상블 모델 생성 헬퍼."""
    return PivotEnsemble(
        classifier=classifier,
        regressor=regressor,
        classifier_weight=classifier_weight,
    )
