"""Pivot Prediction Models

피봇 후보 확정/취소 예측 모델들.

Models:
- PivotConfirmationClassifier: 이진 분류 (확정/취소)
- PivotProbabilityRegressor: 회귀 (확정 확률)
- PivotLifespanPredictor: 시계열 (후보 수명)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
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
    """후보 잔여 수명(봉수) 시계열 예측 모델.

    입력: 후보 등록 후 '지금까지' 관측된 부분 시퀀스 (seq_len, input_dim)
    출력: 지금 시점부터 확정/취소까지의 '잔여' 봉수 (log1p 스케일)

    Architecture:
        Input (seq_len, input_dim) → LSTM(hidden) → Linear(1)

    Note:
        학습 타깃을 '전체 수명'이 아닌 '잔여 수명'으로 정의해야 추론(진행 중
        부분 시퀀스)과 분포가 일치하고, 시퀀스 길이가 곧 정답이 되는 누수를
        피한다. 패딩이 있는 배치는 forward(x, lengths) 로 호출할 것.
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
    
    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) or (seq_len, input_dim)
            lengths: (batch,) 각 시퀀스의 실제 유효 길이.
                패딩이 포함된 입력이면 반드시 전달해야 한다. 전달하면
                pack_padded_sequence 로 패딩을 건너뛰고 '실제 마지막 스텝'의
                은닉 상태를 readout 으로 사용한다. None이면 패딩이 없다고 보고
                마지막 타임스텝을 그대로 사용한다.

        Returns:
            (batch, 1) or (1,) 예상 잔여 수명(봉수, log1p 스케일)

        Note:
            후방(post) 패딩 + 마지막 타임스텝 readout 조합은 패딩 0벡터에서
            출력을 읽어 예측을 망친다. lengths 를 넘겨 packing 을 사용하면
            이 문제를 제거한다.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.as_tensor(lengths, dtype=torch.long)
            # pack 은 길이를 CPU long 텐서로 요구. 최소 1 보장.
            lengths_cpu = lengths.detach().to("cpu").long().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # h_n[-1] = 최종 LSTM 레이어의 마지막(실제) 스텝 은닉 상태 (batch, hidden)
        last_output = h_n[-1]
        output = self.fc(last_output)
        return output

    def predict(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        with torch.no_grad():
            lifespan = self.forward(x, lengths)
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
