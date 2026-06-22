"""Pivot Probability Regressor Training Script

후보 피봇 확정 확률 회귀 모델 학습 스크립트.

Usage:
    python prediction/train_pivot_regressor.py \
        --data_path data/pivot_candidates.pkl \
        --output_dir prediction/weights \
        --epochs 50 \
        --batch_size 32 \
        --lr 0.001
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pivot_models import PivotProbabilityRegressor
from features import ADAPT_KEYS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class PivotDataset(Dataset):
    """피봇 후보 데이터셋 (회귀용)."""
    
    def __init__(self, records: List[Dict], feature_keys: List[str]):
        self.records = records
        self.feature_keys = feature_keys
        self.X, self.y = self._prepare_data()
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        
        for record in self.records:
            # 등록 시점 피처
            features = record["registered_features"]
            
            # 피처 벡터 생성 (ADAPT_KEYS 순서)
            feature_vector = []
            for key in self.feature_keys:
                val = features.get(key, 0.0)
                if isinstance(val, (int, float, np.number)):
                    feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            
            X.append(feature_vector)
            y.append(record["label"])  # 1=확정, 0=취소 (확률로 사용)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # NaN/Inf 처리
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return X, y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def load_dataset(data_path: str) -> Tuple[List[Dict], Dict]:
    """데이터셋 로드."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    records = data["completed_candidates"]
    statistics = data["statistics"]
    
    _logger.info(f"데이터셋 로드 완료: {len(records)} 건")
    _logger.info(f"통계: {statistics}")
    
    return records, statistics


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터셋 분할."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    _logger.info(f"데이터 분할: Train={train_size}, Val={val_size}, Test={test_size}")
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """한 epoch 학습."""
    model.train()
    total_loss = 0.0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X).squeeze(-1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """평가."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(-1)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * X.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # 회귀 메트릭
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    # 분류 메트릭 (임계값 0.5)
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    accuracy = sum(1 for p, l in zip(binary_preds, all_labels) if p == l) / len(all_labels)
    
    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy,
    }


def train_regressor(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden_dim: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """회귀 모델 학습."""
    _logger.info(f"학습 시작: device={device}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 로드
    records, statistics = load_dataset(data_path)
    
    if len(records) < 100:
        _logger.warning(f"데이터가 부족합니다: {len(records)} 건 (최소 100건 권장)")
    
    # 데이터셋 생성
    dataset = PivotDataset(records, ADAPT_KEYS)
    
    # 데이터 분할
    train_loader, val_loader, test_loader = split_dataset(dataset)
    
    # 모델 생성
    input_dim = len(ADAPT_KEYS)
    model = PivotProbabilityRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 학습 루프
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 검증
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 스케줄러
        scheduler.step(val_metrics["loss"])
        
        # 로그
        _logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val MSE: {val_metrics['mse']:.4f} - "
            f"Val R2: {val_metrics['r2']:.4f}"
        )
        
        # 체크포인트 저장
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            
            checkpoint_path = output_path / "pivot_regressor_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'val_metrics': val_metrics,
            }, checkpoint_path)
            _logger.info(f"체크포인트 저장: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # 테스트 평가
    _logger.info("테스트 평가...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    _logger.info(f"테스트 결과: {test_metrics}")
    
    # 최종 모델 저장
    final_path = output_path / "pivot_regressor_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'test_metrics': test_metrics,
        'statistics': statistics,
    }, final_path)
    _logger.info(f"최종 모델 저장: {final_path}")
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Pivot Probability Regressor Training")
    parser.add_argument("--data_path", type=str, required=True, help="데이터셋 경로 (.pkl)")
    parser.add_argument("--output_dir", type=str, default="prediction/weights", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=50, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--hidden_dim", type=int, default=128, help="히든 차원")
    parser.add_argument("--device", type=str, default="cuda", help="장치 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 장치 설정
    if args.device == "cuda" and not torch.cuda.is_available():
        _logger.warning("CUDA 사용 불가, CPU로 전환")
        args.device = "cpu"
    
    train_regressor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )


if __name__ == "__main__":
    main()
