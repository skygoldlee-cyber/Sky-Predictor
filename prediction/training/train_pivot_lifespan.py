"""Pivot Lifespan Predictor Training Script

후보 수명(봉수) 시계열 예측 모델 학습 스크립트.

Usage:
    python prediction/train_pivot_lifespan.py \
        --data_path data/pivot_candidates.pkl \
        --output_dir prediction/weights \
        --epochs 50 \
        --batch_size 16 \
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

from pivot_models import PivotLifespanPredictor
from features import ADAPT_KEYS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class PivotSequenceDataset(Dataset):
    """피봇 후보 시계열 데이터셋.

    [누수 수정] 각 완결 후보에 대해 '전체 시퀀스 → 전체 수명' 한 쌍이 아니라,
    여러 prefix 길이에서 '부분 시퀀스 → 잔여 수명' 쌍을 생성한다.

    - 입력  : real_snaps[:t]  (등록~현재까지 관측된 부분 시퀀스)
    - 타깃  : terminal_bar - real_snaps[t-1].bar_idx  (잔여 봉수)

    이렇게 하면 (1) 시퀀스 길이가 곧 정답이 되는 누수가 사라지고,
    (2) 추론 시 진행 중 후보(부분 시퀀스)와 분포가 일치한다.
    확정/취소 시점의 빈 features({}) 스냅샷은 prefix 에서 제외한다.
    """

    def __init__(
        self,
        records: List[Dict],
        feature_keys: List[str],
        max_seq_len: int = 120,
        min_prefix: int = 3,
    ):
        self.records = records
        self.feature_keys = feature_keys
        self.max_seq_len = max_seq_len
        self.min_prefix = max(1, int(min_prefix))
        self.X, self.y, self.seq_lengths = self._prepare_data()

    def _vec(self, features: Dict) -> List[float]:
        """피처 딕셔너리 → ADAPT_KEYS 순서 벡터."""
        vec = []
        for key in self.feature_keys:
            val = features.get(key, 0.0)
            if isinstance(val, (int, float, np.number)):
                vec.append(float(val))
            else:
                vec.append(0.0)
        return vec

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = []
        y = []
        seq_lengths = []

        for record in self.records:
            sequence = record.get("sequence", [])
            if not sequence:
                continue

            # 등록 봉 / 종료 봉 계산
            registered_bar = record.get("registered_bar")
            lifespan_bars = record.get("lifespan_bars")
            if registered_bar is None or lifespan_bars is None:
                continue
            terminal_bar = int(registered_bar) + int(lifespan_bars)

            # 실 스냅샷만 사용 (features 가 비어있는 종료 스냅샷 제외)
            real_snaps = [
                s for s in sequence
                if isinstance(s.get("features"), dict) and len(s["features"]) > 0
                and s.get("bar_idx") is not None
            ]
            if len(real_snaps) < self.min_prefix:
                continue

            # prefix 길이 t = min_prefix .. len(real_snaps)
            for t in range(self.min_prefix, len(real_snaps) + 1):
                prefix = real_snaps[:t]
                last_bar = int(prefix[-1]["bar_idx"])
                remaining = terminal_bar - last_bar
                if remaining < 0:
                    continue  # 데이터 이상치 방어

                # 부분 시퀀스 → 피처 벡터 (시간 순서 유지)
                seq_features = [self._vec(s["features"]) for s in prefix]

                # 트리밍 (최근 max_seq_len) + 후방 패딩 (packing 으로 무시됨)
                if len(seq_features) > self.max_seq_len:
                    seq_features = seq_features[-self.max_seq_len:]
                seq_len = len(seq_features)
                while len(seq_features) < self.max_seq_len:
                    seq_features.append([0.0] * len(self.feature_keys))

                X.append(seq_features)
                y.append(float(remaining))
                seq_lengths.append(seq_len)

        if not X:
            # 빈 데이터셋 방어
            return (
                np.zeros((0, self.max_seq_len, len(self.feature_keys)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        seq_lengths = np.array(seq_lengths, dtype=np.int64)

        # NaN/Inf 처리
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # 잔여 수명 정규화 (로그 스케일)
        y = np.log1p(y)  # log(1 + x)

        _logger.info(
            "[LifespanDataset] 완결 후보 %d건 → 학습 샘플 %d개 (prefix 확장)",
            len(self.records), len(X)
        )
        return X, y, seq_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(int(self.seq_lengths[idx])),
        )


def load_dataset(data_path: str) -> Tuple[List[Dict], Dict]:
    """데이터셋 로드."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    records = data["completed_candidates"]
    statistics = data["statistics"]
    
    _logger.info(f"데이터셋 로드 완료: {len(records)} 건")
    _logger.info(f"통계: {statistics}")
    
    return records, statistics


def split_records_time_ordered(
    records: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """완결 후보를 '레코드 단위'로 시간순 분할한다.

    [누수 수정] prefix 확장(한 후보 → 여러 샘플) 이후에는 무작위 분할 시
    같은 후보의 prefix 들이 train/val/test 에 흩어져 누수가 발생한다.
    따라서 (date, registered_bar) 기준으로 정렬한 뒤 레코드 경계에서 분할해,
    한 후보의 모든 prefix 가 같은 분할에만 속하도록 한다. 동시에 과거→미래
    순서를 유지해 시계열 평가 원칙(과거 학습/미래 검증)을 지킨다.
    """
    def _key(r: Dict):
        return (str(r.get("date", "")), int(r.get("registered_bar", 0)))

    ordered = sorted(records, key=_key)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_recs = ordered[:n_train]
    val_recs = ordered[n_train:n_train + n_val]
    test_recs = ordered[n_train + n_val:]

    _logger.info(
        "[Split] 레코드 시간순 분할: Train=%d Val=%d Test=%d (총 %d)",
        len(train_recs), len(val_recs), len(test_recs), n
    )
    return train_recs, val_recs, test_recs


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
    
    for X, y, seq_lens in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X, seq_lens).squeeze(-1)
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
        for X, y, seq_lens in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X, seq_lens).squeeze(-1)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * X.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # 역정규화 (exp - 1)
    all_preds_denorm = np.expm1(all_preds)
    all_labels_denorm = np.expm1(all_labels)
    
    # 회귀 메트릭
    mse = mean_squared_error(all_labels_denorm, all_preds_denorm)
    mae = mean_absolute_error(all_labels_denorm, all_preds_denorm)
    r2 = r2_score(all_labels_denorm, all_preds_denorm)
    
    # MAE (봉수 기준)
    mae_bars = mae
    
    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "mae_bars": mae_bars,
        "r2": r2,
    }


def train_lifespan_model(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    hidden_dim: int = 64,
    num_layers: int = 2,
    max_seq_len: int = 120,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """시계열 모델 학습."""
    _logger.info(f"학습 시작: device={device}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 로드
    records, statistics = load_dataset(data_path)

    if len(records) < 30:
        _logger.warning(f"데이터가 부족합니다: {len(records)} 건 (최소 30건 권장)")

    # [누수 수정] 레코드 단위 시간순 분할 → 분할별로 prefix 샘플 생성
    train_recs, val_recs, test_recs = split_records_time_ordered(records)

    train_ds = PivotSequenceDataset(train_recs, ADAPT_KEYS, max_seq_len=max_seq_len)
    val_ds   = PivotSequenceDataset(val_recs,   ADAPT_KEYS, max_seq_len=max_seq_len)
    test_ds  = PivotSequenceDataset(test_recs,  ADAPT_KEYS, max_seq_len=max_seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        _logger.error("학습/검증 샘플이 비었습니다. 데이터 수집을 더 진행하세요.")
        return None, {}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    # 모델 생성
    input_dim = len(ADAPT_KEYS)
    model = PivotLifespanPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
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
            f"Val MAE (bars): {val_metrics['mae_bars']:.2f} - "
            f"Val R2: {val_metrics['r2']:.4f}"
        )
        
        # 체크포인트 저장
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            
            checkpoint_path = output_path / "pivot_lifespan_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'val_metrics': val_metrics,
                # 추론 로딩 견고화: best 체크포인트에도 메타 저장
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'max_seq_len': max_seq_len,
                # 타깃 의미: 잔여 수명(remaining), log1p 스케일
                'target': 'remaining_lifespan_bars_log1p',
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
    final_path = output_path / "pivot_lifespan_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len,
        'target': 'remaining_lifespan_bars_log1p',
        'test_metrics': test_metrics,
        'statistics': statistics,
    }, final_path)
    _logger.info(f"최종 모델 저장: {final_path}")
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Pivot Lifespan Predictor Training")
    parser.add_argument("--data_path", type=str, required=True, help="데이터셋 경로 (.pkl)")
    parser.add_argument("--output_dir", type=str, default="prediction/weights", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=50, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--hidden_dim", type=int, default=64, help="히든 차원")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM 레이어 수")
    parser.add_argument("--max_seq_len", type=int, default=120, help="최대 시퀀스 길이")
    parser.add_argument("--device", type=str, default="cuda", help="장치 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 장치 설정
    if args.device == "cuda" and not torch.cuda.is_available():
        _logger.warning("CUDA 사용 불가, CPU로 전환")
        args.device = "cpu"
    
    train_lifespan_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
