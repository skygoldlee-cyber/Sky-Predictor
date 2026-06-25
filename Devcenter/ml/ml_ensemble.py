"""
모델 앙상블 구현
XGBoost, Random Forest, LSTM 예측 결과 결합
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


class ModelEnsemble:
    """모델 앙상블 클래스"""
    
    def __init__(self, ensemble_method='weighted_average'):
        self.ensemble_method = ensemble_method
        self.models = {}
        self.weights = {
            'xgboost': 0.4,
            'random_forest': 0.4,
            'lstm': 0.2
        }
        
    def load_models(self):
        """학습된 모델 로드"""
        # XGBoost 모델 로드
        xgb_path = MODELS_DIR / "trade_filter_xgboost.pkl"
        if xgb_path.exists():
            self.models['xgboost'] = joblib.load(xgb_path)
            print("XGBoost 모델 로드 완료")
        
        # Random Forest 모델 로드
        rf_path = MODELS_DIR / "entry_timing_rf.pkl"
        if rf_path.exists():
            self.models['random_forest'] = joblib.load(rf_path)
            print("Random Forest 모델 로드 완료")
        
        # LSTM 모델 로드
        lstm_path = MODELS_DIR / "exit_timing_lstm.keras"
        if lstm_path.exists():
            from tensorflow.keras.models import load_model
            self.models['lstm'] = load_model(str(lstm_path))
            print("LSTM 모델 로드 완료")
    
    def predict_ensemble(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """앙상블 예측"""
        predictions = {}
        
        # 각 모델 예측
        if 'xgboost' in self.models and 'xgboost' in X_dict:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X_dict['xgboost'])[:, 1]
        
        if 'random_forest' in self.models and 'random_forest' in X_dict:
            predictions['random_forest'] = self.models['random_forest'].predict_proba(X_dict['random_forest'])[:, 1]
        
        if 'lstm' in self.models and 'lstm' in X_dict:
            predictions['lstm'] = self.models['lstm'].predict(X_dict['lstm'], verbose=0).flatten()
        
        # 앙상블 방법별 결합
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average(predictions)
        elif self.ensemble_method == 'majority_vote':
            return self._majority_vote(predictions)
        elif self.ensemble_method == 'stacking':
            return self._stacking(predictions)
        else:
            return self._weighted_average(predictions)
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 평균 앙상블"""
        if not predictions:
            return np.array([])
        
        # 모든 예측의 길이를 최소 길이로 맞춤
        min_length = min(len(pred) for pred in predictions.values())
        aligned_predictions = {name: pred[:min_length] for name, pred in predictions.items()}
        
        ensemble_pred = np.zeros(min_length)
        total_weight = 0
        
        for model_name, pred in aligned_predictions.items():
            weight = self.weights.get(model_name, 1.0)
            ensemble_pred += pred * weight
            total_weight += weight
        
        return ensemble_pred / total_weight if total_weight > 0 else ensemble_pred
    
    def _majority_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """다수결 앙상블"""
        if not predictions:
            return np.array([])
        
        # 이진 변환
        binary_preds = [np.where(pred > 0.5, 1, 0) for pred in predictions.values()]
        binary_preds = np.array(binary_preds)
        
        # 다수결
        majority = np.mean(binary_preds, axis=0)
        return majority
    
    def _stacking(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """스태킹 앙상블 (간단 버전)"""
        if not predictions:
            return np.array([])
        
        # 예측 결과를 피처로 사용
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # 간단한 메타 모델 (가중 평균)
        weights = np.array([self.weights.get(name, 1.0) for name in predictions.keys()])
        weights = weights / weights.sum()
        
        return np.dot(pred_matrix, weights)
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """성과 기반 가중치 업데이트"""
        # 성과 기반 가중치 조정
        total_perf = sum(performance_metrics.values())
        if total_perf > 0:
            for model_name in self.weights:
                if model_name in performance_metrics:
                    self.weights[model_name] = performance_metrics[model_name] / total_perf
        
        print(f"가중치 업데이트: {self.weights}")


def prepare_ensemble_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """앙상블용 데이터 준비"""
    X_dict = {}
    
    # XGBoost 피처
    xgb_features = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    if all(col in df.columns for col in xgb_features):
        X_dict['xgboost'] = df[xgb_features].fillna(0).values
    
    # Random Forest 피처 (29개 피처)
    rf_base_features = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 추가 피처 계산
    df_temp = df.copy()
    
    # 1. RSI 과매수/과매도 상태
    df_temp['rsi_oversold'] = (df_temp['entry_rsi'] < 30).astype(int)
    df_temp['rsi_overbought'] = (df_temp['entry_rsi'] > 70).astype(int)
    
    # 2. MACD 신호 강도
    df_temp['macd_bullish'] = (df_temp['entry_macd'] > df_temp['entry_macd_signal']).astype(int)
    df_temp['macd_strength'] = abs(df_temp['entry_macd'] - df_temp['entry_macd_signal'])
    
    # 3. 가격과 이동평균선 관계
    df_temp['price_above_ma20'] = (df_temp['entry_close'] > df_temp['entry_ma20']).astype(int)
    df_temp['price_above_ma60'] = (df_temp['entry_close'] > df_temp['entry_ma60']).astype(int)
    
    # 4. Bollinger Bands 위치
    df_temp['bb_position'] = (df_temp['entry_close'] - df_temp['entry_bb_lower']) / (df_temp['entry_bb_upper'] - df_temp['entry_bb_lower'])
    df_temp['bb_lower_touch'] = (df_temp['entry_close'] <= df_temp['entry_bb_lower'] * 1.01).astype(int)
    
    # 5. SuperTrend 방향과 가격 관계
    df_temp['price_above_st'] = (df_temp['entry_close'] > df_temp['entry_supertrend']).astype(int)
    
    # 6. 시간대 특성
    df_temp['is_morning'] = ((df_temp['entry_hour'] >= 9) & (df_temp['entry_hour'] < 12)).astype(int)
    df_temp['is_afternoon'] = ((df_temp['entry_hour'] >= 12) & (df_temp['entry_hour'] < 15)).astype(int)
    
    # 7. 레짐 특성
    df_temp['is_bull'] = (df_temp['regime'] == 1).astype(int)
    df_temp['is_neutral'] = (df_temp['regime'] == 0).astype(int)
    
    # 최종 피처 리스트
    rf_features = rf_base_features + [
        'rsi_oversold', 'rsi_overbought',
        'macd_bullish', 'macd_strength',
        'price_above_ma20', 'price_above_ma60',
        'bb_position', 'bb_lower_touch',
        'price_above_st',
        'is_morning', 'is_afternoon',
        'is_bull', 'is_neutral'
    ]
    
    if all(col in df_temp.columns for col in rf_features):
        X_dict['random_forest'] = df_temp[rf_features].fillna(0).values
    
    # LSTM 피처 (시퀀스 데이터)
    lstm_features = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    if all(col in df.columns for col in lstm_features):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        df_sorted = df.sort_values('entry_time').reset_index(drop=True)
        X = df_sorted[lstm_features].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = scaler.fit_transform(X)
        
        # 시퀀스 생성
        sequence_length = 10
        X_sequences = []
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i+sequence_length])
        
        if X_sequences:
            X_dict['lstm'] = np.array(X_sequences)
    
    return X_dict


def evaluate_ensemble(ensemble: ModelEnsemble, X_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict:
    """앙상블 성과 평가"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # 앙상블 예측
    y_pred_proba = ensemble.predict_ensemble(X_dict)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 성과 평가
    result = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba) if len(y) > 1 else 0
    }
    
    return result


def main():
    """메인 함수"""
    print("=" * 80)
    print("모델 앙상블 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    
    # 앙상블 데이터 준비
    X_dict = prepare_ensemble_data(df)
    print(f"\n앙상블 데이터 준비 완료")
    for model_name, X in X_dict.items():
        print(f"  {model_name}: {X.shape}")
    
    # 앙상블 모델 초기화
    ensemble = ModelEnsemble(ensemble_method='weighted_average')
    
    # 모델 로드
    ensemble.load_models()
    
    # 테스트 데이터 분할 (2025-2026)
    test_mask = (df['entry_time'].dt.year >= 2025)
    test_df = df[test_mask].copy()
    
    if len(test_df) > 0:
        # 테스트 데이터 준비
        X_test_dict = prepare_ensemble_data(test_df)
        y_test = test_df['is_win'].values
        
        # LSTM 시퀀스 길이 조정
        if 'lstm' in X_test_dict:
            lstm_len = len(X_test_dict['lstm'])
            y_test = y_test[-lstm_len:]
        
        # 앙상블 평가
        result = evaluate_ensemble(ensemble, X_test_dict, y_test)
        
        print(f"\n앙상블 성과 (2025-2026):")
        print(f"  정확도: {result['accuracy']:.4f}")
        print(f"  정밀도: {result['precision']:.4f}")
        print(f"  재현율: {result['recall']:.4f}")
        print(f"  F1 점수: {result['f1']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        
        # 가중치 업데이트 (성과 기반)
        performance_metrics = {
            'xgboost': 0.55,
            'random_forest': 0.53,
            'lstm': 0.47
        }
        ensemble.update_weights(performance_metrics)
    else:
        print("테스트 데이터 부족")


if __name__ == "__main__":
    main()
