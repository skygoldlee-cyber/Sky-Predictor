"""
시장 레짐별 모델 구축
레짐 분류 및 레짐별 전문 모델
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


class RegimeClassifier:
    """시장 레짐 분류 모델"""
    
    def __init__(self):
        self.model = None
        self.regime_labels = {0: 'neutral', 1: 'bull', -1: 'bear'}
        
    def train_regime_classifier(self, df: pd.DataFrame):
        """레짐 분류 모델 학습"""
        from sklearn.ensemble import RandomForestClassifier
        
        # 레짐 분류용 피처
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['regime'].values
        
        # 시간 기반 분할
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        train_mask = (df['year'] >= 2019) & (df['year'] <= 2023)
        test_mask = (df['year'] >= 2024)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 레짐 분류 모델 학습
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 성과 평가
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"레짐 분류 모델 학습 완료")
        print(f"테스트 정확도: {accuracy:.4f}")
        
        # 모델 저장
        joblib.dump(self.model, MODELS_DIR / "regime_classifier.pkl")
        
        return accuracy
    
    def predict_regime(self, X: np.ndarray) -> np.ndarray:
        """레짐 예측"""
        if self.model is None:
            self.model = joblib.load(MODELS_DIR / "regime_classifier.pkl")
        
        # 마지막 컬럼(regime) 제외
        X_without_regime = X[:, :-1]
        return self.model.predict(X_without_regime)


class RegimeSpecificModels:
    """레짐별 전문 모델"""
    
    def __init__(self):
        self.regime_models = {}
        self.regime_classifier = RegimeClassifier()
        
    def train_regime_models(self, df: pd.DataFrame):
        """레짐별 모델 학습"""
        print("\n레짐별 모델 학습 시작")
        
        # 레짐 분류 모델 학습
        self.regime_classifier.train_regime_classifier(df)
        
        # 각 레짐별 데이터 분리
        for regime in df['regime'].unique():
            regime_data = df[df['regime'] == regime].copy()
            
            if len(regime_data) < 50:
                print(f"레짐 {regime}: 데이터 부족 ({len(regime_data)}건) - 건너뜀")
                continue
            
            print(f"\n레짐 {regime} 모델 학습 ({len(regime_data)}건)")
            
            # 레짐별 XGBoost 모델 학습
            model = self._train_xgboost_regime(regime_data)
            self.regime_models[f'xgboost_regime_{regime}'] = model
            
            # 모델 저장
            joblib.dump(model, MODELS_DIR / f"xgboost_regime_{regime}.pkl")
    
    def _train_xgboost_regime(self, df: pd.DataFrame):
        """레짐별 XGBoost 모델 학습"""
        import xgboost as xgb
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['is_win'].values
        
        # 시간 기반 분할
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        train_mask = (df['year'] >= 2019) & (df['year'] <= 2023)
        val_mask = (df['year'] == 2024)
        test_mask = (df['year'] >= 2025)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_train) < 30 or len(X_test) < 10:
            print(f"  데이터 부족 - 건너뜀")
            return None
        
        # XGBoost 모델 학습
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            reg_alpha=0.5,
            reg_lambda=2.0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # 성과 평가
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"  테스트 정확도: {accuracy:.4f}")
        
        return model
    
    def predict_with_regime(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """레짐 예측 및 레짐별 모델 예측"""
        # 레짐 예측
        regime_pred = self.regime_classifier.predict_regime(X)
        
        # 레짐별 모델 예측
        predictions = []
        
        for i, regime in enumerate(regime_pred):
            model_key = f'xgboost_regime_{regime}'
            
            if model_key in self.regime_models:
                model = self.regime_models[model_key]
                pred = model.predict_proba(X[i:i+1])[:, 1][0]
            else:
                # 레짐별 모델이 없으면 기본 모델 사용
                if 'xgboost_regime_0' in self.regime_models:
                    model = self.regime_models['xgboost_regime_0']
                    pred = model.predict_proba(X[i:i+1])[:, 1][0]
                else:
                    pred = 0.5  # 기본값
            
            predictions.append(pred)
        
        return np.array(regime_pred), np.array(predictions)


def evaluate_regime_models(regime_models: RegimeSpecificModels, df: pd.DataFrame):
    """레짐별 모델 성과 평가"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # 테스트 데이터 분할 (2025-2026)
    test_mask = (pd.to_datetime(df['entry_time']).dt.year >= 2025)
    test_df = df[test_mask].copy()
    
    if len(test_df) == 0:
        print("테스트 데이터 부족")
        return
    
    # 피처 준비
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['is_win'].values
    
    # 레짐별 예측
    regime_pred, y_pred_proba = regime_models.predict_with_regime(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 성과 평가
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
    }
    
    print(f"\n레짐별 모델 성과 (2025-2026):")
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  정밀도: {result['precision']:.4f}")
    print(f"  재현율: {result['recall']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    print(f"  ROC AUC: {result['roc_auc']:.4f}")
    
    # 레짐 분포
    print(f"\n레짐 분포:")
    unique, counts = np.unique(regime_pred, return_counts=True)
    for regime, count in zip(unique, counts):
        print(f"  레짐 {regime}: {count}건 ({count/len(regime_pred)*100:.1f}%)")
    
    return result


def main():
    """메인 함수"""
    print("=" * 80)
    print("시장 레짐별 모델 구축")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 레짐 분포
    print(f"\n레짐 분포:")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"  레짐 {regime}: {count}건 ({count/len(df)*100:.1f}%)")
    
    # 레짐별 모델 학습
    regime_models = RegimeSpecificModels()
    regime_models.train_regime_models(df)
    
    # 성과 평가
    result = evaluate_regime_models(regime_models, df)
    
    print(f"\n레짐별 모델 구축 완료")


if __name__ == "__main__":
    main()
