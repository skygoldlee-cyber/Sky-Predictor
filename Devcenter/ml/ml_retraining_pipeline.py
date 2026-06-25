"""
정기적 재학습 파이프라인 (Walk-Forward Validation 기반)
시장 레짐 변화 감지 및 모델 업데이트 시스템
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import joblib
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import f

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import ml_trade_filter
import ml_entry_timing
import ml_exit_timing

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"
MODELS_DIR.mkdir(exist_ok=True)


class RetrainingPipeline:
    """정기적 재학습 파이프라인"""
    
    def __init__(self, retraining_interval='3M', train_window_years=5):
        self.retraining_interval = retraining_interval
        self.train_window_years = train_window_years
        self.models = {}
        self.last_retrain_date = None
        self.performance_history = []
        
    def should_retrain(self, current_date: datetime, performance_metrics: Dict) -> Tuple[bool, str]:
        """재학습 필요 여부 판단"""
        # 1. 정기 재학습 주기 확인
        if self.last_retrain_date is None:
            return True, "최초 재학습"
        
        time_elapsed = current_date - self.last_retrain_date
        if time_elapsed >= pd.Timedelta(self.retraining_interval):
            return True, f"정기 재학습 주기 도달 ({time_elapsed.days}일 경과)"
        
        # 2. 성능 저하 확인
        if performance_metrics.get('win_rate', 1.0) < 0.5:
            return True, f"승률 저하: {performance_metrics['win_rate']:.2%}"
        
        # 3. 최근 성과 추이 확인
        if len(self.performance_history) >= 2:
            recent_win_rates = [p['win_rate'] for p in self.performance_history[-2:]]
            if all(wr < 0.5 for wr in recent_win_rates):
                return True, f"승률 2분기 연속 저하: {recent_win_rates}"
        
        return False, "재학습 불필요"
    
    def walk_forward_retrain(self, df: pd.DataFrame, current_date: datetime) -> str:
        """Walk-Forward 방식 재학습"""
        # 훈련 윈도우 설정 (최근 train_window_years)
        train_start = current_date - pd.DateOffset(years=self.train_window_years)
        train_end = current_date
        
        train_data = df[(df['entry_time'] >= train_start) & 
                        (df['entry_time'] < train_end)].copy()
        
        if len(train_data) < 100:
            raise ValueError(f"훈련 데이터 부족: {len(train_data)}건 (최소 100건 필요)")
        
        print(f"\n{'='*80}")
        print(f"Walk-Forward 재학습 시작")
        print(f"{'='*80}")
        print(f"훈련 기간: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
        print(f"훈련 데이터: {len(train_data)}건")
        print(f"승률: {train_data['is_win'].mean() * 100:.2f}%")
        
        # 모델 재학습
        self.retrain_all_models(train_data)
        
        # 모델 버전 관리
        model_version = f"v_{current_date.strftime('%Y%m%d')}"
        self.save_models(model_version)
        
        self.last_retrain_date = current_date
        
        print(f"\n재학습 완료: {model_version}")
        print(f"{'='*80}\n")
        
        return model_version
    
    def retrain_all_models(self, train_data: pd.DataFrame):
        """모든 모델 재학습"""
        # 1. XGBoost 필터링 모델
        print("\n[1/3] XGBoost 필터링 모델 재학습...")
        self.models['trade_filter'] = self.train_xgboost(train_data)
        
        # 2. Random Forest 진입 타이밍 모델
        print("[2/3] Random Forest 진입 타이밍 모델 재학습...")
        self.models['entry_timing'] = self.train_random_forest(train_data)
        
        # 3. LSTM 청산 타이밍 모델
        print("[3/3] LSTM 청산 타이밍 모델 재학습...")
        self.models['exit_timing'] = self.train_lstm(train_data)
    
    def train_xgboost(self, train_data: pd.DataFrame):
        """XGBoost 모델 학습"""
        # 기존 ml_trade_filter 로직 활용
        df = train_data.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        timestamps = df['entry_time']
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].copy().fillna(0).astype(float)
        y = df['is_win'].copy()
        
        # 시간 기반 분할
        train_mask = (timestamps.dt.year >= timestamps.dt.year.min()) & (timestamps.dt.year <= timestamps.dt.year.max() - 1)
        val_mask = (timestamps.dt.year == timestamps.dt.year.max())
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return model
    
    def train_random_forest(self, train_data: pd.DataFrame):
        """Random Forest 모델 학습"""
        # 기존 ml_entry_timing 로직 활용
        df = train_data.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        timestamps = df['entry_time']
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime',
            'rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'
        ]
        
        # 추가 피처 계산
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
        
        X = df[feature_cols].copy().fillna(0).astype(float)
        y = df['is_win'].copy()
        
        # 시간 기반 분할
        train_mask = (timestamps.dt.year >= timestamps.dt.year.min()) & (timestamps.dt.year <= timestamps.dt.year.max() - 1)
        val_mask = (timestamps.dt.year == timestamps.dt.year.max())
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def train_lstm(self, train_data: pd.DataFrame):
        """LSTM 모델 학습"""
        # 기존 ml_exit_timing 로직 활용
        df = train_data.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        timestamps = df['entry_time']
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # 시계열 데이터 준비
        df_sorted = df.sort_values('entry_time').reset_index(drop=True)
        X = df_sorted[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        # 훈련 데이터에만 fit
        train_mask = (timestamps.dt.year >= timestamps.dt.year.min()) & (timestamps.dt.year <= timestamps.dt.year.max() - 1)
        X_scaled = scaler.fit_transform(X[train_mask])
        X_scaled = scaler.transform(X)
        
        # 시퀀스 생성
        sequence_length = 10
        X_sequences = []
        y_sequences = []
        year_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i+sequence_length])
            y_sequences.append(df_sorted['is_win'].iloc[i+sequence_length])
            year_sequences.append(timestamps.iloc[i+sequence_length].year)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        year_sequences = np.array(year_sequences)
        
        # 시간 기반 분할
        train_mask_seq = (year_sequences >= year_sequences.min()) & (year_sequences <= year_sequences.max() - 1)
        val_mask_seq = (year_sequences == year_sequences.max())
        
        X_train, y_train = X_sequences[train_mask_seq], y_sequences[train_mask_seq]
        X_val, y_val = X_sequences[val_mask_seq], y_sequences[val_mask_seq]
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(16, return_sequences=False,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                 validation_data=(X_val, y_val), verbose=0)
        
        return model
    
    def save_models(self, version: str):
        """모델 저장"""
        for model_name, model in self.models.items():
            if model_name == 'exit_timing':
                path = MODELS_DIR / f"{model_name}_{version}.keras"
                model.save(str(path))
            else:
                path = MODELS_DIR / f"{model_name}_{version}.pkl"
                joblib.dump(model, path)
            print(f"  모델 저장 완료: {path}")


class RegimeChangeDetector:
    """시장 레짐 변화 감지 시스템"""
    
    def __init__(self, window_size=30, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.indicators = {}
        
    def detect_regime_change(self, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> List[str]:
        """레짐 변화 감지"""
        change_signals = []
        
        # 1. 변동성 급격 변화
        if self.detect_volatility_shift(historical_data, current_data):
            change_signals.append('변동성 급격 변화')
        
        # 2. 추세 전환
        if self.detect_trend_reversal(historical_data, current_data):
            change_signals.append('추세 전환')
        
        # 3. 통계적 검정
        if self.run_statistical_tests(historical_data, current_data):
            change_signals.append('통계적 구조 변화')
        
        return change_signals
    
    def detect_volatility_shift(self, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
        """변동성 급격 변화 감지"""
        if 'entry_atr' not in historical_data.columns or 'entry_atr' not in current_data.columns:
            return False
            
        hist_vol = historical_data['entry_atr'].mean()
        curr_vol = current_data['entry_atr'].mean()
        
        if curr_vol > hist_vol * self.threshold:
            return True
        return False
    
    def detect_trend_reversal(self, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
        """추세 전환 감지"""
        if 'entry_close' not in historical_data.columns or 'entry_close' not in current_data.columns:
            return False
            
        hist_slope = self.calculate_trend_slope(historical_data['entry_close'])
        curr_slope = self.calculate_trend_slope(current_data['entry_close'])
        
        if hist_slope * curr_slope < 0:
            return True
        return False
    
    def calculate_trend_slope(self, prices: pd.Series) -> float:
        """추세 기울기 계산"""
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope
    
    def run_statistical_tests(self, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
        """통계적 검정 실행 (Chow Test)"""
        if 'entry_close' not in historical_data.columns or 'entry_close' not in current_data.columns:
            return False
            
        combined_data = pd.concat([historical_data, current_data])
        breakpoint = len(historical_data)
        
        y = combined_data['entry_close'].values
        X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
        
        n = len(y)
        k = X.shape[1]
        
        # 전체 기간 회귀
        beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
        rss_full = np.sum((y - X @ beta_full) ** 2)
        
        # 분할 기간 회귀
        X1 = X[:breakpoint]
        y1 = y[:breakpoint]
        X2 = X[breakpoint:]
        y2 = y[breakpoint:]
        
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        
        rss1 = np.sum((y1 - X1 @ beta1) ** 2)
        rss2 = np.sum((y2 - X2 @ beta2) ** 2)
        rss_pooled = rss1 + rss2
        
        # F-statistic 계산
        if n - 2 * k <= 0:
            return False
            
        F = ((rss_full - rss_pooled) / k) / (rss_pooled / (n - 2 * k))
        p_value = 1 - f.cdf(F, k, n - 2 * k)
        
        # 유의수준 0.05에서 기각 시 레짐 변화
        if p_value < 0.05:
            return True
        return False


class ModelVersionManager:
    """모델 버전 관리 및 롤백"""
    
    def __init__(self, max_versions=5):
        self.max_versions = max_versions
        self.model_history = {}
        
    def save_model_version(self, version: str, model, performance: Dict):
        """모델 버전 저장"""
        self.model_history[version] = {
            'model': model,
            'performance': performance,
            'timestamp': pd.Timestamp.now()
        }
        
        # 최대 버전 수 초과 시 가장 오래된 버전 삭제
        if len(self.model_history) > self.max_versions:
            oldest_version = min(self.model_history.keys())
            del self.model_history[oldest_version]
    
    def rollback_to_version(self, target_version: str):
        """특정 버전으로 롤백"""
        if target_version in self.model_history:
            return self.model_history[target_version]['model']
        else:
            raise ValueError(f"버전 {target_version}이 존재하지 않습니다")


def main():
    """메인 함수"""
    print("=" * 80)
    print("정기적 재학습 파이프라인 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 재학습 파이프라인 초기화
    pipeline = RetrainingPipeline(retraining_interval='3M', train_window_years=5)
    
    # 현재 날짜 설정 (데이터의 최신 날짜)
    current_date = df['entry_time'].max()
    
    # 재학습 필요 여부 확인
    performance_metrics = {'win_rate': df['is_win'].mean()}
    should_retrain, reason = pipeline.should_retrain(current_date, performance_metrics)
    
    print(f"\n재학습 필요 여부: {should_retrain}")
    print(f"이유: {reason}")
    
    if should_retrain:
        # Walk-Forward 재학습 실행
        model_version = pipeline.walk_forward_retrain(df, current_date)
        print(f"\n재학습 완료: {model_version}")
    else:
        print("\n재학습 불필요")
    
    # 레짐 변화 감지 테스트
    print("\n" + "=" * 80)
    print("시장 레짐 변화 감지 테스트")
    print("=" * 80)
    
    detector = RegimeChangeDetector(window_size=30, threshold=2.0)
    
    # 데이터 분할 (2024년 이전/이후)
    hist_data = df[df['entry_time'].dt.year < 2024]
    curr_data = df[df['entry_time'].dt.year >= 2024]
    
    change_signals = detector.detect_regime_change(hist_data, curr_data)
    
    print(f"\n감지된 레짐 변화 신호: {change_signals}")
    
    if change_signals:
        print("레짐 변화 감지됨 - 모델 재학습 권장")
    else:
        print("레짐 변화 감지되지 않음")


if __name__ == "__main__":
    main()
