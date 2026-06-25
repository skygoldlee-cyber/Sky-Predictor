# -*- coding: utf-8 -*-
"""
롤링 윈도우 기반 모델 업데이트
미래 거래에 적용하기 위한 동적 모델 업데이트 시스템
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


class RollingWindowModelSelector:
    """롤링 윈도우 기반 모델 선택 시스템"""
    
    def __init__(self, window_size_years=2):
        self.window_size_years = window_size_years
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """기존 모델 로드"""
        # 기본 모델
        self.models['default'] = {
            'trade_filter': xgb.XGBClassifier()
        }
        self.models['default']['trade_filter'].load_model(str(MODELS_DIR / "trade_filter_xgboost.json"))
        
        # 연도별 특화 모델
        for year in range(2019, 2027):
            model_path = MODELS_DIR / f"trade_filter_xgboost_{year}.json"
            if model_path.exists():
                self.models[str(year)] = {
                    'trade_filter': xgb.XGBClassifier()
                }
                self.models[str(year)]['trade_filter'].load_model(str(model_path))
        
        print(f"로드된 모델: {list(self.models.keys())}")
    
    def get_model_for_future_trade(self, trade_date: str) -> xgb.XGBClassifier:
        """미래 거래에 적용할 모델 선택"""
        trade_date = pd.to_datetime(trade_date)
        current_year = trade_date.year
        
        # 현재 연도 모델이 있으면 사용
        if str(current_year) in self.models:
            return self.models[str(current_year)]['trade_filter']
        
        # 가장 최근 연도 모델 사용
        available_years = [int(y) for y in self.models.keys() if y.isdigit()]
        if available_years:
            most_recent_year = max(available_years)
            print(f"{current_year}년 모델 없음, {most_recent_year}년 모델 사용")
            return self.models[str(most_recent_year)]['trade_filter']
        
        # 기본 모델 사용
        print(f"{current_year}년 모델 없음, 기본 모델 사용")
        return self.models['default']['trade_filter']
    
    def train_rolling_window_model(self, end_date: str) -> xgb.XGBClassifier:
        """롤링 윈도우 기반 모델 학습"""
        # 데이터 로드
        df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        
        # 롤링 윈도우 데이터 선택
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.DateOffset(years=self.window_size_years)
        
        df_window = df[(df['entry_time'] >= start_date) & (df['entry_time'] <= end_date)].copy()
        
        if len(df_window) < 100:
            print(f"롤링 윈도우 데이터 부족: {len(df_window)}건")
            return None
        
        print(f"롤링 윈도우 기간: {start_date} ~ {end_date}")
        print(f"롤링 윈도우 데이터: {len(df_window)}건")
        
        # 피처 선택
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df_window[feature_cols].copy().fillna(0).astype(float)
        y = df_window['is_win'].copy()
        
        # 시간 기반 분할
        df_window_sorted = df_window.sort_values('entry_time').reset_index(drop=True)
        split_idx = int(len(df_window_sorted) * 0.7)
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
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
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def update_model_rolling_window(self, current_date: str):
        """롤링 윈도우 기반 모델 업데이트"""
        model = self.train_rolling_window_model(current_date)
        
        if model is not None:
            # 모델 저장
            model_path = MODELS_DIR / "trade_filter_xgboost_rolling.json"
            model.save_model(str(model_path))
            print(f"롤링 윈도우 모델 저장 완료: {model_path}")
            
            # 모델 로드
            self.models['rolling'] = {
                'trade_filter': model
            }
            
            return model
        
        return None


def test_rolling_window_model():
    """롤링 윈도우 모델 테스트"""
    print("=" * 80)
    print("롤링 윈도우 기반 모델 테스트")
    print("=" * 80)
    
    # 롤링 윈도우 모델 선택 시스템 초기화
    selector = RollingWindowModelSelector(window_size_years=2)
    
    # 미래 거래 시뮬레이션
    future_dates = ['2027-01-15', '2028-03-20', '2029-06-10']
    
    for date in future_dates:
        print(f"\n{date} 거래:")
        model = selector.get_model_for_future_trade(date)
        print(f"사용 모델: {model}")
    
    # 롤링 윈도우 모델 업데이트
    print(f"\n{'='*80}")
    print("롤링 윈도우 모델 업데이트")
    print(f"{'='*80}")
    
    current_date = '2026-06-22'
    model = selector.update_model_rolling_window(current_date)
    
    if model is not None:
        print(f"\n롤링 윈도우 모델 업데이트 완료")
        
        # 롤링 윈도우 모델 사용
        for date in future_dates:
            print(f"\n{date} 거래 (롤링 윈도우 모델):")
            if 'rolling' in selector.models:
                print(f"롤링 윈도우 모델 사용")
            else:
                model = selector.get_model_for_future_trade(date)
                print(f"사용 모델: {model}")


if __name__ == "__main__":
    test_rolling_window_model()
