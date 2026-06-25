# -*- coding: utf-8 -*-
"""
동적 모델 선택 시스템
연도별/시장 구조별로 다른 모델을 사용
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


class DynamicModelSelector:
    """동적 모델 선택 시스템"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """모델 로드"""
        # 기본 모델
        self.models['default'] = {
            'trade_filter': xgb.XGBClassifier()
        }
        self.models['default']['trade_filter'].load_model(str(MODELS_DIR / "trade_filter_xgboost.json"))
        
        # 2019-2026년 특화 모델
        for year in range(2019, 2027):
            model_path = MODELS_DIR / f"trade_filter_xgboost_{year}.json"
            if model_path.exists():
                self.models[str(year)] = {
                    'trade_filter': xgb.XGBClassifier()
                }
                self.models[str(year)]['trade_filter'].load_model(str(model_path))
                print(f"{year}년 특화 모델 로드 완료")
        
        # 롤링 윈도우 모델
        if (MODELS_DIR / "trade_filter_xgboost_rolling_realistic.json").exists():
            self.models['rolling'] = {
                'trade_filter': xgb.XGBClassifier()
            }
            self.models['rolling']['trade_filter'].load_model(str(MODELS_DIR / "trade_filter_xgboost_rolling_realistic.json"))
            print("현실적 롤링 윈도우 모델 로드 완료")
        elif (MODELS_DIR / "trade_filter_xgboost_rolling_optimized.json").exists():
            self.models['rolling'] = {
                'trade_filter': xgb.XGBClassifier()
            }
            self.models['rolling']['trade_filter'].load_model(str(MODELS_DIR / "trade_filter_xgboost_rolling_optimized.json"))
            print("최적 롤링 윈도우 모델 로드 완료")
        elif (MODELS_DIR / "trade_filter_xgboost_rolling.json").exists():
            self.models['rolling'] = {
                'trade_filter': xgb.XGBClassifier()
            }
            self.models['rolling']['trade_filter'].load_model(str(MODELS_DIR / "trade_filter_xgboost_rolling.json"))
            print("롤링 윈도우 모델 로드 완료")
        
        # 진입 타이밍 모델
        if (MODELS_DIR / "entry_timing_rf.pkl").exists():
            self.models['default']['entry_timing'] = joblib.load(MODELS_DIR / "entry_timing_rf.pkl")
            print("진입 타이밍 모델 로드 완료")
        
        print(f"로드된 모델: {list(self.models.keys())}")
    
    def get_trade_filter_model(self, year: int, regime: int = None) -> xgb.XGBClassifier:
        """연도별/레짐별 거래 필터링 모델 선택"""
        # 레짐 기반 모델 선택 (우선)
        if regime is not None:
            regime_key = f"{year}_regime_{regime}"
            if regime_key in self.models:
                return self.models[regime_key]['trade_filter']
        
        # 모든 연도에 롤링 윈도우 모델 사용 (현실적 선택)
        if 'rolling' in self.models:
            return self.models['rolling']['trade_filter']
        
        # 롤링 윈도우 모델이 없으면 연도별 모델 사용
        if str(year) in self.models:
            return self.models[str(year)]['trade_filter']
        else:
            return self.models['default']['trade_filter']
    
    def get_entry_timing_model(self) -> Optional[RandomForestClassifier]:
        """진입 타이밍 모델 선택"""
        return self.models['default'].get('entry_timing')
    
    def predict_trade_filter(self, df: pd.DataFrame, year: int, regime: int = None) -> np.ndarray:
        """거래 필터링 예측"""
        model = self.get_trade_filter_model(year, regime)
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].copy().fillna(0).astype(float)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # 현실적 롤링 윈도우 모델 최적 threshold 사용
        threshold = 0.45
        
        return y_pred_proba >= threshold, y_pred_proba, threshold


def test_dynamic_model_selection():
    """동적 모델 선택 시스템 테스트"""
    print("=" * 80)
    print("동적 모델 선택 시스템 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    
    # 동적 모델 선택 시스템 초기화
    selector = DynamicModelSelector()
    
    # 계약수 설정
    contract_size = 3
    
    print(f"\n계약수: {contract_size}계약")
    
    # 연도별 테스트
    total_pnl = 0
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].copy()
        
        print(f"\n{'='*80}")
        print(f"{year}년 테스트")
        print(f"{'='*80}")
        
        # 거래 필터링 예측
        filtered_mask, y_pred_proba, threshold = selector.predict_trade_filter(df_year, year)
        filtered_df = df_year[filtered_mask].copy()
        
        # 3계약 기준 PnL 계산
        filtered_df['net_krw_3contract'] = filtered_df['net_krw'] * contract_size
        total_pnl_year = filtered_df['net_krw_3contract'].sum()
        total_pnl += total_pnl_year
        
        print(f"Threshold: {threshold:.2f}")
        print(f"필터링 전: {len(df_year)}건 (승률: {df_year['is_win'].mean() * 100:.2f}%)")
        print(f"필터링 후: {len(filtered_df)}건 (승률: {filtered_df['is_win'].mean() * 100:.2f}%)")
        print(f"총 PnL (1계약): {filtered_df['net_krw'].sum():,.0f}원")
        print(f"총 PnL ({contract_size}계약): {total_pnl_year:,.0f}원")
        
        # 롤링 윈도우 모델 사용 여부
        if 'rolling' in selector.models:
            print("롤링 윈도우 모델 사용")
        elif str(year) in selector.models:
            print(f"{year}년 특화 모델 사용")
        else:
            print("기본 모델 사용")
    
    # 전체 성과
    print(f"\n{'='*80}")
    print("전체 성과")
    print(f"{'='*80}")
    print(f"계약수: {contract_size}계약")
    print(f"총 PnL: {total_pnl:,.0f}원")


if __name__ == "__main__":
    test_dynamic_model_selection()
