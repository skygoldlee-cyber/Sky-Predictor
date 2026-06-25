"""
시장 레짐 감지 시스템
시장 레짐 변화 감지 및 모델 자동 업데이트 트리거
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """ML 데이터셋 로드"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    df['month'] = df['entry_time'].dt.month
    return df


def calculate_regime_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """레짐 감지를 위한 피처 계산"""
    df = df.copy()
    df = df.sort_values('entry_time')
    
    # 변동성 기반 레짐
    df['volatility'] = df['entry_atr'] / df['entry_close'] * 100
    df['volatility_ma'] = df['volatility'].rolling(window=window, min_periods=1).mean()
    df['volatility_regime'] = np.where(df['volatility'] > df['volatility_ma'] * 1.5, 'high', 
                                       np.where(df['volatility'] < df['volatility_ma'] * 0.5, 'low', 'normal'))
    
    # 추세 기반 레짐
    df['trend_strength'] = (df['entry_close'] - df['entry_close'].rolling(window=window, min_periods=1).mean()) / df['entry_close'] * 100
    df['trend_regime'] = np.where(df['trend_strength'] > 1.0, 'uptrend',
                                  np.where(df['trend_strength'] < -1.0, 'downtrend', 'sideways'))
    
    # 거래량 기반 레질
    if 'entry_volume' in df.columns:
        df['volume_ma'] = df['entry_volume'].rolling(window=window, min_periods=1).mean()
        df['volume_regime'] = np.where(df['entry_volume'] > df['volume_ma'] * 1.5, 'high',
                                       np.where(df['entry_volume'] < df['volume_ma'] * 0.5, 'low', 'normal'))
    else:
        df['volume_regime'] = 'normal'
    
    # 종합 레짐
    df['regime_composite'] = df['volatility_regime'] + '_' + df['trend_regime'] + '_' + df['volume_regime']
    
    return df


def detect_regime_changes(df: pd.DataFrame) -> List[Dict]:
    """레짐 변화 감지"""
    df = df.copy()
    df = df.sort_values('entry_time')
    
    regime_changes = []
    current_regime = None
    
    for idx, row in df.iterrows():
        regime = row['regime_composite']
        
        if current_regime is None:
            current_regime = regime
        elif regime != current_regime:
            regime_changes.append({
                'time': row['entry_time'],
                'old_regime': current_regime,
                'new_regime': regime,
                'volatility': row['volatility'],
                'trend_strength': row['trend_strength']
            })
            current_regime = regime
    
    return regime_changes


def analyze_regime_performance(df: pd.DataFrame) -> Dict:
    """레짐별 성과 분석"""
    regime_performance = {}
    
    for regime in df['regime_composite'].unique():
        regime_data = df[df['regime_composite'] == regime]
        
        if len(regime_data) == 0:
            continue
        
        win_rate = regime_data['is_win'].mean() * 100
        avg_pnl = regime_data['net_krw'].mean()
        total_pnl = regime_data['net_krw'].sum()
        trade_count = len(regime_data)
        
        regime_performance[regime] = {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'trade_count': trade_count
        }
    
    return regime_performance


def build_regime_classifier(df: pd.DataFrame) -> Dict:
    """레짐 분류 모델 구축"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'volatility', 'trend_strength'
    ]
    
    # 결측치 처리
    X = df[feature_cols].copy().fillna(0).astype(float)
    y = df['volatility_regime'].copy()
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 성능 평가
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # 피처 중요도
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    result = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importance': feature_importance,
        'classification_report': classification_report(y_test, y_test_pred, zero_division=0)
    }
    
    return result


def trigger_model_update(regime_change: Dict, current_model_version: str) -> str:
    """레짐 변화 시 모델 업데이트 트리거"""
    new_model_version = f"{current_model_version}_regime_{regime_change['new_regime']}"
    
    trigger_info = {
        'trigger_time': regime_change['time'],
        'old_regime': regime_change['old_regime'],
        'new_regime': regime_change['new_regime'],
        'old_model_version': current_model_version,
        'new_model_version': new_model_version,
        'action': 'model_retrain_required'
    }
    
    return trigger_info


def main():
    """메인 함수"""
    print("=" * 80)
    print("시장 레짐 감지 시스템")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 레짐 피처 계산
    df = calculate_regime_features(df, window=20)
    print(f"\n레짐 피처 계산 완료")
    
    # 레짐 변화 감지
    regime_changes = detect_regime_changes(df)
    print(f"\n레짐 변화 감지 완료: {len(regime_changes)}건")
    
    if regime_changes:
        print(f"\n주요 레짐 변화:")
        for i, change in enumerate(regime_changes[:10]):  # 상위 10개만 표시
            print(f"  {i+1}. {change['time']}: {change['old_regime']} → {change['new_regime']}")
    
    # 레짐별 성과 분석
    regime_performance = analyze_regime_performance(df)
    print(f"\n레짐별 성과 분석:")
    for regime, perf in sorted(regime_performance.items(), key=lambda x: x[1]['win_rate'], reverse=True):
        print(f"  {regime}:")
        print(f"    승률: {perf['win_rate']:.2f}%")
        print(f"    평균 PnL: {perf['avg_pnl']:,.0f}원")
        print(f"    총 PnL: {perf['total_pnl']:,.0f}원")
        print(f"    거래 수: {perf['trade_count']}건")
    
    # 레짐 분류 모델 구축
    regime_classifier = build_regime_classifier(df)
    print(f"\n레짐 분류 모델 구축 완료")
    print(f"  훈련 정확도: {regime_classifier['train_accuracy']:.4f}")
    print(f"  테스트 정확도: {regime_classifier['test_accuracy']:.4f}")
    
    print(f"\n피처 중요도:")
    for feature, importance in sorted(regime_classifier['feature_importance'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # 모델 업데이트 트리거 시뮬레이션
    if regime_changes:
        print(f"\n모델 업데이트 트리거 시뮬레이션:")
        for i, change in enumerate(regime_changes[:5]):  # 상위 5개만 시뮬레이션
            trigger_info = trigger_model_update(change, f"v2.9")
            print(f"  {i+1}. {trigger_info['trigger_time']}:")
            print(f"    레짐 변화: {trigger_info['old_regime']} → {trigger_info['new_regime']}")
            print(f"    모델 버전: {trigger_info['old_model_version']} → {trigger_info['new_model_version']}")
            print(f"    조치: {trigger_info['action']}")
    
    # 모델 저장
    import joblib
    model_path = MODELS_DIR / "regime_classifier.pkl"
    joblib.dump(regime_classifier['model'], model_path)
    print(f"\n레짐 분류 모델 저장: {model_path}")
    
    # 레짐 데이터 저장
    df.to_csv(DATA_DIR / "ml_dataset_with_regime.csv", index=False)
    print(f"레질 데이터 저장: {DATA_DIR / 'ml_dataset_with_regime.csv'}")
    
    # 레짐 변화 저장
    import json
    regime_changes_path = MODELS_DIR / "regime_changes.json"
    with open(regime_changes_path, 'w', encoding='utf-8') as f:
        json.dump(regime_changes, f, indent=2, ensure_ascii=False, default=str)
    print(f"레짐 변화 저장: {regime_changes_path}")
    
    return regime_classifier, regime_changes, regime_performance


if __name__ == "__main__":
    regime_classifier, regime_changes, regime_performance = main()
