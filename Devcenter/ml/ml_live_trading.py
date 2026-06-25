"""
실매매 실행 파일
실시간 마켓 모니터링 및 진입/청산 신호 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """실매매 엔진"""
    
    def __init__(self, entry_threshold: float = 0.6, position_size: int = 1):
        """
        실매매 엔진 초기화
        
        Args:
            entry_threshold: 진입 임계값 (보수적: 0.6, 공격적: 0.5)
            position_size: 포지션 사이즈 (보수적: 1, 공격적: 3)
        """
        self.entry_threshold = entry_threshold
        self.position_size = position_size
        self.model = None
        self.current_position = None
        self.consecutive_losses = 0
        self.max_consecutive_losses = 1
        self.max_loss = 1000000  # 100만원
        
        # 피처 목록
        self.feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime',
            'rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'
        ]
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        model_path = MODELS_DIR / "rf_conservative.pkl"
        
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"모델 로드 완료: {model_path}")
        else:
            logger.warning(f"모델 파일 없음: {model_path}")
            logger.info("새로운 모델 학습 시작...")
            self._train_model()
    
    def _train_model(self):
        """모델 학습"""
        try:
            # 데이터 로드
            df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            
            # 파생 피처 계산
            df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
            df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
            df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
            df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
            
            # 학습 데이터 준비
            X = df[self.feature_cols].copy().fillna(0).astype(float)
            y = df['is_win'].copy()
            
            # 모델 학습
            self.model = RandomForestClassifier(
                n_estimators=20,
                max_depth=3,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model.fit(X, y)
            
            # 모델 저장
            joblib.dump(self.model, MODELS_DIR / "rf_conservative.pkl")
            logger.info("모델 학습 및 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 학습 실패: {e}")
            raise
    
    def calculate_features(self, market_data: Dict) -> Dict:
        """
        시장 데이터로 피처 계산
        
        Args:
            market_data: 시장 데이터 딕셔너리
                - close: 종가
                - rsi: RSI
                - macd: MACD
                - macd_signal: MACD Signal
                - macd_hist: MACD Histogram
                - atr: ATR
                - supertrend: SuperTrend
                - supertrend_dir: SuperTrend 방향
                - ma20: MA20
                - ma60: MA60
                - bb_upper: BB 상단
                - bb_lower: BB 하단
                - bb_middle: BB 중단
                - regime: 레짐
        
        Returns:
            피처 딕셔너리
        """
        now = datetime.now()
        
        # 기본 피처
        features = {
            'entry_rsi': market_data.get('rsi', 50),
            'entry_macd': market_data.get('macd', 0),
            'entry_macd_signal': market_data.get('macd_signal', 0),
            'entry_macd_hist': market_data.get('macd_hist', 0),
            'entry_atr': market_data.get('atr', 1),
            'entry_supertrend': market_data.get('supertrend', market_data.get('close', 0)),
            'entry_supertrend_dir': market_data.get('supertrend_dir', 1),
            'entry_ma20': market_data.get('ma20', market_data.get('close', 0)),
            'entry_ma60': market_data.get('ma60', market_data.get('close', 0)),
            'entry_bb_upper': market_data.get('bb_upper', market_data.get('close', 0) * 1.02),
            'entry_bb_lower': market_data.get('bb_lower', market_data.get('close', 0) * 0.98),
            'entry_bb_middle': market_data.get('bb_middle', market_data.get('close', 0)),
            'entry_hour': now.hour,
            'entry_dayofweek': now.weekday(),
            'entry_month': now.month,
            'regime': market_data.get('regime', 1)
        }
        
        # 파생 피처
        close = market_data.get('close', 0)
        features['rsi_ma_ratio'] = features['entry_rsi'] / features['entry_ma20'] if features['entry_ma20'] > 0 else 1
        features['price_ma_ratio'] = close / features['entry_ma20'] if features['entry_ma20'] > 0 else 1
        features['bb_position'] = (close - features['entry_bb_lower']) / (features['entry_bb_upper'] - features['entry_bb_lower']) if (features['entry_bb_upper'] - features['entry_bb_lower']) > 0 else 0.5
        features['supertrend_alignment'] = 1 if (close > features['entry_supertrend'] and features['entry_supertrend_dir'] == 1) else 0
        
        return features
    
    def predict(self, features: Dict) -> float:
        """
        승률 예측
        
        Args:
            features: 피처 딕셔너리
        
        Returns:
            예측 확률 (0-1)
        """
        if self.model is None:
            logger.error("모델이 로드되지 않음")
            return 0.0
        
        try:
            # 피처 벡터 생성
            feature_vector = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
            
            # 예측
            predicted_prob = self.model.predict_proba(feature_vector)[0, 1]
            
            return predicted_prob
            
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            return 0.0
    
    def generate_entry_signal(self, market_data: Dict) -> Dict:
        """
        진입 신호 생성
        
        Args:
            market_data: 시장 데이터 딕셔너리
        
        Returns:
            신호 딕셔너리
                - signal: 진입 신호 (True/False)
                - probability: 예측 확률
                - position_size: 포지션 사이즈
                - timestamp: 신호 생성 시간
        """
        # 피처 계산
        features = self.calculate_features(market_data)
        
        # 예측
        predicted_prob = self.predict(features)
        
        # 진입 결정
        signal = predicted_prob >= self.entry_threshold
        
        # 리스크 관리
        if signal and self.current_position is not None:
            logger.warning("이미 포지션 보유 중")
            signal = False
        
        if signal and self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"연속 손실 제한 도달: {self.consecutive_losses}")
            signal = False
        
        result = {
            'signal': signal,
            'probability': predicted_prob,
            'position_size': self.position_size if signal else 0,
            'timestamp': datetime.now().isoformat(),
            'features': features
        }
        
        if signal:
            logger.info(f"진입 신호 생성: 확률={predicted_prob:.4f}, 포지션={self.position_size}")
        
        return result
    
    def generate_exit_signal(self, market_data: Dict, current_pnl: float) -> Dict:
        """
        청산 신호 생성
        
        Args:
            market_data: 시장 데이터 딕셔너리
            current_pnl: 현재 PnL
        
        Returns:
            신호 딕셔너리
                - signal: 청산 신호 (True/False)
                - reason: 청산 사유
                - timestamp: 신호 생성 시간
        """
        signal = False
        reason = ""
        
        # 최대 손실 제한
        if current_pnl <= -self.max_loss:
            signal = True
            reason = f"최대 손실 제한 도달: {current_pnl:,.0f}원"
            logger.warning(reason)
        
        # 기타 청산 조건 (필요시 추가)
        # 예: 목표 수익 도달, 시간 경과 등
        
        result = {
            'signal': signal,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        if signal:
            logger.info(f"청산 신호 생성: {reason}")
        
        return result
    
    def update_position(self, pnl: float, is_win: bool):
        """
        포지션 업데이트
        
        Args:
            pnl: PnL
            is_win: 승리 여부
        """
        if not is_win:
            self.consecutive_losses += 1
            logger.warning(f"손실 발생: {pnl:,.0f}원, 연속 손실: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"수익 발생: {pnl:,.0f}원")
    
    def reset_position(self):
        """포지션 리셋"""
        self.current_position = None
        logger.info("포지션 리셋")


def main():
    """메인 함수 (테스트용)"""
    print("=" * 80)
    print("실매매 엔진 테스트")
    print("=" * 80)
    
    # 엔진 초기화
    engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
    
    # 테스트 데이터
    test_market_data = {
        'close': 350.0,
        'rsi': 45.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_hist': 0.2,
        'atr': 2.0,
        'supertrend': 348.0,
        'supertrend_dir': 1,
        'ma20': 349.0,
        'ma60': 347.0,
        'bb_upper': 352.0,
        'bb_lower': 346.0,
        'bb_middle': 349.0,
        'regime': 1
    }
    
    # 진입 신호 테스트
    print("\n진입 신호 테스트:")
    entry_signal = engine.generate_entry_signal(test_market_data)
    print(f"  신호: {entry_signal['signal']}")
    print(f"  확률: {entry_signal['probability']:.4f}")
    print(f"  포지션: {entry_signal['position_size']}")
    
    # 청산 신호 테스트
    print("\n청산 신호 테스트:")
    exit_signal = engine.generate_exit_signal(test_market_data, -1500000)
    print(f"  신호: {exit_signal['signal']}")
    print(f"  사유: {exit_signal['reason']}")
    
    print("\n테스트 완료")


if __name__ == "__main__":
    main()
