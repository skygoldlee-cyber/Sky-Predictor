"""
실매매 실행 파일 전체 테스트 검증
다양한 시나리오 테스트 및 기능 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import logging

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent))
from ml_live_trading import LiveTradingEngine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """모델 로드 테스트"""
    print("\n" + "="*80)
    print("테스트 1: 모델 로드")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        assert engine.model is not None, "모델 로드 실패"
        print("[PASS] 모델 로드 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 모델 로드 실패: {e}")
        return False


def test_feature_calculation():
    """피처 계산 테스트"""
    print("\n" + "="*80)
    print("테스트 2: 피처 계산")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        # 테스트 데이터
        market_data = {
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
        
        features = engine.calculate_features(market_data)
        
        # 피처 개수 확인
        assert len(features) == 20, f"피처 개수 오류: {len(features)} (예상: 20)"
        
        # 필수 피처 확인
        required_features = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime',
            'rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'
        ]
        
        for feature in required_features:
            assert feature in features, f"필수 피처 누락: {feature}"
        
        print(f"[PASS] 피처 계산 성공 ({len(features)}개 피처)")
        print(f"  샘플 피처: rsi={features['entry_rsi']:.2f}, macd={features['entry_macd']:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] 피처 계산 실패: {e}")
        return False


def test_prediction():
    """예측 기능 테스트"""
    print("\n" + "="*80)
    print("테스트 3: 예측 기능")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        market_data = {
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
        
        predicted_prob = engine.predict(market_data)
        
        # 예측 확률 범위 확인
        assert 0 <= predicted_prob <= 1, f"예측 확률 범위 오류: {predicted_prob}"
        
        print(f"[PASS] 예측 기능 성공")
        print(f"  예측 확률: {predicted_prob:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] 예측 기능 실패: {e}")
        return False


def test_entry_signal_conservative():
    """진입 신호 테스트 (보수적 설정)"""
    print("\n" + "="*80)
    print("테스트 4: 진입 신호 (보수적 설정)")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        # 높은 확률 시나리오
        market_data_high = {
            'close': 350.0,
            'rsi': 30.0,  # 과매도
            'macd': 1.0,
            'macd_signal': 0.5,
            'macd_hist': 0.5,
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
        
        signal_high = engine.generate_entry_signal(market_data_high)
        print(f"  높은 확률 시나리오: 신호={signal_high['signal']}, 확률={signal_high['probability']:.4f}")
        
        # 낮은 확률 시나리오
        market_data_low = {
            'close': 350.0,
            'rsi': 70.0,  # 과매수
            'macd': -0.5,
            'macd_signal': 0.3,
            'macd_hist': -0.8,
            'atr': 2.0,
            'supertrend': 352.0,
            'supertrend_dir': -1,
            'ma20': 349.0,
            'ma60': 347.0,
            'bb_upper': 352.0,
            'bb_lower': 346.0,
            'bb_middle': 349.0,
            'regime': 0
        }
        
        signal_low = engine.generate_entry_signal(market_data_low)
        print(f"  낮은 확률 시나리오: 신호={signal_low['signal']}, 확률={signal_low['probability']:.4f}")
        
        print(f"[PASS] 진입 신호 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 진입 신호 테스트 실패: {e}")
        return False


def test_entry_signal_aggressive():
    """진입 신호 테스트 (공격적 설정)"""
    print("\n" + "="*80)
    print("테스트 5: 진입 신호 (공격적 설정)")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.5, position_size=3)
        
        market_data = {
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
        
        signal = engine.generate_entry_signal(market_data)
        
        print(f"  공격적 설정: 신호={signal['signal']}, 확률={signal['probability']:.4f}, 포지션={signal['position_size']}")
        print(f"[PASS] 공격적 설정 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 공격적 설정 테스트 실패: {e}")
        return False


def test_exit_signal():
    """청산 신호 테스트"""
    print("\n" + "="*80)
    print("테스트 6: 청산 신호")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        market_data = {
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
        
        # 최대 손실 도달 시나리오
        exit_signal_loss = engine.generate_exit_signal(market_data, -1500000)
        print(f"  최대 손실 도달: 신호={exit_signal_loss['signal']}, 사유={exit_signal_loss['reason']}")
        assert exit_signal_loss['signal'] == True, "최대 손실 청산 신호 실패"
        
        # 정상 PnL 시나리오
        exit_signal_normal = engine.generate_exit_signal(market_data, 500000)
        print(f"  정상 PnL: 신호={exit_signal_normal['signal']}, 사유={exit_signal_normal['reason']}")
        
        print(f"[PASS] 청산 신호 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 청산 신호 테스트 실패: {e}")
        return False


def test_position_management():
    """포지션 관리 테스트"""
    print("\n" + "="*80)
    print("테스트 7: 포지션 관리")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        # 승리 시나리오
        engine.update_position(500000, True)
        assert engine.consecutive_losses == 0, "승리 후 연속 손실 초기화 실패"
        print(f"  승리 후 연속 손실: {engine.consecutive_losses}")
        
        # 손실 시나리오
        engine.update_position(-500000, False)
        assert engine.consecutive_losses == 1, "손실 후 연속 손실 증가 실패"
        print(f"  손실 후 연속 손실: {engine.consecutive_losses}")
        
        # 연속 손실 시나리오
        engine.update_position(-500000, False)
        assert engine.consecutive_losses == 2, "연속 손실 증가 실패"
        print(f"  연속 손실 후: {engine.consecutive_losses}")
        
        # 포지션 리셋
        engine.reset_position()
        print(f"  포지션 리셋 완료")
        
        print(f"[PASS] 포지션 관리 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 포지션 관리 테스트 실패: {e}")
        return False


def test_consecutive_loss_limit():
    """연속 손실 제한 테스트"""
    print("\n" + "="*80)
    print("테스트 8: 연속 손실 제한")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        # 연속 손실 설정
        engine.consecutive_losses = 1
        engine.max_consecutive_losses = 1
        
        market_data = {
            'close': 350.0,
            'rsi': 30.0,
            'macd': 1.0,
            'macd_signal': 0.5,
            'macd_hist': 0.5,
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
        
        # 연속 손실 제한 도달 시 진입 신호 차단
        signal = engine.generate_entry_signal(market_data)
        print(f"  연속 손실 제한 도달 시 신호: {signal['signal']}")
        assert signal['signal'] == False, "연속 손실 제한 진입 차단 실패"
        
        print(f"[PASS] 연속 손실 제한 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 연속 손실 제한 테스트 실패: {e}")
        return False


def test_position_conflict():
    """포지션 충돌 테스트"""
    print("\n" + "="*80)
    print("테스트 9: 포지션 충돌")
    print("="*80)
    
    try:
        engine = LiveTradingEngine(entry_threshold=0.6, position_size=1)
        
        # 포지션 보유 상태 설정
        engine.current_position = {'entry_time': datetime.now()}
        
        market_data = {
            'close': 350.0,
            'rsi': 30.0,
            'macd': 1.0,
            'macd_signal': 0.5,
            'macd_hist': 0.5,
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
        
        # 포지션 보유 중 진입 신호 차단
        signal = engine.generate_entry_signal(market_data)
        print(f"  포지션 보유 중 신호: {signal['signal']}")
        assert signal['signal'] == False, "포지션 충돌 진입 차단 실패"
        
        print(f"[PASS] 포지션 충돌 테스트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] 포지션 충돌 테스트 실패: {e}")
        return False


def run_all_tests():
    """전체 테스트 실행"""
    print("="*80)
    print("실매매 실행 파일 전체 테스트 검증")
    print("="*80)
    
    tests = [
        test_model_loading,
        test_feature_calculation,
        test_prediction,
        test_entry_signal_conservative,
        test_entry_signal_aggressive,
        test_exit_signal,
        test_position_management,
        test_consecutive_loss_limit,
        test_position_conflict
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] 테스트 실행 중 오류: {e}")
            results.append(False)
    
    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"통과: {passed}/{total}")
    print(f"실패: {total - passed}/{total}")
    
    if passed == total:
        print("\n[PASS] 모든 테스트 통과")
        return True
    else:
        print(f"\n[FAIL] {total - passed}개 테스트 실패")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
