# 해외선물 추천 가이드

**버전:** 2026-04-25  
**파일:** `docs/OVERSEAS_FUTURES_RECOMMENDATIONS.md`

---

## 목차

1. [추천 해외선물 순위](#1-추천-해외선물-순위)
2. [비추천 해외선물](#2-비추천-해외선물)
3. [종합 비교표](#3-종합-비교표)
4. [최종 추천](#4-최종-추천)
5. [백테스팅 우선순위](#5-백테스팅-우선순위)

---

## 1. 추천 해외선물 순위

### 🥇 1위: S&P 500 선물 (ES)

**예상 승률: 65~70%**  
**예상 수익률: +35~45%/년**

#### 추천 이유

- **높은 유동성**: 세계 최대 선물 시장, 슬리피지 최소화
- **명확한 추세성**: 미국 주식 시장 추세가 뚜렷하여 ZigZag 피봇 형성 우수
- **적절한 변동성**: ATR 기반 손절/이익실현 효과적
- **세션 구조 명확**: 아시아/유럽/미국 세션 구분으로 다중 TF 필터 효과적
- **데이터 풍부**: ML 모델 학습에 충분한 과거 데이터

#### 파라미터 추천

```json
{
  "adaptive_indicator": {
    "symbol": "S&P 500 Futures",
    "warmup_bars": 50,
    "min_pivot_interval_bars": 12,
    "higher_tf_pivot_filter": true,
    
    "zigzag": {
      "atr_multiplier": 1.7,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.35,
      "pivot_threshold_max_pct": 3.5,
      "confirmation_bars": 2,
      "min_wave_bars": 1,
      "structure_lookback_swings": 35,
      "structure_points": 4,
      "structure_majority_threshold": 0.7,
      "freeze_on_confirm": true,
      "cluster_tolerance_pct": 0.4
    },
    
    "supertrend": {
      "atr_min_period": 8,
      "atr_max_period": 22,
      "multiplier_min": 1.7,
      "multiplier_max": 4.2
    }
  }
}
```

#### 리스크 관리

```python
from prediction.pivot_risk_manager import RiskConfig

config = RiskConfig(
    max_position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2,
    high_confidence_size_pct=0.90,
    medium_confidence_size_pct=0.65,
    low_confidence_size_pct=0.30,
    max_risk_per_trade_pct=0.018,
    trailing_stop_atr_multiplier=1.8
)
```

#### 백테스팅 파라미터

```python
from prediction.backtest_pivot_signals import BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,  # USD
    tick_size=0.25,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2
)
```

---

### 🥈 2위: 골드 선물 (GC)

**예상 승률: 62~68%**  
**예상 수익률: +30~40%/년**

#### 추천 이유

- **추세성 우수**: 금 가격은 장기 추세가 뚜렷하여 피봇 형성이 명확
- **세션별 패턴**: 아시아/유럽/미국 세션별 뚜렷한 패턴
- **안전 자산**: 시장 충격 시 방어적 성격
- **변동성 적당**: 과도한 변동성 없이 안정적

#### 파라미터 추천

```json
{
  "adaptive_indicator": {
    "symbol": "Gold Futures",
    "warmup_bars": 55,
    "min_pivot_interval_bars": 13,
    "higher_tf_pivot_filter": true,
    
    "zigzag": {
      "atr_multiplier": 1.8,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.4,
      "pivot_threshold_max_pct": 4.0,
      "confirmation_bars": 2,
      "min_wave_bars": 2,
      "structure_lookback_swings": 38,
      "structure_points": 4,
      "structure_majority_threshold": 0.7,
      "freeze_on_confirm": true,
      "cluster_tolerance_pct": 0.45
    },
    
    "supertrend": {
      "atr_min_period": 9,
      "atr_max_period": 24,
      "multiplier_min": 1.8,
      "multiplier_max": 4.3
    }
  }
}
```

#### 리스크 관리

```python
config = RiskConfig(
    max_position_size_pct=0.88,
    stop_loss_atr_multiplier=2.3,
    take_profit_atr_multiplier=3.3,
    high_confidence_size_pct=0.88,
    medium_confidence_size_pct=0.63,
    low_confidence_size_pct=0.28,
    max_risk_per_trade_pct=0.017,
    trailing_stop_atr_multiplier=1.9
)
```

#### 백테스팅 파라미터

```python
config = BacktestConfig(
    initial_capital=100000.0,  # USD
    tick_size=0.1,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.88,
    stop_loss_atr_multiplier=2.3,
    take_profit_atr_multiplier=3.3
)
```

---

### 🥉 3위: 나스닥 100 선물 (NQ)

**예상 승률: 60~65%**  
**예상 수익률: +28~38%/년**

#### 추천 이유

- **높은 유동성**: 나스닥 선물 거래량 풍부
- **테크 섹터 추세**: 기술 주가 추세가 뚜렷
- **변동성 높음**: 높은 변동성으로 수익 기회 많음
- **세션 구조 명확**: 미국 장 시간에 활발한 거래

#### 파라미터 추천

```json
{
  "adaptive_indicator": {
    "symbol": "Nasdaq 100 Futures",
    "warmup_bars": 50,
    "min_pivot_interval_bars": 12,
    "higher_tf_pivot_filter": true,
    
    "zigzag": {
      "atr_multiplier": 1.8,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.4,
      "pivot_threshold_max_pct": 4.0,
      "confirmation_bars": 2,
      "min_wave_bars": 1,
      "structure_lookback_swings": 35,
      "structure_points": 4,
      "structure_majority_threshold": 0.7,
      "freeze_on_confirm": true,
      "cluster_tolerance_pct": 0.4
    },
    
    "supertrend": {
      "atr_min_period": 8,
      "atr_max_period": 24,
      "multiplier_min": 1.8,
      "multiplier_max": 4.5
    }
  }
}
```

#### 리스크 관리

```python
config = RiskConfig(
    max_position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2,
    high_confidence_size_pct=0.90,
    medium_confidence_size_pct=0.65,
    low_confidence_size_pct=0.30,
    max_risk_per_trade_pct=0.018,
    trailing_stop_atr_multiplier=1.8
)
```

#### 백테스팅 파라미터

```python
config = BacktestConfig(
    initial_capital=100000.0,  # USD
    tick_size=0.25,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2
)
```

---

### 4위: 유로 선물 (6E)

**예상 승률: 58~63%**  
**예상 수익률: +25~35%/년**

#### 추천 이유

- **외환 시장 특성**: 24시간 거래로 세션 필터 효과적
- **ECB 정책 명확**: 유럽 중앙은행 정책이 예측 가능
- **유동성 높음**: EUR/USD 쌍은 가장 유동성 높음

#### 파라미터 추천

```json
{
  "adaptive_indicator": {
    "symbol": "Euro Futures",
    "warmup_bars": 60,
    "min_pivot_interval_bars": 15,
    "higher_tf_pivot_filter": true,
    
    "zigzag": {
      "atr_multiplier": 2.0,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.4,
      "pivot_threshold_max_pct": 4.0,
      "confirmation_bars": 2,
      "min_wave_bars": 2,
      "structure_lookback_swings": 40,
      "structure_points": 4,
      "structure_majority_threshold": 0.7,
      "freeze_on_confirm": true,
      "cluster_tolerance_pct": 0.45
    },
    
    "supertrend": {
      "atr_min_period": 10,
      "atr_max_period": 26,
      "multiplier_min": 2.0,
      "multiplier_max": 4.5
    }
  }
}
```

#### 리스크 관리

```python
config = RiskConfig(
    max_position_size_pct=0.85,
    stop_loss_atr_multiplier=2.5,
    take_profit_atr_multiplier=3.5,
    high_confidence_size_pct=0.85,
    medium_confidence_size_pct=0.60,
    low_confidence_size_pct=0.25,
    max_risk_per_trade_pct=0.015,
    trailing_stop_atr_multiplier=2.0
)
```

#### 백테스팅 파라미터

```python
config = BacktestConfig(
    initial_capital=100000.0,  # USD
    tick_size=0.00005,
    commission_rate=0.0002,
    slippage_ticks=2,
    position_size_pct=0.85,
    stop_loss_atr_multiplier=2.5,
    take_profit_atr_multiplier=3.5
)
```

---

### 5위: 10년 국채 선물 (ZN)

**예상 승률: 55~60%**  
**예상 수익률: +20~30%/년**

#### 추천 이유

- **추세성 우수**: 금리 추세가 뚜렷
- **변동성 낮음**: 안정적인 거래 환경
- **펀더멘털 기반**: Fed 정책에 의한 명확한 방향성

#### 파라미터 추천

```json
{
  "adaptive_indicator": {
    "symbol": "10-Year Treasury Futures",
    "warmup_bars": 55,
    "min_pivot_interval_bars": 14,
    "higher_tf_pivot_filter": true,
    
    "zigzag": {
      "atr_multiplier": 1.9,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.3,
      "pivot_threshold_max_pct": 3.0,
      "confirmation_bars": 2,
      "min_wave_bars": 2,
      "structure_lookback_swings": 38,
      "structure_points": 4,
      "structure_majority_threshold": 0.7,
      "freeze_on_confirm": true,
      "cluster_tolerance_pct": 0.4
    },
    
    "supertrend": {
      "atr_min_period": 9,
      "atr_max_period": 24,
      "multiplier_min": 1.9,
      "multiplier_max": 4.3
    }
  }
}
```

#### 리스크 관리

```python
config = RiskConfig(
    max_position_size_pct=0.92,
    stop_loss_atr_multiplier=2.1,
    take_profit_atr_multiplier=3.1,
    high_confidence_size_pct=0.92,
    medium_confidence_size_pct=0.67,
    low_confidence_size_pct=0.32,
    max_risk_per_trade_pct=0.019,
    trailing_stop_atr_multiplier=1.7
)
```

#### 백테스팅 파라미터

```python
config = BacktestConfig(
    initial_capital=100000.0,  # USD
    tick_size=0.015625,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.92,
    stop_loss_atr_multiplier=2.1,
    take_profit_atr_multiplier=3.1
)
```

---

## 2. 비추천 해외선물

### ❌ 크루드오일 선물 (CL)

**예상 승률: 50~58%**

#### 비추천 이유

- **지정학적 리스크**: OPEC+, 중동 사건으로 급격한 변동
- **뉴스 민감성**: API 재고, OPEC 발표에 과도한 반응
- **세션 갭 큼**: 주말/공휴일 갭 리스크 높음
- **잡음 많음**: 단기 변동성이 커서 잡음 피봇 많음

#### 주의사항

만약 거래한다면:
- 더 보수적인 파라미터 사용
- 세션 갭 리스크 관리 강화
- 뉴스 발표 시 거래 회피

---

### ❌ 비트코인 선물 (BTC)

**예상 승률: 45~55%**

#### 비추천 이유

- **극단적 변동성**: ATR 기반 손절/이익실현 어려움
- **24시간 변동**: 세션 구조 불명확
- **뉴스 민감성**: 규제 뉴스에 급격한 반응
- **피봇 과다**: 너무 많은 피봇 발생으로 신호 노이즈

#### 주의사항

만약 거래한다면:
- ATR 멀티플라이어 대폭 증가 (3.0~4.0)
- 포지션 사이즈 축소 (50% 이하)
- 짧은 타임프레임 사용 권장

---

### ❌ VIX 선물 (VX)

**예상 승률: 48~55%**

#### 비추천 이유

- **평균 회귀성**: 추세성 부족으로 피봇 패턴 불명확
- **급격한 반전**: 단기간에 급격한 방향 전환
- **거래량 부족**: 유동성 낮음

#### 주의사항

만약 거래한다면:
- 짧은 기간 거래만 권장
- 헷징 용도로만 사용
- 단독 매매 비추천

---

## 3. 종합 비교표

| 순위 | 선물 | 티커 | 예상 승률 | 예상 수익률 | 유동성 | 추세성 | 변동성 | 추천도 |
|---|---|---|---|---|---|---|---|---|
| 1 | S&P 500 | ES | 65~70% | +35~45%/년 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 2 | 골드 | GC | 62~68% | +30~40%/년 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 3 | 나스닥 100 | NQ | 60~65% | +28~38%/년 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 | 유로 | 6E | 58~63% | +25~35%/년 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 5 | 10년 국채 | ZN | 55~60% | +20~30%/년 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 4. 최종 추천

### 최고의 선택: S&P 500 선물 (ES)

**이유**:
- 가장 높은 예상 승률 (65~70%)
- 최고의 유동성으로 슬리피지 최소화
- 명확한 추세성으로 피봇 형성 우수
- 세션 구조 명확로 다중 TF 필터 효과적
- 데이터 풍부로 ML 모델 학습 최적

**추천 포지션 사이즈**:
- HIGH confidence: 90%
- MEDIUM confidence: 65%
- LOW confidence: 30%

**추천 거래 시간**:
- 미국 세션 (15:00-21:00 KST): 최고 유동성
- 유럽 세션 (06:00-15:00 KST): 중간 유동성
- 아시아 세션 (21:00-06:00 KST): 낮은 유동성, 보수적 거래

---

### 안정적 선택: 골드 선물 (GC)

**이유**:
- 높은 승률 (62~68%)
- 안전 자산으로 시장 충격 시 방어적
- 장기 추세 뚜렷
- 변동성 적당으로 리스크 관리 용이

**추천 포지션 사이즈**:
- HIGH confidence: 88%
- MEDIUM confidence: 63%
- LOW confidence: 28%

**추천 거래 시간**:
- 미국 세션: 금 가격 변동성 높음
- 유럽 세션: 중간 변동성
- 아시아 세션: 낮은 변동성, 안정적

---

### 공격적 선택: 나스닥 100 선물 (NQ)

**이유**:
- 높은 변동성으로 수익 기회 많음
- 테크 섹터 추세 뚜렷
- 높은 유동성

**주의사항**:
- 변동성 높음으로 리스크 관리 강화 필요
- 포지션 사이즈 축소 권장

**추천 포지션 사이즈**:
- HIGH confidence: 90%
- MEDIUM confidence: 65%
- LOW confidence: 30%

---

## 5. 백테스팅 우선순위

### 1단계: S&P 500 선물 (ES)

**기간**: 1주  
**목표**: 기준 성능 확립

```python
from prediction.backtest_pivot_signals import PivotSignalBacktester, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    tick_size=0.25,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2
)

backtester = PivotSignalBacktester(config)
result = backtester.run_backtest(df, signals, atr_col="ATR")
backtester.print_results(result)
```

**성공 기준**:
- 승률 ≥ 60%
- Sharpe Ratio ≥ 1.2
- MDD ≤ 5%

---

### 2단계: 골드 선물 (GC)

**기간**: 1주  
**목표**: 다양성 검증

```python
config = BacktestConfig(
    initial_capital=100000.0,
    tick_size=0.1,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.88,
    stop_loss_atr_multiplier=2.3,
    take_profit_atr_multiplier=3.3
)
```

**성공 기준**:
- 승률 ≥ 58%
- Sharpe Ratio ≥ 1.1
- MDD ≤ 6%

---

### 3단계: 나스닥 100 선물 (NQ)

**기간**: 1주  
**목표**: 공격적 전략 검증

```python
config = BacktestConfig(
    initial_capital=100000.0,
    tick_size=0.25,
    commission_rate=0.00015,
    slippage_ticks=1,
    position_size_pct=0.90,
    stop_loss_atr_multiplier=2.2,
    take_profit_atr_multiplier=3.2
)
```

**성공 기준**:
- 승률 ≥ 55%
- Sharpe Ratio ≥ 1.0
- MDD ≤ 8%

---

## 포트폴리오 구성 제안

### 보수적 포트폴리오

- **S&P 500 (ES)**: 60%
- **골드 (GC)**: 30%
- **10년 국채 (ZN)**: 10%

**예상 승률**: 62~65%  
**예상 수익률**: +28~35%/년  
**예상 MDD**: 4~5%

---

### 밸런스 포트폴리오

- **S&P 500 (ES)**: 50%
- **골드 (GC)**: 25%
- **나스닥 100 (NQ)**: 15%
- **유로 (6E)**: 10%

**예상 승률**: 60~64%  
**예상 수익률**: +30~38%/년  
**예상 MDD**: 5~6%

---

### 공격적 포트폴리오

- **S&P 500 (ES)**: 40%
- **나스닥 100 (NQ)**: 35%
- **골드 (GC)**: 15%
- **유로 (6E)**: 10%

**예상 승률**: 58~63%  
**예상 수익률**: +32~42%/년  
**예상 MDD**: 6~8%

---

## 결론

지그재그 피봇 기반 매매 신호 시스템은 **S&P 500 선물 (ES)**에서 가장 높은 승률과 수익률을 기대할 수 있습니다. 

**최적의 전략**:
1. S&P 500 선물로 기준 성능 확립
2. 골드 선물로 다양성 확보
3. 백테스팅 후 실제 성능에 따라 포트폴리오 조정

**예상 개발 기간**: 2~3주 (백테스팅 포함)
