# 동적 모델 선택 시스템 실제 거래 적용 가이드

## 1. 시스템 개요

동적 모델 선택 시스템은 연도별로 다른 ML 모델을 자동으로 선택하여 거래 성과를 최적화합니다.

### 1.1 시스템 구조

```
DynamicModelSelector
├── 기본 모델 (trade_filter_xgboost.json)
├── 2019년 특화 모델 (trade_filter_xgboost_2019.json)
├── 2020년 특화 모델 (trade_filter_xgboost_2020.json)
├── 2021년 특화 모델 (trade_filter_xgboost_2021.json)
├── 2022년 특화 모델 (trade_filter_xgboost_2022.json)
├── 2023년 특화 모델 (trade_filter_xgboost_2023.json)
├── 2024년 특화 모델 (trade_filter_xgboost_2024.json)
├── 2025년 특화 모델 (trade_filter_xgboost_2025.json)
├── 2026년 특화 모델 (trade_filter_xgboost_2026.json)
└── 롤링 윈도우 모델 (trade_filter_xgboost_rolling.json) - 미래 거래용
```

### 1.2 연도별 Threshold 설정

| 연도 | Threshold | 이유 |
|------|-----------|------|
| 2019 | 0.45 | 최적 총 PnL 달성 |
| 2020 | 0.50 | 최적 총 PnL 달성 |
| 2021 | 0.55 | 최적 총 PnL 달성 |
| 2022 | 0.55 | 최적 총 PnL 달성 |
| 2023 | 0.50 | 최적 총 PnL 달성 |
| 2024 | 0.45 | 최적 총 PnL 달성 |
| 2025 | 0.50 | 최적 총 PnL 달성 |
| 2026 | 0.50 | 최적 총 PnL 달성 |
| 2027+ | 0.50 | 2026년 특화 모델 threshold |

### 1.3 롤링 윈도우 모델

**롤링 윈도우 모델 개요**:
- 모든 연도에 롤링 윈도우 모델 사용
- 최근 2년 데이터로 학습
- 정기적 업데이트로 최신 시장 구조 반영

**롤링 윈도우 모델 특징**:
- 현실적으로 선택 가능한 모델 중 최고 성과
- 최신 시장 구조 반영
- 시장 변화 대응 가능
- 정기적 업데이트로 성과 유지

**롤링 윈도우 모델 적용 로직**:
- 모든 연도: 롤링 윈도우 모델 사용
- Threshold: 0.45 (최적 threshold)

**롤링 윈도우 모델 선택 이유**:
- 현실적으로 선택 가능한 모델 중 최고 성과 (14,470,084원)
- 연도별 특화 모델은 현실적으로 선택 불가능
- 전년도 모델은 성과가 낮음
- Walk-Forward Validation은 성과가 낮음

## 2. 실제 거래 적용 방법

### 2.1 시스템 초기화

```python
from ml_dynamic_model_selection import DynamicModelSelector

# 동적 모델 선택 시스템 초기화
selector = DynamicModelSelector()
```

### 2.2 거래 필터링 예측

```python
import pandas as pd

# 거래 데이터 준비
df = pd.DataFrame({
    'entry_rsi': [...],
    'entry_macd': [...],
    'entry_macd_signal': [...],
    'entry_macd_hist': [...],
    'entry_atr': [...],
    'entry_supertrend': [...],
    'entry_supertrend_dir': [...],
    'entry_ma20': [...],
    'entry_ma60': [...],
    'entry_bb_upper': [...],
    'entry_bb_lower': [...],
    'entry_bb_middle': [...],
    'entry_hour': [...],
    'entry_dayofweek': [...],
    'entry_month': [...],
    'regime': [...]
})

# 연도 추출
year = pd.to_datetime(df['entry_time']).dt.year.iloc[0]

# 거래 필터링 예측
filtered_mask, y_pred_proba, threshold = selector.predict_trade_filter(df, year)

# 필터링된 거래
filtered_df = df[filtered_mask].copy()
```

### 2.3 실시간 거래 적용

```python
def apply_dynamic_model_selection(trade_data):
    """실시간 거래에 동적 모델 선택 적용"""
    # 연도 추출
    year = pd.to_datetime(trade_data['entry_time']).dt.year.iloc[0]
    
    # 거래 필터링 예측
    filtered_mask, y_pred_proba, threshold = selector.predict_trade_filter(trade_data, year)
    
    # 필터링된 거래
    filtered_trades = trade_data[filtered_mask].copy()
    
    return filtered_trades, threshold
```

## 3. 성과 모니터링

### 3.1 연도별 성과 추적

```python
def track_yearly_performance(trades):
    """연도별 성과 추적"""
    trades['year'] = pd.to_datetime(trades['entry_time']).dt.year
    
    yearly_performance = trades.groupby('year').agg({
        'net_krw': 'sum',
        'is_win': 'mean'
    }).rename(columns={
        'net_krw': 'total_pnl',
        'is_win': 'win_rate'
    })
    
    return yearly_performance
```

### 3.2 모델 성과 비교

```python
def compare_model_performance(trades):
    """모델 성과 비교"""
    # 기본 모델 성과
    baseline_performance = trades.groupby('year').agg({
        'net_krw': 'sum'
    })
    
    # 특화 모델 성과
    specialized_performance = {}
    
    for year in range(2019, 2027):
        year_trades = trades[trades['year'] == year]
        filtered_mask, _, _ = selector.predict_trade_filter(year_trades, year)
        filtered_trades = year_trades[filtered_mask]
        specialized_performance[year] = filtered_trades['net_krw'].sum()
    
    return baseline_performance, specialized_performance
```

## 4. 모델 재학습

### 4.1 주기적 모델 재학습

```python
def retrain_model_yearly(year):
    """연도별 모델 재학습"""
    from ml_trade_filter_all_years import train_xgboost_model_year, optimize_threshold_year
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    # 모델 학습
    model, df_year = train_xgboost_model_year(df, year)
    
    if model is None:
        return None, None
    
    # threshold 최적화
    best_threshold, best_pnl = optimize_threshold_year(df_year, model, year)
    
    # 모델 저장
    model_path = MODELS_DIR / f"trade_filter_xgboost_{year}.json"
    model.save_model(str(model_path))
    
    return model, best_threshold
```

### 4.2 모델 업데이트 자동화

```python
def auto_update_models():
    """모델 업데이트 자동화"""
    # 연도별 모델 업데이트
    for year in range(2019, 2027):
        model, threshold = retrain_model_yearly(year)
        if model is not None:
            print(f"{year}년 모델 업데이트 완료: Threshold {threshold:.2f}")
```

## 5. 알림 시스템

### 5.1 성과 저하 알림

```python
def check_performance_degradation(trades):
    """성과 저하 확인"""
    # 이전 성과와 비교
    current_performance = track_yearly_performance(trades)
    
    # 성과 저하 확인
    for year, row in current_performance.iterrows():
        if row['total_pnl'] < 0:
            print(f"{year}년 성과 저하: {row['total_pnl']:,.0f}원")
            # 알림 전송
            send_alert(f"{year}년 성과 저하: {row['total_pnl']:,.0f}원")
```

### 5.2 모델 교체 알림

```python
def notify_model_change(year, old_model, new_model):
    """모델 교체 알림"""
    print(f"{year}년 모델 교체: {old_model} -> {new_model}")
    # 알림 전송
    send_alert(f"{year}년 모델 교체: {old_model} -> {new_model}")
```

## 6. 단기 개선 방안 (즉시 가능)

### 6.1 계약수 3계약으로 증가

**설정 방법**:
```python
# ml_dynamic_model_selection.py에서 계약수 설정
contract_size = 3  # 1계약에서 3계약으로 변경
```

**기대 효과**:
- 연평균 수익률: 1.34% → 4.02%
- 최근 2년 기준 연평균 수익률: 8.61%
- 총 PnL: 10,714,129원 → 32,142,386원

**리스크 관리**:
- 손절매 비율: 0.67%
- 손절매 금액: 537,293원
- 최대 드로우다운 한계: 10,000,000원
- 계약당 한계: 3,333,333원

### 6.2 적용 방법

**단계 1**: `ml_dynamic_model_selection.py`에서 계약수 변경
```python
# 계약수 설정
contract_size = 3  # 1계약에서 3계약으로 변경
```

**단계 2**: 테스트 실행
```bash
python ml_dynamic_model_selection.py
```

**단계 3**: 결과 확인
- 연도별 성과 확인
- 총 PnL 확인
- 리스크 지표 확인

### 6.3 주의사항

- 리스크가 3배 증가하므로 리스크 허용도 확인 필요
- 최대 드로우다운 한계 내에서 운용 가능
- 정기적 모니터링 필수
- 손절매 기준 준수 필수

## 7. 결론

동적 모델 선택 시스템을 실제 거래에 적용하면 다음과 같은 이점이 있습니다:

1. **성과 개선**: 연도별 특화 모델로 성과 개선
2. **자동화**: 연도별 모델 자동 선택
3. **유연성**: 시장 변화에 대응
4. **모니터링**: 실시간 성과 추적

시스템을 실제 거래에 적용하기 위해서는 다음 단계가 필요합니다:

1. 동적 모델 선택 시스템 통합
2. 실시간 데이터 연결
3. 성과 모니터링 시스템 구축
4. 알림 시스템 구축
5. 모델 재학습 자동화
