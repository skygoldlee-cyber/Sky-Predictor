# long_or_flat_strategy.py 보완 사항

## 현재 상태

### 전략 특징
- MA20/60 + ADX 기반 레짐 감지
- 롱-또는-플랫 전략
- 당일 청산 (08:45 진입, 15:45 청산)
- 하락장 방어 효과 (MA20 < MA60일 때 플랫)

### 백테스트 성과
- 거래수: 179
- 승률: 55.87%
- 총 손익: +66,911,035 원
- 기대값: 1.50 pt/거래
- Sharpe: 1.296
- Max Drawdown: -35,534,659 원

---

## 보완 우선순위

### 1. 자동화된 주문 시스템 연동 (고우선순위)

#### 현재 문제점
- 신호만 제공, 수동 진입/청산 필요
- 매일 08:45, 15:45에 수동 주문 필요
- 실수 가능성 높음

#### 개선 방안
- **HTS API 연동**: 자동 매수/매도 주문
- **스케줄러**: Windows Task Scheduler로 자동 실행
- **주문 확인**: 주문 체결 여부 실시간 확인
- **예외 처리**: 주문 실패 시 알림 및 재시도

#### 구현 예시
```python
def execute_order(signal: int, config: dict):
    """자동 주문 실행"""
    if signal == 1:
        # 08:45 시가 매수
        place_buy_order()
        # 15:45 종가 매도 예약
        schedule_sell_order()
```

---

### 2. 실시간 모니터링 및 알림 시스템 (고우선순위)

#### 현재 문제점
- 장 중 상황 모니터링 불가
- 급격한 하락 시 대응 불가
- 청산 실패 시 알림 없음

#### 개선 방안
- **텔레그램 봇**: 신호, 진입, 청산 알림
- **실시간 모니터링**: 장 중 포지션 상태 확인
- **손절 알림**: 일정 손실률 도달 시 경고
- **시스템 상태**: 데이터 수집, 주문 상태 모니터링

#### 구현 예시
```python
def send_telegram_alert(message: str):
    """텔레그램 알림 전송"""
    bot.send_message(chat_id, message)
```

---

### 3. 손절/익절 로직 추가 (중우선순위)

#### 현재 문제점
- 당일 청산만으로 급격한 하락 대응 불가
- 손절 없이 끝까지 보유
- 변동성 높은 날 대응 부족

#### 개선 방안
- **손절 라인**: -2pt, -3pt 등 손절 설정
- **익절 라인**: +3pt, +5pt 등 익절 설정
- **트레일링 스탑**: 이익 보호를 위한 트레일링 스탑
- **시간 기반 청산**: 15:00 이후 청산 가속

#### 구현 예시
```python
def check_exit_conditions(current_pnl: float, config: dict):
    """청산 조건 확인"""
    if current_pnl <= config['stop_loss']:
        return 'stop_loss'
    elif current_pnl >= config['take_profit']:
        return 'take_profit'
    elif current_time >= '15:00':
        return 'time_exit'
    return None
```

---

### 4. 포지션 사이징 (중우선순위)

#### 현재 문제점
- 고정 수량으로 진입
- 리스크 관리 부족
- 자본 효율성 낮음

#### 개선 방안
- **고정 금액**: 매일 일정 금액 투자
- **리스크 기반**: Max Drawdown 기반 사이징
- **Kelly Criterion**: 기대값 기반 최적 사이징
- **변동성 기반**: ATR 기반 사이징

#### 구현 예시
```python
def calculate_position_size(capital: float, risk_per_trade: float, atr: float):
    """포지션 사이징 계산"""
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / (atr * 2)
    return position_size
```

---

### 5. 다양한 리스크 관리 기능 (중우선순위)

#### 현재 문제점
- Max Drawdown만으로 리스크 관리 부족
- 연속 손실 시 대응 부족
- 과도한 거래 방지 기능 없음

#### 개선 방안
- **일일 손실 한도**: 하루 최대 손실 설정
- **연속 손실 한도**: N일 연속 손실 시 휴식
- **거래 빈도 제한**: 과도한 거래 방지
- **변동성 필터**: 변동성 너무 높은 날 스킵

#### 구현 예시
```python
def check_risk_limits(trade_history: list, config: dict):
    """리스크 한도 확인"""
    daily_loss = calculate_daily_loss(trade_history)
    consecutive_losses = count_consecutive_losses(trade_history)
    
    if daily_loss > config['max_daily_loss']:
        return False, '일일 손실 한도 초과'
    if consecutive_losses >= config['max_consecutive_losses']:
        return False, '연속 손실 한도 초과'
    return True, 'OK'
```

---

### 6. 백테스트 결과 시각화 (저우선순위)

#### 현재 문제점
- 텍스트로만 결과 출력
- 추이 파악 어려움
- 성과 비교 불편

#### 개선 방안
- **그래프 시각화**: 누적 수익, Drawdown 그래프
- **월별 성과**: 월별 수익/손실 차트
- **승률 분석**: 기간별 승률 변화
- **리스크 메트릭**: Sharpe, Sortino 등 시각화

#### 구현 예시
```python
def plot_backtest_results(result: BacktestResult):
    """백테스트 결과 시각화"""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(result.cumulative_pnl)
    plt.title('누적 수익')
    plt.subplot(2, 1, 2)
    plt.plot(result.drawdown)
    plt.title('Drawdown')
    plt.show()
```

---

### 7. 파라미터 최적화 자동화 (저우선순위)

#### 현재 문제점
- 수동으로 파라미터 조정
- 최적 파라미터 찾기 어려움
- 시장 상황 변화 대응 부족

#### 개선 방안
- **Optuna 통합**: 자동 파라미터 최적화
- **Walk-Forward**: Walk-Forward 분석
- **앙상블**: 여러 파라미터 조합 앙상블
- **동적 파라미터**: 시장 상황에 따른 파라미터 조정

#### 구현 예시
```python
def optimize_parameters(df: pd.DataFrame, config: dict):
    """파라미터 최적화"""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, config), n_trials=100)
    return study.best_params
```

---

### 8. 로그 및 보고서 개선 (저우선순위)

#### 현재 문제점
- 기본 로그만 저장
- 상세 분석 어려움
- 이력 관리 부족

#### 개선 방안
- **상세 로그**: 진입/청산 시간, 가격, 수익 상세 기록
- **일일 보고서**: 매일 성과 요약 보고서
- **주간/월간 보고서**: 주간/월간 성과 분석
- **이상 징후 탐지**: 비정상적인 패턴 탐지

#### 구현 예시
```python
def generate_daily_report(trade: dict, config: dict):
    """일일 보고서 생성"""
    report = {
        'date': trade['date'],
        'signal': trade['signal'],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'pnl': trade['pnl'],
        'ma20': trade['ma20'],
        'ma60': trade['ma60'],
        'adx': trade['adx']
    }
    save_report(report, config)
```

---

### 9. 데이터 수집 자동화 (저우선순위)

#### 현재 문제점
- 수동으로 데이터 수집 필요
- 데이터 누락 가능성
- 데이터 품질 확인 필요

#### 개선 방안
- **자동 수집**: 매일 자동으로 데이터 수집
- **데이터 품질 확인**: 데이터 누락/이상치 확인
- **백업**: 데이터 백업 시스템
- **복구**: 데이터 손상 시 복구 기능

#### 구현 예시
```python
def auto_collect_data(config: dict):
    """자동 데이터 수집"""
    if not check_data_quality(config):
        collect_data(config)
        validate_data(config)
```

---

### 10. 예외 상황 처리 (저우선순위)

#### 현재 문제점
- 예외 상황 대응 부족
- 에러 발생 시 중단
- 복구 기능 없음

#### 개선 방안
- **예외 처리**: 다양한 예외 상황 처리
- **재시도 로직**: 일시적 오류 시 재시도
- **롤백**: 오류 발생 시 롤백
- **알림**: 예외 상황 발생 시 알림

#### 구현 예시
```python
def safe_execute(func, max_retries=3):
    """안전한 실행"""
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                send_alert(f"실패: {e}")
                raise
            time.sleep(2 ** i)
```

---

## 구현 순서 추천

### 단계 1: 기본 자동화 (1-2주)
1. 자동화된 주문 시스템 연동
2. 실시간 모니터링 및 알림 시스템

### 단계 2: 리스크 관리 (2-3주)
3. 손절/익절 로직 추가
4. 포지션 사이징
5. 다양한 리스크 관리 기능

### 단계 3: 분석 및 최적화 (2-3주)
6. 백테스트 결과 시각화
7. 파라미터 최적화 자동화

### 단계 4: 시스템 안정화 (1-2주)
8. 로그 및 보고서 개선
9. 데이터 수집 자동화
10. 예외 상황 처리

---

## 결론

현재 `long_or_flat_strategy.py`는 기본적인 신호 생성 기능을 제공하지만, 실제 자동화된 트레이딩 시스템으로 운영하기 위해서는 위 보완 사항들이 필요합니다.

우선순위별로 단계적으로 구현하여 시스템을 안정적으로 개선하는 것을 추천합니다.
