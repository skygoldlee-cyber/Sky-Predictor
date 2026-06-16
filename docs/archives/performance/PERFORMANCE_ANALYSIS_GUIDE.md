# 성과 분석 고도화 가이드

> **상세 성과 분석 기능 사용 가이드**
> 작성일: 2026-04-26

---

## 개요

본 가이드는 고도화된 성과 분석 기능을 사용하여 거래 성과를 심층 분석하는 방법을 설명합니다. 기본 통계부터 고급 지표, 시각화, Excel 보고서 생성까지 포함합니다.

## 목차

1. 개요
2. 주요 기능
3. 분석 항목 상세
4. 사용 예시
5. 시각화 기능
6. 보고서 생성
7. 권장 사항

---

## 2. 주요 기능

### 2.1 기본 통계
- 총 거래 수
- 승리/패배 거래 수
- 승률
- 총 수익
- 평균 수익
- 최대 수익/손실
- 평균 보유 시간

### 2.2 시간대별 분석
- 오전 (09:00-11:30)
- 오후 (13:00-14:30)
- 장 마감 (14:30-15:30)
- 거래 수, 승률, 총 수익

### 2.3 요일별 분석
- 월~금 요일별 성과
- 거래 수, 승률, 총 수익

### 2.4 슬롯별 분석
- 슬롯 A/B/C별 성과
- 거래 수, 승률, 총 수익, 평균 보유 시간

### 2.5 신뢰도별 분석
- HIGH/MEDIUM/LOW 신뢰도별 성과
- 거래 수, 승률, 총 수익

### 2.6 청산 사유별 분석
- 목표수익, 손절, 강제청산, 역신호 등
- 거래 수, 승률, 총 수익

### 2.7 보유 기간 분석
- 단기 (0-5분), 중기 (5-15분), 장기 (15-30분), 초장기 (30분+)
- 거래 수, 승률, 평균 수익

### 2.8 리스크 메트릭
- **MDD (Maximum Drawdown)**: 최대 낙폭
- **Sharpe Ratio**: 리스크 조정 수익률
- **Sortino Ratio**: 하방 리스크 조정 수익률
- **Profit Factor**: 총 수익 / 총 손실

### 2.9 고급 성과 지표
- **Calmar Ratio**: 연간 수익률 / 최대 낙폭
- **Win/Loss Ratio**: 평균 승리 / 평균 패배
- **Expectancy**: 기대값 (거래당 평균 수익)
- **Risk of Ruin**: 파산 위험도
- **Recovery Factor**: 총 수익 / 최대 낙폭

### 2.10 포지션 사이징 분석
- 사이징 방법별 성과
- 총 리스크 노출
- 평균 거래당 리스크
- 평균 포지션 사이즈

---

## 3. 분석 항목 상세

### 3.1 리스크 메트릭 상세

#### MDD (Maximum Drawdown)
- **설명**: 고점에서 저점까지의 최대 낙폭
- **계산**: (누적 수익 - 최대 누적 수익) / 최대 누적 수익
- **해석**: 낮을수록 좋음 (보통 -10% 이하 권장)

#### Sharpe Ratio
- **설명**: 리스크 단위당 수익률
- **계산**: (평균 수익률 / 표준편차) * √252
- **해석**: 
  - 1.0 이상: 우수
  - 0.5~1.0: 양호
  - 0.5 미만: 개선 필요

#### Sortino Ratio
- **설명**: 하방 리스크만 고려한 수익률
- **계산**: (평균 수익률 / 하방 표준편차) * √252
- **해석**: Sharpe Ratio보다 보수적, 1.0 이상 권장

#### Profit Factor
- **설명**: 총 수익 대비 총 손실 비율
- **계산**: 총 수익 / 총 손실
- **해석**: 
  - 2.0 이상: 우수
  - 1.5~2.0: 양호
  - 1.5 미만: 개선 필요

### 3.2 고급 성과 지표 상세

#### Calmar Ratio
- **설명**: 연간 수익률 대비 최대 낙폭
- **계산**: 연간 수익률 / 최대 낙폭
- **해석**: 3.0 이상 권장

#### Win/Loss Ratio
- **설명**: 평균 승리 대비 평균 패배
- **계산**: 평균 승리 / 평균 패배
- **해석**: 2.0 이상 권장

#### Expectancy (기대값)
- **설명**: 거래당 평균 수익
- **계산**: (승률 × 평균 승리) - (패배율 × 평균 패배)
- **해석**: 양수여야 함, 클수록 좋음

#### Risk of Ruin (파산 위험도)
- **설명**: 자본이 고갈될 확률
- **계산**: ((1-승률)/승륟)^(초기자본/평균손실)
- **해석**: 
  - 0.1% 미만: 안전
  - 1% 미만: 양호
  - 1% 이상: 위험

#### Recovery Factor
- **설명**: 낙폭 회복 능력
- **계산**: 총 수익 / 최대 낙폭
- **해석**: 3.0 이상 권장

---

## 4. 사용 예시

### 4.1 기본 사용

```python
from prediction.performance_analyzer import PerformanceAnalyzer
from trading.state import TradeRecord

# 거래 데이터 로드
trades = load_trades()  # TradeRecord 리스트

# 분석기 초기화
analyzer = PerformanceAnalyzer()

# 리포트 생성
report = analyzer.generate_report(trades)

# 리포트 출력
analyzer.print_report(report)
```

### 4.2 시각화

```python
from pathlib import Path

# 자본 곡선 시각화
analyzer.plot_equity_curve(trades, save_path=Path("equity_curve.png"))

# 시간대별 성과 시각화
analyzer.plot_performance_by_time(trades, save_path=Path("performance_by_time.png"))
```

### 4.3 Excel 보고서 저장

```python
from pathlib import Path

# Excel 보고서 저장
analyzer.save_report_to_excel(report, Path("performance_report.xlsx"))
```

### 4.4 특정 분석 항목 확인

```python
# 기본 통계
print(report.basic_stats)
print(f"총 거래 수: {report.basic_stats['total_trades']}")
print(f"승률: {report.basic_stats['win_rate']:.2%}")

# 리스크 메트릭
print(report.risk_metrics)
print(f"MDD: {report.risk_metrics['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {report.risk_metrics['sharpe_ratio']:.2f}")

# 고급 지표
print(report.advanced_metrics)
print(f"Expectancy: {report.advanced_metrics['expectancy']:.2f}")
print(f"Risk of Ruin: {report.advanced_metrics['risk_of_ruin']:.2%}")

# 슬롯별 분석
print(report.slot_analysis)
for slot, stats in report.slot_analysis.items():
    print(f"{slot}: 승률 {stats['win_rate']:.2%}, 총 수익 {stats['total_profit']:.2f}pt")

# 사이징 분석
print(report.sizing_analysis)
sizing_methods = report.sizing_analysis['sizing_method_analysis']
for method, stats in sizing_methods.items():
    print(f"{method}: 승률 {stats['win_rate']:.2%}, 평균 사이즈 {stats['avg_position_size']:.2f}")
```

---

## 5. 시각화 기능

### 5.1 자본 곡선 (Equity Curve)

누적 수익의 시계열 그래프로 전체 성과 추이를 한눈에 파악할 수 있습니다.

**특징**:
- 누적 수익 추이
- 0선 표시 (손익 분기점)
- 날짜 축 포맷팅

**사용법**:
```python
analyzer.plot_equity_curve(trades, save_path="equity_curve.png")
```

### 5.2 시간대별 성과

4개의 서브플롯으로 시간대별 성과를 다각도로 분석합니다.

**서브플롯**:
1. 시간대별 거래 수
2. 시간대별 승률
3. 시간대별 평균 수익
4. 시간대별 총 수익

**사용법**:
```python
analyzer.plot_performance_by_time(trades, save_path="performance_by_time.png")
```

---

## 6. 보고서 생성

### 6.1 Excel 보고서

모든 분석 결과를 정형화된 Excel 파일로 저장합니다.

**포함 내용**:
- 기본 통계
- 리스크 메트릭
- 고급 성과 지표
- 슬롯별 분석
- 시간대별 분석
- 포지션 사이징 분석

**스타일**:
- 헤더 스타일 (파란색 배경, 굵은 글씨)
- 열 너비 자동 조정
- 가독성 높은 형식

**사용법**:
```python
from pathlib import Path

analyzer.save_report_to_excel(report, Path("performance_report.xlsx"))
```

### 6.2 의존성

시각화 및 Excel 저장을 위해 다음 패키지가 필요합니다:

```bash
pip install matplotlib seaborn openpyxl
```

패키지가 설치되지 않은 경우 해당 기능은 건너뛰고 경고 메시지가 출력됩니다.

---

## 7. 권장 사항

### 7.1 정기 분석

- **주간**: 기본 통계, 리스크 메트릭 확인
- **월간**: 시간대별/요일별/슬롯별 분석
- **분기별**: 고급 지표, 사이징 효과 분석

### 7.2 성과 기준

| 지표 | 우수 | 양호 | 개선 필요 |
|------|------|------|----------|
| 승률 | 60%+ | 50-60% | 50% 미만 |
| Sharpe Ratio | 1.0+ | 0.5-1.0 | 0.5 미만 |
| Sortino Ratio | 1.0+ | 0.5-1.0 | 0.5 미만 |
| Profit Factor | 2.0+ | 1.5-2.0 | 1.5 미만 |
| Calmar Ratio | 3.0+ | 1.5-3.0 | 1.5 미만 |
| Win/Loss Ratio | 2.0+ | 1.5-2.0 | 1.5 미만 |
| MDD | -10% 이상 | -10~-20% | -20% 미만 |
| Risk of Ruin | 0.1% 미만 | 0.1-1% | 1% 이상 |

### 7.3 개선 방향

#### 승률 낮을 때
- 진입 조건 강화
- 신뢰도 기준 상향
- 시간대 필터링

#### 리스크 높을 때
- 포지션 사이즈 축소
- 손절 타이트화
- 리스크 패리티 사이징

#### 수익성 낮을 때
- 목표 수익 조정
- 보유 기간 최적화
- 사이징 방법 변경

### 7.4 분석 체크리스트

- [ ] 기본 통계 확인 (승률, 수익)
- [ ] 리스크 메트릭 확인 (MDD, Sharpe)
- [ ] 고급 지표 확인 (Expectancy, Risk of Ruin)
- [ ] 시간대별 분석 (최적 시간대 파악)
- [ ] 슬롯별 분석 (슬롯 성과 확인)
- [ ] 사이징 분석 (사이징 효과 확인)
- [ ] 청산 사유 분석 (손실 원인 파악)
- [ ] 추세 모니터링 (성과 추이 확인)

---

## 8. 주의사항

### 8.1 데이터 충분성
- 최소 30거래 이상 권장
- 100거래 이상 시 신뢰도 높음

### 8.2 시장 상황 고려
- 과거 성과가 미래 보장 아님
- 시장 레짐 변화 주의
- 백테스트와 실시간 차이 확인

### 8.3 리스크 관리
- 단일 지표에 과도 의존 금지
- 복합 지표 활용
- 리스크 한도 준수

### 8.4 지표 해석
- Sharpe Ratio: 고변동 시 왜곡 가능
- MDD: 회복 기간 고려 필요
- Expectancy: 거래 빈도 고려 필요

---

## 9. 고급 활용

### 9.1 A/B 테스트

다른 파라미터나 전략 비교:

```python
# 전략 A
report_a = analyzer.generate_report(trades_a)

# 전략 B
report_b = analyzer.generate_report(trades_b)

# 비교
print(f"전략 A Sharpe: {report_a.risk_metrics['sharpe_ratio']:.2f}")
print(f"전략 B Sharpe: {report_b.risk_metrics['sharpe_ratio']:.2f}")
```

### 9.2 시계열 분석

월별/분기별 성과 추이 분석:

```python
# 월별 분할
trades_by_month = group_trades_by_month(trades)

for month, month_trades in trades_by_month.items():
    report = analyzer.generate_report(month_trades)
    print(f"{month}: 승률 {report.basic_stats['win_rate']:.2%}")
```

### 9.3 자동화

주간/월간 자동 리포트 생성:

```python
from pathlib import Path
from datetime import datetime

# 주간 리포트
date_str = datetime.now().strftime("%Y%m%d")
report_path = Path(f"reports/weekly_{date_str}.xlsx")
analyzer.save_report_to_excel(report, report_path)
```

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-04-26
