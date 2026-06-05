# 적응형 지표 가이드

SkyPredictor의 적응형 지표(Adaptive ZigZag, Adaptive SuperTrend, AdaptiveParamEngine)를 통합한 가이드입니다.

---

## 개요

### Adaptive ZigZag
Adaptive ZigZag는 ATR(Average True Range) 기반 동적 임계값과 ER(Efficiency Ratio) 적응형 필터링을 사용하여 스윙 고점/저점을 탐지하는 기술적 지표입니다. 시장 변동성에 따라 임계값을 자동 조정하여 추세 반전을 정확하게 감지합니다.

### Adaptive SuperTrend
ATR 기반 추세 추종 지표로, 변동성에 따라 상승/하락선을 동적으로 조정합니다.

### AdaptiveParamEngine
외부 레짐 분류기 없이, `AdaptiveZigZag` 내부 버퍼만으로 장중 피봇 탐색 파라미터를 실시간 자동 조정하는 자기완결형 엔진입니다.

---

## Adaptive ZigZag 알고리즘

### 핵심 알고리즘 구조

#### 메인 처리 흐름 (`update` 메서드)
```
1. OHLC 데이터 수집
2. True Range & ATR 계산
3. ATR 변화 모니터링 (급격 변동 감지)
4. 적응형 임계값 계산 (ATR × 배율)
5. Pending Confirmation 처리
6. 방향 결정/전환
7. 스윙 추가 및 클러스터링
8. 상태 업데이트
```

#### 데이터 구조
```python
@dataclass
class ZigzagConfig:
    """ZigZag 설정"""
    atr_period: int = 14              # ATR 기간
    atr_multiplier: float = 2.0       # ATR 배수
    min_wave_bars: int = 7           # 최소 파동 봉 수
    confirmation_bars: int = 3       # 확인 봉 수
    er_threshold: float = 0.6         # Efficiency Ratio 임계값
    use_atr_based_filtering: bool = True  # ATR 기반 필터링 사용
```

### ATR 기반 필터링

#### min_wave_atr_ratio
- 시간대별 최소 파동 ATR 비율
- 장중 변동성에 따라 동적 조정
- `session_min_wave_atr_ratio_table` 설정

#### 합의도(Consensus) 필터링
- 다른 지표와의 일치 확인
- SuperTrend와의 방향 일치
- 피봇 후보 필터링

---

## Adaptive SuperTrend

### 알고리즘
```python
@dataclass
class SuperTrendConfig:
    """SuperTrend 설정"""
    atr_period: int = 10
    atr_multiplier: float = 3.0
    er_threshold: float = 0.6
```

### 특징
- ATR 기반 상승/하락선
- 변동성에 따른 동적 조정
- 추세 반전 신호 제공

---

## AdaptiveParamEngine

### 특징
- **자기완결형**: 외부 의존성 없이 ZigZag 내부 버퍼만으로 동작
- **실시간 조정**: 매 봉마다 파라미터 동적 조정
- **레짐 기반**: 시장 상태에 따른 최적 파라미터 프로파일 적용
- **EMA 스무딩**: 파라미터 급격한 변동 방지 (α=0.15)
- **피드백 루프**: 피봇 밀도에 따른 미세 조정

### 통합 방법

#### 1. AdaptiveZigZag.__init__()에 엔진 초기화 추가
```python
# ── [자기완결형 적응 엔진] ─────────────────────────────────────
self._adaptive_engine = None
try:
    from .adaptive_param_engine import AdaptiveParamEngine
    self._adaptive_engine = AdaptiveParamEngine(self.config)
    _logger.info("[AdaptiveZigZag] 적응형 파라미터 엔진 활성화")
except Exception as e:
    _logger.warning("[AdaptiveZigZag] 적응형 파라미터 엔진 초기화 실패: %s", e)
```

#### 2. _get_runtime_params()에 엔진 기반 로직 추가
```python
def _get_runtime_params(self) -> ZigzagRuntimeParams:
    """실시간 파라미터 계산"""
    if self._adaptive_engine:
        return self._adaptive_engine.get_params(self)
    return self._default_params
```

---

## 설정 가이드

### config.json 설정
```json
{
  "adaptive_indicator": {
    "enabled": true,
    "zigzag": {
      "atr_period": 14,
      "atr_multiplier": 2.0,
      "min_wave_bars": 7,
      "confirmation_bars": 3,
      "er_threshold": 0.6,
      "use_atr_based_filtering": true
    },
    "supertrend": {
      "atr_period": 10,
      "atr_multiplier": 3.0,
      "er_threshold": 0.6
    },
    "session_min_wave_atr_ratio_table": [
      ["09:00", "09:30", 0.8],
      ["09:30", "10:30", 1.2],
      ["10:30", "15:00", 1.0]
    ]
  }
}
```

### 시간대별 파라미터
- **개장 직후 (09:00-09:30)**: 낮은 ATR 비율 (불안정성 감소)
- **활성 시간 (09:30-10:30)**: 높은 ATR 비율 (활발한 움직임)
- **장중 (10:30-15:00)**: 표준 ATR 비율

---

## 피봇 확정 알고리즘

### 현행 알고리즘 요약
1. **후보 등록**: 임계값 돌파 시 후보 등록
2. **확인 기간**: `confirmation_bars` 동안 유지 확인
3. **합의도 검사**: 다른 지표와의 일치 확인
4. **피봇 확정**: 조건 만족 시 피봇으로 확정
5. **클러스터링**: 인접 피봇 그룹화

### 피봇 탐지 파라미터
- **atr_period**: ATR 계산 기간 (기본 14)
- **er_period**: Efficiency Ratio 계산 기간 (기본 10)
- **atr_multiplier**: ATR 배수 (기본 1.5)
- **atr_multiplier_min**: 최소 ATR 배수 (기본 1.0)
- **atr_multiplier_max**: 최대 ATR 배수 (기본 4.0)
- **min_wave_bars**: 최소 파동 봉 수 (기본 1)
- **confirmation_bars**: 확인 봉 수 (기본 2)
- **confirmation_bars_ranging**: 횡보 구간 확인 봉 수 (기본 1)
- **confirmation_bars_unknown**: 미확인 구간 확인 봉 수 (기본 1)
- **min_wave_pct**: 최소 파동 비율 (기본 0.0)
- **pivot_threshold_min_pct**: 피봇 임계값 최소 비율 (기본 0.3%)
- **pivot_threshold_max_pct**: 피봇 임계값 최대 비율 (기본 3.0%)
- **major_swing_ratio**: 메이저 스윙 비율 (기본 2.0)
- **max_swings**: 최대 스윙 수 (기본 50)
- **cluster_tolerance_pct**: 클러스터 허용 비율 (기본 0.3%)
- **structure_lookback_swings**: 구조 분석용 최근 스윙 수 (기본 8)
- **structure_points**: 구조 판단 최소 고점/저점 수 (기본 3)
- **structure_majority_threshold**: 구조 다수결 임계값 (기본 0.7)

### 피봇 확정 로직
1. **방향 전환 감지**: 현재 방향과 반대 방향으로 가격이 임계값을 돌파
2. **후보 등록**: `_pending_confirm`에 후보 정보 저장
3. **확인 기간**: `confirmation_bars` 동안 후보 유지 확인
4. **피봇 확정**: 확인 기간 동안 후보가 유지되면 피봇으로 확정
5. **클러스터링**: 인접한 피봇들을 그룹화하여 대표 피봇 선정
6. **교번 보장**: `_enforce_hl_alternation`으로 연속 HIGH/LOW 피봇 방지
7. **구조 분석**: `_analyze_structure`로 상승/하락/횡보 판정

### 피봇 기반 매매 전략
상세한 피봇 기반 매매 전략은 [PIVOT_BASED_TRADING_STRATEGY.md](PIVOT_BASED_TRADING_STRATEGY.md)를 참조하세요.

### 관련 문서
- [multi_timeframe_zigzag.md](multi_timeframe_zigzag.md) - 다중 시간프레임 ZigZag 구현 가이드
- [OVERSEAS_FUTURES_ADAPTIVE_ZIGZAG_APPLICABILITY.md](OVERSEAS_FUTURES_ADAPTIVE_ZIGZAG_APPLICABILITY.md) - 해외선물 적용 가능성 검토
- [zigzag_tuning.md](zigzag_tuning.md) - ZigZag 파라미터 튜닝 가이드
- [regime_zigzag_tuning.md](regime_zigzag_tuning.md) - 레짐 기반 ZigZag 튜닝

### 보완 및 개선점

#### 1. confirmation_bars 동적 조절
- **현재**: 고정된 `confirmation_bars` 값 사용
- **개선**: 레짐(상승/하락/횡보)에 따라 가변 조절
  - 추세장: 낮은 값 (빠른 확정)
  - 횡보장: 높은 값 (노이즈 필터링)
- **구현**: `_calc_confirmation_bars()` 함수 추가

#### 2. 합의도 필터링 강화
- **현재**: 단순 방향 일치 확인
- **개선**: 지표별 가중치 부여 및 종합 점수 계산
  - SuperTrend 방향 일치: 가중치 0.4
  - 구조 판정 일치: 가중치 0.3
  - 오더북 불균형: 가중치 0.3
- **구현**: `_calc_consensus_score()` 함수 추가

#### 3. 피봇 품질 점수화
- **현재**: 메이저 스윙 분류 (`major_swing_ratio`)
- **개선**: 피봇 품질 점수 계산
  - 파동 크기 점수
  - 구조 일치 점수
  - 지지/저항 거리 점수
- **구현**: `_calc_pivot_quality_score()` 함수 추가

#### 4. 시간대별 파라미터 테이블
- **현재**: 전역 파라미터 사용
- **개선**: 시간대별 파라미터 테이블 적용
  - 개장 직후 (09:00-09:30): 낮은 임계값
  - 활성 시간 (09:30-10:30): 높은 임계값
  - 장중 (10:30-15:00): 표준 임계값
- **구현**: `session_param_table` 설정 추가

#### 5. 피드백 루프 기반 파라미터 조정
- **현재**: 고정 파라미터
- **개선**: 피봇 확정/취소 이력 기반 파라미터 조정
  - 확정률이 낮으면 `atr_multiplier` 증가
  - 취소율이 높으면 `confirmation_bars` 증가
- **구현**: `AdaptiveParamEngine` 확장

#### 6. 다중 시간프레임 합의도
- **현재**: 단일 시간프레임 (분봉)
- **개선**: 다중 시간프레임 합의도 확인
  - 1분봉, 5분봉, 15분봉 피봇 합의
  - 모든 시간프레임 일치 시 높은 신뢰도 부여
- **구현**: `MultiTimeframeZigZag` 클래스 추가

---

## 성능 최적화

### 파라미터 튜닝
- ATR 기간: 10-20 사이 조정
- ATR 배수: 1.5-3.0 사이 조정
- 확인 봉 수: 2-5 사이 조정

### 레짐 기반 파라미터
- **추세 시**: 낮은 임계값, 빠른 반응
- **횡보 시**: 높은 임계값, 신호 필터링
- **고변동**: ATR 배수 증가

---

## 모니터링 및 디버깅

### 로그 확인
```bash
# ZigZag 로그
grep "\[ZZ\]" logs/prediction.log

# SuperTrend 로그
grep "\[ST\]" logs/prediction.log

# ParamEngine 로그
grep "\[PARAM\]" logs/prediction.log
```

### 성능 메트릭
- 피봇 탐지 정확도
- 거짓 신호 비율
- 평균 지연 시간
- 수익률 (백테스트)

---

## 문제 해결

### 1. 피봇 미탐지
**증상**: 명확한 반전점인데 피봇이 생성되지 않음
**해결**: 
- `min_wave_bars` 감소
- `atr_multiplier` 감소
- `er_threshold` 감소

### 2. 거짓 신호 증가
**증상**: 실제 반전 없이 피봇이 자주 생성됨
**해결**:
- `confirmation_bars` 증가
- `atr_multiplier` 증가
- 합의도 필터링 강화

### 3. 지연 시간 증가
**증상**: 피봇이 너무 늦게 생성됨
**해결**:
- `confirmation_bars` 감소
- ATR 기간 감소
- 합의도 필터링 완화
