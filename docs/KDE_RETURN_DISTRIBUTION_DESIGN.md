# KDE 기반 수익률 분포 & 평균회귀 신호 설계

> 문서 ID: `DESIGN-KDE-RETURN-001`
> 작성일: 2026-07-19
> 상태: Phase 1 완료, Phase 2/3 진행 예정
> 목적: SkyPredictor 기존 아키텍처에 확률밀도함수(KDE)를 이용한 평균회귀 로직을 추가하는 설계

---

## 1. 개요

### 1.1 핵심 아이디어

최근 N개 수익률의 **확률밀도함수(PDF)**를 Kernel Density Estimation(KDE)으로 추정하고, 현재 수익률이 그 분포에서 얼마나 **꼬리 쪽**에 위치하는지(CDF 백분위)를 계산합니다. 극단적 꼬리(CDF < 하위 임계값 또는 CDF > 상위 임계값)에 도달했을 때, 추가 조건(ZigZag 구조, VWAP 위치, 추세 필터)을 충족하면 평균회귀 진입 신호를 발생시킵니다.

### 1.2 기존 아키텍처와의 관계

기존 SkyPredictor는 다음 인프라를 이미 보유하고 있습니다.

| 기존 컴포넌트 | 이미 제공하는 기능 | 새 로직에서의 역할 |
|--------------|-------------------|-------------------|
| `indicators/adaptive_zigzag.py` | ZigZag 고점/저점, 피봇 생명주기 | 평균회귀 진입의 가격 구조 확인 |
| `indicators/adaptive_supertrend.py` | ADX, ATR, ER, 추세 방향/강도 | 추세장 필터, 변동성 측정 |
| `prediction/mixins/tick_mixin.py` | VWAP 편차(`vwap_dev`) | 현재가가 VWAP 대비 어디에 있는지 판단 |
| `services/market_regime_classifier.py` | 시장 레짐 분류 | 추세/횡보 레짐 기반 전환 |
| `prediction/predictor.py` | 딥러닝 방향 예측 | 평균회귀 신호를 추가 예측 채널로 통합 가능 |
| `trading/gate.py`, `trading/pivot_gate.py` | 진입/청산 게이트 | 신호 → 주문 실행 연결 지점 |

새로 추가되는 부분은 **수익률 분포 추정기**와 **평균회귀 신호 생성기** 두 모듈뿐이며, 나머지는 기존 지표/게이트를 재사용합니다.

---

## 2. 시스템 구조

### 2.1 전체 데이터 플로우

```
실시간 1분/5분 OHLCV
    │
    ▼
┌─────────────────────────────────────┐
│  ReturnCalculationMixin (신규)       │
│  - 1분/5분/로그 수익률 계산          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  ReturnDistributionEstimator (신규)│
│  - 롤링 윈도우 버퍼                 │
│  - KDE 적합                         │
│  - PDF/CDF/Z-score/꼬리 확률 출력    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  MeanReversionSignalGenerator (신규)│
│  - CDF 임계값 진입 조건             │
│  - ZigZag/VWAP/ATR/ADX 종합 판단    │
│  - 신호 강도(weak/normal/strong)     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  기존 predictor / gate / telegram    │
│  - 신호를 기존 매매 파이프라인에     │
│    병합 또는 독립 채널로 운영        │
└─────────────────────────────────────┘
```

### 2.2 모듈 배치

```
indicators/
├── return_distribution.py          # KDE 추정기
├── mean_reversion_signal.py        # 신호 생성기
└── (기존) adaptive_supertrend.py
    (기존) adaptive_zigzag.py

prediction/
├── features/
│   └── features.py                   # pdf_* / cdf_* 피처 추가
└── mixins/
    └── kde_mixin.py (신규)          # Predictor에 KDE 채널 연동

trading/
└── gate.py / pivot_gate.py          # 신호 수용 인터페이스

config/
└── config.json                      # kde_mean_reversion 섹션 추가
```

---

## 3. 핵심 클래스 설계

### 3.1 `ReturnDistributionEstimator`

#### 책임
- 최근 수익률 샘플을 버퍼링
- KDE 적합 및 갱신
- 현재 수익률의 PDF, CDF, Z-score, 꼬리 확률 반환

#### 인터페이스

```python
class ReturnDistributionEstimator:
    def __init__(
        self,
        window: int = 3000,          # 롤링 윈도우 길이
        min_samples: int = 500,      # KDE 적합 최소 샘플 수
        bandwidth: str = "scott",    # "scott" | "silverman" | float
        decay: Optional[float] = None,  # None=uniform, float=지수감쇠 가중
    ) -> None: ...

    def update(self, return_value: float) -> None: ...

    def evaluate(self, return_value: float) -> DistributionMetrics: ...
```

#### 반환값 (`DistributionMetrics`)

| 필드 | 설명 | 용도 |
|------|------|------|
| `pdf` | 현재 수익률에서의 확률밀도값 | 분포 중심 vs 꼬리 판단 (낮을수록 드문 값) |
| `cdf` | 누적확률 (0~1) | 하위/상위 백분위 판단 |
| `z_score` | (r - μ) / σ | 추가 극단성 측정 |
| `left_tail_prob` | P(R ≤ r) | 과매도 측정 |
| `right_tail_prob` | P(R ≥ r) | 과매수 측정 |
| `median` | KDE 중앙값 | 평균회귀 목표 참고 |
| `is_ready` | KDE 적합 가능 여부 | 초기 warmup 방어 |

#### CDF 계산 방법

`scipy.stats.gaussian_kde`는 PDF만 직접 제공하므로, 실시간 성능을 위해 다음 중 하나를 선택합니다.

1. **Grid 기반 룩업 테이블** (권장): KDE를 미리 grid(예: 1,000~2,000 포인트)에 평가한 뒤 누적합으로 CDF를 만들어 보간.
2. **`scipy.integrate.quad`**: 정확도는 높으나 실시간 호출 시 부담. 백테스트용 또는 초기 검증용.
3. **statsmodels KDE**: `KDEUnivariate.cdf` 사용 가능하나 의존성 추가.

> **설계 결정**: 실시간 파이프라인에는 grid 기반 CDF, 백테스트/오프라인 검증에는 `quad` 또는 `statsmodels`를 병행 사용.

### 3.2 `MeanReversionSignalGenerator`

#### 책임
- KDE 지표와 기존 지표(ZigZag, VWAP, ADX 등)를 결합해 매매 신호 생성
- 신호 강도 및 억제 사유 기록

#### 인터페이스

```python
class MeanReversionSignalGenerator:
    def __init__(
        self,
        lower_tail_threshold: float = 0.02,   # 하위 2%
        upper_tail_threshold: float = 0.98,   # 상위 2%
        adx_max_for_mean_reversion: float = 25.0,
        vwap_dev_threshold: float = 0.0005,   # 0.05%
        atr_increase_lookback: int = 5,
        cooldown_bars: int = 3,
    ) -> None: ...

    def generate(
        self,
        metrics: DistributionMetrics,
        zigzag_state: ZigZagState,           # 기존 AdaptiveZigZag 상태
        supertrend_state: SuperTrendState,   # 기존 AdaptiveSuperTrend 상태
        vwap_dev: float,                     # tick_mixin에서 계산
        current_price: float,
    ) -> MeanReversionSignal: ...
```

#### 신호 생성 규칙

**Long 후보 조건 (모두 충족)**

```
metrics.left_tail_prob  < lower_tail_threshold   (하위 2% 이하)
AND zigzag_state 가 최근 저점 또는 하락 후반
AND current_price < VWAP * (1 - vwap_dev_threshold)
AND supertrend_state.adx < adx_max_for_mean_reversion   (약추세/횡보)
AND ATR이 최근 atr_increase_lookback 대비 증가 또는 유지
AND cooldown_bars 이내 동일 방향 신호 없음
```

**Short 후보 조건**은 부호/부등호를 반대로 적용합니다.

#### 신호 강도

| 강도 | 조건 |
|------|------|
| **STRONG** | tail_prob < 0.01 또는 tail_prob > 0.99 + ZigZag 피봇 확정 + ADX < 20 |
| **NORMAL** | tail_prob < 0.02 또는 tail_prob > 0.98 + VWAP 방향 일치 |
| **WEAK**   | tail_prob < 0.05 또는 tail_prob > 0.95, 추가 필터 미충족 시 HOLD |

---

## 4. 기존 컴포넌트 연동

### 4.1 `AdaptiveIndicatorManager` 연동

`indicators/indicator_integration.py`의 `AdaptiveIndicatorManager`가 분 단위로 업데이트될 때, KDE 추정기도 함께 갱신합니다.

```python
class AdaptiveIndicatorManager:
    def __init__(...):
        ...
        self.return_dist = ReturnDistributionEstimator(window=3000)
        self.mr_signal = MeanReversionSignalGenerator(...)

    def update(self, high, low, close):
        ...
        # 수익률 계산 (이전 종가 대비)
        if self._last_close > 0:
            ret = np.log(close / self._last_close)
            self.return_dist.update(ret)
        self._last_close = close

        metrics = self.return_dist.evaluate(ret) if self.return_dist.is_ready else None
        mr_sig = self.mr_signal.generate(
            metrics=metrics,
            zigzag_state=self.zigzag.state,
            supertrend_state=self.supertrend.state,
            vwap_dev=vwap_dev,
            current_price=close,
        )
        ...
```

### 4.2 `TransformerPredictor` 연동

`prediction/predictor.py`의 `TransformerPredictor`는 딥러닝 예측 외에 평균회귀 채널을 추가할 수 있습니다.

```python
class TransformerPredictor:
    def predict(self, *, input: ModelInput) -> TransformerPredictionResult:
        ...
        mr_action = self._mr_signal.generate(...)
        # 투표: 딥러닝 신호와 평균회귀 신호가 동일할 때만 강도 상승
        final_action = self._combine(dl_action, mr_action)
```

> 단, 초기 단계에서는 딥러닝 예측과 **독립적으로 운영**하는 것을 권장합니다. 두 채널이 상충할 경우 기존 예측을 우선하고, MR 신호는 로깅/백테스트용으로 축적합니다.

### 4.3 `Gate` 연동

`trading/gate.py` 또는 `trading/pivot_gate.py`에서 `MeanReversionSignal`을 수신할 수 있도록 인터페이스를 확장합니다.

```python
class PivotGate:
    def on_mean_reversion_signal(self, signal: MeanReversionSignal):
        if signal.action in ("BUY", "SELL") and signal.strength == "STRONG":
            ...
```

---

## 5. 피처 엔지니어링

### 5.1 추가 피처 목록

`prediction/features/features.py`에 다음 피처를 추가해 딥러닝/ML 모델 입력으로 활용합니다.

| 피처명 | 설명 | 모델 |
|--------|------|------|
| `ret_1m` | 1분 로그수익률 | 모든 모델 |
| `ret_5m` | 5분 로그수익률 | 모든 모델 |
| `pdf_1m` | 1분 수익률 KDE PDF 값 | XGBoost/RF |
| `cdf_1m` | 1분 수익률 KDE CDF 값 | XGBoost/RF |
| `left_tail_1m` | 하위 꼬리 확률 | XGBoost/RF |
| `right_tail_1m` | 상위 꼬리 확률 | XGBoost/RF |
| `z_score_1m` | Z-score | XGBoost/RF/LSTM |
| `dist_to_median_1m` | 현재 수익률과 KDE 중앙값 차이 | XGBoost/RF |
| `vwap_dev` | 이미 존재, MR 조건용 | 모든 모델 |
| `ast_adx_norm` | 이미 존재, 추세 필터용 | 모든 모델 |

### 5.2 피처 정규화

- `pdf_*`: 로그 변환 권장 (long-tail)
- `cdf_*`: 이미 [0, 1] 범위
- `z_score_*`: 클리핑 ±5 후 표준화

---

## 6. 설정 (config.json)

```json
{
  "kde_mean_reversion": {
    "enabled": false,
    "mode": "standalone",
    "estimator": {
      "window": 3000,
      "min_samples": 500,
      "bandwidth": "scott",
      "decay": null
    },
    "signal": {
      "lower_tail_threshold": 0.02,
      "upper_tail_threshold": 0.98,
      "adx_max_for_mean_reversion": 25.0,
      "vwap_dev_threshold": 0.0005,
      "atr_increase_lookback": 5,
      "cooldown_bars": 3,
      "require_zigzag_pivot": true,
      "require_vwap_cross": true
    },
    "integration": {
      "combine_with_dl": false,
      "log_only_when_disagree": true,
      "strength_filter": ["STRONG", "NORMAL"]
    }
  }
}
```

- `mode`: `"standalone"`(독립 실행), `"combined"`(딥러닝과 결합), `"disabled"`
- `strength_filter`: 실제 주문에 사용할 신호 강도

---

## 7. 구현 우선순위

### Phase 1: 오프라인 프로토타입 ✅ 완료

1. `indicators/return_distribution.py` 구현 ✅
2. `indicators/mean_reversion_signal.py` 구현 ✅
3. 백테스트 데이터로 KDE 피처 생성 ✅
4. `Devcenter/ml/ml_dataset.csv`에 KDE 피처 추가 ✅
5. 기존 XGBoost 필터 모델과 비교 ✅

### Phase 2: 실시간 통합 (남은 설계 항목)

1. `AdaptiveIndicatorManager`에 KDE/평균회귀 신호 연동
2. 실시간 1분봉 파이프라인에서 KDE 피처 생성
3. `TransformerPredictor`에 KDE 채널 로깅/결합 인터페이스 추가
4. Paper trading / 소규모 실전 테스트
5. 거래 방향에 따른 direction-aware KDE 피처 실시간 갱신

### Phase 3: 고도화 (남은 설계 항목)

1. 다중 timeframe KDE (1분/5분/15분) — 현재 1분/5분 완료
2. Regime별 KDE 분리 (BULL/BEAR/NEUTRAL마다 별도 KDE)
3. Bayesian 업데이트 (HMM 또는 온라인 감마 분포)
4. 슬리피지/체결 지연을 고려한 실행 전략
5. KDE 피처를 딥러닝 모델(Transformer/Mamba/PatchTST) 입력에 통합
6. Walk-forward / 롤링 윈도우 검증으로 통계적 유의성 확보

---

## 8. 리스크 및 대응

| 리스크 | 설명 | 대응 |
|--------|------|------|
| **과적합** | 과거 분포에 맞춰져 미래 변화 반응 늦음 | 롤링 윈도우 + decay + Regime별 분리 |
| **강추세 손실** | 평균회귀는 추세장에서 연속 손실 | ADX/SuperTrend 필터 + 반대 신호 금지 |
| **무더기 진입** | 연속적인 극단값에서 중복 진입 | cooldown_bars + 최대 포지션 제한 |
| **KDE 왜곡** | 샘플 부족/이상치로 분포 왜곡 | min_samples, bandwidth 로버스트 선택, 클리핑 |
| **실시간 지연** | KDE 재계산 및 CDF 적분 비용 | grid 기반 CDF 룩업 + 버퍼링 |

---

## 9. 검증 지표

백테스트/실전 적용 시 다음 지표를 중심으로 평가합니다.

| 지표 | 목표 | 비고 |
|------|------|------|
| 승률 | 기존 대비 향상 또는 유지 | 필터링 시 거래 수 감소 감안 |
| 평균 PnL | 상승 | 꼬리 진입으로 기대 수익 증가 |
| 최대 드로우다운 | 감소 또는 유지 | 추세장 손실 통제 |
| 거래 수 | 적정 수준 유지 | 너무 적으면 기회비용 |
| 거짓 신호 비율 | < 30% | VWAP/ZigZag 필터 효과 측정 |
| 신호-예측 일치율 | 딥러닝과의 상관관계 | 향후 결합/분리 판단 근거 |

---

## 10. 관련 문서

- `docs/PREDICTION_MODELS_ORGANIZATION.md`
- `docs/ARCHITECTURE.md`
- `docs/ADAPTIVE_INDICATOR_GUIDE.md`
- `docs/ZIGZAG_PIVOT_COMPREHENSIVE_GUIDE.md`
- `docs/ML_PREDICTION_GUIDE.md`
- `Devcenter/ml/ML_MODELS_INVENTORY.md`

---

## 11. Phase 1 구현 결과

### 11.1 생성된 모듈

| 파일 | 설명 |
|------|------|
| `indicators/return_distribution.py` | 롤링 KDE 추정기 (PDF/CDF/Z-score/꼬리확률) |
| `indicators/mean_reversion_signal.py` | KDE + ZigZag/VWAP/ATR/ADX 평균회귀 신호 생성기 |
| `tests/indicators/test_return_distribution.py` | KDE 추정기 및 신호 생성기 단위 테스트 |
| `Devcenter/ml/generate_kde_features.py` | CSV 백테스트 데이터용 KDE 피처 생성 CLI |
| `Devcenter/ml/generate_kde_features_parquet.py` | DuckDB 1분봉 parquet용 KDE 피처 생성 CLI |
| `Devcenter/ml/compare_kde_filter.py` | ml_dataset.csv 병합 + XGBoost 기반성능 비교 |
| `Devcenter/ml/optimize_kde_params.py` | KDE window/bandwidth grid search |

### 11.2 최적 KDE 파라미터

```json
{
  "window": 5000,
  "bandwidth": 0.002,
  "grid_points": 1024,
  "refit_every": 100
}
```

### 11.3 XGBoost 필터 성능 비교

Test set (2025-06-25 ~ 2026-06-19, 240건 기준):

| 지표 | Baseline | KDE-enhanced | Diff |
|------|----------|--------------|------|
| accuracy | 0.5125 | 0.5292 | +0.0167 |
| precision | 0.4588 | 0.4848 | +0.0260 |
| recall | 0.3545 | 0.4364 | +0.0818 |
| f1 | 0.4000 | 0.4593 | +0.0593 |
| roc_auc | 0.4870 | **0.5508** | **+0.0638** |

- KDE 모델은 **threshold ≥ 0.55**에서 baseline 대비 총 PnL 우수
- KDE 추가로 피처 중요도가 `entry_month`/`entry_hour` 등 달력 피처 집중에서 기술적 지표로 분산

### 11.4 남은 설계 항목

- Phase 2: 실시간 `AdaptiveIndicatorManager` 연동, `TransformerPredictor` 로깅/결합
- Phase 3: Regime별 KDE, 다중 timeframe, 딥러닝 입력 통합, walk-forward 검증
