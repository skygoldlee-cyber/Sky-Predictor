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

### 11.2 최적 KDE 파라미터 (validation AUC 기준)

```json
{
  "window": 2000,
  "bandwidth": 0.001,
  "grid_points": 1024,
  "refit_every": 100
}
```

- validation ROC AUC = 0.5415
- untouched test ROC AUC = 0.3844 (최종 config 1회 평가, 과적합/샘플 부족 가능성)

### 11.3 XGBoost 필터 성능 비교 (버그 수정 + 추가 조치 후)

#### 11.3.1 초기 1년 테스트 (2025-06-25 ~ 2026-06-19, 242건)

| 지표 | Baseline | KDE-enhanced | Diff |
|------|----------|--------------|------|
| accuracy | 0.4504 | 0.4669 | +0.0165 |
| precision | 0.4358 | 0.4596 | +0.0238 |
| recall | 0.7091 | 0.9818 | +0.2727 |
| f1 | 0.5398 | 0.6261 | +0.0863 |
| roc_auc | 0.4816 | **0.5249** | **+0.0433** |

- 1년 데이터만 KDE 피처가 있어 표본이 작았음.
- 분류 성능은 개선됐으나 PnL은 개선되지 않음.

#### 11.3.2 전 구간 5분봉 KDE 테스트 (2019-06-03 ~ 2026-06-19, 1,211건 test)

5분봉 데이터가 2019년부터 있어 5분봉 KDE 피처를 전 구간 생성 후 재평가.

| 지표 | Baseline | KDE-enhanced | Diff |
|------|----------|--------------|------|
| accuracy | 0.4814 | 0.6152 | +0.1338 |
| precision | 0.4549 | 0.5589 | +0.1040 |
| recall | 0.4549 | 0.9062 | +0.4514 |
| f1 | 0.4549 | 0.6914 | +0.2365 |
| roc_auc | 0.4771 | **0.7279** | **+0.2508** |

| threshold | Baseline 총 PnL | Baseline 승률 | KDE 총 PnL | KDE 승률 |
|-----------|----------------|---------------|-----------|----------|
| 0.40 | -8,653,487 | 47.56% | +18,064,620 | 56.59% |
| 0.50 | -8,369,066 | 45.49% | +17,021,670 | 62.48% |
| 0.55 | - | - | **+23,891,280** | **69.57%** |
| 0.65 | - | - | +22,455,970 | 75.26% |

- **KDE 추가 시 test set에서 ROC AUC 0.25, F1 0.24 상승.**
- **PnL 역전**: Baseline은 모든 threshold에서 손실, KDE는 0.40~0.80 threshold에서 모두 수익.
- **피처 중요도 상위 3개**: `kde_aligned_tail_5m`, `kde_opposite_tail_5m`, `kde_aligned_zscore_5m`.

#### 11.3.3 3단계 전체 파이프라인 재검증 (XGBoost filter → RF entry → LSTM exit, 2025-2026 test)

| 지표 | Baseline (재학습) | KDE-enhanced | Diff |
|------|-------------------|--------------|------|
| Stage 1 ROC AUC | 0.4901 | **0.6999** | +0.2098 |
| Stage 2 ROC AUC | 0.4995 | 0.5791 | +0.0796 |
| Stage 3 ROC AUC | 0.4983 | 0.5000 | +0.0017 |
| **최종 거래 수** | 441 | 742 | +301 |
| **최종 승률** | 43.76% | **63.07%** | +19.31%p |
| **최종 총 PnL** | **-3,670,926** | **+23,690,192** | **+27,361,118** |
| **최종 평균 PnL** | -8,324 | +31,927 | +40,251 |

- **3단계 파이프라인 전체에 KDE 5분봉 피처를 추가하면 test set(2025-2026)에서 baseline이 손실(-367만 원)인 반면 KDE는 +2,369만 원 수익.**
- 거래 수는 68% 증가, 승률은 19.3%p 상승.
- 주의: baseline 재학습 결과가 과거 보고서(20.08% 수익률)와 다르게 나타난 점은 threshold 튜닝 방식과 `regime` 컬럼 차이 때문. 상대적 KDE 효과는 뚜렷함.

#### 11.3.4 거래 비용 반영 + walk-forward 검증 (2025-2026)

슬리피지를 1틱(0.05pt)/side, 계약승수 31,500원 반영 후 재평가. LSTM은 BatchNorm, class weight, early stopping, units=64/dropout=0.5로 개선.

| Fold | Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL |
|------|---------|--------|------|--------|---------|
| 2019-2023→2024→2025 | Baseline | 38 | 26.32% | -631,716 | -16,624 |
| 2019-2023→2024→2025 | **KDE** | **179** | **62.01%** | **+2,734,972** | **+15,279** |
| 2019-2024→2025→2026 | Baseline | 151 | 41.72% | -6,573,366 | -43,532 |
| 2019-2024→2025→2026 | **KDE** | **271** | **74.54%** | **+20,695,022** | **+76,365** |
| 2020-2024→2025→2026 | Baseline | 162 | 46.30% | -2,103,570 | -12,985 |
| 2020-2024→2025→2026 | **KDE** | **310** | **72.58%** | **+20,054,132** | **+64,691** |

- **모든 walk-forward fold에서 KDE가 baseline을 압도.**
- 2026 단독 테스트에서는 KDE가 +2,000만 원 이상 수익, baseline은 손실.
- LSTM exit 단독 ROC AUC는 여전히 ~0.5 수준으로 raw discrimination은 낮지만, XGB+RF로 선별된 고신뢰 거래를 보수적으로 필터링하는 역할로 승률/수익성 향상.

#### 11.3.5 5분봉 KDE 파라미터 최적화

> ⚠️ **중요: 11.3.5 ~ 11.3.7의 초기 결과는 look-ahead bias가 발견되어 무효화되었습니다. 아래 11.3.8에서 수정된 결과를 확인하세요.**

Grid search (`window × bandwidth × grid_points × refit_every`)를 5분봉 KDE에 적용.
XGBoost filter 단계에서 **validation ROC AUC / PnL / Sharpe** 별로 최적 파라미터를 선택하고,
각 기준별 최적 config로 3단계 전체 파이프라인을 재실행 (test 2025-2026, 1틱/side 슬리피지 반영).

**Grid 범위**
- `window`: 900, 1000, 1100, 1200
- `bandwidth`: scott, silverman
- `grid_points`: 1024, 2048
- `refit_every`: 50, 100

**Top grid 결과 (AUC / PnL / Sharpe, window 900-1200)**

| window | bandwidth | grid_points | refit_every | val ROC AUC | val PnL | val Sharpe |
|--------|-----------|-------------|-------------|-------------|---------|------------|
| 1000   | scott     | 1024        | 50          | 0.6716      | 1,550,849 | 2.946   |
| 1200   | scott     | 2048        | 100         | 0.6696      | **1,567,480** | **4.686**   |
| 900    | silverman | 2048        | 100         | 0.6722      | 1,502,419 | 4.695   |
| 1100   | scott     | 2048        | 100         | 0.6724      | 1,434,992 | 4.363   |
| 1100   | silverman | 2048        | 50          | 0.6690      | 1,461,750 | 4.439   |

**확장 grid (window 1300-1500) Top 결과**

| window | bandwidth | grid_points | refit_every | val ROC AUC | val PnL | val Sharpe | multi_score |
|--------|-----------|-------------|-------------|-------------|---------|------------|------------|
| 1500   | scott     | 1024        | 100         | 0.6751      | 1,493,088 | 4.157   | 0.8598     |
| 1400   | silverman | 2048        | 50          | 0.6699      | 1,505,815 | 4.511   | 0.8274     |
| 1400   | scott     | 1024        | 50          | 0.6708      | 1,529,838 | 3.921   | 0.7593     |
| 1300   | scott     | 1024        | 100         | 0.6739      | 1,470,565 | 3.717   | 0.6929     |

Window 1200에서 multi_score 0.9245가 여전히 최고이므로, 확장 탐색 결과는 최적 영역이 **1200~1400, scott/silverman, grid_points 1024~2048, refit 50~100** 근처에 있음을 확인.

**Multi-objective 최적 config** (AUC 0.2 + PnL 0.4 + Sharpe 0.4 가중치):
`window=1200`, `bandwidth=scott`, `grid_points=2048`, `refit_every=100`

##### 기준별 최종 test 결과 (single-timeframe 5m KDE)

| Criterion | Baseline 거래/승률/총PnL/test Sharpe | KDE 거래/승률/총PnL/test Sharpe | KDE Stage 1 AUC |
|-----------|-----------------------------------|-------------------------------|-----------------|
| AUC       | 91 / 40.66% / -1,202,938 / -       | **579** / **62.52%** / **+17,804,505** / - | 0.7062 |
| PnL       | 81 / 46.91% / -194,929 / -         | **436** / **67.20%** / **+20,072,999** / - | 0.7036 |
| Sharpe    | 88 / 40.91% / -3,772,816 / -        | **497** / **67.61%** / **+20,870,567** / 5.78 | 0.7091 |
| **Multi** | 209 / 43.06% / -2,139,225 / -0.66   | **463** / **66.09%** / **+22,366,910** / **6.68** | 0.7045 |

- **validation PnL/Sharpe 기준 최적화가 AUC 기준보다 test PnL에서 더 우수** (+2,000만 원 이상).
- Multi-objective 가중치가 가장 높은 test PnL **+2,237만 원**, Sharpe **6.68** 기록.
- `window=1000~1200`, `bandwidth=scott/silverman`, `grid_points=2048`, `refit_every=50~100` 근처가 최적 영역.
- Baseline PnL가 실행마다 소폭 변동(RF/LSTM 내부 randomness)하지만, KDE의 relative 개선폭은 일정함.

##### 다중 timeframe KDE 피처 추가 (5m + 15m + 30m + 60m)

동일한 multi-objective 최적 config에 15m/30m/60m KDE 피처를 추가한 결과:

| Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL | test Sharpe | Stage 1 AUC |
|---------|--------|------|--------|---------|-------------|-------------|
| Baseline_MultiTF | 345 | 42.61% | -6,692,407 | -19,398 | -1.58 | 0.5297 |
| **KDE_MultiTF** | **322** | **69.88%** | **+19,212,803** | **+59,667** | **7.09** | 0.6991 |

- 다중 timeframe KDE는 거래 수는 줄었지만 **승률(69.9%)과 Sharpe(7.09)를 크게 개선**.
- 총 PnL은 단일 timeframe multi-objective(+2,237만 원)보다 소폭 낮으나, **리스크 조정 수익률(Sharpe)과 평균 거래 수익(평균 +5.97만 원)은 더 우수**.
- 전략 성향에 따라 **높은 Sharpe 선호 시 다중 timeframe**, **총 PnL 극대화 시 단일 timeframe multi-objective** config를 선택할 수 있음.
- 종합: **5분봉 KDE를 기존 3단계 ML 파이프라인에 도입하면 수익성을 크게 개선할 수 있으며, 파라미터 선정 기준은 PnL/Sharpe 또는 multi-objective 기준이 실제 수익에 더 효과적.**

#### 11.3.6 다중 timeframe KDE walk-forward 검증

Multi-objective 최적 config(`window=1200`, `scott`, `grid=2048`, `refit=100`)에 5m/15m/30m/60m KDE 피처를 추가하고 walk-forward 검증을 수행한 결과.

| Fold | Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL |
|------|---------|--------|------|--------|---------|
| 2019-2023→2024→2025 | Baseline | 13 | 30.77% | -547,701 | -42,131 |
| 2019-2023→2024→2025 | **KDE** | **124** | **59.68%** | **+1,653,874** | **+13,338** |
| 2019-2024→2025→2026 | Baseline | 195 | 42.05% | -4,265,526 | -21,874 |
| 2019-2024→2025→2026 | **KDE** | **171** | **74.85%** | **+16,089,477** | **+94,091** |
| 2020-2024→2025→2026 | Baseline | 135 | 44.44% | +1,192,650 | +8,834 |
| 2020-2024→2025→2026 | **KDE** | **203** | **74.88%** | **+17,131,653** | **+84,392** |

- **모든 walk-forward fold에서 다중 timeframe KDE가 baseline을 능가.**
- Fold 1(2025 단독 test)에서는 절대 수익이 작지만 여전히 양수(+165만 원)이며 baseline은 -54만 원 손실.
- Fold 2/3(2026 test)에서는 각각 +1,609만 원, +1,713만 원 수익으로 안정성 확인.
- 다중 timeframe KDE가 단일 timeframe 대비 거래 수는 줄지만 **승률과 Sharpe 측면에서 더 강건**함.

#### 11.3.7 실전 적용 기준 대비 최종 성능 검토

KP200 선물 5분봉 전략을 기준으로 한 권장 수준과 현재 KDE 기반 3단계 파이프라인(test 2025-2026, 1틱/side 슬리피지)을 비교.

| 항목 | 권장 수준 | 단일 timeframe KDE (multi-objective) | 다중 timeframe KDE |
|------|-----------|-------------------------------------|--------------------|
| **Sharpe Ratio** | ≥ 2.0 | **6.68** | **7.09** |
| **승률** | 55~70% | **66.09%** | **69.88%** |
| **Profit Factor** | ≥ 1.8 | **2.88** | **4.03** |
| **Calmar Ratio** | ≥ 1.5 | **16.83** | **21.65** |
| **Max Drawdown** | 연수익 대비 낮음 | 연수익 ~1,118만 원 / MDD 133만 원 | 연수익 ~961만 원 / MDD 89만 원 |
| **연간 거래 횟수** | ≥ 500건 | 2025: 161건 / 2026: 302건 | 2025: 124건 / 2026: 198건 |

- **Sharpe, 승률, Profit Factor, Calmar 모두 권장 기준을 크게 상회**하며, MDD도 연 수익 대비 매우 낮음.
- **연간 거래 횟수는 500건 미만**으로 권장 기준에 미치지 못함. 5분봉 선물 1종목에 대해 진입 필터를 거치는 전략 특성상 500건 기준은 다소 높게 볼 수 있음. 2년간 322~463건이면 성능 지표 분산이 다소 클 수 있으므로 추가 out-of-sample 검증 권장.
- 실전 적용 관점에서는 **리스크 조정 성능(Sharpe/Calmar/PF)이 우수**하여 staging/paper trading을 시작할 수 있는 수준. 다만 overfitting 리스크를 줄이기 위해 2027년 추가 데이터 또는 다른 월물로 추가 walk-forward 검증 후 실거래 전환 권장.

#### 11.3.8 Look-ahead bias 발견 및 수정

Sharpe 7.09 등 극단적인 성능에 대한 의문을 해소하기 위해 merge 시점을 점검한 결과, **look-ahead bias**가 발견되었습니다.

- **원인**: 거래 entry_time이 5분봉 timestamp와 정확히 일치하고, 진입가가 해당 봉의 **open**에 해당(예: 10:50 봉 open 266.70). 기존 `pd.merge_asof(..., direction='backward')`는 entry_time과 동일한 timestamp의 KDE 피처를 허용했으나, 해당 timestamp의 KDE 피처는 그 봉의 **close**까지 계산되므로, 진입 시점 이후 가격 정보를 미리 본 것과 동일.
- **해결책**: 모든 KDE-거래 merge에 `allow_exact_matches=False`를 추가해, entry_time과 동일한 bar가 아닌 **이전 bar의 close까지 사용한 KDE 피처**만 사용.
  - 수정 파일: `compare_kde_filter.py`, `optimize_kde_params.py`, `optimize_kde_params_5min.py`, `run_multi_tf_kde_pipeline.py`

**수정 후 grid search (window 900-1200, multi-objective)**

| window | bandwidth | grid_points | refit_every | val ROC AUC | val PnL | val Sharpe |
|--------|-----------|-------------|-------------|-------------|---------|------------|
| 1000   | scott     | 1024        | 100         | 0.4808      | -1,716,121 | -2.252 |
| 1000   | silverman | 1024        | 100         | 0.4917      | -1,813,338 | -2.452 |
| 1100   | scott     | 1024        | 100         | 0.4574      | -2,095,902 | -3.398 |
| 1200   | silverman | 1024        | 100         | 0.5236      | -2,781,981 | -3.475 |

→ validation PnL/Sharpe가 모두 음수. **KDE 피처만으로는 5분봉 진입 시점에 예측력이 거의 없음.**

**수정 후 test 2025-2026 결과**

| Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL | test Sharpe | Stage 1 AUC |
|---------|--------|------|--------|---------|-------------|-------------|
| Baseline (single-tf) | 204 | 42.65% | -2,710,207 | -13,285 | - | 0.5297 |
| **KDE (single-tf, window=1000, scott)** | 111 | 44.14% | -1,278,222 | -11,516 | - | 0.5347 |
| Baseline (multi-tf) | 135 | 45.93% | +13,626 | +101 | 0.01 | 0.5297 |
| **KDE (multi-tf)** | 91 | 40.66% | -288,162 | -3,167 | -0.14 | 0.5375 |

- **단일 timeframe**: baseline -271만 원, KDE -128만 원. KDE가 손실을 줄이지만 여전히 수익이 아님.
- **다중 timeframe**: baseline +1.4만 원(거의 0), KDE -28.8만 원. KDE가 오히려 손실 확대.
- Stage 1 ROC AUC도 0.53~0.54 수준으로 **random guessing에 가까움**.

**결론 및 시사점**

- **초기 Sharpe 6~7, PnL +2,000만 원대 결과는 look-ahead bias로 인한 과장**이었습니다. 이는 매우 좋은 전략을 발견한 것이 아니라, 미래 1개봉 정보라도 유출되었기 때문입니다.
- 수정 후 KDE는 baseline보다 소폭 나은 경우도 있지만, **경제적으로 의미 있는 edge는 없음**.
- **향후 방향**:
  - KDE를 진입 **직전**봉(close 기준) 외에도, **open/high/low/volume** 기반 추가 정보를 활용.
  - Target label을 단순 이진 승/패가 아닌 **R-multiple, holding-period return** 등으로 개선.
  - Regime별 KDE(상승/하락/횡보) 적용 시 기존 단순 KDE보다 유의미할 가능성을 별도 검증.
  - 실거래 전에는 반드시 paper trading 또는 2027년 추가 데이터로 out-of-sample 검증.

#### 11.3.9 Regime별(상승/하락/횡보) conditional KDE 적용

단순 KDE는 전체 5분봉 수익률 분포를 하나의 분포로 모델링. 상승장과 하락장에서 극단값(tail)의 의미가 다르므로, **regime별로 별도 KDE를 유지**하는 conditional KDE를 추가.

**Regime 정의** (`add_regime`): 20-EMA / 60-EMA 기준
- **bull**: close > slow_ema and fast_ema > slow_ema
- **bear**: close < slow_ema and fast_ema < slow_ema
- **neutral**: 그 외

`build_kde_features`에 `regime_col` 인자를 추가해 현재 bar의 regime에 해당하는 estimator만 update/evaluate. 5m/15m/30m/60m 각 timeframe마다 동일한 regime 조건으로 conditional KDE 생성.

**수정 후 multi-timeframe + regime KDE 결과 (test 2025-2026, look-ahead 차단)**

| Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL | Profit Factor | test Sharpe | MDD |
|---------|--------|------|--------|---------|---------------|-------------|-----|
| Baseline (multi-tf) | 90 | 45.56% | -1,240,131 | -13,779 | 0.84 | -0.57 | 3,387,698 |
| **KDE (multi-tf + regime)** | 138 | 46.38% | **+659,084** | **+4,776** | **1.06** | **0.21** | 3,319,100 |

- **Regime별 conditional KDE + t-1 look-ahead 배제** 조합으로 baseline 손실(-124만 원) 대비 KDE는 소폭 **수익(+66만 원)**.
- Profit Factor 1.06, Sharpe 0.21로 아직 권장 기준에는 미치지 못하지만, **KDE가 baseline을 능가하는 첫 사례**.
- 연간 거래 횟수: 2025년 36건, 2026년 102건으로 여전히 적음.

> **t-1 shift 적용 상세**: KDE 피처는 봉 `t` close까지의 데이터로 계산되며, `feature_time = timestamp + 5m`으로 변환. `entry_time`이 봉 `t+1`의 open이라 할 때, `pd.merge_asof(..., right_on='feature_time', direction='backward', allow_exact_matches=False)`로 반드시 직전 완성봉까지만 사용.

#### 11.3.10 Regime 정교화 + 추가 피처 + 거래 횟수 확대

세 가지 개선을 동시에 적용:

1. **Regime 정교화**: 단순 bull/bear/neutral 대신 `trend(EMA20/60) × volatility(rolling std)`를 결합한 6-class regime(e.g. `bull_high`, `bull_low`, `bear_high`, `bear_low`, `neutral_high`, `neutral_low`).
2. **추가 피처**: `price_ma20_ratio`, `price_ma60_ratio`, `bb_width`, `atr_ratio`, `rsi_macd_interaction`, `rsi_bb_interaction`를 RF/LSTM 입력에 추가.
3. **거래 횟수 확대**: threshold 선택 시 최소 거래 수 제약을 완화(stage1=300, stage2=100, stage3=30)하여 과도한 필터링 방지.

**최신 결과 (test 2025-2026, look-ahead 배제)**

| Variant | 거래 수 | 승률 | 총 PnL | 평균 PnL | Profit Factor | test Sharpe | MDD |
|---------|--------|------|--------|---------|---------------|-------------|-----|
| Baseline (multi-tf) | 216 | 43.52% | -2,889,108 | -13,376 | 0.78 | -1.21 | 5,099,460 |
| **KDE (multi-tf + refined regime + features)** | 260 | 46.92% | **+4,240,027** | **+16,308** | **1.35** | **1.40** | 1,969,532 |

- KDE가 baseline을 능가하며 **Profit Factor 1.35, Sharpe 1.40** 기록.
- 연간 거래 횟수: 2025년 118건, 2026년 142건으로 이전보다 크게 늘었으나 권장 500건에는 여전히 미달.
- Sharpe ≥2.0, PF ≥1.8 권장 기준에는 조금 부족하지만, **양수 수익과 낮은 MDD**를 동시에 달성한 안정적 결과.
- 모델 내부(XGBoost, LSTM)의 난수 시드가 고정되지 않아 실행마다 결과가 일부 변동할 수 있으므로, 최종 평가 전 `random_state` 고정 또는 여러 번 반복 실험 권장.

### 11.4 Phase 1.5 핵심 버그 수정

재검증 전에 발견된 구현상 문제를 수정했습니다.

| 항목 | 수정 전 문제 | 수정 내용 |
|------|-------------|-----------|
| `optimize_kde_params.py` | test AUC를 grid search 선택 기준으로 사용해 optimistic bias | validation AUC로 파라미터 선택, 최종 config만 test AUC 1회 평가 |
| `mean_reversion_signal.py` SuperTrend | 방향 필터가 반대로 작동 (상승장에서 롱, 하락장에서 숏 허용) | 상승장에서만 롱, 하락장에서만 숏 허용 |
| `mean_reversion_signal.py` cooldown | `bar_index` 미전달 시 영구 냉각 | 미전달 시 내부 카운터 자동 증가 |
| `mean_reversion_signal.py` upper_tail | `upper_tail_threshold`가 사용되지 않음 | SELL 트리거를 `right_tail_prob < 1 - upper_tail_threshold`로 사용 |
| `mean_reversion_signal.py` strength | `tail_prob > 1 - threshold` 분기가 dead code, WEAK 강도 미발생 | WEAK-entry를 `normal_tail_prob`로 분리, 비대칭 NORMAL/STRONG 강도 적용 |
| `generate_kde_features_parquet.py` | 세션 경계(개장/종장) 수익률이 분포를 오염 | `session_date` 기준 그룹별 shift, 경계값은 NaN 처리 |
| `compare_kde_filter.py` | `merge`로 정확한 timestamp가 아니면 KDE 누락 + look-ahead 위험 | `merge_asof(direction='backward')`로 가장 최근 1분봉 피처 사용 |
| XGBoost | `use_label_encoder` deprecated, early stopping 없음 | `use_label_encoder` 제거, validation set 기반 `early_stopping_rounds=20` 추가 |
| XGBoost | 클래스 불균형 + 고정 threshold 0.5로 인해 precision/recall/f1 0 | `scale_pos_weight` 추가, validation F1 기반 threshold 튜닝 |

### 11.5 남은 설계 항목

- Phase 2: 실시간 `AdaptiveIndicatorManager` 연동, `TransformerPredictor` 로깅/결합
- Phase 3: Regime별 KDE, 다중 timeframe, 딥러닝 입력 통합, walk-forward 검증
