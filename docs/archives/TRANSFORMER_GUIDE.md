# Transformer 로직 가이드 (상세)

본 문서는 저장소의 Transformer(수치 예측) 관련 로직을 한 곳에 정리한 문서입니다.

- 대상 코드:
  - `prediction/features.py` (`OB_KEYS`, `CD_KEYS`, `OPT_KEYS`, `build_sequence`)
  - `prediction/model.py` (`PriceTransformer`)
  - `prediction/predictor.py` (`NumericPredictor`, `TransformerPredictor`, `TFTPredictor`, `EnsemblePredictor`)
  - `prediction/tft_model.py` (`TemporalFusionTransformer`)
  - `prediction/time_features.py` (TFT time features: past_known/future_known)
  - `prediction/data_builder.py` (학습 데이터셋 생성: JSONL → NPZ)
  - `train.py` (오프라인 학습)
  - `train_tft.py` (TFT 오프라인 학습)
  - `merge_datasets.py` (최근 N일 rolling merge)

### 목차

| 섹션 | 내용 |
|------|------|
| **0** | Transformer의 목적과 역할 — 시스템 내 위치, 왜 Transformer인가, 예측 목표 |
| **1** | 입력 시퀀스 정의 — OB/CD/OPT 블록 상세, 피처 해석 가이드 |
| **2** | `build_sequence` — 블록별 조립 로직, `_ts_epoch` 기반 candle 매핑 |
| **3** | 모델 정의 `PriceTransformer` — 아키텍처, CLS token, Pre-LayerNorm |
| **4** | 추론 래퍼 `TransformerPredictor` — 초기화, inference, fallback, 피처 정규화, 성능 평가 |
| **5** | 학습 데이터셋 생성 — train/serve 정합, CD/OPT 구현 방향 |
| **6** | 학습 실행 — 시간순 분할, 손실 함수, 모니터링 |
| **7** | 운영 체크리스트 |
| **8** | 자주 발생하는 문제와 해결 방법 |

---

## 0) Transformer의 목적과 역할

### 0.1 시스템 내 위치

본 시스템은 **두 모델의 역할을 명확히 분리**합니다.

```
[Transformer]               [LLM]
수치 예측 전담               해석·판단 전담
──────────────              ──────────────
입력: 오더북 + 분봉           입력: Transformer 출력 + 시장 컨텍스트
출력: P(up) ∈ [0, 1]        출력: 전략 판단 JSON (action/risk_level/rationale)
```

Transformer는 **오직 숫자로만 판단**합니다. 뉴스, 이벤트, 거시경제 해석은 LLM의 영역입니다. Transformer가 "확률"을 내면 LLM이 그 확률의 맥락을 해석하여 최종 전략 판단을 내립니다.

### 0.2 왜 Transformer인가

KP200 선물 호가 데이터는 **시계열 구조**를 가집니다. 매 초마다 갱신되는 오더북 스냅샷(OBI, spread, 잔량 등)과 분봉 캔들은 서로 시간적 맥락이 있습니다.

- **단순 MLP**: 각 시점을 독립적으로 처리 — 시간 패턴 학습 불가
- **LSTM/GRU**: 순환 구조 — 기울기 소실, 병렬화 어려움
- **Transformer**: Self-attention으로 시퀀스 내 임의 위치 간 관계를 직접 학습 — 60초 전 OBI 변화와 현재 spread 간의 관계도 포착 가능

### 0.3 예측 목표

```
입력: 최근 seq_len초(기본 60초)의 오더북·분봉·옵션 피처
출력: N분 후 가격이 현재보다 높을 확률 P(up)
레이블: 1 (up) = N분 후 종가 > 현재가, 0 (down) = 그 반대
```

N = `prediction_minutes` (기본 5분). 즉 **5분 후 방향성 이진 분류** 문제입니다.

---

## 1) 입력 시퀀스 정의

Transformer 입력은 고정 길이 시계열 텐서입니다.

- **shape**: `(seq_len, feature_dim)`
- **dtype**: `float32`

`feature_dim`은 아래 설정 조합에 따라 달라집니다:

| option_feature_set | adaptive_indicator.enabled | feature_dim | 구성 |
|---|---:|---:|---|
| `v1` | false | 19 | OB(7) + CD(5) + OPT(7) |
| `v1` | true  | 47 | OB(7) + CD(5) + OPT(7) + ADAPT(28) |
| `v2` | false | 28 | OB(7) + CD(5) + OPT(16) |
| `v2` | true  | 56 | OB(7) + CD(5) + OPT(16) + ADAPT(28) |

```
시퀀스 구조 (seq_len=60, feature_dim 예시)

시간 →  t-59s  t-58s  ...  t-1s   t(현재)
         ┌─────────────────────────────────┐
OB  [7]  │ obi spread level1_ratio ...     │
CD  [5]  │ ret1 ret3 slope3 ...            │
OPT [7/16]  │ option snapshot + (v2일 때) optm_* ... │  (스칼라, 전 행 동일)
ADAPT[28]   │ ast_* + azz_* + cross_*              │  (adaptive indicators, 전 행 동일)
         └─────────────────────────────────┘
         col 0~6    col 7~11    col 12~(OPT end)   col (OPT end+1)~
```

> **중요**: 모든 인덱스 참조는 하드코딩 대신 `OB_KEYS`, `CD_KEYS`, `OPT_KEYS` 상수로 해야 합니다. 키 순서가 바뀌면 magic index는 silently wrong이 됩니다.

---

### 1.1 Orderbook 블록 (col 0~6, 7개)

| 인덱스 | 키 | 의미 | 범위(참고) |
|--------|-----|------|-----------|
| 0 | `obi` | Order Book Imbalance = (총매수잔량 - 총매도잔량) / 총잔량 | [-1, 1] |
| 1 | `spread` | 최우선 매도호가 - 최우선 매수호가 | ≥ 0 (틱 단위) |
| 2 | `level1_ratio` | L1 매수잔량 불균형 = (bidrem1 - offerrem1) / 합계 | [-1, 1] |
| 3 | `bid_slope` | 매수 잔량 기울기 = (bidrem5 - bidrem1) / 4 | 음수: 얕은 매수벽 |
| 4 | `offer_slope` | 매도 잔량 기울기 = (offerrem5 - offerrem1) / 4 | 양수: 깊은 매도벽 |
| 5 | `totbidrem` | 5단계 총 매수잔량 합계 | ≥ 0 |
| 6 | `totofferrem` | 5단계 총 매도잔량 합계 | ≥ 0 |

**OBI 해석 가이드**:
- `obi > 0`: 매수 우세 — 단기 상승 압력
- `obi < 0`: 매도 우세 — 단기 하락 압력
- `|obi| > 0.3`: 강한 편향 신호
- `level1_ratio`는 L1만의 불균형 — 대형 주문 진입/취소 감지에 유용

**slope 해석 가이드**:
- `bid_slope < 0` + `offer_slope > 0`: "V자형" 호가창 — ATM 근처 집중
- `bid_slope ≈ 0` + `offer_slope >> 0`: 매도벽이 뒤쪽에 쌓임 — 상승 저항

---

### 1.2 Candle(분봉) 블록 (col 7~11, 5개)

| 인덱스 | 키 | 계산식 | 의미 |
|--------|-----|--------|------|
| 7 | `ret1` | `Close.pct_change(1)` | 직전 분봉 대비 수익률 |
| 8 | `ret3` | `Close.pct_change(3)` | 3분봉 수익률 (단기 모멘텀) |
| 9 | `slope3` | `Close.diff(3) / 3` | 3분봉 선형 기울기 |
| 10 | `vol_accel` | `Volume / rolling(5).mean()` | 거래량 가속도 (1.0 = 평균) |
| 11 | `range_pct` | `(High - Low) / Close` | 분봉 변동폭 비율 |

**해석 가이드**:
- `ret3 > 0` + `vol_accel > 1.5`: 상승 모멘텀 확인 — BUY 신호 강화
- `range_pct` 급증: 변동성 확대 — confidence 저하 요인
- `vol_accel < 0.5`: 거래 소강 — 신호 신뢰도 감소

**시퀀스 매핑 방식**: 오더북은 초(1Hz) 해상도이고 캔들은 분봉(1/60Hz) 해상도입니다. 각 초 단위 ob_record에 해당 시점이 속하는 분봉의 캔들 피처를 매핑합니다.

```
ob_record._ts_epoch 기반 매핑 (권장)

초 시퀀스:  ...│13:04:45│13:04:46│13:04:47│...│13:05:00│13:05:01│...
               ↓         ↓         ↓             ↓         ↓
캔들 매핑:  ...│ 13:04분봉 피처    │...│ 13:05분봉 피처    │...
```

`_ts_epoch`가 없으면 선형 매핑(`row * bars / seq_len`)으로 fallback합니다 (정밀도 낮음).

---

### 1.3 Option scalar 블록 (col 12~18, 7개)

| 인덱스 | 키 | 의미 | 기본값 (데이터 없을 때) |
|--------|-----|------|----------------------|
| 12 | `pcr_volume` | Put/Call Ratio (거래량 기준) | 1.0 |
| 13 | `iv_skew` | ATM Put IV / ATM Call IV | 1.0 |
| 14 | `max_pain_dist_pct` | (현재가 - Max Pain 행사가) / 현재가 × 100 | 0.0 |
| 15 | `atm_iv` | ATM Call IV (절대값) | 0.0 |
| 16 | `atm_spread_pct` | ATM 옵션 스프레드 비율 ((ask-bid)/mid×100) | 0.0 |
| 17 | `atm_orderbook_imb` | ATM 옵션 L1 잔량 불균형 | 0.0 |
| 18 | `atm_liquidity_log` | ATM 옵션 5단계 유동성(log1p 합산) | 0.0 |

이 블록은 **스칼라** 값이므로 `seq_len` 전체 행에 동일한 값이 복제됩니다 (`np.tile`).

`prediction.option_feature_set="v2"`인 경우 OPT 블록이 16차원으로 확장되며, `optm_*`(옵션 분봉 미세움직임) 9개 피처가 추가됩니다.

---

### 1.4 Adaptive indicator scalar 블록 (ADAPT_KEYS, 28개)

AdaptiveSuperTrend + AdaptiveZigZag + cross 피처(총 28개)가 추가됩니다.

- 키 목록은 `prediction/features.py`의 `ADAPT_KEYS`로 고정되어 있으며, 모델 입력 차원에 직접 영향을 줍니다.
- 이 블록도 스칼라 값(“현재 분봉 기준”)이므로 `seq_len` 전체 행에 동일한 값이 복제됩니다.

**해석 가이드**:
- `pcr_volume > 1.2`: 풋 매수 우세 — 시장 하방 헤지 증가, 약세 신호
- `pcr_volume < 0.8`: 콜 매수 우세 — 강세 기대
- `iv_skew > 1.1`: 풋 프리미엄 높음 — 하방 리스크 프리미엄 증가
- `max_pain_dist_pct > 0`: 현재가 > Max Pain — 만기 수렴 시 하락 압력 예상
- `atm_iv` 급등: 시장이 단기 변동성 확대 예상 — confidence 조정 필요

**옵션 미구독 시**: 옵션 데이터가 없으면 기본값(PCR=1.0, skew=1.0, dist=0.0, iv=0.0)이 사용됩니다. 학습/서빙 모두 동일 기본값을 사용해야 train-serve 정합이 유지됩니다.

---

## 2) `build_sequence` (서빙 입력 생성)

- **구현**: `prediction/features.py::build_sequence`
- **역할**: 서빙 시 실시간 데이터에서 모델 입력 텐서를 조립

### 2.1 입력 → 출력 흐름

```
입력
├── ob_records  : List[dict]         FH0 기반 오더북 피처 (1Hz, maxlen=seq_len)
│                                    각 dict = {obi, spread, ..., _ts_epoch(옵션)}
├── candle_df   : pd.DataFrame       calc_candle_features() 결과 (분봉 인덱스)
│                                    컬럼: ret1, ret3, slope3, vol_accel, range_pct
└── opt_features: dict               옵션 스칼라 dict
                                     키: pcr_volume, iv_skew, max_pain_dist_pct, atm_iv

출력
└── ndarray (seq_len, feature_dim)   float32  모델 직접 입력 가능
```

### 2.2 OB 블록 조립

```python
# features.build_sequence()의 OB 조립 로직(거의 동일한 의사코드)

ob_keys = list(OB_KEYS)
cd_keys = list(CD_KEYS)
opt_keys = list(OPT_KEYS)

ob_arr = np.zeros((seq_len, len(ob_keys)), dtype=np.float32)
tail = ob_records[-seq_len:] if ob_records else []
start = seq_len - len(tail)
for i, rec in enumerate(tail):
    ob_arr[start + i] = [float(rec.get(k, 0.0) or 0.0) for k in ob_keys]
```

**Zero-padding의 의미**: 시스템 시작 직후 오더북 버퍼가 덜 찬 경우입니다. 패딩 위치의 모든 피처가 0이므로 모델이 "신호 없음"으로 학습해야 합니다 — 학습 데이터에서도 동일한 상황이 재현되어야 합니다.

### 2.3 CD 블록 조립 — `_ts_epoch` 기반 매핑 (권장)

```python
# features.build_sequence()의 candle 매핑 로직(거의 동일한 의사코드)

cd_arr = np.zeros((seq_len, len(cd_keys)), dtype=np.float32)

if candle_df is not None and (not candle_df.empty):
    # tail의 모든 레코드에 _ts_epoch가 있어야 timestamp-align 경로를 사용
    use_ts = bool(tail) and all(r.get("_ts_epoch") is not None for r in tail)

    if use_ts:
        cdf = candle_df

        # candle_df index가 DatetimeIndex가 아니면 변환 시도
        if not isinstance(cdf.index, pd.DatetimeIndex):
            try:
                cdf = cdf.copy()
                cdf.index = pd.to_datetime(cdf.index)
            except Exception:
                cdf = candle_df

        if isinstance(cdf.index, pd.DatetimeIndex) and len(cdf.index) >= 1:
            last_min = cdf.index.max()
            last_complete_min = last_min

            # 현재 분봉이 "미완성"으로 들어온 경우 마지막 완성 분봉으로 clamp
            try:
                now_min = datetime.fromtimestamp(float(tail[-1].get("_ts_epoch"))).replace(second=0, microsecond=0)
                if last_min.replace(second=0, microsecond=0) == now_min and len(cdf.index) >= 2:
                    last_complete_min = cdf.index[-2]
            except Exception:
                last_complete_min = last_min

            for i, rec in enumerate(tail):
                ts = float(rec.get("_ts_epoch"))
                minute = datetime.fromtimestamp(ts).replace(second=0, microsecond=0)
                if minute > last_complete_min:
                    minute = last_complete_min

                if minute in cdf.index:
                    cd_arr[start + i] = cdf.loc[minute, cd_keys].values.astype(np.float32)
                else:
                    # best-effort: nearest previous candle (searchsorted)
                    pos = int(cdf.index.searchsorted(minute, side="right")) - 1
                    if pos >= 0:
                        cd_arr[start + i] = cdf.iloc[pos][cd_keys].values.astype(np.float32)
        else:
            use_ts = False

    # fallback: 선형 매핑(legacy)
    if not use_ts:
        cd_vals = candle_df[cd_keys].values.astype(np.float32)
        bars = int(len(cd_vals))
        for row in range(int(seq_len)):
            bar_idx = min(bars - 1, int(row * bars / int(seq_len)))
            cd_arr[row] = cd_vals[bar_idx]
```

> `_ts_epoch`는 현재 `pipeline.py`의 FH0 버퍼링에서 `ob.setdefault("_ts_epoch", int(sec_key))`로 첨부됩니다.

### 2.4 OPT 블록 조립

```python
# features.build_sequence()의 OPT 조립 로직(거의 동일한 의사코드)

opt_row = np.zeros((len(opt_keys),), dtype=np.float32)
if opt_features:
    for j, k in enumerate(opt_keys):
        v = opt_features.get(k)
        if v is not None:
            opt_row[j] = float(v)

opt_arr = np.tile(opt_row, (seq_len, 1))
```

### 2.5 최종 조립

```python
sequence = np.concatenate([ob_arr, cd_arr, opt_arr], axis=1)
# shape: (seq_len, feature_dim)  dtype: float32
```

---

## 3) 모델 정의: `PriceTransformer`

- **구현**: `prediction/model.py`
- **목적**: `(batch, seq_len, feature_dim)` → `P(up)` ∈ [0, 1]

### 3.1 아키텍처 개요

```
입력: (B, 60, 16)
    │
    ▼ Linear(16 → d_model=64)
(B, 60, 64)
    │
    │ [CLS] token 삽입
    ▼
(B, 61, 64)  ← [CLS], t-59, t-58, ..., t(현재)
    │
    ▼ PositionalEncoding (sinusoidal, max_len=61)
    │
    ▼ TransformerEncoder (n_layers=2, n_heads=4, d_ff=128)
(B, 61, 64)
    │
    │ CLS 위치(index 0)만 추출
    ▼
(B, 64)
    │
    ▼ LayerNorm → Linear(64→32) → GELU → Dropout → Linear(32→1) → Sigmoid
    ▼
(B,)  ← P(up) per sample
```

### 3.2 CLS token 방식

```python
cls = self.cls_token.expand(batch, -1, -1)   # (B, 1, 64)
x   = torch.cat([cls, x], dim=1)             # (B, 61, 64)
x   = self.encoder(x)
out = self.head(x[:, 0])                     # CLS 위치만 사용
```

**왜 CLS token인가**: BERT 스타일의 CLS token은 encoder가 시퀀스 전체 정보를 하나의 벡터로 요약하도록 유도합니다. "마지막 시점만 사용(`x[:, -1]`)"하는 방식은 최근 데이터에 편향되어 60초 전 패턴을 무시할 수 있습니다. CLS token은 self-attention을 통해 임의 위치의 정보를 자유롭게 집계합니다.

### 3.3 하이퍼파라미터 기본값

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `feature_dim` | 16 | 입력 피처 수 (ob+cd+opt) |
| `d_model` | 64 | Transformer 내부 차원 |
| `n_heads` | 4 | Multi-head attention 헤드 수 |
| `n_layers` | 2 | Encoder 레이어 수 |
| `d_ff` | 128 | FFN 중간 차원 (d_model × 2) |
| `seq_len` | 60 | 입력 시퀀스 길이 |
| `dropout` | 0.1 | Dropout 비율 |

**파라미터 수**: 약 40,000개 (경량 모델). 실시간 CPU inference에 적합합니다.

### 3.4 Pre-LayerNorm 구조

```python
encoder_layer = nn.TransformerEncoderLayer(
    ...,
    norm_first=True,   # Pre-LN: 학습 안정성 향상
)
```

`norm_first=True`는 각 sub-layer 전에 LayerNorm을 적용하는 Pre-LN 구조입니다. Post-LN 대비 학습 초기 기울기 소실 문제가 적고, 더 높은 학습률을 사용할 수 있습니다.

### 3.5 torch 없는 환경

```python
# model.py
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

if not _TORCH_AVAILABLE:
    class PriceTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required")
```

torch가 없으면 `PriceTransformer`를 import는 할 수 있지만 생성할 수 없습니다. `predictor.py`는 이 경우를 감지해 rule-based fallback으로 전환합니다.

---

## 4) 추론 래퍼: `TransformerPredictor`

- **구현**: `prediction/predictor.py`
- **역할**: 서빙 경로에서 모델 추론 또는 fallback 처리

### 4.1 초기화 흐름

```python
TransformerPredictor(
    weights_path="prediction/weights/transformer_5m.pt",
    feature_dim=19,
    seq_len=60,
    device="cpu",
    buy_threshold=0.62,      # config.json에서 주입
    sell_threshold=0.38,     # config.json에서 주입
)
```

```
weights_path 존재?
    ├─ YES → PriceTransformer.load() → model.eval()
    │        [Transformer inference 경로]
    └─ NO  → self._model = None
             [Rule-based fallback 경로]
```

### 4.2 weights 로딩

```python
state = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
```

- `map_location=device`: GPU weights를 CPU에서 로딩 가능
- `weights_only=True`: 임의 코드 실행 방지 (보안)
- `model.eval()`: Dropout/BatchNorm을 inference 모드로 전환

### 4.3 Transformer inference 경로

```python
seq = input.sequence
x = torch.tensor(seq[np.newaxis], dtype=torch.float32)
# shape: (1, seq_len, feature_dim) — batch_size=1

with torch.no_grad():          # 기울기 계산 비활성화 (메모리/속도)
    prob = float(model(x).item())   # Sigmoid 출력: [0, 1]
```

**`torch.no_grad()` 필수**: 서빙에서 기울기를 계산하면 메모리와 시간이 낭비됩니다.

### 4.4 분류 임계값 적용

```python
if p >= buy_threshold:    # 기본 0.62
    signal = "BUY"
elif p <= sell_threshold: # 기본 0.38
    signal = "SELL"
else:
    signal = "HOLD"       # [0.38, 0.62] 구간은 HOLD
```

```python
margin = abs(p - 0.5)
if margin >= confidence_high_margin and spread <= confidence_spread_max_for_high:
    confidence = "HIGH"    # p ≥ 0.65 or p ≤ 0.35
elif margin >= confidence_mid_margin:
    confidence = "MEDIUM"  # p ∈ [0.58, 0.65) or (0.35, 0.42]
else:
    confidence = "LOW"     # p ∈ [0.42, 0.58]
```

위 `confidence_*` 값은 `config.json`의 아래 키로 조정할 수 있습니다:

- `prediction.confidence_high_margin`
- `prediction.confidence_mid_margin`
- `prediction.confidence_spread_max_for_high`

**임계값 설정 가이드**:

| 목적 | buy_threshold | sell_threshold | 특징 |
|------|--------------|----------------|------|
| 기본값 | 0.62 | 0.38 | HOLD 구간 24% |
| 신중형 | 0.65 | 0.35 | HOLD 구간 30%, 진입 빈도 감소 |
| 공격형 | 0.55 | 0.45 | HOLD 구간 10%, 오신호 증가 가능 |

모델 캘리브레이션 후 Precision-Recall 곡선을 보고 운영 목표(승률 vs 빈도)에 맞게 조정하세요.

### 4.5 Rule-based fallback 상세

weights 없거나 inference 실패 시 OBI/캔들 기반 휴리스틱으로 prob를 산출합니다.

```
pressure_ob = 0.75 × obi + 0.25 × level1_ratio   ← OB 압력 (가중 평균)
mom         = clip(ret3 × 50, -1, 1)              ← 3분봉 모멘텀 정규화
vol_boost   = clip(vol_accel - 1.0, 0, 1)         ← 거래량 가속 (0~1)

pressure    = pressure_ob + 0.10 × mom × (0.5 + 0.5 × vol_boost)
spread_pen  = clip(spread / 5.0 × 0.25, 0, 0.25) ← spread 페널티

prob        = clip(0.5 + clip(pressure, -0.48, 0.48) - spread_pen, 0, 1)
```

**fallback의 한계**: rule-based는 과거 데이터로 학습된 것이 아닙니다. OBI가 높아도 실제 체결이 없으면 허수호가일 수 있고, 분봉 모멘텀만으로 5분 후 방향을 맞추기 어렵습니다. **weights 파일 생성 후에는 반드시 실제 inference 경로로 전환해야 합니다.**

### 4.6 `inference_mode` 투명성 (권장 추가)

현재 `TransformerPredictionResult`에는 inference 경로 정보가 없습니다. LLM이 컨텍스트를 받을 때 rule-based 결과인지 알 수 없습니다.

```python
# 권장 수정
@dataclass
class TransformerPredictionResult:
    prob: float
    signal: str
    confidence: str
    feature_snapshot: Dict[str, Any]
    inference_mode: str = "transformer"   # "transformer" | "rule_based"
```

LLM 컨텍스트에도 포함:
```python
snapshot["transformer"]["inference_mode"] = t_res.inference_mode
```

이를 통해 LLM은 "rule-based 결과이므로 더 보수적으로 판단"하는 로직을 rationale에 반영할 수 있습니다.

---

### 4.6.1 TFT(Temporal Fusion Transformer) predictor (현행 구현)

본 저장소는 predictor 입력을 `ModelInput`으로 표준화했으며, TFT를 위해 time feature 텐서를 함께 제공합니다.

Known-future covariates(현행 파이프라인 생성 기준):
- `future_known.shape == (prediction_minutes*60, 11)`
- 컬럼(순서):
  - `dte_scaled`
  - `dow_onehot_0..6`
  - `tod_sin`, `tod_cos` (정규장 09:00~15:30 기준; 장외 시간은 0~1로 clamp)
  - `is_expiry_week`

권장:
- 학습/서빙에서 feature ordering이 어긋나지 않도록 `ModelInput.schema_version`을 활용해 정합성을 검증하세요.
- TFT는 `ModelInput.sequence`(past_unknown)와 time features(past_known/future_known)를 함께 사용하되,
  외부 출력 계약은 기존과 동일하게 `P(up)` → `prob/signal/confidence`로 유지합니다.

## 4.7 전체 서빙 파이프라인 — 데이터 흐름 상세

실시간 운영 중 하나의 예측이 만들어지기까지 전체 경로입니다.

```
eBest 실시간 콜백
    │
    ├─ FC0 틱 ──────────────────→ tick_processor
    │                              └─ futures_ticks deque 추가
    │                              └─ futures_minute_data[HHMM] 집계 (OHLCV: Open/High/Low/Close/Volume)
    │
    ├─ OC0 틱 ──────────────────→ tick_processor
    │                              └─ call_options / put_options dict 갱신
    │
    ├─ OH0 틱 (옵션 호가) ───────→ JSONL 로그/정규화(tick_norm)
    │                              └─ tick_processor.process_option_quote_tick()로 라우팅
    │                                 call_options/put_options 스냅샷에 bid/ask + 5단계 depth/qty 반영
    │
    ├─ JIF 틱 (장운영) ───────────→ JSONL 로그/모니터링
    │                              └─ 장구분(jangubun) + 장상태(jstatus)
    │                              └─ 현행 구현에서는 피처/예측 입력으로 사용하지 않음
    │
    └─ FH0 틱 ──────────────────→ pipeline._ob_records 버퍼 (1Hz)
                                   └─ calc_orderbook_features() 호출
                                   └─ ob["_ts_epoch"] = sec_key 기록

                                              ↓ (5분마다 get_prediction() 호출)

pipeline.get_prediction()
    │
    ├─ tick_processor.get_futures_minute_df()
    │   └─ calc_candle_features(df) ──→ candle_df
    │
    ├─ option_features.build_option_snapshot(calls, puts, price)
    │   └─ PCR / IV Skew / Max Pain / ATM IV ──→ opt_features dict
    │
    ├─ features.build_sequence(ob_records, candle_df, opt_features)
    │   └─ ndarray (60, 19) ──→ sequence
    │
    ├─ TransformerPredictor.predict(sequence)
    │   ├─ [weights 있음] torch inference → prob ∈ [0,1]
    │   └─ [weights 없음] rule-based → prob ∈ [0,1]
    │   └─ threshold 적용 → signal (BUY/SELL/HOLD) + confidence
    │
    ├─ context_builder.build_context(snapshot)
    │   └─ LLM 시스템 프롬프트 + 시장 컨텍스트 JSON 조립
    │
    └─ llm_judge.judge(context)  (use_llm=True일 때)
        └─ Claude/GPT/Gemini API 호출 → action/risk_level/rationale/caution
        └─ 결과 통합 → 최종 prediction dict 반환
```

### 4.7.1 OH0 틱 사용 포인트

현재 옵션 피처(`OPT_KEYS`: PCR/IV/MaxPain 등)는 `tick_processor.call_options/put_options`에 누적된 스냅샷을 기반으로 계산되며,
이 스냅샷은 주로 **OC0(옵션 체결)** 에 의해 갱신됩니다.

현행 구현에서는 **OH0(옵션 호가)** 를 `tick_processor.process_option_quote_tick()`에서 처리하여,
옵션 스냅샷에 다음과 같은 마이크로스트럭처 정보가 반영됩니다.

- **bid/ask 및 5단계 depth/잔량** (옵션 유동성/스프레드/호가 쏠림)
- **체결이 드문 옵션**의 최신 상태(OC0만으로는 갱신이 느림)

관련 구현 포인트는 다음과 같습니다.

- `tick_normalizer.normalize_realtime_tick(OH0)` → depth/qty(list) 기반 정규화
- `tick_processor.process_tick()` → `TRCode.OPTIONS_QUOTE(OH0)` 라우팅
- `option_features.calc_atm_microstructure()` → OH0 기반 신규 피처(스프레드/불균형/유동성) 계산

**레이턴시 예산**: 5분 주기 예측에서 각 단계의 목표 시간입니다.

| 단계 | 목표 | 비고 |
|------|------|------|
| `build_sequence` | < 1ms | numpy 연산, CPU |
| Transformer inference | < 5ms | CPU, batch=1, ~40k params |
| LLM API 호출 | < 8s | SDK timeout 설정 필요 |
| 전체 `get_prediction` | < 10s | LLM이 지배적 |

---

## 4.8 Self-Attention 동작 원리 — 왜 호가 데이터에 유효한가

Transformer의 핵심은 **Self-Attention**입니다. 시퀀스의 각 위치(초)가 다른 모든 위치에 "얼마나 주의를 기울일지"를 학습합니다.

### KP200 선물에서의 의미

```
시간 t-30s: OBI 급등 (매수 폭발)
시간 t-15s: spread 축소 (체결 집중)
시간 t-5s:  vol_accel 급증
시간 t:     현재 상태

Self-Attention이 학습하는 예시 패턴:
  "t-30s의 OBI 급등이 t-5s의 vol_accel 급증과
   결합하면 5분 후 상승 확률이 높다"
```

단순 규칙(OBI > 0.3 AND vol_accel > 1.5)과 다른 점은 **타이밍과 순서**까지 학습한다는 것입니다. "30초 전에 OBI가 올라갔다"는 정보가 "지금 OBI가 높다"는 정보와 독립적으로 attention weight에 반영됩니다.

### Multi-Head Attention (n_heads=4)

4개의 head가 서로 다른 관계를 병렬로 학습합니다.

```
Head 1: OBI 추세 — 과거 OBI가 현재 OBI와 얼마나 일관되는가
Head 2: 모멘텀 패턴 — 캔들 수익률 변화 패턴
Head 3: 유동성 변화 — spread/totbidrem/totofferrem의 변화
Head 4: 다중 신호 — OB + 캔들 + 옵션의 복합 관계
```

### Positional Encoding의 역할

Self-Attention 자체는 위치 정보가 없습니다 (토큰의 순서를 모릅니다). Positional Encoding을 더해 "이 값이 60초 전 것인지 1초 전 것인지"를 모델에게 알려줍니다.

```
x[t] = feature_projection(seq[t]) + PE(t)

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**CLS token의 위치**: `PE(0)`이 CLS token에 할당됩니다. CLS는 위치 0에 고정되어 시퀀스 전체 정보를 집계하는 특수한 "집약 포인트" 역할을 합니다. 이 설계 덕분에 CLS token은 어느 시점의 피처도 균등하게 참조할 수 있습니다.

---

## 4.9 피처 정규화 전략

현재 코드에는 **명시적 정규화(StandardScaler 등)가 없습니다**. 각 피처가 서로 다른 스케일을 가지므로 이에 대한 이해가 필요합니다.

### 현재 각 피처의 실제 스케일

| 피처 | 전형적 범위 | 정규화 여부 |
|------|------------|-----------|
| `obi` | [-1, 1] | 수식 자체가 정규화됨 |
| `spread` | [0.05, 2.0] | 미정규화 — 틱 단위 절대값 |
| `level1_ratio` | [-1, 1] | 수식 자체가 정규화됨 |
| `bid_slope` | [-200, 200] | **미정규화 — 스케일 큼** |
| `offer_slope` | [-200, 200] | **미정규화 — 스케일 큼** |
| `totbidrem` | [100, 5000] | **미정규화 — 절대 잔량** |
| `totofferrem` | [100, 5000] | **미정규화 — 절대 잔량** |
| `ret1` | [-0.005, 0.005] | 비율이나 스케일 매우 작음 |
| `ret3` | [-0.015, 0.015] | 비율이나 스케일 매우 작음 |
| `vol_accel` | [0, 5] | 비율 (1.0 = 평균) |
| `pcr_volume` | [0.5, 2.0] | 비율 |
| `atm_iv` | [0.10, 0.40] | 절대 IV (소수) |

**문제점**: `totbidrem`(~1000)과 `obi`(~0.3)가 같은 Linear 레이어에 들어가면, 스케일이 큰 피처가 학습 초기에 weight를 지배할 수 있습니다. Transformer의 LayerNorm이 어느 정도 보정하지만, 명시적 정규화 없이는 학습 안정성과 수렴 속도에 영향이 있습니다.

### 권장 정규화 방안

**방법 1 — 피처 수식 자체 정규화 (권장, 서빙-학습 정합 보장)**

```python
# features.py에서 수식 단계에 정규화 내재화
"totbidrem":   totbid / (totbid + totask + 1e-9),   # → [0, 1] 비율
"totofferrem": totask / (totbid + totask + 1e-9),   # → [0, 1] 비율
"bid_slope":   bid_slope / (totbid / 5 + 1),        # 평균 단계 잔량 대비 기울기
"spread":      spread / close_price * 100 if close_price > 0 else spread,  # bps
"ret1":        ret1 * 1000,                          # ±5 수준으로 확대
"ret3":        ret3 * 1000,
```

이 방식은 학습/서빙 모두 동일한 수식을 통과하므로 별도 scaler 파일이 필요 없습니다.

**방법 2 — 학습 통계 저장 후 서빙에서 적용**

```python
# data_builder.py — 학습 데이터 통계 계산 후 저장
X_flat = X.reshape(-1, feature_dim)
feat_mean = X_flat.mean(axis=0).astype(np.float32)   # (16,)
feat_std  = X_flat.std(axis=0).astype(np.float32) + 1e-9

np.savez("dataset_5m.npz", X=X, y=y,
         feat_mean=feat_mean, feat_std=feat_std)

# predictor.py — 서빙 시 동일 통계로 정규화
feat_mean = data.get("feat_mean")
feat_std  = data.get("feat_std")
if feat_mean is not None:
    sequence = (sequence - feat_mean) / feat_std
```

이 방식은 학습 데이터 분포에 의존하므로, 시장 체제가 크게 바뀌면 재학습이 필요합니다.

---

## 4.10 모델 성능 평가 방법론

학습 후 `val_acc`만으로는 실제 운영 성능을 판단하기 어렵습니다. 다음 지표들을 함께 확인해야 합니다.

### 이진 분류 지표

```python
from sklearn.metrics import classification_report, roc_auc_score

# val_ds 전체에 대한 예측
probs = []
labels = []
model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        probs.extend(model(xb.to(device)).cpu().numpy())
        labels.extend(yb.numpy())

probs  = np.array(probs)
labels = np.array(labels)
preds  = (probs >= 0.5).astype(int)

print(classification_report(labels, preds, target_names=["down", "up"]))
print(f"AUC-ROC: {roc_auc_score(labels, probs):.4f}")
```

**지표 해석**:

| 지표 | 의미 | 목표 |
|------|------|------|
| Accuracy | 전체 정확도 | > 53% |
| Precision (up) | BUY 신호 중 실제 상승 비율 | > 55% (오신호 줄이기) |
| Recall (up) | 실제 상승 중 BUY 신호 비율 | 운영 전략에 따라 조정 |
| AUC-ROC | 분류 능력 종합 | > 0.55 |
| F1 | Precision/Recall 균형 | 참고용 |

### Calibration 확인

모델이 "prob=0.65"를 출력할 때 실제로 65% 확률로 상승해야 합니다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, fraction_pos, "s-", label="모델")
plt.plot([0, 1], [0, 1], "k--", label="완벽한 캘리브레이션")
plt.xlabel("예측 확률"); plt.ylabel("실제 양성 비율")
plt.title("Calibration Curve"); plt.legend(); plt.savefig("calibration.png")
```

**캘리브레이션이 나쁜 경우**: 모델이 0.5 근처에 확률을 집중시키거나, 0.6이라 해도 실제 50%밖에 안 맞으면 threshold 설정이 무의미해집니다. 이 경우 Platt Scaling이나 Isotonic Regression으로 사후 보정을 고려하세요.

### 방향성 적중률 (실제 운영 지표)

```python
# BUY/SELL 신호만 평가 (HOLD 제외)
buy_threshold  = 0.62
sell_threshold = 0.38

buy_mask  = probs >= buy_threshold
sell_mask = probs <= sell_threshold

buy_hit  = np.mean(labels[buy_mask]  == 1) if buy_mask.any()  else 0.0
sell_hit = np.mean(labels[sell_mask] == 0) if sell_mask.any() else 0.0
hold_rate = 1 - (buy_mask | sell_mask).mean()

print(f"BUY  신호 적중률: {buy_hit:.1%}  (n={buy_mask.sum()})")
print(f"SELL 신호 적중률: {sell_hit:.1%}  (n={sell_mask.sum()})")
print(f"HOLD 비율: {hold_rate:.1%}")
```

이것이 `ebest_live.py`의 `eval_dir_hit_count / eval_dir_count`와 동일한 개념입니다. 운영 중 이 값이 50% 이상이면 모델이 유효합니다.

---

## 5) 학습 데이터셋 생성 (train/serve 정합)

- **구현**: `prediction/data_builder.py`
- **입력**: `ticks_replay_*.jsonl` (eBest 실시간 틱 저장본)
- **출력**: `dataset_*.npz` (X: ndarray, y: ndarray)

### 5.1 데이터 흐름

```
ticks_replay_*.jsonl
    │
    ├─ FC0 레코드 → 분봉 OHLCV 재구성 → calc_candle_features() → CD 블록
    ├─ FH0 레코드 → calc_orderbook_features() → OB 블록 (60초 버퍼)
    └─ OC0 레코드 → build_option_snapshot() → OPT 블록

FC0 레코드 (샘플 생성 트리거)
    │ price > 0 AND ob_buf 충분?
    ├─ YES
    │   │ N분 후 가격 있음?
    │   ├─ YES → (X, y) 생성
    │   └─ NO  → skip
    └─ NO → skip

출력: X (N, 60, 16), y (N,) — 0 or 1
```

### 5.2 레이블 생성 규칙

```python
cur_m  = hhmm_to_minutes(chetime[:4])        # 현재 분 (HH×60 + MM)
tgt_hhmm = f"{(cur_m + horizon) // 60 % 24:02d}{(cur_m + horizon) % 60:02d}"

label = 1 if minute_prices[tgt_hhmm] > price else 0
```

- `horizon = prediction_minutes` (기본 5)
- `minute_prices[hhmm]` = 해당 분에 마지막으로 체결된 FC0 가격

**동률 처리**: `price_future > price_now`이므로 동률(`==`)은 0(down)으로 처리됩니다. 실제로 선물 틱 단위(0.05p)에서 동률은 드물지만, 필요하면 `>=`로 변경할 수 있습니다.

### 5.3 OB 블록 조립 (현재 구현 — 정상)

```python
ob_arr = np.zeros((seq_len, len(OB_KEYS)), dtype=np.float32)
tail   = list(ob_buf)[-seq_len:]
start  = seq_len - len(tail)
for i, r in enumerate(tail):
    ob_arr[start + i] = [float(r.get(k, 0.0) or 0.0) for k in OB_KEYS]
```

### 5.4 CD 블록 조립 (현재 구현)

```python
# data_builder.py는 FC0 틱으로 분봉 OHLCV를 재구성하고, calc_candle_features()로
# candle_df를 만든 뒤 샘플 시점의 "마지막 완성 분봉" 피처를 seq_len 전체에 타일링합니다.

cd_arr = np.zeros((seq_len, len(CD_KEYS)), dtype=np.float32)
if candle_df is not None and (not candle_df.empty):
    # Use the last completed candle at sampling time.
    if minute in candle_df.index:
        row = candle_df.loc[minute, CD_KEYS].values.astype(np.float32)
        cd_arr[:] = row
```

### 5.5 OPT 블록 조립 (현재 구현)

```python
# data_builder.py는 OC0 틱으로 calls/puts 스냅샷을 누적하고, build_option_snapshot()을
# 호출해 OPT 스칼라 피처를 만들고 seq_len 전체에 타일링합니다.

opt_arr = np.zeros((seq_len, len(OPT_KEYS)), dtype=np.float32)
opt_snap = build_option_snapshot(calls, puts, float(price))
opt_row = np.array([
    float(opt_snap.get("pcr_volume") or 1.0),
    float(opt_snap.get("iv_skew") or 1.0),
    float(opt_snap.get("max_pain_dist_pct") or 0.0),
    float(opt_snap.get("atm_iv") or 0.0),
], dtype=np.float32)
opt_arr[:] = np.tile(opt_row, (seq_len, 1))
```

> 참고: 옵션 미구독/데이터 부족 시에는 기본값(PCR=1.0, skew=1.0, dist=0.0, iv=0.0)이 들어가며, 이 경우 OPT 블록이 0이 아닐 수도 있습니다.

### 5.6 데이터 품질 체크

```python
# 권장 검증 코드
import numpy as np

data = np.load("dataset_5m.npz")
X, y = data["X"], data["y"]

print(f"Shape: X={X.shape}, y={y.shape}")
print(f"Pos rate: {y.mean():.1%}")  # 이상적으로 45~55%

# 각 블록 값 분포 확인
ob_block  = X[:, :, :7]
cd_block  = X[:, :, 7:12]
opt_block = X[:, :, 12:]

print(f"OB  - mean={ob_block.mean():.4f}, std={ob_block.std():.4f}")
print(f"CD  - mean={cd_block.mean():.4f}, std={cd_block.std():.4f}")
print(f"OPT - mean={opt_block.mean():.4f}, std={opt_block.std():.4f}")

# CD/OPT가 모두 0이면 train-serve skew 존재
assert cd_block.std() > 0, "CD block is all zeros — data_builder 미구현"
assert opt_block.std() > 0, "OPT block is all zeros — data_builder 미구현"
```

---

## 6) 학습 실행 (`train.py`)

### 6.1 실행 순서

```bash
# Step 1) replay 데이터 수집 (운영 중 --out-ticks 옵션으로 저장)
python main.py --out-ticks ticks_replay_20250210.jsonl --duration-sec 25200

# Step 2) 데이터셋 생성
python -m prediction.data_builder \
  --files ticks_replay_20250210.jsonl ticks_replay_20250211.jsonl \
  --out dataset_5m.npz --seq-len 60 --horizon 5

# Step 3) 학습
python train.py \
  --data dataset_5m.npz \
  --out prediction/weights/transformer_5m.pt \
  --epochs 50 --batch-size 256 --lr 1e-3
```

TFT 학습(현행, 별도 스크립트):

```bash
# 최근 N일 데이터 merge (rolling 재학습)
python merge_datasets.py --pattern "dataset_tft_*.npz" --last-days 20 --out dataset_tft_merged_last20.npz

# TFT 학습
python train_tft.py --data dataset_tft_merged_last20.npz --out prediction/weights/tft_5m.pt
```

### 6.2 시간순 분할 (필수)

```python
# 올바른 방식 — 시간 순서 유지
n_train    = int(N * 0.8)
train_ds   = TensorDataset(X[:n_train], y[:n_train])
val_ds     = TensorDataset(X[n_train:], y[n_train:])

# 잘못된 방식 — 데이터 누수
train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, N - n_train])
```

**랜덤 분할의 문제**: 10:00의 샘플이 검증에, 10:05의 샘플이 학습에 들어가면, 모델이 사실상 미래 데이터를 보고 학습하는 것과 같습니다. 검증 accuracy가 실제보다 낙관적으로 보여 모델 선택이 왜곡됩니다.

### 6.3 손실 함수 — 클래스 불균형 처리

```python
# 상승/하락 비율이 50:50이 아닐 때 pos_weight로 보정
pos_rate   = float(y.mean())                      # 상승 비율
pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])  # 하락 많으면 상승에 더 큰 가중치

loss = -(
    pos_weight × y × log(prob + ε)
    + (1 - y)  × log(1 - prob + ε)
).mean()
```

KP200 선물 데이터에서 5분 후 상승 비율은 보통 48~52% 수준입니다. `pos_rate ≈ 0.5`이면 `pos_weight ≈ 1.0`으로 일반 BCE와 동일합니다. 데이터가 한 방향으로 편향된 날(급등/급락장)이 많으면 이 보정이 중요합니다.

### 6.4 학습 모니터링 지표

```
epoch  50/50 loss=0.6721 val_acc=55.23%
  -> checkpoint saved (best=55.23%): prediction/weights/transformer_5m.pt
```

- `val_acc > 55%`: 의미 있는 학습 진행 중
- `val_acc ≈ 50%`: 랜덤 추측 수준 — 피처 품질 또는 데이터 양 확인 필요
- `val_acc > 60%`: 과적합 의심 — 학습/검증 분포가 다를 수 있음 (train-serve skew)
- `loss`가 안 떨어지면: lr 조정 또는 데이터 품질 문제

### 6.5 학습 데이터 권장 기준

| 항목 | 최소 | 권장 |
|------|------|------|
| 샘플 수 | 1,000개 | 10,000개 이상 |
| 거래일 수 | 5일 | 20일 이상 |
| 시장 상황 | — | 상승/하락/횡보 포함 |
| pos rate | 45~55% | 48~52% |

---

## 7) 운영 체크리스트

### 7.1 빌드 전 확인

- [ ] `feature_dim=19` (ob 7 + cd 5 + opt 7) 정합 확인 (권장: 단일 상수로 관리)
- [ ] `data_builder.py` CD/OPT 블록이 0이 아닌 실제 값으로 채워짐
- [ ] `train.py` 랜덤 분할 → 시간순 분할로 교체
- [ ] `build_sequence()`에 `_ts_epoch` 기반 candle 매핑 추가
- [ ] `pipeline.py` FH0 버퍼링 시 `_ts_epoch` 기록

### 7.1.1 만기주(Expiry week) weights 동결 운영

옵션 만기주는 시장 미세구조/옵션 수급 변화로 학습 데이터 분포가 급격히 변할 수 있어,
"최근 20일 rolling 재학습"로 만들어진 최신 weights가 오히려 불안정해질 수 있습니다.

현행 구현은 만기주 중에서도 **월~목(만기일) 4거래일에 한해**, 만기주 시작 직전까지 학습된 weights로 동결합니다.

- 만기주 판별: `utils.get_expiry_week_info()` (두 번째 목요일이 포함된 주)
- 동결 적용 기간: 만기주 week_start(월요일) ~ 만기일(두 번째 목요일)
- 동결 기준일(cutoff): 만기주 week_start(월요일) - 1일 (즉, 직전 일요일)
- 사용 weights:
  - `prediction/weights/transformer_5m_YYYYMMDD.pt`
  - `prediction/weights/tft_5m_YYYYMMDD.pt`
  - 위 파일이 없으면 기본 파일(`transformer_5m.pt`, `tft_5m.pt`)로 fallback

학습 스크립트는 best checkpoint를 저장한 뒤 날짜 태그 사본을 자동 생성합니다.

- `train.py` / `train_tft.py`: `--tag-date YYYYMMDD` (미지정 시 당일 날짜로 자동)

실행 시 선택 내역은 로그로 출력됩니다.

- 예: `[Weights] selection=expiry_week_freeze_mon_thu cutoff=YYYY-MM-DD transformer=... tft=...`

### 7.1.2 옵션 구독 설정(현행)

옵션 실시간 구독 범위는 `config.json`의 `options_subscription`으로 제어됩니다.

- `itm`: ATM 기준 ITM 쪽 개수
- `otm_open_min`: OTM 구독 조건 — 옵션 시가(open) 하한
- `max_otm_calls`, `max_otm_puts`: OTM 콜/풋 구독 상한 (0이면 무제한)
- `wait_sec`: 옵션 구독 전 대기 시간

OTM 구독은 로그인 후 `t2301`로 받은 open map 기준으로 동적으로 결정되며,
open map 수신이 실패하면 **OTM은 0으로 축소**되어 ITM/ATM만 구독합니다.

- 예: `[eBest] t2301 open map unavailable; subscribing ITM/ATM only (OTM=0)`

### 7.2 학습 후 확인

- [ ] `prediction/weights/transformer_5m.pt` 생성 확인
- [ ] `predictor.py`가 weights 로딩 성공 로그 출력: `[Predictor] Transformer weights loaded:`
- [ ] `get_prediction()` 결과에 `"prob"` 값이 rule-based 범위(0.3~0.7)를 벗어나는지 확인
- [ ] `inference_mode = "transformer"` 로 표시되는지 확인 (권장 추가 후)

### 7.3 서빙 중 확인

- [ ] `fo0_age_sec < 2.0` — FH0 실시간 수신 중
- [ ] `ob_records_len ≈ seq_len(60)` — 오더북 버퍼 충분
- [ ] `consensus = true` — Transformer + LLM 방향 일치
- [ ] LLM `rationale`에 PCR/IV Skew/Max Pain 언급 — 옵션 컨텍스트 정상 주입

### 7.4 테스트 실행

```bash
# 단위 테스트 (pytest 권장)
python -m pytest -q

# 전체 스모크 테스트
python main.py --test
```

---

## 8) 자주 발생하는 문제와 해결 방법

| 증상 | 원인 | 해결 |
|------|------|------|
| `prob` 항상 0.48~0.52 | weights 없어 rule-based, OBI ≈ 0 | FH0 구독 확인, weights 학습 |
| `prob` 분포 이상 (≈0.5 집중) | train-serve skew (CD/OPT 0패딩) | data_builder CD/OPT 채우기 |
| `signal` 항상 HOLD | threshold 너무 넓거나 OBI 신호 약 | 임계값 조정 또는 더 많은 데이터 |
| LLM rationale 빈약 | 시스템 프롬프트 인코딩 손상 | context_builder.py UTF-8 확인 |
| `ob_records_len = 0` | FH0 미구독 또는 `_invalid` 연속 | FH0 구독 확인, 스키마 키 확인 |
| val_acc 학습 중 감소 | 과적합 또는 시간순 분할 미적용 | random split → 시간순 분할 |
| `torch` ImportError | torch 미설치 | `pip install torch` 또는 weights 삭제 후 rule-based 사용 |

---

*문서 갱신일: 2026-02-14 | 기준 코드 버전: v1.0.0*
