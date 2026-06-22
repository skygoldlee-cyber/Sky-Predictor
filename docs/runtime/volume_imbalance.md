# Volume Imbalance (OBI) 로직 정리

## 용어

이 프로젝트에서 말하는 **Volume Imbalance**는 실시간 호가(FH0)로부터 계산되는 **Orderbook Imbalance(OBI)** 를 의미합니다.

- `obi`: 호가 잔량 기반 **전체(또는 대체된 총합) 불균형 지표**
- `level1_ratio`: 1호가(L1) 잔량 기반 **L1 불균형 지표**

> 구현 파일 기준으로는 “volume imbalance”라는 이름보다는 `obi`, `level1_ratio`로 사용됩니다.

## 데이터 소스

### 1) eBest realtime: `FH0` (선물 실시간 호가)

- `PredictionPipeline.add_realtime_tick()`에서 `trcode == FH0` 인 tick을 처리하며,
  `calc_orderbook_features()`로 전달 가능한 형태인지 검사한 뒤(orderbook snapshot처럼 보이는지) 피처 계산에 사용합니다.

### 2) tick 정규화(`tick_normalizer.py`)

- `FH0/OH0`는 `tick_norm`에 리스트 형태로 정규화됩니다.
  - `offerhos`, `bidhos` (len=5)
  - `offerrems`, `bidrems` (len=5)
  - `totofferrem`, `totbidrem` 등

- `PredictionPipeline.add_realtime_tick()`는 `tick_norm`이 존재하면 이를 다시 raw dict 키(`offerho1..5`, `bidrem1..5` 등)로 **언팩(unpack)** 하여
  `calc_orderbook_features()`가 기대하는 “FH0 raw-key schema”를 최대한 맞춥니다.

## 계산 로직 (핵심)

구현: `prediction/features.py::calc_orderbook_features(quote)`

### 1) 입력 필드 후보

환경/피드에 따라 키가 다를 수 있어, 여러 키 후보를 순차 탐색하여 값이 있으면 사용합니다.

- L1 가격(스프레드 계산용)
  - ask: `offerho` or `offerho1` …
  - bid: `bidho` or `bidho1` …

- 잔량(수량)
  - L1: `offerrem`/`offerrem1`, `bidrem`/`bidrem1` …
  - Depth(1~5): `offerrem1..5`, `bidrem1..5`

- 총 잔량(가능하면 사용)
  - `totofferrem`, `totbidrem` (or 대체 키)

### 2) 기본 검증(Invalid 처리)

- bid/ask 가격이 없으면 **스프레드 계산이 불가능**하고, OBI에서 사용할 호가 레벨 기준(최소 L1 기준점)이 흔들리며,
  분자/분모 산정 과정에서 “의미 있는 상태”를 유지하기가 어려워 downstream 피처가 불안정해질 수 있으므로 `_invalid=True`로 반환하고,
  `PredictionPipeline`은 이 경우 OB 레코드에 누적하지 않습니다.

### 3) 스프레드

- `spread = offer1 - bid1`
- 순간적인 update ordering으로 음수가 될 수 있어 음수면 `abs()`로 보정합니다.

### 4) 총 잔량(total_offer/total_bid) 결정 규칙

우선순위(앞 단계가 유효하면 그 값을 **사용하고 종료**, 다음 단계는 fallback):

- (A) feed가 주는 total 필드
  - `totofferrem`, `totbidrem`
- (B) Depth(1~5) 잔량 합계
  - total 필드가 없고 `offerrem1..5`/`bidrem1..5`가 존재하면 `sum()`
- (C) L1 잔량 fallback
  - total이 여전히 없으면 `offerrem1`/`bidrem1` (또는 L1 잔량)만으로 total을 구성

### 5) OBI (Orderbook Imbalance)

- 분모 0 방지(최소 1.0):

```python
total = max(total_offer + total_bid, 1.0)
```

- **정의**

```text
obi = (total_bid - total_offer) / (total_bid + total_offer)
```

- 범위: 이론적으로 `[-1, +1]`
  - `+1`에 가까울수록 매수(매수잔량 우세)
  - `-1`에 가까울수록 매도(매도잔량 우세)

### 6) L1 imbalance (`level1_ratio`)

- `level1_denom = bidrem1 + offerrem1` (0 방지용으로 최소 1.0)

```text
level1_ratio = (bidrem1 - offerrem1) / (bidrem1 + offerrem1)
```

### 7) 기타 부가 피처

- `bid_slope = (bidrem5 - bidrem1) / 4`
- `offer_slope = (offerrem5 - offerrem1) / 4`
- 해석(직관):
  - `bid_slope`가 음수면 bid depth가 바깥(L5)으로 갈수록 잔량이 줄어드는 형태(= 아래로 갈수록 얕아짐)
  - `offer_slope`가 음수면 offer depth가 바깥(L5)으로 갈수록 잔량이 줄어드는 형태(= 위로 갈수록 얕아짐)
  - 절대값이 클수록(양/음 어느 쪽이든) depth 형태가 L1 대비 급격히 변합니다.
- `totbidrem`, `totofferrem`도 함께 반환(디버그/컨텍스트/피처 구성에 사용)

## 버퍼링/다운샘플링 (실시간에서 어떻게 누적되는가)

구현: `prediction/pipeline.py::PredictionPipeline.add_realtime_tick()`

- `FH0` orderbook snapshot을 내부 리스트(`self._ob_records`)에 누적합니다.
- `_ob_records`는 무한히 쌓이지 않으며, `deque(maxlen=seq_len)`로 유지됩니다.
  - 구현: `prediction/pipeline.py`에서 `self._ob_records = deque(maxlen=int(self._seq_len))`
  - 즉, 모델 입력 시퀀스 길이(`seq_len`)와 **같은 길이만큼만** 유지됩니다.
- 실시간 갱신이 매우 빠르므로 **1Hz(초당 1개)**로 다운샘플합니다.
  - 같은 초(`sec_key`)에 여러 건이 오면 마지막 1건으로 덮어씀
- 중복 스킵(시그니처 기반)
  - `obi/spread/level1_ratio/totbidrem/totofferrem` 값이 동일하고 같은 초면 스킵

`build_sequence()`와의 관계:

- `PredictionPipeline.get_prediction()`에서 `build_sequence(list(self._ob_records), candle_df, seq_len=self._seq_len, ...)` 형태로 호출됩니다.
- 따라서 `_ob_records`는 항상 `build_sequence()`가 참조하는 윈도우 크기(`seq_len`)에 맞춰 고정 길이로 관리됩니다.

## 어디에 사용되나

### 1) 모델 입력 피처(수치 예측)

- `prediction/features.py`의 `OB_KEYS` 시퀀스에 포함되어 모델 입력으로 들어갑니다.
- `build_sequence()`가 OB+CD+OPT(+ADAPT) 시퀀스를 구성할 때 활용됩니다.

### 2) 룰 기반 fallback

구현: `prediction/predictor.py::_rule_based()`

- 룰 베이스에서 주문장 압력을 다음처럼 구성합니다.

```text
pressure_ob = 0.75 * obi + 0.25 * level1_ratio
```

- 가중치 근거(현재 구현):
  - 경험적/휴리스틱한 초기값이며, `obi`(depth 기반 총 불균형)를 중심으로 두고 `level1_ratio`(L1 불균형)를 보조로 섞는 목적입니다.
  - 이후 튜닝 시에는 해당 비율을 조정하거나, 스프레드/유동성(예: `atm_spread_pct`, `atm_liquidity_log`) 조건에 따라 동적으로 가중치를 바꾸는 방향도 고려할 수 있습니다.

- 이후 모멘텀(ret3) 및 거래량 가속(vol_accel)을 약하게 섞어 최종 `prob`/`signal`을 만듭니다.

### 2.1) (가벼운 가드레일) ATM 옵션 호가(OH0) 기반 보수화

방향(BUY/SELL) 결정을 옵션 호가로 직접 바꾸지는 않되, **ATM 근처 옵션의 미세구조가 나쁜 구간에서는 신뢰도를 낮추거나 HOLD로 보수화**합니다.

- 입력: `build_option_snapshot()`이 계산한 OH0 기반 ATM 미세구조
  - `atm_spread_pct`: (ask-bid)/mid * 100 의 평균(ATM call/put)
  - `atm_liquidity_log`: ATM call/put의 depth(5단계) 잔량 합을 `log1p()` 스케일
- 적용 위치: `PredictionPipeline.get_prediction()` (numeric predictor 이후)
- 정책(현재 구현):
  - 조건 평가
    - wide: `atm_spread_pct >= 1.5`
    - illiq: `atm_liquidity_log <= 2.0`
  - wide & illiq 이고 signal이 `BUY/SELL`이면
    - `signal = HOLD`, `confidence = LOW`
  - 그 외(옵션 미세구조가 다소 나쁨)에는
    - `confidence`만 한 단계 다운그레이드(HIGH→MEDIUM, MEDIUM→LOW)

> 목적: 옵션호가 노이즈를 방향신호로 “가산”하기보다, 유동성/스프레드 악화 구간에서 과감한 신호를 완화하여 false positive 및 체결 리스크를 줄이기 위함입니다.

### 3) LLM 컨텍스트(요약)

구현: `prediction/context_builder.py::_summarize_orderbook()`

- 최근 `ob_records`의 `obi/spread/level1_ratio/totbidrem/totofferrem`을
  - `last`
  - `mean`
  - `delta (last-first)`
  로 요약하여 LLM 프롬프트에 주입됩니다.

## 주의/엣지 케이스

- **키/스키마 변동**
  - 환경에 따라 `totofferrem/totbidrem`이 없거나, depth 잔량 키가 달라질 수 있어 다중 키 후보 탐색을 사용합니다.
- **bid/ask 가격이 없는 tick**
  - OBI 자체는 잔량만 있어도 만들 수 있지만, 코드에서는 bid/ask 가격 누락 시 `_invalid=True`로 처리하여 버퍼 누적을 막습니다.
- **total 필드가 없을 때**
  - depth 합계 → L1 fallback으로 “최소한 의미 있는 OBI”를 유지합니다.

