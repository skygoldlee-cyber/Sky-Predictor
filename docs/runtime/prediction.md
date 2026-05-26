# prediction/ (Runtime Pipeline)

## 1) prediction/pipeline.py

### 역할

- 런타임 핵심 오케스트레이터
- tick을 받아 내부 버퍼(FO0 특징, 옵션 스냅샷, 분봉 DF)를 갱신
- 수치 예측(Transformer/TFT/앙상블/룰베이스) + LLM 판단을 결합해 최종 결과 생성

### 핵심 클래스/메서드

| 이름 | 종류 | 설명 |
|---|---|---|
| `PredictionPipeline` | class | 실시간 예측 파이프라인 |
| `PredictionPipeline.add_realtime_tick(payload)` | method | realtime payload를 받아 내부 상태 업데이트 |
| `PredictionPipeline.get_prediction()` | method | 현재까지 누적된 정보로 1회 예측 수행(결과 dict 반환) |
| `PredictionPipeline.set_market_snapshots(t2101, t2301)` | method | 배경 스냅샷 주입(가능할 때) |

## 2) prediction/predictor.py

### 역할

- 수치 예측기(Transformer/TFT) 로딩 및 추론
- 가중치/차원 불일치 시 룰 기반 fallback

| 이름 | 종류 | 설명 |
|---|---|---|
| `TransformerPredictionResult` | dataclass | 수치 예측 결과(prob/signal/confidence/snapshot) |
| `ModelInput` | dataclass | 모델 입력 컨테이너(sequence/past_known/future_known/meta 등) |
| `TransformerPredictor` | class | Transformer 추론(또는 fallback) |
| `_classify(prob, spread, buy_threshold, sell_threshold, confidence_*)` | function | 확률→BUY/SELL/HOLD + confidence 매핑(임계값은 config로 조정 가능) |
| `create_numeric_predictor(...)` | function | numeric_predictor 옵션에 따라 predictor 조합 생성 |
| `RuleBasedPredictor` | class | `numeric_predictor: "rule_based"` — 휴리스틱 확률만 (`compute_rule_based_probability`) |

**참고**: `heuristic_fallback`, `rule_based_weights`, `rule_based_mom_multiplier`, 라이브 휴리스틱 플립 간격 키는 [`Prediction_Algorithm.md`](../Prediction_Algorithm.md)·[`config.md`](config.md) 참고.

## 3) prediction/features.py

### 역할

- FH0 오더북 스냅샷 → 수치 피처(OB)
- 분봉 OHLCV → 캔들 피처(CD)
- 옵션 스냅샷 → 옵션 피처(OPT)
- (옵션) adaptive 지표 피처(ADAPT)
- 최종적으로 `(seq_len, feature_dim)` 시퀀스를 구성

| 이름 | 종류 | 설명 |
|---|---|---|
| `OB_KEYS`, `CD_KEYS`, `OPT_KEYS`, `ADAPT_KEYS` | constants | 모델 입력 피처 키/순서(정본) |
| `calc_orderbook_features(quote)` | function | FH0-like dict → OB 피처 dict |
| `calc_candle_features(df)` | function | OHLCV DF → CD 피처 DF |
| `build_sequence(..., adaptive_features=None)` | function | OB+CD+OPT(+ADAPT) 시퀀스 생성 |

### 3.1 옵션 피처 셋(v1/v2)

`config.json`의 `prediction.option_feature_set`으로 OPT 블록을 선택합니다.

- `v1`: 기존 OPT(7)
- `v2`: 확장 OPT(기존 7 + option_minute_ohlcv 기반 미세움직임 9)

`v2`의 `optm_*`(미세움직임) 피처는 `tick_processor.get_option_minute_df()`를 통해
옵션 분봉 OHLCV가 제공될 때만 값이 채워집니다.

- `config.json`의 `option_minute_ohlcv.enabled=true`가 필요
- 실시간에서는 `OC0` 틱이 옵션 분봉 집계를 구동합니다(심볼 범위는 ATM±`atm_window` 정책에 따름)

> ⚠️ `option_feature_set`이 바뀌면 OPT 차원이 달라져 `feature_dim`이 변경됩니다. v2로 모델 입력에 포함하려면 dataset 재생성 및 재학습이 필요합니다.

### 3.2 운영용 임계값(요약)

아래 값들은 `config.json`의 `prediction.*`에서 조정할 수 있습니다.

- `confidence_high_margin`, `confidence_mid_margin`, `confidence_spread_max_for_high`
- `disagreement_hold_prob_diff_max`
- `guard_basis_hold_thr`, `guard_basis_downgrade_thr`
- `guard_atm_spread_pct_thr`, `guard_atm_liq_log_thr`
- `heuristic_fallback`, `rule_based_weights`, `rule_based_mom_multiplier`, `heuristic_flip_*`

v2 추가 키(9개):

- `optm_call_ret`
- `optm_put_ret`
- `optm_straddle_ret`
- `optm_call_range_pct`
- `optm_put_range_pct`
- `optm_straddle_range_pct`
- `optm_call_vol`
- `optm_put_vol`
- `optm_straddle_vol`

## 4) prediction/option_features.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `calc_pcr(calls, puts)` | function | PCR(volume/oi) 계산 |
| `calc_iv_skew(calls, puts, underlying_price)` | function | ATM put/call IV skew |
| `calc_max_pain(calls, puts, underlying_price)` | function | max pain strike 및 거리 |
| `calc_atm_microstructure(calls, puts, underlying_price)` | function | OH0 기반 ATM 미세구조 피처 |
| `build_option_snapshot(calls, puts, underlying_price)` | function | 위 피처들을 합쳐 snapshot dict 생성 |

## 5) prediction/context_builder.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `build_llm_context(snapshot, ob_records, adaptive_context)` | function | 컨텍스트 블록 생성 |
| `build_llm_prompt(context, prediction_minutes)` | function | strict JSON 응답을 강제하는 프롬프트 생성 |

## 6) prediction/llm_judge.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `LLMJudgment` | dataclass | 정규화된 LLM 출력 |
| `LLMJudge` | class | provider 선택/호출/fallback/파싱 담당 |
| `LLMJudge.judge(system, user, timeout=None)` | method | 실제 호출 및 `LLMJudgment` 반환 |

## 7) prediction/time_features.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `build_time_features(dt)` | function | TFT용 시간 피처 벡터 생성 |

## 8) prediction/weights_selector.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `WeightSelection` | dataclass | 선택된 weight 경로와 이유 |
| `select_weights_for_datetime(now, ...)` | function | 만기주 freeze 정책 포함해 사용할 weight 결정 |
