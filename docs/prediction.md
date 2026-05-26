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
| `RuleBasedPredictor` | class | `numeric_predictor: "rule_based"` 전용. `compute_rule_based_probability()`로 OBI·모멘텀 등 휴리스틱 확률만 산출 |
| `compute_rule_based_probability` / `_merge_rule_based_weights` | function | 휴리스틱 확률·레짐 가중 병합 (`rule_based_weights`, `rule_based_mom_multiplier` 반영) |

**설정 (`prediction.*`)**: `heuristic_fallback`(LLM 실패 시 휴리스틱으로 `llm_action` 보강), `rule_based_weights`, `rule_based_mom_multiplier`. 라이브 플립 간격은 `heuristic_flip_min_interval_sec`, `heuristic_flip_include_hold_transition`(상세는 [`Prediction_Algorithm.md`](Prediction_Algorithm.md) 9장).

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

### 3.1 옵션 피처 셋(v1/v2/v3/v4)

`config.json`의 `prediction.option_feature_set`으로 OPT 블록을 선택합니다.

| 버전 | 피처 수 | 설명 |
|---|---|---|
| `v1` | 7 | 기본 OPT (PCR, IV skew, max pain, microstructure) |
| `v2` | 16 | v1 + option_minute_ohlcv 기반 미세움직임 9개 |
| `v3` | 23 | v2 + 만기주 콜-풋 패리티 이탈(parity divergence) 7개 |
| `v4` | 29 | v3 + 만기주 프리미엄 블리드(premium bleed) 6개 |

`v2`의 `optm_*`(미세움직임) 피처는 `tick_processor.get_option_minute_df()`를 통해
옵션 분봉 OHLCV가 제공될 때만 값이 채워집니다.

- `config.json`의 `option_minute_ohlcv.enabled=true`가 필요
- 실시간에서는 `OC0` 틱이 옵션 분봉 집계를 구동합니다(심볼 범위는 ATM±`atm_window` 정책에 따름)

**v3 추가 키(7개) — 만기주 콜-풋 패리티 이탈:**

- `parity_spread_pct` : C-P 이론값 대비 이탈 비율(%)
- `call_delta_proxy` : C/(C+P) 델타 근사값 [0,1]. ATM 이론값 = 0.5
- `straddle_price` : ATM C+P 스트래들 가격
- `straddle_vs_fut_move` : 스트래들 / |F-K| 배율
- `call_vs_fut_ret_diff` : 콜 수익률 - (0.5 × 선물 수익률)
- `dte_weight_norm` : 만기 근접도 [0,1]. 당일=1.0, 7일 이상≈0
- `parity_divergence_score` : 종합 이탈 스코어 [-1,1]

**v4 추가 키(6개) — 만기주 프리미엄 블리드:**

- `straddle_decay_vs_fut` : 스트래들 수익률 - |선물 수익률|×0.5. 음수=비정상 수축
- `iv_crush_proxy` : BS ATM IV 근사 변화율. 음수=IV 감소
- `fut_ret` : 직전 틱 선물 수익률
- `straddle_now` : 현재 ATM 스트래들 가격
- `straddle_prev` : 직전 틱 ATM 스트래들 가격
- `premium_bleed_score` : 종합 프리미엄 수축 스코어 [-1,1]. -1=강한 수축

> ⚠️ `option_feature_set`이 바뀌면 OPT 차원이 달라져 `feature_dim`이 변경됩니다. v3/v4는 dataset 재생성 및 재학습이 필요합니다.

> ℹ️ v3/v4의 `_prev_*` 상태(직전 선물가/ATM 옵션가)는 `PredictionPipeline._build_option_snapshot_safe(update_prev=True)`가 OB 버퍼 경로(1Hz)에서 자동 갱신합니다.

### 3.2 가드레일 구조 (v3/v4)

v3 이상에서는 수치 예측 결과에 다음 가드레일이 **순서대로** 적용됩니다.

| 순서 | 가드레일 | 동작 조건 |
|---|---|---|
| 1 | `_apply_option_guardrail` | PCR/IV 기반 이상 감지 |
| 2 | `_apply_basis_guardrail` | 선/현물 베이시스 과다 |
| 3 | `_apply_parity_guardrail` | v3/v4: \|parity_divergence_score\| ≥ 0.5, dte_w ≥ 0.033 |
| 4 | `_apply_bleed_guardrail` | **v4 전용**: premium_bleed_score ≤ -0.75, dte_w ≥ 1.0, 선물 상승 중 |

### 3.3 운영용 임계값(요약)

아래 값들은 `config.json`의 `prediction.*`에서 조정할 수 있습니다.

- `confidence_high_margin`, `confidence_mid_margin`, `confidence_spread_max_for_high`
- `disagreement_hold_prob_diff_max`
- `guard_basis_hold_thr`, `guard_basis_downgrade_thr`
- `guard_atm_spread_pct_thr`, `guard_atm_liq_log_thr`
- `heuristic_fallback`, `rule_based_weights`, `rule_based_mom_multiplier`, `heuristic_flip_min_interval_sec`, `heuristic_flip_include_hold_transition`

## 4) prediction/option_features.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `calc_pcr(calls, puts)` | function | PCR(volume/oi) 계산 |
| `calc_iv_skew(calls, puts, underlying_price)` | function | ATM put/call IV skew |
| `calc_max_pain(calls, puts, underlying_price)` | function | max pain strike 및 거리 |
| `calc_atm_microstructure(calls, puts, underlying_price)` | function | OH0 기반 ATM 미세구조 피처 |
| `calc_gex(calls, puts, underlying_price)` | function | Gamma Exposure(GEX) 계산 |
| `calc_parity_divergence(calls, puts, underlying_price, ...)` | function | **v3/v4**: ATM 콜-풋 패리티 이탈 지표 계산 (8개 피처 반환) |
| `calc_premium_bleed(calls, puts, underlying_price, ...)` | function | **v4**: 선물 상승 중 옵션 프리미엄 수축 지표 계산 (6개 피처 반환) |
| `build_option_snapshot(calls, puts, underlying_price, *, option_feature_set, prev_*)` | function | v1~v4 분기 처리 후 snapshot dict 생성 |

### 설계 문서 참조

- 패리티 이탈 설계: [`call_put_parity_divergence_design.md`](../call_put_parity_divergence_design.md)
- 프리미엄 블리드 설계: [`premium_bleed_design.md`](../premium_bleed_design.md)

## 5) prediction/context_builder.py

| 이름 | 종류 | 설명 |
|---|---|---|
| `build_llm_context(snapshot, ob_records, adaptive_context)` | function | 컨텍스트 블록 생성. v3/v4에서 `[PARITY_ANALYSIS]`, `[PREMIUM_BLEED]` 섹션 자동 포함 |
| `build_llm_prompt(context, prediction_minutes)` | function | strict JSON 응답을 강제하는 프롬프트 생성 |
| `_describe_parity_divergence(opt_snap)` | function | **v3/v4**: 패리티 이탈 수치 → LLM용 자연어 변환. dte_w < 0.1이면 빈 문자열 반환 |
| `_describe_premium_bleed(opt_snap)` | function | **v4**: 프리미엄 블리드 수치 → LLM용 자연어 변환. dte_w < 0.1이면 빈 문자열 반환 |

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
