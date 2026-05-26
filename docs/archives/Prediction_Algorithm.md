# 예측 알고리즘 개요

본 문서는 런타임 **수치 예측(Transformer / TFT / Mamba 앙상블)** 과 **가드레일·LLM·피드백**까지 이어지는 흐름을 코드 기준으로 정리한다.  
세부 API·파일 목록은 `docs/prediction.md`, `docs/Transformer_LLM_Pipeline.md` 등을 참고한다.

---

## 1. 전체 데이터 흐름

```
실시간 틱 / 분봉
    → TickMixin (FO0 버퍼, 분봉 DF)
    → OptionMixin (옵션 스냅샷, OI, IV)
    → AdaptiveMixin (레짐·지표, 선택)
    → PredictionMixin.get_prediction()
         ├─ 수치: build_sequence + ModelInput → numeric_predictor.predict()
         ├─ 가드레일: 옵션/베이시스/패리티/블리드/OI·진폭
         ├─ LLM: 스냅샷·프롬프트 → 판단 (설정 시)
         └─ 피드백 큐 (앙상블 가중 갱신용)
```

핵심 클래스: `prediction/pipeline.py`의 `PredictionPipeline` (Mixin 조합).

---

## 2. 수치 모델 입력 (`ModelInput`)

| 필드 | 의미 |
|------|------|
| `sequence` | 오더북·캔들·옵션·(선택)적응/멀티스케일 피처로 만든 `(seq_len, feature_dim)` 배열 |
| `past_known` | TFT용 과거 시간 피처 `(seq_len, FUTURE_KNOWN_DIM)` |
| `future_known` | TFT용 미래 시간 피처 `(horizon, FUTURE_KNOWN_DIM)` |
| `feature_snapshot` | 최신 호가/스프레드 등 스냅샷 dict |

시퀀스 생성: `prediction/features.py`의 `build_sequence()` 등.

---

## 3. 단일 모델 출력과 `_classify`

모든 경로는 최종적으로 **상승 확률 `prob ∈ [0,1]`** 을 만든 뒤 공통 함수로 신호·신뢰도를 정한다.

- **신호**: `prob ≥ buy_threshold` → BUY, `prob ≤ sell_threshold` → SELL, 그 사이 HOLD (기본 예: 0.62 / 0.38).
- **신뢰도**: `|prob − 0.5|` 마진과 **스프레드** 상한(`confidence_spread_max_for_high`)으로 HIGH / MEDIUM / LOW.

구현: `prediction/predictor.py`의 `_classify()`.

---

## 4. Transformer (`TransformerPredictor`)

- 가중치가 있으면 PyTorch로 추론, 없거나 실패 시 **`_rule_based()`** (호가 OBI·슬로프·단기 모멘텀 등 휴리스틱).
- **`rule_based_weights`** (`config.json` → `prediction`): 레짐 키별 가중치 오버라이드(내부 `_merge_rule_based_weights`로 기본값과 병합). 미설정 시 코드 기본 가중치.
- **`rule_based_mom_multiplier`**: 위 휴리스틱 확률 계산 시 **모멘텀 스케일**에 곱하는 전역 배율(레짐별 세부 rule은 아님).
- `numeric_predictor: "rule_based"`이면 **`RuleBasedPredictor`** 가 동일한 `compute_rule_based_probability()` 경로로만 확률을 낸다(학습 가중치 없음).
- 체크포인트에 정규화 통계가 있으면 z-score 적용, 없으면 시퀀스 런타임 z-score 폴백.
- Sigmoid 극단값(≈0.998 / 0.002)이면 **confidence를 LOW로 강등** (포화 의심).

적응형 쪽은 `AdaptiveMixin`에서 `ast_dir` / `ast_signal` / `azz` 등을 **`_parse_adaptive_heuristic_features()`** 한 번으로 파싱해 중복 분기를 줄인다.

---

## 5. TFT (`TFTPredictor`)

- `past_unknown` + `past_known` + `future_known` 를 입력으로 단일 확률 출력.
- 가중치 없음·차원 불일치 시 HOLD·0.5 등으로 **우아하게 축소**.

---

## 6. 앙상블 (`EnsemblePredictor`, `numeric_predictor: "ensemble"`)

### 6.1 가중 확률

- **2자**: Transformer + TFT — `ens_prob = w_t·p_t + (1−w_t)·p_f` (`w_t`는 설정·적응형).
- **Mamba 포함**: 3자 가중합 (Mamba 가중 `mamba_weight` 반영).
- TFT/Mamba 비가용 시 해당 분기만 생략.

### 6.2 Conformal 구간과 신뢰도

- 캘리브레이션된 Conformal이 있으면 `prob_lower`, `prob_upper`를 구한 뒤, **구간 폭이 넓을수록** HIGH/MEDIUM을 낮춘다 (`adjust_confidence_by_conformal_interval_width`).
- 설정: `confidence_conformal_width_max_for_high`, `confidence_conformal_width_max_for_medium` (`config.json` → `prediction`).

### 6.3 불일치 HOLD (연속 거리)

- 참여 모델 확률들의 **쌍별 \|Δp\| 최대값**이 임계 이상이면 **HOLD + LOW** (`disagreement_hold`).
- 기본 임계: `disagreement_hold_prob_diff_max`.
- **레짐별** 오버라이드: `disagreement_hold_prob_diff_max_by_regime` (예: `RANGE`, `STRONG_UP`).  
  `set_regime()`으로 현재 레짐이 설정되면 해당 키가 있을 때만 대체.

### 6.4 합의 시 신뢰도 상향 (선택)

- `ensemble_agreement_confidence_boost`가 켜져 있고, TFT가 참여하며 **방향 합의**, `prob_diff`가 `ensemble_agreement_prob_diff_max` 미만, 이미 MEDIUM이며 스프레드가 HIGH 허용 범위이면 **HIGH**로 올린다 (`*_agreement_boost` 접미사).

### 6.5 적응형 가중 (`AdaptiveEnsembleWeightTracker`)

- 피드백으로 Transformer/TFT 정확도를 **Brier 스타일 점수**로 누적하고, 레짐·만기(DTE)에 따라 **윈도 크기**와 초기 `w_t` 편향을 조정.

---

## 7. 가드레일 (요약)

순서는 파이프라인 구현에 따른다. 대표적으로:

- **옵션·베이시스·패리티·블리드** 등 (`GuardrailMixin`).
- **OI**: `_oi_levels`의 Peak 거리·집중도·Zero Gamma·Vol Trigger 등으로 신호/신뢰도 보정 (`_apply_oi_guardrail`).
- **진폭 + OI**: 예상 진폭 소진(`amplitude_exhaustion`)과 Peak 근접을 동시에 보면 추가 억제.

OI 데이터가 없어 가드레일이 스킵되면 메트릭 `guardrail_oi_skipped_no_data`가 증가할 수 있다.

---

## 8. 선물 진폭과 OI 지지·저항

- `calc_oi_levels()`: Call/Put OI 피크 등 **지지·저항 대표가**와 거리·집중도 등을 산출.
- `calc_expected_amplitude()`:  
  - IV 기반 일중 진폭(`iv_amp`),  
  - OI 박스폭 `call_peak − put_peak` 기반 `oi_amp`,  
  - 집중도로 **가중 혼합** (`expected = oi_weight·oi_amp + (1−oi_weight)·iv_amp`).  
  - `oi_range_pct`가 크면 OI 가중을 추가로 낮춰 IV 비중을 키운다.

자세한 식은 `prediction/oi_features.py` 주석 참고.

---

## 9. LLM 층

- 수치 결과·오더북·옵션·적응 컨텍스트를 스냅샷으로 묶어 프롬프트 생성 (`context_builder`, `prediction_mixin`).
- Rate limit(429) 시 off-boundary 호출 등에서 LLM 생략 정책 적용.

### 9.1 LLM 실패 시 `heuristic_fallback` (`prediction/llm_mixin.py`)

`PredictionConfig.heuristic_fallback`은 **LLM 호출 실패·타임아웃** 시 최종 `llm_action`을 어떻게 둘지 결정한다.

| 값 | 동작 |
|----|------|
| `false` | 수치 파이프라인 신호만 유지. 근거 문자열에 `heuristic_fallback=false`를 명시. |
| `true`(기본) | `model_outputs["heuristic"]`에 adaptive 휴리스틱이 BUY/SELL로 **준비된 경우** 해당 방향을 `llm_action`에 반영하고, `llm_provider`를 `heuristic_fallback`으로 표시. 그렇지 않으면 수치 신호만. |

단일 LLM·듀얼 LLM 모두 동일 헬퍼(`_llm_failure_fallback_action`)를 탄다.

### 9.2 라이브: 휴리스틱 방향 플립 간격 (`ebest_live.py`)

적응형 휴리스틱 **방향이 바뀔 때** 너무 잦은 전환을 줄이기 위해 최소 간격을 둔다.

- **`heuristic_flip_min_interval_sec`**: 초 단위. `null`이면 `max(60, prediction_minutes×30)` 초와 동일한 기본 공식을 쓴다. 파이프라인에 주입되면 `predictor._heuristic_flip_min_interval_sec`로 읽힌다.
- **`heuristic_flip_include_hold_transition`**: `false`(기본)이면 BUY↔SELL만 플립으로 간주. `true`이면 **HOLD↔BUY/SELL** 전환도 간격 제한·메트릭 대상에 포함.

메트릭(`pipeline._metrics`): `heur_flip_triggered`, `heur_flip_skipped_interval` (`predictor._metrics_inc`가 있을 때만 증가).

---

## 10. 피드백

- 예측 시점의 `transformer_prob` / `tft_prob` 등을 큐에 넣고, 이후 실제 가격 움직임과 비교해 **앙상블 가중**을 갱신한다 (`FeedbackMixin`).

---

## 11. 검증·튜닝 (오프라인)

- `prediction/calibration_metrics.py`: Brier, ECE.
- `prediction/calibration_report.py`: 검증 세트 요약 문자열.
- `prediction/calibration_thresholds.py`: `config.json`의 `prediction`에서 자주 튜닝하는 키 목록.
- `scripts/rule_based_backtest_hook.py`: 샘플 데이터로 `build_validation_report` + `format_tunable_keys_reference` 출력(룰베이스·캘리브레이션 키 점검용 진입점).

---

## 12. 관련 설정 키 (발췌)

| 키 (prediction) | 용도 |
|-----------------|------|
| `numeric_predictor` | `transformer` / `tft` / `ensemble` / `rule_based` |
| `heuristic_fallback` | LLM 실패 시 adaptive 휴리스틱으로 `llm_action` 보강 여부(위 9.1) |
| `heuristic_flip_min_interval_sec`, `heuristic_flip_include_hold_transition` | 라이브 휴리스틱 방향 전환 최소 간격·HOLD 포함 여부(위 9.2) |
| `rule_based_weights`, `rule_based_mom_multiplier` | 휴리스틱 확률 가중·모멘텀 배율(위 4장) |
| `buy_threshold`, `sell_threshold` | BUY/SELL 임계 |
| `confidence_*`, `confidence_conformal_width_*` | 신뢰도·Conformal 폭 보정 |
| `disagreement_hold`, `disagreement_hold_prob_diff_max`, `disagreement_hold_prob_diff_max_by_regime` | 불일치 HOLD |
| `ensemble_agreement_confidence_boost`, `ensemble_agreement_prob_diff_max` | 합의 시 HIGH 승격 |
| `conformal_alpha`, `conformal_path` | Conformal 예측 구간 |

전체 스키마는 `config.py`의 `PredictionConfig` 및 `docs/runtime/config.md` 참고.

---

## 13. 주요 소스 파일

| 파일 | 역할 |
|------|------|
| `prediction/pipeline.py` | 파이프라인 조립 |
| `prediction/predictor.py` | 수치 예측·앙상블·분류 |
| `prediction/prediction_mixin.py` | `get_prediction` 흐름 |
| `prediction/features.py` | 피처·시퀀스 |
| `prediction/oi_features.py` | OI 레벨·진폭 |
| `prediction/guardrail_mixin.py` | 가드레일 |
| `prediction/calibration_metrics.py` | Brier/ECE |

---

*문서 버전: 코드베이스 기준 통합 설명. 세부 수치는 배포 환경의 `config.json`이 정본이다.*
