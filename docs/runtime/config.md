# config.py (Configuration)

## 역할

- `config.json` + `config.secrets.json` 병합 로드
- dataclass 기반 설정 구조 제공
- 런타임에서 안전하게 접근할 수 있도록 검증

## 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `_deep_merge_dict(base, override)` | function | dict 재귀 머지(시크릿 병합에 사용) |
| `_load_json_file(path)` | function | JSON 파일 안전 로드(best-effort) |
| `_resolve_secrets_path(config_path, secrets_path)` | function | secrets 파일 경로 결정 |
| `load_config(config_path="config.json")` | function | 런타임 편의 로더(실패 시 defaults) |

## 핵심 dataclass

| 이름 | 종류 | 설명 |
|---|---|---|
| `AIProviderConfig` | dataclass | Claude/OpenAI/Gemini 키 보관 |
| `EBestConfig` | dataclass | eBest appkey/appsecretkey |
| `OptionSubscriptionConfig` | dataclass | 옵션 구독 범위(ITM/OTM 등) |
| `OptionMinuteOhlcvConfig` | dataclass | 옵션 틱→분봉 OHLCV 집계 설정 |
| `MinuteLookbackConfig` | dataclass | `tick_processor` 분봉 DF 조회 기본 lookback(`minute_lookback`) |
| `PredictionConfig` | dataclass | 예측 관련 파라미터(시퀀스 길이/threshold 등) |
| `AdaptiveIndicatorSettings` | dataclass | adaptive_indicator 사용 여부/파라미터/warmup_bars |
| `AppConfig` | dataclass | 전체 설정 루트(로딩/검증 포함) |

## 런타임에서 중요한 설정 키(요약)

| config key | 영향 범위 |
|---|---|
| `prediction_minutes` / `prediction.minutes` | LLM/모델 horizon |
| `seq_len` / `prediction.seq_len` | FO0 버퍼 길이(초) 및 모델 입력 길이 |
| `min_minute_bars_required` | 분봉이 충분히 쌓이기 전 예측 방지 |
| `minute_lookback.futures/options` | `tick_processor.get_*_minute_df()`의 기본 조회 길이 |
| `adaptive_indicator.enabled` | feature_dim 19/47 및 컨텍스트 블록 포함 여부 |
| `adaptive_indicator.warmup_bars` | adaptive 지표 warmup 길이 |
| `option_minute_ohlcv.enabled` | 옵션 틱 분봉 집계 on/off |
| `prediction.confidence_high_margin` / `prediction.confidence_mid_margin` / `prediction.confidence_spread_max_for_high` | confidence 분류 임계값(확률 마진/스프레드) |
| `prediction.disagreement_hold_prob_diff_max` | 앙상블 disagree 시 HOLD로 강제할지 결정하는 확률 차이 임계값 |
| `prediction.guard_basis_hold_thr` / `prediction.guard_basis_downgrade_thr` | spot-선물 basis 가드레일(hold/downgrade) 임계값 |
| `prediction.guard_atm_spread_pct_thr` / `prediction.guard_atm_liq_log_thr` | ATM 옵션 스프레드/유동성 가드레일 임계값 |
| `prediction.heuristic_fallback` | LLM 실패·타임아웃 시 adaptive 휴리스틱으로 최종 판단을 보강할지 여부 (`llm_mixin`) |
| `prediction.heuristic_flip_min_interval_sec` | 라이브에서 휴리스틱 방향 전환(BUY/SELL 등) 최소 간격(초). 미설정 시 `ebest_live` 기본 공식 사용 |
| `prediction.heuristic_flip_include_hold_transition` | 위 간격 제한에 HOLD↔BUY/SELL 전환을 포함할지 여부 |
| `prediction.rule_based_weights` | 레짐별 휴리스틱 가중치 오버라이드(dict). `rule_based`·Transformer 폴백 공통 |
| `prediction.rule_based_mom_multiplier` | 휴리스틱 확률의 모멘텀 항 전역 배율 |
| `prediction.pcr_atm_strikes_each_side` | `calc_pcr` ATM±N 행사가 윈도(0~50, 기본 5). 0이면 ATM 1줄만 합산 |
| `telegram.option_flow_status_enabled` | 옵션 마이크로 플로우 별도 메시지 전송 on/off |
| `telegram.option_flow_status_cooldown_sec` | 옵션 마이크로 플로우 별도 메시지 전송 전용 쿨다운(초) |
| `telegram.option_flow_status_intraday_only` | 옵션 마이크로 플로우 메시지를 장중에만 전송할지 여부 |
| `telegram.option_flow_status_disable_after_close` | 장마감 후 옵션 마이크로 플로우 메시지 비활성화 여부 |
| `telegram.option_flow_interp_sr_warn` / `telegram.option_flow_interp_sr_hot` | `surge_ratio` 해석 임계값(`유입 증가`/`유입 급증`) |
| `telegram.option_flow_interp_pt_low` / `telegram.option_flow_interp_pt_high` | `per_tick_move_pt` 해석 임계값(`충격 낮음`/`충격 큼`) |
| `telegram.option_flow_interp_pcr_v_low` / `telegram.option_flow_interp_pcr_v_high` | `pcr_volume` 해석 임계값(콜/풋 우위 판정) |
| `telegram.option_flow_interp_pcr_oi_low` / `telegram.option_flow_interp_pcr_oi_high` | `pcr_oi` 해석 임계값(콜/풋 우위 판정) |
| `meaningful_option_levels` | 옵션 의미가 레벨 리스트. OC0 당일 최고/최저가가 해당 레벨과 정확히 일치할 때 GUI `RT:` 라인에 `SRH`/`SRL`로 표시 |
