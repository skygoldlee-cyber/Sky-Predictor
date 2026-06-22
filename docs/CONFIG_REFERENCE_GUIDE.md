# config.json 전체 설정 가이드

SkyPredictor 시스템의 모든 설정을 설명하는 참조 가이드.

## 개요

config.json은 SkyPredictor 시스템의 모든 설정을 포함하는 중앙 설정 파일입니다. 각 섹션별 설정을 상세히 설명합니다.

### 목적

- 모든 설정의 중앙화된 참조
- 설정값별 설명 및 권장 범위
- 설정 변경 가이드

### 대상 독자

- 시스템 운영자
- 개발자
- 트레이딩 전략 개발자

## 전체 구조

```json
{
  "ai_providers": {},
  "ebest": {},
  "options_subscription": {},
  "option_minute_ohlcv": {},
  "minute_lookback": {},
  "market_holidays": [],
  "telegram": {},
  "meaningful_option_levels": [],
  "use_llm": true,
  "prediction": {},
  "adaptive_indicator": {},
  "trade_gate": {}
}
```

## 섹션별 설정

### 1. ai_providers

AI 제공자 (Anthropic, OpenAI, Gemini) 설정입니다.

```json
{
  "ai_providers": {
    "anthropic": {},
    "openai": {},
    "gemini": {}
  }
}
```

**설명**: 각 제공자의 API Key는 환경변수 또는 `config.secrets.json`에 저장합니다.

---

### 2. ebest

eBest API 설정입니다.

```json
{
  "ebest": {}
}
```

**설명**: eBest 연동 설정. 상세 설정은 환경변수로 관리합니다.

---

### 3. options_subscription

옵션 구독 설정입니다.

```json
{
  "options_subscription": {
    "itm": 10,
    "otm_open_min": 0.5,
    "max_otm_calls": 30,
    "max_otm_puts": 40,
    "wait_sec": 2,
    "preopen_oh0_window": 10,
    "oi_itm_count": 10,
    "oi_otm_count": 10,
    "oi_rebalance_interval_sec": 60
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| itm | int | 10 | ITM 옵션 개수 | 5 ~ 20 |
| otm_open_min | float | 0.5 | OTM 오픈 최소 거리 (%) | 0.3 ~ 1.0 |
| max_otm_calls | int | 30 | 최대 OTM 콜 개수 | 20 ~ 50 |
| max_otm_puts | int | 40 | 최대 OTM 풋 개수 | 30 ~ 60 |
| wait_sec | int | 2 | 구독 대기 시간 (초) | 1 ~ 5 |
| preopen_oh0_window | int | 10 | 장전 OH0 윈도우 (분) | 5 ~ 15 |
| oi_itm_count | int | 10 | OI ITM 개수 | 5 ~ 20 |
| oi_otm_count | int | 10 | OI OTM 개수 | 5 ~ 20 |
| oi_rebalance_interval_sec | int | 60 | OI 리밸런싱 간격 (초) | 30 ~ 120 |

---

### 4. option_minute_ohlcv

옵션 분봉 OHLCV 설정입니다.

```json
{
  "option_minute_ohlcv": {
    "enabled": true,
    "atm_window": 2
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| enabled | bool | true | 활성화 여부 | true/false |
| atm_window | int | 2 | ATM 윈도우 | 1 ~ 5 |

---

### 5. minute_lookback

분봉 룩백 설정입니다.

```json
{
  "minute_lookback": {
    "futures": 45,
    "options": 45
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| futures | int | 45 | 선물 룩백 (분) | 30 ~ 60 |
| options | int | 45 | 옵션 룩백 (분) | 30 ~ 60 |

---

### 6. market_holidays

휴장일 설정입니다.

```json
{
  "market_holidays": [
    "2026-05-05",
    "2026-06-06"
  ]
}
```

**설명**: ISO 8601 형식 (YYYY-MM-DD)

---

### 7. telegram

텔레그램 알림 설정입니다.

```json
{
  "telegram": {
    "enabled": true,
    "option_flow_status_enabled": true,
    "option_flow_status_cooldown_sec": 300,
    "option_flow_status_intraday_only": true,
    "option_flow_status_disable_after_close": true,
    "option_flow_interp_sr_warn": 1.5,
    "option_flow_interp_sr_hot": 2.0,
    "option_flow_interp_pt_low": 0.008,
    "option_flow_interp_pt_high": 0.03,
    "option_flow_interp_pcr_v_low": 0.9,
    "option_flow_interp_pcr_v_high": 1.1,
    "option_flow_interp_pcr_oi_low": 0.95,
    "option_flow_interp_pcr_oi_high": 1.05
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| enabled | bool | true | 텔레그램 활성화 | true/false |
| option_flow_status_enabled | bool | true | 옵션 흐름 상태 활성화 | true/false |
| option_flow_status_cooldown_sec | int | 300 | 쿨다운 (초) | 180 ~ 600 |
| option_flow_status_intraday_only | bool | true | 장내만 알림 | true/false |
| option_flow_status_disable_after_close | bool | true | 장종료 후 비활성화 | true/false |
| option_flow_interp_sr_warn | float | 1.5 | SR 경고 임계값 | 1.0 ~ 2.0 |
| option_flow_interp_sr_hot | float | 2.0 | SR 핫 임계값 | 1.5 ~ 2.5 |
| option_flow_interp_pt_low | float | 0.008 | PT 낮음 임계값 | 0.005 ~ 0.015 |
| option_flow_interp_pt_high | float | 0.03 | PT 높음 임계값 | 0.02 ~ 0.05 |
| option_flow_interp_pcr_v_low | float | 0.9 | PCR-V 낮음 임계값 | 0.8 ~ 1.0 |
| option_flow_interp_pcr_v_high | float | 1.1 | PCR-V 높음 임계값 | 1.0 ~ 1.2 |
| option_flow_interp_pcr_oi_low | float | 0.95 | PCR-OI 낮음 임계값 | 0.9 ~ 1.0 |
| option_flow_interp_pcr_oi_high | float | 1.05 | PCR-OI 높음 임계값 | 1.0 ~ 1.1 |

---

### 8. prediction

예측 엔진 설정입니다.

```json
{
  "prediction": {
    "numeric_predictor": "ensemble",
    "model_class": "patch_tst",
    "patch_len": 8,
    "stride": 4,
    "mamba_enabled": false,
    "mamba_weights_path": "",
    "mamba_weight": 0.33,
    "multiscale_5m": false,
    "conformal_alpha": 0.12,
    "conformal_path": "",
    "option_feature_set": "v4",
    "pcr_atm_strikes_each_side": 3,
    "multiscale_time_scales": [1, 5, 15],
    "multiscale_enabled": true,
    "llm_timeout_sec": 20.0,
    "gemini_timeout_sec": 45.0,
    "min_minute_bars_required": 21,
    "dual_llm": false,
    "dual_llm_primary_provider": "gemini",
    "buy_threshold": 0.64,
    "sell_threshold": 0.36,
    "transformer_weight": 0.5,
    "tft_weights_path": "",
    "tft_horizon": 300,
    "disagreement_hold": true,
    "disagreement_hold_prob_diff_max": 0.08,
    "guard_basis_hold_thr": 2.5,
    "guard_basis_downgrade_thr": 1.5,
    "guard_atm_spread_pct_thr": 1.5,
    "guard_atm_liq_log_thr": 2.0,
    "llm_min_interval_sec": 30.0,
    "llm_provider_cooldown_on_timeout_sec": 90.0,
    "tick_size": 0.05,
    "feedback_threshold_ticks": 10,
    "feedback_skip_hold_ticks": 2,
    "feedback_weight_high": 1.0,
    "feedback_weight_mid": 0.5,
    "feedback_weight_low": 0.25,
    "feedback_use_price_snapshot": true,
    "feedback_snapshot_tolerance_sec": 30.0,
    "feedback_snapshot_required": false,
    "fc0_stale_threshold_sec": 10.0,
    "fc0_stale_cooldown_sec": 60.0,
    "oi_alert_cooldown_sec": 120,
    "transformer_weights_path": "prediction/weights/patch_tst_5m.pt"
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| numeric_predictor | string | "ensemble" | 예측 엔진 | transformet, tft, ensemble, rule_based |
| model_class | string | "patch_tst" | 모델 클래스 | transformer, patch_tst |
| patch_len | int | 8 | PatchTST 패치 길이 | 4 ~ 16 |
| stride | int | 4 | PatchTST 스트라이드 | 2 ~ 8 |
| mamba_enabled | bool | false | Mamba 활성화 | true/false |
| mamba_weight | float | 0.33 | Mamba 가중치 | 0.2 ~ 0.5 |
| buy_threshold | float | 0.64 | BUY 임계값 | 0.6 ~ 0.7 |
| sell_threshold | float | 0.36 | SELL 임계값 | 0.3 ~ 0.4 |
| transformer_weight | float | 0.5 | Transformer 가중치 | 0.3 ~ 0.7 |
| disagreement_hold | bool | true | 불일치 시 HOLD | true/false |
| llm_timeout_sec | float | 20.0 | LLM 타임아웃 | 15 ~ 30 |
| llm_min_interval_sec | float | 30.0 | LLM 최소 간격 | 20 ~ 60 |

---

### 9. adaptive_indicator

적응형 지표 설정입니다.

```json
{
  "adaptive_indicator": {
    "enabled": true,
    "dual_mode": true,
    "symbol": "KOSPI 지수",
    "kospi_symbol": "KOSPI 지수",
    "futures_symbol": "KP200 선물",
    "warmup_bars": 45,
    "supertrend": {},
    "ranging_filter": {},
    "zigzag": {},
    "kospi_zigzag": {},
    "futures_zigzag": {}
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| enabled | bool | true | 활성화 여부 | true/false |
| dual_mode | bool | false | 듀얼 모드 | true/false |
| warmup_bars | int | 45 | 워밍업 봉 수 | 30 ~ 60 |

---

### 10. trade_gate

트레이딩 게이트 설정입니다.

```json
{
  "trade_gate": {
    "enabled": true,
    "max_daily_trades": 3,
    "min_consecutive_signals": 2,
    "min_confidence": "MEDIUM",
    "min_prob_buy": 0.62,
    "max_prob_sell": 0.38,
    "require_consensus": true,
    "target_profit_pt": 2.0,
    "stop_loss_pt": 1.0,
    "market_open_time": "08:45",
    "slot_a_end": "10:30",
    "slot_b_end": "13:00",
    "force_close_time": "15:30",
    "reverse_close_count": 2,
    "iv_dynamic_enabled": true,
    "iv_target_mult": 0.5,
    "iv_stop_mult": 0.25,
    "iv_target_min": 1.5,
    "iv_target_max": 5.0,
    "iv_stop_min": 0.75,
    "iv_stop_max": 2.5,
    "gamma_gate_enabled": false,
    "confidence_dynamic_enabled": true,
    "confidence_high_target_mult": 1.5,
    "confidence_high_stop_mult": 0.8,
    "confidence_medium_target_mult": 1.0,
    "confidence_medium_stop_mult": 1.0,
    "confidence_low_target_mult": 0.7,
    "confidence_low_stop_mult": 1.3,
    "max_consecutive_losses": 3,
    "max_daily_loss_pt": 5.0,
    "slot_performance_enabled": false,
    "trailing_stop_enabled": false,
    "trailing_stop_activation_pt": 1.0,
    "trailing_stop_distance_pt": 0.5,
    "history_save_enabled": true,
    "history_dir": "trade_history"
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| enabled | bool | true | 활성화 여부 | true/false |
| max_daily_trades | int | 3 | 일일 최대 트레이드 | 1 ~ 5 |
| min_confidence | string | "MEDIUM" | 최소 신뢰도 | LOW, MEDIUM, HIGH |
| min_prob_buy | float | 0.62 | 최소 매수 확률 | 0.55 ~ 0.7 |
| max_prob_sell | float | 0.38 | 최대 매도 확률 | 0.3 ~ 0.45 |
| target_profit_pt | float | 2.0 | 목표 수익 (pt) | 1.5 ~ 3.0 |
| stop_loss_pt | float | 1.0 | 손절 (pt) | 0.5 ~ 2.0 |
| iv_dynamic_enabled | bool | true | ATM IV 기반 동적 목표/손절 활성화 | true/false |
| confidence_dynamic_enabled | bool | true | 신뢰도 기반 동적 목표/손절 활성화 | true/false |
| confidence_high_target_mult | float | 1.5 | HIGH confidence 목표 배수 | 1.2 ~ 2.0 |
| confidence_high_stop_mult | float | 0.8 | HIGH confidence 손절 배수 | 0.5 ~ 1.0 |
| confidence_medium_target_mult | float | 1.0 | MEDIUM confidence 목표 배수 | 0.8 ~ 1.2 |
| confidence_medium_stop_mult | float | 1.0 | MEDIUM confidence 손절 배수 | 0.8 ~ 1.2 |
| confidence_low_target_mult | float | 0.7 | LOW confidence 목표 배수 | 0.5 ~ 0.9 |
| confidence_low_stop_mult | float | 1.3 | LOW confidence 손절 배수 | 1.0 ~ 1.5 |
| max_consecutive_losses | int | 3 | 최대 연속 손실 횟수 (0=비활성) | 0 ~ 5 |
| max_daily_loss_pt | float | 5.0 | 일일 최대 손실 (pt, 0=비활성) | 0 ~ 10 |
| slot_performance_enabled | bool | false | 슬롯별 성과 기반 할당 활성화 | true/false |
| trailing_stop_enabled | bool | false | Trailing Stop-loss 활성화 | true/false |
| trailing_stop_activation_pt | float | 1.0 | Trailing 시작 이익 (pt) | 0.5 ~ 2.0 |
| trailing_stop_distance_pt | float | 0.5 | Trailing 거리 (pt) | 0.3 ~ 1.0 |

---

## 사용 예시

### 기본 설정

```json
{
  "prediction": {
    "numeric_predictor": "ensemble",
    "buy_threshold": 0.64,
    "sell_threshold": 0.36
  }
}
```

### 고성능 설정

```json
{
  "prediction": {
    "numeric_predictor": "ensemble",
    "model_class": "patch_tst",
    "mamba_enabled": true,
    "dual_llm": true
  }
}
```

### 보수적 설정

```json
{
  "prediction": {
    "buy_threshold": 0.7,
    "sell_threshold": 0.3,
    "disagreement_hold": true
  },
  "trade_gate": {
    "min_confidence": "HIGH",
    "require_consensus": true
  }
}
```

## 주의사항

1. **API Key**: AI 제공자 API Key는 절대 config.json에 직접 저장하지 마세요. 환경변수 또는 `config.secrets.json`을 사용하세요.
2. **백업**: 설정 변경 전 반드시 백업을 만드세요.
3. **검증**: 새 설정은 백테스트로 검증 후 적용하세요.
4. **재시작**: 설정 변경 후 시스템 재시작이 필요합니다.

## 관련 문서

- [머신러닝 엔진 개요](./ML_ENGINE_OVERVIEW.md)
- [트레이딩 시그널 생성 가이드](./TRADING_SIGNAL_GENERATION_GUIDE.md)
- [듀얼 모드 구조 가이드](./DUAL_MODE_GUIDE.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
