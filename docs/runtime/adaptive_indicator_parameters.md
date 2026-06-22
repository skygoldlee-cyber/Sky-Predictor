# Adaptive Indicator Parameters (AST / AZZ)

이 문서는 `config.json`의 `adaptive_indicator.supertrend`(Adaptive SuperTrend, AST) 및
`adaptive_indicator.zigzag`(Adaptive ZigZag, AZZ) 파라미터의 **의미 / 권장 범위 / 튜닝 가이드**를 정리합니다.

- 이 문서의 목적은 “값을 무엇으로 설정해야 하는가”에 대한 **운영/튜닝 기준점**을 제공하는 것입니다.
- 여기의 범위는 하드 제한이 아니라 **권장 시작점(starting point)** 입니다.
- 최종적으로는 `Replay`(ticks 리플레이)로 최근 데이터에 대해 스모크 확인을 권장합니다.

---

## 1) 적용 위치

`config.json`:

```json
{
  "adaptive_indicator": {
    "enabled": true,
    "warmup_bars": 50,
    "supertrend": { /* AST */ },
    "zigzag": { /* AZZ */ }
  }
}
```

GUI:

- GUI의 `Adaptive indicators` 섹션에서 값을 입력하고 `Start` 또는 `Replay` 실행 시
  해당 값이 `config.json`에 저장됩니다.

---

## 2) 공통 원칙(튜닝 방향)

- **민감도↑(더 빨리 반응)**
  - AST: `multiplier_*` ↓, `atr_*_period` ↓, `smooth_period` ↓
  - AZZ: `pivot_threshold_min_pct` ↓, `atr_multiplier_min/max` ↓, `confirmation_bars` ↓, `min_wave_bars` ↓

- **안정성↑(노이즈/휩쏘 감소)**
  - AST: `multiplier_*` ↑, `smooth_period` ↑, `atr_max_period` ↑
  - AZZ: `pivot_threshold_min_pct` ↑, `confirmation_bars` ↑, `min_wave_bars` ↑, `freeze_on_confirm=true`

---

## 3) Adaptive SuperTrend (AST) — `adaptive_indicator.supertrend`

AST는 SuperTrend의 핵심 파라미터(ATR 기간, multiplier)를 **시장 상태(ER/ADX/BB 폭)**에 따라 동적으로 조정합니다.

### 3.1 핵심 파라미터

#### (1) `atr_min_period`, `atr_max_period`

- **의미**
  - 적응형 ATR 기간의 하한/상한.
  - ER이 높을수록(추세) 짧은 기간(`atr_min_period`)에 가까워지고,
    ER이 낮을수록(횡보) 긴 기간(`atr_max_period`)에 가까워집니다.

- **권장 범위(시작점)**
  - `atr_min_period`: 5 ~ 10
  - `atr_max_period`: 14 ~ 30

- **튜닝 팁**
  - 휩쏘가 많으면 `atr_max_period`를 늘려 횡보 구간 ATR을 더 안정적으로 만들 수 있습니다.

#### (2) `multiplier_min`, `multiplier_max`

- **의미**
  - SuperTrend 밴드 폭을 결정하는 multiplier의 하한/상한.
  - ADX/BB 보정이 들어간 후에도 최종 multiplier가 이 구간에서 움직이도록 설계합니다.

- **권장 범위(시작점)**
  - `multiplier_min`: 1.5 ~ 2.5
  - `multiplier_max`: 3.5 ~ 5.5

- **튜닝 팁**
  - 너무 자주 방향 flip이 발생하면: `multiplier_min/max`를 상향
  - 신호가 너무 늦으면: `multiplier_min/max`를 하향

#### (3) `er_period`

- **의미**
  - Kaufman Efficiency Ratio(ER) 계산 기간.

- **권장 범위(시작점)**
  - 8 ~ 20

- **튜닝 팁**
  - 값을 줄이면 시장 상태 판단이 더 빨라지지만, 노이즈에 민감해질 수 있습니다.

#### (4) `adx_period`

- **의미**
  - ADX 계산 기간.
  - ADX는 multiplier 조정(추세 강도 반영)에 사용됩니다.

- **권장 범위(시작점)**
  - 10 ~ 20

#### (5) `smooth_period`

- **의미**
  - SuperTrend 라인(`value`)을 EMA로 스무딩하는 기간.
  - `1`이면 스무딩 없음.

- **권장 범위(시작점)**
  - 1 ~ 7

- **튜닝 팁**
  - 휩쏘가 많으면 `3~5` 권장

### 3.2 보정 파라미터(선택)

#### (1) `use_bb_correction`, `bb_period`, `bb_std`

- **의미**
  - 볼린저 밴드 폭을 이용해 multiplier를 보정할지 여부.

- **권장 범위(시작점)**
  - `bb_period`: 14 ~ 30
  - `bb_std`: 1.5 ~ 2.5

#### (2) `adx_mult_norm_cap`, `bb_correction_floor`, `bb_correction_ref_pct`

- **의미**
  - ADX/BB 기반 정규화 및 보정 스케일을 결정하는 파라미터.

- **권장 범위(시작점)**
  - `adx_mult_norm_cap`: 40 ~ 80
  - `bb_correction_floor`: 0.5 ~ 0.9
  - `bb_correction_ref_pct`: 0.03 ~ 0.10

---

## 4) Adaptive ZigZag (AZZ) — `adaptive_indicator.zigzag`

AZZ는 ATR과 ER을 이용해 전환 임계값(threshold)을 동적으로 계산하고,
스윙/피보/지지저항/구조를 추적합니다.

### 4.1 임계값(Threshold) 관련

#### (1) `atr_period`

- **의미**
  - ATR 계산 기간.

- **권장 범위(시작점)**
  - 10 ~ 20

#### (2) `er_period`

- **의미**
  - ER 계산 기간. ER은 동적 multiplier 선택에 사용됩니다.

- **권장 범위(시작점)**
  - 8 ~ 20

#### (3) `atr_multiplier_min`, `atr_multiplier_max`

- **의미**
  - ER에 따라 선택되는 multiplier 범위.

    ER이 높으면(추세) `max`에 가까워집니다.
    ER이 낮으면(횡보) `min`에 가까워집니다.

- **권장 범위(시작점)**
  - `atr_multiplier_min`: 0.8 ~ 1.8
  - `atr_multiplier_max`: 3.0 ~ 6.0

- **튜닝 팁**
  - 스윙이 너무 안 잡히면: `atr_multiplier_max`를 낮추거나 `pivot_threshold_min_pct`를 낮추세요.
  - 스윙이 너무 잦으면: `atr_multiplier_min/max`를 올리거나 `pivot_threshold_min_pct`를 올리세요.

#### (4) `atr_multiplier`

- **의미**
  - ER 워밍업 구간에서 기본값/중간값 성격으로 사용되는 multiplier.
  - 운영에선 보통 `atr_multiplier_min/max`가 더 중요하며,
    `atr_multiplier`는 “기본값”으로 두는 편이 안전합니다.

- **권장 범위(시작점)**
  - 1.0 ~ 3.0

#### (5) `pivot_threshold_min_pct`, `pivot_threshold_max_pct`

- **의미**
  - 최종 threshold(%)의 하한/상한.
  - 지나치게 작거나 커지는 상황을 방지하는 safety clamp입니다.

- **권장 범위(시작점)**
  - `pivot_threshold_min_pct`: 0.2 ~ 1.0
  - `pivot_threshold_max_pct`: 2.0 ~ 6.0

### 4.2 스윙 확정/노이즈 억제

#### (1) `confirmation_bars`

- **의미**
  - reversal 감지 후 바로 확정하지 않고, N봉 뒤에 확정하는 지연.

- **권장 범위(시작점)**
  - 0 ~ 5

- **튜닝 팁**
  - 만기주/노이즈 구간: 2~4 권장

#### (2) `freeze_on_confirm`

- **의미**
  - `confirmation_bars > 0`에서 후보 스윙(price/idx)을 고정할지 여부.

- **권장**
  - `true` 권장(소급 변경 repainting 완화)

#### (3) `min_wave_bars`, `min_wave_pct`

- **의미**
  - reversal 감지 후에도 “최소 파동 길이” 조건을 만족할 때만 확정하도록 하는 필터.

- **권장 범위(시작점)**
  - `min_wave_bars`: 1 ~ 15
  - `min_wave_pct`: 0.0 ~ 3.0

- **튜닝 팁**
  - 스윙이 너무 잦으면: `min_wave_bars`를 올리세요.
  - `min_wave_pct`는 0이면 비활성(권장 시작점은 0).

### 4.3 스윙 분류 / 상태 제한

#### (1) `major_swing_ratio`

- **의미**
  - 직전 동일 타입 스윙 대비 크기가 `atr × ratio` 이상이면 major 스윙으로 분류.

- **권장 범위(시작점)**
  - 1.5 ~ 3.5

#### (2) `max_swings`

- **의미**
  - 유지할 최근 스윙 개수 제한.

- **권장 범위(시작점)**
  - 10 ~ 60

### 4.4 지지/저항/구조 분석

#### (1) `cluster_tolerance_pct`

- **의미**
  - 지지/저항 클러스터링 허용 오차(%).

- **권장 범위(시작점)**
  - 0.2 ~ 0.8

#### (2) `structure_lookback_swings`, `structure_points`

- **의미**
  - 구조(uptrend/downtrend/ranging) 판정에 사용할 최근 스윙 개수/샘플 개수.

- **권장 범위(시작점)**
  - `structure_lookback_swings`: 6 ~ 20
  - `structure_points`: 2 ~ 4

---

## 5) 만기주(Expiry week) 추천 시작 프리셋

만기주 프리셋은 `ADAPTIVE_INDICATOR_GUIDE.md`의 8.7 섹션을 참고하세요.
