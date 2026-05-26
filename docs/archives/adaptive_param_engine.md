# AdaptiveParamEngine - 장중 자기완결형 적응형 파라미터 조정 엔진

## 개요

`AdaptiveParamEngine`은 외부 레짐 분류기나 MarketRegimeClassifier 없이, `AdaptiveZigZag` 내부 버퍼만으로 장중 피봇 탐색 파라미터를 실시간 자동 조정하는 자기완결형 엔진입니다.

## 특징

- **자기완결형**: 외부 의존성 없이 ZigZag 내부 버퍼만으로 동작
- **실시간 조정**: 매 봉마다 파라미터 동적 조정
- **레짐 기반**: 시장 상태에 따른 최적 파라미터 프로파일 적용
- **EMA 스무딩**: 파라미터 급격한 변동 방지 (α=0.15)
- **피드백 루프**: 피봇 밀도에 따른 미세 조정

## 통합 방법

### 1. AdaptiveZigZag.__init__()에 엔진 초기화 추가

```python
# ── [자기완결형 적응 엔진] ─────────────────────────────────────
self._adaptive_engine = None
try:
    from .adaptive_param_engine import AdaptiveParamEngine
    self._adaptive_engine = AdaptiveParamEngine(self.config)
    _logger.info("[AdaptiveZigZag] 적응형 파라미터 엔진 활성화")
except Exception as e:
    _logger.warning("[AdaptiveZigZag] 적응형 파라미터 엔진 초기화 실패: %s", e)
```

### 2. _get_runtime_params()에 엔진 기반 로직 추가

```python
# Layer C: 적응형 엔진 (자기완결형)
if self._adaptive_engine is not None:
    try:
        adjusted = self._adaptive_engine.compute(
            atr_values=list(self._atr_values),
            all_swings=self._all_swings,
            bar_idx=self._bar_idx,
            er=float(self._calc_er()),
            der=float(self._calc_der()),
            direction=self._current_direction,
            last_confirmed_bar_idx=self._last_confirmed_bar_idx,
        )
        # 결합: config 수정 없이 런타임 dict로만
        return {
            "atr_multiplier": float(np.clip(
                cfg.atr_multiplier * adjusted.mult,
                cfg.atr_multiplier_min, cfg.atr_multiplier_max,
            )),
            "confirmation_bars": adjusted.confirmation_bars,
            "min_wave_atr_ratio": float(np.clip(
                a_atr_ratio * adjusted.wave_ratio_mult,
                0.5, 5.0,
            )),
            "min_wave_bars": a_wave_bars,
            "pivot_threshold_min_pct": float(np.clip(
                cfg.pivot_threshold_min_pct * adjusted.thr_mult,
                cfg.pivot_threshold_min_pct * 0.5,
                cfg.pivot_threshold_max_pct,
            )),
        }
    except Exception as e:
        _logger.warning("[AdaptiveZigZag] 적응형 엔진 계산 실패, 기본값 사용: %s", e)

# 폴백: 기존 방식
...
```

## 신호 합성 구조

```
┌──────────────────────────────────────────────────────────────┐
│                   AdaptiveParamEngine.compute()              │
│                                                              │
│  Signal-A: ER  ─────────────────────┐                       │
│  (Efficiency Ratio 0~1)             │                        │
│                                     ├→ _classify_regime()   │
│  Signal-B: ATR 백분위 ──────────────┤   4가지 레짐 결정    │
│  (현재 변동성 위치 0~100%)          │                        │
│                                     │                        │
│  Signal-C: DER ─────────────────────┘                       │
│  (방향 ER, -1~+1)                                           │
│           │                                                  │
│           ↓                                                  │
│  REGIME_TABLE 조회 → 기준 배율 (mult/wave/thr/cb)            │
│           │                                                  │
│  피봇 밀도 피드백 → 배율 미세 보정 (±15%)                   │
│           │                                                  │
│  EMA 스무딩 (α=0.15) → 파라미터 깜빡임 방지                 │
│           │                                                  │
│  클램핑 → AdaptiveAdjustment 반환                           │
└──────────────────────────────────────────────────────────────┘
```

## 레짐별 파라미터 프로파일

| 레짐 | ER | ATR% | atr_mult | wave_ratio | cb | 효과 |
|------|-----|------|----------|-------------|----|----|
| trend_strong_up/dn | >0.60 | — | ×1.30 → 2.6 | ×1.20 | 1 | 작은 되돌림 무시, 빠른 확정 |
| trend_weak_up/dn | 0.35~0.60 | — | ×1.10 → 2.2 | ×1.05 | 2 | 중간 민감도 |
| chop_low_vol | <0.35 | <75% | ×0.80 → 1.6 | ×0.85 | 2 | 작은 반전도 포착 |
| chop_high_vol | <0.35 | >75% | ×1.40 → 2.8 | ×1.40 | 3 | 흔들기 노이즈 억제 |
| volatile | <0.35 | >75% | ×1.50 → 3.0 | ×1.50 | 2 | 급변동 관망 |
| unknown | — | — | ×1.00 → 2.0 | ×1.00 | 2 | 기본값 |

*기준값: config.atr_multiplier=2.0 기준*

## 레짐 결정 로직

```
ATR 백분위 > 75% + ER < 0.35  → volatile (급변동)
ER > 0.60                     → trend_strong_* (방향에 따라)
ER > 0.35                     → trend_weak_*
ER ≤ 0.35 + ATR > 75%        → chop_high_vol
ER ≤ 0.35 + ATR ≤ 75%        → chop_low_vol
```

## 파라미터 설명

### 입력 신호

- **ER (Efficiency Ratio)**: 추세 강도 0~1
  - 1.0에 가까울수록 강한 추세
  - 0.5에 가까울수록 횡보
  
- **ATR 백분위**: 현재 변동성 상대 위치 0~100%
  - 최근 60봉 ATR 분포에서 현재 ATR의 위치
  
- **DER (Direction ER)**: 방향성 ER -1~+1
  - 양수: 상승 추세
  - 음수: 하락 추세
  
- **피봇 밀도**: 최근 30봉 내 확정 피봇 수
  - 과다 (≥4): 임계값 상향
  - 부족 (<1): 임계값 하향

### 출력 배율

- **mult**: atr_multiplier 배율 (0.6~2.0 클램프)
- **wave_ratio_mult**: min_wave_atr_ratio 배율 (0.6~2.0 클램프)
- **thr_mult**: pivot_threshold_min_pct 배율 (0.6~1.5 클램프)
- **confirmation_bars**: 확정 대기 봉수 (1~4 클램프)

## EMA 스무딩

ER은 봉마다 변동하므로 레짐이 매 봉 바뀌면 atr_multiplier가 급격히 진동합니다. α=0.15의 EMA는 약 6봉의 시정수를 가져 레짐 전환을 부드럽게 반영합니다.

```python
self._ema_mult = a * base_mult + (1 - a) * self._ema_mult
```

- α=0.15: 약 6봉 시정수
- α=0.10: 약 9봉 시정수 (더 천천히)
- α=0.30: 약 3봉 시정수 (더 빠르게)

## config.json 권장 설정

엔진이 배율로 조정하므로 config 기준값은 "중립 상태"를 나타내도록 설정하면 됩니다:

```json
"futures_zigzag": {
  "use_atr_based_filtering": true,
  "atr_multiplier":     2.0,   // 중립 기준 (엔진이 1.6~3.0으로 조정)
  "atr_multiplier_min": 1.2,   // 하한 클램프
  "atr_multiplier_max": 4.0,   // 상한 클램프
  "pivot_threshold_min_pct": 0.5,
  "pivot_threshold_max_pct": 2.0,
  "confirmation_bars": 1       // 엔진이 1~4로 조정
}
```

## 모니터링

### PivotQualityMonitor 통합

`pivot_quality_monitor.py`에 적응형 엔진 상태 표시 기능이 추가되었습니다:

```python
# 적응형 엔진 조정 상태
adaptive_regime: str        # 현재 레짐 라벨
adaptive_mult: float        # ATR 배율
adaptive_er: float          # Efficiency Ratio
adaptive_atr_pct: float     # ATR 백분위
adaptive_density: str       # 피봇 밀도 신호
```

GUI에서 실시간으로 다음 정보를 확인할 수 있습니다:
- 레짐 (trend_strong_up, chop_low_vol 등)
- 배율 (현재 조정된 배율)
- ER (추세 강도)
- ATR% (변동성 위치)
- 밀도 (sparse/normal/dense)

## 원웨이장에서의 효과

원웨이장은 계속 상승하거나 계속 하락하는 추세적인 장입니다.

### confirmation_bars=1
- **추세 지속 시**: 빠른 확정으로 추세 진입 시점 포착에 유리
- **방향 전환 시**: 빠른 신호로 전환 포착 가능
- **적합도**: 높음

### freeze_on_confirm=false
- **추세 지속 시**: 극값 갱신으로 추세 강도 반영
- **방향 전환 시**: 새 극값으로 빠르게 전환
- **적합도**: 높음

### 적응형 엔진 시너지
원웨이장에서 레짐이 `trend_strong_up` 또는 `trend_strong_dn`으로 분류되면:
- **cb=1**: 자동으로 빠른 확정 적용
- **mult×1.30**: 큰 되돌림만 피봇으로 인식
- **결과**: 추세 방향의 주요 변곡점만 포착

## 지연확정 문제 대안

### 이중 신호 시스템 (권장)

지연확정이 10봉 이상인 경우:

- **예비 신호**: pending 등록 시점 (0봉 지연, 빠름)
- **확인 신호**: 실제 확정 시점 (지연 있음, 신뢰도 높음)

pending 등록 시 즉시 알림 → 확정 시 최종 확인

### 동적 confirmation_bars 조정

- 레짐이 `trend_strong_*`이면 cb=1로 강제
- 레짐이 `chop_*`이면 cb 증가하여 가짜 피봇 억제
- 지연이 길어지면 자동으로 cb 감소

### 신뢰도 기반 조기 확정

- ATR 백분위 > 80% + ER > 0.7 → 조기 확정 허용
- 피봇 밀도가 낮을 때 (30봉 내 1개 미만) → 조기 확정 허용

## 로그 예시

```
[AdaptiveParamEngine] regime=trend_strong_up er=0.65 atr_pct=82% density=normal mult=1.28 wave=1.18 thr=0.92 cb=1
[AdaptiveParamEngine] regime=chop_low_vol er=0.25 atr_pct=45% density=sparse mult=0.82 wave=0.88 thr=0.85 cb=2
[AdaptiveParamEngine] regime=volatile er=0.20 atr_pct=88% density=dense mult=1.45 wave=1.48 thr=1.18 cb=2
```

## 트러블슈팅

### 엔진이 활성화되지 않음

로그 확인:
```
[AdaptiveZigZag] 적응형 파라미터 엔진 활성화
```

활성화 로그가 없으면 `adaptive_param_engine.py` 파일이 존재하는지 확인하세요.

### 파라미터가 조정되지 않음

1. `_adaptive_engine`이 None인지 확인
2. `compute()` 호출 시 예외 발생 확인
3. 로그에 실패 메시지 확인

### 레짐이 항상 unknown

1. ER 계산 확인 (`_calc_er()`)
2. DER 계산 확인 (`_calc_der()`)
3. ATR 백분위 계산 확인

## 참고

- KP200 선물 1분봉 기준 설정
- ATR ≈ 0.3~0.8pt, 일간 범위 ≈ 5~12pt
- 주요 변곡점 파동 ≈ 2~5pt (일간 범위의 25~40%)
