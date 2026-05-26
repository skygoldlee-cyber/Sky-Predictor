# 만기주 콜 등가 이탈 탐지 설계

**대상 프로젝트:** Transformer  
**연관 파일:** `prediction/option_features.py`, `prediction/features.py`, `prediction/pipeline.py`, `prediction/context_builder.py`, `constants.py`  
**최종 갱신:** 2026-03-07 (v2 — 코드 검토 후 공식·임계값 수정 반영)

---

## 문제 정의

콜-풋 패리티(Call-Put Parity)에 따르면 ATM 등가 콜과 풋의 가격 움직임은 선물가격 변화와 이론적으로 일정한 관계를 유지해야 한다:

```
C - P = F - K·e^(-rT)   (단순화: C - P ≈ F - K)
```

만기가 가까워질수록(T → 0) 이 관계가 무너지는 경우가 발생한다. 구체적으로:

- **ATM 콜 가격이 선물 상승에 비해 덜 오르거나 / 더 빠르게 하락**하는 경우
- **콜-풋 가격 합(스트래들 가격)이 선물 변동성에 비해 수축/과팽창**하는 경우
- **등가 콜의 델타가 0.5에서 이탈**하거나 (만기 근접 gamma squeeze)
- **콜-풋 스프레드가 베이시스를 벗어나는** 경우

이를 탐지하면 다음 시나리오를 사전에 포착할 수 있다:
1. 방향성 포지션 청산/롤오버에 따른 일시적 가격 왜곡
2. 대규모 헤지 수요로 인한 ATM 옵션 프리미엄 급등
3. 이론가 대비 과매도/과매수 ATM 옵션 (아비트라지 기회 신호)

---

## 추가된 피처 모음: `calc_parity_divergence()`

### 위치: `prediction/option_features.py`

```python
def calc_parity_divergence(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    days_to_expiry: float = 7.0,
    risk_free_rate: float = 0.035,
    prev_underlying_price: Optional[float] = None,
    prev_atm_call_price: Optional[float] = None,
    prev_atm_put_price: Optional[float] = None,
) -> Dict[str, float]:
    """
    ATM 콜-풋 패리티 이탈 지표를 계산한다.

    반환 피처:
        parity_spread        : C - P - (F - K·e^{-rT}). 이론상 0에 수렴해야 함.
        parity_spread_pct    : parity_spread / F * 100. 비율 정규화.
        call_delta_proxy     : C / (C + P). ATM 이론값 0.5에서의 이탈.
        straddle_price       : C + P. 내재 변동성 크기의 직접 지표.
        straddle_vs_fut_move : straddle_price / max(|F - K|, 0.1).
        call_vs_fut_ret_diff : (콜 수익률) - (0.5 × 선물 수익률).
                               직전 틱 대비 콜이 선물을 얼마나 추종하는지.
        dte_weight_norm      : min(1 / (max(dte, 0.1) × 10), 1.0).
                               만기 근접일수록 가중치 증가.
        parity_divergence_score : 위 지표들의 가중 합성 이탈 스코어 [-1, 1].

    Notes:
        - tick_processor.process_option_tick()이 저장하는 'price' 필드를 사용한다.
        - prev_* 인자가 None이면 call_vs_fut_ret_diff는 0.0으로 반환된다.
        - Best-effort: 모든 계산 실패 시 zeros를 반환한다.
    """
```

#### 핵심 계산 로직

**1. ATM 콜/풋 가격 추출**
```python
# calc_iv_skew와 동일한 ATM 탐색 로직 재사용
# round(F * 2) / 2 로 0.5pt 단위 ATM 앵커 설정, ±2.5pt 이내 탐색
atm_strike = _find_atm_strike(calls, puts, underlying_price)
# _get_atm_option_price: "price" 필드 1차, bid/ask 중간값 fallback
atm_call_price = _get_atm_option_price(calls, atm_strike)
atm_put_price  = _get_atm_option_price(puts, atm_strike)
```

**2. 패리티 스프레드 (핵심 지표)**
```python
F = underlying_price
K = atm_strike
T = days_to_expiry / 365.0
r = risk_free_rate

# 이론 패리티: C - P = F - K * exp(-rT)
theoretical_diff = F - K * math.exp(-r * T)
actual_diff      = atm_call_price - atm_put_price
parity_spread    = actual_diff - theoretical_diff        # 0이면 완전 이론적
parity_spread_pct = parity_spread / F * 100.0 if F > 0 else 0.0
```

**3. 콜 델타 프록시 (0.5에서의 이탈)**
```python
# BS 델타를 직접 계산하지 않고, C/(C+P) 비율로 근사
# ATM에서는 이론상 0.5. 만기 근접 + 가격 이탈 시 0.5에서 벗어남
straddle = atm_call_price + atm_put_price
call_delta_proxy = atm_call_price / straddle if straddle > 0.0 else 0.5
```

**4. 콜 수익률 vs 선물 수익률 차이 (직전 틱 비교)**
```python
# prev_* 인자가 있을 때만 계산
if prev_underlying_price and prev_atm_call_price:
    fut_ret  = (F - prev_underlying_price) / prev_underlying_price
    call_ret = (atm_call_price - prev_atm_call_price) / prev_atm_call_price
    # ATM 콜의 이론 델타 ≈ 0.5이므로 call_ret ≈ 0.5 * fut_ret 이어야 함
    call_vs_fut_ret_diff = call_ret - (0.5 * fut_ret)
```

**5. DTE 가중치 (만기 근접일수록 이탈 신호 증폭)**
```python
# min(1 / (max(dte, 0.1) * 10), 1.0)
# dte= 0일 → 1.000
# dte= 1일 → 0.100
# dte= 3일 → 0.033
# dte= 7일 → 0.014   ← 7일 이상이면 신호 사실상 소멸
# dte=10일 → 0.010
# dte=30일 → 0.003
dte_weight_norm = min(1.0 / (max(dte, 0.1) * 10.0), 1.0)
```

> **v1 구현 대비 변경점:** 초기 코드는 `1/(1+dte*0.15)` 공식을 사용하여
> 7일 전에도 dte_weight=0.49로 과도하게 높았다. 수정 후 공식은 만기 7일 이상에서
> 신호가 사실상 소멸(0.014)하여 만기주 전용 특성이 올바르게 반영된다.

**6. 종합 이탈 스코어**
```python
raw_score = (
    np.clip(parity_spread_pct * 0.4,        -1.0, 1.0) * 0.4   # 패리티 이탈 비율
    + np.clip((call_delta_proxy - 0.5) * 2.0, -1.0, 1.0) * 0.3  # 델타 비대칭
    + np.clip(call_vs_fut_ret_diff * 10.0,   -1.0, 1.0) * 0.3   # 수익률 추종 이탈
)
# 만기 근접일수록 신호 진폭 증가 (최대 2배)
parity_divergence_score = np.clip(raw_score * (1.0 + dte_weight_norm), -1.0, 1.0)
```

> **v1 구현 대비 변경점:** delta 계수 `4.0 → 2.0`, ret_diff 계수 `5.0 → 10.0`.
> 비중(0.4 / 0.3 / 0.3)은 유지.

---

## `build_option_snapshot()` 수정 — 시그니처 확장

> **구조 변경 핵심:**
> 초기 설계는 `pipeline.py`가 `snap` dict에 `_prev_*` 키를 삽입하면
> `build_option_snapshot` 내부에서 `snap.pop()`으로 소비하는 방식이었다.
> 이 방식은 `calc_parity_divergence`가 1차(prev=None) + 2차(prev 있음) **이중 실행**되는
> 구조적 결함을 유발했다. 수정 후에는 `prev_*`를 함수 파라미터로 직접 전달한다.

```python
def build_option_snapshot(
    calls, puts, underlying_price,
    *,
    tick_processor=None,
    option_feature_set="v1",
    prev_underlying_price: Optional[float] = None,   # ← 추가
    prev_atm_call_price: Optional[float] = None,     # ← 추가
    prev_atm_put_price: Optional[float] = None,      # ← 추가
) -> Dict[str, Any]:
```

```python
# v3 분기 내부 — snap.pop() 방식 제거, 파라미터 직접 사용
if fs == "v3":
    dte = float(get_expiry_week_info().get("days_to_expiry") or 7.0)
    snap.update(
        calc_parity_divergence(
            calls or {},
            puts or {},
            float(underlying_price or 0.0),
            days_to_expiry=dte,
            prev_underlying_price=prev_underlying_price,
            prev_atm_call_price=prev_atm_call_price,
            prev_atm_put_price=prev_atm_put_price,
        )
    )
```

---

## `features.py` 수정: OPT_KEYS_V3 추가

```python
# V1 = 7개,  V2 = 16개,  V3 = 23개
OPT_KEYS_V3 = OPT_KEYS_V2 + [
    "parity_spread_pct",        # 패리티 이탈 비율 (%)
    "call_delta_proxy",         # 콜 델타 근사값 [0, 1]
    "straddle_price",           # ATM 스트래들 가격
    "straddle_vs_fut_move",     # 스트래들 대비 선물 이동 배율
    "call_vs_fut_ret_diff",     # 콜 수익률 vs 선물 수익률 차이
    "dte_weight_norm",          # 만기 거리 가중치 [0, 1]
    "parity_divergence_score",  # 종합 이탈 스코어 [-1, 1]
]
```

---

## `pipeline.py` 수정: 직전 상태 캐싱 + 단일 호출 구조

```python
# PredictionPipeline.__init__ 에 추가
self._prev_atm_call_price: Optional[float] = None
self._prev_atm_put_price: Optional[float] = None
self._prev_underlying_price: Optional[float] = None
```

### `_build_option_snapshot_safe()` — prev 단일 전달 구조

```python
def _build_option_snapshot_safe(self, *, current_price: float, update_prev: bool = True):
    """
    update_prev=True  : OB 버퍼(1Hz) 경로 — 매 초 _prev_* 갱신
    update_prev=False : get_prediction() 경로 — 이중 갱신 방지
    """
    # v3일 때만 prev 값 전달. 그 외에는 None → diff 계산 비활성화.
    prev_u = self._prev_underlying_price if self._option_feature_set == "v3" else None
    prev_c = self._prev_atm_call_price   if self._option_feature_set == "v3" else None
    prev_p = self._prev_atm_put_price    if self._option_feature_set == "v3" else None

    snap = build_option_snapshot(
        ...,
        option_feature_set=str(self._option_feature_set),
        prev_underlying_price=prev_u,
        prev_atm_call_price=prev_c,
        prev_atm_put_price=prev_p,
    )

    if update_prev:
        self._prev_underlying_price = float(current_price)
        # atm_strike는 snap["atm_strike"] 재사용, 없으면 call_options에서 직접 탐색
        atm_k = snap.get("atm_strike") or (nearest strike from call_options)
        if atm_k > 0:
            cp = _get_atm_option_price(call_options, atm_k)
            pp = _get_atm_option_price(put_options, atm_k)
            self._prev_atm_call_price = cp if cp > 0 else self._prev_atm_call_price
            self._prev_atm_put_price  = pp if pp > 0 else self._prev_atm_put_price
```

---

## LLM 컨텍스트 보강 (`context_builder.py`)

### `_describe_parity_divergence()` 신설

```python
def _describe_parity_divergence(opt_snap: dict) -> str:
    """
    dte_weight_norm < 0.1 (만기 7일 이상)이면 빈 문자열 반환.
    의미 있는 이탈 신호가 없을 때도 빈 문자열 반환.
    """
    score    = float(opt_snap.get("parity_divergence_score") or 0.0)
    dte_w    = float(opt_snap.get("dte_weight_norm") or 0.0)
    spread   = float(opt_snap.get("parity_spread_pct") or 0.0)
    delta_p  = float(opt_snap.get("call_delta_proxy") or 0.5)
    ret_diff = float(opt_snap.get("call_vs_fut_ret_diff") or 0.0)

    if dte_w < 0.1:   # 만기 7일 이상 → 신호 무의미
        return ""

    lines = []
    if abs(score) >= 0.3:
        direction = "콜 과매도(저평가)" if score < 0 else "콜 과매수(고평가)"
        lines.append(f"패리티 이탈 감지: {direction} (score={score:.2f}, dte_weight={dte_w:.3f})")
    if abs(spread) >= 0.1:
        lines.append(f"  패리티 스프레드: {spread:+.2f}% (C-P 실제값 - 이론값, 0에 가까울수록 균형)")
    if abs(delta_p - 0.5) >= 0.05:
        lines.append(f"  콜 델타 비대칭: {delta_p:.3f} (ATM 이론값=0.50, 이탈 클수록 한쪽 방향 포지션 쏠림)")
    if abs(ret_diff) >= 0.002:
        dir2 = "과소추종(콜 저평가 가능)" if ret_diff < 0 else "과다추종(콜 고평가 가능)"
        lines.append(f"  콜 수익률 추종 이탈: {ret_diff:+.4f} ({dir2})")

    return "\n".join(lines)
```

### `build_llm_context()` 내 [PARITY_ANALYSIS] 섹션

```python
# [OPTIONS_SNAPSHOT] 출력 직후 삽입
parity_desc = _describe_parity_divergence(opt_snap)
if parity_desc:
    lines.append("")
    lines.append("[PARITY_ANALYSIS]")
    lines.append(parity_desc)
```

생성되는 LLM 컨텍스트 예시 (만기 2일 전, 강한 콜 과매도):
```
[OPTIONS_SNAPSHOT]
{ "pcr_volume": 1.23, "atm_iv": 18.5, "parity_divergence_score": -0.71, ... }

[PARITY_ANALYSIS]
패리티 이탈 감지: 콜 과매도(저평가) (score=-0.71, dte_weight=0.056)
  패리티 스프레드: -0.35% (C-P 실제값 - 이론값, 0에 가까울수록 균형)
  콜 델타 비대칭: 0.438 (ATM 이론값=0.50, 이탈 클수록 한쪽 방향 포지션 쏠림)
  콜 수익률 추종 이탈: -0.0031 (과소추종(콜 저평가 가능))
```

---

## 탐지 활용: 가드레일 (`pipeline.py`)

```python
def _apply_parity_guardrail(self, *, signal, confidence, opt_snap) -> Tuple[str, str, Optional[str]]:
    """
    option_feature_set == "v3" 일 때만 동작.
    적용 순서: option_guardrail → basis_guardrail → parity_guardrail (마지막)
    """
```

### 가드레일 임계값

| 조건 | dte_weight_norm | 만기 잔존일 | 동작 |
|---|---|---|---|
| `score >= 0.8` | `>= 1.0` | 만기 당일 (dte=0) | BUY/SELL → **HOLD** + LOW |
| `score >= 0.5` | `>= 0.033` | dte ≤ 3일 | MEDIUM → **LOW** 강등 |
| — | `< 0.1` | dte > 7일 | 가드레일 비활성 |

> **v1 구현 대비 변경점:**
> - HOLD 임계값: `dte_w >= 0.9` → `dte_w >= 1.0` (만기 당일에만)
> - 강등 임계값: `dte_w >= 0.33` → `dte_w >= 0.033` (dte_weight 공식 변경에 맞춤)
> - HIGH→MEDIUM 강등 조건 제거 (초기 설계에 없던 조항)

---

## `constants.py` 추가

```python
# v1: 기본 7개  /  v2: v1 + micro-movement 9개  /  v3: v2 + parity divergence 7개
DEFAULT_OPTION_FEATURE_SET = "v3"
```

---

## 데이터 의존성

`tick_processor.py`의 `process_option_tick()`에 `"price": safe_float(tick.get("price"))` 저장 확인 완료 (351번 줄). `_get_atm_option_price()`가 이 필드를 1차로 읽고 bid/ask 중간값으로 fallback한다.

---

## 구현 완료 순서

| 단계 | 파일 | 작업 | 상태 |
|---|---|---|---|
| 1 | `tick_processor.py` | 옵션 틱 `price` 필드 저장 확인 | ✅ 기존 구현 확인 |
| 2 | `option_features.py` | `calc_parity_divergence()` 함수 추가 | ✅ |
| 3 | `option_features.py` | `build_option_snapshot()` 시그니처 확장 + v3 분기 | ✅ snap.pop → 파라미터 방식 교체 |
| 4 | `features.py` | `OPT_KEYS_V3` 정의 및 `get_opt_keys()` 분기 추가 | ✅ |
| 5 | `pipeline.py` | `_prev_atm_*` 상태 캐싱 추가 | ✅ |
| 6 | `pipeline.py` | `_build_option_snapshot_safe()` 단일 호출 구조 정리 | ✅ 이중실행 제거 |
| 7 | `pipeline.py` | `_apply_parity_guardrail()` 추가 및 연동 | ✅ |
| 8 | `context_builder.py` | `_describe_parity_divergence()` + [PARITY_ANALYSIS] | ✅ |
| 9 | `constants.py` | `DEFAULT_OPTION_FEATURE_SET = "v3"` | ✅ |

---

## 핵심 인사이트

만기주의 콜 등가 이탈은 단순히 "가격이 다르다"가 아니라 **선물-옵션 간 정보 비대칭 신호**다. 통상적으로:

- **콜이 선물보다 느리게 오를 때**: 콜 매도 포지션 대량 청산 중 → 선물 상승 모멘텀이 일시적으로 옵션 시장에 반영 안 됨 → 콜 매수 기회
- **콜이 선물보다 빠르게 오를 때**: 외가 콜 매수 급증 (방향성 배팅) → 상방 브레이크아웃 신호 또는 gamma squeeze 시작
- **스트래들 가격이 선물 변동폭 대비 과도하게 높을 때**: IV 과팽창 → 방향성 예측 자체가 어려운 구간 → HOLD 가중

`parity_divergence_score`는 이 세 케이스를 하나의 스칼라로 압축한 값이며,
`[PARITY_ANALYSIS]` 섹션을 통해 LLM이 "만기 N일 전, 콜이 선물 움직임을 X% 덜 추종하고 있음"이라는 맥락을 자연어로 받아 판단에 활용한다.

**dte_weight_norm이 0.1 미만(만기 7일 이상)이면 이 신호 전체를 무시**하는 것이 올바른 해석이다. 패리티 이탈은 만기주 고유 현상이며, 일반 거래일에는 노이즈에 가깝다.
