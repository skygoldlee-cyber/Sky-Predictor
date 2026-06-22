# 만기주 프리미엄 블리드(Premium Bleed) 탐지 설계

**대상 프로젝트:** Transformer  
**연관 파일:** `prediction/option_features.py`, `prediction/features.py`, `prediction/context_builder.py`, `prediction/pipeline.py`, `telegram_notifier.py`  
**최종 갱신:** 2026-03-07 (v1 — 신규 설계)

---

## 문제 정의

콜-풋 패리티 이탈(`parity_divergence_score`, v3)은 C-P 스프레드의 방향성 왜곡을 탐지하지만, **선물이 상승하는 동안 스트래들 전체 가치(C+P)가 수축하는 현상**은 별도로 감지하지 못한다.

### 현상 설명

만기 당일~2일 전 구간에서 다음 패턴이 발생한다:

```
선물가격: 355.00 → 356.05  (+0.30% 상승)
ATM 콜:   3.20  →  2.95   (- 7.8% 하락)
ATM 풋:   1.02  →  0.90   (- 11.8% 하락)
스트래들: 4.22  →  3.85   (- 8.8% 수축)
```

선물이 상승했으므로 이론적으로 콜은 올라야 하고 풋은 약간 하락해야 한다. 그러나 **콜까지 함께 하락**하며 스트래들 전체가 수축한다.

### 주요 원인

| 원인 | 설명 |
|---|---|
| **Theta 급가속** | 만기 당일 ATM 옵션의 시간가치 소멸 속도가 지수적으로 증가 |
| **IV Crush** | 만기 이벤트 통과를 앞두고 내재변동성(IV) 급락 |
| **Vega 붕괴** | 만기 근접 시 Vega → 0, 변동성 변화가 옵션 가격에 반영 불가 |
| **MM 롤오버** | 시장조성자들이 다음 만기로 포지션 이동하며 프리미엄 방어 포기 |
| **감마 스퀴즈 소진** | 전일까지 쌓인 감마 헤지 수요가 사라지는 구간 |

이 현상을 탐지하면 다음 판단에 활용할 수 있다:
- 선물 상승 신호가 있더라도 옵션 시장이 이미 방향성을 포기한 구간
- 신규 방향성 진입보다 기존 포지션 청산이 지배적인 국면
- IV Crush로 인해 옵션 매수 전략이 불리한 구간

---

## v3와 v4의 탐지 영역 비교

| | `parity_divergence_score` (v3) | `premium_bleed_score` (v4) |
|---|---|---|
| **탐지 대상** | C-P 스프레드 vs 이론값 이탈 | 스트래들 전체 수축 vs 선물 방향 |
| **주요 원인** | 한쪽 방향 포지션 쏠림 | IV Crush, Theta 가속, 롤오버 |
| **신호 방향** | 양방향 (콜 과매수/과매도) | 주로 음수 (수축 방향) |
| **유효 구간** | 만기 3일 이내 | 만기 **당일~2일** 전 |
| **LLM 활용** | 방향성 왜곡 경고 | 옵션 시장 이탈 경고 |
| **가드레일 HOLD 임계** | `score >= 0.8`, `dte_w >= 1.0` | `score <= -0.75`, `dte_w >= 1.0` + 선물 상승 중 |

두 지표를 **AND 조건**으로 결합하면 오탐률이 크게 감소한다. 패리티 이탈과 프리미엄 수축이 동시에 발생하면 만기 당일 가장 위험한 구간이다.

---

## 신규 피처: `calc_premium_bleed()`

### 위치: `prediction/option_features.py`

```python
def calc_premium_bleed(
    calls, puts, underlying_price,
    *,
    days_to_expiry: float = 7.0,
    prev_underlying_price: Optional[float] = None,
    prev_atm_call_price: Optional[float] = None,
    prev_atm_put_price: Optional[float] = None,
) -> Dict[str, float]:
```

### 반환 피처 (6개)

| 피처명 | 설명 | 범위 |
|---|---|---|
| `straddle_decay_vs_fut` | 스트래들 수익률 - \|선물 수익률\| × 0.5. **핵심 지표.** 음수일수록 비정상 수축 | [-∞, ∞] |
| `iv_crush_proxy` | ATM IV 근사 변화율 = (σ_now - σ_prev) / σ_prev | [-∞, ∞] |
| `fut_ret` | 직전 틱 대비 선물 수익률. 방향 확인용 | [-∞, ∞] |
| `straddle_now` | 현재 ATM 스트래들 가격 (C + P) | [0, ∞) |
| `straddle_prev` | 직전 틱 ATM 스트래들 가격. 0이면 prev 없음 | [0, ∞) |
| `premium_bleed_score` | 종합 수축 스코어. -1 = 강한 수축, +1 = 강한 팽창 | [-1, 1] |

### 핵심 계산 로직

**1. 스트래들 수축률 vs 선물 수익률 (핵심 지표)**

```python
# 이론: 선물 1% 상승 시 ATM 스트래들은 거의 변화 없어야 함
# (콜 상승 ≈ 풋 하락으로 서로 상쇄되기 때문)
fut_ret       = (F - prev_F) / prev_F
straddle_ret  = (straddle_now - straddle_prev) / straddle_prev

# 기대치보다 스트래들이 더 많이 수축한 정도
straddle_decay_vs_fut = straddle_ret - abs(fut_ret) * 0.5
# 음수 = 이론 기대치보다 프리미엄이 더 많이 빠짐 → IV Crush / Theta 가속 신호
```

**2. IV Crush 근사 (BS 완전 구현 없이)**

```python
# ATM 이론 스트래들 ≈ F × σ × √T × √(2/π)
# 역산: σ_proxy = straddle / (F × √T × √(2/π))
T = days_to_expiry / 365.0
scale = F * math.sqrt(T) * math.sqrt(2.0 / math.pi)
sigma_now  = straddle_now  / scale
sigma_prev = straddle_prev / scale

iv_crush_proxy = (sigma_now - sigma_prev) / sigma_prev
# 음수 = 내재변동성 감소, 양수 = 내재변동성 증가
```

**3. DTE 가중치 (v3와 동일 공식 재활용)**

```
dte_weight_norm = min(1 / (max(dte, 0.1) × 10), 1.0)

dte = 0일  → 1.000 (만기 당일)
dte = 1일  → 0.100
dte = 3일  → 0.033
dte = 7일  → 0.014  ← 7일 이상이면 신호 사실상 소멸
```

**4. 종합 블리드 스코어**

```python
decay_component = clip(straddle_decay_vs_fut × 20, -1, 1)  # 수축 강도 (비중 60%)
iv_component    = clip(iv_crush_proxy × 5,         -1, 1)  # IV 방향 (비중 40%)

raw_score = decay_component × 0.6 + iv_component × 0.4

# 만기 근접일수록 신호 증폭 (최대 2배)
premium_bleed_score = clip(raw_score × (1 + dte_weight_norm), -1, 1)
```

**노이즈 필터:** 선물 수익률 `|fut_ret| < 0.0003` (0.03% 미만)이면 스코어 0.0 반환.  
KP200 선물 1틱 ≈ 0.05pt, 350pt 기준 약 0.014%이므로 2틱 미만 움직임은 무시한다.

---

## `features.py` 수정: OPT_KEYS_V4 추가

```python
# v4: v3 피처에 프리미엄 블리드 6개 피처 추가
OPT_KEYS_V4 = OPT_KEYS_V3 + [
    "straddle_decay_vs_fut",    # 스트래들 수익률 - |선물수익률|*0.5. 음수 = 비정상 수축.
    "iv_crush_proxy",           # ATM IV 방향 근사 변화율. 음수 = IV 감소.
    "fut_ret",                  # 직전 틱 선물 수익률. 방향 확인용.
    "straddle_now",             # 현재 ATM 스트래들 가격 (C+P).
    "straddle_prev",            # 직전 틱 ATM 스트래들 가격.
    "premium_bleed_score",      # 종합 프리미엄 수축 스코어 [-1, 1].
]
```

총 피처 수: v1 = 7개 / v2 = 16개 / v3 = 23개 / **v4 = 29개**

---

## `build_option_snapshot()` v4 분기

```python
# v3/v4: parity divergence 계산 (공통)
if fs in ("v3", "v4"):
    snap.update(calc_parity_divergence(...))

# v4 전용: premium bleed 계산 (parity 이후 실행, dte 역산 재활용)
if fs == "v4":
    snap.update(calc_premium_bleed(
        calls, puts, underlying_price,
        days_to_expiry=_dte_days,       # dte_weight_norm 역산값
        prev_underlying_price=prev_u,
        prev_atm_call_price=prev_c,
        prev_atm_put_price=prev_p,
    ))
```

`_prev_*` 상태 캐싱은 `pipeline._build_option_snapshot_safe()`에서 기존 v3 로직과 동일하게 v4에도 적용된다 (`if self._option_feature_set in ("v3", "v4")`).

---

## LLM 컨텍스트 보강 (`context_builder.py`)

### `_describe_premium_bleed()` 신설

```python
def _describe_premium_bleed(opt_snap: dict) -> str:
    """
    premium_bleed_score가 없거나 dte_weight_norm < 0.1이면 빈 문자열 반환.
    straddle_prev == 0 (prev 없음)이면 빈 문자열 반환.
    |score| < 0.3이면 빈 문자열 반환.
    """
```

### `build_llm_context()` 내 `[PREMIUM_BLEED]` 섹션

```python
# [PARITY_ANALYSIS] 출력 직후 삽입
bleed_desc = _describe_premium_bleed(opt_snap)
if bleed_desc:
    lines.append("[PREMIUM_BLEED]")
    lines.append(bleed_desc)
```

생성되는 LLM 컨텍스트 예시 (만기 1일 전, 선물 상승 + 스트래들 수축):
```
[OPTIONS_SNAPSHOT]
{ "premium_bleed_score": -0.74, "straddle_now": 3.85, ... }

[PREMIUM_BLEED]
프리미엄 블리드 감지: 선물 상승 중 스트래들 수축(프리미엄 블리드) (score=-0.74, dte_weight=0.100)
  선물 수익률: +0.3100%
  스트래들 변화: -8.77% (4.22 → 3.85) [decay_vs_fut=-0.0310]
  IV Crush ⬇️: -8.30% (BS ATM IV 근사)
  → 선물 상승에도 옵션 프리미엄이 수축 중: Theta 가속/롤오버/IV Crush 가능.
     방향성 신규 진입보다 관망 권장.
```

---

## 가드레일 (`pipeline.py`)

### `_apply_bleed_guardrail()` 신설

parity_guardrail **이후** 마지막으로 적용된다. option_feature_set == "v4" 일 때만 동작한다.

### 가드레일 임계값

| 조건 | dte_weight_norm | 만기 잔존일 | 동작 |
|---|---|---|---|
| `score <= -0.75` + 선물 상승 중(`fut_ret > 0`) | `>= 1.0` | 만기 당일 (dte=0) | BUY/SELL → **HOLD** + LOW |
| `\|score\| >= 0.5` | `>= 0.033` | dte ≤ 3일 | MEDIUM → **LOW** 강등 |
| — | `< 0.1` | dte > 7일 | 가드레일 비활성 |

HOLD 전환 조건에 `fut_ret > 0` (선물 상승 중) 조건을 추가한 이유: 선물이 하락하면서 프리미엄이 수축하는 것은 정상이므로 이 경우에는 HOLD 강제 전환을 하지 않는다.

### 가드레일 적용 순서 (v4 기준)

```
1. _apply_option_guardrail()   (PCR/IV 기반)
2. _apply_basis_guardrail()    (선/현물 베이시스 기반)
3. _apply_parity_guardrail()   (v3/v4: C-P 패리티 이탈)
4. _apply_bleed_guardrail()    (v4 전용: 프리미엄 블리드)  ← 신규
```

---

## 텔레그램 알림 시스템 (`telegram_notifier.py`)

### 알림 종류 두 가지

**① 예측 메시지 내 `💧 블리드 가드레일` 블록 (기존 흐름 연동)**

가드레일이 신호를 변경한 경우에만 예측 메시지 안에 포함된다.

```
💧 *블리드 가드레일*: 🟢`BUY` → 🟡`HOLD`  신뢰도 HIGH → LOW
   `premium_bleed_critical(score=-0.76,dte_w=1.00,fut_ret=+0.0031)`
```

**② `send_premium_bleed_alert()` 독립 알림 (실시간 감시)**

예측 주기(60초)와 무관하게, 5초 주기 블리드 모니터 스레드가 신호를 감지하면 즉시 전송한다.

독립 알림 예시:
```
🔥 프리미엄 블리드 알림 | 14:23:07

💰 선물가: 357.50  |  만기: 1일 전
📊 선물 방향: 상승 📈  |  스트래들: 수축 💧
🎯 블리드 스코어: -0.74  (강한 수축)

📉 스트래들: 4.22 → 3.85 (-8.77%)
📈 선물 수익률: +0.3100%
💧 Decay vs Fut: -0.0310
🌊 IV Crush ⬇️: -8.30%

💡 해석: 선물 상승 중 옵션 프리미엄 비정상 수축.
   Theta 급가속 / IV Crush / MM 롤오버 가능.
   → 방향성 신규 진입 자제, 기존 포지션 청산 국면 가능.
```

### `send_premium_bleed_alert()` 필터링 조건

```python
def send_premium_bleed_alert(
    opt_snap, current_price,
    *,
    dte_days=None,
    min_score=0.3,       # |premium_bleed_score| 최소값
    cooldown_sec=None,   # None이면 인스턴스 기본값 300초 사용
    force=False,
) -> bool:
```

| 조건 | 동작 |
|---|---|
| `dte_w < 0.1` (만기 7일 초과) | 전송 안 함 |
| `straddle_prev == 0` (prev 없음) | 전송 안 함 |
| `\|score\| < min_score` | 전송 안 함 |
| 쿨다운 미경과 (`elapsed < cooldown_sec`) | 전송 안 함 |
| `force=True` | 위 조건 모두 무시하고 강제 전송 |

### `_bleed_monitor_loop()` — 독립 감시 스레드

`PipelineTelegramBridge.start()`에서 v4 설정 시 자동으로 시작된다.

```
[BleedMonitor 스레드]
  매 5초: pipeline._build_option_snapshot_safe(update_prev=False) 호출
           → opt_snap 조회 (상태 갱신 없음)
           → send_premium_bleed_alert() 호출
              → 쿨다운/점수 필터 통과 시 텔레그램 전송
```

`update_prev=False`로 호출하는 이유: 블리드 모니터가 상태를 갱신하면 OB 버퍼 경로(1Hz)의 `_prev_*` 값이 덮어써져 다음 예측 틱의 diff 계산이 오염된다.

---

## 활성화 방법

`config.json` 또는 파이프라인 생성자에서 feature set을 v4로 변경한다:

```python
pipeline = PredictionPipeline(
    ...,
    option_feature_set="v4",   # "v3" → "v4" 로 변경
)
```

`bridge.start()` 호출 시 v4가 감지되면 `BleedMonitor` 스레드가 자동으로 함께 시작된다. 별도 설정은 필요하지 않다.

쿨다운 또는 최소 점수를 조정하려면:

```python
# TelegramNotifier 인스턴스에서 직접 조정
notifier._bleed_alert_cooldown_sec = 180.0   # 3분으로 단축
bridge._bleed_min_score = 0.5                # 더 강한 신호만 전송
```

---

## 구현 완료 순서

| 단계 | 파일 | 작업 | 상태 |
|---|---|---|---|
| 1 | `option_features.py` | `calc_premium_bleed()` 신규 | ✅ |
| 2 | `option_features.py` | `build_option_snapshot()` v4 분기 추가 | ✅ |
| 3 | `features.py` | `OPT_KEYS_V4` 정의 + `get_opt_keys()` v4 분기 | ✅ |
| 4 | `context_builder.py` | `_describe_premium_bleed()` 신규 | ✅ |
| 5 | `context_builder.py` | `[PREMIUM_BLEED]` 섹션 `build_llm_context()` 삽입 | ✅ |
| 6 | `pipeline.py` | v4 유효성 검증에 추가 | ✅ |
| 7 | `pipeline.py` | `_prev_*` 상태 캐싱 v4 포함 | ✅ |
| 8 | `pipeline.py` | `_apply_bleed_guardrail()` 신규 + 연동 | ✅ |
| 9 | `telegram_notifier.py` | `format_premium_bleed_alert()` 신규 | ✅ |
| 10 | `telegram_notifier.py` | `send_premium_bleed_alert()` 신규 (쿨다운 포함) | ✅ |
| 11 | `telegram_notifier.py` | `💧 블리드 가드레일` 블록 `format_prediction_message()` 삽입 | ✅ |
| 12 | `telegram_notifier.py` | `_bleed_monitor_loop()` 신규 + `start()` 자동 시작 | ✅ |

---

## 핵심 인사이트

`premium_bleed_score`가 포착하는 것은 단순히 "옵션이 쌌다"가 아니라 **옵션 시장 참여자들이 방향성 배팅을 포기하는 신호**다.

- **score < -0.5, 선물 상승 중**: 콜 매도 포지션 정리 + IV Crush 동시 발생. 선물 상승이 가짜 브레이크아웃일 가능성이 높다.
- **score > +0.5**: 프리미엄 급팽창. 대형 이벤트 직전 또는 감마 스퀴즈 시작. 방향성보다 변동성 전략이 유효한 구간.
- **score ≈ 0**: 옵션 시장이 선물 움직임을 정상적으로 추종 중. 방향성 신호 신뢰 가능.

`[PREMIUM_BLEED]` 섹션을 통해 LLM은 "선물이 상승하고 있지만 옵션 시장 참여자들은 이미 만기 포지션 청산 모드로 전환했다"는 맥락을 자연어로 받아 판단에 반영한다.

**dte_weight_norm < 0.1 (만기 7일 이상)이면 이 신호 전체를 무시**하는 것이 올바르다. 프리미엄 블리드는 만기주 고유 현상이며, 일반 거래일에는 정상적인 theta decay로 해석될 뿐이다.
