# ZigZag 피봇 확정 지연 분석

**대상**: `AdaptiveZigZag` — `kospi_indicators/kospi_indicators/adaptive_zigzag.py`  
**기준 타임프레임**: 1분봉 (KP200 선물)  
**분석 기준 파라미터**: `confirmation_bars=2` (dataclass 기본) / `1` (운영값), `min_wave_bars=5`, `freeze_on_confirm=True`

---

## 1. 핵심 공식

```
총 lag = 파동 추적 봉수 + confirmation_bars
```

| 구성 요소 | 내용 | 조절 가능 여부 |
|---|---|---|
| 파동 추적 봉수 | 극값(피봇봉) ~ 반전 임계 돌파봉(후보등록봉) 사이 경과 봉수 | 불가 — ZigZag 알고리즘 본질적 후행성 |
| `confirmation_bars` | 후보 등록 후 확정까지 대기 봉수 | 가능 (0~N, 현재 운영값=1) |
| `min_wave_bars` 제약 | 직전 확정 후 최소 5봉 미만이면 후보 등록 차단 | 가능 (0~N, 현재=5) |

---

## 2. 단계별 흐름

```
[피봇봉]     [파동 추적 중 ...]     [후보등록봉 T]     [대기]     [확정봉]
  T-N  →  T-N+1 ... T-2 ... T-1  →       T          →  T+1  →    T+2
  (극값 발생)   (임계 미달)              (rem=2 세팅)    (rem=1)   (rem=0)
```

### 세부 단계

**단계 1 — 극값 형성 (피봇봉)**  
`direction=1` (상승 파동) 구간에서 `_pending_high`가 갱신되며 최고점을 추적. 이 봉이 실제 피봇 위치(`swing_idx`)가 된다.

**단계 2 — 반전 임계 돌파 (후보등록봉 T)**  
조건 충족 시 `pending_confirm` 생성:
- `pending_high − current_low ≥ thr_abs` (ATR 기반 임계값 돌파)
- `_is_wave_length_ok()` → 직전 확정 후 `min_wave_bars(=5)봉` 이상 경과

등록 시 `remaining = confirmation_bars` 로 초기화.

**단계 3 — 확정 카운트다운 (매 봉 `rem -= 1`)**

| 봉 | remaining | 상태 |
|---|---|---|
| T (등록봉) | `confirmation_bars` | 후보 등록 |
| T+1 | `confirmation_bars - 1` | 대기 |
| T+2 | 0 | **확정** (`rem ≤ 0`) |

`confirmation_bars=1` 운영값 기준:

| 봉 | remaining | 상태 |
|---|---|---|
| T (등록봉) | 1 | 후보 등록 |
| T+1 | 0 | **확정** |

**단계 4 — lag 산출**  
`lag_bars = confirmed_bar_idx − swing_idx`

- **케이스 A** — 등록봉 = 피봇봉 (`swing_idx == T`): `lag = 0 + confirmation_bars`
- **케이스 B** — 피봇봉이 N봉 앞 (`swing_idx = T - N`): `lag = N + confirmation_bars`

실운용에서는 케이스 B가 대부분. N = 상승(하락) 파동 지속 봉수.

---

## 3. 파동 유형별 예상 지연 (1분봉, 운영값 `confirmation_bars=1` 기준)

| 파동 유형 | 파동 추적 봉수 | + confirmation | 총 지연(분) | 설명 |
|---|---|---|---|---|
| 급등락 임펄스 | 1~3봉 | +1봉 | **2~4분** | 1~2분 내 빠른 반전. 임계 돌파가 즉시 발생 |
| 통상 단기 파동 | 3~8봉 | +1봉 | **4~9분** | KP200 일중 일반적 스윙 |
| 중기 파동 | 8~20봉 | +1봉 | **9~21분** | 장중 주요 고점/저점 |
| 장기 파동 | 20~60봉 | +1봉 | **21~61분** | 반일 추세 내 주요 피봇 |
| min_wave_bars 제약 발동 | 0~4봉 대기 후 등록 | +1봉 | **1~5분** | 직전 확정 직후 빠른 반전. 강제 대기 |

> `confirmation_bars=2` (dataclass 기본) 기준으로는 위 수치에 각각 +1분 추가.

---

## 4. confirmation_bars 설정값별 비교

| 설정값 | 후보→확정 | 총 최소 lag | 특성 |
|---|---|---|---|
| `0` | 즉시 | 파동 추적봉수만 | 임계 돌파 즉시 확정. whipsaw 매우 취약 |
| `1` | 1봉 후 | 파동 추적 + 1분 | SkyEbest·PatchTST 운영값. 속도·안정성 균형 |
| `2` | 2봉 후 | 파동 추적 + 2분 | 라이브러리 dataclass 기본값. false positive 감소 |

### azz_pending_urgency 계산 (PatchTST ML 피처)

```
azz_pending_urgency = 1.0 - rem / confirmation_bars
```

`confirmation_bars=1` 기준:

| 경과 봉수 | rem | urgency | 상태 |
|---|---|---|---|
| 0봉 (등록 직후) | 1 | 0.000 | 방금 등록 |
| 1봉 | 0 | 1.000 | **확정** |

`confirmation_bars=2` 기준:

| 경과 봉수 | rem | urgency | 상태 |
|---|---|---|---|
| 0봉 (등록 직후) | 2 | 0.000 | 방금 등록 |
| 1봉 | 1 | 0.500 | 대기 중 |
| 2봉 | 0 | 1.000 | **확정** |

---

## 5. min_wave_bars 제약이 지연에 미치는 영향

`min_wave_bars=5`: 직전 피봇 확정 후 5봉 미만이면 새 후보 등록 자체를 차단.

```
직전 확정봉 이후 경과:  0봉  1봉  2봉  3봉  4봉  [5봉~: 등록 가능]
                       ← 차단 구간 (최대 4봉 대기) →
```

- **영향**: 빠른 연속 반전 시 최대 4봉 추가 대기 후 등록 → 총 lag 최대 `4 + confirmation_bars`
- **목적**: 연속 피봇 생성 방지, 노이즈성 스윙 필터링
- **최악 케이스**: 직전 확정 1봉 후 즉시 반전 → 4봉 대기 + 1봉 confirmation = **5봉(분) 지연**

---

## 6. freeze_on_confirm 옵션 영향

현재 `freeze_on_confirm=True` (양쪽 프로젝트 운영값).

| 옵션 | 동작 | 지연 영향 |
|---|---|---|
| `True` (현재) | 후보 등록 시점 가격 고정. 대기 중 신고점/저점 갱신 차단 | 지연 일정. 피봇 위치가 매 봉 흔들리지 않음 |
| `False` | 대기 중 신고점/저점으로 후보 가격 갱신. 갱신 발생 시 `remaining = max(1, cb//2+1)` 로 부분 리셋 | 추가 지연 가능. `cb=2`이면 리셋 시 최대 +2봉 반복 |

`freeze_on_confirm=False` + `confirmation_bars=2` 의 최악 케이스:  
매 봉 신고점 갱신 시 `remaining → 2` 리셋 반복 → 이론적으로 무한 지연 가능 (실제로는 시장이 추세 전환하면 종료).

---

## 7. 지연 성격 요약

| 성격 | 구성 요소 | 조절 여부 | 비고 |
|---|---|---|---|
| **제거 불가** — ZigZag 후행성 | 파동 추적 봉수 | 불가 | 알고리즘 본질. ATR 임계를 낮추면 노이즈 증가로 트레이드오프 |
| **조절 가능** — 확정 대기 | `confirmation_bars` | 가능 (0~N) | 낮출수록 빠르지만 whipsaw 위험 증가 |
| **조절 가능** — 등록 차단 | `min_wave_bars` | 가능 (0~N) | 낮출수록 연속 피봇 허용. 현재=5봉 |
| **PatchTST 보완책** | `azz_pending_*` 피처 | — | 확정 전 후보 상태를 ML에 미리 노출. 지연 제거 대신 조기 인식으로 보완 |

---

## 8. PatchTST pending 피처를 통한 후행성 보완 구조

ZigZag의 후행성을 없앨 수 없으므로, PatchTST는 **확정 전 후보 상태를 ML 피처로 노출**하여 모델이 확정 전에 반전 가능성을 인식하도록 설계.

```
피봇봉(T-N) → 후보등록(T) → 대기(T+1) → 확정(T+2)
                  ↑
          azz_pending_type    = +1(HIGH후보) / -1(LOW후보) / 0(없음)
          azz_pending_dist    = 후보가격과 현재가 거리 비율 (±5% 클리핑)
          azz_pending_urgency = 1 - rem/confirmation_bars  (0=방금등록, 1=다음봉확정)
          azz_pending_age     = exp(-waited/5)             (0봉=1.0, 5봉≈0.37, 10봉≈0.14)
```

ML 모델은 `urgency=0.5` (cb=2 기준 1봉 대기 중)일 때 이미 반전 신호를 선반영할 수 있음.  
SkyEbest는 해당 피처를 소비하는 ML 파이프라인이 없으므로 미구현.

---

## 9. 현재 설정 기준 요약

```
confirmation_bars = 1  (운영값, 양쪽 프로젝트 공통)
min_wave_bars     = 5
freeze_on_confirm = True
max_wait_bars     = 0  (무제한, 필요 시 설정 가능)

최소 지연:  1봉  (파동 추적=0봉, min_wave_bars 통과, 즉시 등록)
통상 지연:  4~9분  (단기 파동 3~8봉 + confirmation 1봉)
최대 지연:  (min_wave_bars-1) + confirmation_bars = 4+1 = 5봉
           ※ 장기 파동은 파동 추적 시간이 지배적 (수십~수백 분)
```
