# 피봇반전 로직 검증 메모 — Look-ahead 없음 & OOS 결론

**작성일**: 2026-06-21
**대상**: KOSPI200 연결선물(`futures_1min` → 5분봉 리샘플), 주간세션 08:45–15:45, 당일청산
**검증 코드**: `indicators/hybrid_adaptive_pivot.py`(검출기), `pivot_optuna_v2.py`(검출·백테스트), `pivot_walkforward_v3.py`(신규: nested walk-forward)
**데이터**: 2025-06-25 ~ 2026-06-19, 5분봉

---

## 0. 한 줄 결론

피봇반전 파이프라인은 **검출부터 체결까지 미래 정보가 들어가는 지점이 없다(look-ahead 없음)**. 그리고 선택구간(opt)과 분리된 **홀드아웃 테스트에서 최적 파라미터조차 베타(장중 상시 롱)를 못 이기고 손실을 낸다**. 즉 "엣지 없음"은 코드 버그가 아니라 진짜 신호다.

---

## 1. Look-ahead 없음 — 코드 레벨 근거

### 1.1 검출기(`HybridAdaptivePivot.update`)는 인과적

- 방향 +1(고점 탐색) 상태에서 러닝 맥스 `pending_high` 를 추적하다가, **극점에서 `thr` 만큼 되돌려진 것을 본 뒤에야**(`reversal = pending_high - low >= thr`) 후보를 등록한다. 후보 등록 시점이 이미 극점보다 늦다.
- 확정 신호(`new_pivot_signal = "new_high"/"new_low"`)는 `_process_pending` 에서 `remaining` 이 0이 되는 봉에서 발생하며, 그 봉 인덱스가 곧 `confirm_pos` 다. 확정 시점까지 사용된 데이터는 그 봉 이하뿐이다.
- `_calc_er` 은 `closes[-(er_period+1):-1]` 로 **현재 봉을 제외한** 과거 종가만 쓴다. `_wave_ok` 도 현재·과거 high/low 만 본다.

### 1.2 `confirmation_bars=0` 도 안전(소급 확정 없음)

후보 등록은 `_run_logic` 끝(Step C/D)에서 일어나고, 그 봉에서는 확정 신호를 반환하지 않는다. 확정은 **빨라야 다음 봉**의 `_process_pending`(`rem==0` 즉시확정)에서 나온다. 따라서 *극점 봉 → 신호 봉* 사이에 최소 1봉 지연이 보장되며, 극점 봉으로 거슬러 확정되는 경로가 없다. 단위 테스트 `test_confirmation_bars_zero_allows_immediate` 가 이 동작을 검증한다.

### 1.3 진입·청산(`pivot_optuna_v2.backtest`)도 인과적

- 진입은 `epos = confirm_pos + 1` 의 **시가**(`next_open`). 신호 확정 봉의 *다음* 봉에서 들어간다.
- 반전 청산이 다음 진입과 같은 봉·같은 시가에서 맞물린다(포지션 연속).
- 당일청산: 다음 피봇이 다음 거래일이면 당일 마지막 봉 종가로 강제청산, 마지막 봉 진입은 스킵.
- 손절/익절 인트라바 스캔은 동일봉에서 손절을 익절보다 먼저 가정(보수적). 비용모델(편도 수수료+2틱 슬리피지×250,000승수)도 진입·청산가 기준으로 일관.

### 1.4 검출기 테스트

`tests/test_hybrid_adaptive_pivot.py` 18개 중 17개 통과. 실패 1개(`test_cancel_ratio_default`)는 로직이 아니라 **오래된 기대값**(테스트는 0.3 단언, 실제 기본값 0.1) → 테스트를 0.1로 고치거나 config 기본값을 복원할 것.

> **결론**: in-sample 의 양수 Sharpe(viability 문서의 1.9~3.1)는 look-ahead 가 아니라 **선택편향/과적합**에서 나온 것이고, 진짜 OOS 가 음수인 것은 정상이다.

---

## 2. 방법론 교체 — `make_objective` → nested walk-forward

### 2.1 기존 `make_objective` 의 문제

`purged_walkforward_folds` 로 구간을 나누지만, 목적함수가 **모든 fold 에서 backtest 를 돌려 평균을 내고 Optuna 가 그 평균을 최대화**한다. 파라미터를 평가하는 그 데이터로 선택하므로 전부 in-sample 이다. purge/embargo 는 fold 경계 상태 누수만 막을 뿐 선택편향을 막지 못한다. 이름과 달리 walk-forward(예측 검증)가 아니다.

### 2.2 신규 `pivot_walkforward_v3.nested_walkforward_optimize`

1. **최종 홀드아웃 분리**: 거래일을 시간순으로 잘라 마지막 `test_frac`(기본 20%)을 Optuna 가 한 번도 보지 않는 테스트 셋으로 떼어둔다(사이 embargo 거래일).
2. **전방 평가 블록**: opt 구간을 비중첩 전방 윈도우(기본 21거래일)로 나눠 각 블록에서만 metric 을 재고 `mean − λ·std` 로 구간 일관성을 본다. (검출은 거래일별 리셋 + 지표는 전체 1회 계산 후 슬라이스 → 블록 간 누수 없음)
3. **편향 없는 최종 수치**: study 종료 후 best 파라미터를 홀드아웃에서 **딱 한 번** 평가. 베타/롱-또는-플랫과 같은 홀드아웃에서 비교하고 Sharpe 표준오차도 같이 낸다.

핵심 차이: **선택 기준(opt 구간 일관성)** 과 **OOS 추정치(홀드아웃)** 를 물리적으로 분리한다.

---

## 3. 실측 결과 (n_trials=60, val_block=21일, test_frac=0.2, seed=42)

- 구성: OPT **190거래일** / TEST(홀드아웃) **48거래일** / 전방블록 **9개** / embargo 2일
- 선택구간 내 일관성 점수(in-sample, OOS 아님): **−1.226** — 정직한 전방블록 기준으로는 일관되게 양수인 파라미터 자체가 없었다.

### 3.1 홀드아웃(편향 없는 OOS) 비교

| 전략 | 거래(일) | 승률 | PnL(원) | Sharpe | Sharpe SE | MaxDD(원) |
|------|---:|---:|---:|---:|---:|---:|
| **피봇반전 (best params)** | 56 | 50.00% | **−8,018,211** | **−0.515** | ±2.29 | −44,330,271 |
| 베타(장중 상시 롱) | 48 | 52.08% | +27,053,913 | 1.254 | ±2.30 | −23,349,005 |
| 롱-또는-플랫(MA20/60+ADX) | 34 | 50.00% | +20,989,323 | 1.170 | ±2.73 | −23,349,005 |

- **피봇 < 베타 2SE**: `False` — 피봇이 베타를 유의하게 이기지 못하는 정도가 아니라, **음수**다.
- 베타가 피봇을 27M+ 앞서고, MaxDD 도 피봇이 2배 가까이 깊다.

### 3.2 통계적 주의

홀드아웃이 48거래일뿐이라 Sharpe SE 가 ±2.3 수준으로 매우 넓다 → 단일 수치로는 **베타의 +1.254 조차 0과 구분되지 않는다**. 따라서 결론은 어느 한 숫자가 아니라 **세 가지가 같은 방향을 가리킨다는 일관성**에서 나온다:
1. opt 구간 전방블록 일관성 점수 음수(−1.23),
2. 홀드아웃 OOS 음수(−0.515, PnL −8M),
3. 이전 `pivot_short_bear_walkforward` 의 OOS 음수(−1.667).

---

## 4. 권고

1. **운영 로직 교체 보류**: 현재 데이터에서 피봇반전을 롱-또는-플랫/베타의 대체로 채택하지 않는다(본인 viability 결론과 일치, 이제는 편향 없는 근거로 뒷받침).
2. **선택 기준은 항상 nested 프로토콜**: 단일 in-sample Sharpe 폐기. `pivot_walkforward_v3` 처럼 opt↔홀드아웃 분리 + 베타 대비 SE 비교를 디폴트로.
3. **데이터 확장 후 재검증**: 2년+ 데이터에서 홀드아웃이 길어져 SE 가 좁아진 뒤에야 "유의하게 베타를 이기는가"가 판별 가능. 그 전엔 연구 브랜치로만 보관.
4. **테스트 정리**: `test_cancel_ratio_default` 기대값(0.3↔0.1) 일치시키기.

---

## 5. 재현 방법

```bash
# 검출기 모듈(indicators/)이 import 경로에 있어야 함
PYTHONPATH=/path/to/SkyPredictor python pivot_walkforward_v3.py
# → data/backtest_results/walkforward_v3_result.json 생성
```

`indicators.hybrid_adaptive_pivot` 의 `[HAP][확정]` 경고 로그는 양이 많으므로
`logging.getLogger("indicators.hybrid_adaptive_pivot").setLevel(logging.ERROR)` 로 끈다(데모 `main()` 에 반영됨).
