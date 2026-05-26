# 하이브리드 피봇탐색 알고리즘 평가 프레임워크

> **대상 파일**: `indicators/hybrid_adaptive_pivot.py`, `prediction/pivot_parameter_db.py`,  
> `scripts/run_daily_backtest.py`  
> **최종 수정**: 2026-05-26

---

## 1. 배경 및 문제 정의

`HybridAdaptivePivot`은 매일 장마감 후 백테스트 결과를 SQLite DB(`pivot_parameters.db`)에 누적하고, 다음 날 `ParameterRecommender.recommend()`가 레짐별 최적 파라미터를 DB에서 조회하여 추천하는 구조다.

```
[당일 매매] → [백테스트] → [DB 저장] → [익일 파라미터 추천] → [당일 매매] → ...
```

이 피드백 루프가 **실제로 성능을 개선하는지** 확인하는 외부 검증 루프가 현재 코드에 없다.

---

## 2. 현재 composite_score 구조

`_calc_composite_score()`는 5개 지표의 가중합(0~1)으로 계산된다.

| 지표 | 가중치 | 설명 |
|------|--------|------|
| `confirmation_rate` | 35% | 피봇 후보 → 확정 전환율 |
| `lag_score` | 25% | 확정 래그 (20봉 기준 역수 정규화) |
| `pivot_quality` | 20% | 파동 품질 점수 |
| `alternation_rate` | 10% | 고점/저점 교대율 |
| `fp_score` | 10% | 오탐 억제 점수 |

이 점수가 높은 날의 파라미터를 DB에서 조회해 추천하지만, 그 추천이 **미래 날에도 유효한지** 검증되지 않았다.

---

## 3. 구조적 위험 3가지

### 3-1. 순환 강화 (Circular Reinforcement)

시장이 좋았던 날의 파라미터가 DB에 쌓이고 → 그 파라미터가 추천되고 → 비슷한 날에는 잘 맞지만 → 장세가 바뀌면 낙후된 파라미터가 반복 추천될 수 있다.

**탐지 방법**: Walk-forward 검증에서 `score_db < score_fallback`이 연속으로 나타나면 순환 강화 신호.

### 3-2. 레짐 분류 의존성

`AdaptiveParamEngine`이 레짐을 오분류하면 잘못된 DB 버킷에서 파라미터를 조회한다.  
레짐 정확도를 별도로 측정해야 한다.

```
실제 레짐: trend_strong_up
분류 결과: chop_low_vol → 잘못된 atr_weight=0.35 적용
```

### 3-3. 버킷팅 정밀도 손실

`save_session_parameters()`에서 연속형 파라미터를 반올림 저장한다.

```python
atr_weight_bin = round(config.get("atr_weight", 0), 1)  # 0.47 → 0.5
base_pct_bin   = round(config.get("base_pct", 0), 2)    # 0.283 → 0.28
```

이 반올림 손실이 누적되면 DB 조회 시 실제 최적값과 괴리가 생긴다.

---

## 4. 평가 방법론

### 4-1. Layer 1 — Walk-forward 검증 (핵심)

시간 누수(look-ahead bias) 없이 DB 추천의 실질 효과를 측정한다.

```
훈련 윈도우: D-30 ~ D-1  →  파라미터 추천
테스트 윈도우: D일         →  실제 composite_score 측정
                           →  REGIME_FALLBACK 대비 비교
윈도우 슬라이드: 하루씩 전진
N회 반복 후 평균 개선률 계산
```

**판정 기준**:
- 평균 개선률 `> +0.03` → DB 추천 유효
- 평균 개선률 `< 0` → DB 추천이 오히려 해로움 (순환 강화 의심)

### 4-2. Layer 2 — 베이스라인 비교

3가지 파라미터 소스를 동일 날짜에 동시 평가한다.

| 소스 | 설명 | 코드 위치 |
|------|------|-----------|
| `REGIME_FALLBACK` | 하드코딩 폴백 | `ParameterRecommender.REGIME_FALLBACK` |
| 고정 파라미터 | `base_pct=0.3, atr_weight=0.5` 고정 | 없음 (새로 추가) |
| DB 추천 | `ParameterRecommender.recommend()` | 기존 코드 |

DB 추천이 세 가지 중 가장 높아야 DB가 의미 있다.

### 4-3. Layer 3 — 핵심 실용 지표

| 지표 | 목표 임계값 | 설명 |
|------|-------------|------|
| 방향 정확도 | `≥ 60%` | 피봇 후 N봉 이내 가격이 예측 방향으로 이동한 비율 |
| 레짐 일치율 | `≥ 70%` | 추천 레짐 vs 사후 판단 실제 레짐 일치 비율 |
| 샘플 커버리지 | `≥ 80%` | DB 조회 시 `min_sample` 조건 충족 비율 (미달 시 폴백) |
| 파라미터 안정성 | `CV < 0.2` | 연속 날의 추천 파라미터 변동 계수 |

---

## 5. 수정 내용 요약

### 5-1. `prediction/pivot_parameter_db.py`

- `WalkForwardEvaluator` 클래스 추가
  - `run(lookback_days, test_days)`: 슬라이딩 윈도우 검증 실행
  - `_run_single_day_comparison()`: DB vs FALLBACK vs 고정값 3방향 비교
  - `_calc_stability_cv()`: 파라미터 변동 계수 계산

### 5-2. `scripts/run_daily_backtest.py`

- `run_walk_forward_evaluation()` 함수 추가
- `--evaluate` CLI 옵션 추가: 장마감 후 자동으로 Walk-forward 검증 실행
- 검증 결과를 `logs/evaluation/` 디렉토리에 JSON + 요약 텍스트로 저장

### 5-3. 연속형 파라미터 저장 개선 (버킷팅 병행)

`pivot_parameters_session` 테이블에 `_raw` 컬럼을 추가해 버킷팅 이전 원본값을 함께 저장하는 방식 적용.  
단, 기존 테이블 스키마 호환성을 위해 JSON 컬럼으로 저장한다.

---

## 6. 최소 실행 계획

30일 데이터 확보 시점부터 즉시 측정 가능하다.

```bash
# 장마감 후 백테스트 + Walk-forward 평가 동시 실행
python scripts/run_daily_backtest.py --with-ohlcv --evaluate

# 결과 확인
cat logs/evaluation/walkforward_latest.json
```

**성공 판정 체크리스트**:
- [ ] 평균 `improvement_vs_fallback > 0.03`
- [ ] 레짐별 샘플 커버리지 `≥ 80%`
- [ ] 파라미터 안정성 `CV < 0.2`
- [ ] 방향 정확도 `≥ 60%`

모든 항목 충족 시 → DB 추천 방식 신뢰 가능  
2개 이상 미충족 시 → `min_sample` 임계값 상향 또는 `lookback_days` 축소 검토

---

## 7. 코드 파일 목록

| 파일 | 변경 유형 | 주요 내용 |
|------|-----------|-----------|
| `prediction/pivot_parameter_db.py` | 기능 추가 | `WalkForwardEvaluator` 클래스, 안정성 지표 |
| `scripts/run_daily_backtest.py` | 기능 추가 | `--evaluate` 옵션, 평가 결과 저장 |
