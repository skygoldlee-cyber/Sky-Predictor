# Transformer + TFT + LLM 예측 파이프라인 상세 동작 알고리즘

> **SkyEbest 프로젝트 — KP200 선물 실시간 예측 시스템**  
> 대상 코드: `prediction/pipeline.py`, `prediction/predictor.py`, `prediction/model.py`, `prediction/tft_model.py`, `adaptive_indicator/`

---

## 1. 시스템 전체 구조 개요

```mermaid
graph TB
    subgraph INPUT["데이터 입력 레이어"]
        FC0["FC0 선물 틱<br/>(체결/호가)"]
        OC0["OC0 옵션 틱<br/>(체결/호가)"]
        FO0["FO0 호가창<br/>(OrderBook)"]
    end

    subgraph PIPELINE["PredictionPipeline"]
        TP["RealTimeTickProcessor<br/>분봉 생성 / 옵션 집계"]
        OBBuf["OB 버퍼<br/>deque(maxlen=seq_len)<br/>1Hz 다운샘플링"]
        ADAPT["AdaptiveIndicatorManager<br/>SuperTrend + ZigZag"]
        FEAT["Feature Engineering<br/>OB(7) + CD(5) + OPT(7/16) + ADAPT(28)"]
        NUM["NumericPredictor<br/>Transformer / TFT / Ensemble"]
        GUARD["가드레일<br/>옵션 유동성 + 베이시스"]
        LLM["LLMJudge<br/>Claude / GPT / Gemini"]
        OUT["최종 예측 출력<br/>signal / confidence / consensus"]
    end

    FC0 & OC0 & FO0 --> TP
    FO0 --> OBBuf
    TP --> ADAPT
    OBBuf & ADAPT & TP --> FEAT
    FEAT --> NUM
    NUM --> GUARD
    GUARD --> LLM
    LLM --> OUT
```

---

## 2. 데이터 수집 및 전처리

### 2-1. 틱 수신 흐름 (`add_realtime_tick`)

```mermaid
flowchart TD
    A[틱 데이터 수신] --> B{trcode 분류}
    B -->|FC0 선물체결| C[tick_processor.process_tick]
    B -->|OC0 옵션체결| C
    B -->|FO0 호가창| D[tick_norm 언패킹<br/>offerhos/bidhos 배열 → offerho1~5]
    D --> E{호가 유효성 검사<br/>bid/ask > 0?}
    E -->|Invalid| F[버퍼링 스킵]
    E -->|Valid| G[가격 단위 복원<br/>int → float /100]
    G --> H[1Hz 다운샘플링<br/>sec_key = timestamp.second]
    H --> I{같은 초, 같은 signature?}
    I -->|중복| J[스킵]
    I -->|새 데이터| K[calc_orderbook_features 호출]
    K --> L[ob_records deque에 추가/갱신]

    style F fill:#ffcccc
    style J fill:#ffcccc
```

### 2-2. OrderBook 피처 산출 (`calc_orderbook_features`)

| 피처 | 계산식 | 의미 |
|------|--------|------|
| `obi` | `(총매수잔량 - 총매도잔량) / (총매수 + 총매도 + ε)` | 호가 불균형 [-1, 1] |
| `spread` | `ask1 - bid1` | 최우선 스프레드 |
| `level1_ratio` | `bid1잔량 / (bid1 + ask1잔량 + ε)` | L1 비율 |
| `bid_slope` | `Σ(bidrem_i / bidho_i) / 5` | 매수 호가 기울기 |
| `offer_slope` | `Σ(offerrem_i / offerho_i) / 5` | 매도 호가 기울기 |
| `totbidrem` | 총 매수 잔량 | 시장 수요 |
| `totofferrem` | 총 매도 잔량 | 시장 공급 |

---

## 3. 분봉 피처 엔지니어링 (`calc_candle_features`)

```mermaid
flowchart LR
    A[분봉 OHLCV DataFrame] --> B[수익률 계산]
    B --> C["ret1 = log(close_t / close_t-1)"]
    B --> D["ret3 = log(close_t / close_t-3)"]
    B --> E["slope3 = OLS slope (3분)"]
    A --> F[변동성 계산]
    F --> G["range_pct = (high-low)/close"]
    A --> H[거래량 가속도]
    H --> I["vol_accel = vol_t / mean(vol_t-3:t)"]

    C & D & E & G & I --> J[CD_KEYS 5개<br/>ret1 ret3 slope3 vol_accel range_pct]
```

---

## 4. Adaptive Indicator 시스템

### 4-1. AdaptiveIndicatorManager 흐름

```mermaid
flowchart TD
    A[분봉 확정 바 수신] --> B{Warmed Up?}
    B -->|No - 최초 초기화| C["tail(warmup_bars=45) 로 배치 업데이트<br/>마지막 완성된 바 제외"]
    C --> D[마지막 완성 바로 update 호출]
    D --> E[warmed = True]
    B -->|Yes| F{새 분봉 타임스탬프?}
    F -->|Yes - 신규 바| G[incremental update 1회]
    F -->|No - 동일 바| H[캐시된 피처 재사용]
    F -->|과거 바 감지| I[reset + 재초기화]

    G & D --> J[AdaptiveSuperTrend.update]
    G & D --> K[AdaptiveZigZag.update]
    J & K --> L[ADAPT_KEYS 28개 피처 생성]
    L --> M[LLM 컨텍스트 문자열 생성]
```

### 4-2. Adaptive SuperTrend 알고리즘

```mermaid
flowchart TD
    A["입력: High, Low, Close"] --> B["Efficiency Ratio (ER) 계산<br/>ER = |close변화| / Σ|close변화(er_period)|"]
    B --> C["ADX 계산 (adx_period)"]
    C --> D["동적 ATR 기간 계산<br/>atr_period = lerp(atr_min, atr_max, 1-ER)"]
    D --> E["동적 배율 계산<br/>multiplier = lerp(mult_min, mult_max, 1-ER)"]
    E --> F["Wilder 스무딩으로 ATR 계산"]
    F --> G{BB 보정 사용?}
    G -->|Yes| H["볼린저밴드 폭으로 ATR 보정"]
    G -->|No| I[원본 ATR 사용]
    H & I --> J["SuperTrend 밴드 계산<br/>Upper = (H+L)/2 + mult*ATR<br/>Lower = (H+L)/2 - mult*ATR"]
    J --> K{방향 결정}
    K -->|Close > Upper 이전 밴드| L["direction = +1 (상승)"]
    K -->|Close < Lower 이전 밴드| M["direction = -1 (하락)"]
    L & M --> N["smooth_period 스무딩 적용"]
    N --> O[SuperTrendState 반환]
```

### 4-3. Adaptive ZigZag 알고리즘

```mermaid
flowchart TD
    A["입력: High, Low, Close"] --> B["ER 기반 동적 threshold 계산<br/>thr = lerp(min_thr_pct, max_thr_pct, 1-ER)"]
    B --> C{스윙 방향 감지}
    C -->|High > 이전 저점 * (1+thr)| D["High 스윙 포인트 후보"]
    C -->|Low < 이전 고점 * (1-thr)| E["Low 스윙 포인트 후보"]
    D & E --> F["confirmation_bars 확인<br/>min_wave_bars, min_wave_pct 필터"]
    F --> G["cluster_tolerance_pct 기반<br/>중복 스윙 제거"]
    G --> H["최대 max_swings 개 유지"]
    H --> I["Fib 레벨 계산<br/>fib618, fib382"]
    H --> J["구조 분석<br/>Higher Highs / Lower Lows"]
    I & J --> K[ZigZagState 반환]
```

---

## 5. 시퀀스 피처 빌드 (`build_sequence`)

```mermaid
graph LR
    subgraph SOURCES["입력 소스"]
        OB["ob_records<br/>OB_KEYS × 7"]
        CD["candle_df<br/>CD_KEYS × 5"]
        OPT["opt_snap<br/>OPT_KEYS × 7 or 16"]
        ADPT["adaptive_features<br/>ADAPT_KEYS × 28"]
    end

    subgraph OUTPUT["출력 시퀀스"]
        SEQ["shape: (seq_len, feature_dim)<br/>feature_dim = 7+5+7+28 = 47<br/>(또는 v2: 7+5+16+28 = 56)"]
    end

    SOURCES --> ALIGN["타임스탬프 정렬<br/>ob_records ↔ candle_df 매칭"]
    ALIGN --> CONCAT["axis=1 concat<br/>각 타임스텝별 피처 결합"]
    CONCAT --> PAD["seq_len 패딩<br/>(앞쪽 0 패딩)"]
    PAD --> SEQ
```

---

## 6. 수치 예측기 (Numeric Predictor)

### 6-1. 예측기 선택 구조

```mermaid
flowchart TD
    A["create_numeric_predictor(mode)"] --> B{mode}
    B -->|transformer| C[TransformerPredictor]
    B -->|tft| D[TFTPredictor]
    B -->|ensemble/combined| E[EnsemblePredictor]
    B -->|rule_based| F[RuleBasedPredictor]

    C --> G{weights 파일 존재?}
    G -->|Yes + feature_dim 일치| H[PriceTransformer.load<br/>PyTorch 모델]
    G -->|No / dim 불일치| I[Rule-based Fallback]

    D --> J{weights 파일 존재?}
    J -->|Yes + dim 일치| K[TemporalFusionTransformer.load]
    J -->|No / dim 불일치| L[HOLD 반환]

    E --> C
    E --> D
```

### 6-2. PriceTransformer 아키텍처

```mermaid
graph TB
    IN["입력 x<br/>shape: (batch, seq_len, feature_dim)"]
    PROJ["Linear Projection<br/>feature_dim → d_model(64)"]
    CLS["[CLS] Token<br/>학습 가능 파라미터 (1, 1, 64)"]
    CAT["Concat<br/>(batch, seq_len+1, 64)"]
    PE["Positional Encoding<br/>Sin/Cos + Dropout(0.1)"]
    ENC["TransformerEncoder<br/>n_layers=2, n_heads=4<br/>d_ff=128, norm_first=True"]
    HEAD["Classification Head<br/>LayerNorm → Linear(64→32) → GELU<br/>→ Dropout → Linear(32→1) → Sigmoid"]
    OUT["P(up) ∈ [0,1]"]

    IN --> PROJ
    CLS --> CAT
    PROJ --> CAT
    CAT --> PE
    PE --> ENC
    ENC --> |"CLS token [:, 0, :]"| HEAD
    HEAD --> OUT
```

### 6-3. 신호 분류 로직 (`_classify`)

```mermaid
flowchart TD
    A["prob ∈ [0,1]"] --> B{prob ≥ buy_threshold\n(default: 0.62)?}
    B -->|Yes| C["signal = BUY"]
    B -->|No| D{prob ≤ sell_threshold\n(default: 0.38)?}
    D -->|Yes| E["signal = SELL"]
    D -->|No| F["signal = HOLD"]

    A --> G["margin = |prob - 0.5|"]
    G --> H{margin ≥ 0.15 AND\nspread ≤ conf_spread_max?}
    H -->|Yes| I["confidence = HIGH"]
    H -->|No| J{margin ≥ 0.08?}
    J -->|Yes| K["confidence = MEDIUM"]
    J -->|No| L["confidence = LOW"]
```

### 6-4. EnsemblePredictor 합산 흐름

```mermaid
flowchart TD
    A[ModelInput] --> B[TransformerPredictor.predict]
    A --> C[TFTPredictor.predict]
    B --> D[transformer_prob]
    C --> E{TFT 사용 가능?}
    E -->|Yes| F["tft_prob<br/>ens_prob = w_t * t_prob + w_f * f_prob"]
    E -->|No| G["ens_prob = transformer_prob<br/>method = transformer_only"]
    F --> H{방향 불일치 AND\ndisagreement_hold=True AND\nprob_diff < 0.1?}
    H -->|Yes| I["signal = HOLD\nconfidence = LOW\nmethod = disagreement_hold"]
    H -->|No| J[_classify 적용]
    G --> J
    I --> K[EnsemblePredictionResult]
    J --> K

    D --> H
    F --> H
```

---

## 7. 가드레일 (Guardrail)

### 7-1. 옵션 유동성 가드레일

```mermaid
flowchart TD
    A["opt_snap 검사"] --> B{ATM 옵션 존재\n(call_cnt > 0 or put_cnt > 0)?}
    B -->|No| Z[신호 유지]
    B -->|Yes| C{wide = atm_spread_pct ≥ 1.5%?}
    C --> D{illiq = atm_liq_log ≤ 2.0?}

    C & D --> E{wide AND illiq?}
    E -->|Yes, signal=BUY/SELL| F["→ HOLD, LOW"]
    E -->|No| G{confidence = HIGH?}
    G -->|Yes| H["→ signal, MEDIUM"]
    G -->|No| I{confidence = MEDIUM?}
    I -->|Yes| J["→ signal, LOW"]
    I -->|No| Z
```

### 7-2. 베이시스 가드레일

```mermaid
flowchart TD
    A["IJ 실시간 스냅샷 조회\nbasis = futures_price - spot_index"] --> B{|basis| ≥ 2.5?}
    B -->|Yes, BUY/SELL| C["→ HOLD, LOW"]
    B -->|No| D{|basis| ≥ 1.5?}
    D -->|No| E[신호 유지]
    D -->|Yes| F{confidence = HIGH?}
    F -->|Yes| G["→ signal, MEDIUM"]
    F -->|No| H{confidence = MEDIUM?}
    H -->|Yes| I["→ signal, LOW"]
    H -->|No| E
```

---

## 8. LLM 판단 레이어 (`LLMJudge`)

### 8-1. 단일 LLM 모드

```mermaid
sequenceDiagram
    participant P as PredictionPipeline
    participant E as ThreadPoolExecutor
    participant J as LLMJudge
    participant LLM as LLM Provider

    P->>E: submit(judge, system, user)
    E->>J: judge(system, user, timeout=8.0s)
    J->>LLM: API 호출 (선호 provider → fallback 순)
    LLM-->>J: JSON 응답
    J->>J: 파싱 (action/risk_level/rationale/caution)
    J-->>E: LLMJudgment
    E-->>P: judgment (또는 TimeoutError)

    alt Timeout 발생
        P->>P: executor 리셋
        P->>P: transformer 결과로 Fallback
    end
```

### 8-2. Dual LLM 모드 (GPT + Gemini 병렬)

```mermaid
flowchart TD
    A[LLM 판단 요청] --> B{dual_llm = True?}
    B -->|Yes| C[GPT 비동기 제출]
    B -->|Yes| D[Gemini 비동기 제출]
    C & D --> E["각각 timeout 대기\n(llm_timeout_sec)"]
    E --> F{두 모델 결과}
    F --> G{gpt_action ≠ gemini_action\nAND disagreement_hold?}
    G -->|불일치| H["→ HOLD, LOW\nprovider = dual_disagreement_hold"]
    G -->|일치| I[primary_provider 결과 사용]
    B -->|No| J[단일 LLM 실행]
```

### 8-3. LLM 프롬프트 구조 (`build_llm_context`)

```mermaid
graph TD
    A["snapshot 딕셔너리"] --> B["Transformer/Ensemble 예측값<br/>prob, signal, confidence"]
    A --> C["시장 데이터<br/>current_price, basis, spot_index"]
    A --> D["호가창 스냅샷<br/>obi, spread, level1_ratio"]
    A --> E["옵션 피처<br/>PCR, IV skew, ATM OB imbalance"]
    A --> F["Adaptive 지표<br/>SuperTrend direction, ZigZag swings"]
    A --> G["배경 데이터<br/>T2101(주요 지수), T2301(국채)"]
    B & C & D & E & F & G --> H["컨텍스트 JSON 직렬화"]
    H --> I["System Prompt<br/>역할 정의 + 출력 형식"]
    H --> J["User Prompt<br/>컨텍스트 + 분석 요청"]
    I & J --> K["LLM API 호출"]
    K --> L["JSON 파싱<br/>{action, risk_level, rationale, caution}"]
```

---

## 9. `get_prediction()` 전체 흐름

```mermaid
flowchart TD
    START([get_prediction 호출]) --> A[현재 시각 & 가격 취득]
    A --> B{선물 가격 존재?}
    B -->|No| ERR1[insufficient_data 반환]
    B -->|Yes| C[ATM 옵션 허용 종목 갱신]
    C --> D[분봉 DataFrame 조회]
    D --> E{bars ≥ min_required?}
    E -->|No| ERR2[insufficient_minutes 반환]
    E -->|Yes| F[Adaptive 지표 업데이트<br/>_compute_adaptive_bundle]
    F --> G[옵션 스냅샷 빌드<br/>build_option_snapshot]
    G --> H[수치 예측 실행<br/>_build_and_predict_numeric]
    H --> I{예측 성공?}
    I -->|No| ERR3[numeric_failed 반환]
    I -->|Yes| J[옵션 가드레일 적용]
    J --> K[베이시스 가드레일 적용]
    K --> L[LLM 컨텍스트 스냅샷 구성]
    L --> M[LLM 프롬프트 빌드]
    M --> N[LLM 판단 실행<br/>_run_llm_judgment]
    N --> O{consensus =\nsignal == llm_action?}
    O --> P[결과 딕셔너리 조립]
    P --> END([반환])

    style ERR1 fill:#ffcccc
    style ERR2 fill:#ffcccc
    style ERR3 fill:#ffcccc
    style END fill:#ccffcc
```

---

## 10. 출력 결과 구조

```mermaid
graph LR
    subgraph OUTPUT["get_prediction() 반환값"]
        A["핵심 예측\n• prob (앙상블 확률)\n• signal (BUY/SELL/HOLD)\n• confidence (HIGH/MEDIUM/LOW)"]
        B["모델 세부\n• transformer_prob\n• tft_prob\n• ensemble_method\n• model_agreement"]
        C["LLM 판단\n• llm_action\n• llm_provider\n• risk_level\n• rationale / caution"]
        D["시장 컨텍스트\n• current_price\n• spot_index / basis\n• regime (STRONG_UP 등)"]
        E["진단 정보\n• fo0_age_sec\n• ob_records_len\n• consensus\n• model_outputs (각 모델별)"]
    end
```

---

## 11. 실시간 Regime 분류

```mermaid
flowchart TD
    A[adaptive_supertrend_state] --> B{direction 추출\n+1 / -1 / 0}
    B --> C{trend_strength 추출}
    C -->|strong| D{direction > 0?}
    C -->|weak| E{direction > 0?}
    C -->|neutral/없음| F["RANGE"]
    D -->|Yes| G["STRONG_UP"]
    D -->|No| H["STRONG_DOWN"]
    E -->|Yes| I["WEAK_UP"]
    E -->|No| J["WEAK_DOWN"]
    B -->|0 or None| K["regime = None"]

    style G fill:#ccffcc
    style I fill:#eeffcc
    style H fill:#ffcccc
    style J fill:#ffeecc
```

---

## 12. 주요 설정 파라미터 요약

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `seq_len` | 60 | OB 버퍼 길이 (초, 1Hz) |
| `prediction_minutes` | 5 | 예측 지평선 |
| `min_minute_bars_required` | 20 | 최소 분봉 수 |
| `buy_threshold` | 0.62 | BUY 신호 임계값 |
| `sell_threshold` | 0.38 | SELL 신호 임계값 |
| `confidence_high_margin` | 0.15 | HIGH 신뢰도 마진 |
| `llm_timeout_sec` | 8.0 | LLM 응답 제한 시간 |
| `disagreement_hold` | True | 모델 불일치 시 HOLD |
| `disagreement_hold_prob_diff_max` | 0.1 | 불일치 허용 prob 차이 |
| `transformer_weight` | 0.5 | 앙상블 내 Transformer 비중 |
| `fo0_stale_sec` | 10 | FO0 스테일 경고 임계 |
| `guard_basis_hold_thr` | 2.5 | 베이시스 HOLD 임계 |
| `guard_atm_spread_pct_thr` | 1.5 | ATM 스프레드 임계(%) |
| `adaptive_warmup_bars` | 45 | Adaptive 지표 워밍업 분봉 수 |

---

## 13. 피처 차원 구성 요약

```mermaid
pie title Feature Dimension (기본 v1 + Adaptive)
    "OB_KEYS (호가창)" : 7
    "CD_KEYS (분봉)" : 5
    "OPT_KEYS_V1 (옵션)" : 7
    "ADAPT_KEYS (적응형 지표)" : 28
```

| 그룹 | 키 수 | 내용 |
|------|-------|------|
| OB (OrderBook) | 7 | obi, spread, level1_ratio, bid_slope, offer_slope, totbidrem, totofferrem |
| CD (Candle) | 5 | ret1, ret3, slope3, vol_accel, range_pct |
| OPT v1 | 7 | pcr_volume, iv_skew, max_pain_dist_pct, atm_iv, atm_spread_pct, atm_orderbook_imb, atm_liquidity_log |
| OPT v2 추가 | +9 | optm_call_ret, optm_put_ret, optm_straddle_ret 등 |
| ADAPT | 28 | ast_direction~ast_band_width_pct (9개) + azz_direction~azz_structure_down (19개) + cross (4개) |
| **합계 (v1+Adapt)** | **47** | 기본 운영 차원 |

---

*문서 생성일: 2026-02-28 | SkyEbest Transformer Pipeline v2*
