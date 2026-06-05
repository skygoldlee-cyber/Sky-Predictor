# Adaptive ZigZag 피봇 로직 상세 문서

> 대상 코드: `kospi_indicators/kospi_indicators/adaptive_zigzag.py`, `views/charts/UnifiedTA.py`,  
> `views/charts/technical_analysis.py`, `services/kospi_adaptive_zz_confirm.py`, `services/kospi_zz_rth_slice.py`  
> 최종 수정: Patch_TST (FIX-1~8, P-FIX-A/C 포함)

---

## 1. 전체 구조 개요

```mermaid
flowchart TD
    A[Minute OHLCV\nOriginal df] --> B[kospi_zz_rth_slice.py\nslice_df_ind_for_kospi_zz]
    B -->|df_zz + pos_map| C[UnifiedTA.get_zig_zag\nuse_adaptive=True]
    C -->|AdaptiveZigZagConfig assembly| D[AdaptiveZigZag.update\n× n bars loop]
    D -->|ZigZagResult list| E[technical_analysis\ncalc_ZigZag_adaptive]
    E -->|pivot_markers dict| F[_stabilize_index_adaptive_confirmed\nSuppress transient frames]
    F -->|Stabilized pivot_markers| G[shift_adaptive_pivot_markers\nslice_idx → original_idx]
    G --> H[plot_manager\nupdate_main_plots]
    H --> I[_zz_polyline_from_pivot_markers\nOrange polyline]
    H --> J[_plot_pivot_bucket\n▽ ▲ ◆ markers]

    K[Headless tick path\nIJ_ tick] --> L[maybe_kospi_zz_headless_on_df]
    L -->|_kospi_headless_zz_snapshot| F

    style A fill:#1e3a5f,color:#fff
    style K fill:#1e3a5f,color:#fff
    style D fill:#2d5016,color:#fff
    style F fill:#5c3317,color:#fff
    style H fill:#4a1c6e,color:#fff
```

---

## 2. 세션 슬라이스 (`kospi_zz_rth_slice.py`)

ZigZag 엔진은 **정규 거래 시간 구간만** 입력으로 받습니다. 프리마켓·장외 구간을 포함하면 스윙 기준이 왜곡됩니다.

| 종목 | 슬라이스 시작 | 슬라이스 종료 |
|------|-------------|-------------|
| KOSPI (001) | 09:00 KST | 15:30 KST |
| KP200 (선물) | 08:45 KST | 15:30 KST |
| 옵션 | 08:45 KST | 15:30 KST |

```mermaid
flowchart LR
    subgraph INPUT["Original df (full minute bars)"]
        P1["08:45\nPremarket"]
        P2["09:00\nRegular Start"]
        P3["15:30\nRegular End"]
        P4["15:45\nAfter Hours"]
    end

    subgraph SLICE["Slice Result"]
        S1["df_zz\nRTH only"]
        S2["pos_map\nslice_idx → orig_idx"]
    end

    P1 -. "KOSPI: Excluded\nKP200/Options: Included" .-> SLICE
    P2 --> SLICE
    P3 --> SLICE
    P4 -. "Excluded" .-> SLICE

    SLICE -->|"shift_adaptive_pivot_markers(pos_map)"| R["Original df\nIndex-based markers"]

    style P1 fill:#5c1111,color:#fff
    style P4 fill:#5c1111,color:#fff
    style P2 fill:#1a4a1a,color:#fff
    style P3 fill:#1a4a1a,color:#fff
```

슬라이스 함수 `slice_df_ind_for_kospi_zz()`는 슬라이스된 `df_zz`와 함께 **`pos_map`** (슬라이스 idx → 원본 df idx 매핑 배열)을 반환합니다. 피봇 마커를 차트 전체 인덱스로 다시 변환할 때 이 매핑을 사용합니다 (`shift_adaptive_pivot_markers()`).

> **config.ini 연관 파라미터 없음** — 세션 시간은 하드코딩됨.  
> `MARKET_OPEN_TIME` 설정은 차트 Open 보정(`_fix_index_minute_open`)에만 영향.

---

## 3. 파라미터 상세

### 3-1. `AdaptiveZigZagConfig` (엔진 레벨)

`kospi_indicators/kospi_indicators/adaptive_zigzag.py`에 정의된 dataclass.

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `atr_period` | `14` | ATR 계산 기간 (Wilder RMA). 차트 경로는 `10` 고정. |
| `er_period` | `10` | Efficiency Ratio 계산 기간. ER = `|종가 변화량| / Σ|봉간 변화량|`. 추세 강도 지수. |
| `atr_multiplier` | `1.5` | 기준 ATR 배수. `get_zig_zag()`에서 `atr_mult`로 동적 재산출. |
| `atr_multiplier_min` | `1.0` | ER이 0일 때(횡보) 적용할 최소 ATR 배수. |
| `atr_multiplier_max` | `4.0` | ER이 1일 때(강추세) 적용할 최대 ATR 배수. |
| `pivot_threshold_min_pct` | `0.3` | 피봇 임계값 하한 (%). 아무리 작은 ATR이어도 최소 0.3% 이상이어야 방향 전환을 인정. |
| `pivot_threshold_max_pct` | `3.0` | 피봇 임계값 상한 (%). 아무리 큰 ATR이어도 3% 초과로 올라가지 않음. |
| `confirmation_bars` | `2` | **피봇 확정 대기 봉 수.** `config.ini` → `ADAPTIVE_ZZ_CONFIRMATION_BARS`로 오버라이드. |
| `freeze_on_confirm` | `True` | **확정 대기 중 가격 갱신 차단 여부.** `True`이면 후보 등록 시점 가격이 확정 가격이 됨. `config.ini` → `ADAPTIVE_ZZ_FREEZE_ON_CONFIRM`. |
| `major_swing_ratio` | `2.0` | 이전 스윙 대비 ATR 배수 초과 시 주요 스윙(is_major=True)으로 분류. |
| `max_swings` | `20` | 보관할 최대 스윙 포인트 수. 초과 시 오래된 것부터 삭제. |
| `min_wave_bars` | `5` | 직전 스윙 확정 후 최소 경과 봉 수. 단타 등락에 의한 과도한 스윙 발생 억제. 봉 수 < 80이면 `1`로 완화. |
| `min_wave_pct` | `0.0` | 파동 크기의 최소 % 조건. 0이면 비활성. |
| `max_wait_bars` | `0` | **후보 자동 취소 봉 수.** 0이면 무제한 대기. >0이면 해당 봉 수 경과 후 자동 취소. |
| `structure_lookback_swings` | `8` | 시장 구조(상승/하락/횡보) 분석에 사용할 최근 스윙 수. |
| `structure_points` | `3` | 구조 판단에 필요한 최소 고점/저점 수. |
| `fib_ratios` | `[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]` | 피보나치 계산 비율 목록. |
| `cluster_tolerance_pct` | `0.3` | S/R 클러스터링 허용 오차 (%, 현재 미사용). |
| `zz_log_chart_key` | `None` | KOSPI 엔진 상세 로그 활성화 키. `"001"` 또는 `"KOSPI"` 설정 시 로그 활성화. |
| `zz_log_fn` | `None` | 로그 출력 콜백 (`view.log_message`). |
| `zz_idx_to_time_fn` | `None` | 봉 idx → HH:MM 변환 함수 (로그용). |
| `zz_status_fn` | `None` | UI 상태바 업데이트 콜백. |

### 3-2. `config.ini` / `settings.py` 제어 파라미터

| config.ini 키 | settings.py 변수 | 기본값 | 유효 범위 | 역할 |
|---------------|-----------------|--------|-----------|------|
| `ADAPTIVE_ZZ_CONFIRMATION_BARS` | `ADAPTIVE_ZZ_CONFIRMATION_BARS` | `1` | `0~10` | 후보 피봇이 확정되기까지 대기할 봉 수. **0이면 즉시 확정** (리페인팅 없음 대신 노이즈 감수). 1~2 권장. |
| `ADAPTIVE_ZZ_FREEZE_ON_CONFIRM` | `ADAPTIVE_ZZ_FREEZE_ON_CONFIRM` | `ON` | `ON/OFF` | `ON`이면 후보 등록 시점 가격으로 확정 (피봇 위치 고정). `OFF`이면 대기 중 신고점/신저점이 오면 가격·인덱스 갱신. |
| `ADAPTIVE_ZZ_EXCLUDE_LAST_BAR_LIVE` | `ADAPTIVE_ZZ_EXCLUDE_LAST_BAR_LIVE` | `ON` | `ON/OFF` | 라이브 모드에서 엔진 입력 끝부분에서 미완성 봉을 제외할지 여부. `ON` 권장. |
| `ADAPTIVE_ZZ_EXCLUDE_LAST_OPEN_BARS` | `ADAPTIVE_ZZ_EXCLUDE_LAST_OPEN_BARS` | `1` | `1~5` | 제외할 봉 수. **1 권장** — 현재 형성 중인 마지막 봉(`N-1`)만 제외. 2 이상은 확정된 완성 봉(`N-2`)까지 제외해 피봇 위치가 1봉 지연되고 `CONFIRMATION_BARS=1`과 충돌하므로 비권장. |

### 3-3. `get_zig_zag()` 내 동적 파라미터 계산

```mermaid
flowchart TD
    ZF["zz_factor (UI input, e.g., 3.0)"]
    PC["pc_adaptive = zz_factor × 0.5\n= 1.5"]
    ATR["ATR calculation (period=10, Wilder RMA)\natr_pct_last = ATR / Close × 100"]
    MN["mult_needed = pc_adaptive / atr_pct_last"]
    MX["atr_mult_max = clip mult_needed 1.0 30.0\natr_mult_min = clip atr_mult_max × 0.35 0.5 atr_mult_max"]
    TH["min_thr = clip atr_pct_last × 0.5 0.05 1.0\nmax_thr = clip pc_adaptive × 6.0 min_thr+0.1 30.0"]
    CFG["AdaptiveZigZagConfig\natr_multiplier = atr_mult_max\natr_multiplier_min = atr_mult_min\natr_multiplier_max = atr_mult_max\npivot_threshold_min_pct = min_thr\npivot_threshold_max_pct = max_thr\nconfirmation_bars = ADAPTIVE_ZZ_CONFIRMATION_BARS\nfreeze_on_confirm = ADAPTIVE_ZZ_FREEZE_ON_CONFIRM"]

    ZF --> PC --> ATR --> MN --> MX --> TH --> CFG

    style ZF fill:#1e3a5f,color:#fff
    style CFG fill:#2d5016,color:#fff
```

**`zz_factor`가 클수록 스윙 임계값이 커져 피봇 수가 줄어듭니다** (더 큰 파동만 잡음).

---

## 4. 임계값 계산 상세 (`_calc_threshold_pct`)

스윙 방향 전환을 인정하기 위한 최소 가격 변동 비율을 봉마다 동적으로 계산합니다.

```mermaid
flowchart LR
    subgraph ER["ER Calculation (er_period=10)"]
        direction TB
        C0["C_0 10 bars ago"]
        Cn["C_n current"]
        SUM["Σ|C_i - C_i-1|"]
        ERVAL["ER = |C_n - C_0| / Σ|...|
Range: 0.0~1.0"]
        C0 & Cn & SUM --> ERVAL
    end

    subgraph MULT["Multiplier Decision (FIX-1)"]
        direction TB
        MMIN["atr_multiplier_min"]
        MMAX["atr_multiplier_max"]
        MULTV["mult = mmin + ER × mmax - mmin
Ranging ER≈0 → mult min
Trending ER≈1 → mult max"]
        MMIN & MMAX --> MULTV
    end

    subgraph THR["Threshold Calculation"]
        direction TB
        ATR2["ATR"]
        CLOSE["Close"]
        BASE["base_pct = ATR/Close × 100 × mult"]
        CLIP["threshold_pct = clip base_pct
pivot_threshold_min_pct pivot_threshold_max_pct"]
        ABS["threshold_abs = Close × threshold_pct / 100"]
        ATR2 & CLOSE --> BASE --> CLIP --> ABS
    end

    ER --> MULT --> THR

    style MULT fill:#2d4a1a,color:#fff
```

**직관적 해석:**
- 횡보장(ER≈0): mult 최소 → 임계값 낮음 → 작은 파동도 스윙으로 잡음
- 추세장(ER≈1): mult 최대 → 임계값 높음 → 큰 파동만 인정, 노이즈 제거

---

## 5. 피봇 상태 머신 (`update()`)

### 5-1. 매 봉 실행 순서

```mermaid
sequenceDiagram
    participant OHLC as Bar Data (H/L/C)
    participant S1 as 1. ATR·Threshold
    participant S2 as 2. 3-a: pending processing
    participant S3 as 3. 3-b: direction judgment
    participant ST as ZigZagState

    OHLC->>S1: TR calculation
    S1->>S1: WilderRMA → ATR
    S1->>S1: ER → mult → threshold_pct
    S1-->>S2: threshold_abs

    S2->>S2: max_wait_bars exceeded? → cancel
    S2->>S2: freeze=OFF then price update
    S2->>S2: remaining -= 1
    alt remaining <= 0
        S2->>ST: SwingPoint add (confirmed)
        S2->>S2: pending_confirm = None
        Note over S3: P-FIX-B: skip 3-b this bar
    else remaining > 0
        S2-->>S3: continue
        S3->>S3: direction-specific logic
        S3->>ST: pending_confirm register/update
    end

    ST-->>OHLC: ZigZagState return
```

### 5-2. `_pending_confirm` 처리 상세 (3-a)

```mermaid
flowchart TD
    START[Bar start: pending_confirm exists?]
    NO_PEND[Proceed to 3-b]
    WAIT_CHK{"max_wait_bars > 0\n& waited >= max_wait?"}
    CANCEL_WAIT["Cancel: ZZ_ENGINE_CANCEL\nreason=max_wait_bars\npending_confirm = None"]
    FREEZE_CHK{"freeze_on_confirm\n== False?"}
    UPD_PRICE{"type=high: high > c_price?\ntype=low: low < c_price?"}
    UPDATE["Price·index update\nupdated = True"]
    DEC["remaining -= 1"]
    REM_RESET{"updated=True\n& remaining > 0?"}
    RESET_REM["reset_to = max(1, conf_bars//2+1)\nif remaining < reset_to:\n    remaining = reset_to\nFIX-6: prevent early confirm"]
    ZERO_CHK{"remaining <= 0?"}
    CONFIRM["SwingPoint add\n_all_swings.append\nPivot confirm log\npending_confirm = None\nP-FIX-B: skip 3-b this bar"]
    SEED["New direction pending seed init"]
    CONT[Proceed to 3-b]

    START -- "No" --> NO_PEND
    START -- "Yes" --> WAIT_CHK
    WAIT_CHK -- "Yes" --> CANCEL_WAIT
    WAIT_CHK -- "No" --> FREEZE_CHK
    FREEZE_CHK -- "True (ON)" --> DEC
    FREEZE_CHK -- "False (OFF)" --> UPD_PRICE
    UPD_PRICE -- "Yes" --> UPDATE --> DEC
    UPD_PRICE -- "No" --> DEC
    DEC --> REM_RESET
    REM_RESET -- "Yes" --> RESET_REM --> ZERO_CHK
    REM_RESET -- "No" --> ZERO_CHK
    ZERO_CHK -- "Yes" --> CONFIRM --> SEED
    ZERO_CHK -- "No" --> CONT

    style CONFIRM fill:#1a4a1a,color:#fff
    style CANCEL_WAIT fill:#5c1111,color:#fff
    style RESET_REM fill:#4a3a00,color:#fff
```

### 5-3. 방향 결정 / 전환 (3-b)

```mermaid
stateDiagram-v2
    [*] --> INIT: First bar

    state INIT
        note: direction = 0
        note: pending_high / pending_low tracking
    end state

    state UP
        note: direction = 1 uptrend
        note: pending_high new high update
    end state

    state DOWN
        note: direction = -1 downtrend
        note: pending_low new low update
    end state

    state PEND_H
        note: pending_confirm = high
        note: remaining = confirmation_bars
        note: ZZ_ENGINE_REGISTER
    end state

    state PEND_L
        note: pending_confirm = low
        note: remaining = confirmation_bars
        note: ZZ_ENGINE_REGISTER
    end state

    INIT --> UP: ph_idx > pl_idx and ph-pl>=thr
    INIT --> DOWN: pl_idx > ph_idx and ph-pl>=thr
    UP --> PEND_H: ph-low>=thr and wave_ok
    DOWN --> PEND_L: high-pl>=thr and wave_ok
    PEND_H --> DOWN: direction = -1
    PEND_L --> UP: direction = 1
    PEND_H --> UP: remaining<=0 then confirmed HIGH
    PEND_L --> DOWN: remaining<=0 then confirmed LOW
```

### 5-4. 파동 길이 조건 (`_is_wave_length_ok`)

```mermaid
flowchart LR
    CHK1{"min_wave_bars > 0\n& last_confirmed >= 0?"}
    CHK2{"Current bar - last_confirmed\n< min_wave_bars?"}
    CHK3{"min_wave_pct > 0\n& close > 0?"}
    CHK4{"threshold_abs/close×100\n< min_wave_pct?"}
    OK["✅ Wave allowed\nreturn True"]
    NG["❌ Wave rejected\nreturn False"]

    CHK1 -- "Yes" --> CHK2
    CHK1 -- "No" --> CHK3
    CHK2 -- "Yes (too fast)" --> NG
    CHK2 -- "No" --> CHK3
    CHK3 -- "Yes" --> CHK4
    CHK3 -- "No" --> OK
    CHK4 -- "Yes (wave too small)" --> NG
    CHK4 -- "No" --> OK

    style OK fill:#1a4a1a,color:#fff
    style NG fill:#5c1111,color:#fff
```

---

## 6. `pivot_markers` 딕셔너리 구조

```mermaid
classDiagram
    class pivot_markers {
        +dict confirmed
        +dict unconfirmed
        +int anchor_idx
    }
    class confirmed {
        +List~int~ idx
        +List~float~ y
        +List~str~ type
    }
    class unconfirmed {
        +List~int~ idx
        +List~float~ y
        +List~str~ type
    }

    pivot_markers --> confirmed : Confirmed pivot list
    pivot_markers --> unconfirmed : Waiting candidates
    pivot_markers --> anchor_idx : Open anchor\nPolyline included, markers excluded
```

```python
pivot_markers = {
    "confirmed": {
        "idx":  [int, ...],   # 슬라이스된 df 기준 iloc 인덱스
        "y":    [float, ...], # 피봇 가격
        "type": [str, ...],   # "H" 또는 "L"
    },
    "unconfirmed": {
        "idx":  [int, ...],   # 현재 대기 중인 후보 인덱스
        "y":    [float, ...],
        "type": [str, ...],
    },
    "anchor_idx": int,        # 시가 앵커 피봇의 idx
}
```

**`anchor_idx`**: `confirmed` 버킷이 비어있고 `unconfirmed`가 1개뿐일 때 `_inject_open_anchor_pivot()`이 시가를 앵커 피봇으로 주입합니다. 폴리라인 시작점 역할이며, 마커(▽▲)는 표시하지 않습니다.

---

## 7. 피봇 안정화 로직 (`kospi_adaptive_zz_confirm.py`)

### `_stabilize_index_adaptive_confirmed()`

```mermaid
flowchart TD
    IN["pivot_markers input\nnew calculation result"]
    CHK_SYM{"KOSPI/KP200 symbol?\n& use_adaptive_zz?"}
    CHK_RSN{"reason is quiet tick?\nunknown / empty / tick / timer"}
    HEADLESS{"Headless snapshot exists?"}
    HS_CMP{"len(cur) < len(hs_hhmm)\n& tail match\n& cur ⊆ hs_hhmm?"}
    USE_HS["Return headless snapshot\nstabilization after chart open"]
    PREV_CMP{"Previous stable value exists?"}
    STAB_CMP{"len(cur) < len(prev_hhmm)\n& tail match\n& cur ⊆ prev_hhmm?"}
    USE_PREV["Return previous stable pivot_markers\nignore transient frames"]
    UPDATE_CACHE["_zz_confirmed_stable_cache update"]
    RETURN["Return pivot_markers as is"]

    IN --> CHK_SYM
    CHK_SYM -- "No" --> RETURN
    CHK_SYM -- "Yes" --> CHK_RSN
    CHK_RSN -- "Explicit event" --> RETURN
    CHK_RSN -- "Quiet tick" --> HEADLESS
    HEADLESS -- "Yes" --> HS_CMP
    HEADLESS -- "No" --> PREV_CMP
    HS_CMP -- "Yes (shrink)" --> USE_HS
    HS_CMP -- "No" --> PREV_CMP
    PREV_CMP -- "No" --> UPDATE_CACHE --> RETURN
    PREV_CMP -- "Yes" --> STAB_CMP
    STAB_CMP -- "Yes (shrink)" --> USE_PREV
    STAB_CMP -- "No" --> UPDATE_CACHE --> RETURN

    style USE_HS fill:#1e3a5f,color:#fff
    style USE_PREV fill:#1e3a5f,color:#fff
    style UPDATE_CACHE fill:#2d5016,color:#fff
```

### 헤드리스 ZZ 스냅샷 흐름

```mermaid
sequenceDiagram
    participant TICK as IJ_ tick (during session)
    participant HL as maybe_kospi_zz_headless_on_df
    participant SNAP as app._kospi_headless_zz_snapshot
    participant CHART as Index chart (on open)
    participant STAB as _stabilize_index_adaptive_confirmed

    loop Every tick (chart not open)
        TICK->>HL: KOSPI minute df
        HL->>HL: AdaptiveZigZag calculation
        HL->>SNAP: confirmed_hhmm, pivot_markers save
    end

    Note over CHART: User opens chart
    CHART->>HL: calc_ZigZag_adaptive run
    HL->>STAB: New pivot_markers deliver
    STAB->>SNAP: Headless snapshot query
    SNAP-->>STAB: hs_hhmm, hs_pm
    STAB->>STAB: New < headless? tail match? subset?
    STAB-->>CHART: Stabilized pivot_markers return
    Note over CHART: Suppress pivot shrinkage visual
```

---

## 8. 차트 렌더 흐름 (`plot_manager.py`)

```mermaid
flowchart TD
    PM["plot_manager\nupdate_main_plots"]
    TM_CHK{"touch_mode\n== last_bar?"}
    SKIP_ZZ["ZigZag·ST recalc skip\nhigh·low·current markers only"]
    FULL_INTERVAL{"CHART_TOUCH_FULL_REFRESH\n_INTERVAL_SEC elapsed?"}
    FULL_RENDER["FULL render execute"]
    ZZ_POLY["_zz_polyline_from_pivot_markers\nfplt.plot style color orange width 2.0"]
    BUCKET_C["_plot_pivot_bucket confirmed\nAll confirmed pivots"]
    BUCKET_U["_plot_pivot_bucket unconfirmed\nKOSPI/KP200: only_last True\nOthers: all"]
    MH["confirmed H: ▽\nstyle v color orange"]
    ML["confirmed L: △\nstyle ^ color orange"]
    MU["unconfirmed: ◆\nstyle d color orange"]
    ANCH["anchor_idx → marker exclude"]
    DBG["Debug text overlay\nRecent 6 H/L labels"]

    PM --> TM_CHK
    TM_CHK -- "last_bar" --> SKIP_ZZ
    TM_CHK -- "full" --> FULL_RENDER
    SKIP_ZZ --> FULL_INTERVAL
    FULL_INTERVAL -- "Elapsed" --> FULL_RENDER
    FULL_INTERVAL -- "Not elapsed" --> SKIP_ZZ
    FULL_RENDER --> ZZ_POLY
    FULL_RENDER --> BUCKET_C & BUCKET_U
    BUCKET_C --> MH & ML & ANCH & DBG
    BUCKET_U --> MU

    style SKIP_ZZ fill:#3a2a00,color:#fff
    style FULL_RENDER fill:#1a4a1a,color:#fff
```

---

## 9. 주요 버그 수정 이력

```mermaid
timeline
    title Adaptive ZigZag Bug Fix Timeline

    section Core Logic
        FIX-1 : ER direction reversal fix
              : mult = mmin + er x (mmax-mmin)
              : Trending threshold normalization
        FIX-2 : pending opposite type replace allow
              : Prevent swing miss on fast reversal

    section Stability
        FIX-3 : _all_swings del to slicing
        FIX-4 : _find_nearest_sr empty list to 0.0
        FIX-5 : Direction switch move inside block
              : Suppress repeated REGISTER logs

    section Confirmation Logic
        FIX-6 : remaining partial reset
              : Prevent early confirmation
        FIX-7 : max_wait_bars auto cancel
        FIX-8 : pending unconfirmed marker always show

    section Patch_TST
        P-FIX-A : freeze=True still price update on replace_opposite
                : Fix strong trend pivot miss
        P-FIX-B : Block 3-b entry on confirm bar
        P-FIX-C : Opposite pending seed init after initial direction confirm
```

| 코드 | 내용 | 영향 |
|------|------|------|
| **FIX-1** | ER 방향 역전 수정: `mult = mmin + er*(mmax-mmin)` | 추세장에서 임계값이 정상적으로 커짐 |
| **FIX-2** | `pending_confirm` 반대 타입이면 교체 허용 | 빠른 반전 시 스윙 누락 방지 |
| **FIX-3** | `_all_swings` 관리: del → 슬라이싱 재할당 | 코드 명확성 개선 |
| **FIX-4** | `_find_nearest_sr` 빈 리스트 → 0.0 반환 | downstream > 0 체크로 처리 가능 |
| **FIX-5** | 방향 전환·pending 리셋을 신규 후보 등록 블록 안으로 이동 | 동일 타입 반복 REGISTER 로그 억제 |
| **FIX-6** | 신고점/신저점 갱신 시 remaining 부분 리셋 | 마지막 봉 신고점에 의한 조기 확정 방지 |
| **FIX-7** | `max_wait_bars`: 오래된 pending 자동 취소 | 장기 대기 후보 처리 |
| **FIX-8** | `_pending_confirm`이 있으면 항상 unconfirmed 마커 표시 | 워밍업 이후 unconfirmed 마커 누락 수정 |
| **P-FIX-A** | `freeze_on_confirm=True`여도 `replace_opposite` 시 가격 갱신 후 재등록 | 강추세에서 낮은 고점이 확정되는 피봇 누락 수정 |
| **P-FIX-B** | 3-a 확정 봉에서 3-b 진입 차단 | 확정+즉시 재등록 혼재 방지 |
| **P-FIX-C** | 초기 방향 확정 후 반대 방향 pending을 현재 봉 H/L로 초기화 | `_pending_high=0.0`/`_pending_low=inf` 기준 오작동 수정 |

---

## 10. 로그 레퍼런스

```mermaid
flowchart LR
    subgraph ENGINE["Engine Level Logs (KOSPI/KP200)"]
        REG["ZZ_ENGINE_REGISTER\nCandidate register\ntype, swing_at, price, rem"]
        UPD["ZZ_ENGINE_UPDATE\nPrice·index update\nfreeze=OFF only"]
        CAN["ZZ_ENGINE_CANCEL\nCandidate cancel\nmax_wait_bars\nreplace_opposite\nexception"]
        CHN["ZZ_ENGINE_CHAIN\nLifecycle summary\nregister/update/confirm/cancel\nfinal result"]
        PND["ZZ_PENDING_STATUS\nWait status on minute update"]
        REG --> CHN
        UPD --> CHN
        CAN --> CHN
    end

    subgraph TA["TechnicalAnalysis Level"]
        ADP["ZZ adaptive\nPivot stats\ncount, last timestamp"]
        LEG["ZZ legacy\nLegacy pivot stats"]
    end

    subgraph PM["PlotManager Level"]
        CHG["ZZ Points changed\nconfirmed timestamp change"]
        DIAG["ZZ_PIVOT_DIAG\nMarker diagnosis\nconfirmed/unconfirmed count"]
        PLOT["ZZ_PIVOT_PLOT\nActual coordinate diagnosis"]
    end

    style ENGINE fill:#1e1e3a,color:#fff
    style TA fill:#1a3a1a,color:#fff
    style PM fill:#3a1a1a,color:#fff
```

> **KOSPI 엔진 로그 활성화 조건**: `zz_log_chart_key`에 `"001"` 또는 `"KOSPI"` 설정 시. 차트에서는 심볼이 KOSPI/KP200일 때 `_wants_kospi_zz_engine_log()`가 True를 반환하면 활성화.

---

## 11. 설정 권장값 (운영 기준)

```ini
[SETTINGS]
# 확정 대기: 1봉. 0이면 즉시 확정으로 리페인팅 없음
ADAPTIVE_ZZ_CONFIRMATION_BARS = 1

# 후보 등록 시점 가격 고정. ON 권장 (피봇 위치 안정)
ADAPTIVE_ZZ_FREEZE_ON_CONFIRM = ON

# 마지막 미완성 봉 제외. ON 권장
ADAPTIVE_ZZ_EXCLUDE_LAST_BAR_LIVE = ON

# 제외 봉 수. 1 권장 — N-1(형성 중)만 제외. 2 이상은 확정 봉까지 제외해 피봇 1봉 지연 발생.
ADAPTIVE_ZZ_EXCLUDE_LAST_OPEN_BARS = 1
```

| UI `zz_factor` | 특성 | 용도 |
|----------------|------|------|
| 1.0~2.0 | 민감 (피봇 많음) | 단기 스캘핑, 세밀한 파동 추적 |
| 3.0~5.0 | 중간 (권장) | 일반 분봉 차트 |
| 6.0 이상 | 둔감 (피봇 적음) | 장기 추세 파악 |

---

## 12. 전체 데이터 흐름 다이어그램

```mermaid
flowchart TD
    subgraph INPUT["Input Preprocessing"]
        RAW[Original OHLCV]
        EXCL["EXCLUDE_LAST_OPEN_BARS=1\nLast bar excluded N-1 only"]
        SLICE["slice_df_ind_for_kospi_zz\nKOSPI: 09:00~\nKP200/Options: 08:45~"]
        RAW --> EXCL --> SLICE
    end

    subgraph PARAM["Parameter Assembly (get_zig_zag)"]
        ATR_C["ATR period 10 calculation"]
        PC["pc_adaptive = zz_factor × 0.5"]
        MUL["atr_mult_max/min calculation"]
        THR["min_thr / max_thr calculation"]
        CFG_A["AdaptiveZigZagConfig"]
        ATR_C & PC --> MUL --> THR --> CFG_A
    end

    subgraph ENGINE["Engine (AdaptiveZigZag.update × n bars)"]
        direction LR
        E1["1. ATR·ER·threshold calculation"]
        E2["2. 3-a: pending_confirm processing\nremaining decrement → confirm or wait"]
        E3["3. 3-b: direction judgment·candidate registration\npending creation on direction change"]
        E1 --> E2 --> E3
    end

    subgraph POST["Post-processing (calc_ZigZag_adaptive)"]
        PM1["ZigZagResult → pivot_markers assembly"]
        PM2["anchor_idx injection\n_inject_open_anchor_pivot"]
        PM3["_stabilize_index_adaptive_confirmed\nSuppress transient frames"]
        PM4["shift_adaptive_pivot_markers\nslice_idx → original_idx"]
        PM1 --> PM2 --> PM3 --> PM4
    end

    subgraph RENDER["Chart Render (plot_manager)"]
        R1["Orange polyline\n_zz_polyline_from_pivot_markers"]
        R2["confirmed ▽▲ markers\n_plot_pivot_bucket"]
        R3["unconfirmed ◆ markers\nKOSPI: last 1"]
        R4["H/L debug text overlay"]
    end

    HEADLESS["Headless ZZ Snapshot\nUpdated every tick when chart not open"]

    INPUT -->|df_zz, pos_map| PARAM
    PARAM -->|AdaptiveZigZagConfig| ENGINE
    ENGINE -->|ZigZagResult list| POST
    HEADLESS -->|hs_hhmm, hs_pm| POST
    POST -->|Stabilized pivot_markers| RENDER

    style INPUT fill:#1e1e3a,color:#fff
    style ENGINE fill:#1a3a1a,color:#fff
    style POST fill:#3a2a00,color:#fff
    style RENDER fill:#3a1a3a,color:#fff
    style HEADLESS fill:#1e3a3a,color:#fff
```
