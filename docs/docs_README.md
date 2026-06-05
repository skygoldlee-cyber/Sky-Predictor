# KOSPI200 Price Predictor

 KOSPI200 선물/옵션 틱 데이터를 기반으로 **분봉을 집계**하고, (선물) 오더북/체결 정보를 활용해 **단기(5/10/15분) 방향성 판단 + 전략 코멘트(LLM)** 를 제공합니다.

 지원 모드:
- `Heuristic` 예측 (기본 동작)
- `LLM` 판단 (Claude / GPT / Gemini, API Key 필요)
- `eBest Live` 실시간 구독 모드 (eBest 래퍼 필요)

LLM 관련 (옵션):
- `dual_llm` 모드에서는 GPT/Gemini를 **동시에 호출**해 둘 다의 결과를 로그/출력에 포함할 수 있습니다.
  - 설정 키: `prediction.dual_llm` (bool), `prediction.dual_llm_primary_provider` (`gpt`|`gemini`)
  - `dual_llm_primary_provider`는 최종 `llm_action/risk_level/rationale`에 반영되는 provider를 의미합니다.

추가 입력(시장 배경):
- KOSPI200 현물지수(`IJ_`, key=`101`)를 실시간으로 수신하여 선물-현물 괴리(`basis`)를 계산합니다.
  - GUI `RT:` 라인에 `spot`/`basis`가 함께 표시됩니다.
  - LLM 컨텍스트에는 `market_background.ij_` 및 `market.spot_index/basis`로 포함됩니다.
  - `basis`가 과도할 때는 신뢰도 하향 또는 `HOLD`로 전환하는 가드레일이 적용됩니다.

Transformer 관련 로직(입력 피처, 모델, 학습/추론)은 별도 문서에 정리되어 있습니다.
 - [ML_PREDICTION_GUIDE.md](ML_PREDICTION_GUIDE.md) - 머신러닝 예측 시스템 통합 가이드

운영/학습 루틴(매일 ticks 저장 → dataset 생성 → 최근 N일 merge → 재학습 → 성능평가)은 아래 런북을 참고하세요.
- [DAILY_TICK_TRAINING_RUNBOOK.md](DAILY_TICK_TRAINING_RUNBOOK.md)

듀얼 모델(TFT 포함) 설계 가이드는 아래 문서에 정리되어 있습니다.
- [TFT_DUAL_MODEL_DESIGN_GUIDE.md](TFT_DUAL_MODEL_DESIGN_GUIDE.md)

---
 
## 파일 구조

```
project/
├── main.py                 # CLI 엔트리포인트
├── pyproject.toml          # 프로젝트 설정 (버전 관리)
├── config/                 # 설정 모듈
│   ├── __init__.py         # load_config, AppConfig (re-export)
│   ├── config.py           # config.json 로드/검증
│   └── constants.py        # 상수/Enum
├── core/                   # 핵심 모듈
│   ├── cli_args.py         # 커맨드라인 인자 파싱
│   ├── logging_utils.py    # 로깅 설정(파일 로테이션, tee)
│   └── utils.py            # safe_* / 날짜/통계 유틸
├── app/                    # 애플리케이션 모듈
│   ├── app_setup.py        # 애플리케이션 설정
│   ├── pipeline_builder.py # 파이프라인 빌더
│   └── run_modes.py        # 실행 모드 (test/replay/live)
├── logs/                   # 로그 파일 디렉토리
│   ├── prediction.log      # 메인 로그 파일
│   └── watchdog.log        # watchdog 로그 파일
├── tests/                  # pytest 테스트 디렉터리
│   ├── test_smoke.py       # 스모크 테스트
│   └── prediction/         # 예측 관련 테스트
├── telegram_notifier.py     # Telegram 알림/명령 수신
├── Transformer.spec         # PyInstaller spec(단일 실행파일 빌드)
├── assets/                 # 아이콘 등 리소스
│   └── beacon.ico           # 실행파일 아이콘
├── tick_normalizer.py      # eBest 실시간 tick 표준화(tick_norm 생성)
├── tick_processor.py       # 실시간 틱 처리 + 분봉 집계
├── adaptive_indicator/      # Adaptive 지표 모듈
├── prediction/             # 현재 권장 파이프라인(Transformer + LLM 역할 분리)
│   ├── pipeline.py          # PredictionPipeline (예측 + 판단 오케스트레이션)
│   ├── features.py          # FH0 오더북 feature + 분봉 feature + 시퀀스 빌드
│   ├── option_features.py   # 옵션 지표(PCR/IV Skew/Max Pain)
│   ├── predictor.py         # Transformer/TFT/Ensemble 예측기 (가중치 없으면 rule-based fallback)
│   ├── model.py             # PriceTransformer (PyTorch)
│   ├── tft_model.py          # TemporalFusionTransformer (PyTorch)
│   ├── time_features.py      # TFT time features (past_known/future_known)
│   ├── context_builder.py   # LLM 컨텍스트/프롬프트 구성
│   └── llm_judge.py         # Claude/GPT/Gemini 호출 + JSON 파싱/정규화
│   ├── data_builder.py      # ticks_replay_*.jsonl/.jsonl.gz → dataset npz
│   └── weights/             # 가중치 저장(transformer_5m.pt, tft_5m.pt)
├── ebest_live.py           # eBest 연동(예측 루프 오케스트레이션)
├── ebest_api.py            # eBest 인증/REST 요청 헬퍼
├── ebest_options.py        # 옵션 심볼 필터링/ATM 유틸
├── ebest_callbacks.py      # realtime/message callback + ACK 판별
├── train.py                # 오프라인 학습(dataset npz → transformer_5m.pt)
├── train_tft.py             # 오프라인 학습(TFT dataset npz → tft_5m.pt)
├── merge_datasets.py        # 최근 N일 dataset_*.npz merge
├── docs/                   # 문서 (runtime/training 등)
├── tests.py                # (호환) main.py --test용 pytest wrapper
├── config.json             # 설정 파일(선택)
├── config.secrets.json      # 민감 설정 파일(선택, gitignore 권장)
└── requirements.txt        # 의존성(필수/옵션 분리)
```

---

참고:
- Transformer 입력 시퀀스의 `feature_dim`은 다음 설정 조합에 따라 달라집니다.
  - `prediction.option_feature_set` (`v1`/`v2`/`v3`/`v4`)
  - `adaptive_indicator.enabled` (true/false)

LLM 모델명 운영 정책(중요):
- 모델명은 `constants.py`에서 중앙 관리합니다.
- Provider 측 모델명이 변경되거나 권한/리전 이슈로 호출이 실패할 수 있습니다.
- 런타임에서는 모델 관련 오류 발생 시 fallback 모델로 자동 재시도합니다.
  - Gemini: `GEMINI_MODEL` → `GEMINI_FALLBACK_MODELS`
  - Claude: `CLAUDE_MODEL` → `CLAUDE_FALLBACK_MODELS`
- 운영 중 모델 오류가 반복되면, `constants.py`의 기본 모델을 최신 유효 모델로 업데이트하세요.
 
## 설치

기본 실행(heuristic-only)에 필요한 최소 의존성:

```bash
pip install -r requirements.txt
```

---

## 배포(단일 실행파일)

PyInstaller로 단일 실행파일을 빌드할 수 있습니다.

- Spec 파일: `Transformer.spec`
- 아이콘: `assets/beacon.ico`

빌드(Windows PowerShell, 프로젝트 루트에서):

```powershell
pyinstaller --noconfirm --clean --onefile Transformer.spec
```

출력:

- `dist/Transformer.exe`

### 선택 의존성

- `scipy`:
  - `utils.norm_cdf/norm_pdf` 정확도/성능 개선
- LLM 사용 시:
  - `anthropic`, `openai`, `google-genai`
- eBest Live 모드(Qt + asyncio event loop) 사용 시:
  - `PySide6`, `qasync`

---
 
## 설정(`config.json`)

`config.json`이 없으면 기본값으로 동작하지만, LLM/eBest를 쓰려면 설정을 권장합니다.

- GUI 모드는 설정 파일 선택 UI가 없으며, 프로젝트 루트의 `config.json`을 사용합니다.

### 텔레그램(Telegram)

- 텔레그램 연동 및 명령어는 `telegram_notifier.py`의 `PipelineTelegramBridge`가 담당합니다.
- 브릿지 시작 시 `🚀 <b>SkyEbest 예측 시스템 시작</b>`이 1회 전송됩니다.
- 브릿지 종료 시 `🛑 <b>SkyEbest 예측 시스템 종료</b>`이 1회 전송됩니다.
- 예측 결과에 `error`가 포함된 경우는 텔레그램으로 전송하지 않으며, 내부 로그에만 남습니다.
- GUI 로그에서 텔레그램 송신/수신 로그(`[TG][SEND]`, `[TG][RECV]`)를 확인할 수 있습니다.

자세한 런타임 동작/명령어는 아래 문서를 참고하세요.

- [docs/runtime/telegram.md](docs/runtime/telegram.md)

보안상 API Key류는 `config.json`에 직접 넣지 않고, 아래 우선순위로 로드됩니다.

1) 환경변수
2) `config.json`과 같은 폴더의 `config.secrets.json` (gitignore 대상)
3) `config.json`

`config.secrets.json` 경로는 환경변수 `APP_SECRETS_CONFIG`로 오버라이드할 수 있습니다.

```json
{
  "ai_providers": {
    "anthropic": {},
    "openai": {},
    "gemini": {}
  },
  "ebest": {},
  "options_subscription": {
    "itm": 6,
    "otm_open_min": 0.30,
    "max_otm_calls": 20,
    "max_otm_puts": 30,
    "wait_sec": 2
  },
  "option_minute_ohlcv": {
    "enabled": false,
    "atm_window": 2
  },
  "prediction_minutes": 5,
  "min_minute_bars_required": 20,
  "seq_len": 60,
  "fo0_stale_sec": 10,
  "fo0_log_schema": true,
  "preferred_provider": "",
  "prediction": {
    "numeric_predictor": "ensemble",
    "dual_llm": false,
    "dual_llm_primary_provider": "gpt",
    "buy_threshold": 0.62,
    "sell_threshold": 0.38,
    "confidence_high_margin": 0.15,
    "confidence_mid_margin": 0.08,
    "confidence_spread_max_for_high": 1.0,
    "transformer_weight": 0.5,
    "tft_weights_path": "",
    "tft_horizon": 300,
    "disagreement_hold": true,
    "disagreement_hold_prob_diff_max": 0.1,
    "guard_basis_hold_thr": 2.5,
    "guard_basis_downgrade_thr": 1.5,
    "guard_atm_spread_pct_thr": 1.5,
    "guard_atm_liq_log_thr": 2.0
  },
  "adaptive_indicator": {
    "enabled": true,
    "symbol": "KOSPI200 선물",
    "warmup_bars": 30,
    "supertrend": {
      "atr_min_period": 7,
      "atr_max_period": 21,
      "multiplier_min": 1.5,
      "multiplier_max": 4.0,
      "er_period": 10,
      "adx_period": 14,
      "use_bb_correction": true,
      "adx_mult_norm_cap": 60.0,
      "bb_correction_floor": 0.7,
      "bb_correction_ref_pct": 0.05,
      "bb_period": 20,
      "bb_std": 2.0,
      "smooth_period": 3
    },
    "zigzag": {
      "atr_multiplier": 1.5,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.3,
      "pivot_threshold_max_pct": 3.0,
      "major_swing_ratio": 2.0,
      "max_swings": 20,
      "confirmation_bars": 2,
      "cluster_tolerance_pct": 0.3
    }
  }
}
```

`config.secrets.json` 예시(로컬에만 생성; 커밋 금지):

```json
{
  "ai_providers": {
    "anthropic": { "api_key": "" },
    "openai": { "api_key": "" },
    "gemini": { "api_key": "" }
  },
  "ebest": {
    "appkey": "",
    "appsecretkey": ""
  }
}
```

참고:
- `prediction.numeric_predictor`:
  - `transformer`: Transformer(가중치 없으면 rule-based fallback)
  - `tft`: TFT 단독
  - `ensemble`: Transformer + TFT 앙상블
  - `combined`: `ensemble` 별칭(호환)
  - `rule_based`: rule-based 고정
- `prediction.option_feature_set`:
  - `v1`: 기본 OPT(7) — PCR, IV skew, max pain, microstructure
  - `v2`: 확장 OPT(16) — v1 + option_minute_ohlcv 기반 옵션 미세움직임 9개
  - `v3`: 만기주 패리티 이탈 OPT(23) — v2 + 콜-풋 패리티 이탈 지표 7개
  - `v4`: 만기주 프리미엄 블리드 OPT(29) — v3 + 프리미엄 블리드 지표 6개. BleedMonitor 알림 자동 활성
  - ⚠️ v2~v4는 `feature_dim`이 변경되므로 dataset 재생성 및 재학습이 필요합니다.
- `prediction.transformer_weight`: 앙상블 내 Transformer 가중치(0~1)
- `prediction.tft_weights_path`: TFT 가중치 경로(비어 있으면 predictor 기본값)
- `prediction.tft_horizon`: TFT horizon(학습/서빙 일치 필요)
- `prediction.disagreement_hold`: 앙상블 disagreement 시 HOLD 강제 여부

Adaptive indicators:
- `adaptive_indicator.enabled`: AdaptiveSuperTrend/AdaptiveZigZag 피처 + LLM 컨텍스트 생성 활성화
- `adaptive_indicator.warmup_bars`: 시작 시 분봉 DF에서 최근 N개로 지표 상태를 워밍업(기본 45; 최솟값 45 — ADX 이중 RMA 수렴 ≈28봉 + ZigZag 구조 신호 안정화 버퍼)
- `adaptive_indicator.supertrend.*`: Adaptive SuperTrend 파라미터
- `adaptive_indicator.zigzag.*`: Adaptive ZigZag 파라미터

Option minute OHLCV:
- `option_minute_ohlcv.enabled`: OC0 옵션 체결 틱을 심볼별로 분봉 OHLCV로 집계할지 여부
- `option_minute_ohlcv.atm_window`: 콜/풋 ATM 기준 ±N개만 분봉 집계 (예: 2면 ATM±2)

참고:
- `prediction.option_feature_set=v2/v3/v4`의 추가 OPT 피처(`optm_*`)는 option_minute_ohlcv 데이터가 있어야 유효합니다.

환경변수로도 설정 가능:
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
- `EBEST_APPKEY`, `EBEST_APPSECRET`

PowerShell에서 `config.secrets.json` 경로를 강제로 지정하려면:

```powershell
$env:APP_SECRETS_CONFIG=".\config.secrets.json"
python main.py
```

PowerShell 예시(현재 터미널 세션에서만):

```powershell
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:EBEST_APPKEY="..."
$env:EBEST_APPSECRET="..."
python main.py
```

PowerShell 예시(사용자 환경변수로 영구 저장; 새 터미널부터 반영):

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
setx ANTHROPIC_API_KEY "..."
setx EBEST_APPKEY "..."
setx EBEST_APPSECRET "..."
```

참고:
- `main.py`는 `config.config.AppConfig`를 사용합니다.
- 레거시 모듈(`predictor_pkg/`, `predictor.py`, `financial_derivatives_predict.py`)은 제거되었습니다.

---
 
## 실행 방법

### 0) GUI 모드(기본)

`main.py`는 기본 실행이 GUI입니다.

```bash
python main.py
```

GUI 의존성:

- PySide6
- qasync
- qt-material

GUI 주요 항목:

- **Use Transformer / Use TFT**: 수치 예측기 모드 선택(Transformer/TFT/앙상블)
- **Include options**: 옵션 실시간 데이터 포함 여부
- **Option month (YYMM)**: 옵션 월물(비우면 자동)
- **Save ticks / Out ticks**: 실시간 틱을 `.jsonl`로 저장(비우면 기본 파일명 자동 생성)
- **Compress ticks (.gz)**: 저장된 틱 로그를 `.jsonl.gz`로 스트리밍 압축 저장
- **Replay**: 저장된 tick 로그(`.jsonl`/`.jsonl.gz`)를 predictor에 재주입하여 디버깅(replay). 실행 중에는 버튼이 Pause/Resume 토글로 동작

GUI 표시(요약):

- **Summary**: 방향 예측 종합 상태(Consensus)
  - 우측 LED는 `CONS`(BUY/SELL/HOLD)를 색으로 표시하고(초록/빨강/노랑), 중앙 텍스트로 votes(`N/3`)를 표시
- Adaptive indicators 활성/비활성은 GUI에서 토글하지 않으며 `config.json` 또는 CLI(`--adaptive-enabled/--no-adaptive-enabled`)로 제어

GUI RT 라벨(`RT:`) 참고:

- 수신 카운트: `FC0/FH0/OC0/OH0/JIF`
- 선물 가격: `fut_5m_ago`, `fut_now`
- 마지막 심볼/가격 요약:
  - `FC0=<symbol> <price>` / `OC0=<symbol> <price>`
  - `OH0C=<count> <symbol>` / `OH0P=<count> <symbol>`
- 평가(누적): `DIR=xx.x% (hit/total)`
- 옵션 의미가(당일 extreme, exact match):
  - `SRH=<symbol> H<price>` / `SRL=<symbol> L<price>`

### 0.1) CLI 모드(--cli)

GUI 대신 기존 CLI로 실행하려면 `--cli`를 사용합니다.

---

## 오프라인 학습(데이터셋 → 가중치)

오프라인 학습은 다음 흐름으로 진행합니다.

- `ticks_replay_*.jsonl` → `prediction.data_builder`로 NPZ 생성
- `train.py` / `train_tft.py`로 가중치(`.pt`) 학습

학습/데이터셋 생성은 `config.json`을 함께 전달하는 방식을 권장합니다.
- dataset의 `feature_dim`은 `prediction.option_feature_set`(v1/v2/v3/v4) 및 `adaptive_indicator.enabled`(true/false)에 따라 달라집니다.
  - v1: 19(adaptive=off) / 47(adaptive=on)
  - v2: 28 / 56
  - v3: 35 / 63
  - v4: 41 / 69
- dataset npz에는 schema metadata가 함께 저장되며, `train.py`/`train_tft.py`는 학습 시작 전에 설정과의 불일치를 검증합니다.

예시:

```bash
# Transformer/TFT 공용 dataset 생성
python -m prediction.data_builder --config config.json \
  --files ticks_replay_YYYYMMDD.jsonl.gz \
  --out dataset_YYYYMMDD.npz --seq-len 60 --horizon 5

# TFT dataset 생성
python -m prediction.data_builder --config config.json \
  --files ticks_replay_YYYYMMDD.jsonl.gz \
  --out dataset_tft_YYYYMMDD.npz --seq-len 60 --horizon 5 --tft --tft-horizon-sec 300

# 학습
python train.py --config config.json --data dataset_YYYYMMDD.npz --out prediction/weights/transformer_5m.pt
python train_tft.py --config config.json --data dataset_tft_YYYYMMDD.npz --out prediction/weights/tft_5m.pt
```

```bash
python main.py --cli --help
```

### 1) 테스트

```bash
python main.py --test
```

또는 pytest 직접 실행:

```bash
python -m pytest -q
```

### 2) Heuristic 예측만 실행(실시간 연결 없이)

```bash
python main.py --heuristic-only --no-ebest-live
```

디버그(LLM 프롬프트 확인):

```bash
python main.py --dump-llm-prompt
```

- LLM `user` 프롬프트 문자열을 최초 1회 로그로 덤프합니다.
- 출력 태그: `[LLM_USER_PROMPT_DUMP] (first occurrence only)`

### 2.1) 수치 예측기 모드/앙상블 파라미터 CLI 오버라이드

```bash
python main.py --numeric-predictor ensemble --transformer-weight 0.6 --tft-horizon 300
python main.py --numeric-predictor transformer
python main.py --numeric-predictor tft --tft-weights-path prediction/weights/tft_5m.pt
python main.py --no-disagreement-hold
```

주의:
- 실시간 틱이 없으면 예측에 필요한 분봉이 부족할 수 있습니다.
- `main.py`는 기본값이 `--ebest-live`이므로, 단순 예측 목적이면 `--no-ebest-live`를 사용하세요.

### 3) eBest Live 모드

```bash
python main.py
```

참고(LLM warmup/log):

- `Use LLM: True`인 경우, `main.py`는 시작 시 LLM을 1회 warmup 호출하여(OpenAI/Gemini) 초기화/연결 상태를 로그로 먼저 출력합니다.
  - 09:00 이전(예: 08:45) 실행에서도 LLM 관련 문제를 조기에 확인하기 위함입니다.
  - warmup 실패는 non-fatal이며 런타임은 계속 진행됩니다.
- OpenAI/Gemini 초기화 상태 로그는 성공/실패/스킵(키 없음) 여부가 항상 출력됩니다.
  - Task Scheduler로 실행 시에는 환경변수(API key) 주입 여부에 따라 스킵 로그가 나올 수 있습니다.

#### 3.1) ticks_replay 저장 및 압축

- `main.py`는 `--compress-ticks`가 활성화된 경우 tick 로그를 **`.jsonl.gz`로 스트리밍 압축 저장**합니다.
- 압축 비활성화(원본 `.jsonl` 저장):

```bash
python main.py --no-compress-ticks
```

- 날짜별 파일명으로 저장(권장):

```bash
python main.py --include-options --out-ticks ticks_replay_YYYYMMDD.jsonl
```

- `ebest_live.py`는 `import ebest`에 의존합니다.
- 프로젝트/환경에 eBest 래퍼 모듈(`ebest.OpenApi`)이 준비되어 있어야 합니다.

참고:
- 실시간 콜백은 원본 raw `tick`을 유지하면서, 문서 스키마 기준으로 표준화한 `tick_norm`을 함께 생성해 파이프라인으로 전달합니다.
- Live 모드에서는 `JIF`(실시간 장운영 정보)도 구독/기록할 수 있습니다.
  - `JIF`는 모니터링 목적이며, 현행 구현에서 예측 피처 입력으로 사용하지 않습니다.
  - `jstatus` 변경 시 `[JIF_STATUS] ...` 로그가 출력되며, 1Hz rate-limit된 `[JIF] ...` 로그도 출력됩니다.
  - `jangubun == "5"` 이고 `jstatus == "41"` 수신 시 `[JIF_CLOSE] ...` 로그 후 라이브 루프가 정상 종료됩니다.

옵션 호가(OH0) 구독 참고:

- `OH0`는 부하를 줄이기 위해 **ATM±2** 근방만 구독하며, 60초마다 ATM을 재계산해서 필요한 심볼을 **추가 구독**합니다.
  - 신규 구독 로그: `[OH0_REFRESH] ...`

옵션 의미가 레벨 설정:

- `config.json`의 `meaningful_option_levels`에 의미가(가격 레벨) 리스트를 정의합니다.
  - OC0 당일 최고/최저가가 해당 레벨과 정확히 일치할 때 `SRH`/`SRL`로 표시됩니다.

---
 
## 예측 결과 포맷

`predictor.get_prediction()`은 다음 키들을 포함한 dict를 반환합니다.

현재 `PredictionPipeline` 기준 주요 출력:

- `prob`: 상승 확률(0~1, 현재는 rule-based 또는 추후 Transformer)
- `signal`: `BUY|SELL|HOLD`
- `confidence`: `HIGH|MEDIUM|LOW`
- `llm_action`: `BUY|SELL|HOLD`
- `llm_provider`: `claude|gpt|gemini` (실제로 사용된 provider)
- `llm_timed_out`: LLM 호출 타임아웃 여부 (True/False)
- `risk_level`: `LOW|MEDIUM|HIGH`
- `rationale`, `caution`
- `consensus`: Transformer(signal)와 LLM(action) 일치 여부
- `ob_records_len`: FH0 오더북 버퍼 길이(1Hz 기준)
- `fo0_age_sec`: 마지막 FH0 수신 이후 경과 시간(초)
- 위의 `fo0_*` 명칭은 과거 호환 용어이며, 현행 구현에서 오더북 입력 TR은 `FH0`입니다.
- `options`: 옵션 지표 스냅샷(PCR/IV Skew/Max Pain 등, best-effort)

참고:
- 위의 출력이 본 저장소의 **현행 구현 전부**입니다.
- 과거 버전에서 제공되던 `ensemble`, `reversal`, `expiry_blending`, `model_outputs` 등의 필드는 레거시 코드 제거와 함께 더 이상 반환되지 않습니다.

신뢰도(`confidence`) 범위:
- 전 경로에서 `_CONFIDENCE_MIN`~`_CONFIDENCE_MAX`(기본 10~85)로 클램핑됩니다.

방향 계산(direction):
- `utils.calc_direction()`로 공통화되어, `threshold_pct=0`에서도 부동소수점 노이즈로 방향이 튀는 것을 방지합니다.

스냅샷/지표 주의:
- Ichimoku `chikou_span`은 미래 데이터를 참조(lookahead)하므로 snapshot/LLM payload에서 제외됩니다.

LLM 입력(snapshot) 개요:
- LLM provider로 전달되는 입력은 `prediction/pipeline.py`가 만든 snapshot + 오더북 요약이며,
  `prediction/context_builder.py`에서 JSON 형태로 직렬화되어 프롬프트에 포함됩니다.

자세한 입력 payload는 `LLM_INPUT_TABLE.md`를 참고하세요.

---
 
## 문제 해결

### ImportError: No module named 'ebest'

- eBest Live 모드는 `ebest` 래퍼가 필요합니다.
- heuristic-only 실행은 `--no-ebest-live`로 우회하세요.

### Gemini 파싱 오류/`LLM_PARSE_FAIL`

- LLM(Gemini)이 JSON을 끝까지 완결하지 못하면(문자열 줄바꿈/출력 잘림 등) `prediction/llm_judge.py`에서 JSON 파싱 실패가 발생할 수 있습니다.
- `google-genai` 1.x에서는 `GenerateContentConfig` 파라미터명이 camelCase(`responseMimeType`, `systemInstruction`, `maxOutputTokens`)입니다.
  - 본 프로젝트는 버전별로 파라미터명을 감지하여 가능하면 `responseMimeType="application/json"`(또는 구버전 `response_mime_type`)을 설정해 JSON 응답을 강제합니다.
- 의존성은 **`main.py`를 실행하는 동일한 Python 환경**에 설치되어야 합니다.
  - 버전 확인 예시:
    - `python -c "import google.genai as gg; print(getattr(gg,'__version__',None))"`

### LLM 예측이 조용히 fallback 되는 것 같을 때

- LLM 호출/결과 생성 단계에서 예외가 발생하면 경고 로그가 남습니다.
  - `[LLM_PREDICT_FAIL] ...`

### 분봉 데이터 부족(insufficient_minutes)

- 예측은 최소 분봉이 필요합니다.
- 기본값은 `constants.MIN_MINUTE_BARS_REQUIRED`이며, `config.json`의 `min_minute_bars_required`로 변경할 수 있습니다.
- Live 모드로 일정 시간 틱을 수집한 뒤 예측을 실행하세요.

---
 
## 문서

- `ARCHITECTURE.md`: 모듈 아키텍처/데이터 흐름
- `ADAPTIVE_INDICATOR_GUIDE.md`: 적응형 지표 통합 가이드
- `ML_PREDICTION_GUIDE.md`: 머신러닝 예측 시스템 통합 가이드
- `TFT_DUAL_MODEL_DESIGN_GUIDE.md`: TFT 듀얼 모델 설계 가이드
- `docs/runtime/README.md`: 런타임(실시간 예측) 핵심 모듈 함수/클래스 레퍼런스
- `docs/training/README.md`: 오프라인 데이터셋 생성/학습 경로 함수/클래스 레퍼런스

## 테스트

- pytest 전체 실행: `python -m pytest -q`
- 전체 스모크(우산) 테스트: `python -m pytest -q tests/test_smoke.py`
