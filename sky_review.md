# SkyPredictor 코드 리뷰 (파트별 상세)

> 대상: `SkyPredictor.zip` (압축 해제 기준 **226개 `.py`, 약 102,000 LOC**, 23개 최상위 모듈)
> 리뷰일: 2026-06-05
> 범위: 아키텍처 · 보안 · 모듈별 구조/품질 · 횡단 이슈 · 우선순위 액션

---

## 0. 요약 (Executive Summary)

전체적으로 **성숙도가 높은 시스템**입니다. Mixin 기반 파이프라인 분해, SSOT 원칙, 광범위한 문서(`docs/`)와 테스트(52개 파일), `pip-compile` 고정 의존성 등 엔지니어링 규율이 잘 잡혀 있습니다. 다만 **자격증명 노출**이라는 치명적 보안 이슈가 이번에도 재현되었고, 일부 **God 모듈**과 **광범위한 예외 삼킴**이 운영 안정성/디버깅의 발목을 잡는 구조적 부채로 남아 있습니다.

| 심각도 | 이슈 | 위치 | 상태 |
|---|---|---|---|
| 🔴 Critical | 라이브 자격증명이 ZIP에 포함 (eBest 거래/OpenAI/Gemini/Telegram) | `config.secrets.json` | **즉시 재발급 필요** |
| 🔴 High | 대시보드가 `0.0.0.0`에 인증 없이 바인딩 | `prediction/web_trade_dashboard.py` | 미해결 |
| 🟠 Medium | 예외를 조용히 삼키는 핸들러 약 **698개** | 전역 (GUI/ebestapi/mixins/telegram 집중) | 이전 지적 잔존 |
| 🟠 Medium | `PredictionPipeline.__init__` 798줄 (God 생성자) | `prediction/pipeline.py` | 미해결 |
| 🟡 Low | 핵심 모듈 내 `print()` 약 330+개 (logging 미사용) | `indicators`(169) 외 | 미해결 |
| 🟡 Low | `config.py`에 secret 마스킹/`__repr__` 보호 부재 | `config/config.py` | 권장 |

---

## 1. 🔴 보안 — 가장 먼저 처리할 항목

### 1-1. 라이브 자격증명 노출 (Critical)

`config.secrets.json` 안에 **실제로 동작하는 운영 자격증명**이 그대로 들어 있습니다. 값은 이 문서에 옮기지 않았으며, 키 종류와 형식만 표시합니다.

| 키 경로 | 종류 | 위험 |
|---|---|---|
| `ai_providers/openai/api_key` | OpenAI 프로젝트 키 (sk- 형식, 164자) | 과금/모델 오남용 |
| `ai_providers/gemini/api_key` | Google API 키 (AIza 형식) | 과금/오남용 |
| `ebest/appkey`, `ebest/appsecretkey` | **eBest 증권 OpenAPI 자격증명** | **실거래 계좌 접근 — 금전 손실 직접 연결** |
| `telegram/bot_token` | Telegram 봇 토큰 | 봇 탈취, 알림/명령 위조 |

**가장 위험한 것은 eBest 키**입니다. 증권사 거래 API 자격증명이므로 노출 시 제3자가 주문을 낼 수 있습니다.

**즉시 조치**
1. eBest, OpenAI, Gemini, Telegram 4종 **전부 폐기 후 재발급**. (ZIP이 한 번이라도 외부로 나갔다면 이미 노출된 것으로 간주)
2. 신규 키는 파일이 아닌 **환경변수**로만 주입 — 코드는 이미 `EBEST_APPKEY`, `OPENAI_API_KEY` 등 env 우선 로딩을 지원합니다(`config/config.py`).
3. 패키징 스크립트에서 `config.secrets.json`을 명시적으로 제외.

> 참고: `.gitignore`는 `config.secrets.json`을 **올바르게 제외**하고 있습니다. 문제는 ZIP 압축이 `.gitignore`를 따르지 않는다는 점입니다. 즉 git 저장소는 안전하지만 **배포/공유용 ZIP을 만드는 경로**가 구멍입니다. `git archive`를 쓰거나, 패키징 시 `--exclude config.secrets.json`을 강제하세요.

### 1-2. 대시보드 무인증 + 전체 인터페이스 바인딩 (High)

`prediction/web_trade_dashboard.py`의 `run_api(host="0.0.0.0", port=8000)`은 인증 없이 거래 요약·활성 포지션·손익·리스크 지표를 노출합니다. 같은 네트워크의 누구나 `/api/summary` 등을 읽을 수 있습니다.

- 기본값을 `127.0.0.1`로 변경하고, 외부 노출이 필요하면 토큰/Basic Auth 또는 리버스 프록시 뒤에 둘 것.

### 1-3. Secret 로깅 방어 부재 (Low–Medium)

`config.py`의 설정 dataclass에 `__repr__` 마스킹이나 로깅 sanitizer가 없습니다. 디버그 로그/예외 traceback에 객체가 찍히면 키가 평문으로 남을 수 있습니다. secret 필드는 `repr=False` 또는 `*** 마스킹 __repr__`를 권장합니다.

---

## 2. 파트별 리뷰

### Part A. 진입점 & 앱 계층 — `main.py`, `app/` (5파일, ~950 LOC)

**구조**: `main.py`는 얇은 진입점으로 잘 분리되어 있습니다. CLI 파싱(`core/cli_args.py`), 로깅(`app_setup.py`), 파이프라인 구성(`pipeline_builder.py`), 실행 모드(`run_modes.py`: test/replay/live/simple)로 책임이 나뉘어 있어 가독성이 좋습니다.

**좋은 점**
- `ZZLogFilter`로 장마감 시 `[ZZ]` 로그 폭주를 억제하는 등 운영 경험이 반영됨.
- `importlib.metadata`로 버전 동적 조회 + 폴백.

**개선**
- `main.py`에 줄바꿈이 `\r\n`(CRLF)로 저장되어 있습니다. 전 파일이 CRLF인지 확인하고 `.gitattributes`로 정규화 권장(혼용 시 diff 노이즈).

### Part B. 설정 — `config/` (3파일, ~2,147 LOC, `config.py`만 1,736 LOC)

**좋은 점**
- `_deep_merge_dict` + `_secrets_paths_to_merge`로 다단계 secret 병합(나중 파일 우선), 환경변수 최우선 오버라이드까지 견고하게 설계됨.
- `config.example.json`을 별도 제공해 온보딩 친화적.

**개선**
- `config.py` 1,736줄은 단일 파일로 과대합니다. (1) secret 로딩/병합, (2) dataclass 스키마, (3) 검증/디폴트 세 영역으로 분할 권장.
- 1-3의 secret 마스킹 적용.

### Part C. 코어 유틸 — `core/` (6파일, ~2,066 LOC)

`cli_args` / `interfaces` / `logging_utils`(681 LOC) / `strike_utils` / `utils`(약 600 LOC). `interfaces.py`로 추상화 경계를 둔 점이 좋습니다. `utils.py`가 잡다한 헬퍼의 집합소가 되지 않도록(파일 600줄) 도메인별로 묶어 관찰 권장.

### Part D. 데이터 처리 — `data/` (5파일, ~2,675 LOC)

- `tick_processor.py`(2,046 LOC)가 핵심이자 **God 모듈 후보**. FC0/FH0/OC0/OH0 처리, 분봉 집계, ATM 옵션 필터, 메모리 정리까지 한 파일에 집중되어 있습니다. 책임별(체결 집계 / 호가 / 옵션 / 메모리관리) 클래스 분리 검토.
- 샘플 CSV(`minute_bars/*.csv`)가 동봉되어 있는데, 운영 데이터라면 패키징에서 제외하는 편이 깔끔합니다.

### Part E. eBest 연동 — `ebestapi/` (5파일, ~5,016 LOC)

- `live.py`(2,722 LOC)가 라이브 루프의 중심. 재연결/세션/실시간 구독을 담당하는 만큼 크지만, 조용히 삼키는 예외가 **109개**로 모듈 중 두 번째로 많습니다. 네트워크/세션 계층의 silent `except`는 "연결이 끊겼는데 끊긴 줄 모르는" 장애로 이어지므로, 최소한 `logger.warning`과 재시도 카운터를 남기세요.
- `print()`는 0개로 로깅 규율이 잘 지켜진 모듈입니다(👍).

### Part F. 기술적 지표 — `indicators/` (23파일, ~17,582 LOC)

- `adaptive_zigzag.py`(**4,187 LOC, 최대 파일**)는 거대하지만 **메서드 분해가 매우 잘 되어 있습니다**: 방향별 처리(`_process_direction_one/zero/minus_one`), ATR/임계값 계산, 멀티 타임프레임 갱신, 이벤트 emit 등이 단일 책임 메서드로 쪼개져 있어 단일 클래스 치고는 추적 가능합니다. 그래도 `SwingPoint`/`ZigZagState`/이벤트 emit 계층을 별도 파일로 추출하면 4천 줄 → 관리 가능한 단위로 나눌 수 있습니다.
- 상수가 `ZigZagConstants` 클래스로 모여 있고 EDGE-CASE 주석으로 의도가 남아 있는 점이 좋습니다(과거 14건 알고리즘 리뷰의 흔적).
- ⚠️ **`print()` 169개** — 지표 모듈 전체에서 가장 많습니다. 백테스트/최적화 스크립트(`optimize_zigzag_lag.py`)의 진행 출력이라면 용인되나, 라이브 경로에서 호출되는 지표 코드의 `print`는 `logger.debug`로 교체 필요(터미널 오염 + 성능).

### Part G. 예측 시스템 — `prediction/` (56파일, ~25,186 LOC) — **가장 큰 파트**

서브 패키지 분해가 모범적입니다: `features/`(7), `models/`(5: transformer/tft/mamba/pivot), `mixins/`(8), `training/`(3), `backtest/`(2), `weights/`.

**핵심 오케스트레이터 `pipeline.py`**
- 🟠 `PredictionPipeline.__init__`가 **122~919행, 약 798줄**이며 내부에 if/for/try 제어 블록이 22개. 생성자가 사실상 "조립 + 부트스트랩 + 콜백 등록"을 모두 수행하는 **God 생성자**입니다.
  - 분리안: `_build_components()`, `_wire_callbacks()`, `_init_state()`로 추출하고 `__init__`은 위임만. 테스트에서 부분 초기화가 가능해지고, 800줄 diff 충돌도 줄어듭니다.
- Mixin 분해(Tick/Option/Adaptive/Prediction/LLM/Guardrail/Amplitude/Feedback)는 책임 경계가 명확해 좋습니다. 다만 mixin 간 `self.<attr>` 암묵 의존이 많아질수록 초기화 순서 결합이 강해집니다 — mixin이 기대하는 속성을 docstring/Protocol로 명시 권장.

**모델 계층**
- Transformer + TFT + Ensemble + (Mamba/PatchTST 실험)으로 다중 아키텍처를 갖춤. `adaptive_ensemble_weights.py` + Brier Score 피드백, `conformal.py`(컨포멀 예측), `calibration_*`(ECE/캘리브레이션), `transformer_quality_tracker.py`(방향 정확도/per-LLM 추적)까지 — **예측 품질 관측 인프라가 인상적입니다**.

**개선**
- `mixins/` 내 silent `except` **95개**. 예측 hot-path에서 조용히 삼키면 "신호가 안 나오는데 에러도 없는" 디버깅 지옥이 됩니다. 최소 `logger.exception` + 메트릭 카운터(`_metrics_inc`)를 남기세요(이미 메트릭 헬퍼가 있으니 활용).
- `predictor.py`(1,450 LOC)도 분할 후보.

### Part H. 매매 실행 — `trading/` (6파일, ~3,785 LOC)

- `gate.py`(1,459 LOC)의 `TradeExecutionGate`는 신호→진입/청산/리버스/일일요약까지 상태기계가 잘 모델링되어 있습니다(`_try_enter`, `_check_close_inner`, `_handle_reverse_signal`, `_calc_dynamic_targets`). `TradeGateConfig.from_dict`로 설정 역직렬화도 분리.
- `position_sizing.py`, `pivot_gate.py`, `state.py` 분리도 적절.
- 실거래 직결 모듈이므로 **테스트가 가장 두꺼워야** 합니다 — `tests/test_trade_gate.py`(1,450 LOC)가 존재하는 점은 매우 좋습니다(👍). 강제청산 시각(`_is_after_force_close`)·동적 타깃 경계값 테스트가 포함됐는지 확인 권장.

### Part I. 텔레그램 — `telegram/` (6파일, ~5,530 LOC)

- `notifier.py`(1,801 LOC) + `bridge.py`(1,407 LOC)가 중심. `PipelineTelegramBridge`가 파이프라인↔알림을 잇고, 백테스트 결과 저장·즉시예측(`predict_now`)·폴링 명령까지 처리.
- `bridge.py:886`의 파일 저장 경로는 **서버 생성 파일명**(`pivot_backtest_{날짜}.json`)과 config 기반 디렉토리를 쓰므로 **path traversal 위험은 없습니다**(과거 지적 사항은 이 경로에선 해소됨). 다만 `history_dir`을 외부 설정에서 받는 만큼, 경로 화이트리스트/`os.path.realpath` 검증을 한 줄 추가하면 방어적입니다.
- silent `except` 85개, `print()` 44개 — 알림 실패가 조용히 묻히면 "알림이 안 오는 줄도 모르는" 상황이 생깁니다. 전송 실패는 반드시 로깅.

### Part J. GUI — `gui/` (34파일, ~13,769 LOC)

- **분해 품질 우수**: `controller.py`(3,180)를 중심으로 `controller_config_reload/logview/market/rt_helpers/startup/ui/window.py`로 관심사를 쪼갰고, `components/`, `engines/`, `renderers/`, `utils/`, `data/` 하위 구조가 깔끔합니다. fplt 렌더러 디버깅 이력(`fplt_renderer.py` 1,779 LOC)이 반영됨.
- ⚠️ **silent `except` 230개 + renderers 27개 = GUI가 전체 1위.** Qt 콜백/페인트 루프에서 예외를 삼키는 관행 때문인데, UI 멈춤/렌더 깨짐의 원인 추적이 어려워집니다. `gui/utils/error_handlers.py`가 이미 있으니 이를 통한 **중앙 집중 예외 처리 + 로깅**으로 수렴시키세요.
- `chart_viewer.py`(3,229)도 분할 후보.

### Part K. 서비스 · 도구 · 스크립트 — `services/`, `tools/`, `scripts/`

- `tools/MD_to_HTML.py`가 3,420줄 단일 파일 — 문서 변환기치고 비대합니다. (단 운영 핵심 경로는 아니므로 우선순위 낮음.)
- `scripts/`(7파일)는 일회성/배치 성격이라 `print` 사용 용인.

### Part L. 테스트 — `tests/` (53파일, ~14,199 LOC)

- **커버리지 폭이 넓습니다**: zigzag(flip/atr_ratio/hybrid/options), prediction smoke·replay·confidence, gui(chart/renderer/controller), trade_gate, adaptive indicator 등. smoke 테스트로 회귀를 빠르게 잡는 전략이 좋습니다.
- 보강 권장: (1) `config.secrets.json` **부재 시**에도 env로 정상 부팅되는지, (2) 대시보드 인증, (3) silent `except`를 logging으로 바꾼 뒤 "에러가 실제로 기록되는지" 검증 테스트.

---

## 3. 횡단(Cross-cutting) 이슈

### 3-1. 광범위한 예외 삼킴 — 약 698개 (이전 ~699 지적 잔존)

전체 `except Exception` 2,147개 중 약 **698개가 `pass` 또는 무로깅으로 조용히 삼킴**. 분포:

| 모듈 | 조용한 핸들러 수 |
|---|---|
| `gui` (+renderers) | ~257 |
| `ebestapi` | 109 |
| `prediction/mixins` | 95 |
| `telegram` | 85 |
| `prediction` (기타) | 58 |
| `indicators` | 24 |

> 권장 패턴: 좁은 예외 타입으로 한정하고, 최소 `logger.warning/exception`를 남기며, hot-path는 메트릭 카운터로 발생 빈도를 관측. "삼켜야만 하는" 경우(예: 선택적 알림 전송)는 `# intentional: optional path` 주석으로 의도를 명시.

### 3-2. `print()` 대 `logging`

핵심 모듈(`indicators` 169, `prediction` 74, `telegram` 44, `trading` 33, `data` 10)에 `print`가 산재. 라이브 경로의 `print`는 (1) 로그 레벨/포맷 우회, (2) 파일 로깅 누락, (3) 성능 저하를 유발. `logger`로 일괄 치환 권장(`ebestapi`처럼 0개가 목표).

### 3-3. God 모듈 / 대형 파일

| 파일 | LOC | 권장 |
|---|---|---|
| `indicators/adaptive_zigzag.py` | 4,187 | 분해 양호, 상태/이벤트 추출 권장 |
| `gui/chart_viewer.py` | 3,229 | 뷰/상태/이벤트 분리 |
| `gui/controller.py` | 3,180 | 추가 분할 |
| `ebestapi/live.py` | 2,722 | 세션/구독/재연결 분리 |
| `data/tick_processor.py` | 2,046 | 체결/호가/옵션 분리 |
| `config/config.py` | 1,736 | secret/스키마/검증 분리 |
| `prediction/pipeline.py` | 1,185 | **`__init__` 798줄 우선 분해** |

---

## 4. 우선순위 액션 플랜

**즉시 (오늘)**
1. eBest·OpenAI·Gemini·Telegram 자격증명 **전부 재발급**.
2. 패키징에서 `config.secrets.json` 강제 제외(`git archive` 또는 `--exclude`).
3. 대시보드 `host` 기본값 `127.0.0.1`로 변경.

**단기 (이번 주)**
4. `pipeline.py.__init__` → `_build_components/_wire_callbacks/_init_state` 분해.
5. hot-path(`ebestapi`, `prediction/mixins`, `telegram`) silent `except`에 로깅 + 메트릭 우선 주입.
6. config dataclass secret 필드 `repr=False`.

**중기**
7. 핵심 모듈 `print` → `logger` 일괄 치환.
8. 대형 파일 분해(위 표 순서대로).
9. 보안/부팅 회귀 테스트 추가(secret 부재, 대시보드 인증).

---

*본 리뷰는 정적 분석 + 구조 표본 검토 기준입니다. 런타임 동작·실거래 로직의 정확성은 별도 동적 테스트로 검증하시길 권장합니다. 매매·금융 관련 판단은 참고 정보이며, 최종 결정과 책임은 운영자에게 있습니다.*
