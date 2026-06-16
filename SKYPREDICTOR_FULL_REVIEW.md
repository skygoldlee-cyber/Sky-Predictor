# SkyPredictor 파트별 상세 코드 리뷰

> 대상: `SkyPredictor.zip` 전체 (KOSPI200 선물/옵션 예측·자동매매 시스템)
> 규모: Python **약 103,000 LOC** (테스트 제외 ~89K), 17개 상위 패키지
> 리뷰 방식: 디렉토리/클래스 구조 스캔 + 횡단 정적 분석 + 핵심 파트 표적 정독
> 리뷰 일자: 2026-06-16

---

## 0. 개요

| 파트 | LOC | 파일 | 역할 |
|---|---:|---:|---|
| `prediction/` | 26,108 | 57 | Transformer/TFT/Mamba + LLM 앙상블, 피봇 ML, 파이프라인 |
| `indicators/` | 17,573 | 23 | AdaptiveZigZag, HybridAdaptivePivot 등 지표 |
| `tests/` | 14,205 | 53 | 테스트 스위트 |
| `gui/` | 13,769 | 34 | PySide6 대시보드/차트 |
| `telegram/` | 5,530 | 6 | 알림/명령 브리지 |
| `ebestapi/` | 5,032 | 5 | eBest 실시간/주문 API |
| `trading/` | 3,785 | 6 | 매매 게이트/리스크 |
| `tools/` | 3,432 | 1 | MD→HTML 변환 도구 |
| `data/` | 2,675 | 5 | 틱 처리/수집 |
| `config/` | 2,147 | 3 | 설정/시크릿 로딩 |
| `core/` | 2,066 | 6 | 공용 유틸 |
| `training/` | 1,842 | 6 | 모델 학습 스크립트 |
| 기타 | ~5,000 | | scripts, services, app, events, utils, telegram |

스택: PySide6(GUI) · PyTorch(모델) · sqlite3 · requests(eBest/텔레그램) · pandas/numpy.

리뷰 한계 명시: 103K LOC 전체를 라인 단위로 정독하지는 않았습니다. 횡단 정적 분석으로 시스템 전반의 패턴 문제를 정량화하고, 머니패스(주문/리스크)·오케스트레이션·핵심 지표·예측 파트를 표적 정독했습니다. 수치는 실제 grep 결과이며, 개별 `file:line`은 확인된 위치입니다.

---

## 1. 종합 진단

구조는 최근 대규모 리팩터링(Mixin 분해, Protocol 기반 예측기, 파이프라인 분리)으로 **모듈 경계는 양호**합니다. 그러나 운영 안전성을 좌우하는 **횡단 관심사**에서 시스템 전체에 퍼진 문제가 큽니다. 단일 버그보다 "전반에 반복되는 패턴"이 핵심 리스크입니다.

| # | 심각도 | 파트 | 한 줄 요약 |
|---|---|---|---|
| C1 | **P0** | config | 실 API 키(OpenAI/Gemini/eBest/Telegram)가 `config.secrets.json`으로 **배포 ZIP에 동봉** |
| C2 | **P0** | 전역 | `except: pass`류 **무음 예외 689건**, `except Exception` **2,152건** → 장애·주문 실패가 조용히 사라짐 |
| C3 | **P1** | 전역 | 운영 코드에 `print()` **456건** → 로깅 체계 우회, 운영 추적 불가 |
| C4 | **P1** | 전역 | `datetime.now()` **229건** → 백테스트/학습 시 벽시계 혼입(누수·재현불가) |
| C5 | **P1** | prediction/indicators | God 객체: `adaptive_zigzag.py` 4,179 LOC·단일 클래스 99 메서드; `PredictionPipeline.__init__` ~187줄 |
| C6 | **P1** | 전역 | `__del__`에서 자원정리(`pipeline.py`, `llm_judge.py`) → 인터프리터 종료 시 안티패턴 |
| C7 | **P2** | 전역 | 부동소수 동등비교 **191건**, `# type: ignore` 51건, 직렬화 `pickle.load` 5건 |
| C8 | **P0/P1** | prediction(pivot ML) | 별도 리뷰의 수명모델 누수·워크포워드 룩어헤드(이미 수정본 제공) + 잔여 P1들 |

> eval/exec는 14/4건 매칭되나 전부 `model.eval()`·`dialog.exec()`로 **builtin eval/exec 아님(오탐, 위험 없음)**. TLS `verify=False`·`shell=True` 0건(양호).

---

## 2. 횡단 관심사 (가장 우선)

### C1. 시크릿 노출 (P0)
`config.secrets.json`(651B)이 ZIP에 포함되어 있고, 다음 **실키로 보이는 값**이 들어 있습니다(값은 마스킹):

- `ai_providers.openai.api_key` (len 164, `sk…` 형식)
- `ai_providers.gemini.api_key` (len 39, `AI…` 형식)
- `ebest.appkey`(36) / `ebest.appsecretkey`(32)
- `telegram.bot_token`(46) / `telegram.chat_id`(10)

`.gitignore`에는 `config.secrets.json`, `.env.local.ps1`이 등록되어 **깃 추적은 차단**되어 있습니다. 문제는 **공유된 ZIP(배포 산출물)에 그대로 동봉**되었다는 점입니다. 즉 git은 막았으나 패키징/공유 단계에서 누출됩니다.

권장(즉시):
1. 노출된 **모든 키를 폐기·재발급**(OpenAI/Gemini/eBest/Telegram). eBest 앱키는 주문 권한이 있어 특히 위험합니다.
2. 배포/공유용 아카이브 생성 시 시크릿·`logs/`·`data/*.db`·`*.pkl` 제외(`git archive` 또는 명시적 exclude 사용, `zip -x`).
3. 런타임은 환경변수/시크릿 매니저에서만 로드하고, 저장소·아카이브에는 `config.secrets.example.json`(빈 값)만 둘 것.

### C2. 무음 예외 처리 (P0)
`except … : pass` 형태 **689건**, 광역 `except Exception` **2,152건**. 트레이딩 시스템에서 이는 가장 위험한 패턴입니다. 주문 전송 실패, API 타임아웃, 데이터 결손이 로그도 없이 정상처럼 진행될 수 있습니다. 예: `indicators/adaptive_param_engine.py`의 `compute`가 모든 예외를 흡수하고 중립값 반환(레짐 적응이 조용히 꺼져도 인지 불가).

권장: (a) 머니패스(주문/체결/리스크)에서 광역 except 금지 — 구체 예외만, 그 외는 전파. (b) 불가피한 흡수는 반드시 `logger.exception()` + 카운터/헬스플래그. (c) 자동화 가능: `except Exception: pass`를 린트 규칙(`flake8-bugbear B902`, `tryceratops`)으로 CI 차단.

### C3. print → 로깅 (P1)
운영 경로에 `print()` 456건. 운영 환경에서 표준출력은 유실되며 레벨/타임스탬프/모듈 추적이 없습니다. 로거로 일괄 치환하고, 라이브러리 모듈에서는 `logging.getLogger(__name__)` 사용.

### C4. 시간 처리 (P1)
`datetime.now()/utcnow()` 229건. 백테스트·세션 집계·워크포워드에서 벽시계가 혼입되면 (1) 과거 재생 시 모든 레코드가 "오늘"이 되고(피봇 리뷰에서 collector 확인), (2) 워크포워드 룩어헤드(별도 수정본 제공)가 발생합니다. 권장: "현재 시각"을 **주입 가능한 클록**(`now_fn` 또는 `as_of`)으로 추상화하고, 백테스트/학습 경로에서는 데이터의 봉 시각을 사용.

### C5. God 객체 (P1)
- `indicators/adaptive_zigzag.py` — 4,179 LOC, **단일 클래스 99 메서드**. 변경 위험·테스트 곤란.
- `prediction/pipeline.py` `PredictionPipeline.__init__` ~187줄(이후 `_init_parameters` 245줄 등). Mixin으로 잘게 나눴으나 초기화 자체가 거대.
- `gui/chart_viewer.py`(3,229), `gui/controller.py`(3,180), `ebestapi/live.py`(2,738).
권장: 책임 단위(상태/지표계산/세션/렌더) 분리, 초기화는 빌더/팩토리로 위임.

### C6. `__del__` 자원정리 (P1)
`prediction/pipeline.py`, `prediction/llm_judge.py`가 `__del__`에서 정리 수행. `__del__`은 호출 시점 비결정적이고 인터프리터 종료 중 예외가 무시되며 순환참조 시 호출 안 될 수 있습니다. 권장: 명시적 `close()` + 컨텍스트 매니저(`__enter__/__exit__`), `__del__`은 최후 보루로만.

### C7. 기타
- 부동소수 동등비교 191건(`== 0.0` 등) — 지표 임계 판정에서 오작동 위험. 허용오차(`math.isclose`) 사용.
- `pickle.load` 5건 — 모두 로컬 피봇 데이터셋(`.pkl`) 로드. 네트워크 입력은 아니나, 변조 가능 경로면 위험. 신뢰 디렉토리로 제한 또는 안전 포맷(parquet/npz).
- `# type: ignore` 51건 — 타입 회피 누적. mypy 점진 강화.

---

## 3. 파트별 상세 리뷰

### 3.1 진입/부트스트랩 (`main.py`, `app/`, `core/`)
- `main.py`(89줄)는 얇은 진입점 — 양호.
- `app/`(950 LOC)·`core/`(2,066 LOC)는 공용 유틸/부트스트랩. 횡단 이슈(예외/로깅) 동일 적용 대상.
- 점검 권장: 시작 시 시크릿/설정 검증(필수 키 누락 시 조기 실패), 모델·DB·API 핸들의 단일 소유/종료 순서 보장.

### 3.2 설정·시크릿 (`config/`, 2,147 LOC)
- `config/config.py` 1,736 LOC — 설정 로딩 단일 거대 모듈. C1(시크릿)·C5(거대) 직접 해당.
- 권장: 설정 스키마 검증(pydantic 등)으로 타입/필수값 강제, 시크릿 로딩과 일반 설정 분리, 환경별(.env) 오버라이드 일원화.

### 3.3 데이터 계층 (`data/`, `ebestapi/`)
- `data/tick_processor.py`(2,046) — 틱→봉 변환/스냅샷. 실시간 경로라 예외 무음·`print`·부동소수 비교 위험이 집중되는 곳. 표적 점검 권장.
- `ebestapi/live.py`(2,738) — 실시간 구독·이벤트 훅(`_candidate_hook`, `_handle_zz_confirm` 등) 중심. 머니패스라 C2(무음 예외) 0-허용 정책을 최우선 적용. 재연결/레이트리밋/부분체결 처리의 명시적 상태기계 권장.
- 네트워크 호출에 타임아웃·재시도(지수 백오프)·idempotency 키가 일관 적용되는지 확인.

### 3.4 지표 (`indicators/`, 17,573 LOC)
- **`adaptive_zigzag.py`(4,179, 99메서드)** — 최대 God 객체. 인과성/룩어헤드는 별도 SSOT 리팩터 이력이 있으나, 규모 자체가 리스크. 분해 + 골든 회귀 테스트 고정 권장.
- **`hybrid_adaptive_pivot.py`(HAP)** — 별도 피봇 리뷰에서 검출 코어(`_run_logic/_process_pending/_confirm_pivot`)는 인과성 양호로 확인. 초기 방향 피봇(`_init_direction`)이 `_register_candidate`를 거치지 않는 점은 회계 시 주의(워크포워드 evaluator에서 보정함).
- `adaptive_param_engine.py` `compute`의 전체 예외 흡수(C2) → 최소 경고/헬스 노출.
- `pivot_score_integrator.py` — **6레이어 중 HAP 점수가 결과/피처에서 누락**(`ps_hap_score` 없음), `ps_active_layers`를 6이 아닌 5로 정규화(P1, 피봇 리뷰 상세).

### 3.5 예측 (`prediction/`, 26,108 LOC)
구조는 가장 잘 잡혀 있습니다. `Protocol`(`NumericPredictor`, `PredictionResult`) 기반으로 `TransformerPredictor`/`TFTPredictor`/`RuleBasedPredictor`/`EnsemblePredictor`를 다형 처리하고, 룰베이스 폴백 경고까지 둡니다.

발견:
- `EnsemblePredictor.predict`가 Transformer→Mamba→TFT→LLM을 적응 가중으로 결합. torch 미가용 시 룰베이스로 degrade하며 경고(양호). 다만 가중 결합·`disagreement_hold` 임계가 하드코딩/분산되어 있어 config화·캘리브레이션 권장.
- `predictor.py`에 172줄·102줄 `__init__`(God 생성자). 모델 로딩/검증을 팩토리로 분리 권장.
- **피봇 ML 서브시스템(별도 리뷰의 P0):** ① 수명 모델 타깃 누수(전체 시퀀스→전체 수명)·후방 패딩 readout, ② 워크포워드 룩어헤드(`as_of_date` 미사용), ③ 워크포워드 surrogate 자기충족 — **세 건 모두 수정본 파일을 별도 제공**(`pivot_models.py`, `train_pivot_lifespan.py`, `pivot_lifespan_inference.py`, `pivot_parameter_db.py`, `pivot_walkforward_eval.py`). 잔여 P1(분류기·회귀기 앙상블 비다양성, `query_best_parameters` bare-column, 백테스트 Close-only 체결)은 미수정 상태로 후속 권장.

### 3.6 트레이딩/리스크 (`trading/`, 3,785 LOC)
- `trading/gate.py`(1,459) `TradeExecutionGate` — 진입/청산/메시지/명령으로 책임 분해는 양호. 단 `_try_enter`(L642–900, ~258줄), `_check_close_inner`, `_execute_close`가 거대 메서드.
- 머니패스이므로: (a) 광역 except 금지(C2), (b) 동적 타깃/강제청산(`_calc_dynamic_targets`, `_is_after_force_close`) 경계조건 단위테스트 강화, (c) 일별 카운트/연속신호 상태(`_update_consecutive`)의 날짜 롤오버를 주입 클록으로 테스트.
- 선물 포지션 사이징은 피봇 리뷰에서 지적한 "주식식 `value/price`" 문제(계약승수·증거금 미반영) 확인 필요 — 도메인 모델 점검 권장(투자자문 아님).

### 3.7 알림 (`telegram/`, 5,530 LOC)
- `telegram/notifier.py`(1,801) — 단일 거대 모듈. 네트워크 호출 타임아웃/재시도/레이트리밋, 토큰 노출(C1) 점검. 알림 실패가 매매 로직을 막지 않도록 격리(비동기 큐) 권장.

### 3.8 GUI (`gui/`, 13,769 LOC)
- `chart_viewer.py`(3,229, 70메서드)·`controller.py`(3,180)·`renderers/fplt_renderer.py`(1,779)·`engines/chart_engine.py`(1,561) — 렌더/상태/컨트롤 결합도가 높음.
- finplot 렌더링은 과거 다수 버그(타임스탬프 epoch-마이크로초, step-mode, 깜빡임) 이력. UI 스레드와 데이터 스레드 경계(QThread/시그널) 명확화, 렌더러를 순수 함수화하여 테스트 가능하게.
- GUI 예외가 전체 앱을 죽이지 않도록 슬롯 단위 가드(로깅 포함) 적용.

### 3.9 도구/스크립트/학습 (`tools/`, `scripts/`, `training/`)
- `tools/MD_to_HTML.py`(3,432, 단일 파일) — 본 시스템과 무관한 보조 도구가 코드베이스 최상위에 큰 비중. 별도 저장소/패키지로 분리 권장(빌드/의존성 오염 방지).
- `training/` 5종 학습 스크립트 — 시계열 분할/시드/체크포인트 메타 일관성 점검(피봇 학습 리뷰와 동일 패턴: `random_split`·`batch_size` 인자 무시·best 체크포인트 메타 누락 가능성).
- `scripts/run_daily_backtest.py` — 세션 지표를 **거래 백테스트 기반**으로 적재(`pivot_confirmation_rate=win_rate` 등). 이는 검출 기하 기반 워크포워드 evaluator와 척도가 다름을 인지하고 사용할 것(피봇 리뷰 주의 참조).

### 3.10 테스트 (`tests/`, 14,205 LOC, 53파일)
- 규모 양호(`test_trade_gate.py` 1,454 등 머니패스 커버). 다만 GUI/지표 위주이며, **누수·시간의존(C4)·동시성(C6) 회귀 테스트**가 약함.
- 권장 추가: (a) 워크포워드 "테스트일 이후 행 제거 시 결과 불변" 가드, (b) 주입 클록으로 날짜 롤오버 테스트, (c) 머니패스 예외 전파 테스트(주문 실패가 삼켜지지 않는지).

---

## 4. 우선순위 액션 플랜

1. **C1 시크릿 폐기·재발급 + 배포 산출물에서 시크릿 제외** — 즉시. (eBest 주문권한 키 최우선)
2. **C2 머니패스 무음 예외 제거** — `ebestapi/`, `trading/`, `data/tick_processor.py`부터. CI 린트로 재발 차단.
3. **피봇 ML P0(별도 수정본) 반영 + 재학습** — 수명 모델은 옛 가중치 폐기, 워크포워드는 real evaluator 연결.
4. **C4 시간 추상화** — 백테스트/학습/세션 집계의 `datetime.now()` 제거, 주입 클록 도입.
5. **C3 로깅 일원화** — `print`→logger, 모듈별 로거.
6. **C5/C6 God 객체·`__del__` 리팩터** — adaptive_zigzag, pipeline 초기화, GUI 렌더러; 명시적 close/컨텍스트 매니저.
7. **C7 정리** — 부동소수 비교, pickle, type-ignore 점진 개선.

1~3을 먼저 처리해야 운영 안전성과 모델 신뢰성이 확보됩니다. 나머지는 유지보수성 개선입니다.

---

## 5. 부록 — 이전 피봇 예측 리뷰와의 연결

본 문서는 시스템 전반의 파트별 리뷰입니다. 피봇 **예측 알고리즘**에 대한 라인 단위 상세와 P0 수정본(수명 모델 누수, 워크포워드 룩어헤드/자기충족, real 모드 evaluator)은 앞서 제공한 별도 산출물에 있습니다. 두 문서는 상호 보완적이며, 액션 플랜의 3번 항목이 그 수정본 반영에 해당합니다.
