# main.py (Runtime Entry)

## 역할

- CLI/GUI 실행 모드 제공
- `config.json` 로드 및 오버라이드 적용
- `PredictionPipeline` 생성 후 라이브(eBest) 또는 리플레이 루프 실행

## LLM warmup (08:45 사전 실행)

- `main.py`는 `PredictionPipeline` 생성 직후 LLM을 1회 warmup 호출합니다.
  - 목적: 09:00 이전(예: 08:45) 실행에서도 LLM 초기화/연결 문제를 조기에 드러내기 위함
  - warmup은 짧은 타임아웃으로 실행되며 실패해도 런타임은 계속 진행됩니다.
  - warmup 결과는 `[LLM_WARMUP] ...` 로그로 출력됩니다.

## LLM 초기화 로그(always-on)

- OpenAI/Gemini provider는 초기화 성공/실패/스킵(키 없음) 여부가 항상 `INFO` 로그로 출력됩니다.
  - 성공 시: `OpenAI client initialized`, `OpenAI supported models ...`, `OpenAI selected model ...`
  - 실패/스킵 시에도 동일한 항목이 `(skipped|not initialized)` 형태로 출력됩니다.

## LLM 프롬프트 덤프(디버그)

- `--dump-llm-prompt` / `--no-dump-llm-prompt`: LLM `user` 프롬프트 문자열을 최초 1회 로그로 덤프합니다.
  - 출력 태그: `[LLM_USER_PROMPT_DUMP] (first occurrence only)`
  - 프롬프트는 길 수 있으며(스냅샷 JSON 포함) 디버그 목적에서만 사용하세요.

## Dual LLM (GPT + Gemini 동시 호출)

- 1회 예측 라운드에서 GPT/Gemini를 각각 호출하여 둘 다의 결과를 출력/로그에 포함할 수 있습니다.
  - 설정 키: `prediction.dual_llm` (bool), `prediction.dual_llm_primary_provider` (`gpt`|`gemini`)
  - CLI: `--dual-llm`, `--dual-llm-primary-provider`
  - `dual_llm_primary_provider`는 최종 `llm_action/risk_level/rationale`에 반영되는 provider를 의미합니다.
  - `dual_llm=true`일 때는 provider별 결과를 모두 보관/출력할 수 있으며(로그/결과 dict), 최종 채택은 primary provider 기준입니다.

## ticks 저장(JSONL/GZ)

- live 모드에서 `--out-ticks`가 지정되면 tick 로그를 로컬 파일(`.jsonl`)로 저장
- `--compress-ticks`(기본값: True)이면 tick 로그를 `.jsonl.gz`로 스트리밍 압축 저장
  - 원본 `.jsonl`을 유지하려면 `--no-compress-ticks`

GUI 모드:

- GUI에서는 tick 저장이 항상 활성화되어 있으며, 출력 경로는 자동 생성됩니다.
- `Summary` 박스에 `Tick file: ...`로 실제 저장 경로가 표시됩니다.

## GUI replay (Pause/Resume)

- GUI의 `Replay` 버튼은 tick 로그(`.jsonl`/`.jsonl.gz`)를 리플레이합니다.
- 리플레이 실행 중에는 버튼이 `Pause/Resume` 토글로 동작합니다.

## GUI replay 파일 선택

- GUI에서 replay 파일 경로를 직접 입력하지 않고 선택할 수 있도록 `Pick replay file` 버튼을 제공합니다.
  - tick 로그 파일(`.jsonl`/`.jsonl.gz`)을 선택하면 GUI의 replay 입력 필드가 해당 경로로 채워집니다.

## GUI replay 옵션(Speed/Max lines)

- `Replay speed`와 `Replay max lines`는 한 줄로 수평 배치됩니다.

## GUI RT 상태표시(`RT:`)

- `Start` 버튼 아래의 라벨(`status_lbl`)은 1초 주기로 실시간 수신 상태를 표시합니다.
  - 기본 표시: `FC0/FH0/OC0/OH0/JIF/IJ_` 수신 누적 카운트
  - 선물 가격: `fut_5m_ago`, `fut_now` (FC0 기반)
  - 옵션 가격: `call_now`, `put_now` (OC0 price 기반)
  - 현물지수(KP200): `spot`, `basis` (IJ_ 또는 IJ 스냅샷 기반)
  - 평가(누적):
    - `DIR=xx.x% (hit/total)`: 5분 방향예측 누적 성공율(평가 완료된 건에 한해 표시)
  - 옵션 의미가(당일 extreme):
    - `SRH=<symbol> H<price>`: 당일 옵션 가격 최고치(High)가 `meaningful_option_levels`의 의미가 레벨과 정확히 일치할 때 표시(예: `B0163800 H2.50`)
    - `SRL=<symbol> L<price>`: 당일 옵션 가격 최저치(Low)가 `meaningful_option_levels`의 의미가 레벨과 정확히 일치할 때 표시

- RT 라인 항목 사이에는 `|` 구분자가 포함되어 가독성을 높입니다.

- 텔레그램 송신/수신 로그도 GUI 로그 뷰에 함께 출력됩니다.
  - `[TG][SEND] ...`, `[TG][RECV] ...`

## GUI Prediction (Minutes/Modes)

- `Minutes`와 `Modes`는 한 줄로 수평 배치됩니다.
- `Minutes`는 `5/10/15`를 빠르게 선택할 수 있습니다.
- 선택이 `(config)`인 경우 `config.json`의 `prediction.minutes`를 읽어 `(effective: Xm)` 형태로 표시합니다.

## GUI Summary (Consensus LED)

- GUI의 `Summary` 박스 우측에는 Consensus LED가 표시됩니다.
  - 색상: BUY=green, SELL=red, HOLD=yellow
  - 중앙 텍스트: votes(`N/3`)

## Heuristic flip 즉시 예측/알림

- live 모드에서는 기본적으로 `prediction.minutes`(예: 5분) 경계에서만 예측이 수행됩니다.
- 다만 **Heuristic action이 BUY↔SELL로 flip** 되는 경우에는 5분 경계가 아니어도 즉시 LLM 포함 예측을 수행하고,
  GUI 메시지 강조 및 Telegram 즉시 전송을 트리거합니다.

### Heuristic action 계산 규칙(현재)

- `PredictionPipeline`의 `model_outputs["heuristic"]["action"]`은 **AST + AZZ 합성 규칙**으로 결정됩니다.
- 개념적으로:
  - AST 방향(`ast_signal` 우선, 없으면 `ast_direction`)으로 **BUY/SELL 방향 후보**를 정하고
  - AZZ에서 **신규 스윙 확정(`azz_new_swing`)** 이 발생했고, 그 방향이 AST 방향과 **일치**할 때만 `BUY` 또는 `SELL`
  - 위 조건이 아니면 `HOLD`

즉, flip은 "AST 방향이 바뀜"만으로 발생하지 않고,
**AZZ 신규 스윙 확정이 동반되어 Heuristic action이 BUY↔SELL로 실제 전환되는 시점**에 발생합니다.

### live 루프에서 flip 감지 및 즉시 예측 트리거

- `ebest_live.py`에서 오프바운더리(예측 분 경계가 아님)에서도 adaptive bundle을 계산하여 heuristic action을 확인합니다.
- 이전 heuristic action과 현재 action이 `BUY↔SELL`로 바뀌면 아래 동작을 수행합니다.
  - **마커 로그 출력**: `[HEUR_FLIP_TRIGGER] <prev> -> <cur> (off-boundary)`
  - 다음 예측 시각을 `now`로 당겨 **즉시 예측이 실행**되도록 스케줄링

### GUI/Telegram 동작

- GUI는 `[HEUR_FLIP_TRIGGER]` 로그 라인을 감지하면 로그 뷰에서 **강조 표시(주황색/bold)** 합니다.
- 동시에 `PipelineTelegramBridge.predict_now(force=True, include_dir_summary=True)`를 비동기로 호출하여
  **즉시 예측 + Telegram 전송(DIR_SUMMARY 포함)** 을 수행합니다.
- 중복 전송을 막기 위해 GUI 측에서는 flip 트리거 Telegram 전송에 **짧은 디바운스**가 적용됩니다.

## GUI 테마(qt-material)

- GUI는 `qt-material` 테마를 적용할 수 있습니다(설치되어 있을 때 자동 적용).
- 입력 위젯(`QLineEdit` 등)의 글자색은 어두운 테마에서도 가독성이 유지되도록 런타임에서 보정됩니다.

## GUI 기본 창 크기

- 기본 창 크기: `1100x1200`

## GUI 창 위치(센터링)

- GUI 창은 실행 시 **현재 마우스 커서가 위치한 모니터의 중앙**으로 이동합니다.
- 커서 기반 스크린을 얻지 못하는 경우에는 Qt가 판단한 윈도우 스크린 또는 primary screen을 fallback으로 사용합니다.

## GUI 설정 파일

- GUI는 설정 파일을 선택하는 UI를 제공하지 않으며, 항상 프로젝트 루트의 `config.json`을 사용합니다.

## GUI 버전 표시

- 상단 타이틀바에는 버전을 표시하지 않습니다.
- 버전은 하단 상태 라인의 **우측 끝(시간 표시와 동일 라인 끝)**에 `vX.Y.Z` 형태로 표시됩니다.

## GUI Adaptive indicators

- Adaptive indicator 항목들은 `config.json` 값을 GUI 입력칸에 로드하여 표시합니다.
- GUI에서 수정한 값은 `Start`/`Replay` 실행 시 `config.json`의 `adaptive_indicator` 섹션에 저장됩니다.

## GUI Telegram

- `Enable Telegram`은 기본값이 ON(체크됨)입니다.
- 브릿지 시작 시 `🚀 <b>SkyEbest 예측 시스템 시작</b>`이 1회 전송됩니다.
- 예측 결과에 `error`가 포함된 경우는 텔레그램으로 전송하지 않으며, 내부 로그에만 남습니다.
- JIF 장마감 신호로 라이브 루프가 종료될 때 브릿지가 정지되며 `🛑 <b>SkyEbest 예측 시스템 종료</b>`가 마지막으로 1회 전송됩니다.

## 로그 타임스탬프 포맷

- 로그 타임스탬프는 초 단위까지만 표시됩니다(밀리초 미표시).

## 핵심 함수/클래스

| 이름 | 종류 | 설명 | 주요 I/O |
|---|---|---|---|
| `parse_arguments()` | function | CLI 인자 파싱 | out: `argparse.Namespace` |
| `_make_args_from_gui(...)` | function | GUI 입력값을 args로 변환 | in: GUI 값, out: args |
| `display_startup_info(config, args, logger)` | function | 시작 시 설정/모드 요약 로깅 | in: `AppConfig`, args |
| `run_test_mode()` | function | 테스트 모드 실행 | out: exit code |
| `run_replay_mode_with_predictor(replay_file, predictor, speed, max_lines, pause_event, stop_event)` | function | ticks `.jsonl`/`.jsonl.gz` 리플레이 실행 | in: 파일 경로 |
| `run_simple_prediction(predictor, args)` | function | predictor에 입력을 넣고 1회 예측 수행(헬퍼) | out: result dict |
| `main()` | function | 전체 런타임 엔트리포인트 | out: exit code |
| `_QtLogEmitter` | class | GUI 로그 전달용 시그널 래퍼 | Qt signal |
| `_QtLogHandler` | class | Python logging → GUI로 전달 | logging.Handler |

## 주요 의존

- `config.load_config()`
- `prediction.PredictionPipeline`
- `prediction.weights_selector.select_weights_for_datetime()`
- `ebest_live` 런타임 루프
