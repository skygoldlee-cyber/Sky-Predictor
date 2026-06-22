# Live Run Troubleshooting

이 문서는 실시간 실행(`ebest_live.py`) 중 발생할 수 있는 대표적인 문제와 대응 방법을 정리합니다.

---

## 1) LLM(GPT/Gemini) 응답이 간헐적으로 누락됨

### 증상

- 로그에서 예측 라운드가 도는데 특정 라운드에 `[GEMINI]` 또는 `[GPT]` 블록이 보이지 않음
- `llm_provider`가 `timeout`/`error`로 떨어지거나 Transformer 결과로 fallback

### 원인

- 네트워크/서버 지연으로 인한 타임아웃
- 일시적 API 오류(429/5xx)
- 빈 응답 또는 JSON 파싱 실패

### 개선/대응

- 재시도/백오프가 적용됩니다.
  - `constants.py`:
    - `API_MAX_RETRIES`
    - `API_RETRY_DELAY_SECONDS`
    - `API_BACKOFF_MULTIPLIER`
  - `prediction/pipeline.py`의 provider 호출에서 `gpt`, `gemini`에 대해 재시도 + 지수 backoff를 수행합니다.
- 타임아웃 상향(운영 설정)
  - `prediction/pipeline.py`의 `llm_timeout_sec`를 8초 → 12~15초 등으로 조정하면 누락 빈도가 줄어들 수 있습니다.
- 원인 추적
  - 듀얼 LLM 모드에서는 provider가 실패해도 `model_outputs.gpt` / `model_outputs.gemini`에 `timed_out`, `error`가 기록됩니다.

---

## 2) Gemini 404 NOT_FOUND (모델이 없거나 generateContent 미지원)

### 증상

- 로그에 아래와 유사한 메시지가 포함됨
  - `gemini generate_content failed: 404 NOT_FOUND ... model is not found for API version v1beta, or is not supported for generateContent`

### 원인

- `GEMINI_MODEL`로 설정된 모델이 현재 사용 중인 API 버전/SDK에서
  - 존재하지 않거나
  - `generateContent`를 지원하지 않는 경우

### 대응

- 모델명을 교체하거나 fallback 모델 리스트를 보강합니다.
  - `constants.py`의 `GEMINI_MODEL` / `GEMINI_FALLBACK_MODELS`
- Gemini SDK가 제공하는 모델 목록을 확인합니다.
  - `prediction/llm_judge.py`는 클라이언트 초기화 시 지원 모델을 로그로 남길 수 있습니다.

---

## 3) 로그가 너무 길어서 보기 불편함 (가로 스크롤 발생)

### 증상

- `rationale`/`caution`가 한 줄로 길게 찍혀 우측 스크롤이 필요

### 대응

- 모델 출력 블록(`[GPT]`, `[GEMINI]`, `[PIPELINE]`, `[HEURISTIC]`)은 긴 문자열을 여러 줄로 wrap하여 출력하도록 조정되어 있습니다.
- LLM provider 블록에서는 큰 `raw` 필드를 출력에서 제외하여 로그를 간결하게 유지합니다.

---

## 4) `market_opened=False`가 계속 유지됨 (JIF 장시작 이벤트 미수신)

### 증상

- `[WAIT] ... market_opened=False ...`가 지속됨
- `[JIF_OPEN]` 로그가 보이지 않음
- 08:45 이후 프로그램을 시작한 경우 `JIF`의 "장시작" 전이가 이미 지나가서 수신되지 않는 것이 정상일 수 있음

### 원인

- 런타임 시작 시점이 장시작 이후라 `JIF_OPEN` 전이 이벤트가 발생하지 않음
- 래퍼/환경 차이로 `JIF` realtime tick이 콜백까지 전달되지 않는 경우

### 개선/대응

- 런타임은 `JIF_OPEN`이 없어도 멈추지 않도록 fallback이 동작합니다.
  - realtime tick(FC0/FH0/OC0/OH0/IJ_)이 1회라도 수신되면 게이트를 오픈합니다.
    - 로그: `[GATE_BY_TICK] market_opened=True by realtime tick ...`
  - 추가 안전장치로 장중 시간(09:00~15:45, KST)에는 일정 시간 대기 후 게이트를 오픈할 수 있습니다.
    - 로그: `[GATE_FALLBACK] ...`

---

## 5) `[IJ_REFRESH] ... jisu=0.00`만 반복됨 (현물지수 스냅샷이 0으로만 들어옴)

### 증상

- 로그가 아래처럼 반복됨
  - `[IJ_REFRESH] tr_key=101 jisu=0.00 time=`
- GUI `RT:` 라인에서 `spot`/`basis`가 표시되지 않음(또는 0/None)

### 원인

- eBest wrapper/서버 환경에 따라 `IJ` REST 응답 블록 키/필드명이 달라 파싱이 실패하거나 0으로 들어올 수 있음
- 해당 시점에 `IJ` 스냅샷이 정상값을 제공하지 않는 경우(장외/세션/권한)

### 개선/대응

- `ebest_api.py::_ebest_fetch_ij_snapshot()`은 여러 request/response 형태를 best-effort로 시도하도록 보강되어 있습니다.
- 그래도 0.0이면, 우선 `IJ_` realtime 수신이 있는지(카운트 증가/틱 로그) 확인하고,
  필요 시 `IJ` TR의 key/블록 키를 환경에 맞게 조정해야 합니다.
