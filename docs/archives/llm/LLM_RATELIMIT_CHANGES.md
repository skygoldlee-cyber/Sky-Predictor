# LLM Rate Limit / Fallback 변경 내역

작성일: 2026-03-03

본 문서는 `logs/prediction.log`에서 관측된 아래 이슈들에 대한 **최소 변경(운영 안정성 우선)** 수정 내역을 정리합니다.

- OpenAI/Gemini **429 Too Many Requests** 빈발
- Gemini fallback 모델 호출 중 **404 Not Found** 발생(불필요한 연쇄 호출 유발)

---

## 1) `constants.py`

**파일**
- `c:\Project\Transformer\constants.py`

**수정 목적**
- 429 발생 시 재시도 폭주를 줄이기 위해 **재시도 횟수는 줄이고**, 재시도 간격/백오프를 키워 **천천히 재시도**하도록 조정
- Gemini 404를 유발하는 fallback 모델 시도를 줄이기 위해 fallback 목록을 보수적으로 운영

**변경 사항**
- **GPT 최신/자동 선택용 fallback 추가**
  - `GPT_FALLBACK_MODELS` 추가: `("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4")`

- **API 재시도 파라미터 조정 (429 완화 / 선호안 3)**
  - `API_MAX_RETRIES`: `3` -> `2`
  - `API_RETRY_DELAY_SECONDS`: `1.0` -> `2.0`
  - `API_BACKOFF_MULTIPLIER`: `2.0` -> `3.0`

- **429 대응 쿨다운 상수 추가(향후/운영 옵션)**
  - `LLM_COOLDOWN_SECONDS_ON_429 = 300.0`

- **Gemini fallback 모델 목록 보수화 (404 완화)**
  - `GEMINI_MODEL`을 고정 버전 대신 최신 alias로 전환: `gemini-flash-latest`
  - `GEMINI_FALLBACK_MODELS`에서 `gemini-1.5-*` 계열을 기본 fallback에서 제외
  - 현재 fallback:
    - `("gemini-2.5-flash", "gemini-2.0-flash")`

---

## 2) `prediction/llm_judge.py`

**파일**
- `c:\Project\Transformer\prediction\llm_judge.py`

**수정 목적**
- Gemini 모델명이 변경/권한/리전 문제 등으로 **404 Not Found / invalid model**이 발생할 때, 동일 모델을 반복 시도하며 호출이 증가하는 문제 완화

- OpenAI(GPT)에서도 `models.list()` 결과를 기반으로
  - 지정 모델이 없으면 fallback 모델로 자동 선택
  - model-not-found 계열 에러 발생 시 bad model로 기록 후 fallback 시도

**변경 사항**
- Gemini 호출에서 모델 에러(모델 not found/invalid/permission/unsupported 등)가 감지되면:
  - 해당 모델명을 `self._gemini_bad_models`에 기록
  - 이후 동일 프로세스 lifetime 동안 해당 모델은 **추가 시도에서 스킵**

**효과**
- `gemini-1.5-flash`, `gemini-1.5-pro` 등에서 발생하던 404가 있을 경우
  - 동일 라운드/이후 라운드에서 불필요한 반복 호출을 줄여
  - 429를 간접적으로도 완화

---

## 3) `prediction/pipeline.py` (사용자 수동 수정)

**파일**
- `c:\Project\Transformer\prediction\pipeline.py`

**수정 주체**
- USER가 IDE에서 직접 적용한 변경(diff 기반)

**변경 사항 요약**
- LLM user prompt dump 시:
  - `"출력은 반드시 JSON 단일 객체만"` 문구 변형까지 제거(공백/마침표 유무 포함)
  - 덤프 로그에 user 프롬프트 본문을 `logger.info(_u.strip())`로 출력하도록 강화
- 예측 루프에서 LLM 판단 호출을 `self._run_llm_judgment(...)`로 수행하고:
  - 실패 시 `{error: "llm_failed"}` 형태로 반환
- `get_prediction()` 최종 반환 dict 재구성:
  - `model_outputs`, `llm_*` 필드 포함
  - `llm_raw`가 있을 때만 output에 포함
- 런타임 진단용 메서드 추가:
  - `get_metrics()` : 내부 `_metrics` + (가능시) adaptive weight 정보 반환
  - `reset_adaptive_weights()` : numeric predictor가 지원할 때 adaptive weight 초기화

- 429 발생 시 급변(off-boundary) LLM 예측 스킵:
  - `_run_llm_judgment()`에서 429("Too Many Requests")가 감지되면 `LLM_COOLDOWN_SECONDS_ON_429` 만큼 rate-limit 상태를 기록
  - `get_prediction(off_boundary=True)` 호출(= `HEUR_FLIP_TRIGGER`로 인한 추가 예측)에서는 rate-limit 상태일 때 LLM 호출을 스킵하고 numeric 결과로 `llm_*` 필드를 채움
  - 5분 경계의 정규 스케줄 예측은 그대로 LLM을 시도(= off-boundary만 제한)

---

## 4) 운영 상 권장 적용 순서

- **Step 1**: `constants.py` 재시도 파라미터 튜닝 적용 (현재 반영됨)
- **Step 2**: `llm_judge.py`의 Gemini bad-model 스킵 로직으로 404 연쇄 호출 완화 (현재 반영됨)
- **Step 3**: 여전히 429가 잦다면
  - `dual_llm` 비활성(동시 호출 감소) 또는
  - provider 호출을 조건부(예: confidence 낮을 때만)로 제한하는 정책 고려
