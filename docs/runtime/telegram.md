# telegram_notifier.py (Telegram Integration)

## 역할

- `PredictionPipeline.get_prediction()` 결과를 텔레그램으로 전송
- 텔레그램 명령을 폴링으로 수신하여 런타임을 제어
- `main.py`에서 CLI/GUI 실행 시 `PipelineTelegramBridge`로 자동 연동 가능

## 설정

### 1) 활성화 스위치

- 텔레그램 활성화 여부는 **엔트리포인트(`main.py`)에서 결정**합니다.
  - CLI: `config.json`의 `telegram.enabled=true`일 때 텔레그램 브리지를 시작합니다.
  - GUI: `Enable Telegram` 체크박스가 켜져 있으면 텔레그램 브리지를 시작합니다.
    - 체크박스가 꺼져 있더라도 `telegram.enabled=true`이면 시작합니다.
    - GUI에서 `Enable Telegram`은 기본값이 ON(체크됨)입니다.

`config.json` 예시:

```json
{
  "telegram": {
    "enabled": true
  }
}
```

### 2) Secrets(봇 토큰 / 채팅 ID)

우선순위:

1. 환경변수
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
2. `config.secrets.json`
   - 기본 경로: `config.json`과 같은 폴더의 `config.secrets.json`
   - 환경변수 `APP_SECRETS_CONFIG`로 경로 오버라이드 가능

`telegram_notifier.py`는 `config.json`을 직접 읽지 않고, 위 secrets/환경변수에서 `bot_token/chat_id`만 로드합니다.

`config.secrets.json` 예시:

```json
{
  "telegram": {
    "bot_token": "...",
    "chat_id": "..."
  }
}
```

## 런타임 동작

### 브리지: `PipelineTelegramBridge`

- `predict_interval_sec` 주기로 `pipeline.get_prediction()`을 호출
- `only_actionable=true`면 `HOLD`는 전송하지 않음
- `only_consensus=true`면 `consensus=true`인 경우에만 전송

- 예측 결과에 `error`가 포함된 경우(예: `insufficient_minutes`)는 텔레그램으로 전송하지 않습니다.
  - 내부 로그에는 `[TG][SUPPRESS] ...` 형태로 남습니다.

### 전송 메시지

- 기본: 예측 결과를 MarkdownV2 포맷으로 전송
- 추가: `prediction_minutes` 경계(예: 5분)마다 한 번, `[DIR_SUMMARY]` 요약을 같은 메시지 하단에 함께 포함할 수 있습니다.
  - 구현은 결과 dict의 `model_outputs`에서 `heuristic/gpt/gemini` action을 기반으로 재구성합니다.

- 브릿지 시작 시: `🚀 <b>SkyEbest 예측 시스템 시작</b>` 메시지를 1회 전송합니다.
- 브릿지 종료 시: `🛑 <b>SkyEbest 예측 시스템 종료</b>` 메시지를 1회 전송하며, 폴링도 함께 정지합니다.

### 옵션 마이크로 플로우 해석(경계값)

별도 메시지 `📡 옵션 마이크로 플로우`는 아래 임계값으로 해석 문구를 생성합니다.

- 설정 키 위치: `config.json > telegram`
  - `option_flow_interp_sr_warn`, `option_flow_interp_sr_hot`
  - `option_flow_interp_pt_low`, `option_flow_interp_pt_high`
  - `option_flow_interp_pcr_v_low`, `option_flow_interp_pcr_v_high`
  - `option_flow_interp_pcr_oi_low`, `option_flow_interp_pcr_oi_high`

기본값 기준:

| 지표 | 경계값 | 비교 연산 | 출력 문구 |
|---|---:|---|---|
| `surge_ratio` | `1.50` | `sr >= 1.50` and `< 2.00` | `유입 증가` |
| `surge_ratio` | `2.00` | `sr >= 2.00` | `유입 급증(변동성 확대 경계)` |
| `per_tick_move_pt` | `0.008` | `pt <= 0.008` | `틱당 충격 낮음(흡수 가능)` |
| `per_tick_move_pt` | `0.030` | `pt >= 0.030` | `틱당 충격 큼(얇은 호가/급변 가능)` |

참고:

- 경계값은 **포함(이상/이하)** 으로 판정합니다.
  - 예: `sr=1.50` → `유입 증가`, `sr=2.00` → `유입 급증`
  - 예: `pt=0.008` → `충격 낮음`, `pt=0.030` → `충격 큼`
- PCR 해석은 `pcr_volume` + `pcr_oi`를 함께 사용합니다.
  - `pcr_v >= pcr_v_high` and `pcr_oi >= pcr_oi_high` → `풋 우위(하방/헤지 성향)`
  - `pcr_v <= pcr_v_low` and `pcr_oi <= pcr_oi_low` → `콜 우위(상방 성향)`

### GUI 로그 출력

- 텔레그램 송신/수신 이벤트는 logger에 남으며, GUI의 로그 뷰(`log_view`)에도 출력됩니다.
  - 송신 성공: `[TG][SEND] ...`
  - 수신 텍스트: `[TG][RECV] ...`

## 텔레그램 명령

- `/predict` — 즉시 예측 실행(강제 전송)
- `/status` — 현재 상태/마지막 예측 요약
- `/pause` — 알림 일시정지
- `/resume` — 알림 재개(+ 즉시 1회 예측)
- `/json` — 마지막 예측 결과 JSON 전송(디버그)
- `/help` — 도움말
