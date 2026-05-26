# telegram_notifier.py (Telegram Integration)

## 역할

- `PredictionPipeline.get_prediction()` 결과를 텔레그램으로 전송
- 텔레그램 명령을 폴링으로 수신하여 런타임을 제어
- `main.py`에서 CLI/GUI 실행 시 `PipelineTelegramBridge`로 자동 연동 가능
- **v4 전용**: 만기주 프리미엄 블리드(선물 상승 + 옵션 수축) 실시간 독립 알림

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

1. 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
2. `config.secrets.json` (기본 경로: `config.json`과 같은 폴더)
   - 환경변수 `APP_SECRETS_CONFIG`로 경로 오버라이드 가능

`telegram_notifier.py`는 `config.json`을 직접 읽지 않고, 위 secrets/환경변수에서 `bot_token/chat_id`만 로드합니다.

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
- 예측 결과에 `error`가 포함된 경우는 전송하지 않음 (내부 로그: `[TG][SUPPRESS] ...`)

### 전송 메시지 종류

| 메시지 종류 | 조건 | 포맷 |
|---|---|---|
| 예측 결과 | 매 `predict_interval_sec` | MarkdownV2 예측 메시지 |
| DIR_SUMMARY | `prediction_minutes` 경계 틱 | 예측 메시지 하단 포함 |
| 🛡 패리티 가드레일 | v3/v4: parity_divergence가 신호 변경 시 | 예측 메시지 내 블록 |
| 💧 블리드 가드레일 | v4: premium_bleed가 신호 변경 시 | 예측 메시지 내 블록 |
| 🔥 프리미엄 블리드 독립 알림 | **v4 전용**: BleedMonitor 감지 | 독립 MarkdownV2 메시지 |
| 시스템 시작/종료 | 브릿지 start()/stop() | `🚀` / `🛑` HTML 메시지 |

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

### v4 전용: 프리미엄 블리드 독립 알림

`option_feature_set=v4`일 때 `start()`가 **BleedMonitor 스레드**를 자동 시작합니다.

**동작 파라미터:**

| 파라미터 | 기본값 | 위치 |
|---|---|---|
| 폴링 주기 | 5초 | `bridge._bleed_monitor_interval_sec` |
| 쿨다운 | 300초 | `notifier._bleed_alert_cooldown_sec` |
| 최소 점수 | 0.3 | `bridge._bleed_min_score` |

**전송 조건 (모두 충족 필요):**
- `dte_weight_norm >= 0.1` (만기 7일 이내)
- `straddle_prev > 0` (직전 틱 데이터 존재)
- `|premium_bleed_score| >= min_score`
- 쿨다운 경과 (`force=True`로 우회 가능)

**독립 알림 예시:**
```
🔥 프리미엄 블리드 알림 | 14:23:07

💰 선물가: 357.50  |  만기: 1일 전
📊 선물 방향: 상승 📈  |  스트래들: 수축 💧
🎯 블리드 스코어: -0.74  (강한 수축)

📉 스트래들: 4.22 → 3.85 (-8.77%)
📈 선물 수익률: +0.3100%
💧 Decay vs Fut: -0.0310
🌊 IV Crush ⬇️: -8.30%

💡 해석: 선물 상승 중 옵션 프리미엄 비정상 수축.
   Theta 급가속 / IV Crush / MM 롤오버 가능.
   → 방향성 신규 진입 자제, 기존 포지션 청산 국면 가능.
```

**로그 태그:**
- 시작: `[TG][BLEED] 프리미엄 블리드 모니터 시작 (간격: 5초)`
- 전송: `[TG][BLEED] 프리미엄 블리드 알림 전송 (score=..., dte_w=...)`
- 억제: `[TG][BLEED] 쿨다운 중 ...` / `score < min_score — 전송 생략`

**쿨다운/점수 런타임 조정:**
```python
notifier._bleed_alert_cooldown_sec = 180.0   # 3분으로 단축
bridge._bleed_min_score = 0.5                # 강한 신호만 전송
```

### GUI 로그 출력

- 송신 성공: `[TG][SEND] ...`
- 수신 텍스트: `[TG][RECV] ...`

## 텔레그램 명령

| 명령 | 동작 | 비고 |
|---|---|---|
| `/predict` | 즉시 예측 실행(강제 전송) | `force=True`, DIR_SUMMARY 포함 |
| `/status` | 현재 상태/마지막 예측 요약 | 상태·신호·확률·총 전송 횟수·**예측 주기** 표시 |
| `/pause` | 알림 일시정지 | `_user_pause_event.set()` |
| `/resume` | 알림 재개(+ 즉시 1회 예측) | `_user_pause_event.clear()` |
| `/interval 300` | 예측 주기 변경 | 단위: 초, 허용 범위 10–3600, 인수 생략 시 현재 주기 조회 |
| `/regime` | 현재 시장 레짐 조회 | 캐시된 마지막 결과 사용, 없으면 즉시 1회 예측 |
| `/reset` | 신호 억제 상태 초기화 | `_last_signal` 클리어 → 다음 예측 무조건 전송, `_last_boundary_minute` 초기화 |
| `/json` | 마지막 예측 결과 JSON 전송(디버그) | 3800자 초과 시 생략 |
| `/help` | 도움말 | |

### 명령 처리 공통 규칙

- 인수가 있는 명령(`/interval`)은 **공백으로 구분**: `/interval 300`
- 장 종료(`market_closed=True`) 중에는 모든 명령이 차단되고 안내 메시지 반환
- 허가되지 않은 `chat_id`의 명령은 무시 (보안: DS-02)

## 주요 클래스/함수 레퍼런스

| 이름 | 종류 | 설명 |
|---|---|---|
| `TelegramNotifier` | class | 텔레그램 봇 전송/폴링 핵심 클래스 |
| `TelegramNotifier.send_prediction(result, *, force, include_dir_summary)` | method | 예측 결과 전송. 신호 변경 시에만 전송(중복 억제) |
| `TelegramNotifier.send_premium_bleed_alert(opt_snap, current_price, ...)` | method | **v4**: 프리미엄 블리드 독립 알림. 쿨다운/점수 필터 내장 |
| `TelegramNotifier.send_text(text, parse_mode)` | method | 임의 텍스트 전송 |
| `PipelineTelegramBridge` | class | PredictionPipeline ↔ TelegramNotifier 연결 브리지 |
| `PipelineTelegramBridge.start()` | method | 예측 루프 시작. v4이면 BleedMonitor 스레드도 자동 시작 |
| `PipelineTelegramBridge.stop()` | method | 예측 루프 + BleedMonitor 정지 |
| `PipelineTelegramBridge._bleed_monitor_loop()` | method | **v4**: 5초 주기 프리미엄 블리드 감시 루프 |
| `format_prediction_message(result, ...)` | function | 예측 결과 dict → MarkdownV2 메시지 포매터 |
| `format_premium_bleed_alert(opt_snap, current_price, ...)` | function | **v4**: 프리미엄 블리드 독립 알림 포매터 |
| `create_notifier_from_config(secrets_path)` | function | secrets 자동 로드 후 TelegramNotifier 생성 |
| `create_bridge_from_config(pipeline, secrets_path)` | function | secrets 자동 로드 후 PipelineTelegramBridge 생성 |

## 설계 문서 참조

- 프리미엄 블리드 설계: [`../premium_bleed_design.md`](../premium_bleed_design.md)
- 패리티 이탈 설계: [`../call_put_parity_divergence_design.md`](../call_put_parity_divergence_design.md)
