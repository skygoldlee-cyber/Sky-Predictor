# `telegram_notifier.py` 코드 리뷰

리뷰 기준: `docs/runtime/telegram.md` 사양 + `main.py` / `config.py` 실제 연동 코드 교차 검증

---

## 버그 (즉시 수정 필요)

### BUG-1 · PCR 옵션 라인 이중 이스케이프

**파일**: `telegram_notifier.py` — `format_prediction_message()` 옵션 요약 블록

```python
# 현재 코드
parts.append(f"PCR\\(V\\): {pcr_v:.2f}")   # 이미 \\( 로 이스케이프
...
lines.append(f"📐 *옵션*: {esc('  |  '.join(parts))}")  # esc()가 또 처리
```

`"PCR\\(V\\): 0.82"` 문자열이 `esc()`를 다시 통과하면 `\\` → `\\\\`, `(` → `\\(` 가 중첩되어 텔레그램에 `PCR\\\(V\\\): 0\.82` 로 표시됩니다.

**수정**: `parts`에 이미 이스케이프된 문자열을 넣고 있으므로 `esc()` 를 제거하거나, 반대로 `parts`에서 수동 이스케이프를 제거하고 `esc()`에 맡깁니다.

```python
# 수정안 A — parts를 raw 값으로 구성하고 esc() 에게 위임
if pcr_v is not None:
    parts.append(f"PCR(V): {pcr_v:.2f}")
if pcr_oi is not None:
    parts.append(f"PCR(OI): {pcr_oi:.2f}")
lines.append(f"📐 *옵션*: {esc('  |  '.join(parts))}")  # esc()가 ( ) 처리
```

---

### BUG-2 · `format_error_message` — `message` 필드 MarkdownV2 이스케이프 누락

**파일**: `telegram_notifier.py` — `format_error_message()`

```python
# 현재 코드
f"{message[:300]}"   # 이스케이프 없음
```

`message`에 `.`, `-`, `(`, `)` 등이 포함되면(예: `"분봉 데이터 부족 (현재: 5개, 필요: 20개)"`) MarkdownV2 파싱 에러가 발생하고 `_send_message_plain()` 폴백으로 전송됩니다. 매번 폴백이 트리거되므로 로그 노이즈 및 이중 API 호출이 발생합니다.

**수정**:

```python
def format_error_message(result: Dict[str, Any]) -> str:
    def esc(s: str) -> str:
        for ch in r"\_*[]()~`>#+-=|{}.!":
            s = s.replace(ch, f"\\{ch}")
        return s

    error = result.get("error", "unknown")
    message = str(result.get("message", ""))
    now = datetime.now().strftime("%H:%M:%S")
    return (
        f"🚨 *예측 오류* \\| {now}\n"
        f"`{esc(error)}`\n"
        f"{esc(message[:300])}"
    )
```

---

### BUG-3 · GUI `_run_pipeline`에서 텔레그램 브리지가 항상 누락

**파일**: `main.py` — `async def _run_pipeline()`

CLI 경로는 `config.telegram.enabled` + `create_notifier_from_config()` + `PipelineTelegramBridge` 생성 코드가 모두 구현되어 있습니다. 반면 GUI의 `_run_pipeline()`은 `bridge = None` 한 줄만 있고 이후 브리지 생성 코드가 없습니다. 즉, **GUI에서는 `Enable Telegram` 체크박스가 있지만 아무 동작도 하지 않습니다.**

문서 사양:
> GUI: `Enable Telegram` 체크박스가 켜져 있으면 텔레그램 브리지를 시작합니다.
> 체크박스가 꺼져 있더라도 `telegram.enabled=true`이면 시작합니다.

**수정**: GUI `_run_pipeline()` 내 `predictor =` 생성 이후에 아래 블록 추가

```python
# predictor = PredictionPipeline(...) 이후
try:
    tg_enabled = (
        telegram_enable_chk.isChecked()
        or bool(getattr(getattr(config, "telegram", None), "enabled", False))
    )
    if tg_enabled:
        cfg_path_obj = Path(str(cfg_path or "config.json"))
        secrets = (
            os.environ.get("APP_SECRETS_CONFIG")
            or str(cfg_path_obj.parent / "config.secrets.json")
        )
        notifier = create_notifier_from_config(secrets)
        if notifier._token and notifier._chat_id:
            bridge = PipelineTelegramBridge(
                predictor, notifier,
                predict_interval_sec=60,
                only_consensus=True,
            )
            bridge.start()
            bridge.start_polling()
        else:
            _append_log("[TELEGRAM] 토큰/채팅ID 미설정 — 브리지 생략")
except Exception as _e:
    _append_log(f"[TELEGRAM] bridge init failed: {_e}")
```

---

## 설계 문제 (수정 권장)

### DESIGN-1 · `create_bridge_from_config`가 `main.py`에서 사용되지 않음

`telegram_notifier.py`에 `create_bridge_from_config()`가 정의되어 있고 docstring에 `main.py`에서 쓰는 예시까지 있지만, 실제 `main.py`에서는 `create_notifier_from_config()` + `PipelineTelegramBridge()` 직접 생성 방식을 사용합니다. 함수가 존재하지만 누구도 호출하지 않는 dead code 상태입니다.

**선택지**:

- **A (권장)**: `main.py`를 `create_bridge_from_config()` 사용으로 통일. CLI와 GUI 모두 이 함수 하나로 브리지를 생성하면 중복 로직 제거.
- **B**: `create_bridge_from_config()`를 제거하고 현재 `main.py` 방식을 유지.

---

### DESIGN-2 · `load_telegram_config`가 이전 버전의 `config.json` 읽기 로직을 포함

현재 `telegram_notifier.py`의 `load_telegram_config()`는 `bot_token`/`chat_id`만 반환하며 `config.json`을 직접 읽지 않습니다. 그러나 코드 히스토리를 보면 이전 리뷰 사이클에서 `config.json`의 `telegram.enabled`를 읽는 로직이 추가/삭제를 반복했고, 현재 docstring의 `Returns` 항목에도 `"enabled": bool` 이 남아있을 가능성이 있습니다.

문서 사양:
> `telegram_notifier.py`는 `config.json`을 직접 읽지 않고, secrets/환경변수에서 `bot_token/chat_id`만 로드합니다.

현재 구현은 사양에 맞습니다. 다만 `load_telegram_config` 의 return type이 `Dict[str, Any]`인데 실제로는 `str` 값만 담으므로 `Dict[str, str]`이 더 정확합니다.

---

### DESIGN-3 · `/predict` 명령의 DIR_SUMMARY 포함 여부가 경계 틱 기준

`predict_now()` 는 `/predict` 명령과 주기 루프 모두에서 호출됩니다. 현재 `predict_now()`는 `_is_boundary_tick()`을 내부에서 호출하므로, 사용자가 `/predict` 명령으로 강제 실행해도 **해당 시점이 경계 분이 아니면 DIR_SUMMARY가 포함되지 않습니다.**

이는 사용자 입장에서 혼란스럽습니다. `/predict`는 강제 실행이므로 항상 DIR_SUMMARY를 포함하는 것이 자연스럽습니다.

**수정**: `_handle_command("/predict")` → `predict_now(force=True)` 호출 시 `include_dir_summary=True` 를 명시적으로 전달하도록 `predict_now`에 파라미터 추가

```python
def predict_now(
    self,
    force: bool = False,
    include_dir_summary: Optional[bool] = None,  # None=자동(경계 기준)
) -> Optional[Dict[str, Any]]:
    ...
    _include = include_dir_summary if include_dir_summary is not None else self._is_boundary_tick(result)
    self._notifier.send_prediction(result, force=force, include_dir_summary=_include)
```

`_handle_command` 에서:

```python
if cmd == "/predict":
    self._notifier.send_text("⏳ 예측 중...", parse_mode="HTML")
    self.predict_now(force=True, include_dir_summary=True)  # 강제 실행 시 항상 요약 포함
```

---

## 경고 / 개선 사항

### WARN-1 · `main.py` 이중 `if __name__ == "__main__":` 가드

**파일**: `main.py` 마지막 2줄

```python
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":      # ← 중복
    sys.exit(main())
```

`main()`이 두 번 호출될 수 없지만(두 번째 조건도 동일하므로), 코드 정리가 필요합니다. 병합/복사 시 생긴 중복으로 보입니다. 두 번째 블록을 삭제하세요.

---

### WARN-2 · `TelegramNotifier` 내부 상태(`_token`, `_chat_id`)에 외부 직접 접근

`main.py` 여러 곳에서 `notifier._token`, `notifier._chat_id`를 직접 읽어 `bool()`로 유효성 검사합니다.

```python
token_ok = bool(getattr(notifier, "_token", "")) and bool(getattr(notifier, "_chat_id", ""))
```

Private 필드에 외부에서 직접 접근하는 것은 캡슐화 위반입니다.

**수정**: `TelegramNotifier`에 공개 property 추가

```python
@property
def is_configured(self) -> bool:
    """봇 토큰과 채팅 ID가 모두 설정되어 있으면 True."""
    return bool(self._token) and bool(self._chat_id)
```

`main.py`에서 `token_ok` 조건을 `notifier.is_configured`로 교체.

---

### WARN-3 · `_send_message_plain` 폴백의 `\\` 치환 로직이 불완전

```python
plain = text.replace("\\", "").replace("*", "").replace("`", "").replace("_", "")
```

MarkdownV2 이스케이프 시퀀스 외에도 `~`, `>`, `|`, `{`, `}` 등의 특수문자가 plain 텍스트에 남아있을 수 있고, 백슬래시만 제거하면 한글·숫자에 붙은 `.` 등이 예상치 않게 노출됩니다. 정규식 기반 정리 또는 `html.escape()` 사용 후 `parse_mode="HTML"`로 전환하는 것이 더 견고합니다.

---

### WARN-4 · 폴링 루프에서 `allowed_updates` 직렬화 문제

```python
params: Dict[str, Any] = {"timeout": 1, "allowed_updates": ["message"]}
...
data = self._http_get(self._api_url("getUpdates"), params)
```

`_http_get`은 `urllib.parse.urlencode(params)`를 사용합니다. `allowed_updates`는 리스트인데 `urlencode`는 리스트를 `"['message']"` 형태의 문자열로 직렬화하여 Telegram API가 인식하지 못합니다. Telegram은 이를 무시하고 모든 업데이트를 반환하므로 동작은 하지만 불필요한 `edited_message`, `channel_post` 등 이벤트가 포함될 수 있습니다.

**수정**: `allowed_updates` 를 JSON 문자열로 직렬화하거나, POST body에 포함

```python
# GET params에서는 allowed_updates 제거하고
params = {"timeout": 1}
if self._last_update_id > 0:
    params["offset"] = self._last_update_id + 1
# → 또는 getUpdates를 POST로 전환해 JSON body에 allowed_updates 포함
```

---

## 요약 테이블

| ID | 심각도 | 위치 | 설명 |
|----|--------|------|------|
| BUG-1 | 🔴 버그 | `format_prediction_message` | PCR 옵션 라인 이중 이스케이프 |
| BUG-2 | 🔴 버그 | `format_error_message` | `message` 필드 MarkdownV2 이스케이프 누락 |
| BUG-3 | 🔴 버그 | `main.py` GUI `_run_pipeline` | 텔레그램 브리지가 실제로 시작되지 않음 |
| DESIGN-1 | 🟡 설계 | `create_bridge_from_config` | Dead code — main.py 에서 미사용 |
| DESIGN-2 | 🟡 설계 | `load_telegram_config` | 반환 타입 `Dict[str, Any]` → `Dict[str, str]` |
| DESIGN-3 | 🟡 설계 | `predict_now` + `_handle_command` | `/predict` 명령 시 DIR_SUMMARY 항상 포함 권장 |
| WARN-1 | 🟠 경고 | `main.py` 하단 | `if __name__` 블록 중복 |
| WARN-2 | 🟠 경고 | `main.py` | `_token`/`_chat_id` private 필드 직접 접근 |
| WARN-3 | 🟠 경고 | `_send_message_plain` | 폴백 텍스트 정리 불완전 |
| WARN-4 | 🟠 경고 | `_get_updates` | `allowed_updates` 리스트 urlencode 직렬화 오류 |
