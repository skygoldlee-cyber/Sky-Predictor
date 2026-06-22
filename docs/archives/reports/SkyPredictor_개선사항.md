# SkyPredictor 리팩터링 후 개선 사항

> 리팩터링 기준일: 2026-04-25  
> 코드베이스 규모: 95개 소스 파일 / 53,798 LOC / 256 테스트

---

## 1. 아직 큰 파일들 — 추가 분할 대상

| 파일 | LOC | 문제 |
|---|---|---|
| `telegram/notifier.py` | 3,003 | `TelegramNotifier`(~1,100줄) + `PipelineTelegramBridge`(~1,400줄) 두 클래스가 공존 |
| `gui/controller.py` | 2,683 | `run()` 메서드 하나가 2,500줄 이상 — UI 빌드 / 파이프라인 실행 / 이벤트 핸들러 혼재 |
| `ebestapi/live.py` | 2,472 | `run_ebest_live_mode()` 단일 함수가 2,400줄 |
| `config.py` | 1,215 | 17개 dataclass + 로드/검증 로직이 단일 파일 |
| `data/tick_processor.py` | 1,462 | 실시간 틱 처리 + 분봉 빌드 + OHLCV 누적이 혼재 |

### 권장 분할 방향

#### `telegram/notifier.py` → `telegram/` 2파일
```
telegram/
├── notifier.py         ← TelegramNotifier (HTTP 전송 레이어)
└── pipeline_bridge.py  ← PipelineTelegramBridge (파이프라인 연동 브릿지)
```

#### `gui/controller.py` → `gui/` 3파일
```
gui/
├── controller.py          ← GuiController 클래스 껍데기 + _enter_gui_main_loop
├── controller_pipeline.py ← _run_pipeline / _run_replay (파이프라인 실행 로직)
└── controller_build_ui.py ← run() 내 위젯 생성 코드 (현재 ~1,800줄)
```

#### `ebestapi/live.py` → `ebestapi/` 2파일
```
ebestapi/
├── live.py          ← run_ebest_live_mode 진입점 + 상태 머신
└── live_handlers.py ← 틱 수신 / 구독 관리 / OC0 재구독 로직
```

#### `config.py` → `config/` 패키지
```
config/
├── __init__.py         ← load_config, AppConfig (re-export)
├── schema.py           ← 모든 dataclass 정의
├── loader.py           ← JSON 파싱 + 검증 로직
└── defaults.py         ← 상수 기본값
```

---

## 2. 예외 처리 개선 — `except Exception: pass` 남용

현재 전체 1,579개의 `except Exception:` 블록 중 상당수가 `pass` 또는 무의미한 처리.  
특히 심각한 파일:

| 파일 | except 개수 |
|---|---|
| `gui/controller.py` | 204 |
| `ebestapi/live.py` | 168 |
| `telegram/notifier.py` | 90 |
| `ebestapi/callbacks.py` | 61 |

### 문제 패턴
```python
# ❌ 현재: 오류를 삼켜 디버깅 불가
try:
    some_critical_operation()
except Exception:
    pass

# ✅ 권장: 최소한 DEBUG 로그라도 남김
try:
    some_critical_operation()
except Exception as e:
    logger.debug("[ComponentName] 오류 무시: %s", e)
```

### 조치 방향
- UI 레이어: `except Exception` → `except Exception as e: logger.debug(...)` 로 교체
- 비즈니스 로직: 구체적인 예외 타입 명시 (`ValueError`, `KeyError` 등)
- 실시간 피드: 예외 카운터 추가 후 임계치 초과 시 알림

---

## 3. `asyncio.get_event_loop()` deprecated 사용

Python 3.10+에서 `asyncio.get_event_loop()`는 실행 중인 루프가 없으면 `DeprecationWarning` 발생.  
`gui/controller.py`에서 10곳 사용.

```python
# ❌ deprecated (Python 3.10+)
await asyncio.get_event_loop().run_in_executor(None, blocking_fn)

# ✅ 권장
await asyncio.get_running_loop().run_in_executor(None, blocking_fn)
```

---

## 4. 타입 힌트 현대화

| 항목 | 현황 | 권장 |
|---|---|---|
| `Optional[X]` | 418곳 | Python 3.10+: `X \| None` |
| `from __future__ import annotations` | 82개 파일 적용 | 미적용 13개 파일 추가 |
| 반환 타입 없는 함수 | 247개 (`def` 864 - `->` 617) | 점진적 추가 |

```python
# ❌ 현재 (Python 3.9 스타일)
from typing import Optional, Dict, List
def foo(x: Optional[str]) -> Optional[Dict[str, List[int]]]:
    ...

# ✅ 권장 (Python 3.10+ 스타일)
def foo(x: str | None) -> dict[str, list[int]] | None:
    ...
```

---

## 5. 테스트 커버리지 공백

256개 테스트가 있으나 핵심 모듈 다수가 테스트 미보유.

| 모듈 | LOC | 테스트 |
|---|---|---|
| `ebestapi/live.py` | 2,472 | ❌ 없음 |
| `ebestapi/callbacks.py` | 1,097 | ❌ 없음 |
| `telegram/notifier.py` | 3,003 | ❌ 없음 |
| `telegram/bridge.py` | 1,066 | ❌ 없음 |
| `core/utils.py` | 686 | ❌ 없음 |
| `core/logging_utils.py` | 588 | ❌ 없음 |
| `app/run_modes.py` | 316 | ❌ 없음 |
| `trading/gate.py` | 978 | ✅ 있음 (73개) |

### 우선순위 테스트 추가 대상
1. `core/utils.py` — 유틸 함수들 (순수 함수라 테스트 쉬움)
2. `core/logging_utils.py` — `setup_logging` 파라미터 검증
3. `telegram/notifier.py` — 메시지 포맷, 재시도 로직
4. `ebestapi/callbacks.py` — 틱 파싱, OI 업데이트

---

## 6. `pandas` deprecated 패턴 — 510곳

`DataFrame.append()`, `inplace=True`, `pd.Int64Index` 등 pandas 1.x 스타일이 510곳.  
pandas 2.x에서 `FutureWarning` 또는 오류 발생 가능.

```python
# ❌ pandas 1.x (deprecated)
df = df.append(new_row, ignore_index=True)

# ✅ pandas 2.x
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
```

주요 발생 파일: `data/tick_processor.py`, `prediction/data_builder.py`, `prediction/option_features.py`

---

## 7. 스레드 · 비동기 혼용 정리

`threading.Thread` 직접 생성이 27곳.  
`asyncio` 이벤트 루프와 혼용되어 데드락 위험 존재.

### 문제 구조
```
asyncio 루프 (qasync)
  └─ run_in_executor → 스레드풀
  └─ threading.Thread (직접 생성) ← 별도 관리, 종료 보장 안 됨
```

### 권장 방향
- 백그라운드 루프 태스크 → `asyncio.create_task()` 통합
- 불가피한 스레드 → `threading.Thread(daemon=True)` + 레지스트리 관리 (기존 `asyncio task registry` 패턴 확장)
- `telegram/notifier.py`의 폴링 루프: 이미 스레드 기반 → `asyncio.to_thread()` 마이그레이션 고려

---

## 8. `PredictionPipeline` Mixin 상속 깊이

```python
class PredictionPipeline(
    LLMMixin,
    AmplitudeMixin,
    GuardrailMixin,
    FeedbackMixin,
    AdaptiveMixin,
    OptionMixin,
    PredictionMixin,
    TickMixin,
):
```

8개 Mixin 다중 상속 → MRO 복잡도 증가, 메서드 출처 추적 어려움.

### 권장 방향
- **컴포지션 우선**: Mixin → 독립 서비스 클래스로 분리 후 `PredictionPipeline`에 주입
- 예: `LLMMixin` → `LLMJudgeService(pipeline_ref)` 로 분리

```python
# 장기 목표
class PredictionPipeline:
    def __init__(self):
        self._llm = LLMJudgeService(self)
        self._amplitude = AmplitudeAnalyzer(self)
        self._guardrail = GuardrailService(self)
```

---

## 9. 설정값 중복 선언

`"logs/prediction.log"` 문자열이 8곳에 분산:
- `constants.py`: `DEFAULT_LOG_FILE = "logs/prediction.log"` ✅ 단일 소스
- `gui/controller.py`: `QLineEdit("logs/prediction.log")` ← 하드코딩 4곳
- `app/app_setup.py`: `"logs/prediction.log"` 2곳

### 권장
```python
# gui/controller.py
from config import DEFAULT_LOG_FILE
log_file_edit = QLineEdit(DEFAULT_LOG_FILE)
```

---

## 10. 의존성 방향 개선

현재 `prediction/` 이 `data/tick_processor` 를 직접 import:
```
prediction/pipeline.py     → from data.tick_processor import RealTimeTickProcessor
prediction/data_builder.py → from data.tick_processor import RealTimeTickProcessor
```

레이어 규칙 위반 (`prediction` > `data` 여야 하나 직접 결합).

### 권장: 인터페이스 도입
```python
# core/interfaces.py (신규)
from typing import Protocol
class TickDataProvider(Protocol):
    def get_futures_minute_df(self, minutes: int) -> pd.DataFrame: ...
    def get_daily_session_ohlc(self) -> dict: ...

# prediction/pipeline.py
from core.interfaces import TickDataProvider
class PredictionPipeline:
    def __init__(self, tick_provider: TickDataProvider): ...
```

---

## 11. `ebestapi/` 패키지명 — 문서화 필요

외부 eBest SDK(`import ebest`)와 내부 래퍼(`ebestapi/`)가 이름이 달라  
신규 개발자에게 혼란을 줄 수 있음.

### 권장
- `ebestapi/__init__.py` 상단에 명시적 설명 추가:
  ```python
  """
  ebestapi — 내부 eBest API 래퍼 패키지.

  외부 eBest SDK는 `import ebest` (별도 설치 필요).
  본 패키지는 해당 SDK를 감싸는 내부 연동 레이어.
  """
  ```
- `README.md` 또는 `docs/` 에 패키지 구조도 추가

---

## 12. `tools/MD_to_HTML.py` 독립성 강화

3,420 LOC의 독립 도구가 프로젝트에 포함.  
현재 `tools/` 에 위치하나 별도 저장소 또는 독립 패키지로 분리 고려.

---

## 13. 차트 시각화 개선 (2026-04-25)

### 수정 내용

#### 13.1 차트 시간 인덱스 KST 표시
- **파일**: `gui/chart_viewer.py`
- **변경**: x축 시간 포맷을 UTC에서 KST로 변경
- **구현**: `_setup_xaxis_format()` 메서드 추가
```python
def kst_formatter(x, pos):
    kst = datetime.datetime.fromtimestamp(x, datetime.timezone.utc).astimezone(
        datetime.timezone(datetime.timedelta(hours=9))
    ).replace(tzinfo=None)
    return kst.strftime("%H:%M")
```

#### 13.2 십자선 기능 추가
- **파일**: `gui/chart_viewer.py`
- **변경**: 마우스 호버 시 십자선 표시 활성화
- **구현**: finplot 기본 기능 활용

#### 13.3 차트 갱신 주기 설정
- **파일**: `config.json`, `gui/chart_viewer.py`, `gui/controller.py`
- **변경**: 차트 갱신 주기를 config.json에서 설정 가능
- **기본값**: 500ms
```json
"chart": {
  "refresh_ms": 500
}
```

#### 13.4 시가선/종가선 추가
- **파일**: `gui/chart_viewer.py`
- **변경**: 시가선(노란색), 종가선(마젠타색) 별도 표시
- **구현**: `_render_price_lines()` 메서드 추가
```python
시가선: #FFFF00 (노란색)
종가선: #FF00FF (마젠타색)
```

#### 13.5 캔들 색상 설정
- **파일**: `gui/chart_viewer.py`
- **변경**: 양봉/음봉 색상 명시적 설정
- **구현**: 개별 캔들 렌더링으로 변경
```python
양봉: #7CFC00 (lawngreen)
음봉: #FF5252 (빨간색)
```

#### 13.6 지그재그 피봇 선 연결 제거
- **파일**: `gui/chart_viewer.py`
- **변경**: 피봇 간 선 연결 제거, 마커만 표시
- **구현**: `_render_pivots()` 메서드에서 폴리라인 제거

#### 13.7 거래량 차트를 OBV로 변경
- **파일**: `gui/chart_viewer.py`
- **변경**: 거래량 바 차트를 OBV(On Balance Volume)로 변경
- **구현**: `_render_volume()` 메서드 수정
```python
OBV 선: #00FFFF (CYAN)
OBV 제로라인: #FFFF00 (노란색)
```

#### 13.8 LLM 비활성 시 LED 숨김
- **파일**: `gui/controller.py`
- **변경**: LLM 비활성 시 GPT/GEM LED 숨김
- **구현**: `use_llm` 설정 확인 후 `setVisible(False)` 적용

#### 13.9 메인 윈도우 KOSPI 실시간 데이터 표시
- **파일**: `gui/controller_rt_helpers.py`
- **변경**: 메인 윈도우 status_lbl에 KOSPI 지수 표시
- **구현**: `format_rt_status_line()` 메서드 수정
- **불필요 항목 삭제**: 선물 틱 수(FC0, FH0), KOSPI 틱 수(IJ_), 수신 시간(spot_time), 장운영 정보(JIF)

### 최종 차트 구성

#### 상단 차트 (메인)
1. 캔들스틱: OHLC 캔들 (양봉: lawngreen, 음봉: 빨간색)
2. 시가선: 노란색 선
3. 종가선: 마젠타색 선
4. 피봇 마커: 고점/저점/후보 마커 (선 연결 없음)

#### 하단 차트
1. OBV 선: CYAN 색상
2. OBV 제로라인: 노란색 점선

### 메인 윈도우 상태 표시

#### 변경 전
```
RT FC0=... FH0=... OC0(C/P)=... OH0(C/P)=... JIF=... IJ_=... | KOSPI=...@... | fut_5m_ago=... | fut_now=... | call_now=... | put_now=...
```

#### 변경 후
```
RT OC0(C/P)=... OH0(C/P)=... KOSPI=... | fut_5m_ago=... | fut_now=... | call_now=... | put_now=...
```

### 전체 색상 팔레트
- **양봉**: lawngreen (#7CFC00)
- **음봉**: 빨간색 (#FF5252)
- **시가선**: 노란색 (#FFFF00)
- **종가선**: 마젠타색 (#FF00FF)
- **OBV 선**: CYAN (#00FFFF)
- **OBV 제로라인**: 노란색 (#FFFF00)
- **피봇 마커**: 주황색 (#FFA500)
- **미확정 마커**: 노란색 (#FFFF00)
- **배경**: 어두운 검은색 (#0D0D0D)

---

## 14. 최신 개선 완료 사항 (2026-05-03)

### 14.1 Adaptive ZigZag 알고리즘 수정 완료 ✅

**파일**: `indicators/adaptive_zigzag.py`, `gui/chart_viewer.py`, `docs/ADAPTIVE_ZIGZAG_COMPLETE.md`

**수정 내용**:
- [REVIEW-FIX-2] ATR 급변 감지 비대칭 수정: 논리적 일관성 복구 완료
  - ATR 급증 시 임계값 높여 노이즈 억제 (1.3배)
  - ATR 급락 시 임계값 낮춰 민감도 회복 (0.7배)
- [REVIEW-FIX-4] ER 계산 look-ahead 편향 수정: 실시간 환경 보정 완료
  - 현재 봉(미완결) 제외하여 완결봉만 사용
- [REVIEW-FIX-5] 렌더링 캐시 무효화: swing_version 카운터 추가
  - 클러스터링 in-place 갱신 시 버전 증가로 캐시 무효화

**문서화**: `ADAPTIVE_ZIGZAG_COMPLETE.md`로 통합 완료 (알고리즘 설명 + 코드 리뷰)

---

### 14.2 Summary 라벨 비활성화 ✅

**파일**: `gui/controller.py`

**수정 내용**:
- FC0 age 및 피드백 관련 6개 라벨 비활성화
  - FC0 age
  - FB snap
  - FB strict skip
  - FB HOLD skip
  - Weight updates
  - Adaptive weights

**효과**: UI 단순화, 불필요한 정보 노출 제거

---

## 15. 추가 개선 권장사항 (2026-05-03)

### 15.1 ER 공식 재검토 ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`, `docs/ADAPTIVE_ZIGZAG_COMPLETE.md`

**문제**: 현재 ER 공식(`mmin + er*(mmax-mmin)`)이 Perry Kaufman의 KAMA 원리와 반대 방향
- 강한 추세(ER 높음) → 임계값 높음 → 전환 신호 억제
- 횡보(ER 낮음) → 임계값 낮음 → 가짜 신호 과다 발생

**권장**: 백테스트로 두 공식 비교 후 결정
- 현재: `mmin + er*(mmax-mmin)`
- 원래: `mmax - er*(mmax-mmin)`

---

### 15.2 pending 취소 조건 명시화 ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`

**문제**: `_process_pending_confirmation`에서 취소 조건 미명시
- `max_wait_bars` 초과 후 취소 시 pending 가격 처리 방법 불명확
- 취소 시 `_pending_high/_pending_low` 리셋 누락 가능

**권장**: 취소 로직 명시화 및 리셋 처리 추가

---

### 15.3 세션 파라미터 상품별 분리 ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`, `config.py`

**문제**: 세션 파라미터가 KST 고정
- 해외선물(CME, EUREX 등)은 KST 기준 세션이 다름
- 상품별 세션 테이블 분리 필요

**권장**: 상품별 세션 파라미터 구조 도입
```python
session_params = {
    "KP200": { "early_start": "09:00", "early_end": "09:30", ... },
    "ES": { "early_start": "22:30", "early_end": "23:00", ... },
}
```

---

### 15.4 ATR 기간 타임프레임별 튜닝 ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`, `config.py`

**문제**: ATR 기간(14봉)이 해외선물 분봉에 부적합
- 해외선물 1분봉: 하루 1,440봉, 14봉은 14분만 반영
- 권장: 1분봉 60~120봉, 5분봉 24~48봉

**권장**: 타임프레임별 ATR 기간 설정

---

### 15.5 시간 기반 confirmation ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`

**문제**: `confirmation_bars`가 봉 수에 종속
- 1분봉에서 2분, 15분봉에서 30분 의미
- 타임프레임 변경 시 전략 성격 변화

**권장**: 시간(초) 기반 확정 조건 도입
```python
confirmation_seconds: int = 120  # 2분 고정
confirmation_bars = max(1, confirmation_seconds // bar_interval_seconds)
```

---

### 15.6 DER 구현 문서화 ⏳ 대기

**파일**: `indicators/adaptive_zigzag.py`, `docs/ADAPTIVE_ZIGZAG_COMPLETE.md`

**문제**: `_calc_der` 구현이 문서에 불완전
- 음수 반환 시 `abs()` 처리 확인 필요
- 설계 의도 명확화 필요

**권장**: DER 구현 문서화 및 음수 처리 검증

---

## 우선순위 로드맵 (업데이트)

| 우선순위 | 항목 | 예상 공수 | 효과 | 상태 |
|---|---|---|---|---|
| 🔴 즉시 | **10. 의존성 인터페이스** | 1일 | 테스트 용이성 대폭 향상 | ⏳ 대기 |
| 🔴 즉시 | **2. except pass 제거** | 2일 | 운영 디버깅 품질 개선 | ⏳ 대기 |
| 🔴 즉시 | **9. 설정값 중복 제거** | 반일 | 오류 발생 지점 단일화 | ⏳ 대기 |
| 🟡 단기 | **15.2 pending 취소 조건 명시화** | 반일 | 로직 명확화 | ⏳ 대기 |
| 🟡 단기 | **15.1 ER 공식 재검토** | 1일 | 백테스트 검증 | ⏳ 대기 |
| 🟡 단기 | **1. 큰 파일 분할** | 3~5일 | 가독성 · 유지보수성 | ⏳ 대기 |
| 🟡 단기 | **5. 테스트 커버리지** | 3일 | 회귀 방지 | ⏳ 대기 |
| 🟡 단기 | **3. asyncio 현대화** | 1일 | Python 3.12 호환성 | ⏳ 대기 |
| 🟢 중기 | **15.3 세션 파라미터 상품별 분리** | 2일 | 해외선물 대응 | ⏳ 대기 |
| 🟢 중기 | **15.4 ATR 기간 타임프레임별 튜닝** | 1일 | 시장 적합성 확보 | ⏳ 대기 |
| 🟢 중기 | **4. 타입 힌트 현대화** | 1주 | IDE 지원 강화 | ⏳ 대기 |
| 🟢 중기 | **6. pandas 2.x 마이그레이션** | 1주 | 미래 호환성 | ⏳ 대기 |
| 🟢 중기 | **7. 스레드·비동기 통합** | 2주 | 안정성 개선 | ⏳ 대기 |
| 🔵 장기 | **8. Mixin → 컴포지션** | 2~3주 | 아키텍처 개선 | ⏳ 대기 |
| 🔵 장기 | **15.5 시간 기반 confirmation** | 2일 | 타임프레임 독립성 | ⏳ 대기 |
| 🔵 장기 | **15.6 DER 구현 문서화** | 반일 | 설계 의도 명확화 | ⏳ 대기 |

---

## 완료된 개선사항 (2026-05-04 기준)

| 항목 | 완료일 | 비고 |
|---|---|---|
| Adaptive ZigZag ATR 급변 수정 | 2026-05-03 | REVIEW-FIX-2 |
| Adaptive ZigZag ER look-ahead 수정 | 2026-05-03 | REVIEW-FIX-4 |
| Adaptive ZigZag 렌더링 캐시 무효화 | 2026-05-03 | REVIEW-FIX-5 |
| Summary 라벨 비활성화 | 2026-05-03 | FC0 age 등 6개 라벨 |
| 문서 통합 | 2026-05-03 | ADAPTIVE_ZIGZAG_COMPLETE.md |
| GUI 버튼 이모지 추가 | 2026-05-03 | Start, Replay, Reload config, Reset weights |
| 백그라운드 데이터 컴퓨팅 | 2026-05-03 | QThread 사용하여 UI 응답성 개선 |
| 코드 리뷰 수정 (심각) | 2026-05-03 | 5개 항목 수정 |
| 코드 리뷰 수정 (경고) | 2026-05-03 | 2개 항목 수정 |
| 코드 리뷰 수정 (개선) | 2026-05-03 | MAX_BARS 동적 조정, 시계 통일 |
| 거래 이벤트 로그 추가 | 2026-05-03 | 피봇 상태 포함 |
| 코드 리뷰 수정 (심각 v2) | 2026-05-03 | 스레드 안전성, 경쟁 상태, 재진입 방지 |
| 코드 리뷰 수정 (중요 v2) | 2026-05-03 | 메모리 누수 방지, 해시 충돌 수정 |
| 코드 리뷰 수정 (심각 v3) | 2026-05-03 | 빈 배열 처리 통일, 취소 플래그 실제 작동 |
| 코드 리뷰 수정 (중요 v3) | 2026-05-03 | 로그 배치, x_coord 불일치, 타이머 타이밍 |
| 코드 리뷰 수정 (개선 v3) | 2026-05-03 | zlib.adler32, current_price, self._random, 로깅 추가 |
| 코드 리뷰 수정 (버그 v4) | 2026-05-03 | wait() 블로킹, df 검증, 디바운싱, logger 위치 |
| 피봇 마커 정보 표시 버그 수정 | 2026-05-04 | 좌표계 단위 불일치, numpy 인덱싱 오류 수정 |

---

## 16. GUI 개선 사항 (2026-05-03)

### 16.1 버튼 이모지 추가 ✅

**파일**: `gui/controller.py`

**수정 내용**:
- Start 버튼: 🚀 Start
- Replay 버튼: 🔄 Replay
- Reload config 버튼: 🔃 Reload config
- Reset weights 버튼: ↩️ Reset weights

**효과**: UI 시각적 개선, 버튼 식별성 향상

---

### 16.2 백그라운드 데이터 컴퓨팅 구현 ✅

**파일**: `gui/chart_viewer.py`

**문제**: 차트 렌더링 시 데이터 컴퓨팅이 메인 스레드에서 수행되어 UI가 응답하지 않는 문제 발생

**해결**: 데이터 컴퓨팅을 백그라운드 스레드로 이동

**구현**:
- `DataComputeThread` 클래스: `QThread`를 상속받아 시그널을 Thread 자체에 정의
- 스레드 관리: 메인 위젯에서 스레드 인스턴스 추적 및 정리
- 시그널 핸들러: `_on_compute_finished`, `_on_compute_error`
- 스레드 취소: 이미 실행 중인 스레드가 있으면 취소하고 새 스레드 시작
- 안전한 메모리 정리: 완료 후 `deleteLater()` 호출
- force_clear 값 스레드 내부 저장: 스레드별로 force_clear 값을 저장하여 덮어쓰기 방지

**코드 구조**:
```python
class DataComputeThread(QThread):
    finished = Signal(object, object, bool)  # (df, pm, force_clear)
    error = Signal(str, bool)  # (error_message, force_clear)
    
    def __init__(self, compute_func, force_clear: bool):
        super().__init__()
        self._compute_func = compute_func
        self._force_clear = force_clear
    
    def run(self):
        try:
            df, pm = self._compute_func()
            self.finished.emit(df, pm, self._force_clear)
        except Exception as e:
            self.error.emit(str(e), self._force_clear)
```

**효과**: 차트 렌더링 시간이 길어져도 UI 응답성 유지

---

## 17. 코드 리뷰 수정 사항 (2026-05-03)

### 17.1 심각 (버그 / 런타임 오류) 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_build_control_bar 반환 타입 불일치 수정**
   - `_build_pivot_event_log` 호출에 try/except 추가
   - 예외 발생 시에도 ctrl_w 반환 보장

2. **_on_crosshair_moved 캡슐화 위반 수정**
   - `_renderer._x_coords_cache` 대신 `self._x_coords_cache` 사용
   - ChartViewerWidget 내부 캐시로 이동하여 캡슐화 개선

3. **_update_with_virtual_ticks 캐시 키 타입 불일치 수정**
   - 캐시 키를 tuple에서 str로 통일 (`_get_cache_key()` 사용)

4. **_do_refresh_after_clear force_clear 값 덮어쓰기 위험 수정**
   - force_clear 값을 스레드 내부에 저장
   - 시그널에 force_clear 포함하여 전달
   - `_compute_thread_force_clear` 변수 제거

5. **_build_widget _loading_lbl 초기화 누락 수정**
   - `__init__`에서 `self._loading_lbl: Optional[Any] = None` 초기화 추가

---

### 17.2 경고 (잠재적 문제) 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_on_parameter_dialog ImportError 처리 추가**
   - `from gui.parameter_dialog import ParameterDialog`를 try/except로 감싸
   - ImportError 발생 시 로그 출력 후 반환

2. **DataComputeThread 시그니처 문서화**
   - 시그널 파라미터 타입 명시적 문서화
   - `finished = Signal(object, object, bool)  # (df: pd.DataFrame, pm: Optional[Dict[str, Any]], force_clear: bool)`

---

### 17.3 개선 권장 (완료) ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **MAX_BARS 상수 동적 조정**
   - `MAX_BARS`를 클래스 상수에서 인스턴스 변수 `_max_bars`로 변경
   - `set_max_bars(minutes)` 메서드 추가
   - minutes 값에 따라 동적으로 조정:
     - minutes >= 9999: 1000 (장전체, 실제 데이터 길이 411봉 기준)
     - minutes >= 120: minutes + 100 (2시간 이상)
     - 그 외: 500 (기본값)
   - `__init__`와 `_on_range_changed`에서 호출

2. **시계 혼용 통일**
   - 캐시 TTL 비교에 `time.time()` → `time.monotonic()`으로 통일
   - `_is_cache_valid`와 `_update_with_virtual_ticks` 수정

---

## 18. 거래 이벤트 로그 추가 (2026-05-03)

### 18.1 거래 이벤트 로그 UI 추가 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **거래 이벤트 로그 UI 추가**
   - `_trade_event_log` QTextEdit 위젯 추가
   - 최대 높이 150px, 최대 라인 수 100
   - 피봇 이벤트 로그와 유사한 스타일 적용

2. **거래 이벤트 로그 기능**
   - `_add_trade_event_log` 메서드 추가
   - 거래 유형(ENTRY/EXIT), 액션(BUY/SELL), 가격 표시
   - 피봇 상태(확정/미확정/없음) 표시
   - 가장 가까운 피봇 정보 표시

3. **FpltRenderer 콜백 시스템**
   - `set_trade_event_callback` 메서드 추가
   - `_render_trade_markers`에서 콜백 호출
   - 거래 시점의 피봇 상태 자동 감지

**로그 예시**:
```
[10:30:15] 📥 진입 BUY@352.50 | 피봇: ⏳ 미확정 (H@352.00)
[10:35:20] 📤 청산 SELL@353.00 | 피봇: ✅ 확정 (H@352.00)
```

**효과**: 거래 이벤트와 해당 시점의 피봇 상태를 한눈에 확인 가능

---

## 19. 코드 리뷰 수정 사항 v2 (2026-05-03)

### 19.1 심각한 문제 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **스레드 안전성 위반 수정**
   - `DataComputeThread`에 `_stop_requested` 플래그와 `request_stop()` 메서드 추가
   - `_do_refresh_after_clear`에서 `terminate()` 대신 `request_stop()` 사용
   - 최대 2초 대기 후에도 종료되지 않으면 강제 종료
   - 공유 상태 불일치 위험 감소

2. **_toggle_blink와 _render_pivots 경쟁 상태 수정**
   - `_unconf_marker_names`를 `list`에서 `set`으로 변경
   - `_toggle_blink`에 `RuntimeError` 처리 추가 (C++ 객체 삭제 감지)
   - 마커 이름 제거 시 `discard()` 사용
   - 경쟁 상태로 인한 크래시 방지

3. **_render_chart 재진입 방지 로직 수정**
   - `force_clear`일 때만 취소 요청 허용
   - 취소 요청 후 200ms 뒤 재시도 로직 추가
   - 증분 업데이트 요청 무시
   - 렌더링 중복 요청 방지

---

### 19.2 중요한 설계 문제 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_upsert의 setOpacity 남용 수정**
   - 빈 배열일 때 `setOpacity(0.0)` 대신 실제 제거 (`_remove`)
   - 깜빡임 마커는 `_toggle_blink`에서만 `setOpacity` 사용
   - 메모리 누수 방지

2. **_pm_hash 충돌 가능성 수정**
   - `hash()` 대신 `hashlib.sha256` 사용
   - 프로세스 재시작 시 해시 일관성 보장
   - 해시 충돌 가능성 감소

---

### 19.3 개선 권장 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. _build_control_bar 책임 분리 - 로그 영역 분리 ✅ 완료
2. except Exception: pass 로깅 추가 (주요 위치) ✅ 완료
   - _feed_zigzag: ZigZag 초기화 오류, 봉 업데이트 오류
   - _build_pivot_markers: 타임스탬프 매핑 오류, SwingType import 실패, 피봇 마커 처리 오류

---

## 20. 코드 리뷰 수정 사항 v3 (2026-05-03)

### 20.1 심각한 문제 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_upsert 빈 배열 처리 통일**
   - NaN 필터링 후 빈 배열과 처음부터 빈 배열의 처리 통일
   - 모두 `_remove()`로 실제 제거
   - 메모리 누수 완전 방지

2. **_render_chart 취소 플래그 실제 렌더링 중단 구현**
   - FpltRenderer에 `_cancel_check_callback` 추가
   - render 메서드에서 주기적으로 취소 확인
   - 취소 요청 시 렌더링 중단 및 False 반환

---

### 20.2 중요한 설계 문제 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_build_widget 로그 영역 배치 수정**
   - 차트를 위에, 로그를 아래에 배치
   - 차트 표시 영역 확보

2. **_on_crosshair_moved x_coord 불일치 수정**
   - `np.arange(len(df_index))` 대신 `datetime64[ns]` 변환 후 비교
   - finplot 내부 좌표와 일치하도록 수정

3. **_do_refresh_after_clear 타이머 재시작 타이밍 수정**
   - `_delayed`의 finally에서 타이머 재시작 제거
   - `_on_compute_finished`에서만 재시작 (이미 구현됨)
   - 중복 갱신 방지

---

### 20.3 성능 / 코드 품질 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_pm_hash SHA-256 대신 zlib.adler32 사용**
   - SHA-256은 과도한 비용
   - zlib.adler32 사용 (빠르고 충분히 안전)

2. **_check_position_risk current_price 수정**
   - `pos.current_price` 대신 `_get_current_price()` 사용
   - 실제 현재가로 리스크 체크

3. **VirtualTickGenerator self._random 사용**
   - 모듈 레벨 random 대신 self._random 사용
   - 재현성 확보

---

## 21. 코드 리뷰 수정 사항 v4 (2026-05-03)

### 21.1 잠재적 버그 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **DataComputeThread wait() 블로킹 수정**
   - wait(2000) 제거 - UI 블로킹 방지
   - 취소 요청만 하고 종료 대기하지 않음
   - 스레드는 finished 시그널로 자연스럽게 정리됨

2. **_on_compute_finished df 타입 검증 추가**
   - df가 None인지 체크 추가
   - df.empty 체크 전에 df 검증
   - AttributeError 방지

---

### 21.2 성능 / 코드 품질 수정 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:

1. **_on_crosshair_moved 디바운싱 추가**
   - 50ms 디바운싱 타이머 추가
   - 마우스 이동마다 통계 계산 실행 방지
   - QTimer.singleShot 사용

2. **logger 선언 위치 수정**
   - logger를 파일 최상단으로 이동
   - DataComputeThread 클래스 정의 전에 선언
   - 가독성/유지보수 개선

3. **_load_trade_events 파일 I/O 성능 모니터링**
   - 파일 읽기 시간 측정 추가
   - 100ms 이상 걸리면 경고 로그
   - 성능 문제 조기 감지

---

## 22. 피봇 마커 정보 표시 버그 수정 (2026-05-04)

### 22.1 좌표계 단위 불일치 수정 ✅

**파일**: `gui/chart_viewer.py`

**문제**: 마우스를 피봇 마커 근처로 가져가도 우측 상단에 피봇 관련 정보가 표시되지 않음

**원인**: `_do_crosshair_update` 메서드에서 좌표계 단위 불일치
- `mapSceneToView(pos).x()`가 반환하는 값이 datetime64[ns]가 아니라 봉 인덱스(0-based)임
- 이전 코드는 datetime64[ns]로 가정하여 잘못된 변환 수행

**수정 내용**:
- `x_coord`를 그대로 봉 인덱스로 사용하도록 수정
- 범위 검사 추가 (0 미만이면 0, df_index 길이 이상이면 마지막 인덱스)

```python
# 수정 전
x_idx_ns = self._renderer._to_x(df_index).astype(np.float64)
nearest_idx = int(np.argmin(np.abs(x_idx_ns - x_coord)))

# 수정 후
nearest_idx = int(x_coord)
if nearest_idx < 0:
    nearest_idx = 0
if nearest_idx >= len(df_index):
    nearest_idx = len(df_index) - 1
```

---

### 22.2 numpy 인덱싱 오류 수정 ✅

**파일**: `gui/chart_viewer.py`

**문제**: `np.where(mask)[best_match_idx]` 문법 오류

**원인**: `np.where(mask)`는 튜플을 반환하므로 `[best_match_idx]`는 항상 첫 번째 원소(전체 배열)를 반환

**수정 내용**:
- 올바른 인덱싱으로 수정: `np.where(mask)[0][best_match_local]`

```python
# 수정 전
orig_idx = np.where(mask)[best_match_idx]

# 수정 후
orig_indices = np.where(mask)[0]
orig_idx = int(orig_indices[best_match_local])
```

---

### 22.3 디버그 로그 제거 ✅

**파일**: `gui/chart_viewer.py`

**수정 내용**:
- `_do_crosshair_update` 메서드의 디버그 로그 제거
- `logger.info("[ChartViewer] x_coord=%.2f, nearest_idx=%d, df_index len=%d", ...)` 제거
- `logger.info("[ChartViewer] pivot 선택: orig_idx=%d, pivot_idx_v=%s, pivot_price=%.2f", ...)` 제거

**효과**: 로그 출력 최소화, 운영 환경의 로그 부하 감소
