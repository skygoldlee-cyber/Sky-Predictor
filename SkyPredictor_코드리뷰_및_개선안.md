# SkyPredictor 전체 코드 리뷰 및 개선안

> 대상: `SkyPredictor.zip` (2026-05-29 기준)
> 규모: Python 파일 **225개 / 약 101,865 LOC** (테스트 53파일·14,236 LOC 포함)
> 검토 방법: 정적 분석(ruff), 구문 컴파일 검사, 의존성·시크릿·테스트 수집 검사, 핵심 모듈 수동 리뷰

---

## 0. 한눈에 보기

| 영역 | 상태 | 핵심 메시지 |
|------|------|------------|
| 🔴 **보안** | **즉시 조치** | 실제 API 키·토큰이 업로드 압축본에 포함됨 → **전수 폐기·재발급 필요** |
| 🟠 **버그/안정성** | 높음 | Python 3.10에서 import 실패하는 모듈 1개, `NameError` 잠복 버그 1개 (광범위 except에 가려짐) |
| 🟠 **예외 처리** | 높음 | `except Exception: pass` 약 **699곳** — 실거래 시스템에서 장애를 침묵시킴 |
| 🟡 **코드 품질** | 중간 | ruff 경고 843건(대부분 자동 수정 가능), 600줄 God-Method 다수 |
| 🟡 **테스트/CI** | 중간 | 리팩터링 후 깨진 테스트 2개(수집 불가), README가 주장하는 CI 미존재 |
| 🟢 **문서/패키징** | 낮음 | 의존성 lock 파일 없음, README 예시 설정 파일 누락 |

**총평** — 모듈 구조 분리, dataclass 기반 설정, optional-dependency 분리 등 **아키텍처 기초는 탄탄**합니다. 다만 (1) 시크릿 노출, (2) 광범위 `except`가 실제 버그를 숨기는 구조, (3) 리팩터링 후 정리되지 않은 잔재(깨진 테스트·죽은 코드)가 가장 큰 리스크입니다. 아래 우선순위대로 처리하면 단기간에 안정성을 크게 끌어올릴 수 있습니다.

---

## 1. 🔴 보안 — 가장 먼저 처리

### 1-1. 실제 시크릿이 압축 파일에 포함됨 (CRITICAL)

`config.secrets.json`에 **빈 값이 아닌 실제 운영 자격증명**이 들어 있습니다.

- OpenAI API Key (`sk-proj-...`)
- Gemini API Key (`AIza...`)
- eBest `appkey` / `appsecretkey`
- Telegram `bot_token` / `chat_id`

`.gitignore`에는 정상적으로 제외되어 있으나, **이번에 업로드된 zip 압축본에는 그대로 포함**되어 보안 경계를 벗어났습니다. eBest 키는 **실거래 주문 권한**과 직결되므로 노출 시 금전적 피해로 이어질 수 있습니다.

**조치 (오늘 안에):**

1. **전 키 폐기·재발급** — OpenAI, Gemini, eBest, Telegram 봇 토큰 모두. (노출된 토큰은 더 이상 신뢰할 수 없음)
2. 배포·전달 시 시크릿 파일이 함께 포장되지 않도록 **패키징/zip 스크립트에서 명시적으로 제외**.
3. 중장기적으로 시크릿을 **환경변수 또는 OS 키체인**으로 이전하고, 파일 기반은 폐기.
4. (선택) Telegram 봇은 재발급 후 `chat_id` 화이트리스트 검증을 명령 핸들러에 추가.

> 이 항목 하나만으로도 본 리뷰에서 가장 시급합니다. 코드 개선보다 우선합니다.

---

## 2. 🟠 버그 / 안정성

### 2-1. `telegram/formatters.py` — Python 3.10/3.11에서 import 실패 (HIGH)

**파일:** `telegram/formatters.py` 라인 **787, 791, 795, 799**

```python
# 787행 (예시)
f"  {_c_arrow} 콜: `{esc(f"{_c_chg:+.1%}")}` "   # f-string 내부에서 같은 따옴표(") 재사용
...
f"  ◆ 콜: 집계 중 \\({esc("open_price")} 미주입\\)"
```

f-string 표현식 안에서 **바깥과 동일한 따옴표를 재사용**하는 문법은 PEP 701로 **Python 3.12부터** 허용됩니다. `pyproject.toml`의 `requires-python = ">=3.10"`을 따르는 3.10/3.11 환경에서는 **`SyntaxError`로 모듈 전체가 import되지 않습니다.**

현재 개발 PC가 3.12이기 때문에 증상이 가려져 있을 뿐입니다(`py_compile` 통과). 알림 포매팅의 핵심 모듈이라 배포 환경이 3.10/3.11이면 텔레그램 알림 전체가 죽습니다.

**수정:** 내부 따옴표를 작은따옴표로 바꾸면 끝.
```python
f"  {_c_arrow} 콜: `{esc(f'{_c_chg:+.1%}')}` "
f"  ◆ 콜: 집계 중 \\({esc('open_price')} 미주입\\)"
```
또는 표현식을 미리 변수로 빼서 f-string 중첩을 없앱니다.

### 2-2. `prediction/data_builder.py` — `adaptive_dict` 미정의 (HIGH, 잠복)

**파일:** `prediction/data_builder.py` 라인 **480, 481, 490, 491, 492, 495**

```python
kospi_zz_s   = _fn(adaptive_dict.get("kospi_zigzag")  or {}, base=zz_s)
futures_zz_s = _fn(adaptive_dict.get("futures_zigzag") or {}, base=zz_s)
...
dual_mode      = bool(adaptive_dict.get("dual_mode", False) or False)
pivot_proximity_alert = adaptive_dict.get("pivot_proximity_alert") or {}
```

`adaptive_dict`는 이 함수 어디에서도 **정의되지 않습니다**(전수 검색 확인). 따라서 이 블록은 첫 참조에서 `NameError`를 던지고, 이를 감싼 `except Exception:`이 **조용히 삼켜 버립니다.** 결과적으로:

- KOSPI/선물 **개별 ZigZag 설정 경로가 항상 죽고**,
- `dual_mode`, `kospi_symbol`, `futures_symbol`, `pivot_proximity_alert` 설정이 **전부 폴백 값으로만 동작**합니다.

config에 dual_mode/proximity_alert를 넣어도 "효과가 없는" 현상이 있었다면 이게 원인일 가능성이 높습니다.

**수정:** 함수가 실제로 받는 dict 변수명으로 교체해야 합니다. 함수 상단에서 `st`, `zz`를 `cfg.adaptive_indicator`에서 뽑는 것으로 보아, 같은 source dict(예: `adaptive` 또는 `adaptive_indicator_dict`)를 의도한 것으로 추정됩니다. 정확한 원본 변수를 확인해 일괄 치환하세요. **이 한 줄 수정으로 dual-mode 기능 전체가 되살아납니다.**

### 2-3. 광범위 예외 처리가 버그를 침묵시킴 (HIGH, 구조적)

비-테스트 코드 기준 `except Exception` 약 **2,138곳**, 그중 **`except Exception: pass` 형태가 약 699곳**입니다.

| 디렉터리 | `except Exception: pass` 수 |
|----------|---:|
| gui | 263 |
| prediction | 166 |
| ebestapi | 109 |
| telegram | 85 |
| indicators | 25 |
| training | 19 |
| core / utils / 기타 | ~32 |

2-2의 버그가 **개발 단계에서 발견되지 않은 이유가 바로 이것**입니다. 실거래 시스템에서 피드 수신·주문·신호 경로의 예외가 `pass`로 사라지면, 시스템이 멈춘 줄도 모르고 "정상 동작 중"으로 보입니다.

**점진적 개선 전략 (전부 한 번에 고칠 필요 없음):**

1. **핫패스 우선** — `ebestapi/`(피드·주문), `trading/`(게이트·포지션), `prediction/`(신호 생성) 의 `except: pass`부터.
2. 최소한 **로깅을 강제**: `except Exception: pass` → `except Exception: log.exception("...")`. 로그만 남겨도 침묵 장애가 사라집니다.
3. **삼켜도 되는 곳**(GUI 위젯 cosmetic 갱신 등)은 의도를 주석으로 명시 → `# best-effort, UI 비핵심`.
4. ruff에 `BLE001`(blind-except) 규칙을 켜서 신규 추가를 차단(아래 5-3).

---

## 3. 🟡 코드 품질

### 3-1. 정적 분석(ruff) 결과 — 843건, 대부분 자동 수정 가능

```
417  F401  unused-import          (미사용 import)
 99  F541  f-string-no-placeholder(플레이스홀더 없는 f"")
 94  F821  undefined-name         (대부분 타입주석, 아래 참고)
 80  F841  unused-variable
 72  E702  multiple-statements-semicolon (한 줄 세미콜론)
 33  E402  module-import-not-at-top
 32  E741  ambiguous-variable-name (l, I 등)
  4        invalid-syntax (= 2-1 항목)
```

- **즉시:** `ruff check . --fix` 로 약 443건 자동 정리(미사용 import/변수 등). 위험도 거의 없음.
- **F821 94건**은 대부분 `from __future__ import annotations`가 있는 파일의 타입 주석(`List`, `QDialog`, `PivotCandidateCollector` 등)이라 **런타임 크래시는 아님**. 다만 타입 체크·가독성을 위해 누락 import를 보강하는 게 좋습니다. (단, `data_builder.py`의 `adaptive_dict`는 주석이 아닌 실코드 → 2-2의 실제 버그)
- **E702 72건**(세미콜론으로 여러 문장): 디버깅 흔적일 가능성. `ruff format` 또는 `black`으로 정리.

### 3-2. God-Method / 초대형 파일 — 분해 필요

| 위치 | 규모 | 메모 |
|------|------|------|
| `indicators/adaptive_zigzag.py` `update()` | **약 595줄** (1245–1840) | 핫패스 단일 메서드. 단계별(스텝 갱신→pending 평가→확정→대안검증)로 분해 권장 |
| 동 파일 `_process_pending_confirmation()` | 약 309줄 (936–1245) | |
| `config/config.py` `_from_dict()` | 약 366줄 (602–968) | 섹션별 파서 함수로 분리 |
| 동 파일 `validate()` | 약 396줄 (1108–1504) | 규칙별 검증 함수 + 결과 누적 패턴 권장 |
| `indicators/adaptive_zigzag.py` (전체) | 3,941줄 | |
| `gui/chart_viewer.py` | 3,218줄 | |
| `gui/controller.py` | 3,158줄 | `docs/TODO.md`에도 분해 과제로 등록되어 있음 |

이미 mixin 분리(예: `prediction/mixins/*`, `gui/controller_*.py`)를 잘 활용 중이므로, **같은 패턴을 `AdaptiveZigZag.update()`와 `config._from_dict/validate`에 적용**하면 됩니다. 핫패스 메서드는 분해 전후 동일성 검증을 위해 회귀 스냅샷 테스트(동일 입력→동일 pivot 시퀀스)를 먼저 두는 것을 권장합니다.

### 3-3. 중복·미정리 모듈 — 정체성 명확화

- **`trade_dashboard.py`가 두 곳**에 존재: `gui/trade_dashboard.py`(323줄) vs `prediction/trade_dashboard.py`(267줄). 이름이 같아 혼동·잘못된 import를 유발합니다. 하나를 `*_view`/`*_state` 등으로 재명명하거나 통합하세요.
- **파라미터 엔진 3종**: `adaptive_param_engine.py`(348), `adaptive_parameter_adjuster.py`(1112), `realtime_parameter_tuner.py`(299). 역할 경계가 문서화되어 있지 않으면 어떤 것이 운영 경로인지 불분명합니다.
- **피봇/전환점 탐지기 다수**: `adaptive_zigzag`, `atr_adaptive_pivot`, `percent_adaptive_pivot`, `hybrid_adaptive_pivot`, `kalman_turning_point`, `market_structure_break`, `fractal_confirmation`, `multi_timeframe_zigzag`. 운영에서 실제로 import되는 것은 사실상 `adaptive_zigzag` 계열입니다. 나머지가 **실험/리서치 코드**라면 `indicators/experimental/` 하위로 모으거나 docstring 상단에 `# STATUS: experimental`을 명시해 운영 코드와 구분하세요.

### 3-4. `main.py`의 전역 부작용 (MEDIUM)

```python
logging.getLogger = getLogger_with_zz_filter   # 표준 라이브러리 함수 전역 교체
```

- 표준 `logging.getLogger`를 모듈 import 시점에 **전역 교체**하는 것은 다른 라이브러리·서드파티 로깅과 충돌할 수 있는 강한 부작용입니다. 필터는 **루트 로거에 1회 부착**하는 방식으로 충분합니다.
- 같은 파일의 `_startup_internet_time_sync()`는 **정의만 되어 있고 호출되지 않습니다**(실제 경로는 `gui/controller_startup.run_startup_internet_time_sync`). **죽은 코드**이므로 제거하세요.

---

## 4. 🟡 테스트 / CI

### 4-1. 리팩터링 후 깨진 테스트 (수집 자체 실패)

`pytest --collect-only` 시 다음이 **import 에러로 수집 불가**:

| 테스트 | 잘못된 import | 올바른 경로 |
|--------|---------------|-------------|
| `tests/test_oi_levels.py` | `from prediction.option_features import ...` | `prediction.features.option_features` |
| `tests/test_rule_based_core.py` | `from prediction.adaptive_mixin import ...` | `prediction.mixins.adaptive_mixin` |

모듈을 패키지로 이동(`features/`, `mixins/`)하면서 **테스트의 import 경로가 갱신되지 않았습니다.** 이 테스트들은 지금 **단 한 번도 실행되지 않고 있습니다.** import 경로만 고치면 됩니다.
(그 외 `torch`/`PySide6` 미설치로 인한 수집 에러는 환경 문제로, optional-dep 설치 시 해소됩니다.)

### 4-2. CI가 문서상으로만 존재

README는 GitHub Actions(`.github/workflows/ci.yml`)와 가이드를 안내하지만, **프로젝트에 `.github` 디렉터리가 없습니다.** "lint + 깨진 테스트"가 자동 검출되지 않는 상태입니다.

**최소 CI 1개만 추가해도 4-1·3-1·2-1이 자동으로 잡힙니다:**

```yaml
# .github/workflows/ci.yml (예시 골격)
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ["3.10", "3.12"] }   # 3.10 매트릭스가 2-1을 잡아줌
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}" }
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q -m "not slow and not integration"
```

> `requires-python = ">=3.10"`을 유지하려면 CI에 3.10을 반드시 포함하세요. 그래야 2-1 같은 버전 의존 문법 오류가 머지 전에 걸립니다. (3.10 지원이 불필요하면 반대로 `>=3.12`로 올리는 것도 방법)

---

## 5. 🟢 문서 / 패키징

### 5-1. README의 설정 예시 파일 누락
README는 `cp config/config.example.json config/config.json` 을 안내하지만 **`config/config.example.json`이 없습니다.** 신규 사용자 셋업이 첫 단계에서 막힙니다. 시크릿을 제거한 예시 파일을 추가하세요.

### 5-2. 의존성 lock 파일 부재
`pyproject.toml`이 `>=` 하한만 지정(`torch>=2.0,<3.0`, `numpy>=1.24` 등)하고 **lock 파일이 없습니다.** 수치 결과가 라이브러리 버전에 민감한 ML·트레이딩 시스템에서 이는 **재현성 리스크**입니다. `pip-tools`(`requirements.lock`) 또는 `uv lock`으로 운영 환경을 고정하세요.

### 5-3. ruff 규칙 확장 (선택)
현재 ruff는 기본 규칙만 활성화로 보입니다. 다음을 켜면 본 리뷰 항목 대부분이 지속적으로 차단됩니다.
```toml
[tool.ruff.lint]
select = ["E","F","W","I","UP","B","BLE","SIM"]
# BLE001: blind-except 차단(2-3), I: import 정렬, UP: 구버전 문법, B: 흔한 버그 패턴
```

### 5-4. 타입 체크 강화 (점진)
`# type: ignore` 47곳, mypy는 `disallow_untyped_defs = false`로 느슨합니다. 핵심 패키지(`indicators`, `prediction`, `trading`)부터 모듈 단위로 점진 strict 적용을 권장합니다.

---

## 6. 잘하고 있는 점 (유지·확장 권장)

- **모듈화가 실제로 진행됨** — `prediction/mixins/*`, `gui/controller_*.py`, `prediction/features/*` 등 거대 God-Object를 mixin/패키지로 분해한 흔적이 명확합니다. 이 패턴을 남은 핫패스(3-2)에 계속 적용하면 됩니다.
- **bare `except:` 0건** — 모든 예외가 최소 `except Exception`으로 좁혀져 있습니다(다음 단계는 2-3의 로깅화).
- **전 파일 구문 컴파일 통과** — 구조적 손상 없음.
- **dataclass·Enum·Protocol 적극 활용**, optional-dependency(`dev/llm/gui/ml`) 분리로 설치 부담을 줄인 점 우수.
- **풍부한 설계·아키텍처 문서**(`docs/architecture/*`)와 53개 테스트 파일 — 기반이 갖춰져 있어 위 개선들이 빠르게 정착될 수 있습니다.

---

## 7. 우선순위 실행 체크리스트

**오늘 (보안)**
- [ ] OpenAI·Gemini·eBest·Telegram **자격증명 전수 폐기·재발급**
- [ ] 패키징/zip 스크립트에서 `config.secrets.json` 제외 보장

**이번 주 (버그·안정성)**
- [ ] `telegram/formatters.py` 787/791/795/799 내부 따옴표 수정 (2-1)
- [ ] `prediction/data_builder.py`의 `adaptive_dict` → 실제 원본 dict로 치환 (2-2)
- [ ] `tests/test_oi_levels.py`, `tests/test_rule_based_core.py` import 경로 수정 (4-1)
- [ ] `ruff check . --fix` 적용 후 diff 검토 (3-1)
- [ ] `.github/workflows/ci.yml` 추가 (3.10 매트릭스 포함) (4-2)

**이번 달 (구조·견고성)**
- [ ] 핫패스(`ebestapi`,`trading`,`prediction`)의 `except: pass` → 로깅화 (2-3)
- [ ] `main.py`의 `getLogger` 전역 교체 제거 + 죽은 `_startup_internet_time_sync` 삭제 (3-4)
- [ ] `config/config.json.example` 추가, 의존성 lock 도입 (5-1, 5-2)
- [ ] `trade_dashboard.py` 중복 정리, 실험 indicator 분류·라벨링 (3-3)

**분기 (점진 개선)**
- [ ] `AdaptiveZigZag.update()` / `config._from_dict·validate` 분해 + 회귀 스냅샷 테스트 (3-2)
- [ ] ruff 규칙 확장(BLE/I/UP/B) + mypy strict 점진 적용 (5-3, 5-4)

---

## 부록 A. 검증에 사용한 수치 (재현 가능)

| 항목 | 값 | 산출 방법 |
|------|----|-----------|
| Python 파일 / 총 LOC | 225 / 101,865 | `find -name '*.py'` |
| 비-테스트 `except Exception` | ~2,138 | `grep` |
| `except Exception: pass` | ~699 | 멀티라인 `grep -P` |
| ruff 총 경고 | 843 (자동수정 443) | `ruff check . --statistics` |
| 3.10 비호환 f-string | 4 (formatters.py) | ruff `invalid-syntax` |
| 미정의 `adaptive_dict` 참조 | 6곳 (data_builder.py) | `grep -n adaptive_dict` |
| 수집 실패 테스트(경로 오류) | 2 | `pytest --collect-only` |
| 최대 파일 | 3,941줄 (adaptive_zigzag.py) | `wc -l` |
| 최대 메서드 | ~595줄 (`update()`) | 라인 1245–1840 |

> 본 리뷰는 정적 분석·구조 검토 기반입니다. 핵심 수정(특히 2-1, 2-2) 적용 시에는 기존 동작과의 동일성을 보장하기 위해 회귀 테스트(동일 틱 입력 → 동일 pivot/신호 출력)를 함께 두시길 권합니다.
