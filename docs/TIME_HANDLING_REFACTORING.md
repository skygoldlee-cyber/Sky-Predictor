# Time Handling Refactoring

## 개요

SkyPredictor 프로젝트의 시간 처리 로직을 테스트 및 백테스트 지원을 위해 개선했습니다. 직접적인 `datetime.now()` 호출을 주입 가능한 `now_fn` 패턴으로 변경하여 시간 의존성을 추상화했습니다.

## 목표

- **테스트 가능성 향상**: 시간 함수를 주입하여 결정적 테스트 가능
- **백테스트 지원**: 과거 시점 시뮬레이션 지원
- **코드 일관성**: 전체 프로젝트에서 일관된 시간 처리 패턴 적용

## 리팩토링 패턴

### 기본 패턴

```python
from typing import Callable, Optional
from datetime import datetime

def example_function(now_fn: Optional[Callable[[], datetime]] = None):
    """예시 함수.
    
    Args:
        now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
    """
    _now = now_fn if now_fn is not None else datetime.now
    current_time = _now()
    # current_time 사용
```

### 클래스 패턴

```python
class ExampleClass:
    def __init__(self, now_fn: Optional[Callable[[], datetime]] = None):
        """초기화.
        
        Args:
            now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
        """
        self._now_fn = now_fn if now_fn is not None else datetime.now
    
    def some_method(self):
        current_time = self._now_fn()
        # current_time 사용
```

## 리팩토링된 파일

### Training Scripts

#### `training/train_tft.py`
- `run()` 함수에 `now_fn` 파라미터 추가
- `main()` 함수에 `now_fn` 파라미터 추가 및 `run()`으로 전달
- `datetime.now()` → `_now()`로 변경

#### `training/train_patch_tst.py`
- `run()` 함수에 `now_fn` 파라미터 추가
- `main()` 함수에 `now_fn` 파라미터 추가 및 `run()`으로 전달
- `datetime.now()` → `_now()`로 변경

#### `training/train_mamba.py`
- `run()` 함수에 `now_fn` 파라미터 추가
- `main()` 함수에 `now_fn` 파라미터 추가 및 `run()`으로 전달
- `datetime.now()` → `_now()`로 변경

#### `training/train.py`
- `run()` 함수에 `now_fn` 파라미터 추가
- `main()` 함수에 `now_fn` 파라미터 추가 및 `run()`으로 전달
- `datetime.now()` → `_now()`로 변경

### Indicators

#### `indicators/adaptive_session_table.py`
- `AdaptiveSessionTable.__init__()`에 `now_fn` 파라미터 추가
- `self._now_fn` 인스턴스 변수 저장
- `update()` 메서드에서 `datetime.now()` → `self._now_fn()`로 변경

#### `indicators/adaptive_parameter_adjuster.py`
- `AdaptiveParameterAdjuster.__init__()`에 `now_fn` 파라미터 추가
- `self._now_fn` 인스턴스 변수 저장
- `TimeStrategy.adjust()`에서 `datetime.now()` → `self.adjuster._now_fn()`로 변경

### Core Utilities

#### `core/utils.py`
- `get_previous_business_day()`에 `now_fn` 파라미터 추가
- `get_expiry_week_info()`에 `now_fn` 파라미터 추가
- `get_option_month_yyyymm()`에 `now_fn` 파라미터 추가
- `parse_chetime()`에 `now_fn` 파라미터 추가
- `parse_ebest_tick_datetime()`에 `now_fn` 파라미터 추가
- `get_default_ticks_output_path()`에 `now_fn` 파라미터 추가
- 모든 함수에서 `datetime.now()` → `_now()`로 변경

### Data Processing

#### `data/merge_datasets.py`
- `main()` 함수에 `now_fn` 파라미터 추가
- `datetime.now()` → `_now()`로 변경

## 사용 예시

### 테스트에서 시간 주입

```python
from datetime import datetime
from training.train_tft import run
from argparse import Namespace

# 고정 시간 설정
fixed_time = datetime(2025, 1, 15, 10, 30, 0)

# 테스트에서 시간 함수 주입
args = Namespace(...)
run(args, now_fn=lambda: fixed_time)
```

### 백테스트에서 시간 시뮬레이션

```python
from datetime import datetime, timedelta
from indicators.adaptive_session_table import AdaptiveSessionTable

# 시간 시뮬레이션
current_time = datetime(2025, 1, 15, 9, 0, 0)
session_table = AdaptiveSessionTable(base_config, now_fn=lambda: current_time)

# 시간 진행
current_time += timedelta(minutes=1)
session_table.update(bar_time, pivot_quality=0.75)
```

## 변경되지 않은 파일

다음 파일의 `datetime.now()` 호출은 유지됩니다:

- **Test files** (`tests/`): 테스트에서 직접 호출 사용
- **GUI files** (`gui/`): 인터랙티브 컴포넌트
- **API integration** (`ebestapi/`): 외부 API 호출
- **Logging utilities** (`core/logging_utils.py`): 타임스탬프 로깅
- **Data processing** (`data/tick_processor.py`, `data/backtest_data_saver.py`): 실시간 데이터 처리
- **App utilities** (`app/`): 설정 및 실험 스크립트

## 이점

1. **테스트 가능성**: 시간 의존 테스트를 결정적으로 작성 가능
2. **백테스트 지원**: 과거 시점 시뮬레이션 용이
3. **코드 일관성**: 핵심 로직에서 일관된 시간 처리 패턴
4. **유지보수성**: 시간 관련 버그 수정 용이

## 백워드 호환성

모든 변경은 백워드 호환성을 유지합니다:
- `now_fn` 파라미터는 기본값 `None`으로 설정
- `None`인 경우 기존 `datetime.now()` 사용
- 기존 코드는 수정 없이 정상 작동

## 관련 문서

- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
- [CHANGELOG.md](../CHANGELOG.md) - 변경 로그

---

**작성일**: 2026-06-16  
**버전**: 1.0
