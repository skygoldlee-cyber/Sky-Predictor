# 피봇 정보 패널 (Crosshair) 구현 문서

## 개요
차트에서 마우스가 피봇 마커 근처에 있을 때 우측상단에 피봇 정보를 표시하는 기능입니다.

## 파일 위치
- **주요 파일**: `gui/chart_viewer.py`
- **관련 파일**: `gui/fplt_renderer.py` (HistoricalPivot, PivotProbabilityCalculator 클래스)

## 구조

### 1. 초기화

#### 1.1 위젯 초기화 (`ChartViewerWidget.__init__`)
```python
# 피봇 정보 패널 (crosshair용)
self._pivot_info_panel: Optional[Any] = None
```

#### 1.2 패널 생성 (`_build_widget`)
```python
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

self._pivot_info_panel = QLabel(win)
self._pivot_info_panel.setStyleSheet("""
    QLabel {
        background-color: rgba(0, 0, 0, 180);
        color: #7CFC00;
        border: 1px solid #7CFC00;
        border-radius: 3px;
        padding: 5px;
        font-size: 11px;
        font-family: Consolas, monospace;
    }
""")
self._pivot_info_panel.hide()
self._pivot_info_panel.setParent(win)
self._pivot_info_panel.raise_()
```

**스타일 특징**:
- 반투명 검은 배경 (rgba 0,0,0,180)
- 연두색 텍스트 및 테두리 (#7CFC00)
- 모나스페이스 폰트 (Consolas)

#### 1.3 이벤트 연결
```python
if hasattr(ax_main, 'vb') and hasattr(ax_main.vb, 'scene'):
    ax_main.vb.scene().sigMouseMoved.connect(self._on_crosshair_moved)
```

## 2. 이벤트 처리 흐름

### 2.1 Crosshair 이동 이벤트 (`_on_crosshair_moved`)
```python
def _on_crosshair_moved(self, pos) -> None:
    """Crosshair 이동 이벤트 핸들러 (피봇 정보 패널용)."""
    if self._pivot_info_panel is None or self._renderer is None:
        return

    # 디바운싱: 50ms 이내의 연속 호출을 무시
    if self._crosshair_debounce_timer is not None:
        self._crosshair_debounce_timer.stop()
    from PySide6.QtCore import QTimer
    self._crosshair_debounce_timer = QTimer()
    self._crosshair_debounce_timer.setSingleShot(True)
    self._crosshair_debounce_timer.timeout.connect(lambda: self._do_crosshair_update(pos))
    self._crosshair_debounce_timer.start(50)
```

**디바운싱 목적**: 마우스 이동 시 연속적인 이벤트 호출을 방지하여 성능 최적화

### 2.2 실제 업데이트 처리 (`_do_crosshair_update`)

#### 2.2.1 좌표 변환
```python
# Scene 좌표 → View 좌표
vb = getattr(self._renderer.ax_main, 'vb', None)
mouse_point = vb.mapSceneToView(pos)
x_coord = mouse_point.x()   # 뷰 좌표계 x값 (봉 인덱스)
y_coord = mouse_point.y()

# 봉 인덱스 계산 (finplot 뷰 좌표계의 x축은 0-based 봉 인덱스)
nearest_idx = int(round(x_coord))
nearest_idx = max(0, min(nearest_idx, len(df_index) - 1))
```

#### 2.2.2 피봇 데이터 조회
```python
pivot_info = self._renderer._pivot_info
pivot_idx_arr = self._renderer._pivot_idx_arr  # 0-based 봉 인덱스
pivot_y_arr   = self._renderer._pivot_y_arr
```

#### 2.2.3 근접 피봇 필터링 (20봉 이내)
```python
# 20봉 이내 피봇 필터링
mask = np.abs(pivot_idx_arr.astype(float) - nearest_idx) <= 20
if not mask.any():
    self._pivot_info_panel.hide()
    return
```

#### 2.2.4 복합 거리 계산
```python
# 복합 거리 계산 (인덱스 거리 + 가격 거리)
idx_diffs   = np.abs(pivot_idx_arr[mask].astype(float) - nearest_idx)
price_diffs = np.abs(pivot_y_arr[mask] - y_coord)
y_safe      = max(abs(y_coord), 1.0)
distances   = idx_diffs * 0.7 + (price_diffs / y_safe * 100) * 0.3

best_match_local = int(np.argmin(distances))

# 원본 인덱스 복원
orig_indices = np.where(mask)[0]
orig_idx     = int(orig_indices[best_match_local])
pivot_row    = pivot_info.iloc[orig_idx]
```

**거리 계산 가중치**:
- 인덱스 거리: 70%
- 가격 거리: 30% (정규화된 값)

#### 2.2.5 정보 텍스트 생성
```python
pivot_type  = pivot_row['t']
pivot_price = float(pivot_row['y'])
pivot_idx_v = int(pivot_idx_val.iloc[0]) if hasattr(pivot_idx_val, 'iloc') else int(pivot_idx_val)

time_str = (df_index[pivot_idx_v].strftime('%H:%M')
            if pivot_idx_v < len(df_index) else f"idx:{pivot_idx_v}")

# 확정 여부 확인
is_confirmed = (
    'confirmed_at_idx' in pivot_row.index
    and pd.notna(pivot_row['confirmed_at_idx'])
    and int(pivot_row['confirmed_at_idx']) >= 0
)
prefix   = "피봇" if is_confirmed else "후보"
info_text = f"{prefix}: {pivot_type} | 가격: {pivot_price:.2f} | 시간: {time_str}"
```

#### 2.2.6 확정 지연 정보
```python
if is_confirmed:
    confirmed_at = int(pivot_row['confirmed_at_idx'])
    delay = confirmed_at - pivot_idx_v
    info_text += f" | 지연확정: +{delay}봉" if delay > 0 else " | 즉시확정"
```

#### 2.2.7 확정 확률 계산 (미확정 피봇)
```python
else:
    try:
        candidate = HistoricalPivot(
            idx=pivot_idx_v, price=pivot_price,
            pivot_type=pivot_type, confirmed=False,
        )
        prob = self._pivot_prob_calc.calculate_combined_probability(
            candidate, y_coord, confirmation_bars_required=3
        )
        info_text += f" | 확정확률: {prob*100:.1f}%"
    except Exception:
        pass
```

#### 2.2.8 패널 표시
```python
self._pivot_info_panel.setText(info_text)
self._pivot_info_panel.adjustSize()

# 패널 위치 (우측 상단)
parent = self._pivot_info_panel.parent()
if parent:
    pw = self._pivot_info_panel.width()
    self._pivot_info_panel.move(parent.width() - pw - 20, 10)

self._pivot_info_panel.show()
```

## 3. 보조 클래스

### 3.1 HistoricalPivot (`gui/fplt_renderer.py`)
```python
@dataclass
class HistoricalPivot:
    """과거 피봇 데이터."""
    idx: int
    price: float
    pivot_type: str  # "H" or "L"
    confirmed: bool
    confirmation_bars: int = 0
    price_deviation_pct: float = 0.0
    timestamp: Optional[pd.Timestamp] = None
```

### 3.2 PivotProbabilityCalculator (`gui/fplt_renderer.py`)
```python
class PivotProbabilityCalculator:
    """피봇 확정 확률 계산기 (통계 + 기술적 조건 조합)."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.historical_pivots: List[HistoricalPivot] = []
        self.stat_weight = 0.4  # 통계 기반 가중치
        self.tech_weight = 0.6  # 기술적 조건 가중치
```

**확률 계산 방법**:
1. **통계적 확률**: 과거 유사 피봇의 확정 비율
2. **기술적 확률**: confirmation_bars 진행 정도, 가격 이탈 정도
3. **조합 확률**: 통계(40%) + 기술적(60%)

## 4. 표시 형식

### 4.1 확정 피봇
```
피봇: H | 가격: 325.50 | 시간: 10:15 | 지연확정: +2봉
```

### 4.2 즉시 확정 피봇
```
피봇: L | 가격: 324.00 | 시간: 09:45 | 즉시확정
```

### 4.3 미확정 피봇 (후보)
```
후보: H | 가격: 326.00 | 시간: 11:30 | 확정확률: 75.5%
```

## 5. 성능 최적화

### 5.1 디바운싱
- 50ms 딜레이 적용
- 연속적인 마우스 이동 이벤트를 단일 업데이트로 통합

### 5.2 필터링
- 20봉 이내 피봇만 검색
- 복합 거리 계산으로 가장 근접한 피봇 선택

### 5.3 캐싱
- `_x_coords_cache`: X 좌표 캐시
- `_crosshair_debounce_timer`: 타이머 재사용

## 6. 에러 처리

```python
except Exception as e:
    logger.debug("[ChartViewerWidget] crosshair 처리 실패: %s", e, exc_info=True)
    if self._pivot_info_panel:
        self._pivot_info_panel.hide()
```

모든 예외 상황에서 패널을 숨김으로써 UI 오류 방지

## 7. 의존성

### 7.1 내부 모듈
- `gui.fplt_renderer`: HistoricalPivot, PivotProbabilityCalculator

### 7.2 외부 라이브러리
- PySide6 (Qt): QLabel, QTimer, Qt
- numpy: 배열 연산
- pandas: 데이터프레임 처리
- finplot: 차트 렌더링

## 8. 제약 사항

1. **데이터 범위**: 마우스 위치가 데이터 범위를 벗어나면 패널 숨김
2. **피봇 존재**: 20봉 이내에 피봇이 없으면 패널 숨김
3. **렌더러 의존**: `_renderer`가 초기화되어야 동작
