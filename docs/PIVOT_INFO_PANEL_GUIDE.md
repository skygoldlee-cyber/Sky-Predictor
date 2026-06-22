# 피봇 정보 패널 구현 및 배치 가이드

> 대상 파일: `gui/chart_viewer.py`, `gui/fplt_renderer.py`  
> 최종 수정: 2026-06-16  
> 병합 대상: pivot_info_panel_chart_internal_placement.md, pivot_info_panel_implementation.md

---

## 목차

1. [개요](#개요)
2. [현재 구현 (오버레이 방식)](#현재-구현-오버레이-방식)
3. [차트 내부 배치 방법](#차트-내부-배치-방법)
4. [구현 상세](#구현-상세)
5. [마이그레이션 가이드](#마이그레이션-가이드)

---

## 개요

차트에서 마우스가 피봇 마커 근처에 있을 때 우측상단에 피봇 정보를 표시하는 기능입니다. 현재는 오버레이(QLabel) 방식으로 구현되어 있으며, 차트 내부 배치(TextItem) 방식으로 변경할 수 있습니다.

---

## 현재 구현 (오버레이 방식)

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

### 2. 위치 설정

```python
# 패널 위치 (우측 상단)
parent = self._pivot_info_panel.parent()
if parent:
    pw = self._pivot_info_panel.width()
    self._pivot_info_panel.move(parent.width() - pw - 20, 10)
```

**특징**:
- 차트 위젯의 픽셀 좌표 사용
- 차트 줌/패에 영향받지 않음 (고정 위치)
- 차트 데이터와 독립적인 레이어

### 3. 정보 업데이트

```python
def _on_crosshair_moved(self, pos):
    # 피봇 마커 근처 확인
    pivot_info = self._find_nearby_pivot(pos)
    
    if pivot_info:
        self._pivot_info_panel.setText(pivot_info)
        self._pivot_info_panel.show()
    else:
        self._pivot_info_panel.hide()
```

---

## 차트 내부 배치 방법

### 방법 1: PyQtGraph TextItem 사용 (추천)

PyQtGraph의 `TextItem`을 사용하여 차트 좌표계에 텍스트를 배치합니다.

#### 1.1 TextItem 초기화
```python
from pyqtgraph import TextItem

# ChartViewerWidget.__init__ 또는 _build_widget에서 초기화
self._pivot_text_item = TextItem(
    color="#7CFC00",
    anchor=(1, 0),  # 우측 상단 앵커
    border="#7CFC00",
    fill=(0, 0, 0, 180)
)
self._pivot_text_item.setFont(QFont("Consolas", 11))
self._pivot_text_item.hide()

# ViewBox에 추가
if hasattr(self._renderer, 'ax_main'):
    self._renderer.ax_main.addItem(self._pivot_text_item)
```

#### 1.2 위치 업데이트 (데이터 좌표계)
```python
def _do_crosshair_update(self, pos) -> None:
    # ... 기존 로직 ...
    
    # 데이터 좌표계로 위치 설정
    if pivot_info:
        # 우측 상단 (데이터 좌표)
        x_range = self._renderer.ax_main.viewRange()[0]
        y_range = self._renderer.ax_main.viewRange()[1]
        
        x_pos = x_range[1]  # 우측
        y_pos = y_range[1]  # 상단
        
        self._pivot_text_item.setText(pivot_info)
        self._pivot_text_item.setPos(x_pos, y_pos)
        self._pivot_text_item.show()
    else:
        self._pivot_text_item.hide()
```

**장점**:
- 차트 줌/패에 따라 자동 이동
- 데이터 좌표계로 직관적 위치 설정
- 차트와 자연스럽게 통합

#### 1.3 스타일 설정
```python
self._pivot_text_item.setColor("#7CFC00")
self._pivot_text_item.setBorder("#7CFC00")
self._pivot_text_item.setFill((0, 0, 0, 180))  # RGBA
```

### 방법 2: QGraphicsTextItem 사용

Qt의 `QGraphicsTextItem`을 사용하여 더 복잡한 서식 지원.

```python
from PySide6.QtWidgets import QGraphicsTextItem

self._pivot_graphics_item = QGraphicsTextItem()
self._pivot_graphics_item.setDefaultTextColor(QColor("#7CFC00"))
self._pivot_graphics_item.setFont(QFont("Consolas", 11))

# QGraphicsScene에 추가
if hasattr(self._renderer, 'ax_main'):
    scene = self._renderer.ax_main.scene()
    scene.addItem(self._pivot_graphics_item)
```

---

## 구현 상세

### 피봇 정보 형식

```python
pivot_info = f"""
[HIGH] 370.25
Dist: 0.52%
Urgency: 0.78
Prob: 0.85
Age: 3
""".strip()
```

### 피봇 검색 로직

```python
def _find_nearby_pivot(self, pos):
    """마우스 위치 근처의 피봇 검색"""
    x, y = pos
    threshold = 10  # 픽셀
    
    for pivot in self._pivots:
        px, py = self._data_to_pixel(pivot['x'], pivot['y'])
        if abs(px - x) < threshold and abs(py - y) < threshold:
            return self._format_pivot_info(pivot)
    
    return None
```

### 데이터-픽셀 변환

```python
def _data_to_pixel(self, data_x, data_y):
    """데이터 좌표를 픽셀 좌표로 변환"""
    vb = self._renderer.ax_main.vb
    pixel_pos = vb.mapViewToDevice(data_x, data_y)
    return pixel_pos.x(), pixel_pos.y()
```

---

## 마이그레이션 가이드

### 오버레이 → 차트 내부 배치

#### 단계 1: TextItem 초기화
```python
# 기존 QLabel 제거
# self._pivot_info_panel = QLabel(win)  # 삭제

# TextItem 추가
self._pivot_text_item = TextItem(...)
self._renderer.ax_main.addItem(self._pivot_text_item)
```

#### 단계 2: 위치 업데이트 로직 변경
```python
# 기존 픽셀 좌표 사용 로직 제거
# self._pivot_info_panel.move(parent.width() - pw - 20, 10)  # 삭제

# 데이터 좌표계 사용 로직 추가
x_range = self._renderer.ax_main.viewRange()[0]
y_range = self._renderer.ax_main.viewRange()[1]
self._pivot_text_item.setPos(x_range[1], y_range[1])
```

#### 단계 3: 이벤트 핸들러 업데이트
```python
def _on_crosshair_moved(self, pos):
    # 기존 로직 유지
    pivot_info = self._find_nearby_pivot(pos)
    
    if pivot_info:
        # QLabel 대신 TextItem 사용
        self._pivot_text_item.setText(pivot_info)
        self._pivot_text_item.show()
    else:
        self._pivot_text_item.hide()
```

### 주의사항

1. **좌표계 차이**: 오버레이는 픽셀 좌표, TextItem은 데이터 좌표
2. **줌/패 동작**: TextItem은 자동 이동, 오버레이는 고정
3. **성능**: TextItem이 더 효율적 (차트 렌더링과 통합)
4. **호환성**: 기존 QLabel 코드는 완전히 제거 필요

---

## 비교 요약

| 특징 | 오버레이 (QLabel) | 차트 내부 (TextItem) |
|------|-------------------|---------------------|
| **좌표계** | 픽셀 좌표 | 데이터 좌표 |
| **줌/패** | 고정 위치 | 자동 이동 |
| **성능** | 독립 레이어 | 차트 통합 |
| **구현 복잡도** | 단순 | 중간 |
| **자연스러움** | 낮음 | 높음 |
| **서식 지원** | 풍부 (HTML) | 제한적 |

---

**문서 버전**: 1.0  
**작성일**: 2026-06-16  
**마지막 수정**: 2026-06-16  
**병합 대상**: pivot_info_panel_chart_internal_placement.md, pivot_info_panel_implementation.md
