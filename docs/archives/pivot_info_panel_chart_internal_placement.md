# 피봇 정보 패널 차트 내부 배치 방법

## 개요
현재 피봇 정보 패널은 차트 위에 오버레이(QLabel)로 표시되고 있습니다. 이를 차트 내부(데이터 영역 내)에 배치하는 방법을 설명합니다.

## 현재 구현 (오버레이 방식)

### 위치
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
    
    # 차트 내부 우측 상단 위치 계산 (데이터 좌표계)
    if self._renderer._df_index is not None and len(self._renderer._df_index) > 0:
        # 현재 뷰 범위 가져오기
        vb = self._renderer.ax_main.vb
        view_range = vb.viewRange()
        
        # x축: 현재 표시 범위의 우측 끝
        x_max = view_range[0][1]
        
        # y축: 현재 표시 범위의 상단 (고가)
        y_max = view_range[1][1]
        
        # 여백 추가 (데이터 범위의 2%)
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]
        
        x_pos = x_max - x_range * 0.02
        y_pos = y_max - y_range * 0.02
        
        # TextItem 위치 설정
        self._pivot_text_item.setPos(x_pos, y_pos)
        self._pivot_text_item.setText(info_text)
        self._pivot_text_item.show()
```

#### 1.3 장점
- 차트 줌/팬에 따라 자동 이동
- 데이터 좌표계 사용으로 직관적
- PyQtGraph 네이티브 기능으로 안정적

#### 1.4 단점
- 줌/팬 시 위치가 계속 변경됨 (사용자가 따라가기 어려울 수 있음)
- 차트 데이터와 겹칠 수 있음

---

### 방법 2: 고정 데이터 좌표 사용

특정 데이터 좌표(예: 최신 봉의 우측)에 고정하여 배치합니다.

#### 2.1 위치 계산
```python
def _do_crosshair_update(self, pos) -> None:
    # ... 기존 로직 ...
    
    # 최신 봉 인덱스와 가격 범위 사용
    if self._renderer._df_index is not None and len(self._renderer._df_index) > 0:
        df = self._renderer._df
        latest_idx = len(df) - 1
        
        # 최신 봉의 고가/저가 범위
        latest_high = df['High'].iloc[-1]
        latest_low = df['Low'].iloc[-1]
        price_range = latest_high - latest_low
        
        # 최신 봉 우측 상단에 배치
        x_pos = latest_idx + 2  # 2봉 우측
        y_pos = latest_high + price_range * 0.05  # 고가에서 5% 상단
        
        self._pivot_text_item.setPos(x_pos, y_pos)
        self._pivot_text_item.setText(info_text)
        self._pivot_text_item.show()
```

#### 2.2 장점
- 항상 최신 데이터 근처에 표시
- 시간 흐름에 따라 자동 이동

#### 2.3 단점
- 차트 줌 시 위치가 상대적으로 변경됨
- 데이터 범위를 벗어날 수 있음

---

### 방법 3: finplot 내부 레이아웃 활용

finplot의 내부 레이아웃 구조를 활용하여 차트 내부에 위젯을 배치합니다.

#### 3.1 finplot 내부 구조
```python
# finplot은 PyQtGraph의 GraphicsLayoutWidget을 사용
# ax_main은 ViewBox를 포함
import finplot as fplt

ax_main = fplt.create_plot(...)  # PlotItem 반환
# ax_main.vb는 ViewBox
# ax_main.vb.scene()은 QGraphicsScene
```

#### 3.2 ViewBox 내부에 위젯 추가
```python
def _build_widget(self, parent_widget=None):
    # ... 기존 코드 ...
    
    # ViewBox 가져오기
    vb = ax_main.vb
    
    # QGraphicsProxyWidget으로 QLabel을 Scene에 추가
    from PySide6.QtWidgets import QGraphicsProxyWidget
    
    proxy = QGraphicsProxyWidget()
    proxy.setWidget(self._pivot_info_panel)
    
    # Scene에 추가
    vb.scene().addItem(proxy)
    
    # 데이터 좌표계로 위치 설정
    # 이는 복잡하므로 방법 1(TextItem) 추천
```

#### 3.3 장점
- 기존 QLabel 스타일 유지 가능
- Scene 기반으로 유연한 배치

#### 3.4 단점
- 좌표 변환 복잡
- 성능 저하 가능성

---

### 방법 4: 하위 축(하단 패널) 활용

차트 하단에 별도의 작은 패널을 생성하여 정보를 표시합니다.

#### 4.1 하위 축 생성
```python
def _build_widget(self, parent_widget=None):
    import finplot as fplt
    
    # 메인 차트
    ax_main = fplt.create_plot(...)
    
    # 하단 정보 패널 (높이 작게)
    ax_info = fplt.create_plot(...)
    ax_info.set_visible(False)  # 기본 숨김
    
    # 레이아웃에 추가
    win = fplt.ForegroundWindow()
    win.ci.layout.addItem(ax_main, row=0, col=0)
    win.ci.layout.addItem(ax_info, row=1, col=0)
    
    return win
```

#### 4.2 정보 표시
```python
def _do_crosshair_update(self, pos) -> None:
    # ... 기존 로직 ...
    
    # 하단 패널에 텍스트 추가
    if hasattr(self, 'ax_info'):
        self.ax_info.clear()
        self.ax_info.text(info_text, color="#7CFC00")
        self.ax_info.set_visible(True)
```

#### 4.3 장점
- 차트 데이터와 겹치지 않음
- 독립적인 영역으로 깔끔함

#### 4.4 단점
- 차트 높이 감소
- 별도의 축 관리 필요

---

## 추천 구현: 방법 1 (TextItem) + 옵션

### 구현 예시

#### 1. 초기화
```python
# ChartViewerWidget.__init__
def __init__(self, ...):
    # ... 기존 코드 ...
    
    # 피봇 정보 TextItem (차트 내부 배치용)
    self._pivot_text_item: Optional[TextItem] = None
    self._use_internal_panel: bool = True  # 내부/외부 패널 전환 플래그
```

#### 2. TextItem 생성
```python
def _build_widget(self, parent_widget=None):
    # ... 기존 코드 ...
    
    # TextItem 생성
    try:
        from pyqtgraph import TextItem
        from PySide6.QtGui import QFont
        
        self._pivot_text_item = TextItem(
            color="#7CFC00",
            anchor=(1, 0),  # 우측 상단
            border="#7CFC00",
            fill=(0, 0, 0, 180)
        )
        self._pivot_text_item.setFont(QFont("Consolas", 11))
        self._pivot_text_item.hide()
        
        # ViewBox에 추가
        if hasattr(ax_main, 'vb'):
            ax_main.vb.addItem(self._pivot_text_item)
            
    except Exception as e:
        logger.warning("[ChartViewer] TextItem 생성 실패: %s", e)
        self._pivot_text_item = None
    
    # 기존 QLabel 패널도 유지 (외부용)
    # ...
```

#### 3. 위치 업데이트
```python
def _do_crosshair_update(self, pos) -> None:
    # ... 기존 피봇 검색 로직 ...
    
    if self._use_internal_panel and self._pivot_text_item is not None:
        # 차트 내부 배치 (TextItem)
        vb = self._renderer.ax_main.vb
        view_range = vb.viewRange()
        
        # 현재 뷰 범위의 우측 상단 (데이터 좌표)
        x_max = view_range[0][1]
        y_max = view_range[1][1]
        
        # 여백 추가
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]
        
        x_pos = x_max - x_range * 0.02
        y_pos = y_max - y_range * 0.02
        
        self._pivot_text_item.setPos(x_pos, y_pos)
        self._pivot_text_item.setText(info_text)
        self._pivot_text_item.show()
        
        # 외부 패널 숨김
        if self._pivot_info_panel:
            self._pivot_info_panel.hide()
            
    elif self._pivot_info_panel is not None:
        # 기존 오버레이 방식 (QLabel)
        self._pivot_info_panel.setText(info_text)
        self._pivot_info_panel.adjustSize()
        
        parent = self._pivot_info_panel.parent()
        if parent:
            pw = self._pivot_info_panel.width()
            self._pivot_info_panel.move(parent.width() - pw - 20, 10)
        
        self._pivot_info_panel.show()
        
        # 내부 TextItem 숨김
        if self._pivot_text_item:
            self._pivot_text_item.hide()
```

#### 4. 전환 기능 (선택사항)
```python
def _toggle_panel_placement(self, internal: bool) -> None:
    """내부/외부 패널 전환."""
    self._use_internal_panel = internal
    
    # 현재 표시된 패널 숨김
    if self._pivot_info_panel:
        self._pivot_info_panel.hide()
    if self._pivot_text_item:
        self._pivot_text_item.hide()
```

---

## 비교표

| 방법 | 장점 | 단점 | 난이도 |
|------|------|------|--------|
| **현재 (오버레이)** | 고정 위치, 안정적 | 차트와 분리됨 | 낮음 |
| **TextItem** | 차트 좌표계, 줌/팬 연동 | 위치 계속 변경, 데이터 겹침 | 중간 |
| **고정 데이터 좌표** | 최신 데이터 근처 | 상대적 위치 변화 | 중간 |
| **finplot 내부 레이아웃** | 유연한 배치 | 좌표 변환 복잡 | 높음 |
| **하위 축 패널** | 데이터와 분리 | 차트 높이 감소 | 중간 |

---

## 권장 사항

### 기본 사용자: 현재 오버레이 방식 유지
- 이해하기 쉽고 안정적
- 차트 조작에 방해되지 않음

### 고급 사용자: TextItem + 옵션
- 내부/외부 전환 기능 제공
- 사용자 선호에 따라 선택

### 최적화: 하위 축 패널
- 정보 표시 전용 공간 확보
- 차트 데이터와 완전 분리

---

## 추가 고려사항

### 1. 성능
- TextItem은 매 업데이트 시 재그리기 필요
- 디바운싱(현재 50ms) 유지 권장

### 2. 가독성
- 배경색/텍스트색 대비 유지
- 차트 데이터와 겹침 방지

### 3. 반응형
- 차트 크기 변경 시 위치 재계산
- 줌/팬 이벤트 감지

### 4. 사용자 설정
- config.json에 배치 방법 옵션 추가
```json
{
  "chart": {
    "pivot_info_placement": "overlay",  // "overlay", "internal", "bottom"
    "pivot_info_internal_fixed": false  // true: 고정 좌표, false: 뷰 범위 기반
  }
}
```
