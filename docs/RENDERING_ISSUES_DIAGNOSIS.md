# 렌더링 관련 문제점 진단

## 문제 개요

차트 렌더링 시 발생할 수 있는 성능 및 시각적 문제점을 진단합니다.

## 진단 결과

### 1. 객체 제거 로직 문제

**현재 로직:**
```python
def _scene_remove(self, obj: Any) -> bool:
    """scene에서 객체 제거"""
    for attr in ("items", "item", "curve", "scatter"):
        try:
            t = getattr(obj, attr, None)
            if t is not None:
                sc = t.scene() if callable(getattr(t, "scene", None)) else None
                if sc:
                    sc.removeItem(t)
                    return True
        except Exception:
            continue
    return False
```

**문제:**
- 여러 속성을 순회하며 제거 시도 → 성능 저하
- 예외 처리로 실패 시 계속 시도 → 불필요한 연산

**위치:**
- `gui/renderers/fplt_renderer.py` line 448-465

### 2. NaN 처리 문제

**현재 로직:**
```python
def _upsert_st(self, name: str, x, y, ax, **kwargs) -> None:
    # NaN 제거 없이 full-length 배열 전달
    # finplot은 NaN 구간을 선 단절로 처리
```

**문제:**
- NaN이 많은 경우 렌더링 성능 저하
- 선 단절이 많아지면 시각적으로 복잡해짐

**위치:**
- `gui/renderers/fplt_renderer.py` line 479-512

### 3. 캔들스틱 재생성 빈도

**현재 로직:**
```python
def _upsert_candle(self, cdf: pd.DataFrame, ax) -> None:
    try:
        existing = self._plots.get("_candle")
        if existing is not None:
            existing.update_data(cdf.index, cdf.Open, cdf.High, cdf.Low, cdf.Close)
    except (TypeError, ValueError):
        self._remove("_candle")
    except RuntimeError:
        self._remove("_candle")
    except Exception:
        self._remove("_candle")
```

**문제:**
- 예외 발생 시마다 _remove → 재생성 루프
- 데이터 구조 변경 시마다 재생성

**위치:**
- `gui/renderers/fplt_renderer.py` line 715-737

### 4. 피봇 마커 업데이트 빈도

**현재 로직:**
```python
def _plot_pivot_bucket(self, pm: Dict[str, Any], ax) -> None:
    # 매 호출마다 모든 마커 재생성
    for i, idx in enumerate(pm["confirmed"]["idx"]):
        self._upsert(f"_conf_h_{i}", ...)
```

**문제:**
- 피봇이 많을 경우 마커 재생성 비용 증가
- 증분 업데이트 미지원

**위치:**
- `gui/renderers/fplt_renderer.py` line 850-950

### 5. MA 표시/숨김 로직

**현재 로직:**
```python
def set_ma_enabled(self, enabled: bool) -> None:
    for name in ("_ma20", "_ma60"):
        obj = self._plots.get(name)
        if obj is not None:
            try:
                obj.setOpacity(1.0 if enabled else 0.0)
            except Exception:
                if not enabled:
                    self._remove(name)
```

**문제:**
- setOpacity 실패 시 _remove fallback → 깜빡임 유발 가능

**위치:**
- `gui/renderers/fplt_renderer.py` line 1214-1228

## 해결 방안

### 1. 객체 제거 로직 최적화

**수정:**
```python
def _scene_remove(self, obj: Any) -> bool:
    """scene에서 객체 제거 (최적화)"""
    # 가장 안전한 방법 우선
    vb = getattr(self.ax_main, 'vb', None)
    if vb is not None:
        try:
            vb.removeItem(obj)
            return True
        except Exception:
            pass
    
    # fallback: scene에서 직접 제거
    try:
        scene = obj.scene()
        if scene is not None:
            scene.removeItem(obj)
            return True
    except Exception:
        pass
    
    return False
```

**효과:**
- 불필요한 속성 순회 제거
- 예외 처리 간소화

### 2. NaN 필터링 최적화

**수정:**
```python
def _upsert_st(self, name: str, x, y, ax, **kwargs) -> None:
    # 연속 NaN 구간 제거 (불필요한 선 단절 감소)
    mask = ~np.isnan(y)
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    # 필터링 후 데이터가 있는지 확인
    if len(x_filtered) == 0:
        return
    
    # upsert 로직...
```

**효과:**
- 불필요한 NaN 제거 → 렌더링 성능 향상

### 3. 캔들스틱 증분 업데이트

**수정:**
```python
def _upsert_candle(self, cdf: pd.DataFrame, ax) -> None:
    # 데이터 길이 변화 감지
    current_len = len(cdf)
    if current_len == self._last_df_len:
        # 길이 변화 없으면 마지막 봉만 업데이트
        existing = self._plots.get("_candle")
        if existing is not None:
            try:
                last_idx = cdf.index[-1]
                last_open = cdf.Open.iloc[-1]
                last_high = cdf.High.iloc[-1]
                last_low = cdf.Low.iloc[-1]
                last_close = cdf.Close.iloc[-1]
                existing.update_data([last_idx], [last_open], [last_high], [last_low], [last_close])
                self._last_df_len = current_len
                return
            except Exception:
                pass
    
    # 전체 재생성
    self._last_df_len = current_len
    # 기존 로직...
```

**효과:**
- 틱 업데이트 시 전체 재생성 방지

### 4. 피봇 마커 증분 업데이트

**수정:**
```python
def _plot_pivot_bucket(self, pm: Dict[str, Any], ax) -> None:
    # 피봇 해시 비교로 변경 감지
    pm_hash = self._calc_pivot_hash(pm)
    if pm_hash == self._last_pm_hash:
        return
    
    self._last_pm_hash = pm_hash
    
    # 변경된 피봇만 업데이트
    # 기존 로직...
```

**효과:**
- 피봇 변화 없으면 렌더링 스킵

### 5. MA 표시/숨김 로직 개선

**수정:**
```python
def set_ma_enabled(self, enabled: bool) -> None:
    self._ma_enabled = enabled
    for name in ("_ma20", "_ma60"):
        obj = self._plots.get(name)
        if obj is not None:
            try:
                obj.setOpacity(1.0 if enabled else 0.0)
            except Exception as e:
                logger.debug("[FpltRenderer] setOpacity 실패 (%s): %s", name, e)
                # _remove 대신 setVisible 사용
                try:
                    obj.setVisible(enabled)
                except Exception:
                    if not enabled:
                        self._remove(name)
```

**효과:**
- _remove fallback 최소화 → 깜빡임 감소

## 우선순위

1. **높음**: 캔들스틱 증분 업데이트 (틱 업데이트 성능)
2. **중간**: 피봇 마커 증분 업데이트 (피봇 변화 감지)
3. **중간**: 객체 제거 로직 최적화 (성능 향상)
4. **낮음**: NaN 필터링 최적화 (시각적 개선)
5. **낮음**: MA 표시/숨김 로직 개선 (깜빡임 감소)

## 테스트 방법

1. 실시간 틱 업데이트 시 캔들 재생성 빈도 확인
2. 피봇 변화 없을 때 렌더링 스킵 확인
3. 마커 제거 시 성능 측정
4. NaN 데이터 포함 시 렌더링 성능 확인
5. MA 표시/숨김 시 깜빡임 확인
