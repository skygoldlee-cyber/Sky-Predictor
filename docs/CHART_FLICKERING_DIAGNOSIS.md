# 차트 렌더링 깜박임 문제 진단

## 문제 개요

차트 뷰어에서 렌더링 깜박임(flickering) 현상이 발생합니다.

## 진단 결과

### 1. refresh_ms 설정 문제

**현재 설정:**
- `ChartViewerConfig.refresh_ms = 500` (0.5초)
- `DEFAULT_REFRESH_MS = 5000` (5초)

**문제:**
- 기본 설정값이 500ms로 너무 짧음
- 0.5초마다 자동 갱신 시도 → 깜박임 유발

**위치:**
- `gui/chart_viewer.py` line 77
- `gui/chart_viewer.py` line 100

### 2. 캐시 TTL과 refresh_ms 불일치

**현재 설정:**
- `cache_ttl = 5.0` (5초)
- `refresh_ms = 500` (0.5초)

**문제:**
- 캐시는 5초 유효하지만 0.5초마다 갱신 시도
- 캐시가 유효하더라도 갱신 로직이 실행됨

**위치:**
- `gui/chart_viewer.py` line 82
- `gui/utils/cache_manager.py` line 13

### 3. 자동 갱신 로직

**현재 로직:**
```python
def _auto_refresh_callback(self) -> None:
    # 새로운 데이터 수신 확인
    if not self._new_data_received:
        return
    self.refresh()
```

**문제:**
- 실시간 데이터 수신 시 `_new_data_received`가 계속 True 상태
- 0.5초마다 `refresh()` 호출 → 깜박임

**위치:**
- `gui/chart_viewer.py` line 2293-2333

### 4. 캐시 무효화 빈도

**현재 로직:**
- 범위 변경 시 `_clear_cache()` 호출
- 플롯 변경 시 `_clear_cache()` 호출
- CSV 로드 시 `_clear_cache()` 호출

**문제:**
- 캐시가 자주 삭제됨 → 재계산 빈도 증가

**위치:**
- `gui/chart_viewer.py` line 838, 865, 972

## 해결 방안

### 1. refresh_ms 기본값 조정 ✅ 적용 완료

**수정:**
```python
# gui/chart_viewer.py line 77
class ChartViewerConfig:
    refresh_ms: int = 2000  # [FIX-FLICKER-1] 500 → 2000 (2초)

# gui/chart_viewer.py line 100
DEFAULT_REFRESH_MS = 2000  # [FIX-FLICKER-1] 5000 → 2000 (2초)
```

**효과:**
- 갱신 주기 증가 → 깜박임 감소

### 2. 캐시 TTL과 refresh_ms 동기화 ✅ 적용 완료

**수정:**
```python
# gui/chart_viewer.py line 82
cache_ttl: float = 2.0  # [FIX-FLICKER-2] 5.0 → 2.0 (refresh_ms와 동기화)
```

**효과:**
- 캐시 유효 시간과 갱신 주기 일치

### 3. 데이터 수신 플래그 최적화 ✅ 적용 완료

**수정:**
```python
# gui/chart_viewer.py line 2293
def _auto_refresh_callback(self) -> None:
    # [FIX-FLICKER-3] 최소 갱신 간격 체크
    if hasattr(self, '_last_refresh_time'):
        elapsed = time.monotonic() - self._last_refresh_time
        min_interval = self._refresh_ms / 1000.0
        if elapsed < min_interval:
            return
    
    if not self._new_data_received:
        return
    
    self._last_refresh_time = time.monotonic()
    self.refresh()
```

**효과:**
- 최소 갱신 간격 보장 → 과도한 갱신 방지

### 4. 캐시 무효화 최소화 ⏭️ 건너뜀

**사유:**
- 해당 함수 `_on_minutes_changed`가 존재하지 않음
- 캐시 키 변경 시 자동으로 새로운 캐시 사용됨

## 우선순위

1. **높음**: refresh_ms 기본값 조정 ✅ 적용 완료
2. **중간**: 캐시 TTL과 refresh_ms 동기화 ✅ 적용 완료
3. **중간**: 데이터 수신 플래그 최적화 ✅ 적용 완료
4. **낮음**: 캐시 무효화 최소화 ⏭️ 건너뜀 (함수 없음)

## 테스트 방법

1. 현재 설정으로 차트 실행 → 깜박임 확인
2. refresh_ms를 2000ms로 변경 → 깜박임 감소 확인 ✅
3. 캐시 TTL을 2.0으로 변경 → 성능 확인 ✅
4. 데이터 수신 플래그 최적화 적용 → 깜박임 제거 확인 ✅
