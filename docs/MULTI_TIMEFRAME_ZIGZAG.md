# 다중 시간프레임 지그재그 구현 가이드

## 개요

다중 시간프레임 지그재그(Multi-Timeframe ZigZag)는 여러 시간프레임(1분봉, 5분봉, 15분봉)에서 독립적으로 피봇을 감지하고, 이를 결합하여 신뢰도 높은 피봇을 식별하는 기능입니다.

## 목적

- 단일 시간프레임의 노이즈 감소
- 다중 시간프레임 합의(Consensus)를 통한 신호 신뢰도 향상
- 거짓 피봇(False Pivot) 필터링

## 아키텍처

### 구성 요소

```
┌─────────────────────────────────────────────────────────────┐
│                     AdaptiveZigZag                          │
│  (1분봉 메인 인스턴스)                                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  _multi_tf_zz: MultiTimeframeZigZag                 │  │
│  │  - 피봇 합의도 확인                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  _upper_tf_zz_instances: Dict[int, AdaptiveZigZag]   │  │
│  │  - 5분봉 ZigZag 인스턴스                             │  │
│  │  - 15분봉 ZigZag 인스턴스                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  _upper_tf_data_buffers: Dict[int, List[Dict]]      │  │
│  │  - 5분봉 데이터 버퍼 (1분봉 5개)                     │  │
│  │  - 15분봉 데이터 버퍼 (1분봉 15개)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
1분봉 데이터 입력
    ↓
AdaptiveZigZag.update()
    ↓
┌─────────────────────────────────────────────────────────────┐
│  _update_upper_timeframe_data()                              │
│  - 1분봉 데이터를 버퍼에 누적                                 │
│  - 버퍼가 상위 시간프레임 봉 수만큼 쌓이면 리샘플링           │
│    (5분봉: 5개, 15분봉: 15개)                                │
│  - 상위 시간프레임 ZigZag.update() 호출                      │
│  - 상위 시간프레임 피봇 캐시 업데이트                        │
└─────────────────────────────────────────────────────────────┘
    ↓
피봇 확정 시
    ↓
┌─────────────────────────────────────────────────────────────┐
│  _check_multiframe_consensus()                               │
│  - MultiTimeframeZigZag.check_consensus() 호출               │
│  - 상위 시간프레임 피봇 캐시에서 매칭 검색                   │
│  - 합의도 계산 (일치하는 시간프레임 수 / 전체 시간프레임 수)│
│  - 합의도 미통과 시 피봇 필터링                               │
└─────────────────────────────────────────────────────────────┘
```

## 구현 상세

### 1. 초기화 (AdaptiveZigZag.__init__)

```python
# 다중 시간프레임 설정
if config.multi_timeframe_enabled:
    # MultiTimeframeZigZag 초기화
    self._multi_tf_zz = MultiTimeframeZigZag(
        scales=[5, 15],  # 상위 시간프레임
        consensus_threshold=2,  # 합의도 임계값
        price_tolerance_pct=1.0,  # 가격 허용 오차
        index_tolerance_multiplier=2.0  # 인덱스 허용 오차 배수
    )
    
    # 상위 시간프레임 ZigZag 인스턴스 생성
    for scale in [5, 15]:
        # 무한 재귀 방지를 위해 다중 시간프레임 비활성화
        upper_config = AdaptiveZigZagConfig()
        upper_config.multi_timeframe_enabled = False
        upper_zz = AdaptiveZigZag(config=upper_config)
        self._upper_tf_zz_instances[scale] = upper_zz
        self._upper_tf_data_buffers[scale] = []
```

### 2. 상위 시간프레임 데이터 업데이트

```python
def _update_upper_timeframe_data(self, high, low, close, bar_time, open, volume):
    """1분봉 데이터를 상위 시간프레임으로 리샘플링"""
    
    for scale, zz in self._upper_tf_zz_instances.items():
        # 버퍼에 데이터 누적
        buffer = self._upper_tf_data_buffers[scale]
        buffer.append({'high': high, 'low': low, 'close': close, ...})
        
        # 버퍼가 상위 시간프레임 봉 수만큼 쌓이면 리샘플링
        if len(buffer) >= scale:
            # 리샘플링
            resampled = self._resample_buffer(buffer, scale)
            # 상위 시간프레임 ZigZag 업데이트
            zz.update(**resampled)
            # 피봇 캐시 업데이트
            self._update_upper_tf_pivot_cache(scale, zz)
            # 버퍼 초기화
            self._upper_tf_data_buffers[scale] = []
```

### 3. 피봇 합의도 확인

```python
def _check_multiframe_consensus(self, pivot_index, pivot_price, pivot_type, current_close):
    """피봇에 대한 다중 시간프레임 합의도 확인"""
    
    if self._multi_tf_zz is None:
        return True  # 비활성화 시 항상 통과
    
    # 합의도 확인
    result = self._multi_tf_zz.check_consensus(
        pivot_index=pivot_index,
        pivot_price=pivot_price,
        pivot_type=pivot_type
    )
    
    # 합의도 미통과 시 필터링
    if not result['passed']:
        logger.warning("피봇 합의도 부족으로 필터링")
        return False  # 피봇 확정 취소
    
    return True  # 통과
```

### 4. 피봇 매칭 로직 (MultiTimeframeZigZag)

```python
def _find_matching_pivot(self, base_index, base_price, base_type, scale_pivots, scale):
    """기준 피봇과 일치하는 상위 시간프레임 피봇 찾기"""
    
    for pivot in scale_pivots:
        # 피봇 타입 일치 확인
        if pivot['pivot_type'] != base_type:
            continue
        
        # 가격 유사도 확인 (±1.0% 허용)
        price_diff = abs(pivot['price'] - base_price) / base_price
        if price_diff > 0.01:
            continue
        
        # 인덱스 범위 확인 (상위 시간프레임 2봉 범위 허용)
        index_diff = abs(pivot['index'] - base_index)
        max_index_diff = scale * 2
        if index_diff > max_index_diff:
            continue
        
        return pivot  # 매칭 성공
    
    return None  # 매칭 실패
```

## 설정 파라미터

### config.json

```json
{
  "adaptive_indicator": {
    "kospi_zigzag": {
      "multi_timeframe_enabled": true,
      "multi_timeframe_scales": [1, 5, 15],
      "multi_timeframe_consensus_threshold": 2,
      "multi_timeframe_price_tolerance_pct": 1.0,
      "multi_timeframe_index_tolerance_multiplier": 2.0
    }
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `multi_timeframe_enabled` | bool | false | 다중 시간프레임 기능 활성화 여부 |
| `multi_timeframe_scales` | List[int] | [1, 5, 15] | 확인할 시간프레임 목록 (분 단위) |
| `multi_timeframe_consensus_threshold` | int | 2 | 합의도 임계값 (이상일 때만 신호 통과) |
| `multi_timeframe_price_tolerance_pct` | float | 1.0 | 가격 허용 오차 (%) |
| `multi_timeframe_index_tolerance_multiplier` | float | 2.0 | 인덱스 허용 오차 배수 (시간프레임 × 배수) |

## 파라미터 튜닝 가이드

### 합의도 임계값 (consensus_threshold)

- **1**: 가장 민감한 설정, 하나의 상위 시간프레임만 일치해도 통과
- **2**: 균형 잡힌 설정, 2개 이상의 상위 시간프레임 일치 필요
- **3**: 보수적인 설정, 모든 상위 시간프레임 일치 필요

추천: 2 (균형 잡힌 설정)

### 가격 허용 오차 (price_tolerance_pct)

- **0.5%**: 엄격한 가격 일치 요구
- **1.0%**: 표준 설정 (추천)
- **2.0%**: 넓은 가격 허용 오차

추천: 1.0% (표준)

### 인덱스 허용 오차 (index_tolerance_multiplier)

- **1.0**: 엄격한 인덱스 일치 요구
- **2.0**: 표준 설정 (추천)
- **3.0**: 넓은 인덱스 허용 오차

추천: 2.0 (표준)

## 성능 최적화

### 캐싱 메커니즘

1. **피봇 캐시 시그니처**
   - 마지막 피봇의 인덱스와 가격으로 시그니처 생성
   - 동일 시그니처 시 캐시 업데이트 스킵

2. **성능 카운터**
   - `_check_count`: 합의도 확인 총 횟수
   - `_cache_hit_count`: 캐시 히트 횟수
   - `get_performance_stats()`: 성능 통계 반환

### 메모리 최적화

- 버퍼는 상위 시간프레임 봉 수만큼만 유지
- 피봇 캐시는 확정 피봇만 저장
- 불필요한 데이터 즉시 삭제

## 사용 예시

### 기본 사용

```python
from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig

# 설정
config = AdaptiveZigZagConfig(
    multi_timeframe_enabled=True,
    multi_timeframe_scales=[1, 5, 15],
    multi_timeframe_consensus_threshold=2
)

# 인스턴스 생성
zz = AdaptiveZigZag(config)

# 데이터 업데이트
for high, low, close, open, volume in data:
    signal = zz.update(high, low, close, open=open, volume=volume)
    if signal == "new_high":
        print("새로운 고점 피봇 확정")
    elif signal == "new_low":
        print("새로운 저점 피봇 확정")
```

### 합의도 필터링 로그

```
[MultiTF] H 피봇 합의도 통과: index=100, price=105.00, consensus=2/2 (100.0%)
[MultiTF] L 피봇 합의도 부족으로 필터링: index=150, price=100.00, consensus=1/2 (50.0%)
```

## 주의사항

### 무한 재귀 방지

상위 시간프레임 ZigZag 인스턴스는 반드시 다중 시간프레임 기능을 비활성화해야 합니다:

```python
upper_config.multi_timeframe_enabled = False
```

### 데이터 요구사항

최소 데이터 요구량:
- **5분봉**: 최소 50개 1분봉 (5분봉 10개)
- **15분봉**: 최소 150개 1분봉 (15분봉 10개)

실제 운영 추천:
- **5분봉**: 100개 이상의 1분봉
- **15분봉**: 300개 이상의 1분봉

### 버퍼링 지연

- 상위 시간프레임 봉이 완성되기까지 지연 발생
- 5분봉: 최대 5분 지연
- 15분봉: 최대 15분 지연

## 테스트

### 백테스트 스크립트

```bash
python scripts/test_multiframe_zigzag.py
```

### 테스트 항목

1. 기본 다중 시간프레임 기능 테스트
2. 합의도 확인 테스트
3. 설정 로드 테스트
4. 성능 통계 테스트

## 문제 해결

### 합의도가 항상 0인 경우

**원인**: 상위 시간프레임 데이터 부족
**해결**: 충분한 데이터 확보 (최소 300봉 이상)

### 무한 재귀 오류

**원인**: 상위 시간프레임 인스턴스에 다중 시간프레임 활성화
**해결**: 상위 시간프레임 인스턴스의 `multi_timeframe_enabled = False`

### 피봇이 너무 많이 필터링됨

**원인**: 합의도 임계값이 너무 높음
**해결**: `consensus_threshold` 낮추기 (1 또는 2)

## 향후 개선 사항

1. 시간 기반 리샘플링 (현재는 봉 수 기반)
2. 가중치 합의도 (상위 시간프레임에 더 높은 가중치)
3. 실시간 합의도 모니터링
4. 합의도 기반 알림 시스템
