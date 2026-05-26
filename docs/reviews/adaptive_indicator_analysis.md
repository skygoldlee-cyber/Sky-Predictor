Adaptive Indicator 비교 분석 보고서
Transformer Project  vs  SkyEbest Project
Adaptive SuperTrend · Adaptive ZigZag 알고리즘 차이 및 개선 권고


1. 분석 개요
두 프로젝트(Transformer, SkyEbest)는 각각 독립적으로 Adaptive SuperTrend와 Adaptive ZigZag 지표를 구현하고 있습니다. 핵심 알고리즘은 공유하지 않고 코드를 별도로 유지하고 있어 시간이 지남에 따라 구현 차이가 누적되고 있습니다.
이 보고서는 두 구현 간의 의미 있는 차이를 분류하고, 각 차이의 영향도를 평가한 뒤 통합/개선 방향을 권고합니다.

항목	Transformer	SkyEbest
파일 위치	adaptive_indicator/adaptive_supertrend.py adaptive_indicator/adaptive_zigzag.py	views/charts/UnifiedTA.py (통합)
WilderRMA 의존	별도 wilder_smooth.py 모듈	동일 (동일 모듈 참조)
기본 파라미터	동일 (atr_min=7, atr_max=21, mult 1.5~4.0)	동일
compute_from_df 기본 컬럼명	소문자 (high, low, close)	대문자 (High, Low, Close)


2. Adaptive SuperTrend — 구현 차이 상세
2-1. ATR 재초기화 임계값 [영향도: 중]
ATR 기간이 크게 바뀔 때 평균값으로 재초기화하는 기준값이 두 구현에서 다릅니다.

구분	조건 (재초기화 실행)	의미
Transformer	period_change > atr_max_period × 0.3 → 기본값 기준 21×0.3=6봉 이상 변경 시	더 민감하게 재초기화 (자주 발생)
SkyEbest	period_change_ratio > 0.5 → 이전 기간 대비 50% 이상 변경 시	더 보수적 재초기화 (덜 발생)

예) ATR 기간이 14→8로 바뀔 경우:
•	Transformer: |14-8|=6 > 21×0.3=6.3 → 재초기화 안 함 (경계선)
•	SkyEbest: |14-8|/14=0.43 < 0.5 → 재초기화 안 함
•	ATR 기간이 21→7로 급변하면: Transformer는 |21-7|=14 > 6.3 → 재초기화, SkyEbest는 14/21=0.67 > 0.5 → 재초기화
두 방식 모두 논리적이나 Transformer 방식(절댓값 기준)이 atr_max_period 설정 변경 시 의도치 않게 민감도가 달라질 수 있습니다.

▶ 권고: 비율 기반으로 통일
SkyEbest의 비율 기반 방식(50%)이 기간 범위 설정과 무관하게 일관된 동작을 보장합니다.
Transformer도 period_change_ratio > 0.5 방식으로 통일하는 것을 권장합니다.

2-2. bars_in_trend 증가 타이밍 버그 [영향도: 높음]
트렌드 지속 봉수 카운트에 결정적인 로직 차이가 있습니다.

Transformer (버그 있음)
# 플립 발생 시 bars_in_trend = 0 으로 리셋
self._state.bars_in_trend = 0   # buy 신호
...
self._state.bars_in_trend += 1  # 리셋 직후 무조건 +1 실행
→ 플립 봉에서 bars_in_trend가 0으로 리셋된 뒤 즉시 1로 증가. 의도한 동작이지만, Transformer feature "ast_trend_duration"에서는 플립 봉이 이미 1로 시작함.

SkyEbest (개선됨)
just_flipped = True
if not just_flipped:
    self._state.bars_in_trend += 1
→ just_flipped 플래그를 활용해 플립 봉에서는 증가를 건너뜀. 플립 봉의 bars_in_trend=0이 보존됨.

▶ 권고: Transformer를 SkyEbest 방식으로 수정
플립 봉에서 bars_in_trend=0 상태가 Transformer feature에 노출되어야 신호 강도 해석이 정확합니다.
SkyEbest의 just_flipped 패턴을 Transformer adaptive_supertrend.py에 적용하세요.
수정 위치: update() 메서드 신호 생성 이후 ~ 상태 업데이트 전

2-3. 신호 감지 방식 차이 [영향도: 낮음]
구분	방식	비고
Transformer	prev_dir 변수 사용 (직전 봉 방향 저장)	코드 일부 중복
SkyEbest	self._direction[-2] 직접 참조	더 간결, 단 direction 버퍼 2개 필요
두 방식은 동일한 결과를 생성하므로 기능적 차이 없음. 단, SkyEbest는 len(self._direction) >= 2 조건 확인이 필요해 방어 코드가 더 요구됩니다.

2-4. compute_from_df 기본 컬럼명 불일치 [영향도: 중]
⚠️ 주의: API 호환성 문제
Transformer: high_col='high', low_col='low', close_col='close' (소문자)
SkyEbest: high_col='High', low_col='Low', close_col='Close' (대문자)
두 프로젝트에서 동일한 DataFrame을 처리할 경우 컬럼명 불일치로 AttributeError가 발생합니다.
→ 공통 전처리 유틸리티를 만들거나 단일 컨벤션으로 통일하세요.

2-5. LLM 컨텍스트 텍스트 차이 [영향도: 낮음]
Transformer의 get_llm_context()는 ER 해석 문장과 ATR 기간 설명이 더 상세합니다. SkyEbest는 market advice (uptrend/downtrend) 매핑에 오류가 있습니다.

⚠️ SkyEbest LLM 컨텍스트 버그
advice 딕셔너리 키가 'uptrend'/'downtrend'인데, s.trend_strength 값은 'weak'/'neutral'/'strong'입니다.
→ .get(s.trend_strength, ...) 가 항상 기본값(횡보 구조)을 반환합니다.
수정: s.structure 또는 s.direction을 기반으로 advice를 생성해야 합니다.


3. Adaptive ZigZag — 구현 차이 상세
3-1. pending_confirm 교체 조건 [영향도: 중]
스윙 확정 대기(pending_confirm) 창이 이미 존재할 때 새로운 전환을 감지하면 교체 여부 결정 로직이 다릅니다.

구분	교체 조건	효과
Transformer	pending_confirm이 None이거나 / 다른 타입(high↔low)이면 교체 → 같은 타입이면 교체 안 함 (기존 유지)	동일 방향 연속 신호 시 첫 번째 피크 보존
SkyEbest	pending_confirm is None일 때만 새로 등록 → 이미 있으면 교체 없이 스킵	단순하지만 연속 전환 시 두 번째 신호 무시
Transformer 방식이 더 정교합니다. 상승→하락 전환 도중 더 높은 고점이 나타나면 replace 조건 덕분에 더 정확한 스윙 고점을 포착합니다.
SkyEbest는 pending_confirm 이미 존재 시 단순 스킵하므로, 빠른 반전 시장에서 스윙 포인트를 누락할 수 있습니다.

▶ 권고: SkyEbest를 Transformer 방식으로 개선
SkyEbest의 elif self._current_direction == 1: 블록에서
if self._pending_confirm is None: → if self._pending_confirm is None or self._pending_confirm.get('type') != 'high': 로 변경
동일하게 direction==-1 블록도 'low' 타입 기준으로 수정

3-2. ATR 초기화 방식 차이 [영향도: 낮음]
구분	방식
Transformer	명시적 _atr_initialized 플래그 + WilderRMA.ready 확인
SkyEbest	_atr_initialized 없이 WilderRMA 결과 직접 사용, _prev_atr만 유지
SkyEbest가 더 단순하며, WilderRMA 내부적으로 워밍업을 처리하므로 기능상 동일합니다.

3-3. _calc_fibonacci() 키 관리 [영향도: 낮음]
구분	피보나치 딕셔너리 키
Transformer	fib_236, fib_382 등 새로운 키만 사용 (레거시 키 FibLevels 클래스로 변환)
SkyEbest	두 키 모두 등록: '0.382'(레거시) + 'fib_382'(신규)
SkyEbest의 이중 키 방식이 하위 호환성을 보장하지만 딕셔너리 크기가 2배가 됩니다. Transformer는 FibLevels 래퍼 클래스로 레거시 키 접근을 지원합니다.

3-4. compute_from_df — azz_fib_0382 별칭 [영향도: 낮음]
Transformer의 compute_from_df()는 azz_fib_0382, azz_fib_0618 별칭 컬럼을 추가로 출력합니다. SkyEbest는 해당 별칭 없음. Transformer feature pipeline(features.py)이 이 컬럼명에 의존하고 있다면 SkyEbest로 교체 시 KeyError가 발생할 수 있습니다.


4. 공통 알고리즘 개선 권고
4-1. SuperTrend EMA 스무딩의 구조적 문제 [영향도: 높음]
현재 양 구현 모두 방향 전환(flip) 발생 후 스무딩을 적용하고 있습니다. 이는 플립 직후 SuperTrend 라인이 연속적으로 보이지만, 실제 판정 밴드(raw band)와 스무딩된 st_value 사이에 괴리가 발생합니다.
•	direction 판정은 raw lower_band/upper_band 기반으로 수행
•	st_value(출력값)는 스무딩된 EMA 기반
•	use_smooth_for_features=False(기본)일 때 feature는 raw band 사용 → 일관성 유지됨
•	use_smooth_for_features=True로 바꾸면 feature와 판정 기준이 달라져 학습 데이터 오염 가능

▶ 권고: use_smooth_for_features=False 유지 (기본값 OK)
스무딩은 시각화 목적으로만 사용하고 Transformer feature는 항상 raw band 기준을 유지하세요.
LLM 컨텍스트에서도 'SuperTrend 라인' 표시값과 'feature 거리' 기준이 다를 수 있음을 주석으로 명시하세요.

4-2. ZigZag 미확정(pending) 스윙의 Repainting 위험 [영향도: 높음]
현재 구현에서 pending_confirm은 확정되기 전 스윙 후보를 추적하며, confirmation_bars(기본 2봉)가 지난 후 확정됩니다. 그러나:
•	pending_confirm 처리 중에 현재 봉의 고점/저점으로 candidate를 업데이트하므로, 확정 시점에 원래 전환 봉이 아닌 이후 봉의 가격이 기록될 수 있습니다.
•	이는 Repainting(소급 수정)으로 이어져 백테스트 결과와 실시간 결과가 다를 수 있습니다.
•	Transformer는 이를 인지하여 코드 주석에 "repainting mitigation"이라고 표기했으나 완전히 해결된 상태는 아닙니다.

▶ 개선 방안: 확정 후 candidate 업데이트 분리
pending_confirm 윈도우가 열린 후에는 candidate price를 업데이트하지 않고 고정하는 옵션을 추가하세요.
설정: freeze_on_confirm: bool = True (기본값으로 제안)
또는 confirmation_bars=0으로 설정하여 즉시 확정 (노이즈가 늘어나는 tradeoff)

4-3. ADX 초기값 25.0의 영향 [영향도: 낮음]
워밍업 기간(14봉 미만) 동안 ADX가 25.0으로 고정되어 있어 multiplier가 중간값으로 고정됩니다. 이는 초기 봉에서 적응 동작이 비활성화되는 효과를 냅니다. 일반적으로 허용 가능하지만, 데이터가 짧은 세션 초기(장 시작 직후)에 예상치 못한 멀티플라이어가 적용될 수 있습니다.
•	대안: ADX 대신 ER만으로 멀티플라이어 보정 (워밍업 필요 없음)
•	또는 워밍업 중 multiplier_max로 고정 (보수적 필터링)

4-4. ZigZag 구조 분석의 lookback 한계 [영향도: 낮음]
_analyze_structure()는 최근 8개 스윙에서 3개씩 고점/저점을 추출합니다. 활발한 시장에서는 이 창이 너무 짧아 HH-HL 구조를 조기에 'ranging'으로 판정할 수 있습니다.
•	개선: max_swings를 30으로 늘리고 lookback_bars 파라미터로 외부에서 조정 가능하게 만드세요.


5. 개선 우선순위 요약

우선순위	대상	항목	조치
🔴 즉시	Transformer SuperTrend	bars_in_trend 플립 봉에서 +1 누적 버그	just_flipped 패턴 적용
🔴 즉시	SkyEbest SuperTrend	LLM advice 딕셔너리 키 오류	s.structure 기반으로 수정
🟠 권장	SkyEbest ZigZag	pending_confirm 교체 조건 단순화	Transformer 방식으로 개선
🟠 권장	Transformer SuperTrend	ATR 재초기화 임계값 차이	비율 기반(50%)으로 통일
🟡 중기	양쪽	compute_from_df 컬럼명 불일치	소문자 통일 또는 공통 전처리
🟡 중기	양쪽	ZigZag Repainting	freeze_on_confirm 옵션 추가
⚪ 선택	양쪽	ADX 초기값 / 구조 분석 lookback	파라미터 외부화


6. 통합 아키텍처 권고
현재 두 프로젝트가 동일 알고리즘을 별도 파일로 유지하는 구조는 장기적으로 유지보수 부담을 증가시킵니다. 다음과 같은 단계적 통합을 권고합니다.

1단계: 즉각 버그 수정 (이번 주)
•	Transformer: bars_in_trend just_flipped 패턴 적용
•	SkyEbest: LLM advice 키 수정, pending_confirm 교체 조건 개선

2단계: 공통 라이브러리 추출 (1~2주)
•	Transformer의 adaptive_indicator/ 패키지를 기준으로 삼고
•	SkyEbest를 pip 설치 또는 git submodule로 참조
•	SkyEbest 전용 기능(calculate(), ZigZagPoint 변환)은 어댑터 클래스로 래핑

3단계: 통합 테스트 추가 (지속)
•	두 프로젝트에서 동일 입력 → 동일 출력 검증 테스트
•	Repainting 여부 확인 테스트 (과거 봉 소급 변경 없음 검증)


— END OF REPORT —
