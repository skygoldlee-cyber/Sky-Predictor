# 머신러닝 예측 시스템 가이드

SkyPredictor의 머신러닝 예측 엔진 구조와 Transformer/TFT/LLM 파이프라인을 통합한 가이드입니다.

> **참고 문서**: 상세 기술 문서는 아래 문서들을 참조하세요.
> - [TFT_DUAL_MODEL_DESIGN_GUIDE.md](TFT_DUAL_MODEL_DESIGN_GUIDE.md) - TFT 상세 설계
> - [guides/MODELS_GUIDE.md](guides/MODELS_GUIDE.md) - 모델 사용 가이드

---

## 개요

SkyPredictor는 다양한 머신러닝 모델을 조합하여 시장 방향을 예측합니다. 사용자는 `numeric_predictor`와 `model_class` 설정을 통해 예측 엔진을 구성할 수 있습니다.

### 시스템 구조

```
실시간 틱 / 분봉
    → TickMixin (FO0 버퍼, 분봉 DF)
    → OptionMixin (옵션 스냅샷, OI, IV)
    → AdaptiveMixin (레짐·지표, 선택)
    → PredictionMixin.get_prediction()
         ├─ 수치: build_sequence + ModelInput → numeric_predictor.predict()
         ├─ 가드레일: 옵션/베이시스/패리티/블리드/OI·진폭
         ├─ LLM: 스냅샷·프롬프트 → 판단 (설정 시)
         └─ 피드백 큐 (앙상블 가중 갱신용)
```

핵심 클래스: `prediction/pipeline.py`의 `PredictionPipeline` (Mixin 조합).

---

## 예측 엔진 유형 (numeric_predictor)

### 1. Transformer (transformer)
단일 Transformer 모델을 사용하여 예측합니다.

**특징**:
- 시계열 Transformer 기반
- Attention 메커니즘으로 장기 의존성 학습
- 빠른 추론 속도

**사용 방법**:
```json
{
  "numeric_predictor": "transformer",
  "model_class": "transformer"
}
```

### 2. TFT (tft)
단일 Temporal Fusion Transformer 모델을 사용합니다.

**특징**:
- 시계열 특화 Transformer
- 멀티-헤드 어텐션과 게이팅 메커니즘
- 시간 기반 피처 지원

**사용 방법**:
```json
{
  "numeric_predictor": "tft",
  "model_class": "tft"
}
```

### 3. Ensemble (ensemble)
Transformer와 TFT를 결합한 앙상블 모델입니다.

**특징**:
- 여러 모델의 예측을 가중 평균
- `transformer_weight`로 가중치 조절
- 더 안정적인 예측

**사용 방법**:
```json
{
  "numeric_predictor": "ensemble",
  "model_class": "transformer",
  "transformer_weight": 0.5
}
```

### 4. Rule-based (rule_based)
ML 모델 없이 휴리스틱만 사용합니다.

**특징**:
- ML 모델 의존성 제거
- 빠른 응답 속도
- 테스트/디버깅용

---

## Transformer + LLM 역할 분리

5분 가격예측 시스템에서 두 모델의 역할을 명확히 분리합니다.

| 모델 | 역할 | 입력 | 출력 |
|------|------|------|------|
| **NumericPredictor (Transformer/TFT/Ensemble)** | 수치 예측 | 오더북 + 분봉(+옵션+adaptive) 시계열, (TFT는 time features 포함) | 상승 확률 (0~1) + signal/confidence |
| **AdaptiveIndicatorManager** | 시장 국면/구조 판단 | 분봉 OHLCV | heuristic action + regime + ADAPT_KEYS(28) |
| **LLM** | 해석·판단 | 예측 결과 + 시장 컨텍스트 | 전략 판단 텍스트 |

> **현행 구현**: 현재 저장소는 `prediction/PredictionPipeline` 기반으로 동작하며,
> `prediction/predictor.py`는 **Transformer + TFT + Ensemble** 수치 예측기를 포함하며,
> 가중치가 존재하면 torch inference를 사용하고, 가중치가 없거나 torch 미설치 시 rule-based fallback으로 동작합니다.
> LLM은 `prediction/llm_judge.py`에서 **Claude/GPT/Gemini** 를 모두 지원하며, 실패 시 provider fallback 합니다.
> `dual_llm=true` 시 GPT + Gemini를 동시 호출하고, `dual_llm_primary_provider`(기본 `"gpt"`) 결과를 최종 판단에 반영합니다.
> **LLM 실패·타임아웃** 시 `prediction.heuristic_fallback`(기본 `true`)에 따라 adaptive 휴리스틱이 최종 `llm_action`을 보강할 수 있다(`prediction/llm_mixin.py`).

---

## 수치 모델 입력 (`ModelInput`)

| 필드 | 의미 |
|------|------|
| `sequence` | 오더북·캔들·옵션·(선택)적응/멀티스케일 피처로 만든 `(seq_len, feature_dim)` 배열 |
| `past_known` | TFT용 과거 time features (초, 요일, 월 등) |
| `future_known` | TFT용 미래 time features (예측 horizon 동안) |
| `seq_len` | 시퀀스 길이 (초 단위, 1Hz 기준) |

---

## Transformer 모델 구조

### 아키텍처
```python
# prediction/model.py (핵심 발췌)
from config import PAST_UNKNOWN_DIM  # = 86 (v4+adaptive) 또는 94 (v4+adaptive+multiscale)

class PriceTransformer(nn.Module):
    """
    5분 방향 예측 Transformer.
    입력:  (batch, seq_len, PAST_UNKNOWN_DIM)
    출력:  (batch, 1)  — sigmoid 전 logit
    """
    def __init__(
        self,
        feature_dim: int = PAST_UNKNOWN_DIM,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_proj = nn.Linear(d_model, 1)
```

### 피처 레이아웃
- **OB_KEYS (7)**: 오더북 불균형 피처
- **CD_KEYS (5)**: 캔들 데이터 (OHLCV)
- **OPT_KEYS (29)**: 옵션 피처 (v4 기준)
- **ADAPT_KEYS (28)**: 적응형 지표 피처
- **TIME_KEYS (11)**: 시간 기반 피처
- **MS5_KEYS (8)**: 멀티스케일 5분봉 피처 (선택)

---

## TFT 모델 구조

### TFT 학습
```python
# train_tft.py (실제 파일 — 주요 부분 발췌)

from config import FUTURE_KNOWN_DIM, HORIZON_SEC, PAST_UNKNOWN_DIM
from prediction.tft_model import TemporalFusionTransformer

# NPZ 로드 (X, past_known, future_known, y)
data = np.load("dataset_tft_5m.npz")
X   = torch.tensor(data["X"],            dtype=torch.float32)   # (N, seq_len, 16)
PK  = torch.tensor(data["past_known"],   dtype=torch.float32)   # (N, seq_len, 11)
FK  = torch.tensor(data["future_known"], dtype=torch.float32)   # (N, horizon, 11)
y   = torch.tensor(data["y"],            dtype=torch.float32)   # (N,)

N, seq_len, past_unknown_dim = X.shape
```

---

## 가드레일 시스템

### 옵션 유동성 가드레일
- OI 집중도 기반 필터링
- IV 기반 피봇 확인
- 콜/풋 스큐 확인

### 베이시스 가드레일
- 선물-옵션 베이시스 모니터링
- 과대/과소 평가 판단
- 거래 제한

### 패리티 다이버전스
- 콜-풋 스큐 모니터링
- 페널티 기반 조정

---

## LLM 판단 시스템

### LLM Judge
- Claude/GPT/Gemini 지원
- 스냅샷 기반 판단
- Provider fallback

### Dual LLM 모드
- GPT + Gemini 동시 호출
- `dual_llm_primary_provider` 기반 최종 판단
- 일치도 확인

---

## 운영 체크리스트

### 1. 모델 배포
- [ ] 가중치 파일 존재 확인
- [ ] Torch 설치 확인
- [ ] Config 설정 확인

### 2. 피처 확인
- [ ] OB/CD/OPT 피처 정상
- [ ] ADAPT 피처 활성화 확인
- [ ] 멀티스케일 피처 확인

### 3. LLM 설정
- [ ] API 키 설정
- [ ] Provider 설정
- [ ] Dual LLM 모드 확인

### 4. 가드레일
- [ ] 옵션 유동성 가드레일
- [ ] 베이시스 가드레일
- [ ] 패리티 다이버전스

---

## 자주 발생하는 문제와 해결 방법

### 1. 모델 로딩 실패
**증상**: `torch.load()` 실패
**해결**: 가중치 파일 경로 확인, torch 버전 확인

### 2. 피처 차원 불일치
**증상**: `RuntimeError: size mismatch`
**해결**: `option_feature_set` 설정 확인, 피처 재생성

### 3. LLM 타임아웃
**증상**: `TimeoutError` 또는 429 에러
**해결**: `gemini_timeout_sec` 설정, fallback 활성화

### 4. 메모리 부족
**증상**: `CUDA out of memory`
**해결**: 배치 사이즈 감소, 모델 크기 축소
