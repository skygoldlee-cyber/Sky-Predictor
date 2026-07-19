# 예측 모델 구조 정리

> 작성일: 2026-07-19
> 목적: SkyPredictor 프로젝트의 ML 예측 모델 파일/모듈 구조를 단일 문서로 정리하고, import 경로를 일관되게 맞춘다.

---

## 1. 모델 분류

| 구분 | 모델 | 출력 | 용도 |
|------|------|------|------|
| **방향 예측** | PriceTransformer | P(up) 이진 확률 | 기본 추론 모델 |
| **방향 예측** | PatchTSTModel | P(up) 이진 확률 | 연산 효율/국소 패턴 |
| **방향 예측** | MambaModel | P(up) 이진 확률 | 장기 의존성 |
| **방향 예측** | TemporalFusionTransformer | P(up) 이진 확률 | 해석 가능성(VSN) |
| **피봇 예측** | PivotConfirmationClassifier | 확정 확률 | 피봇 후보 확정/취소 분류 |
| **피봇 예측** | PivotProbabilityRegressor | 확정 확률 | 확률 직접 회귀 |
| **피봇 예측** | PivotLifespanPredictor | 잔여 봉수 | 후보 수명 예측 |
| **피봇 예측** | PivotEnsemble | 확정 확률 | 분류+회귀 앙상블 |

---

## 2. 파일/모듈 위치

```
prediction/
├── models/                     # 방향 예측 딥러닝 모델 패키지
│   ├── __init__.py             # 4개 모델 re-export
│   ├── model.py                # PriceTransformer
│   ├── patch_tst_model.py      # PatchTSTModel
│   ├── mamba_model.py          # MambaModel
│   └── tft_model.py            # TemporalFusionTransformer
│
├── pivot_models.py             # 피봇 예측 모델 (classification/regression/lifespan/ensemble)
│   (주의: prediction/models/pivot_models.py 와 다름)
│
├── predictor.py                # 런타임 모델 로드/추론 (방향 예측)
├── pivot_inference.py          # 피봇 추론 래퍼
├── pipeline.py                 # 전체 예측 파이프라인
└── ...

training/
├── train.py                    # PriceTransformer 학습
├── train_patch_tst.py          # PatchTSTModel 학습
├── train_mamba.py              # MambaModel 학습
└── train_tft.py                # TemporalFusionTransformer 학습
```

---

## 3. Import 경로 표준

### 3.1 모델 클래스 import (권장)

```python
from prediction.models import (
    PriceTransformer,
    PatchTSTModel,
    MambaModel,
    TemporalFusionTransformer,
)
```

### 3.2 개별 파일 import

```python
from prediction.models.model import PriceTransformer
from prediction.models.patch_tst_model import PatchTSTModel
from prediction.models.mamba_model import MambaModel
from prediction.models.tft_model import TemporalFusionTransformer
from prediction.pivot_models import (
    PivotConfirmationClassifier,
    PivotProbabilityRegressor,
    PivotLifespanPredictor,
    PivotEnsemble,
)
```

---

## 4. 학습 스크립트 ↔ 모델 매핑

| 학습 스크립트 | 모델 클래스 | 가중치 기본 경로 |
|---------------|------------|------------------|
| `training/train.py` | `PriceTransformer` | `prediction/weights/transformer_5m.pt` |
| `training/train_patch_tst.py` | `PatchTSTModel` | `prediction/weights/patch_tst_5m.pt` |
| `training/train_mamba.py` | `MambaModel` | `prediction/weights/mamba_4h.pt` |
| `training/train_tft.py` | `TemporalFusionTransformer` | `prediction/weights/tft_5m.pt` |

---

## 5. 런타임 모델 선택

`config.json`의 `prediction.model_class` 또는 `predictor.TransformerPredictor`의 `model_class` 인자로 선택한다.

```json
{
  "prediction": {
    "model_class": "transformer"
  }
}
```

| model_class 값 | 사용 모델 |
|----------------|----------|
| `"transformer"` | `PriceTransformer` |
| `"patch_tst"` | `PatchTSTModel` |
| `"mamba"` | `MambaModel` |
| TFT는 `TFTPredictor` 별도 경로 사용 | `TemporalFusionTransformer` |

---

## 6. 데이터셋 요구사항

- PriceTransformer / PatchTSTModel / MambaModel: 동일 `.npz` (X, y) 사용 가능
- TemporalFusionTransformer: 별도 `.npz` (X, past_known, future_known, y) 필요
- data_builder `--tft --tft-horizon-sec` 플래그로 TFT 데이터셋 생성

---

## 7. 수정 이력

### 2026-07-19

- `prediction/patch_tst_model.py` → `prediction/models/patch_tst_model.py` 이동
- `prediction/models/__init__.py`의 `PatchTSTModel` import 경로 수정
- `prediction/models/patch_tst_model.py`의 `from constants import PAST_UNKNOWN_DIM` → `from config import PAST_UNKNOWN_DIM` 수정
- 학습 스크립트들의 모델 import 경로를 `prediction.models.*`로 통일
  - `training/train.py`
  - `training/train_patch_tst.py`
  - `training/train_mamba.py`
  - `training/train_tft.py`
- 관련 문서의 import 경로 일괄 수정

---

## 8. 관련 문서

- `docs/archives/ml/MODELS_GUIDE.md`
- `docs/ML_PREDICTION_GUIDE.md`
- `docs/TFT_DUAL_MODEL_DESIGN_GUIDE.md`
- `Devcenter/ml/ml_dynamic_model_selection_guide.md` (별도 XGBoost 기반 동적 모델 선택)
