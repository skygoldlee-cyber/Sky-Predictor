# Training / Offline Pipeline Documentation Index

이 폴더는 프로젝트의 **오프라인 데이터셋 생성 및 학습(training)** 경로에서 사용하는 핵심 스크립트/모듈들의 **클래스/함수**를 모듈군별로 요약합니다.

- 대상 범위(핵심):
  - `prediction/data_builder.py` (ticks JSONL → dataset NPZ)
  - `train.py` (PriceTransformer 학습)
  - `train_tft.py` (TemporalFusionTransformer 학습)
  - `merge_datasets.py` (일별 NPZ 병합)
- 목적:
  - 데이터 생성/학습 경로를 수정할 때 영향 범위를 빠르게 찾기
  - runtime과 training의 feature_dim/seq_len/horizon 정합성 유지

## 문서 목록

- `data_builder.md`
  - `prediction/data_builder.py` (dataset 생성)
- `train_transformer.md`
  - `train.py` (Transformer 학습)
- `train_tft.md`
  - `train_tft.py` (TFT 학습)
- `datasets.md`
  - `merge_datasets.py` 및 NPZ shape/검증 규칙

## 전체 흐름(요약)

1. `ebest_live.py --out-ticks ...`로 ticks JSONL 생성
   - 기본적으로 `.jsonl.gz`로 스트리밍 압축 저장될 수 있음(`--compress-ticks` 기본 True)
   - `.jsonl`을 그대로 남기려면 `--no-compress-ticks`
2. `prediction/data_builder.py`로 dataset NPZ 생성
3. 필요 시 `merge_datasets.py`로 일별 NPZ 병합
4. `train.py` / `train_tft.py`로 가중치(`.pt`) 생성
5. 생성된 가중치는 runtime에서 `prediction/predictor.py` 또는 `prediction/weights_selector.py`를 통해 로드

## 테스트(스모크)

- 전체 스모크(우산) 테스트: `python -m pytest -q tests/test_smoke.py`
- 세부 스모크:
  - `tests/test_adaptive_indicator_smoke.py`
  - `tests/test_prediction_smoke.py`
