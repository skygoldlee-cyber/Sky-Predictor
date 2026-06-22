# Dataset Utilities / Merge

## 1) merge_datasets.py

### 역할

일별 dataset NPZ(`dataset_*.npz` 또는 `dataset_tft_*.npz`)들을 선택/검증/병합하여 학습용 단일 NPZ로 만듭니다.

- 파일 선택
  - `--pattern`으로 glob
  - `--last N`으로 최근 N개 날짜 선택
  - rollover reset 로직으로 만기 이후 구간만 선택 가능
- shape 검증
  - `X`는 3D, `y`는 1D
  - TFT 모드에서는 `past_known`, `future_known`도 3D이며 N/seq_len 정합성 검증

### 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `_extract_yyyymmdd(path)` | function | 파일명에서 YYYYMMDD 추출 |
| `_select_last_n(files, n)` | function | 날짜 기준 최근 N개 선택 |
| `_most_recent_expiry(now)` | function | 최근 만기(2nd Thu) 계산 |
| `_detect_rollover_start(now, files, marker_val)` | function | rollover 시작일을 관측 기반으로 탐지 |
| `_filter_by_rollover(files, rollover_start)` | function | rollover 이전 데이터 제외 |
| `_load_npz(path)` | function | NPZ 로드 |
| `_validate_and_collect(npz_list, tft)` | function | shape 검증 후 배열 목록 수집 |
| `merge(files, tft, out, max_samples)` | function | concat 병합 후 저장 |
| `main()` | function | CLI 엔트리포인트 |

## 2) NPZ 기본 스키마(요약)

- Transformer dataset
  - `X`: `(N, seq_len, feature_dim)`
  - `y`: `(N,)`

- TFT dataset
  - `X`: `(N, seq_len, past_unknown_dim)`
  - `past_known`: `(N, seq_len, FUTURE_KNOWN_DIM)`
  - `future_known`: `(N, horizon_sec, FUTURE_KNOWN_DIM)`
  - `y`: `(N,)`
