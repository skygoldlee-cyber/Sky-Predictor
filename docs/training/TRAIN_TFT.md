# train_tft.py (TemporalFusionTransformer Training)

## 역할

TFT 모델(`prediction.tft_model.TemporalFusionTransformer`)을 TFT용 NPZ dataset으로 학습하고 `.pt` 가중치를 저장합니다.

## 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `set_seed(seed=42)` | function | 재현성을 위한 seed 설정(torch optional) |
| `load_data(path)` | function | NPZ에서 `X`, `past_known`, `future_known`, `y` 로드 |
| `run(args)` | function | 학습 루프(로깅/검증/체크포인트 저장) |
| `main()` | function | CLI 엔트리포인트(args 파싱 후 `run`) |

## config 정합성 체크(중요)

`--config config.json`을 읽어서 다음을 검증합니다.

- `adaptive_indicator.enabled`에 따라 기대 `past_unknown_dim` 계산
- `config.prediction.seq_len`과 dataset `seq_len` 정합성 확인(best-effort)
- `config.prediction.tft_horizon`과 dataset `future_known` horizon 정합성 확인(best-effort)

추가로 코드 상수와의 정합성도 확인합니다.

- `PAST_UNKNOWN_DIM`, `FUTURE_KNOWN_DIM`, `HORIZON_SEC`와 dataset 차원 불일치 시 `ValueError`

## 출력

- `--out`에 `.pt` 저장
- 날짜 태깅된 checkpoint도 추가 저장(best-effort)
