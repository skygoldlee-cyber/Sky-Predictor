# train.py (PriceTransformer Training)

## 역할

Transformer 모델(`prediction.model.PriceTransformer`)을 NPZ dataset으로 학습하고 `.pt` 가중치를 저장합니다.

## 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `set_seed(seed=42)` | function | 재현성을 위한 seed 설정(torch optional) |
| `load_data(path)` | function | NPZ에서 `X`, `y` 로드 |
| `run(args)` | function | 학습 루프(로깅/검증/체크포인트 저장) |
| `main()` | function | CLI 엔트리포인트(args 파싱 후 `run`) |

## config 정합성 체크(중요)

`--config config.json`을 읽어서 다음을 검증합니다.

- `adaptive_indicator.enabled`에 따라 기대 `feature_dim`을 계산
  - `expected_dim = len(OB_KEYS)+len(CD_KEYS)+len(OPT_KEYS)+ (len(ADAPT_KEYS) if enabled else 0)`
- dataset의 `X.shape[-1]`와 불일치하면 즉시 `ValueError`
- `config.prediction.seq_len`과 dataset `seq_len` 불일치도 best-effort로 검출

## 출력

- `--out`에 `.pt` 저장
- `--tag-date` 또는 현재 날짜를 suffix로 붙인 dated checkpoint도 추가 저장(best-effort)
