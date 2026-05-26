# prediction/data_builder.py (Dataset Builder)

## 역할

ticks 리플레이 로그(`ticks_replay_*.jsonl` / `ticks_replay_*.jsonl.gz`)를 읽어 supervised dataset NPZ를 생성합니다.

- `.jsonl.gz`는 압축 해제 없이 그대로 입력으로 사용할 수 있습니다.
- 디스크 로그 최적화(컴팩트 스키마)로 `offerho*`/`bidho*`가 `가격*100` 정수로 저장된 경우, dataset build 시 자동으로 float 가격으로 복원됩니다.

- 입력(best-effort)
  - `FC0` records: `tick.price`, `tick.chetime`
  - `FH0` records: 오더북 스냅샷(→ `calc_orderbook_features`)
  - `OC0` records: 옵션 스냅샷(→ `build_option_snapshot`)
  - `OH0` records: 옵션 호가 스냅샷(옵션 미세구조 피처 보강; bid/ask/depth)
- 출력
  - Transformer용: `X`, `y`
  - TFT용(`--tft`): `X`, `y`, `past_known`, `future_known`
- feature 구성
  - OB(7) + CD(5) + OPT(v1=7 | v2=16)
  - + ADAPT(28) (config에서 `adaptive_indicator.enabled=true`일 때)

## 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `_load_jsonl(path)` | function | JSONL 레코드 스트림 로드 |
| `_hhmm_to_minutes(hhmm)` | function | HHMM → 분 단위 변환(내부 헬퍼) |
| `_to_dt_minute(chetime)` | function | tick 시간 → minute datetime(floor) |
| `_extract_ts_epoch(chetime)` | function | tick 시간 → epoch seconds(best-effort) |
| `build_dataset(files, seq_len, horizon_min, tft, tft_horizon_sec, config_path)` | function | 메인 dataset 빌더(리턴: X,y[,PK,FK]) |

## config 연동 포인트

- `--config`로 `config.json`을 로드하여 다음을 결정:
  - `adaptive_indicator.enabled`에 따라 ADAPT(28) 포함 여부 결정
  - `adaptive_indicator.warmup_bars`로 오프라인 adaptive 피처 산출 warmup 길이 결정
  - `prediction.option_feature_set`에 따라 OPT 피처 셋 결정
    - `v1`: 기존 OPT(7)
    - `v2`: 확장 OPT(기존 7 + option_minute_ohlcv 기반 미세움직임 9)

또한 `option_minute_ohlcv.enabled=true`이면 dataset build 중에도 `OC0`로 옵션 분봉 OHLCV가 집계되어,
`v2`의 `optm_*` 피처가 0이 아닌 값으로 채워질 가능성이 높아집니다.

> ⚠️ `option_feature_set`이 바뀌면 `feature_dim`이 달라지므로, dataset 재생성 및 재학습이 필요합니다.

### OPT(v2) 추가 키(9개)

`option_feature_set="v2"`일 때 추가되는 피처는 아래 9개입니다.

- `optm_call_ret`
- `optm_put_ret`
- `optm_straddle_ret`
- `optm_call_range_pct`
- `optm_put_range_pct`
- `optm_straddle_range_pct`
- `optm_call_vol`
- `optm_put_vol`
- `optm_straddle_vol`

## 출력 스키마(요약)

- Transformer 모드(`--tft` 미지정)
  - `X`: `(N, seq_len, feature_dim)`
  - `y`: `(N,)` (0=down, 1=up)

- TFT 모드(`--tft`)
  - `X`: `(N, seq_len, past_unknown_dim)`
  - `past_known`: `(N, seq_len, FUTURE_KNOWN_DIM)`
  - `future_known`: `(N, tft_horizon_sec, FUTURE_KNOWN_DIM)`
  - `y`: `(N,)`
