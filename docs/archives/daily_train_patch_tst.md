# 매일 장 종료 후 PatchTST 재훈련 절차

> 대상 프로젝트: `Patch_PST`  
> 작업 디렉토리: `C:\Patch_PST`  
> 훈련 스크립트: `train_patch_tst.py`

---

## 전체 흐름

```
장 종료 (15:45)
    │
    ▼
① 오늘 tick 로그 → 일별 dataset 빌드
    │  ticks_replay_YYYYMMDD_*.jsonl → datasets\dataset_YYYYMMDD.npz
    │
    ▼
② 최근 N일 dataset 병합
    │  dataset_20250101.npz ~ dataset_YYYYMMDD.npz → dataset_merged.npz
    │
    ▼
③ PatchTST 훈련
    │  dataset_merged.npz → patch_tst_5m.pt
    │
    ▼
④ 스냅샷 복사 (만기주 동결용)
       patch_tst_5m.pt → transformer_5m_YYYYMMDD.pt
```

---

## Step 1 — 오늘 tick 로그 → 일별 dataset 빌드

장 중 `ebest_live.py --out-ticks` 옵션으로 기록된 tick 파일(`ticks_replay_YYYYMMDD_HHMMSS.jsonl`)을 오늘 날짜 기준으로 dataset으로 변환한다.

```bat
python -m prediction.data_builder ^
    --files ticks_replay_%TODAY%_*.jsonl ^
    --out datasets\dataset_%TODAY%.npz ^
    --seq-len 60 ^
    --horizon 5 ^
    --config config.json
```

| 옵션 | 값 | 설명 |
|---|---|---|
| `--files` | `ticks_replay_%TODAY%_*.jsonl` | 오늘 날짜 tick 파일 (`.gz` 압축도 그대로 지정 가능) |
| `--out` | `datasets\dataset_%TODAY%.npz` | 날짜별로 개별 저장 |
| `--seq-len` | `60` | `config.json` `data.seq_len` 과 일치 |
| `--horizon` | `5` | 5분 후 방향 예측 |
| `--config` | `config.json` | `option_feature_set: v4` 등 피처 설정 자동 반영 |

> **주의**: `config.json`의 `option_feature_set`이 `v4`이므로 `--config config.json`을 반드시 지정해야 feature_dim 불일치 오류가 발생하지 않는다.

---

## Step 2 — 최근 N일 dataset 병합

일별로 쌓인 `.npz` 파일을 훈련용 하나의 파일로 합친다.

```bat
python merge_datasets.py ^
    --pattern "datasets\dataset_????????.npz" ^
    --last 20 ^
    --out datasets\dataset_merged.npz
```

| 옵션 | 값 | 설명 |
|---|---|---|
| `--pattern` | `datasets\dataset_????????.npz` | 날짜 패턴으로 전체 선택 |
| `--last` | `20` | 가장 최근 20거래일만 사용 |
| `--out` | `datasets\dataset_merged.npz` | 훈련에 사용할 병합 파일 |

`merge_datasets.py`는 **만기 롤오버를 자동 감지**한다. 만기일 다음 첫 거래일부터 `.rollover_start_YYYYMM.txt` 마커를 생성하고, 마커 이전 데이터를 자동으로 제외한다. 별도 설정 없이 동작한다.

---

## Step 3 — PatchTST 훈련

```bat
python train_patch_tst.py ^
    --data datasets\dataset_merged.npz ^
    --out prediction\weights\patch_tst_5m.pt ^
    --epochs 60 ^
    --batch-size 256 ^
    --lr 1e-3 ^
    --patch-len 8 ^
    --stride 4 ^
    --n-layers 3 ^
    --d-ff 256 ^
    --patience 10 ^
    --monitor acc ^
    --tag-date %TODAY%
```

| 옵션 | 값 | 설명 |
|---|---|---|
| `--patch-len` | `8` | **`config.json`의 `patch_len`과 반드시 일치** |
| `--stride` | `4` | **`config.json`의 `stride`와 반드시 일치** |
| `--n-layers` | `3` | PatchTST 기본 (Transformer보다 1층 더) |
| `--d-ff` | `256` | PatchTST 기본 (Transformer는 128) |
| `--patience` | `10` | 10 에폭 개선 없으면 조기 종료 |
| `--monitor` | `acc` | `val_acc` 기준 best 체크포인트 저장 |
| `--tag-date` | `%TODAY%` | 체크포인트에 날짜 기록 (추적용) |

훈련 완료 후 `prediction\weights\patch_tst_5m_history.csv`에 에폭별 `train_loss` / `val_acc` / `val_brier` 이력이 저장된다.

---

## Step 4 — 날짜 스냅샷 복사 (만기주 동결용)

`weights_selector.py`는 만기주(월~목) 진입 시 라이브 모델을 고정하기 위해 `transformer_5m_YYYYMMDD.pt` 패턴의 스냅샷을 자동으로 탐색한다. `config.json`에서 `transformer_weights_path`에 PatchTST 경로를 연결했으므로 스냅샷도 동일한 네이밍 규칙을 따라야 한다.

```bat
copy /Y prediction\weights\patch_tst_5m.pt ^
         prediction\weights\transformer_5m_%TODAY%.pt
```

스냅샷이 없으면 만기주 동결 기능이 비활성화되고 매일 재훈련된 가중치가 그대로 사용된다.

---

## 전체 배치 파일 (`daily_train.bat`)

아래 파일을 프로젝트 루트(`C:\Patch_PST`)에 저장하고, Windows 작업 스케줄러로 **장 종료 후(예: 16:00)** 실행을 등록한다.

```bat
@echo off
setlocal

:: ── 날짜 설정 ──────────────────────────────────────────────────────────────
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set TODAY=%DT:~0,8%
echo [%TODAY%] 일별 훈련 시작

cd /d C:\Patch_PST

:: ── Step 1: 일별 dataset 빌드 ──────────────────────────────────────────────
echo [1/4] 오늘 tick 데이터 -^> dataset
python -m prediction.data_builder ^
    --files ticks_replay_%TODAY%_*.jsonl ^
    --out datasets\dataset_%TODAY%.npz ^
    --seq-len 60 --horizon 5 --config config.json
if errorlevel 1 ( echo [ERROR] data_builder 실패 & exit /b 1 )

:: ── Step 2: 최근 20일 병합 ─────────────────────────────────────────────────
echo [2/4] 최근 20일 병합
python merge_datasets.py ^
    --pattern "datasets\dataset_????????.npz" ^
    --last 20 ^
    --out datasets\dataset_merged.npz
if errorlevel 1 ( echo [ERROR] merge 실패 & exit /b 1 )

:: ── Step 3: PatchTST 훈련 ───────────────────────────────────────────────────
echo [3/4] PatchTST 훈련
python train_patch_tst.py ^
    --data datasets\dataset_merged.npz ^
    --out prediction\weights\patch_tst_5m.pt ^
    --epochs 60 --batch-size 256 --lr 1e-3 ^
    --patch-len 8 --stride 4 ^
    --n-layers 3 --d-ff 256 ^
    --patience 10 --monitor acc ^
    --tag-date %TODAY%
if errorlevel 1 ( echo [ERROR] 훈련 실패 & exit /b 1 )

:: ── Step 4: weights_selector 호환 스냅샷 복사 ──────────────────────────────
echo [4/4] 날짜 스냅샷 저장
copy /Y prediction\weights\patch_tst_5m.pt ^
         prediction\weights\transformer_5m_%TODAY%.pt

echo [완료] %TODAY% 훈련 정상 종료
endlocal
```

---

## 참고 — 주요 파일 경로 정리

| 파일 | 경로 | 설명 |
|---|---|---|
| tick 로그 | `ticks_replay_YYYYMMDD_*.jsonl` | 장 중 자동 생성 |
| 일별 dataset | `datasets\dataset_YYYYMMDD.npz` | Step 1 출력 |
| 병합 dataset | `datasets\dataset_merged.npz` | Step 2 출력, Step 3 입력 |
| 훈련 가중치 | `prediction\weights\patch_tst_5m.pt` | 라이브 모델 (config 연결됨) |
| 날짜 스냅샷 | `prediction\weights\transformer_5m_YYYYMMDD.pt` | 만기주 동결용 |
| 훈련 이력 | `prediction\weights\patch_tst_5m_history.csv` | 에폭별 지표 |
| 롤오버 마커 | `.rollover_start_YYYYMM.txt` | merge_datasets 자동 관리 |
