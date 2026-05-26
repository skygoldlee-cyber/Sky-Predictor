@echo off
setlocal

:: 날짜 설정
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set TODAY=%DT:~0,8%
echo [%TODAY%] 일별 훈련 시작

cd /d C:\Patch_PST

:: Step 1: 일별 dataset 빌드
echo [1/4] 오늘 tick 데이터 → dataset
python -m prediction.data_builder ^
    --files ticks_replay_%TODAY%_*.jsonl ^
    --out datasets\dataset_%TODAY%.npz ^
    --seq-len 60 --horizon 5 --config config.json
if errorlevel 1 ( echo [ERROR] data_builder 실패 & exit /b 1 )

:: Step 2: 최근 20일 병합
echo [2/4] 최근 20일 병합
python merge_datasets.py ^
    --pattern "datasets\dataset_????????.npz" ^
    --last 20 ^
    --out datasets\dataset_merged.npz
if errorlevel 1 ( echo [ERROR] merge 실패 & exit /b 1 )

:: Step 3: PatchTST 훈련
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

:: Step 4: weights_selector 호환 스냅샷 복사
echo [4/4] 날짜 스냅샷 저장
copy /Y prediction\weights\patch_tst_5m.pt ^
         prediction\weights\transformer_5m_%TODAY%.pt

echo [완료] %TODAY% 훈련 정상 종료
endlocal