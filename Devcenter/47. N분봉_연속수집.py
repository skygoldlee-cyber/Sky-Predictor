import asyncio
import ebest
import json
from pathlib import Path
from common import *

import pandas as pd

# config.secrets.json에서 읽어오기
secrets_path = Path(__file__).parent.parent / 'config.secrets.json'
with open(secrets_path, 'r', encoding='utf-8') as f:
    secrets = json.load(f)
appkey = secrets.get('ebest', {}).get('appkey', '')
appsecretkey = secrets.get('ebest', {}).get('appsecretkey', '')

'''
선물/지수 분봉 연속조회 수집.

[사용법]
1. config.secrets.json에 eBest API 키(appkey, appsecretkey) 설정
2. 스크립트 실행: python "47. 1분봉_연속수집.py"
3. 분 단위 입력 (예: 1=1분봉, 5=5분봉, 60=60분봉)
4. 각 종목별로 수집할 건수 입력 (빈칸=스킵)
   - KP 200 연결지수선물: 건수 입력
   - KOSPI 지수: 건수 입력
5. 데이터 자동 필터링 및 CSV 저장

[특징]
- 선물(KP 200 연결지수선물 90199999): t8465 사용 (구 t8415 폐지)
- 지수(KOSPI 001): t8418 사용 (업종 차트 TR)
- 연속조회로 과거 데이터 최대한 수집
- 일자별 데이터 건수 자동 판별(최빈값 기준)
- 불완전한 데이터 자동 제거
- CSV 파일 저장:
  - KOSPI: kospi_YYYYMMDD_{ncnt}min.csv
  - 선물: kp200_YYYYMMDD_{ncnt}min.csv
  - 저장 위치: Devcenter/data/

[참고]
- 연결지수가 "분봉을 얼마나 깊게 보관하는지"는 서버 정책상 불확실하므로,
  매 요청의 최과거 타임스탬프를 찍어 실제 보관 깊이를 눈으로 확인한다.
- 분봉이 얕게 나오면(연결지수 한계) → 월물별로 받아 직접 롤오버 결합이 정공법.
- 1분봉 기준: KOSPI 약 381건/일, 선물 약 411건/일
'''


async def GetFutureMinuteChartData(api, code, count, ncnt=1):
    '''
    선물 분봉 데이터 연속조회.
    code : 종목코드 (연결지수선물 = 90199999)
    count: 목표 수집 건수
    ncnt : 분 단위 (1 = 1분봉)
    return: DataFrame[time(date), hhmm, open, high, low, close, volume]
    '''
    received_count = 0
    cts_date = ''
    cts_time = ''
    tr_cont = 'N'
    tr_cont_key = '0'
    all_data = []
    req_frame_count = 0

    while received_count < count:
        req_frame_count += 1
        req_count = min(500, count - received_count)
        request = {
            't8465InBlock': {
                'shcode': code,
                'ncnt': ncnt,          # 분 단위
                'qrycnt': req_count,   # 요청건수(최대 500)
                'nday': '',            # '' : sdate~edate 범위
                'sdate': '',           # 시작일자
                'stime': '',
                'edate': '99999999',   # 종료일자(최신부터)
                'etime': '',
                'cts_date': cts_date,  # 연속일자
                'cts_time': cts_time,  # 연속시간 (분봉은 시각까지 필요)
                'comp_yn': 'N',
            }
        }
        # t8415 폐지 → t8465 사용. 패키지 맵 미등록 대비 path 명시.
        response = await api.request(
            't8465', request,
            path='/futureoption/chart',
            tr_cont=tr_cont, tr_cont_key=tr_cont_key,
        )
        if not response:
            print(f'요청실패: {api.last_message}')
            break

        data = response.body.get('t8465OutBlock1', None)
        if not data:
            print(f'[{req_frame_count}] 데이터 없음 → 종료')
            break

        all_data = data + all_data
        received_count = len(all_data)

        # 이번 묶음의 최과거/최신 타임스탬프 = 보관 깊이 확인용
        oldest = data[0]
        newest = data[-1]
        print(
            f'[{req_frame_count:>3}] +{len(data):>3}건 누적 {received_count:>6} | '
            f'구간 {oldest.get("date")} {oldest.get("time")} ~ '
            f'{newest.get("date")} {newest.get("time")} | tr_cont={response.tr_cont}'
        )

        if received_count >= count:
            break

        out = response.body.get('t8465OutBlock', {}) or {}
        cts_date = str(out.get('cts_date', '')).strip()
        cts_time = str(out.get('cts_time', '')).strip()
        tr_cont = response.tr_cont
        tr_cont_key = response.tr_cont_key

        # 종료조건: 더 받을 데이터 없음 / cts 소진
        if tr_cont != 'Y' or not (cts_date or cts_time):
            print('연속조회 종료(서버 보관 한계 도달)')
            break

        await asyncio.sleep(1)  # 유량 제한 대비

    df = pd.DataFrame(
        [(x['date'], x['time'], float(x['open']), float(x['high']),
          float(x['low']), float(x['close']), float(x['jdiff_vol'])) for x in all_data],
        columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    )
    # 안전: (date,time) 중복 제거 후 시간순 정렬
    df = df.drop_duplicates(subset=['date', 'time']).sort_values(['date', 'time']).reset_index(drop=True)
    return df


async def GetIndexMinuteChartData(api, code, count, ncnt=1):
    '''
    지수 분봉 데이터 연속조회 (t8418 업종 차트 TR).
    code : 종목코드 (KOSPI = 001)
    count: 목표 수집 건수
    ncnt : 분 단위 (1 = 1분봉)
    return: DataFrame[time(date), hhmm, open, high, low, close, volume]
    '''
    received_count = 0
    cts_date = ''
    cts_time = ''
    tr_cont = 'N'
    tr_cont_key = '0'
    all_data = []
    req_frame_count = 0

    while received_count < count:
        req_frame_count += 1
        req_count = min(500, count - received_count)  # t8418 비압축 시 최대 500건
        request = {
            't8418InBlock': {
                'shcode': code,
                'ncnt': ncnt,          # 분 단위
                'qrycnt': req_count,   # 요청건수(최대 2000건 압축, 500건 비압축)
                'nday': '',            # 0:미사용 1이상:사용
                'sdate': '',           # 시작일자
                'stime': '',           # 현재 미사용
                'edate': '99999999',   # 종료일자(최신부터)
                'etime': '',           # 현재 미사용
                'cts_date': cts_date,  # 연속일자
                'cts_time': cts_time,  # 연속시간
                'comp_yn': 'N',        # 압축여부 (N:비압축)
            }
        }
        response = await api.request(
            't8418', request,
            tr_cont=tr_cont, tr_cont_key=tr_cont_key,
        )
        if not response:
            print(f'요청실패: {api.last_message}')
            break

        data = response.body.get('t8418OutBlock1', None)
        if not data:
            print(f'[{req_frame_count}] 데이터 없음 → 종료')
            break

        all_data = data + all_data
        received_count = len(all_data)

        # 이번 묶음의 최과거/최신 타임스탬프 = 보관 깊이 확인용
        oldest = data[0]
        newest = data[-1]
        print(
            f'[{req_frame_count:>3}] +{len(data):>3}건 누적 {received_count:>6} | '
            f'구간 {oldest.get("date")} {oldest.get("time")} ~ '
            f'{newest.get("date")} {newest.get("time")} | tr_cont={response.tr_cont}'
        )

        if received_count >= count:
            break

        out = response.body.get('t8418OutBlock', {}) or {}
        cts_date = str(out.get('cts_date', '')).strip()
        cts_time = str(out.get('cts_time', '')).strip()
        tr_cont = response.tr_cont
        tr_cont_key = response.tr_cont_key

        # 종료조건: 더 받을 데이터 없음 / cts 소진
        if tr_cont != 'Y' or not (cts_date or cts_time):
            print('연속조회 종료(서버 보관 한계 도달)')
            break

        await asyncio.sleep(1)  # 유량 제한 대비

    df = pd.DataFrame(
        [(x['date'], x['time'], float(x['open']), float(x['high']),
          float(x['low']), float(x['close']), float(x['jdiff_vol'])) for x in all_data],
        columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    )
    # 안전: (date,time) 중복 제거 후 시간순 정렬
    df = df.drop_duplicates(subset=['date', 'time']).sort_values(['date', 'time']).reset_index(drop=True)
    return df


async def sample(api):
    ncnt_str = await ainput('분 단위 (예: 1=1분봉, 5=5분봉, 60=60분봉, 빈칸=종료): ')
    if len(ncnt_str) == 0:
        return
    if not ncnt_str.isdigit():
        return
    ncnt = int(ncnt_str) or 1

    # 선물 → KOSPI 순으로 수집
    targets = [
        {'code': '90199999', 'name': 'KP 200 연결지수선물', 'prefix': 'kp200_', 'func': GetFutureMinuteChartData},
        {'code': '001', 'name': 'KOSPI 지수', 'prefix': 'kospi_', 'func': GetIndexMinuteChartData},
    ]

    for target in targets:
        shcode = target['code']
        name = target['name']
        file_prefix = target['prefix']
        func = target['func']

        print(f'\n{"="*60}')
        print(f'{name} {ncnt}분봉 데이터 수집 시작 ({shcode})')
        print(f'{"="*60}')

        count_str = await ainput(f'{name} 조회할 {ncnt}분봉 건수(ex 5000, 빈칸=스킵): ')
        if len(count_str) == 0:
            print('스킵\n')
            continue
        if not count_str.isdigit():
            print('잘못된 입력, 스킵\n')
            continue
        count = int(count_str) or 5000

        df = await func(api, shcode, count, ncnt=ncnt)
        if df.empty:
            print('수집 결과 없음\n')
            continue

        print(f'\n총 {len(df)}건')
        print(f'최과거: {df.iloc[0]["date"]} {df.iloc[0]["time"]}')
        print(f'최신  : {df.iloc[-1]["date"]} {df.iloc[-1]["time"]}')
        # 보관 깊이 = 실제로 받힌 일자 범위. 분봉이 얕으면 여기서 드러남.
        print(f'수집 일자수: {df["date"].nunique()}일')

        # 하루 단위로 정리
        print('\n=== 일자별 데이터 건수 ===')
        daily_counts = df.groupby('date').size().reset_index(name='count')
        for _, row in daily_counts.iterrows():
            date = row['date']
            count = row['count']
            print(f'{date}: {count}건')

        # 가장 많이 나오는 건수를 기준으로 완전한 데이터 판단
        if not daily_counts.empty:
            mode_count = daily_counts['count'].mode()
            if not mode_count.empty:
                target_count = int(mode_count.iloc[0])
                print(f'\n기준 건수(최빈값): {target_count}건')

                # 완전한 데이터(기준 건수)만 필터링
                complete_dates = daily_counts[daily_counts['count'] == target_count]['date'].tolist()
                df_filtered = df[df['date'].isin(complete_dates)].copy()
                removed_count = len(df) - len(df_filtered)
                print(f'불완전한 데이터 제거: {removed_count}건')
                print(f'필터링 후 총 {len(df_filtered)}건 ({len(complete_dates)}일)')

                if not df_filtered.empty:
                    print(f'필터링 후 최과거: {df_filtered.iloc[0]["date"]} {df_filtered.iloc[0]["time"]}')
                    print(f'필터링 후 최신  : {df_filtered.iloc[-1]["date"]} {df_filtered.iloc[-1]["time"]}')
                else:
                    print('필터링 후 데이터가 없습니다.')
                    print('')
                    continue
            else:
                print('데이터 건수 패턴을 확인할 수 없습니다.')
                df_filtered = df.copy()
                complete_dates = daily_counts['date'].tolist()
        else:
            print('데이터가 없습니다.')
            df_filtered = df.copy()
            complete_dates = []

        # 날짜별로 OHLCV 형태의 CSV로 저장
        print('\n=== CSV 저장 ===')
        output_dir = Path(__file__).parent / 'data'
        output_dir.mkdir(exist_ok=True)

        for date in complete_dates:
            df_date = df_filtered[df_filtered['date'] == date].copy()
            filename = output_dir / f'{file_prefix}{date}_{ncnt}min.csv'
            df_date.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f'{filename}: {len(df_date)}건 저장')

        print(f'\n총 {len(complete_dates)}개 파일 저장 완료')
        print('')


async def main():
    api = ebest.OpenApi()
    if not await api.login(appkey, appsecretkey):
        return print(f'연결실패: {api.last_message}')
    await sample(api)
    await api.close()


if __name__ == '__main__':
    asyncio.run(main())
