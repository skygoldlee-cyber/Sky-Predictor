"""
장마감 이후 t8415/t8418 데이터 수집 스크립트

config.json의 target_date를 사용하여 해당 날짜의 분봉 데이터를 수집하고 CSV로 저장합니다.
- t8415: KOSPI200 선물 분봉 데이터
- t8418: KOSPI 지수 분봉 데이터

TR 코드 변경 주의사항:
- 일부 환경에서는 t8415 → t8465, t8418 → t8427로 변경되었을 수 있음
- ebest API 문서를 확인하여 현재 사용 가능한 TR 코드 확인 필요
- TR 코드 변경 시 해당 함수의 request("t8415", req) 부분 수정 필요

사용법:
    python scripts/fetch_daily_data.py                              # config.json의 target_date 사용
    python scripts/fetch_daily_data.py --target-date 20260514       # 인자로 target_date 지정
    python scripts/fetch_daily_data.py --start-date 20260501 --end-date 20260514  # 날짜 범위 지정 (연속 조회)
    python scripts/fetch_daily_data.py --force                       # 장마감 전에도 강제 실행
    python scripts/fetch_daily_data.py --target-date 20260514 --force  # 인자 + 강제 실행
    python scripts/fetch_daily_data.py --start-date 20260501 --end-date 20260514 --force  # 범위 + 강제 실행

요구사항:
    - config.secrets.json에 ebest 자격증명 (appkey, appsecretkey) 설정 필요
    - config.json에 target_date 설정 필요 (인자로 지정 시 생략 가능)
    - kp200_upcode는 API에서 자동 조회, kospi_upcode는 '001' 고정
    - ebest 모듈 (ebest.OpenApi) 설치 필요
    - ebest API 연결 가능한 환경 필요
    - data/daily_bars/ 디렉토리에 CSV 저장

출력 파일:
    - 단일 날짜: data/daily_bars/minute_bars_kp200_{target_date}.csv
    - 날짜 범위: data/daily_bars/minute_bars_kp200_{start_date}_{end_date}.csv

config.secrets.json 예시:
    {
      "ebest": {
        "appkey": "your_appkey",
        "appsecretkey": "your_appsecretkey"
      }
    }

config.json 예시:
    {
      "ebest": {
        "target_date": "20260508"
      }
    }
"""

import asyncio
import argparse
import json
import sys
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ebestapi.api import _ebest_fetch_kp200_ohlcv_t8415, _ebest_fetch_kospi_ohlcv_t8418, _ebest_fetch_kp200_symbol

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class AiohttpFilter(logging.Filter):
    """aiohttp Unclosed session 경고 필터"""
    def filter(self, record):
        # Unclosed 관련 경고 메시지 필터링
        message = record.getMessage()
        if 'Unclosed' in message or 'client_session' in message or 'connector' in message:
            return False
        return True


# 루트 로거에 필터 적용 (모든 하위 로거에 영향)
root_logger = logging.getLogger()
root_logger.addFilter(AiohttpFilter())

# aiohttp 및 asyncio 관련 로거 레벨 설정
for logger_name in ['aiohttp', 'aiohttp.client', 'aiohttp.internal', 'asyncio']:
    aiohttp_logger = logging.getLogger(logger_name)
    aiohttp_logger.setLevel(logging.CRITICAL)

# ResourceWarning 억제 (aiohttp Unclosed session 경고)
warnings.filterwarnings("ignore", category=ResourceWarning, message="Unclosed.*session")


async def fetch_kp200_continuous(api_client, symbol: str, start_date: str, end_date: str, ncnt: int = 1):
    """연속 조회로 KOSPI200 선물 분봉 데이터 수집 (날짜 범위)"""
    all_rows = []
    cts_date = ""
    cts_time = ""
    page = 1
    
    while True:
        logger.info(f"KP200 연속 조회... page={page}")
        
        # ebest API 요청
        req = {
            "t8415InBlock": {
                "shcode": symbol,
                "ncnt": ncnt,
                "qrycnt": 500,  # 최대 조회 건수
                "nday": "0",
                "sdate": start_date,
                "edate": end_date,
                "cts_date": cts_date,
                "cts_time": cts_time,
                "comp_yn": "N"
            }
        }
        
        try:
            res = await api_client.request("t8415", req)
            body = res.get("body", {})
            rows = body.get("t8415OutBlock1", [])
            
            if len(rows) == 0:
                logger.info("KP200 연속 조회 완료 (더 이상 데이터 없음)")
                break
            
            all_rows.extend(rows)
            logger.info(f"KP200 수신={len(rows)} 누적={len(all_rows)}")
            
            # 연속 조회 파라미터 업데이트
            outblock = body.get("t8415OutBlock", {})
            next_cts_date = outblock.get("cts_date", "")
            next_cts_time = outblock.get("cts_time", "")
            
            if next_cts_date == "":
                logger.info("KP200 연속 조회 완료 (cts_date 비어있음)")
                break
            
            cts_date = next_cts_date
            cts_time = next_cts_time
            page += 1
            
            # API 요청 간 딜레이
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logger.error(f"KP200 연속 조회 오류: {e}")
            break
    
    return all_rows


async def fetch_kospi_continuous(api_client, symbol: str, start_date: str, end_date: str, ncnt: int = 1):
    """연속 조회로 KOSPI 지수 분봉 데이터 수집 (날짜 범위)"""
    all_rows = []
    cts_date = ""
    cts_time = ""
    page = 1
    
    while True:
        logger.info(f"KOSPI 연속 조회... page={page}")
        
        # ebest API 요청
        req = {
            "t8418InBlock": {
                "idxcode": symbol,
                "ncnt": ncnt,
                "qrycnt": 500,  # 최대 조회 건수
                "nday": "0",
                "sdate": start_date,
                "edate": end_date,
                "cts_date": cts_date,
                "cts_time": cts_time,
                "comp_yn": "N"
            }
        }
        
        try:
            res = await api_client.request("t8418", req)
            body = res.get("body", {})
            rows = body.get("t8418OutBlock1", [])
            
            if len(rows) == 0:
                logger.info("KOSPI 연속 조회 완료 (더 이상 데이터 없음)")
                break
            
            all_rows.extend(rows)
            logger.info(f"KOSPI 수신={len(rows)} 누적={len(all_rows)}")
            
            # 연속 조회 파라미터 업데이트
            outblock = body.get("t8418OutBlock", {})
            next_cts_date = outblock.get("cts_date", "")
            next_cts_time = outblock.get("cts_time", "")
            
            if next_cts_date == "":
                logger.info("KOSPI 연속 조회 완료 (cts_date 비어있음)")
                break
            
            cts_date = next_cts_date
            cts_time = next_cts_time
            page += 1
            
            # API 요청 간 딜레이
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logger.error(f"KOSPI 연속 조회 오류: {e}")
            break
    
    return all_rows


async def fetch_and_save_daily_data(target_date: str = None, start_date: str = None, end_date: str = None):
    """config.json의 target_date 또는 인자로 받은 target_date/start_date/end_date로 t8415/t8418 데이터 수집 및 저장"""
    
    # config.json 로드 (target_date 등 일반 설정)
    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # config.secrets.json 로드 (자격증명)
    secrets_path = Path(__file__).parent.parent / 'config.secrets.json'
    try:
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = json.load(f)
    except FileNotFoundError:
        logger.warning("config.secrets.json 파일을 찾을 수 없습니다.")
        secrets = {}
    
    ebest_config = config.get('ebest', {})
    ebest_secrets = secrets.get('ebest', {})
    
    # 날짜 리스트 생성
    date_list = []
    
    if start_date and end_date:
        # 날짜 범위 처리
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            # 날짜 범위 생성 (주말 제외)
            current_dt = start_dt
            while current_dt <= end_dt:
                # 주말(토요일=5, 일요일=6) 제외
                if current_dt.weekday() < 5:  # 월요일(0) ~ 금요일(4)
                    date_list.append(current_dt.strftime('%Y%m%d'))
                current_dt = current_dt + timedelta(days=1)
            
            logger.info(f"날짜 범위: {start_date} ~ {end_date} (평일 {len(date_list)}일, 주말 제외)")
        except ValueError as e:
            logger.error(f"날짜 형식 오류: {e} (YYYYMMDD 형식 필요)")
            return
    elif target_date:
        # 단일 날짜 처리
        date_list = [target_date]
        logger.info(f"단일 날짜: {target_date}")
    else:
        # config.json의 target_date 사용
        target_date = ebest_config.get('target_date', datetime.now().strftime('%Y%m%d'))
        date_list = [target_date]
        logger.info(f"config.json target_date 사용: {target_date}")
    
    kp200_upcode = ebest_config.get('kp200_upcode', 'A0166000')  # fallback
    kospi_upcode = '001'  # KOSPI 지수 코드는 항상 '001'
    
    # ebest API 클라이언트 연결
    try:
        import ebest  # type: ignore
        api_client = ebest.OpenApi()
        
        # 자격증명 로드 (config.secrets.json에서 우선)
        appkey = ebest_secrets.get('appkey', ebest_config.get('appkey', ''))
        appsecretkey = ebest_secrets.get('appsecretkey', ebest_config.get('appsecretkey', ''))
        
        if not appkey or not appsecretkey:
            logger.error("ebest 자격증명이 설정되지 않았습니다. config.secrets.json에 appkey와 appsecretkey를 설정하세요.")
            return
        
        # 로그인
        from ebestapi.api import _ebest_login
        login_success = await _ebest_login(api_client, appkey=appkey, appsecretkey=appsecretkey)
        
        if not login_success:
            logger.error("ebest API 로그인 실패")
            return
        
        logger.info("ebest API 로그인 성공")
        
        # KP200 선물 코드 자동 조회
        logger.info("KP200 선물 코드 조회 중...")
        fetched_symbol = await _ebest_fetch_kp200_symbol(api_client)
        if fetched_symbol:
            kp200_upcode = fetched_symbol
            logger.info(f"KP200 선물 코드 자동 조회 완료: {kp200_upcode}")
        else:
            logger.warning(f"KP200 선물 코드 조회 실패, config.json 값 사용: {kp200_upcode}")
    except ImportError as e:
        logger.error(f"ebest 모듈 import 실패: {e}")
        logger.error("ebest 래퍼(예: ebest.OpenApi)가 설치되어 있어야 합니다.")
        return
    except Exception as e:
        logger.error(f"ebest API 연결 실패: {e}")
        return
    
    try:
        # 날짜 범위인 경우 연속 조회 사용
        if start_date and end_date:
            logger.info(f"연속 조회 모드: {start_date} ~ {end_date}")
            
            # KP200 연속 조회
            logger.info("KOSPI200 선물 연속 조회 시작...")
            kp200_rows = await fetch_kp200_continuous(
                api_client,
                symbol=kp200_upcode,
                start_date=start_date,
                end_date=end_date,
                ncnt=1
            )
            
            if kp200_rows:
                kp200_df = pd.DataFrame(kp200_rows)
                # datetime 컬럼 생성
                kp200_df['Datetime'] = pd.to_datetime(
                    kp200_df['date'] + kp200_df['time'],
                    format='%Y%m%d%H%M%S'
                )
                
                # 불필요한 컬럼 제거
                kp200_df = kp200_df.drop(columns=['date', 'time'], errors='ignore')
                
                # 컬럼명 첫 글자 대문자로 변경
                kp200_df.columns = [col.capitalize() for col in kp200_df.columns]
                kp200_df = kp200_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # 정렬 및 중복 제거
                kp200_df = kp200_df.sort_values('Datetime')
                kp200_df = kp200_df.drop_duplicates(subset=['Datetime'], keep='first')
                
                # CSV 저장
                output_dir = Path('data/daily_bars')
                output_dir.mkdir(parents=True, exist_ok=True)
                kp200_file = output_dir / f'minute_bars_kp200_{start_date}_{end_date}.csv'
                kp200_df.to_csv(kp200_file, index=False)
                logger.info(f"KOSPI200 선물 데이터 저장 완료: {kp200_file} ({len(kp200_df)} rows)")
            else:
                logger.warning("KOSPI200 선물 데이터 수집 실패")
            
            # KOSPI 연속 조회
            logger.info("KOSPI 지수 연속 조회 시작...")
            kospi_rows = await fetch_kospi_continuous(
                api_client,
                symbol=kospi_upcode,
                start_date=start_date,
                end_date=end_date,
                ncnt=1
            )
            
            if kospi_rows:
                kospi_df = pd.DataFrame(kospi_rows)
                # datetime 컬럼 생성
                kospi_df['Datetime'] = pd.to_datetime(
                    kospi_df['date'] + kospi_df['time'],
                    format='%Y%m%d%H%M%S'
                )
                
                # 불필요한 컬럼 제거
                kospi_df = kospi_df.drop(columns=['date', 'time'], errors='ignore')
                
                # 컬럼명 첫 글자 대문자로 변경
                kospi_df.columns = [col.capitalize() for col in kospi_df.columns]
                kospi_df = kospi_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # 정렬 및 중복 제거
                kospi_df = kospi_df.sort_values('Datetime')
                kospi_df = kospi_df.drop_duplicates(subset=['Datetime'], keep='first')
                
                # CSV 저장
                kospi_file = output_dir / f'minute_bars_kospi_{start_date}_{end_date}.csv'
                kospi_df.to_csv(kospi_file, index=False)
                logger.info(f"KOSPI 지수 데이터 저장 완료: {kospi_file} ({len(kospi_df)} rows)")
            else:
                logger.warning("KOSPI 지수 데이터 수집 실패")
            
            logger.info("="*80)
            logger.info(f"연속 조회 완료: {start_date} ~ {end_date}")
            logger.info("="*80)
        else:
            # 단일 날짜인 경우 기존 방식 사용
            logger.info("단일 날짜 모드")
            success_count = 0
            fail_count = 0
            
            for idx, current_date in enumerate(date_list, 1):
                logger.info(f"날짜 {idx}/{len(date_list)}: {current_date}")
                
                # t8415: KOSPI200 선물 분봉 데이터 수집 (1분봉)
                logger.info(f"KOSPI200 선물 코드: {kp200_upcode}")
                logger.info(f"t8415 요청: {kp200_upcode} ({current_date})")
                kp200_bars = await _ebest_fetch_kp200_ohlcv_t8415(
                    api_client,
                    symbol=kp200_upcode,
                    yyyymmdd=current_date,
                    ncnt=1  # 1분봉
                )
                
                if kp200_bars:
                    kp200_df = pd.DataFrame(kp200_bars)
                    # current_date와 time 결합하여 KST 기준 datetime 생성
                    kp200_df['Datetime'] = pd.to_datetime(current_date + kp200_df['time'], format='%Y%m%d%H%M%S')
                    
                    # 불필요한 컬럼 제거 (date, time)
                    kp200_df = kp200_df.drop(columns=['date', 'time'], errors='ignore')
                    
                    # 컬럼명 첫 글자 대문자로 변경
                    kp200_df.columns = [col.capitalize() for col in kp200_df.columns]
                    kp200_df = kp200_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    # CSV 저장
                    output_dir = Path('data/daily_bars')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    kp200_file = output_dir / f'minute_bars_kp200_{current_date}.csv'
                    kp200_df.to_csv(kp200_file, index=False)
                    logger.info(f"KOSPI200 선물 데이터 저장 완료: {kp200_file} ({len(kp200_df)} rows)")
                else:
                    logger.warning(f"KOSPI200 선물 데이터 수집 실패: {current_date}")
                    fail_count += 1
                    continue
                
                # t8418: KOSPI 지수 분봉 데이터 수집 (1분봉)
                logger.info(f"t8418 요청: {kospi_upcode} ({current_date})")
                kospi_bars = await _ebest_fetch_kospi_ohlcv_t8418(
                    api_client,
                    symbol=kospi_upcode,
                    yyyymmdd=current_date,
                    ncnt=1  # 1분봉
                )
                
                if kospi_bars:
                    kospi_df = pd.DataFrame(kospi_bars)
                    # current_date와 time 결합하여 KST 기준 datetime 생성
                    kospi_df['Datetime'] = pd.to_datetime(current_date + kospi_df['time'], format='%Y%m%d%H%M%S')
                    
                    # 불필요한 컬럼 제거 (date, time)
                    kospi_df = kospi_df.drop(columns=['date', 'time'], errors='ignore')
                    
                    # 컬럼명 첫 글자 대문자로 변경
                    kospi_df.columns = [col.capitalize() for col in kospi_df.columns]
                    kospi_df = kospi_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    # CSV 저장
                    kospi_file = output_dir / f'minute_bars_kospi_{current_date}.csv'
                    kospi_df.to_csv(kospi_file, index=False)
                    logger.info(f"KOSPI 지수 데이터 저장 완료: {kospi_file} ({len(kospi_df)} rows)")
                    success_count += 1
                else:
                    logger.warning(f"KOSPI 지수 데이터 수집 실패: {current_date}")
                    fail_count += 1
            
            logger.info("="*80)
            logger.info(f"데이터 수집 완료: 성공 {success_count}일, 실패 {fail_count}일 (총 {len(date_list)}일)")
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}", exc_info=True)
    
    finally:
        # API 클라이언트 정리
        if 'api_client' in locals():
            try:
                # ebest API 클라이언트 종료
                if hasattr(api_client, 'disconnect'):
                    api_client.disconnect()
                logger.info("ebest API 클라이언트 종료 완료")
            except Exception as e:
                logger.warning(f"API 클라이언트 종료 중 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='장마감 이후 t8415/t8418 데이터 수집')
    parser.add_argument('--target-date', type=str, help='목표 날짜 (YYYYMMDD 형식)')
    parser.add_argument('--start-date', type=str, help='시작 날짜 (YYYYMMDD 형식)')
    parser.add_argument('--end-date', type=str, help='종료 날짜 (YYYYMMDD 형식)')
    parser.add_argument('--force', action='store_true', help='장마감 전에도 강제 실행')
    
    args = parser.parse_args()
    
    # 인자 유효성 검증
    if args.target_date and (args.start_date or args.end_date):
        logger.error("--target-date와 --start-date/--end-date는 동시에 사용할 수 없습니다.")
        return
    
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        logger.error("--start-date와 --end-date는 함께 사용해야 합니다.")
        return
    
    # 날짜 형식 검증
    for date_arg in ['target_date', 'start_date', 'end_date']:
        date_value = getattr(args, date_arg)
        if date_value:
            try:
                datetime.strptime(date_value, '%Y%m%d')
            except ValueError:
                logger.error(f"{date_arg.replace('_', '-')} 형식 오류: {date_value} (YYYYMMDD 형식 필요)")
                return
    
    logger.info("="*80)
    logger.info("장마감 이후 t8415/t8418 데이터 수집 시작")
    logger.info("="*80)
    
    # 장마감 확인 (15:30 이후)
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now < market_close and not args.force:
        logger.warning(f"현재 시간: {now.strftime('%H:%M:%S')}")
        logger.warning("장마감 시간: 15:30")
        logger.warning("장마감 이후에 실행해야 합니다.")
        logger.warning("강제 실행하려면 --force 옵션을 사용하세요.")
        return
    
    # 비동기 실행 (인자로 날짜 전달)
    asyncio.run(fetch_and_save_daily_data(
        target_date=args.target_date,
        start_date=args.start_date,
        end_date=args.end_date
    ))


if __name__ == '__main__':
    main()
