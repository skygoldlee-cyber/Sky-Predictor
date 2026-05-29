"""
장마감 이후 t8415/t8418 데이터 수집 스크립트

config.json의 target_date를 사용하여 해당 날짜의 분봉 데이터를 수집하고 CSV로 저장합니다.
- t8415: KOSPI200 선물 분봉 데이터
- t8418: KOSPI 지수 분봉 데이터

사용법:
    python scripts/fetch_daily_data.py              # 장마감 이후에만 실행
    python scripts/fetch_daily_data.py --force      # 장마감 전에도 강제 실행

요구사항:
    - config.secrets.json에 ebest 자격증명 (appkey, appsecretkey) 설정 필요
    - config.json에 target_date 설정 필요 (kp200_upcode는 API에서 자동 조회, kospi_upcode는 '001' 고정)
    - ebest 모듈 (ebest.OpenApi) 설치 필요
    - ebest API 연결 가능한 환경 필요
    - data/daily_bars/ 디렉토리에 CSV 저장

출력 파일:
    - data/daily_bars/minute_bars_kp200_{target_date}.csv
    - data/daily_bars/minute_bars_kospi_{target_date}.csv

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
import json
import sys
import warnings
from datetime import datetime
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


async def fetch_and_save_daily_data():
    """config.json의 target_date로 t8415/t8418 데이터 수집 및 저장"""
    
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
    
    target_date = ebest_config.get('target_date', datetime.now().strftime('%Y%m%d'))
    kp200_upcode = ebest_config.get('kp200_upcode', 'A0166000')  # fallback
    kospi_upcode = '001'  # KOSPI 지수 코드는 항상 '001'
    
    logger.info(f"목표 날짜: {target_date}")
    
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
        # t8415: KOSPI200 선물 분봉 데이터 수집 (1분봉)
        logger.info(f"KOSPI200 선물 코드: {kp200_upcode}")
        logger.info(f"t8415 요청: {kp200_upcode} ({target_date})")
        kp200_bars = await _ebest_fetch_kp200_ohlcv_t8415(
            api_client,
            symbol=kp200_upcode,
            yyyymmdd=target_date,
            ncnt=1  # 1분봉
        )
        
        if kp200_bars:
            kp200_df = pd.DataFrame(kp200_bars)
            # target_date와 time 결합하여 KST 기준 datetime 생성
            kp200_df['Datetime'] = pd.to_datetime(target_date + kp200_df['time'], format='%Y%m%d%H%M%S')
            
            # 불필요한 컬럼 제거 (date, time)
            kp200_df = kp200_df.drop(columns=['date', 'time'], errors='ignore')
            
            # 컬럼명 첫 글자 대문자로 변경
            kp200_df.columns = [col.capitalize() for col in kp200_df.columns]
            kp200_df = kp200_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # CSV 저장
            output_dir = Path('data/daily_bars')
            output_dir.mkdir(parents=True, exist_ok=True)
            kp200_file = output_dir / f'minute_bars_kp200_{target_date}.csv'
            kp200_df.to_csv(kp200_file, index=False)
            logger.info(f"KOSPI200 선물 데이터 저장 완료: {kp200_file} ({len(kp200_df)} rows)")
        else:
            logger.warning("KOSPI200 선물 데이터 수집 실패")
        
        # t8418: KOSPI 지수 분봉 데이터 수집 (1분봉)
        logger.info(f"t8418 요청: {kospi_upcode} ({target_date})")
        kospi_bars = await _ebest_fetch_kospi_ohlcv_t8418(
            api_client,
            symbol=kospi_upcode,
            yyyymmdd=target_date,
            ncnt=1  # 1분봉
        )
        
        if kospi_bars:
            kospi_df = pd.DataFrame(kospi_bars)
            # target_date와 time 결합하여 KST 기준 datetime 생성
            kospi_df['Datetime'] = pd.to_datetime(target_date + kospi_df['time'], format='%Y%m%d%H%M%S')
            
            # 불필요한 컬럼 제거 (date, time)
            kospi_df = kospi_df.drop(columns=['date', 'time'], errors='ignore')
            
            # 컬럼명 첫 글자 대문자로 변경
            kospi_df.columns = [col.capitalize() for col in kospi_df.columns]
            kospi_df = kospi_df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # CSV 저장
            kospi_file = output_dir / f'minute_bars_kospi_{target_date}.csv'
            kospi_df.to_csv(kospi_file, index=False)
            logger.info(f"KOSPI 지수 데이터 저장 완료: {kospi_file} ({len(kospi_df)} rows)")
        else:
            logger.warning("KOSPI 지수 데이터 수집 실패")
        
        logger.info("데이터 수집 완료")
        
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
    logger.info("="*80)
    logger.info("장마감 이후 t8415/t8418 데이터 수집 시작")
    logger.info("="*80)
    
    # 장마감 확인 (15:30 이후)
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now < market_close:
        logger.warning(f"현재 시간: {now.strftime('%H:%M:%S')}")
        logger.warning("장마감 시간: 15:30")
        logger.warning("장마감 이후에 실행해야 합니다.")
        logger.warning("강제 실행하려면 --force 옵션을 사용하세요.")
        
        if '--force' not in sys.argv:
            return
    
    # 비동기 실행
    asyncio.run(fetch_and_save_daily_data())


if __name__ == '__main__':
    main()
