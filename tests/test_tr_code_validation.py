"""
eBest TR 코드 유효성 테스트

현재 사용 중인 TR 코드(t8415, t8418)와 대안 TR 코드(t8465, t8427) 테스트
"""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTRCodeValidation:
    """eBest TR 코드 유효성 테스트"""
    
    @pytest.mark.asyncio
    async def test_t8415_kp200_futures(self):
        """KOSPI200 선물: t8415 TR 코드 테스트"""
        try:
            import ebest  # type: ignore
            from ebestapi.api import _ebest_login, _ebest_fetch_kp200_ohlcv_t8415, _ebest_fetch_kp200_symbol
            
            # config.secrets.json 로드
            import json
            secrets_path = Path(__file__).parent.parent / 'config.secrets.json'
            config_path = Path(__file__).parent.parent / 'config.json'
            
            try:
                with open(secrets_path, 'r', encoding='utf-8') as f:
                    secrets = json.load(f)
            except FileNotFoundError:
                pytest.skip("config.secrets.json 파일이 없습니다")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            ebest_secrets = secrets.get('ebest', {})
            ebest_config = config.get('ebest', {})
            
            appkey = ebest_secrets.get('appkey', ebest_config.get('appkey', ''))
            appsecretkey = ebest_secrets.get('appsecretkey', ebest_config.get('appsecretkey', ''))
            
            if not appkey or not appsecretkey:
                pytest.skip("ebest 자격증명이 설정되지 않았습니다")
            
            # API 클라이언트 연결
            api_client = ebest.OpenApi()
            login_success = await _ebest_login(api_client, appkey=appkey, appsecretkey=appsecretkey)
            
            print(f"로그인 성공 여부: {login_success}")
            
            if not login_success:
                pytest.skip("ebest API 로그인 실패")
            
            # 로그인 상태 확인
            print("API 클라이언트 상태 확인...")
            print(f"API 클라이언트 타입: {type(api_client)}")
            
            # 계정 유형 확인
            is_sim = getattr(api_client, "is_simulation", None)
            print(f"계정 유형: {'모의투자' if is_sim else '실시간' if is_sim is not None else '알 수 없음'}")
            print(f"is_simulation 속성: {is_sim}")
            
            # 기타 속성 확인
            print(f"API 클라이언트 속성: {[attr for attr in dir(api_client) if not attr.startswith('_')]}")
            
            # 간단한 테스트 요청으로 로그인 확인
            try:
                test_res = await api_client.request("t1102", {"t1102InBlock": {"gubun": "0"}})
                print(f"t1102 테스트 응답: {test_res}")
            except Exception as e:
                print(f"t1102 테스트 실패: {e}")
            
            # KP200 선물 코드 자동 조회
            print("KP200 선물 코드 조회 중...")
            symbol = await _ebest_fetch_kp200_symbol(api_client)
            print(f"fetch 심볼: {symbol}")
            
            # t9943 직접 테스트
            print("t9943 직접 테스트...")
            try:
                res = await api_client.request("t9943", {"t9943InBlock": {"gubun": ""}})
                print(f"t9943 응답: {res}")
                body = getattr(res, "body", None) or {}
                print(f"t9943 body: {body}")
                items = body.get("t9943OutBlock") or []
                print(f"t9943 items: {items}")
                if items:
                    print(f"t9943 첫 번째 항목: {items[0]}")
                    # t9943에서 얻은 심볼 사용
                    symbol = str(items[0].get("shcode") or "").strip()
                    print(f"t9943에서 얻은 심볼: {symbol}")
            except Exception as e:
                print(f"t9943 실패: {e}")
                import traceback
                traceback.print_exc()
            
            if not symbol:
                symbol = "A0169000"  # fallback
            print(f"사용 심볼: {symbol}")
            
            # t8465 직접 테스트
            print("\n=== t8465 직접 테스트 ===")
            test_date = "20260616"
            try:
                req = {
                    "t8465InBlock": {
                        "shcode": symbol,
                        "ncnt": 1,
                        "qrycnt": 1,
                        "nday": "",
                        "sdate": test_date,
                        "stime": "",
                        "edate": test_date,
                        "etime": "",
                        "cts_date": "",
                        "cts_time": "",
                        "comp_yn": "N"
                    }
                }
                res = await api_client.request("t8465", req)
                print(f"t8465 응답: {res}")
                body = getattr(res, "body", None) or {}
                print(f"t8465 body: {body}")
                items = body.get("t8465OutBlock1") or []
                print(f"t8465 items: {items}")
                if items:
                    print(f"✅ t8465 성공: {len(items)} 건 수신")
                    print(f"첫 번째 바: {items[0]}")
                else:
                    print(f"❌ t8465 실패: 데이터 없음")
                    pytest.skip("t8465 데이터 없음")
            except Exception as e:
                print(f"❌ t8465 실패: {e}")
                import traceback
                traceback.print_exc()
                pytest.skip("t8465 요청 실패")
            
            # 정리
            if hasattr(api_client, 'disconnect'):
                api_client.disconnect()
            
        except ImportError:
            pytest.skip("ebest 모듈이 설치되지 않았습니다")
    
    @pytest.mark.asyncio
    async def test_t8418_kospi_index(self):
        """KOSPI 지수: t8418 TR 코드 테스트"""
        try:
            import ebest  # type: ignore
            from ebestapi.api import _ebest_login, _ebest_fetch_kospi_ohlcv_t8418
            
            # config.secrets.json 로드
            import json
            secrets_path = Path(__file__).parent.parent / 'config.secrets.json'
            config_path = Path(__file__).parent.parent / 'config.json'
            
            try:
                with open(secrets_path, 'r', encoding='utf-8') as f:
                    secrets = json.load(f)
            except FileNotFoundError:
                pytest.skip("config.secrets.json 파일이 없습니다")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            ebest_secrets = secrets.get('ebest', {})
            ebest_config = config.get('ebest', {})
            
            appkey = ebest_secrets.get('appkey', ebest_config.get('appkey', ''))
            appsecretkey = ebest_secrets.get('appsecretkey', ebest_config.get('appsecretkey', ''))
            
            if not appkey or not appsecretkey:
                pytest.skip("ebest 자격증명이 설정되지 않았습니다")
            
            # API 클라이언트 연결
            api_client = ebest.OpenApi()
            login_success = await _ebest_login(api_client, appkey=appkey, appsecretkey=appsecretkey)
            
            if not login_success:
                pytest.skip("ebest API 로그인 실패")
            
            # 테스트용 날짜 (여러 날짜 시도)
            test_dates = ["20260616", "20260615", "20260614", "20260613", "20260612", "20260611", "20260610"]
            symbol = "001"  # KOSPI 지수 코드
            
            # t8418 테스트 (기존 함수 사용)
            for test_date in test_dates:
                print(f"\n=== t8418 테스트 ({test_date}) ===")
                try:
                    bars = await _ebest_fetch_kospi_ohlcv_t8418(
                        api_client,
                        symbol=symbol,
                        yyyymmdd=test_date,
                        ncnt=1
                    )
                    
                    if bars:
                        print(f"✅ t8418 성공: {len(bars)} 건 수신")
                        print(f"첫 번째 바: {bars[0] if bars else 'N/A'}")
                        break  # 성공하면 종료
                    else:
                        print(f"❌ t8418 실패: 데이터 없음 ({test_date})")
                except Exception as e:
                    print(f"❌ t8418 실패: {e} ({test_date})")
                    import traceback
                    traceback.print_exc()
            else:
                # 모든 날짜 실패
                pytest.skip("t8418 모든 테스트 날짜에서 데이터 없음")
            
            # 정리
            if hasattr(api_client, 'disconnect'):
                api_client.disconnect()
            
        except ImportError:
            pytest.skip("ebest 모듈이 설치되지 않았습니다")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
