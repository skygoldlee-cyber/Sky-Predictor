import asyncio
import ebest
import json
from pathlib import Path

# ■  변경대상 선물옵션 TR 리스트
#
# 기존TR	신규TR			TR명
#
# t2101		t2111		선물/옵션 현재가(시세) 조회
#
# t2105		t2112		선물/옵션 현재가 호가조회
#
# t2201		t2212		선물/옵션 시간대별 체결조회
#
# t2203		t2214		선물/옵션 기간별 주가
#
# t2209		t2216		선물/옵션 틱분별 체결조회 차트
#
# t2405		t2407		선물/옵션 호가잔량 비율 차트
#
# t2421		t2424		선물/옵션 미결제약정 추이
#
# t8414		t8464		선물옵션차트(틱/n틱)
#
# t8415		t8465		선물/옵션챠트(N분)
#
# t8416		t8466		선물/옵션챠트(일주월)
#
# t8432		t8467		지수선물마스터조회API용
#
# FC0		FC9		KOSPI200선물체결
#
# FH0		FH9		KOSPI200선물호가
#
# FX0		FX9		KOSPI200선물가격제한폭확대
#
# YFC		YF9		지수선물예상체결

# config.secrets.json에서 읽어오기
secrets_path = Path(__file__).parent.parent / 'config.secrets.json'
with open(secrets_path, 'r', encoding='utf-8') as f:
    secrets = json.load(f)
appkey = secrets.get('ebest', {}).get('appkey', '')
appsecretkey = secrets.get('ebest', {}).get('appsecretkey', '')

async def test_t8465(api):
    """t8465 TR 코드 테스트"""
    # t9943로 선물 심볼 조회
    print("=== t9943 선물 심볼 조회 ===")
    try:
        res = await api.request("t9943", {"t9943InBlock": {"gubun": ""}})
        print(f"t9943 응답: {res}")
        body = getattr(res, "body", None) or {}
        items = body.get("t9943OutBlock") or []
        if items:
            symbol = str(items[0].get("shcode") or "").strip()
            print(f"선물 심볼: {symbol}")
        else:
            print("선물 심볼 조회 실패")
            return
    except Exception as e:
        print(f"t9943 실패: {e}")
        return
    
    # t2111로 현재 선물 시세 조회
    print(f"\n=== t2111 현재 선물 시세 조회 ({symbol}) ===")
    try:
        inputs = {
            't2111InBlock': {
                'focode': symbol,  # 단축코드
            },
        }
        response = await api.request('t2111', inputs)
        if not response:
            print(f'요청 실패: {api.last_message}')
        else:
            print(f"t2111 응답: {response}")
            body = getattr(response, "body", None) or {}
            print(f"t2111 body: {body}")
    except Exception as e:
        print(f"t2111 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # t8465로 선물 분봉 데이터 조회
    print(f"\n=== t8465 선물 분봉 데이터 조회 ({symbol}) ===")
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
        res = await api.request("t8465", req)
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
    except Exception as e:
        print(f"❌ t8465 실패: {e}")
        import traceback
        traceback.print_exc()

async def main():
    api = ebest.OpenApi()
    if not await api.login(appkey, appsecretkey):
        return print(f'연결실패: {api.last_message}')
    
    await test_t8465(api)
    
    await api.close()

if __name__ == '__main__':
    asyncio.run(main())
