import asyncio
import ebest
import json
from pathlib import Path

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
