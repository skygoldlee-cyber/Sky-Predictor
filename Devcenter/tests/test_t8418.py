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


async def sample(api):
    inputs = {
        't8418InBlock': {
            'shcode': '001',  # KOSPI
            'ncnt': 1,        # 1분봉
            'qrycnt': 10,     # 10건만 테스트
            'nday': '',
            'sdate': '',
            'stime': '',
            'edate': '99999999',
            'etime': '',
            'cts_date': '',
            'cts_time': '',
            'comp_yn': 'N',   # 비압축으로 테스트
        },
    }
    response = await api.request('t8418', inputs)
    if not response:
        print(f'요청 실패: {api.last_message}')
        return

    print('=== 전체 응답 ===')
    print(response)
    print('\n=== t8418OutBlock ===')
    print(response.body.get('t8418OutBlock', {}))
    print('\n=== t8418OutBlock1 ===')
    outblock1 = response.body.get('t8418OutBlock1', [])
    print(f'레코드 수: {len(outblock1)}')
    if outblock1:
        print('첫 번째 레코드:')
        print(outblock1[0])
        if len(outblock1) > 1:
            print('두 번째 레코드:')
            print(outblock1[1])


async def main():
    api = ebest.OpenApi()
    if not await api.login(appkey, appsecretkey):
        return print(f'연결실패: {api.last_message}')
    await sample(api)
    await api.close()


if __name__ == '__main__':
    asyncio.run(main())
