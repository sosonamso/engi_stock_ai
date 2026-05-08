"""
KRX API 테스트
- 투자자별 매매동향 (기관/외국인)
- 종목별 시세 데이터
"""
import os, requests
import pandas as pd

token = os.environ.get("KRX_TOKEN", "")
print(f"토큰 앞 10자리: {token[:10]}...")

headers = {
    "User-Agent": "Mozilla/5.0",
    "Authorization": f"Bearer {token}",
}

base = "https://data.krx.co.kr"

# 1. KRX 정보데이터시스템 API 테스트
print("\n" + "=" * 50)
print("1. KRX 투자자별 매매동향 (삼성전자 005930)")
try:
    url = f"{base}/comm/bldAttendant/getJsonData.cmd"
    data = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT02303",
        "isuCd": "KR7005930003",
        "strtDd": "20260401",
        "endDd": "20260507",
        "money": "1",
        "csvxls_isNo": "false",
    }
    resp = requests.post(url, data=data, headers=headers, timeout=10)
    print(f"상태코드: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(f"에러: {e}")

# 2. 토큰 없이 테스트
print("\n" + "=" * 50)
print("2. 토큰 없이 동일 요청")
try:
    resp2 = requests.post(url, data=data, timeout=10)
    print(f"상태코드: {resp2.status_code}")
    j = resp2.json()
    print(f"키: {list(j.keys())[:5]}")
    if "output" in j and j["output"]:
        df = pd.DataFrame(j["output"])
        print(f"컬럼: {df.columns.tolist()}")
        print(df.head(3).to_string())
except Exception as e:
    print(f"에러: {e}")

# 3. 다른 엔드포인트 시도
print("\n" + "=" * 50)
print("3. KRX 일별 투자자 매매 API")
try:
    url3 = f"{base}/comm/bldAttendant/getJsonData.cmd"
    data3 = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT02204",
        "strtDd": "20260401",
        "endDd": "20260507",
        "isuCd": "KR7005930003",
        "csvxls_isNo": "false",
    }
    resp3 = requests.post(url3, data=data3, timeout=10)
    print(f"상태코드: {resp3.status_code}")
    j3 = resp3.json()
    if "output" in j3 and j3["output"]:
        df3 = pd.DataFrame(j3["output"])
        print(f"컬럼: {df3.columns.tolist()}")
        print(df3.head(3).to_string())
    else:
        print(resp3.text[:300])
except Exception as e:
    print(f"에러: {e}")
