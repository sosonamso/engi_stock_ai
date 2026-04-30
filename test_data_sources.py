"""
데이터 소스 테스트
1. 네이버 증권 테마
2. KRX 업종 분류
"""
import requests
import pandas as pd

headers = {"User-Agent": "Mozilla/5.0"}

# ── 1. 네이버 테마 ────────────────────────────────
print("=" * 50)
print("1. 네이버 증권 테마")
print("=" * 50)
try:
    url = "https://finance.naver.com/sise/sise_group.naver?type=theme"
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"상태코드: {resp.status_code}")
    if resp.status_code == 200:
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))
        print(f"테이블 수: {len(tables)}")
        for i, t in enumerate(tables[:3]):
            print(f"\n[테이블 {i}] shape={t.shape}")
            print(t.head(5))
except Exception as e:
    print(f"에러: {e}")

print()

# ── 2. KRX 업종 분류 ──────────────────────────────
print("=" * 50)
print("2. KRX 업종 분류 API")
print("=" * 50)
try:
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    data = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT03901",
        "mktId": "ALL",
        "trdDd": pd.Timestamp.today().strftime("%Y%m%d"),
        "money": "1",
        "csvxls_isNo": "false",
    }
    resp = requests.post(url, data=data, headers=headers, timeout=10)
    print(f"상태코드: {resp.status_code}")
    if resp.status_code == 200:
        j = resp.json()
        print(f"키: {list(j.keys())[:5]}")
        if "output" in j:
            df = pd.DataFrame(j["output"])
            print(f"컬럼: {df.columns.tolist()}")
            print(df.head(5))
except Exception as e:
    print(f"에러: {e}")

print()

# ── 3. KRX 테마 ───────────────────────────────────
print("=" * 50)
print("3. KRX 테마지수")
print("=" * 50)
try:
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    data = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
        "idxIndMidclssCd": "02",
        "trdDd": pd.Timestamp.today().strftime("%Y%m%d"),
        "csvxls_isNo": "false",
    }
    resp = requests.post(url, data=data, headers=headers, timeout=10)
    print(f"상태코드: {resp.status_code}")
    if resp.status_code == 200:
        j = resp.json()
        print(f"키: {list(j.keys())[:5]}")
        if "output" in j:
            df = pd.DataFrame(j["output"])
            print(f"컬럼: {df.columns.tolist()}")
            print(df.head(10))
except Exception as e:
    print(f"에러: {e}")
