"""
네이버 frgn 페이지 전체 컬럼 확인 + main 시총/주식수
"""
import requests, pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0"}
ticker = "005930"

# 1. frgn 페이지 테이블 3 전체 컬럼 확인
print("=" * 50)
print("1. frgn 페이지 테이블 3 전체")
url = f"https://finance.naver.com/item/frgn.naver?code={ticker}"
resp = requests.get(url, headers=headers, timeout=10)
tables = pd.read_html(StringIO(resp.text))
t3 = tables[3]
print(f"컬럼: {t3.columns.tolist()}")
print(t3.head(5).to_string())

# 2. main 페이지 모든 테이블 컬럼 확인
print("\n" + "=" * 50)
print("2. main 페이지 모든 테이블")
url2 = f"https://finance.naver.com/item/main.naver?code={ticker}"
resp2 = requests.get(url2, headers=headers, timeout=10)
tables2 = pd.read_html(StringIO(resp2.text))
for i, t in enumerate(tables2):
    print(f"\n[테이블 {i}] shape={t.shape}")
    print(t.head(3).to_string())

# 3. sise_investor 페이지 (투자자별)
print("\n" + "=" * 50)
print("3. 투자자별 매매동향")
url3 = f"https://finance.naver.com/item/sise_investor.naver?code={ticker}"
resp3 = requests.get(url3, headers=headers, timeout=10)
print(f"상태: {resp3.status_code}")
if resp3.status_code == 200:
    tables3 = pd.read_html(StringIO(resp3.text))
    print(f"테이블 수: {len(tables3)}")
    for i, t in enumerate(tables3[:3]):
        print(f"\n[테이블 {i}] shape={t.shape}")
        print(t.head(5).to_string())
