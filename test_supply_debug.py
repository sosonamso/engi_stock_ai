"""
수급 파싱 디버그
"""
import requests, pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
ticker = "005930"

# 1. 시총 파싱 디버그
print("=" * 40)
print("시총 파싱 디버그")
url = f"https://finance.naver.com/item/main.naver?code={ticker}"
resp = requests.get(url, headers=headers, timeout=10)
tables = pd.read_html(StringIO(resp.text))
# 테이블 6이 시총
print(f"테이블 6:\n{tables[6]}")

# 2. frgn 수급 파싱 디버그
print("\n" + "=" * 40)
print("수급 파싱 디버그")
url2 = f"https://finance.naver.com/item/frgn.naver?code={ticker}"
resp2 = requests.get(url2, headers=headers, timeout=10)
tables2 = pd.read_html(StringIO(resp2.text))
t3 = tables2[3]
print(f"컬럼: {t3.columns.tolist()}")
print(t3.head(5).to_string())
