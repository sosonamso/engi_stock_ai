"""
네이버 기관 순매수 + 발행주식수 추가 테스트
"""
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
ticker = "005930"

# 1. 투자자별 매매동향 페이지
print("=" * 50)
print("1. 투자자별 매매동향")
url = f"https://finance.naver.com/item/투자자.naver?code={ticker}"
url = f"https://finance.naver.com/item/investor.naver?code={ticker}"
resp = requests.get(url, headers=headers, timeout=10)
print(f"상태코드: {resp.status_code}")
tables = pd.read_html(StringIO(resp.text))
print(f"테이블 수: {len(tables)}")
for i, t in enumerate(tables[:5]):
    print(f"\n[테이블 {i}] shape={t.shape} 컬럼:{t.columns.tolist()[:6]}")
    print(t.head(5).to_string())

# 2. main 페이지 나머지 테이블 (시총/발행주식수)
print("\n" + "=" * 50)
print("2. main 페이지 전체 테이블")
url2 = f"https://finance.naver.com/item/main.naver?code={ticker}"
resp2 = requests.get(url2, headers=headers, timeout=10)
tables2 = pd.read_html(StringIO(resp2.text))
for i, t in enumerate(tables2):
    cols = str(t.columns.tolist())
    vals = str(t.values.tolist()[:2])
    if any(k in cols+vals for k in ['시가총액','발행','상장','주식수']):
        print(f"\n[테이블 {i}] ★ 관련 데이터")
        print(t.head(5).to_string())

# 3. 종목분석 페이지
print("\n" + "=" * 50)
print("3. 기관 포함 수급 API 탐색")
# 네이버 금융 API
url3 = f"https://api.finance.naver.com/service/itemSummary.nhn?itemcode={ticker}"
resp3 = requests.get(url3, headers=headers, timeout=10)
print(f"API 상태: {resp3.status_code}")
print(resp3.text[:500])
