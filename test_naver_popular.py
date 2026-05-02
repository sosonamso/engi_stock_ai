"""
네이버 인기종목 크롤링 테스트
"""
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

print("=== 네이버 인기검색 종목 ===")
try:
    url  = "https://finance.naver.com/sise/lastsearch2.naver"
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"상태코드: {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")

    # 테이블 파싱
    tables = pd.read_html(StringIO(resp.text))
    print(f"테이블 수: {len(tables)}")
    for i, t in enumerate(tables[:3]):
        print(f"\n[테이블 {i}] shape={t.shape}")
        print(t.head(10))

    # 링크에서 티커 추출
    print("\n=== 링크에서 티커 추출 ===")
    tickers = []
    for a in soup.select("a[href*='code=']"):
        href = a["href"]
        if "code=" in href:
            code = href.split("code=")[-1].split("&")[0].strip()
            name = a.text.strip()
            if len(code) == 6 and code.isdigit() and name:
                tickers.append({"ticker": code, "name": name})

    df = pd.DataFrame(tickers).drop_duplicates()
    print(f"종목 수: {len(df)}개")
    print(df.head(20))

except Exception as e:
    print(f"에러: {e}")
