"""
네이버 증권 외국인/기관 순매수 크롤링 테스트
삼성전자(005930), SK하이닉스(000660) 테스트
"""
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_investor_data(ticker):
    """외국인/기관 순매수 크롤링"""
    # 당일 투자자별 매매 동향
    url = f"https://finance.naver.com/item/frgn.naver?code={ticker}"
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"\n[{ticker}] 상태코드: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = pd.read_html(StringIO(resp.text))
    print(f"테이블 수: {len(tables)}")
    for i, t in enumerate(tables[:4]):
        print(f"\n[테이블 {i}] shape={t.shape}")
        print(t.head(5))

    # 시가총액 / 발행주식수도 확인
    url2 = f"https://finance.naver.com/item/main.naver?code={ticker}"
    resp2 = requests.get(url2, headers=headers, timeout=10)
    soup2 = BeautifulSoup(resp2.text, "html.parser")
    tables2 = pd.read_html(StringIO(resp2.text))
    print(f"\n[main 페이지] 테이블 수: {len(tables2)}")
    for i, t in enumerate(tables2[:3]):
        print(f"\n[테이블 {i}] shape={t.shape}")
        print(t.head(5))

# 삼성전자 테스트
get_investor_data("005930")
