"""
네이버 테마별 종목 목록 테스트
"""
import requests
import pandas as pd
from io import StringIO

headers = {"User-Agent": "Mozilla/5.0"}

# 1. 테마 목록 + 코드 가져오기
print("테마 목록 수집...")
url = "https://finance.naver.com/sise/sise_group.naver?type=theme"
resp = requests.get(url, headers=headers, timeout=10)
from bs4 import BeautifulSoup
soup = BeautifulSoup(resp.text, "html.parser")

# 테마 링크 추출
themes = []
for a in soup.select("a[href*='sise_group_detail']"):
    href = a["href"]
    name = a.text.strip()
    if "no=" in href:
        no = href.split("no=")[-1].split("&")[0]
        themes.append({"name": name, "no": no})

print(f"테마 수: {len(themes)}개")
print("샘플:", themes[:5])

# 2. 특정 테마 종목 목록 (반도체 장비 찾기)
semi_theme = next((t for t in themes if "반도체" in t["name"] and "장비" in t["name"]), None)
if semi_theme:
    print(f"\n테마: {semi_theme['name']} (no={semi_theme['no']})")
    detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={semi_theme['no']}"
    resp2 = requests.get(detail_url, headers=headers, timeout=10)
    tables = pd.read_html(StringIO(resp2.text))
    print(f"테이블 수: {len(tables)}")
    for i, t in enumerate(tables):
        print(f"\n[테이블 {i}] shape={t.shape} 컬럼:{t.columns.tolist()}")
        print(t.head(10))
