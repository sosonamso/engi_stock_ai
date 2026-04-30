"""
네이버 증권 테마별 종목 매핑 수집
- 264개 테마 전체 수집
- 종목명 → 티커 매핑
- theme_ticker_map.csv 저장
"""
import time, warnings
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_theme_list():
    """테마 목록 + no 수집"""
    url  = "https://finance.naver.com/sise/sise_group.naver?type=theme"
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    themes = []
    for a in soup.select("a[href*='sise_group_detail']"):
        href = a["href"]
        name = a.text.strip()
        if "no=" in href and name:
            no = href.split("no=")[-1].split("&")[0]
            themes.append({"name": name, "no": no})
    # 중복 제거
    seen = set()
    unique = []
    for t in themes:
        if t["no"] not in seen:
            seen.add(t["no"])
            unique.append(t)
    return unique


def get_theme_stocks(theme_no):
    """테마별 종목명 목록 수집"""
    url  = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={theme_no}"
    resp = requests.get(url, headers=headers, timeout=10)
    try:
        tables = pd.read_html(StringIO(resp.text))
        # 테이블 2가 종목 목록
        for t in tables:
            if "종목명" in t.columns:
                names = t["종목명"].dropna().tolist()
                # "* " 제거, 공백 제거
                names = [str(n).replace("*", "").strip() for n in names if str(n).strip() and str(n) != "nan"]
                return names
    except: pass
    return []


if __name__ == "__main__":
    # 티커-종목명 매핑 로드
    kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
    kr_df["ticker"] = kr_df["ticker"].astype(str).str.zfill(6)
    name_to_ticker = {}
    for _, r in kr_df.iterrows():
        name = str(r["name"]).strip()
        name_to_ticker[name] = r["ticker"]
    print(f"티커 매핑: {len(name_to_ticker)}개")

    # 테마 목록 수집
    print("테마 목록 수집 중...")
    themes = get_theme_list()
    print(f"테마 수: {len(themes)}개")

    # 테마별 종목 수집
    rows = []
    failed = 0

    for i, theme in enumerate(themes):
        if i % 20 == 0:
            print(f"[{i}/{len(themes)}] {theme['name']} | 매핑:{len(rows)}건")

        stocks = get_theme_stocks(theme["no"])
        matched = 0

        for name in stocks:
            ticker = name_to_ticker.get(name)
            if not ticker:
                # 부분 매칭 시도 (우선주 등)
                for k, v in name_to_ticker.items():
                    if name in k or k in name:
                        ticker = v
                        break
            if ticker:
                rows.append({
                    "ticker":    ticker,
                    "name":      name,
                    "theme":     theme["name"],
                    "theme_no":  theme["no"],
                })
                matched += 1
            else:
                failed += 1

        time.sleep(0.3)  # 서버 부하 방지

    print(f"\n수집 완료: {len(rows)}건 (매핑실패: {failed}건)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("theme_ticker_map.csv", index=False, encoding="utf-8-sig")
    print(f"저장: theme_ticker_map.csv")

    # 요약
    print(f"\n테마별 종목 수 (상위 10):")
    for theme, cnt in df_out.groupby("theme")["ticker"].count().sort_values(ascending=False).head(10).items():
        print(f"  {theme}: {cnt}개")

    print(f"\n샘플:")
    print(df_out[df_out["theme"] == "반도체 장비"].head(5).to_string())
