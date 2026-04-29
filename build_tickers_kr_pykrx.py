"""
국장 티커 한글명 수집 (pykrx)
- pykrx로 한글명 + 섹터 + 시총
- tickers_kr.csv 업데이트
- API 키 불필요
"""
import os, time, warnings
import pandas as pd
from pykrx import stock

warnings.filterwarnings("ignore")


def cap_label_kr(mc):
    if not mc or mc == 0: return ""
    mc = float(mc)
    if mc >= 10_000_000_000_000: return "MegaCap"
    if mc >= 1_000_000_000_000:  return "LargeCap"
    if mc >= 200_000_000_000:    return "MidCap"
    return "SmallCap"


if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y%m%d")

    # KOSPI + KOSDAQ 티커 전체
    print("KOSPI 티커 수집 중...")
    kospi_tickers  = stock.get_market_tickers(market="KOSPI")
    print(f"KOSPI: {len(kospi_tickers)}개")

    print("KOSDAQ 티커 수집 중...")
    kosdaq_tickers = stock.get_market_tickers(market="KOSDAQ")
    print(f"KOSDAQ: {len(kosdaq_tickers)}개")

    rows = []

    # KOSPI
    for i, ticker in enumerate(kospi_tickers):
        if i % 100 == 0:
            print(f"KOSPI [{i}/{len(kospi_tickers)}] 수집:{len(rows)}개")
        try:
            name = stock.get_market_ticker_name(ticker)
            # 시총
            df_cap = stock.get_market_cap(today, today, ticker)
            mc = int(df_cap["시가총액"].iloc[0]) if len(df_cap) > 0 else 0
            # 섹터
            df_sec = stock.get_market_sector_classifications(today, "KOSPI")
            sector = ""
            if ticker in df_sec.index:
                sector = str(df_sec.loc[ticker, "섹터"]) if "섹터" in df_sec.columns else ""
        except:
            name = ticker
            mc = 0
            sector = ""

        rows.append({
            "ticker":   ticker,
            "name":     name,
            "market":   "KOSPI",
            "exchange": "KO",
            "sector":   sector,
            "industry": "",
            "market_cap": mc,
            "cap":      cap_label_kr(mc),
        })
        time.sleep(0.05)

    # KOSDAQ
    for i, ticker in enumerate(kosdaq_tickers):
        if i % 100 == 0:
            print(f"KOSDAQ [{i}/{len(kosdaq_tickers)}] 수집:{len(rows)}개")
        try:
            name = stock.get_market_ticker_name(ticker)
            df_cap = stock.get_market_cap(today, today, ticker)
            mc = int(df_cap["시가총액"].iloc[0]) if len(df_cap) > 0 else 0
            df_sec = stock.get_market_sector_classifications(today, "KOSDAQ")
            sector = ""
            if ticker in df_sec.index:
                sector = str(df_sec.loc[ticker, "섹터"]) if "섹터" in df_sec.columns else ""
        except:
            name = ticker
            mc = 0
            sector = ""

        rows.append({
            "ticker":   ticker,
            "name":     name,
            "market":   "KOSDAQ",
            "exchange": "KQ",
            "sector":   sector,
            "industry": "",
            "market_cap": mc,
            "cap":      cap_label_kr(mc),
        })
        time.sleep(0.05)

    df_out = pd.DataFrame(rows)
    df_out.to_csv("tickers_kr.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {len(df_out)}개 → tickers_kr.csv")
    print(f"섹터 분포:\n{df_out['sector'].value_counts().head(10)}")
    print(f"시총 분포:\n{df_out['cap'].value_counts()}")
    print(f"\n샘플:")
    for _, r in df_out.head(5).iterrows():
        print(f"  {r['ticker']} {r['name']} {r['market']} {r['sector']} {r['cap']}")
