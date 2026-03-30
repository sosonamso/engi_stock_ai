"""
티커 메타데이터 수집 (1회성)
- EODHD Fundamental API로 섹터/시총 수집
- tickers_us.csv, tickers_kr.csv 저장
"""
import os, time, requests
import pandas as pd
from eodhd_utils import get_tickers_us, get_tickers_kr, EODHD, BASE

MIN_MARKETCAP_US = 300_000_000   # 미장 최소 시총 3억달러
MIN_MARKETCAP_KR = 50_000_000_000  # 국장 최소 시총 500억원


def get_fundamental(symbol, exchange):
    """섹터/시총 조회"""
    url    = f"{BASE}/fundamentals/{symbol}.{exchange}"
    params = {"api_token": EODHD, "fmt": "json", "filter": "General"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200: return {}
        data = resp.json()
        if not isinstance(data, dict): return {}
        # General 필터시 바로 필드가 오거나 General 키 안에 있을 수 있음
        if "General" in data:
            data = data["General"]
        mc = data.get("MarketCapitalization") or data.get("market_capitalization") or 0
        try: mc = float(mc)
        except: mc = 0.0
        return {
            "sector":     str(data.get("Sector", "")    or ""),
            "industry":   str(data.get("Industry", "")  or ""),
            "market_cap": mc,
        }
    except: return {}


def cap_label(market_cap):
    """시총 그룹 분류"""
    if not market_cap: return "SmallCap"
    mc = float(market_cap)
    if mc >= 200_000_000_000: return "MegaCap"
    if mc >= 10_000_000_000:  return "LargeCap"
    if mc >= 2_000_000_000:   return "MidCap"
    return "SmallCap"


def cap_label_kr(market_cap):
    """국장 시총 그룹 분류 (원화 기준)"""
    if not market_cap: return "SmallCap"
    mc = float(market_cap)
    if mc >= 10_000_000_000_000: return "MegaCap"   # 10조+
    if mc >= 1_000_000_000_000:  return "LargeCap"  # 1조+
    if mc >= 200_000_000_000:    return "MidCap"    # 2000억+
    return "SmallCap"


if __name__ == "__main__":
    # ── 미장 ──────────────────────────────────────
    print("=== 미장 티커 수집 ===")
    us_tickers = get_tickers_us()
    print(f"기본 티커: {len(us_tickers)}개")

    us_rows = []
    for i, (ticker, info) in enumerate(us_tickers.items()):
        if i % 200 == 0:
            print(f"[{i}/{len(us_tickers)}] 수집 중...")

        fund = get_fundamental(ticker, "US")
        mc   = fund.get("market_cap", 0) or 0

        # 시총 필터 (3억달러 미만 제외, mc=0이면 통과)
        if float(mc) > 0 and float(mc) < MIN_MARKETCAP_US:
            time.sleep(0.05)
            continue

        us_rows.append({
            "ticker":    ticker,
            "name":      info["name"],
            "exchange":  info["exchange"],
            "sector":    fund.get("sector", ""),
            "industry":  fund.get("industry", ""),
            "market_cap": mc,
            "cap":       cap_label(mc),
        })
        time.sleep(0.05)  # 분당 1000 제한 대응

    us_df = pd.DataFrame(us_rows)
    us_df.to_csv("tickers_us.csv", index=False, encoding="utf-8-sig")
    print(f"미장 저장: {len(us_df)}개 → tickers_us.csv")
    if 'cap' in us_df.columns: print(f"시총 분포:\n{us_df['cap'].value_counts()}")

    # ── 국장 ──────────────────────────────────────
    print("\n=== 국장 티커 수집 ===")
    kr_tickers = get_tickers_kr()
    print(f"기본 티커: {len(kr_tickers)}개")

    kr_rows = []
    for i, (ticker, info) in enumerate(kr_tickers.items()):
        if i % 200 == 0:
            print(f"[{i}/{len(kr_tickers)}] 수집 중...")

        exchange = info["exchange"]
        fund = get_fundamental(ticker, exchange)
        mc   = fund.get("market_cap", 0) or 0

        # 시총 필터 (500억 미만 제외, mc=0이면 통과)
        if float(mc) > 0 and float(mc) < MIN_MARKETCAP_KR:
            time.sleep(0.05)
            continue

        kr_rows.append({
            "ticker":    ticker,
            "name":      info["name"],
            "market":    info["market"],
            "exchange":  exchange,
            "sector":    fund.get("sector", "기타") or "기타",
            "industry":  fund.get("industry", ""),
            "market_cap": mc,
            "cap":       cap_label_kr(mc),
        })
        time.sleep(0.05)

    kr_df = pd.DataFrame(kr_rows)
    kr_df.to_csv("tickers_kr.csv", index=False, encoding="utf-8-sig")
    print(f"국장 저장: {len(kr_df)}개 → tickers_kr.csv")
    if 'cap' in kr_df.columns: print(f"시총 분포:\n{kr_df['cap'].value_counts()}")
