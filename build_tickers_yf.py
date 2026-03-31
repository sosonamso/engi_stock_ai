"""
티커 메타데이터 수집 (1회성 / 분기 수동 실행)
- EODHD Exchange API로 티커 목록
- yfinance로 섹터/시총 수집
- tickers_us.csv, tickers_kr.csv 저장
"""
import os, time, warnings
import pandas as pd
import yfinance as yf
from eodhd_utils import get_tickers_us, get_tickers_kr

warnings.filterwarnings("ignore")

MIN_MARKETCAP_US = 300_000_000      # 미장 최소 시총 3억달러
MIN_MARKETCAP_KR = 50_000_000_000   # 국장 최소 시총 500억원


def cap_label_us(mc):
    if not mc or mc == 0: return ""
    mc = float(mc)
    if mc >= 200_000_000_000: return "MegaCap"
    if mc >= 10_000_000_000:  return "LargeCap"
    if mc >= 2_000_000_000:   return "MidCap"
    return "SmallCap"


def cap_label_kr(mc):
    if not mc or mc == 0: return ""
    mc = float(mc)
    if mc >= 10_000_000_000_000: return "MegaCap"
    if mc >= 1_000_000_000_000:  return "LargeCap"
    if mc >= 200_000_000_000:    return "MidCap"
    return "SmallCap"


def get_yf_info(ticker, exchange="US"):
    """yfinance로 섹터/시총 조회"""
    try:
        symbol = f"{ticker}.KS" if exchange in ("KO", "KQ") else ticker
        info   = yf.Ticker(symbol).info
        mc     = info.get("marketCap") or 0
        return {
            "sector":     str(info.get("sector", "")    or ""),
            "industry":   str(info.get("industry", "")  or ""),
            "market_cap": float(mc),
        }
    except:
        return {"sector": "", "industry": "", "market_cap": 0.0}


if __name__ == "__main__":
    # ── 미장 ──────────────────────────────────────
    print("=== 미장 티커 수집 ===")
    us_tickers = get_tickers_us()
    print(f"기본 티커: {len(us_tickers)}개")

    us_rows = []
    for i, (ticker, info) in enumerate(us_tickers.items()):
        if i % 200 == 0:
            print(f"[{i}/{len(us_tickers)}] 수집 중... (저장:{len(us_rows)}개)")

        yf_info = get_yf_info(ticker, "US")
        mc      = yf_info["market_cap"]

        # 시총 필터 (mc=0이면 통과, 있으면 기준 적용)
        if mc > 0 and mc < MIN_MARKETCAP_US:
            time.sleep(0.3)
            continue

        cap = cap_label_us(mc)

        us_rows.append({
            "ticker":     ticker,
            "name":       info["name"],
            "exchange":   info["exchange"],
            "sector":     yf_info["sector"],
            "industry":   yf_info["industry"],
            "market_cap": mc,
            "cap":        cap,
        })
        time.sleep(0.3)

    us_df = pd.DataFrame(us_rows)
    us_df.to_csv("tickers_us_yf.csv", index=False, encoding="utf-8-sig")
    print(f"\n미장 저장: {len(us_df)}개 → tickers_us_yf.csv")
    if "cap" in us_df.columns and us_df["cap"].any():
        print(f"시총 분포:\n{us_df['cap'].value_counts()}")
    if "sector" in us_df.columns and us_df["sector"].any():
        print(f"섹터 분포 (상위10):\n{us_df['sector'].value_counts().head(10)}")

    # ── 국장 ──────────────────────────────────────
    print("\n=== 국장 티커 수집 ===")
    kr_tickers = get_tickers_kr()
    print(f"기본 티커: {len(kr_tickers)}개")

    kr_rows = []
    for i, (ticker, info) in enumerate(kr_tickers.items()):
        if i % 200 == 0:
            print(f"[{i}/{len(kr_tickers)}] 수집 중... (저장:{len(kr_rows)}개)")

        exchange = info["exchange"]
        yf_info  = get_yf_info(ticker, exchange)
        mc       = yf_info["market_cap"]

        if mc > 0 and mc < MIN_MARKETCAP_KR:
            time.sleep(0.3)
            continue

        cap = cap_label_kr(mc)

        kr_rows.append({
            "ticker":     ticker,
            "name":       info["name"],
            "market":     info["market"],
            "exchange":   exchange,
            "sector":     yf_info["sector"] or "기타",
            "industry":   yf_info["industry"],
            "market_cap": mc,
            "cap":        cap,
        })
        time.sleep(0.3)

    kr_df = pd.DataFrame(kr_rows)
    kr_df.to_csv("tickers_kr_yf.csv", index=False, encoding="utf-8-sig")
    print(f"\n국장 저장: {len(kr_df)}개 → tickers_kr_yf.csv")
    if "cap" in kr_df.columns and kr_df["cap"].any():
        print(f"시총 분포:\n{kr_df['cap'].value_counts()}")
