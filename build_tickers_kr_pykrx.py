"""
국장 티커 한글명 수집 (FinanceDataReader)
- KRX 전체 종목 한글명 + 섹터 + 시총
- tickers_kr.csv 업데이트
- API 키 불필요
"""
import warnings
import pandas as pd
import FinanceDataReader as fdr

warnings.filterwarnings("ignore")


def cap_label_kr(mc):
    if not mc or mc == 0: return ""
    mc = float(mc)
    if mc >= 10_000_000_000_000: return "MegaCap"
    if mc >= 1_000_000_000_000:  return "LargeCap"
    if mc >= 200_000_000_000:    return "MidCap"
    return "SmallCap"


if __name__ == "__main__":
    print("KRX 전체 종목 수집 중...")
    df = fdr.StockListing('KRX')
    print(f"수집: {len(df)}개")
    print(f"컬럼: {df.columns.tolist()}")
    print(df.head(3))

    rows = []
    for _, r in df.iterrows():
        ticker = str(r.get("Code", r.get("Symbol", ""))).zfill(6)
        name   = str(r.get("Name", ticker))
        market = str(r.get("Market", "KOSPI"))
        sector = str(r.get("Sector", "") or r.get("Industry", "") or "")
        mc     = float(r.get("Marcap", r.get("MarketCap", 0)) or 0)
        exchange = "KO" if market == "KOSPI" else "KQ"

        rows.append({
            "ticker":     ticker,
            "name":       name,
            "market":     market,
            "exchange":   exchange,
            "sector":     sector,
            "industry":   str(r.get("Industry", "") or ""),
            "market_cap": mc,
            "cap":        cap_label_kr(mc),
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out[df_out["ticker"].str.match(r'^\d{6}$')]  # 숫자 6자리만
    df_out.to_csv("tickers_kr.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {len(df_out)}개 → tickers_kr.csv")
    print(f"\n시총 분포:\n{df_out['cap'].value_counts()}")
    print(f"\n샘플:")
    for _, r in df_out.head(5).iterrows():
        print(f"  {r['ticker']} {r['name']} {r['market']} {r['sector']} {r['cap']}")
