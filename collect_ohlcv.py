"""
전체 종목 OHLCV 수집
- tickers_us.csv / tickers_kr.csv 기반
- raw_data/us/{ticker}.csv
- raw_data/kr/{ticker}.csv
- 이미 있는 파일은 마지막 날짜 이후만 incremental update
"""
import os, time, warnings
import pandas as pd
from datetime import datetime, timedelta
from eodhd_utils import get_ohlcv, EODHD

warnings.filterwarnings("ignore")

YEARS      = 20
START_DATE = (datetime.today() - timedelta(days=365*YEARS)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")
US_DIR     = "raw_data/us"
KR_DIR     = "raw_data/kr"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_or_fetch(ticker, exchange, filepath):
    """
    이미 파일 있으면 마지막 날짜 이후만 수집 (incremental)
    없으면 전체 수집
    """
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, index_col="date", parse_dates=True)
        last_date = existing.index.max()
        # 마지막 날짜 다음날부터 수집
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start >= END_DATE:
            return existing  # 이미 최신
        new_df = get_ohlcv(ticker, exchange, start=fetch_start, end=END_DATE)
        if new_df is not None and len(new_df) > 0:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            return combined
        return existing
    else:
        return get_ohlcv(ticker, exchange, start=START_DATE, end=END_DATE)


def save_csv(df, filepath):
    df.to_csv(filepath, index=True)


if __name__ == "__main__":
    ensure_dir(US_DIR)
    ensure_dir(KR_DIR)

    # ── 미장 ──────────────────────────────────────
    if os.path.exists("tickers_us.csv"):
        us_df = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        us_tickers = us_df["ticker"].astype(str).tolist()
        print(f"미장 티커: {len(us_tickers)}개")

        ok = skip = fail = 0
        for i, ticker in enumerate(us_tickers):
            if i % 200 == 0:
                print(f"[미장 {i}/{len(us_tickers)}] ok:{ok} skip:{skip} fail:{fail}")
            filepath = os.path.join(US_DIR, f"{ticker}.csv")
            df = load_or_fetch(ticker, "US", filepath)
            if df is not None and len(df) >= 100:
                save_csv(df, filepath)
                ok += 1
            elif df is not None and len(df) > 0:
                save_csv(df, filepath)
                skip += 1
            else:
                fail += 1
            time.sleep(0.05)

        print(f"미장 완료: ok={ok} skip={skip} fail={fail}")
    else:
        print("tickers_us.csv 없음 → 미장 스킵")

    # ── 국장 ──────────────────────────────────────
    if os.path.exists("tickers_kr.csv"):
        kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
        print(f"국장 티커: {len(kr_df)}개")

        ok = skip = fail = 0
        for i, row in kr_df.iterrows():
            if i % 200 == 0:
                print(f"[국장 {i}/{len(kr_df)}] ok:{ok} skip:{skip} fail:{fail}")
            ticker   = str(row["ticker"])
            exchange = str(row["exchange"])  # KO or KQ
            filepath = os.path.join(KR_DIR, f"{ticker}.csv")
            df = load_or_fetch(ticker, exchange, filepath)
            if df is not None and len(df) >= 100:
                save_csv(df, filepath)
                ok += 1
            elif df is not None and len(df) > 0:
                save_csv(df, filepath)
                skip += 1
            else:
                fail += 1
            time.sleep(0.05)

        print(f"국장 완료: ok={ok} skip={skip} fail={fail}")
    else:
        print("tickers_kr.csv 없음 → 국장 스킵")

    print("\n전체 수집 완료!")
    print(f"미장: {len(os.listdir(US_DIR))}개 파일")
    print(f"국장: {len(os.listdir(KR_DIR))}개 파일")
