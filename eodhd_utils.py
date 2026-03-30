"""
EODHD API 공통 유틸리티
- get_ohlcv: 종목 OHLCV 조회
- get_tickers_us: 미장 전체 티커
- get_tickers_kr: 국장 전체 티커
"""
import os, time, warnings, requests
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

EODHD = os.environ.get("EODHD_TOKEN", "")
BASE  = "https://eodhd.com/api"


def get_ohlcv(symbol, exchange, start=None, end=None):
    """
    EODHD EOD 데이터 조회
    symbol:   종목코드 (예: AAPL, 005930)
    exchange: 거래소   (예: US, KO)
    start:    시작일   (YYYY-MM-DD, None이면 전체)
    end:      종료일   (YYYY-MM-DD, None이면 오늘)
    """
    ticker = f"{symbol}.{exchange}"
    url    = f"{BASE}/eod/{ticker}"
    params = {
        "api_token": EODHD,
        "fmt":       "json",
        "order":     "a",
        "period":    "d",
    }
    if start: params["from"] = start
    if end:   params["to"]   = end

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(10); continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data or not isinstance(data, list):
                return None

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # 컬럼 정리
            col_map = {
                "open":          "Open",
                "high":          "High",
                "low":           "Low",
                "close":         "Close",
                "adjusted_close":"Adj_Close",
                "volume":        "Volume",
            }
            df = df.rename(columns=col_map)
            cols = [c for c in ["Open","High","Low","Close","Adj_Close","Volume"] if c in df.columns]
            df = df[cols].astype(float).dropna()
            return df if len(df) > 0 else None

        except Exception as e:
            print(f"EODHD 오류(시도{attempt+1}) {ticker}: {e}")
            time.sleep(3)
    return None


def get_tickers_us():
    """
    미장 전체 티커 조회 (US 거래소)
    반환: {ticker: {name, sector, exchange, cap}} dict
    """
    url    = f"{BASE}/exchange-symbol-list/US"
    params = {"api_token": EODHD, "fmt": "json"}
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"미장 티커 조회 실패: {resp.status_code}")
            return {}
        data = resp.json()
        result = {}
        for row in data:
            ticker = str(row.get("Code", "")).strip()
            if not ticker: continue
            # 주식만 (ETF, Fund 제외)
            ttype    = str(row.get("Type", "")).lower()
            exchange = str(row.get("Exchange", ""))
            if ttype not in ("common stock", "stock", ""): continue
            # PINK/OTC 제외 → NYSE/NASDAQ/AMEX/BATS만
            if exchange not in ("NYSE", "NASDAQ", "AMEX", "BATS", "NYSE ARCA", "NYSE MKT"):
                continue
            result[ticker] = {
                "name":     str(row.get("Name", "")),
                "exchange": exchange,
                "sector":   str(row.get("Sector", "") or ""),
                "cap":      "",
            }
        print(f"미장 티커: {len(result)}개")
        return result
    except Exception as e:
        print(f"미장 티커 오류: {e}")
        return {}


def get_tickers_kr():
    """
    국장 전체 티커 조회 (KO 거래소)
    반환: {ticker: {name, sector, market}} dict
    """
    result = {}
    for exchange in ["KO", "KQ"]:
        url    = f"{BASE}/exchange-symbol-list/{exchange}"
        params = {"api_token": EODHD, "fmt": "json"}
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                print(f"국장 티커 조회 실패({exchange}): {resp.status_code}")
                continue
            data = resp.json()
            for row in data:
                ticker = str(row.get("Code", "")).strip()
                if not ticker: continue
                ttype = str(row.get("Type", "")).lower()
                if ttype not in ("common stock", "stock", ""): continue
                result[ticker] = {
                    "name":     str(row.get("Name", "")),
                    "sector":   str(row.get("Sector", "") or "기타"),
                    "market":   "KOSPI" if exchange == "KO" else "KOSDAQ",
                    "exchange": exchange,
                }
        except Exception as e:
            print(f"국장 티커 오류({exchange}): {e}")

    print(f"국장 티커: {len(result)}개")
    return result


if __name__ == "__main__":
    # 테스트
    print("=== EODHD API 테스트 ===")
    print(f"토큰: {'있음' if EODHD else '없음'}")

    # AAPL 테스트
    df = get_ohlcv("AAPL", "US", start="2024-01-01", end="2024-01-10")
    if df is not None:
        print(f"\nAAPL: {len(df)}일치")
        print(df.head(3))
    else:
        print("\nAAPL 조회 실패")

    # 삼성전자 테스트
    df_kr = get_ohlcv("005930", "KO", start="2024-01-01", end="2024-01-10")
    if df_kr is not None:
        print(f"\n삼성전자: {len(df_kr)}일치")
        print(df_kr.head(3))
    else:
        print("\n삼성전자 조회 실패")
