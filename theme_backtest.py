"""
테마 로테이션 백테스트
- 특정 테마의 가속도(2주RS - 6주RS)가
  언제 처음 상위권 진입했는지 확인
- 그 시점 이후 주가 흐름 분석
"""
import os, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

KR_DIR      = "raw_data/kr"
TARGET_THEME = "전선"        # 분석할 테마
RS_PERIOD    = 10            # 2주
PREV_START   = 10            # 6주 시작
PREV_END     = 40            # 6주 끝
MIN_CAP      = 500_000_000_000
TOP_RANK     = 5             # 가속도 상위 N위 이내


def calc_rs(close_arr, idx_arr, n):
    """직전 n거래일 RS"""
    if len(close_arr) < n+1 or len(idx_arr) < n+1: return None
    sr = float(close_arr[-1] / close_arr[-n] - 1) * 100
    mr = float(idx_arr[-1]  / idx_arr[-n]  - 1) * 100
    return round(sr - mr, 2)


def calc_rs_prev(close_arr, idx_arr, start, end):
    """이전 구간 RS"""
    if len(close_arr) < end+1 or len(idx_arr) < end+1: return None
    sr = float(close_arr[-start] / close_arr[-end] - 1) * 100
    mr = float(idx_arr[-start]  / idx_arr[-end]  - 1) * 100
    return round(sr - mr, 2)


if __name__ == "__main__":
    # 테마 매핑 로드
    theme_df = pd.read_csv("theme_ticker_map.csv", encoding="utf-8-sig")
    theme_df["ticker"] = theme_df["ticker"].astype(str).str.zfill(6)

    # 티커 메타
    kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
    kr_df["ticker"] = kr_df["ticker"].astype(str).str.zfill(6)
    cap_map  = {r["ticker"]: float(r.get("market_cap", 0) or 0) for _, r in kr_df.iterrows()}
    name_map = {r["ticker"]: r["name"] for _, r in kr_df.iterrows()}

    # 타겟 테마 종목 (시총 5000억 이상)
    target_tickers = [
        t for t in theme_df[theme_df["theme"] == TARGET_THEME]["ticker"].tolist()
        if cap_map.get(t, 0) >= MIN_CAP
    ]
    print(f"타겟 테마: {TARGET_THEME}")
    print(f"시총5000억↑ 종목: {target_tickers}")
    print(f"종목명: {[name_map.get(t, t) for t in target_tickers]}")

    # 전체 테마 목록 (비교용)
    all_themes = theme_df["theme"].unique().tolist()

    # KOSPI 지수 로드 (없으면 EODHD에서 수집)
    kospi_path = os.path.join(KR_DIR, "069500.csv")
    if os.path.exists(kospi_path):
        kospi_df = pd.read_csv(kospi_path, index_col="date", parse_dates=True)
        print(f"KOSPI 지수 캐시: {len(kospi_df)}일치")
    else:
        import requests, os as _os
        token = _os.environ.get("EODHD_TOKEN", "")
        if not token:
            print("EODHD_TOKEN 없음!"); exit(1)
        print("KOSPI 지수 수집 중...")
        url    = f"https://eodhd.com/api/eod/069500.KO"
        params = {"api_token": token, "fmt": "json", "from": "2000-01-01"}
        resp   = requests.get(url, params=params, timeout=30)
        data   = resp.json()
        kospi_df = pd.DataFrame(data).rename(columns={"date":"date","close":"Close"})
        kospi_df["date"] = pd.to_datetime(kospi_df["date"])
        kospi_df = kospi_df.set_index("date")[["Close"]].astype(float)
        os.makedirs(KR_DIR, exist_ok=True)
        kospi_df.to_csv(kospi_path)
        print(f"KOSPI 지수 수집 완료: {len(kospi_df)}일치")
    idx_close = kospi_df["Close"]

    # 전체 테마 종목 OHLCV 로드 (EODHD 직접 수집 - 최신 데이터)
    import requests as _req, time as _time
    token = os.environ.get("EODHD_TOKEN", "")
    market_map_all = {r["ticker"]: r["market"] for _, r in kr_df.iterrows()}

    all_tickers = list(set(
        t for t in theme_df["ticker"].tolist()
        if cap_map.get(t, 0) >= MIN_CAP
    ))
    print(f"\n전체 분석 종목: {len(all_tickers)}개 (EODHD 직접 수집)")

    ohlcv = {}
    for i, ticker in enumerate(all_tickers):
        if i % 50 == 0:
            print(f"[{i}/{len(all_tickers)}] 수집 중...")
        try:
            exchange = "KQ" if market_map_all.get(ticker, "KOSPI") == "KOSDAQ" else "KO"
            url    = f"https://eodhd.com/api/eod/{ticker}.{exchange}"
            params = {"api_token": token, "fmt": "json", "from": "2024-01-01"}
            resp   = _req.get(url, params=params, timeout=15)
            if resp.status_code != 200: continue
            data   = resp.json()
            if not data: continue
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["close","volume"]].rename(
                columns={"close":"Close","volume":"Volume"}).astype(float)
            ohlcv[ticker] = df
        except: pass
        _time.sleep(0.05)

    print(f"수집 완료: {len(ohlcv)}개")

    # 날짜 범위 설정 (최근 1.5년)
    all_dates = sorted(kospi_df.index)
    start_idx = max(0, len(all_dates) - 380)  # 약 1.5년
    dates     = all_dates[start_idx:]
    print(f"백테스트 기간: {dates[0].date()} ~ {dates[-1].date()}")

    # 날짜별 테마 가속도 계산
    results = []

    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[{di}/{len(dates)}] {date.date()}")

        # 각 테마별 가속도 계산
        theme_accels = {}
        for theme_name in all_themes:
            tickers = [
                t for t in theme_df[theme_df["theme"] == theme_name]["ticker"].tolist()
                if cap_map.get(t, 0) >= MIN_CAP and t in ohlcv
            ]
            if len(tickers) < 2: continue

            accels = []
            for ticker in tickers:
                df = ohlcv[ticker]
                sl = df.loc[:date]
                if len(sl) < PREV_END + 1: continue

                idx_sl = idx_close.reindex(sl.index).ffill().dropna()
                if len(idx_sl) < PREV_END + 1: continue

                c = sl["Close"].values
                m = idx_sl.values

                rs2w = calc_rs(c, m, RS_PERIOD)
                rs6w = calc_rs_prev(c, m, PREV_START, PREV_END)
                if rs2w is None or rs6w is None: continue
                accels.append(rs2w - rs6w)

            if accels:
                theme_accels[theme_name] = round(np.mean(accels), 2)

        if not theme_accels: continue

        # 순위 산출
        ranked = sorted(theme_accels.items(), key=lambda x: x[1], reverse=True)
        rank_map = {t: i+1 for i, (t, _) in enumerate(ranked)}

        target_accel = theme_accels.get(TARGET_THEME)
        target_rank  = rank_map.get(TARGET_THEME)

        results.append({
            "date":         date,
            "target_accel": target_accel,
            "target_rank":  target_rank,
            "n_themes":     len(theme_accels),
        })

    df_result = pd.DataFrame(results)

    # 타겟 테마가 TOP_RANK 이내 처음 진입한 날
    top_entries = df_result[
        df_result["target_rank"].notna() &
        (df_result["target_rank"] <= TOP_RANK)
    ]

    print(f"\n{'='*50}")
    print(f"[ {TARGET_THEME} 테마 가속도 상위 {TOP_RANK}위 이내 진입 날짜 ]")
    print(f"{'='*50}")

    if len(top_entries) == 0:
        print("해당 기간 내 상위권 진입 없음")
    else:
        # 연속 구간 찾기
        first_entry = top_entries.iloc[0]
        print(f"첫 진입: {first_entry['date'].date()}")
        print(f"  순위: {int(first_entry['target_rank'])}위 / {int(first_entry['n_themes'])}개")
        print(f"  가속도: {first_entry['target_accel']:+.1f}%")

        # 진입 이후 전선 종목 수익률
        entry_date = first_entry["date"]
        print(f"\n[ 진입 이후 {TARGET_THEME} 종목 수익률 ]")
        print(f"{'종목':15s} {'진입가':>10} {'2주후':>8} {'4주후':>8} {'8주후':>8}")
        print("-" * 55)

        for ticker in target_tickers:
            if ticker not in ohlcv: continue
            df = ohlcv[ticker]
            sl = df.loc[entry_date:]
            if len(sl) < 2: continue
            entry_price = float(sl["Close"].iloc[0])
            r2w  = round((float(sl["Close"].iloc[min(10, len(sl)-1)]) / entry_price - 1)*100, 1) if len(sl) > 10 else None
            r4w  = round((float(sl["Close"].iloc[min(20, len(sl)-1)]) / entry_price - 1)*100, 1) if len(sl) > 20 else None
            r8w  = round((float(sl["Close"].iloc[min(40, len(sl)-1)]) / entry_price - 1)*100, 1) if len(sl) > 40 else None
            name = name_map.get(ticker, ticker)
            print(f"{name:15s} {entry_price:>10,.0f}"
                  f" {(str(r2w)+'%') if r2w else '-':>8}"
                  f" {(str(r4w)+'%') if r4w else '-':>8}"
                  f" {(str(r8w)+'%') if r8w else '-':>8}")

    # 전체 추이 저장
    df_result.to_csv(f"backtest_{TARGET_THEME}.csv", index=False, encoding="utf-8-sig")
    print(f"\n추이 저장: backtest_{TARGET_THEME}.csv")

    # 최근 30일 순위 추이
    print(f"\n[ 최근 30일 {TARGET_THEME} 순위 추이 ]")
    print(f"{'날짜':12s} {'순위':>6} {'가속도':>8}")
    print("-" * 30)
    for _, r in df_result.tail(30).iterrows():
        rank = int(r["target_rank"]) if pd.notna(r["target_rank"]) else "-"
        acc  = f"{r['target_accel']:+.1f}%" if pd.notna(r["target_accel"]) else "-"
        marker = " ★" if pd.notna(r["target_rank"]) and r["target_rank"] <= TOP_RANK else ""
        print(f"{str(r['date'].date()):12s} {str(rank):>6} {acc:>8}{marker}")
