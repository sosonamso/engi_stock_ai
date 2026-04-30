"""
국장 섹터 로테이션 스캐너
- 주도/부상 섹터 판별
- 섹터 내 후보 종목 추출
- 텔레그램 전송
"""
import os, time, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from eodhd_utils import get_ohlcv, EODHD

warnings.filterwarnings("ignore")

TOK          = os.environ.get("TELEGRAM_TOKEN", "")
CID          = os.environ.get("TELEGRAM_CHAT_ID", "")
KR_DIR       = "raw_data/kr"
HISTORY_DAYS = 400   # RS 계산용 충분한 기간
TOP_SECTORS  = 3     # 주도섹터 상위 N개
MIN_STOCKS   = 3     # 섹터 내 최소 종목 수


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text, "parse_mode": "HTML"},
                         timeout=10)
        except: pass


def calc_ret(series, n):
    """n일 수익률"""
    if len(series) < n + 1: return None
    return float(series.iloc[-1] / series.iloc[-n] - 1) * 100


def calc_rs_simple(stock_close, idx_close, n):
    """단순 RS: 종목 n일 수익률 - 지수 n일 수익률"""
    if len(stock_close) < n + 1 or len(idx_close) < n + 1:
        return None
    sr = float(stock_close.iloc[-1] / stock_close.iloc[-n] - 1) * 100
    mr = float(idx_close.iloc[-1] / idx_close.iloc[-n] - 1) * 100
    return round(sr - mr, 2)


def calc_volume_ratio(volume, short=5, long=20):
    """단기/장기 거래량 비율"""
    if len(volume) < long: return None
    short_avg = volume.iloc[-short:].mean()
    long_avg  = volume.iloc[-long:].mean()
    if long_avg == 0: return None
    return round(float(short_avg / long_avg), 2)


def pct_from_52w_high(close):
    """52주 고점 대비 현재가 위치"""
    if len(close) < 2: return None
    window = close.iloc[-252:] if len(close) >= 252 else close
    high52 = window.max()
    if high52 == 0: return None
    return round((float(close.iloc[-1]) / float(high52) - 1) * 100, 1)


def is_above_ma200(close):
    if len(close) < 200: return False
    ma200 = close.rolling(200).mean().iloc[-1]
    return float(close.iloc[-1]) > float(ma200)


if __name__ == "__main__":
    if not EODHD:
        send("EODHD_TOKEN 없음!"); exit(1)

    # 티커 로드
    meta_path = "tickers_kr_yf.csv" if os.path.exists("tickers_kr_yf.csv") else "tickers_kr.csv"
    meta_df   = pd.read_csv(meta_path, encoding="utf-8-sig")
    # 한글명은 tickers_kr.csv에서
    kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig") if os.path.exists("tickers_kr.csv") else None
    if kr_df is not None:
        kr_name_map = {str(r["ticker"]).zfill(6): str(r["name"]) for _, r in kr_df.iterrows()}
    else:
        kr_name_map = {}

    # 섹터 있는 종목만
    meta_df["ticker"] = meta_df["ticker"].astype(str).str.zfill(6)
    meta_df = meta_df[meta_df["sector"].notna() & (meta_df["sector"] != "") & (meta_df["sector"] != "nan")]
    print(f"섹터 있는 종목: {len(meta_df)}개")
    print(f"섹터 종류: {meta_df['sector'].nunique()}개")

    meta_map = {r["ticker"]: r.to_dict() for _, r in meta_df.iterrows()}

    # KOSPI 지수 로드
    kospi_idx  = None
    kospi_path = os.path.join(KR_DIR, "069500.csv")
    if os.path.exists(kospi_path):
        kospi_idx = pd.read_csv(kospi_path, index_col="date", parse_dates=True)
        print(f"KOSPI 지수: {len(kospi_idx)}일치")
    else:
        df_tmp = get_ohlcv("069500", "KO", start="2000-01-01")
        if df_tmp is not None:
            kospi_idx = df_tmp
            print(f"KOSPI 지수 수집: {len(kospi_idx)}일치")

    if kospi_idx is None:
        send("KOSPI 지수 없음!"); exit(1)

    idx_close = kospi_idx["Close"]

    # 날짜 설정
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    data_cutoff = pd.Timestamp(datetime.today() - timedelta(days=10))

    send(f"🔄 섹터 로테이션 스캐너 시작\n{len(meta_map)}개 종목 분석 중...")

    # 종목별 데이터 수집 + 지표 계산
    stock_stats = []
    ticker_list = list(meta_map.keys())

    for i, ticker in enumerate(ticker_list):
        if i % 200 == 0:
            print(f"[{i}/{len(ticker_list)}] 처리 중...")

        info     = meta_map[ticker]
        exchange = str(info.get("exchange", "KO"))

        # OHLCV 로드 (캐시 우선)
        cache_path = os.path.join(KR_DIR, f"{ticker}.csv")
        df = None
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
                df = df[["Open","High","Low","Close","Volume"]].astype(float).dropna()
            except: df = None

        if df is None or len(df) < 60 or df.index[-1] < data_cutoff:
            df = get_ohlcv(ticker, exchange, start=start_date, end=end_date)
            time.sleep(0.05)

        if df is None or len(df) < 60: continue
        if float(df["Close"].iloc[-1]) < 1000: continue

        close  = df["Close"]
        volume = df["Volume"]

        # 지표 계산
        rs_4w  = calc_rs_simple(close, idx_close.reindex(close.index).ffill(), 20)
        rs_12w = calc_rs_simple(close, idx_close.reindex(close.index).ffill(), 60)
        vr     = calc_volume_ratio(volume)
        pct52  = pct_from_52w_high(close)
        above200 = is_above_ma200(close)
        cur    = float(close.iloc[-1])

        if rs_4w is None or rs_12w is None: continue

        # 한글명 우선
        name = kr_name_map.get(ticker, str(info.get("name", ticker)))

        stock_stats.append({
            "ticker":   ticker,
            "name":     name,
            "sector":   str(info.get("sector", "")),
            "market":   str(info.get("market", "KOSPI")),
            "cap":      str(info.get("cap", "")),
            "cur":      round(cur, 0),
            "rs_4w":    rs_4w,
            "rs_12w":   rs_12w,
            "vr":       vr,
            "pct52":    pct52,
            "above200": above200,
        })

    print(f"\n지표 계산 완료: {len(stock_stats)}개")
    if not stock_stats:
        send("분석 가능한 종목 없음"); exit(0)

    df_stats = pd.DataFrame(stock_stats)

    # ── 섹터별 RS 집계 ─────────────────────────────
    sector_df = df_stats.groupby("sector").agg(
        n_stocks  = ("ticker",  "count"),
        rs_4w_avg = ("rs_4w",  "mean"),
        rs_12w_avg= ("rs_12w", "mean"),
        rs_4w_pos = ("rs_4w",  lambda x: (x > 0).mean() * 100),
    ).reset_index()

    sector_df = sector_df[sector_df["n_stocks"] >= MIN_STOCKS]
    sector_df = sector_df.sort_values("rs_4w_avg", ascending=False).reset_index(drop=True)
    sector_df["rank_4w"]  = sector_df["rs_4w_avg"].rank(ascending=False)
    sector_df["rank_12w"] = sector_df["rs_12w_avg"].rank(ascending=False)

    n_sec = len(sector_df)

    # 주도섹터: 4주 + 12주 둘 다 상위 1/3
    leader_secs = sector_df[
        (sector_df["rank_4w"]  <= n_sec / 3) &
        (sector_df["rank_12w"] <= n_sec / 3)
    ]["sector"].tolist()

    # 부상섹터: 4주 상위 1/3 + 12주 중하위 1/2 (최근에 올라오는 중)
    rising_secs = sector_df[
        (sector_df["rank_4w"]  <= n_sec / 3) &
        (sector_df["rank_12w"] >  n_sec / 2) &
        (~sector_df["sector"].isin(leader_secs))
    ]["sector"].tolist()

    print(f"\n주도섹터: {leader_secs}")
    print(f"부상섹터: {rising_secs}")

    # ── 후보 종목 필터 ─────────────────────────────
    target_secs = leader_secs + rising_secs
    candidates  = df_stats[
        df_stats["sector"].isin(target_secs) &
        df_stats["above200"] &
        (df_stats["pct52"] >= -15) &          # 52주 고점 -15% 이내
        (df_stats["vr"].notna()) &
        (df_stats["vr"] <= 0.8)               # 거래량 감소 중 (눌림)
    ].copy()

    # 섹터 내 RS 상위 50%
    filtered = []
    for sec in target_secs:
        sub = candidates[candidates["sector"] == sec]
        if len(sub) == 0: continue
        median_rs = sub["rs_4w"].median()
        top_half  = sub[sub["rs_4w"] >= median_rs]
        filtered.append(top_half)

    if filtered:
        df_cand = pd.concat(filtered).sort_values(["sector","rs_4w"], ascending=[True, False])
    else:
        df_cand = pd.DataFrame()

    print(f"후보 종목: {len(df_cand)}개")

    # CSV 저장
    df_stats.to_csv("sector_stats_kr.csv", index=False, encoding="utf-8-sig")
    sector_df.to_csv("sector_rank_kr.csv", index=False, encoding="utf-8-sig")
    if len(df_cand) > 0:
        df_cand.to_csv("sector_rotation_kr.csv", index=False, encoding="utf-8-sig")

    # ── 텔레그램 메시지 ────────────────────────────
    tv_url = lambda t: f"https://www.tradingview.com/chart/?symbol=KRX:{t}"

    # 섹터 랭킹 메시지
    msg = "🔄 <b>국장 섹터 로테이션</b>\n"
    msg += f"분석: {len(df_stats)}개 종목 | {datetime.today().strftime('%Y-%m-%d')}\n"
    msg += "─" * 22 + "\n\n"

    msg += "📈 <b>섹터 RS 랭킹 (4주 기준)</b>\n"
    for i, row in sector_df.head(8).iterrows():
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        tag   = " 🚀" if row["sector"] in rising_secs else " ⭐" if row["sector"] in leader_secs else ""
        msg  += (f"{emoji} {row['sector']}{tag}\n"
                 f"   4주:{row['rs_4w_avg']:+.1f}% | 12주:{row['rs_12w_avg']:+.1f}% | "
                 f"n={int(row['n_stocks'])}\n")

    msg += "\n"

    if len(df_cand) == 0:
        msg += "⚠️ 후보 종목 없음\n(조건: 200MA↑ + 52주고점-15% + 거래량눌림)"
        send(msg)
    else:
        msg += f"🎯 <b>후보 종목 {len(df_cand)}개</b>\n"
        msg += "(200MA↑ + 52주고점-15%내 + 거래량눌림)\n"
        msg += "─" * 22 + "\n\n"
        send(msg)

        # 종목 메시지 (섹터별)
        cur_sec = None
        stock_msg = ""
        for _, r in df_cand.iterrows():
            sec_label = r["sector"]
            if sec_label != cur_sec:
                if stock_msg:
                    send(stock_msg)
                sec_type = "⭐ 주도" if sec_label in leader_secs else "🚀 부상"
                stock_msg = f"<b>[{sec_type}] {sec_label}</b>\n\n"
                cur_sec = sec_label

            mkt  = "🔵" if r["market"] == "KOSPI" else "🟢"
            vol  = f"{r['vr']:.2f}x" if r["vr"] else "-"
            blk  = (
                f"{mkt} <b>{r['name']}</b>({r['ticker']})\n"
                f"  현재가: {r['cur']:,.0f}원\n"
                f"  52주고점대비: {r['pct52']:+.1f}%\n"
                f"  RS 4주: {r['rs_4w']:+.1f}% | 12주: {r['rs_12w']:+.1f}%\n"
                f"  거래량: {vol} (눌림)\n"
                f"  📊 {tv_url(r['ticker'])}\n\n"
            )
            if len(stock_msg) + len(blk) > 3800:
                send(stock_msg)
                stock_msg = blk
            else:
                stock_msg += blk

        if stock_msg:
            send(stock_msg)

    send(f"✅ 섹터 로테이션 스캔 완료\n후보 {len(df_cand)}개")
