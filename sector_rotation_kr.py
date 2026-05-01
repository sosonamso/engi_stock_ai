"""
국장 테마 로테이션 스캐너 v2
- 시총 5000억 이상 종목만으로 테마 RS 계산
- 2주(10거래일) 코스피 대비 RS
- 상승 종목 비율 70% 이상 필터
- 주도주 Top 5
"""
import os, time, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from eodhd_utils import get_ohlcv, EODHD

warnings.filterwarnings("ignore")

TOK              = os.environ.get("TELEGRAM_TOKEN", "")
CID              = os.environ.get("TELEGRAM_CHAT_ID", "")
KR_DIR           = "raw_data/kr"
HISTORY_DAYS     = 100
MIN_CAP          = 500_000_000_000   # 시총 5000억
LEADER_RS_PCT    = 0.20              # 상위 20%
LEADER_UP_RATIO  = 0.70              # 상승 비율 70%
MAX_STOCKS       = 5                 # 테마별 Top N
RS_PERIOD        = 10                # 2주(10거래일)


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text, "parse_mode": "HTML"},
                         timeout=10)
        except: pass


def calc_rs(stock_close, idx_close, n=10):
    try:
        si = idx_close.reindex(stock_close.index).ffill()
        if len(si.dropna()) < n+1 or len(stock_close) < n+1: return None
        sr = float(stock_close.iloc[-1] / stock_close.iloc[-n] - 1) * 100
        mr = float(si.iloc[-1]          / si.iloc[-n]          - 1) * 100
        return round(sr - mr, 2)
    except: return None


def calc_volume_ratio(volume, short=5, long=20):
    try:
        if len(volume) < long: return None
        return round(float(volume.iloc[-short:].mean() / volume.iloc[-long:].mean()), 2)
    except: return None


def pct_from_52w_high(close):
    try:
        w = close.iloc[-252:] if len(close) >= 252 else close
        return round((float(close.iloc[-1]) / float(w.max()) - 1) * 100, 1)
    except: return None


def is_above_ma200(close):
    if len(close) < 200: return False
    return float(close.iloc[-1]) > float(close.rolling(200).mean().iloc[-1])


if __name__ == "__main__":
    if not EODHD:
        send("EODHD_TOKEN 없음!"); exit(1)

    # 테마 매핑 로드
    if not os.path.exists("theme_ticker_map.csv"):
        send("theme_ticker_map.csv 없음!"); exit(1)
    theme_df = pd.read_csv("theme_ticker_map.csv", encoding="utf-8-sig")
    theme_df["ticker"] = theme_df["ticker"].astype(str).str.zfill(6)

    # 티커 메타 로드
    kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
    kr_df["ticker"] = kr_df["ticker"].astype(str).str.zfill(6)
    name_map   = {r["ticker"]: r["name"]                    for _, r in kr_df.iterrows()}
    market_map = {r["ticker"]: r["market"]                  for _, r in kr_df.iterrows()}
    cap_map    = {r["ticker"]: float(r.get("market_cap",0) or 0) for _, r in kr_df.iterrows()}
    caplab_map = {r["ticker"]: str(r.get("cap","") or "")   for _, r in kr_df.iterrows()}

    # 시총 5000억 이상 티커
    large_tickers = {t for t, c in cap_map.items() if c >= MIN_CAP}
    print(f"시총 5000억↑: {len(large_tickers)}개")

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
    end_date    = datetime.today().strftime("%Y-%m-%d")
    start_date  = (datetime.today() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    data_cutoff = pd.Timestamp(datetime.today() - timedelta(days=10))

    # 분석 대상: 테마에 속하고 시총 5000억 이상
    target_tickers = list(set(theme_df["ticker"].tolist()) & large_tickers)
    print(f"분석 대상: {len(target_tickers)}개")
    send(f"🔄 테마 로테이션 스캐너 v2\n시총5000억↑ {len(target_tickers)}개 종목 분석 중...")

    # 종목별 지표 계산
    stock_stats = {}
    for i, ticker in enumerate(target_tickers):
        if i % 50 == 0:
            print(f"[{i}/{len(target_tickers)}] 처리 중...")

        exchange = "KQ" if market_map.get(ticker, "KOSPI") == "KOSDAQ" else "KO"
        df = get_ohlcv(ticker, exchange, start=start_date, end=end_date)
        time.sleep(0.05)

        if df is None or len(df) < 30: continue
        if float(df["Close"].iloc[-1]) < 1000: continue

        close  = df["Close"]
        volume = df["Volume"]

        rs_2w  = calc_rs(close, idx_close, RS_PERIOD)
        vr     = calc_volume_ratio(volume)
        pct52  = pct_from_52w_high(close)
        above200 = is_above_ma200(close)

        if rs_2w is None: continue

        # 2주 수익률 (절대)
        ret_2w = round(float(close.iloc[-1] / close.iloc[-RS_PERIOD] - 1) * 100, 2) if len(close) >= RS_PERIOD else None

        stock_stats[ticker] = {
            "ticker":    ticker,
            "name":      name_map.get(ticker, ticker),
            "market":    market_map.get(ticker, "KOSPI"),
            "cap":       cap_map.get(ticker, 0),
            "cap_label": caplab_map.get(ticker, ""),
            "cur":       round(float(close.iloc[-1]), 0),
            "rs_2w":     rs_2w,
            "ret_2w":    ret_2w,
            "vr":        vr,
            "pct52":     pct52,
            "above200":  above200,
        }

    print(f"지표 계산 완료: {len(stock_stats)}개")

    # ── 테마별 RS 집계 ─────────────────────────────
    theme_stats = []
    for theme_name, grp in theme_df.groupby("theme"):
        # 시총 5000억 이상 종목만
        tickers = [t for t in grp["ticker"].tolist() if t in stock_stats]
        if len(tickers) < 3: continue

        rs_vals  = [stock_stats[t]["rs_2w"]  for t in tickers]
        ret_vals = [stock_stats[t]["ret_2w"] for t in tickers if stock_stats[t]["ret_2w"] is not None]
        up_count = sum(1 for v in ret_vals if v > 0)
        up_ratio = up_count / len(ret_vals) if ret_vals else 0

        theme_stats.append({
            "theme":      theme_name,
            "n_large":    len(tickers),
            "rs_2w_avg":  round(np.mean(rs_vals), 2),
            "up_ratio":   round(up_ratio * 100, 1),
            "up_count":   up_count,
            "total":      len(ret_vals),
        })

    theme_rank = pd.DataFrame(theme_stats).sort_values("rs_2w_avg", ascending=False).reset_index(drop=True)
    n_theme    = len(theme_rank)
    top_n      = max(1, int(n_theme * LEADER_RS_PCT))

    # 주도테마: RS 상위 20% + 상승비율 70% 이상
    leader_mask = (
        (theme_rank.index < top_n) &
        (theme_rank["up_ratio"] >= LEADER_UP_RATIO * 100)
    )
    leader_themes = theme_rank[leader_mask]["theme"].tolist()
    print(f"주도테마: {len(leader_themes)}개")

    # ── 종목 선별 ──────────────────────────────────
    candidates = []
    for theme_name in leader_themes:
        grp     = theme_df[theme_df["theme"] == theme_name]
        tickers = [t for t in grp["ticker"].tolist() if t in stock_stats]
        stats   = [stock_stats[t] for t in tickers]

        # 조건 필터
        filtered = [s for s in stats
                    if s["above200"]
                    and s["pct52"] is not None and s["pct52"] >= -20
                    and s["vr"] is not None and s["vr"] <= 0.8]

        # RS Top 5
        top5 = sorted(filtered, key=lambda x: x["rs_2w"], reverse=True)[:MAX_STOCKS]
        for s in top5:
            candidates.append({**s, "theme": theme_name})

    print(f"후보 종목: {len(candidates)}개")

    # CSV 저장
    theme_rank.to_csv("theme_rank_kr.csv", index=False, encoding="utf-8-sig")
    if candidates:
        pd.DataFrame(candidates).to_csv("theme_rotation_kr.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv("theme_rotation_kr.csv", index=False, encoding="utf-8-sig")

    # ── 텔레그램 ───────────────────────────────────
    tv_url = lambda t: f"https://www.tradingview.com/chart/?symbol=KRX:{t}"
    cap_emoji = {"MegaCap":"💎","LargeCap":"🔷","MidCap":"🔹","SmallCap":"▪️"}

    # 테마 랭킹
    msg  = f"🔄 <b>국장 테마 로테이션 v2</b>\n"
    msg += f"시총5000억↑ | {datetime.today().strftime('%Y-%m-%d')}\n"
    msg += "─" * 22 + "\n\n"
    msg += f"📈 <b>테마 RS 랭킹 Top10 (2주)</b>\n"

    for i, row in theme_rank.head(10).iterrows():
        emoji = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"{i+1}."
        leader = " ⭐" if row["theme"] in leader_themes else ""
        msg   += (f"{emoji} {row['theme']}{leader}\n"
                  f"   2주RS:{row['rs_2w_avg']:+.1f}% | "
                  f"상승{row['up_count']}/{row['total']}({row['up_ratio']:.0f}%)\n")

    send(msg)

    if not candidates:
        send("⚠️ 주도테마 후보 종목 없음\n(200MA↑ + 52주고점-20% + 거래량눌림)")
    else:
        cur_theme = None
        stock_msg = f"🎯 <b>주도테마 Top5 종목</b>\n"
        stock_msg += "(200MA↑ + 52주고점-20% + 거래량눌림)\n"
        stock_msg += "─" * 22 + "\n\n"

        for r in candidates:
            if r["theme"] != cur_theme:
                if cur_theme is not None:
                    send(stock_msg)
                    stock_msg = ""
                tr = theme_rank[theme_rank["theme"] == r["theme"]].iloc[0]
                rank_idx = theme_rank[theme_rank["theme"] == r["theme"]].index[0] + 1
                stock_msg += (f"⭐ <b>{r['theme']}</b> ({rank_idx}위)\n"
                              f"   2주RS:{tr['rs_2w_avg']:+.1f}% | "
                              f"상승{tr['up_count']}/{tr['total']}({tr['up_ratio']:.0f}%)\n\n")
                cur_theme = r["theme"]

            mkt   = "🔵코스피" if r["market"] == "KOSPI" else "🟢코스닥"
            cap_e = cap_emoji.get(r["cap_label"], "")
            vr    = f"{r['vr']:.2f}x" if r["vr"] else "-"
            blk   = (
                f"{cap_e} <b>{r['name']}</b>({r['ticker']}) {mkt}\n"
                f"  현재가: {r['cur']:,.0f}원\n"
                f"  52주고점: {r['pct52']:+.1f}% | 거래량: {vr}\n"
                f"  RS 2주: {r['rs_2w']:+.1f}%\n"
                f"  📊 {tv_url(r['ticker'])}\n\n"
            )
            if len(stock_msg) + len(blk) > 3800:
                send(stock_msg); stock_msg = blk
            else:
                stock_msg += blk

        if stock_msg: send(stock_msg)

    send(f"✅ 완료 | 주도테마 {len(leader_themes)}개 | 후보 {len(candidates)}개")
