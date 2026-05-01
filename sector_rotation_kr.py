"""
국장 테마 로테이션 스캐너 (네이버 테마 기반)
- 테마별 RS 계산 + 주도/부상 테마 판별
- 테마 내 후보 종목 추출
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
HISTORY_DAYS = 400
MIN_STOCKS   = 5     # 테마 내 최소 종목 수


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text, "parse_mode": "HTML"},
                         timeout=10)
        except: pass


def calc_rs_simple(stock_close, idx_close, n):
    try:
        si = idx_close.reindex(stock_close.index).ffill()
        if len(si.dropna()) < n + 1 or len(stock_close) < n + 1:
            return None
        sr = float(stock_close.iloc[-1] / stock_close.iloc[-n] - 1) * 100
        mr = float(si.iloc[-1] / si.iloc[-n] - 1) * 100
        return round(sr - mr, 2)
    except: return None


def calc_volume_ratio(volume, short=5, long=20):
    try:
        if len(volume) < long: return None
        return round(float(volume.iloc[-short:].mean() / volume.iloc[-long:].mean()), 2)
    except: return None


def pct_from_52w_high(close):
    try:
        window = close.iloc[-252:] if len(close) >= 252 else close
        return round((float(close.iloc[-1]) / float(window.max()) - 1) * 100, 1)
    except: return None


def is_above_ma200(close):
    if len(close) < 200: return False
    return float(close.iloc[-1]) > float(close.rolling(200).mean().iloc[-1])


if __name__ == "__main__":
    if not EODHD:
        send("EODHD_TOKEN 없음!"); exit(1)

    # 테마 매핑 로드
    if not os.path.exists("theme_ticker_map.csv"):
        send("theme_ticker_map.csv 없음! build_theme_map.py 먼저 실행하세요."); exit(1)

    theme_df = pd.read_csv("theme_ticker_map.csv", encoding="utf-8-sig")
    theme_df["ticker"] = theme_df["ticker"].astype(str).str.zfill(6)
    print(f"테마 매핑: {len(theme_df)}건 ({theme_df['theme'].nunique()}개 테마)")

    # 한글명 매핑
    kr_df = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
    kr_df["ticker"] = kr_df["ticker"].astype(str).str.zfill(6)
    name_map   = {r["ticker"]: r["name"]   for _, r in kr_df.iterrows()}
    market_map = {r["ticker"]: r["market"] for _, r in kr_df.iterrows()}
    cap_map    = {r["ticker"]: str(r.get("cap","") or "") for _, r in kr_df.iterrows()}

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

    # 분석 대상 티커 목록
    target_tickers = theme_df["ticker"].unique().tolist()
    print(f"분석 대상: {len(target_tickers)}개 종목")
    send(f"🔄 테마 로테이션 스캐너 시작\n{len(target_tickers)}개 종목 분석 중...")

    # 종목별 지표 계산
    stock_stats = {}
    for i, ticker in enumerate(target_tickers):
        if i % 100 == 0:
            print(f"[{i}/{len(target_tickers)}] 처리 중...")

        # EODHD에서 직접 수집 (캐시 미사용 - 최신 데이터 보장)
        exchange = "KQ" if market_map.get(ticker, "KOSPI") == "KOSDAQ" else "KO"
        df = get_ohlcv(ticker, exchange, start=start_date, end=end_date)
        time.sleep(0.05)

        if df is None or len(df) < 60: continue
        if float(df["Close"].iloc[-1]) < 1000: continue

        close  = df["Close"]
        volume = df["Volume"]

        rs_4w  = calc_rs_simple(close, idx_close, 20)
        rs_12w = calc_rs_simple(close, idx_close, 60)
        vr     = calc_volume_ratio(volume)
        pct52  = pct_from_52w_high(close)
        above200 = is_above_ma200(close)

        if rs_4w is None or rs_12w is None: continue

        # RS 이상값 필터 (±500% 초과는 데이터 오류로 제외)
        if abs(rs_4w) > 500 or abs(rs_12w) > 1000: continue

        stock_stats[ticker] = {
            "ticker":   ticker,
            "name":     name_map.get(ticker, ticker),
            "market":   market_map.get(ticker, "KOSPI"),
            "cap":      cap_map.get(ticker, ""),
            "cur":      round(float(close.iloc[-1]), 0),
            "rs_4w":    rs_4w,
            "rs_12w":   rs_12w,
            "vr":       vr,
            "pct52":    pct52,
            "above200": above200,
        }

    print(f"지표 계산 완료: {len(stock_stats)}개")

    # ── 테마별 RS 집계 ─────────────────────────────
    theme_stats = []
    for theme_name, grp in theme_df.groupby("theme"):
        tickers = grp["ticker"].tolist()
        stats   = [stock_stats[t] for t in tickers if t in stock_stats]
        if len(stats) < MIN_STOCKS: continue

        rs_4w_vals  = [s["rs_4w"]  for s in stats]
        rs_12w_vals = [s["rs_12w"] for s in stats]

        theme_stats.append({
            "theme":      theme_name,
            "n_stocks":   len(stats),
            "rs_4w_avg":  round(np.mean(rs_4w_vals), 2),
            "rs_12w_avg": round(np.mean(rs_12w_vals), 2),
            "rs_4w_pos":  round(sum(1 for v in rs_4w_vals if v > 0) / len(rs_4w_vals) * 100, 1),
        })

    theme_df_rank = pd.DataFrame(theme_stats).sort_values("rs_4w_avg", ascending=False).reset_index(drop=True)
    n_theme = len(theme_df_rank)
    theme_df_rank["rank_4w"]  = theme_df_rank["rs_4w_avg"].rank(ascending=False)
    theme_df_rank["rank_12w"] = theme_df_rank["rs_12w_avg"].rank(ascending=False)

    # 주도테마: 4주 + 12주 둘 다 상위 20%
    leader_themes = theme_df_rank[
        (theme_df_rank["rank_4w"]  <= n_theme * 0.20) &
        (theme_df_rank["rank_12w"] <= n_theme * 0.20)
    ]["theme"].tolist()

    # 부상테마: 4주 상위 20% + 12주 중하위 50% (최근 올라오는 중)
    rising_themes = theme_df_rank[
        (theme_df_rank["rank_4w"]  <= n_theme * 0.20) &
        (theme_df_rank["rank_12w"] >  n_theme * 0.50) &
        (~theme_df_rank["theme"].isin(leader_themes))
    ]["theme"].tolist()

    print(f"\n주도테마 ({len(leader_themes)}개): {leader_themes[:5]}")
    print(f"부상테마 ({len(rising_themes)}개): {rising_themes[:5]}")

    # ── 후보 종목 필터 ─────────────────────────────
    target_themes = leader_themes + rising_themes
    candidates = []

    for theme_name in target_themes:
        grp     = theme_df[theme_df["theme"] == theme_name]
        tickers = grp["ticker"].tolist()
        stats   = [stock_stats[t] for t in tickers if t in stock_stats]

        # 200MA 위 + 52주 고점 -15% 이내 + 거래량 눌림
        filtered = [s for s in stats
                    if s["above200"]
                    and s["pct52"] is not None and s["pct52"] >= -15
                    and s["vr"] is not None and s["vr"] <= 0.8]

        # 테마 내 RS 4주 상위 50%
        if not filtered: continue
        median_rs = np.median([s["rs_4w"] for s in filtered])
        top_half  = [s for s in filtered if s["rs_4w"] >= median_rs]

        for s in sorted(top_half, key=lambda x: x["rs_4w"], reverse=True):
            candidates.append({**s, "theme": theme_name,
                                "theme_type": "⭐ 주도" if theme_name in leader_themes else "🚀 부상"})

    print(f"후보 종목: {len(candidates)}개")

    # CSV 저장
    theme_df_rank.to_csv("theme_rank_kr.csv", index=False, encoding="utf-8-sig")
    if candidates:
        pd.DataFrame(candidates).to_csv("theme_rotation_kr.csv", index=False, encoding="utf-8-sig")

    # ── 텔레그램 메시지 ────────────────────────────
    tv_url = lambda t: f"https://www.tradingview.com/chart/?symbol=KRX:{t}"

    # 테마 랭킹
    msg  = f"🔄 <b>국장 테마 로테이션</b>\n"
    msg += f"분석: {len(stock_stats)}개 종목 | {datetime.today().strftime('%Y-%m-%d')}\n"
    msg += "─" * 22 + "\n\n"
    msg += f"📈 <b>테마 RS 랭킹 Top10 (4주)</b>\n"

    for i, row in theme_df_rank.head(10).iterrows():
        emoji = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"{i+1}."
        tag   = " ⭐" if row["theme"] in leader_themes else " 🚀" if row["theme"] in rising_themes else ""
        msg  += (f"{emoji} {row['theme']}{tag}\n"
                 f"   4주:{row['rs_4w_avg']:+.1f}% | 12주:{row['rs_12w_avg']:+.1f}%"
                 f" | n={int(row['n_stocks'])}\n")

    send(msg)

    if not candidates:
        send("⚠️ 후보 종목 없음\n(200MA↑ + 52주고점-15%내 + 거래량눌림)")
    else:
        # 테마별 종목 메시지
        cur_theme = None
        stock_msg = f"🎯 <b>후보 종목 {len(candidates)}개</b>\n"
        stock_msg += "(200MA↑ + 52주고점-15%내 + 거래량눌림)\n"
        stock_msg += "─" * 22 + "\n\n"

        for r in candidates:
            if r["theme"] != cur_theme:
                if cur_theme is not None and len(stock_msg) > 100:
                    send(stock_msg)
                    stock_msg = ""
                stock_msg += f"<b>[{r['theme_type']}] {r['theme']}</b>\n\n"
                cur_theme = r["theme"]

            mkt = "🔵" if r["market"] == "KOSPI" else "🟢"
            vr  = f"{r['vr']:.2f}x" if r["vr"] else "-"
            cap_emoji = {"MegaCap":"💎초대형","LargeCap":"🔷대형","MidCap":"🔹중형","SmallCap":"▪️소형"}.get(r.get("cap",""), "")
            blk = (
                f"{mkt} <b>{r['name']}</b>({r['ticker']}) {cap_emoji}\n"
                f"  현재가: {r['cur']:,.0f}원\n"
                f"  52주고점대비: {r['pct52']:+.1f}%\n"
                f"  RS 4주: {r['rs_4w']:+.1f}% | 12주: {r['rs_12w']:+.1f}%\n"
                f"  거래량: {vr} 📊 {tv_url(r['ticker'])}\n\n"
            )
            if len(stock_msg) + len(blk) > 3800:
                send(stock_msg)
                stock_msg = blk
            else:
                stock_msg += blk

        if stock_msg:
            send(stock_msg)

    send(f"✅ 테마 로테이션 완료 | 후보 {len(candidates)}개")
