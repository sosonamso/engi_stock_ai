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
from io import StringIO
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from eodhd_utils import get_ohlcv, EODHD

warnings.filterwarnings("ignore")

TOK              = os.environ.get("TELEGRAM_TOKEN", "")
CID              = os.environ.get("TELEGRAM_CHAT_ID", "")
KR_DIR           = "raw_data/kr"
HISTORY_DAYS     = 280
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


def get_popular_stocks():
    """네이버 인기검색 종목 수집"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            "https://finance.naver.com/sise/lastsearch2.naver",
            headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        popular = {}
        rank = 1
        for a in soup.select("a[href*='code=']"):
            href = a["href"]
            if "code=" in href:
                code = href.split("code=")[-1].split("&")[0].strip()
                if len(code) == 6 and code.isdigit():
                    if code not in popular:
                        popular[code] = rank
                        rank += 1
        print(f"인기종목: {len(popular)}개")
        return popular
    except Exception as e:
        print(f"인기종목 수집 실패: {e}")
        return {}


def calc_rs(stock_close, idx_close, n=10):
    """직전 n거래일 RS"""
    try:
        si = idx_close.reindex(stock_close.index).ffill()
        if len(si.dropna()) < n+1 or len(stock_close) < n+1: return None
        sr = float(stock_close.iloc[-1] / stock_close.iloc[-n] - 1) * 100
        mr = float(si.iloc[-1]          / si.iloc[-n]          - 1) * 100
        return round(sr - mr, 2)
    except: return None


def calc_rs_prev(stock_close, idx_close, start=10, end=40):
    """이전 구간 RS (start거래일 전 ~ end거래일 전)"""
    try:
        si = idx_close.reindex(stock_close.index).ffill()
        if len(stock_close) < end+1 or len(si.dropna()) < end+1: return None
        sr = float(stock_close.iloc[-start] / stock_close.iloc[-end] - 1) * 100
        mr = float(si.iloc[-start]          / si.iloc[-end]          - 1) * 100
        return round(sr - mr, 2)
    except: return None


def calc_volume_ratio(volume, recent=10, prev_start=10, prev_end=40):
    """최근 2주 vs 이전 6주 거래량 비율"""
    try:
        if len(volume) < prev_end: return None
        recent_avg = volume.iloc[-recent:].mean()
        prev_avg   = volume.iloc[-prev_end:-prev_start].mean()
        if prev_avg == 0: return None
        return round(float(recent_avg / prev_avg), 2)
    except: return None


def clean_ohlcv(df):
    """이상값 제거 (close/high 전후날 대비 40% 초과 제거)"""
    close = df["Close"].astype(float)
    high  = df["High"].astype(float) if "High" in df.columns else close
    mask  = pd.Series([True] * len(close), index=close.index)
    for i in range(1, len(close) - 1):
        for s in [close, high]:
            prev = s.iloc[i-1]; curr = s.iloc[i]; nxt = s.iloc[i+1]
            if prev > 0 and nxt > 0:
                if abs(curr / prev - 1) > 0.4 and abs(curr / nxt - 1) > 0.4:
                    mask.iloc[i] = False
                    break
    return df[mask]


def clean_close(close):
    """이상값 제거 (전날/다음날 대비 ±40% 초과 제거)"""
    s = close.copy().astype(float)
    mask = pd.Series([True] * len(s), index=s.index)
    for i in range(1, len(s) - 1):
        prev = s.iloc[i-1]; curr = s.iloc[i]; nxt = s.iloc[i+1]
        if prev > 0 and nxt > 0:
            if abs(curr / prev - 1) > 0.4 and abs(curr / nxt - 1) > 0.4:
                mask.iloc[i] = False
    return s[mask]


def pct_from_52w_high(close):
    try:
        w = clean_close(close.iloc[-252:] if len(close) >= 252 else close)
        if len(w) == 0: return None
        return round((float(close.iloc[-1]) / float(w.max()) - 1) * 100, 1)
    except: return None


def date_of_52w_high(close):
    try:
        w = clean_close(close.iloc[-252:] if len(close) >= 252 else close)
        if len(w) == 0: return ''
        return w.idxmax().strftime('%y.%m.%d')
    except: return ''


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
    popular_map = get_popular_stocks()
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

        df    = clean_ohlcv(df)
        if len(df) < 30: continue

        close  = df["Close"]
        volume = df["Volume"]

        rs_2w  = calc_rs(close, idx_close, RS_PERIOD)
        rs_6w  = calc_rs_prev(close, idx_close, start=RS_PERIOD, end=RS_PERIOD+30)
        accel  = round(rs_2w - rs_6w, 2) if rs_2w is not None and rs_6w is not None else None
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
            "rs_6w":     rs_6w,
            "accel":     accel,
            "ret_2w":    ret_2w,
            "vr":        vr,
            "pct52":       pct52,
            "high52_date": date_of_52w_high(close),
            "above200":    above200,
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

        accel_vals = [stock_stats[t]["accel"] for t in tickers
                      if stock_stats[t]["accel"] is not None]
        rs_6w_vals  = [stock_stats[t]["rs_6w"]  for t in tickers
                      if stock_stats[t]["rs_6w"]  is not None]

        theme_stats.append({
            "theme":       theme_name,
            "n_large":     len(tickers),
            "rs_2w_avg":   round(np.mean(rs_vals), 2),
            "rs_6w_avg":   round(np.mean(rs_6w_vals), 2) if rs_6w_vals else 0,
            "accel_avg":   round(np.mean(accel_vals), 2) if accel_vals else 0,
            "up_ratio":    round(up_ratio * 100, 1),
            "up_count":    up_count,
            "total":       len(ret_vals),
        })

    # 가속도(2주RS - 6주RS) 기준 정렬
    theme_rank = pd.DataFrame(theme_stats).sort_values("accel_avg", ascending=False).reset_index(drop=True)
    n_theme    = len(theme_rank)
    top_n      = max(1, int(n_theme * LEADER_RS_PCT))

    # 주도테마: RS 상위 20% + 상승비율 70% 이상
    leader_mask = (
        (theme_rank.index < top_n) &
        (theme_rank["up_ratio"] >= LEADER_UP_RATIO * 100)
    )
    leader_themes = theme_rank[leader_mask]["theme"].tolist()
    print(f"주도테마: {len(leader_themes)}개")

    # ── 종목 선별 (RS Top 5, 조건 없음) ─────────────
    candidates = []
    for theme_name in leader_themes:
        grp     = theme_df[theme_df["theme"] == theme_name]
        tickers = [t for t in grp["ticker"].tolist() if t in stock_stats]
        stats   = [stock_stats[t] for t in tickers]

        # RS 높은 순 Top 5
        top5 = sorted(stats, key=lambda x: x["rs_2w"], reverse=True)[:MAX_STOCKS]
        for s in top5:
            candidates.append({
                **s,
                "theme":    theme_name,
                "pop_rank": popular_map.get(s["ticker"])
            })

    print(f"후보 종목: {len(candidates)}개")

    # ── 신규 진입 알림 ────────────────────────────
    NEW_ENTRY_RANK = 5  # 상위 N위 이내

    prev_rank_map = {}
    if os.path.exists("theme_rank_kr.csv"):
        try:
            prev_df = pd.read_csv("theme_rank_kr.csv", encoding="utf-8-sig")
            for i, row in prev_df.iterrows():
                prev_rank_map[row["theme"]] = i + 1
        except: pass

    new_entries = []
    for i, row in theme_rank.iterrows():
        today_rank = i + 1
        prev_rank  = prev_rank_map.get(row["theme"], 9999)
        if today_rank <= NEW_ENTRY_RANK and prev_rank > NEW_ENTRY_RANK:
            new_entries.append({
                "theme":      row["theme"],
                "today_rank": today_rank,
                "prev_rank":  prev_rank,
                "accel":      row.get("accel_avg", 0),
                "rs_2w":      row["rs_2w_avg"],
                "up_ratio":   row["up_ratio"],
                "up_count":   row["up_count"],
                "total":      row["total"],
            })

    if new_entries:
        alert_msg = "🚨 <b>테마 신규 진입 알림</b>\n"
        alert_msg += f"상위 {NEW_ENTRY_RANK}위 신규 진입 {len(new_entries)}개\n"
        alert_msg += "─" * 22 + "\n\n"
        for e in new_entries:
            # 해당 테마 Top3 종목
            grp     = theme_df[theme_df["theme"] == e["theme"]]
            tickers = [t for t in grp["ticker"].tolist() if t in stock_stats]
            top3    = sorted(tickers, key=lambda t: stock_stats[t]["rs_2w"], reverse=True)[:3]
            stocks_str = " / ".join([
                f"{stock_stats[t]['name']}({t})" for t in top3
            ])
            alert_msg += (
                f"⭐ <b>{e['theme']}</b>\n"
                f"  순위: {e['prev_rank']}위 → {e['today_rank']}위\n"
                f"  가속도: {e['accel']:+.1f}% | 2주RS: {e['rs_2w']:+.1f}%\n"
                f"  상승 {e['up_count']}/{e['total']}({e['up_ratio']:.0f}%)\n"
                f"  종목: {stocks_str}\n\n"
            )
        send(alert_msg)
        print(f"신규 진입 알림: {len(new_entries)}개")
    else:
        print("신규 진입 테마 없음")

    # CSV 저장 (오늘 결과를 prev로 백업 후 저장)
    if os.path.exists("theme_rank_kr.csv"):
        import shutil
        shutil.copy("theme_rank_kr.csv", "theme_rank_kr_prev.csv")
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
                  f"   2주:{row['rs_2w_avg']:+.1f}% 6주:{row.get('rs_6w_avg',0):+.1f}%"
                  f" 가속:{row.get('accel_avg',0):+.1f}%\n"
                  f"   상승{row['up_count']}/{row['total']}({row['up_ratio']:.0f}%)\n")

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
                              f"   2주:{tr['rs_2w_avg']:+.1f}% 6주:{tr.get('rs_6w_avg',0):+.1f}%"
                              f" 가속:{tr.get('accel_avg',0):+.1f}%\n\n")
                cur_theme = r["theme"]

            mkt   = "🔵코스피" if r["market"] == "KOSPI" else "🟢코스닥"
            cap_e = cap_emoji.get(r["cap_label"], "")
            vr    = f"{r['vr']:.2f}x" if r["vr"] else "-"
            accel_str = f" 가속:{r['accel']:+.1f}%" if r.get("accel") is not None else ""
            pop_rank  = popular_map.get(r["ticker"])
            pop_str   = f" 🔥인기{pop_rank}위" if pop_rank else ""
            blk   = (
                f"{cap_e} <b>{r['name']}</b>({r['ticker']}) {mkt}{pop_str}\n"
                f"  현재가: {r['cur']:,.0f}원\n"
                f"  52주고점: {r['pct52']:+.1f}% | 거래량: {vr}\n"
                f"  RS 2주: {r['rs_2w']:+.1f}%{accel_str}\n"
                f"  📊 {tv_url(r['ticker'])}\n\n"
            )
            if len(stock_msg) + len(blk) > 3800:
                send(stock_msg); stock_msg = blk
            else:
                stock_msg += blk

        if stock_msg: send(stock_msg)

    send(f"✅ 완료 | 주도테마 {len(leader_themes)}개 | 후보 {len(candidates)}개")
