"""
국장 백테스트 (EODHD 버전)
- raw_data/kr/ 에서 OHLCV 로드
- 미너비니 트렌드 7조건 + 컵핸들
- 조건 완화: 트렌드 + 컵핸들만
- r5/r10/r20/r60 수익률 저장
- 종목명 표시 (tickers_kr.csv)
"""
import os, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TOK           = os.environ.get("TELEGRAM_TOKEN", "")
CID           = os.environ.get("TELEGRAM_CHAT_ID", "")
LOOKBACK_DAYS = 365 * 20
MAX_HOLD      = 60
KR_DIR        = "raw_data/kr"


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text}, timeout=10)
        except: pass


def check_trend(df):
    if len(df) < 200: return False
    c    = df["Close"]
    m50  = c.rolling(50).mean()
    m150 = c.rolling(150).mean()
    m200 = c.rolling(200).mean()
    cur  = float(c.iloc[-1])
    a    = float(m50.iloc[-1])
    b    = float(m150.iloc[-1])
    m20v = m200.dropna()
    if len(m20v) < 1: return False
    d  = float(m20v.iloc[-1])
    d1 = float(m20v.iloc[-21]) if len(m20v) >= 21 else d
    if any(pd.isna([a, b, d])): return False
    lk = c.iloc[-252:] if len(c) >= 252 else c
    return all([
        cur > b and cur > d, b > d, d > d1,
        a > b and a > d, cur > a,
        cur >= lk.min() * 1.25, cur >= lk.max() * 0.70
    ])


def detect(df):
    cl  = df["Close"].values.astype(float)
    vl  = df["Volume"].values.astype(float)
    n   = len(cl)
    idx = df.index
    if n < 60: return False, {}

    c = cl[-min(200, n):]
    v = vl[-min(200, n):]
    w = len(c)

    candidates = []
    for li in range(w // 2):
        if c[li] != max(c[max(0, li-5):li+6]): continue
        lh  = c[li]
        cup = c[li:]
        if len(cup) < 20: continue
        bi  = li + int(np.argmin(cup))
        bot = c[bi]
        cd  = (lh - bot) / lh
        if not (0.15 <= cd <= 0.50): continue
        if (bi - li) < 35: continue
        rc = c[bi:]
        if len(rc) < 10: continue
        ri  = bi + int(np.argmax(rc))
        rh  = c[ri]
        if rh < lh * 0.90 or rh > lh * 1.15: continue
        hnd = c[ri:]; hl = len(hnd)
        if not (5 <= hl <= 20): continue
        hlow = float(np.min(hnd))
        hd   = (rh - hlow) / rh
        if not (0.05 <= hd <= 0.15): continue
        if (hlow - bot) / (lh - bot) < 0.60: continue
        cur = cl[-1]
        if not (rh * 0.97 <= cur <= rh * 1.05): continue
        candidates.append((ri - li, li, bi, ri, lh, bot, rh, cd, hd, hl))

    if not candidates: return False, {}
    candidates.sort(reverse=True)
    _, li, bi, ri, lh, bot, rh, cd, hd, hl = candidates[0]

    cur = cl[-1]
    vr  = float(np.mean(v[-5:])) / float(np.mean(v[-40:-5])) if len(v) >= 40 else 1.0
    try:
        start_pos = n - len(c)
        cup_start = idx[start_pos + li].strftime("%Y-%m-%d")
        cup_end   = idx[start_pos + ri].strftime("%Y-%m-%d")
    except:
        cup_start = ""; cup_end = ""

    return True, {
        "cd": round(cd*100, 1), "hd": round(hd*100, 1),
        "cdays": ri - li, "hdays": hl,
        "pivot": round(float(rh), 0), "cur": round(float(cur), 0),
        "vr": round(vr, 2), "vs": vr >= 1.40,
        "cup_start": cup_start, "cup_end": cup_end
    }


def calc_rs(df, idx_df, sig_idx):
    def p(d, n): return float(d["Close"].iloc[-1] / d["Close"].iloc[-n] - 1) if len(d) >= n else 0.0
    sl = df.iloc[:sig_idx + 1]
    sm = idx_df.reindex(sl.index).ffill().dropna()
    if len(sm) < 63: return 0.0
    sl2 = sl.reindex(sm.index)
    w_  = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
    s   = sum(w_[i] * p(sl2, p_[i]) for i in range(4))
    m   = sum(w_[i] * p(sm,  p_[i]) for i in range(4))
    return round((s - m) * 100, 1)


def calc_score(rs, vr, cd, hd):
    s_rs = 100 if rs>=25 else 80 if rs>=15 else 60 if rs>=10 else 40 if rs>=5 else 20
    s_vr = 100 if vr>=3.0 else 85 if vr>=2.5 else 70 if vr>=2.0 else 55 if vr>=1.7 else 40
    s_cd = 100 if 20<=cd<=35 else 75 if (15<=cd<20 or 35<cd<=40) else 50 if 40<cd<=50 else 30
    s_hd = 100 if 5<=hd<=10 else 75 if 10<hd<=12 else 50 if hd>12 else 60
    return round(s_rs*0.40 + s_vr*0.35 + s_cd*0.15 + s_hd*0.10)


if __name__ == "__main__":
    if not os.path.exists(KR_DIR):
        send("raw_data/kr/ 없음! collect_ohlcv.py 먼저 실행하세요.")
        exit(1)

    # 티커 메타 로드 (종목명)
    meta_map = {}
    if os.path.exists("tickers_kr.csv"):
        mdf = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
        meta_map = {str(r["ticker"]).zfill(6): r.to_dict() for _, r in mdf.iterrows()}
        print(f"티커 메타: {len(meta_map)}개")

    # 지수 로드
    kospi_idx  = None
    kosdaq_idx = None

    from eodhd_utils import get_ohlcv
    kospi_path  = os.path.join(KR_DIR, "069500.csv")
    kosdaq_path = os.path.join(KR_DIR, "229200.csv")

    def load_or_fetch_idx(path, symbol, exchange):
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            print(f"{symbol}: 캐시 로드 {len(df)}일치")
            return df
        print(f"{symbol}: EODHD에서 수집 중...")
        df = get_ohlcv(symbol, exchange, start="2000-01-01")
        if df is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path)
            print(f"{symbol}: 수집 완료 {len(df)}일치")
        else:
            print(f"{symbol}: 수집 실패 → rs=0 처리")
        return df

    kospi_idx  = load_or_fetch_idx(kospi_path,  "069500", "KO")
    kosdaq_idx = load_or_fetch_idx(kosdaq_path, "229200", "KO")

    # 백테스트 기간
    cutoff       = datetime.today() - timedelta(days=LOOKBACK_DAYS)
    signal_dates = set(pd.bdate_range(cutoff, datetime.today()).map(pd.Timestamp))

    send(f"🇰🇷 국장 백테스트 시작 (EODHD)\n기간: {LOOKBACK_DAYS//365}년\n조건: 트렌드+컵핸들")

    files       = [f for f in os.listdir(KR_DIR) if f.endswith(".csv")]
    all_signals = []

    for i, fname in enumerate(files):
        ticker = fname.replace(".csv", "").zfill(6)
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] 시그널:{len(all_signals)}건")

        try:
            df = pd.read_csv(
                os.path.join(KR_DIR, fname),
                index_col="date", parse_dates=True
            )
            df = df[["Open","High","Low","Close","Adj_Close","Volume"]].astype(float).dropna()
        except:
            continue

        if len(df) < 250: continue

        # 저가주 필터 (1000원 미만)
        if float(df["Close"].iloc[-1]) < 1000: continue

        # 종목 정보
        meta   = meta_map.get(ticker, {})
        name   = str(meta.get("name", ticker))
        market = str(meta.get("market", "KOSPI"))
        sector = str(meta.get("sector", "기타") or "기타")
        cap    = str(meta.get("cap", "") or "")

        # 지수 선택
        idx_df = kospi_idx if market == "KOSPI" else kosdaq_idx

        idx = df.index.tolist()

        for j, sig_ts in enumerate(idx):
            if sig_ts not in signal_dates: continue
            sl = df.iloc[:j + 1]
            if len(sl) < 250: continue

            if not check_trend(sl): continue
            ok, pat = detect(sl)
            if not ok: continue

            # RS 계산
            rs = calc_rs(sl, idx_df, j) if idx_df is not None else 0.0

            entry = float(df["Close"].iloc[j])
            score = calc_score(rs, pat["vr"], pat["cd"], pat["hd"])

            # 거래대금 (20일 평균, 억원)
            trdval_20 = 0.0

            # 수익률
            daily_r = {}
            for hold in [1, 3, 5, 10, 15, 20, 30, 40, 50, 60]:
                fi = j + hold
                if fi < len(idx):
                    daily_r[f"r{hold}"] = round(
                        (float(df["Close"].iloc[fi]) / entry - 1) * 100, 2)
                else:
                    daily_r[f"r{hold}"] = None

            all_signals.append({
                "date":         sig_ts.strftime("%Y-%m-%d"),
                "ticker":       ticker,
                "name":         name,
                "market":       market,
                "sector":       sector,
                "cap":          cap,
                "entry":        entry,
                "pivot":        pat["pivot"],
                "cup_depth":    pat["cd"],
                "handle_depth": pat["hd"],
                "cup_days":     pat["cdays"],
                "handle_days":  pat["hdays"],
                "cup_start":    pat.get("cup_start", ""),
                "cup_end":      pat.get("cup_end", ""),
                "vol_ratio":    pat["vr"],
                "vs":           pat["vs"],
                "rs":           rs,
                "score":        score,
                **daily_r,
            })

    print(f"백테스트 완료: {len(all_signals)}건")

    df_out = pd.DataFrame(all_signals)
    df_out.to_csv("backtest_kr_raw.csv", index=False, encoding="utf-8-sig")
    send(f"🇰🇷 국장 백테스트 완료\n{len(all_signals)}건\n저장: backtest_kr_raw.csv")

    # 요약
    if all_signals:
        for col, label in [("r5","5일"),("r10","10일"),("r20","20일")]:
            vals = df_out[col].dropna()
            if len(vals) == 0: continue
            win = (vals > 0).sum()
            print(f"[{label}] n={len(vals)} 평균:{vals.mean():+.1f}% 승률:{win/len(vals)*100:.1f}%")

        # 샘플 출력 (종목명 확인용)
        print(f"\n샘플 시그널 (상위 5개):")
        for _, r in df_out.head(5).iterrows():
            print(f"  {r['date']} {r['ticker']} {r['name']} "
                  f"({r['market']}) {r['sector']} "
                  f"pivot:{r['pivot']:,.0f}원 rs:{r['rs']:+.1f}")
