"""
국장 스캐너 (EODHD + LightGBM)
- tickers_kr.csv 기반 전체 종목 스캔
- 미너비니 트렌드 7조건 + 컵핸들
- LightGBM 점수 계산 (시그널 종목만)
- 종목명 한글로 표시
"""
import os, time, pickle, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from eodhd_utils import get_ohlcv, EODHD

warnings.filterwarnings("ignore")

TOK          = os.environ.get("TELEGRAM_TOKEN", "")
CID          = os.environ.get("TELEGRAM_CHAT_ID", "")
SCAN_DAYS    = 7
HISTORY_DAYS = 600
WINDOW       = 150
SKIP         = 2
VOL_MA       = 20
MIN_PRICE    = 1000
KR_DIR       = "raw_data/kr"


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text}, timeout=10)
        except: pass


def send_file(filepath, caption=""):
    if TOK:
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id": CID, "caption": caption},
                    files={"document": f}, timeout=30)
        except: pass


def get_recent_dates(n=7):
    dates = []
    d = datetime.today()
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    return dates


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
    n   = len(cl); idx = df.index
    if n < 60: return False, {}

    c = cl[-min(500, n):]
    v = vl[-min(500, n):]
    w = len(c)

    candidates = []
    for li in range(w // 2):
        if c[li] != max(c[max(0, li-5):li+6]): continue
        lh  = c[li]; cup = c[li:]
        if len(cup) < 20: continue
        bi  = li + int(np.argmin(cup)); bot = c[bi]; cd = (lh - bot) / lh
        if not (0.15 <= cd <= 0.50): continue
        if (bi - li) < 35: continue
        rc = c[bi:]
        if len(rc) < 10: continue
        ri  = bi + int(np.argmax(rc)); rh = c[ri]
        if rh < lh * 0.90 or rh > lh * 1.15: continue
        hnd = c[ri:]; hl = len(hnd)
        if not (5 <= hl <= 20): continue
        hlow = float(np.min(hnd)); hd = (rh - hlow) / rh
        if not (0.05 <= hd <= 0.15): continue
        if (hlow - bot) / (lh - bot) < 0.60: continue
        cur = cl[-1]
        # 패턴 감지는 넓게 (피벗 70~110%)
        if not (rh * 0.70 <= cur <= rh * 1.10): continue
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


def calc_rs(df, idx_df):
    def p(d, n): return float(d["Close"].iloc[-1] / d["Close"].iloc[-n] - 1) if len(d) >= n else 0.0
    sm = idx_df.reindex(df.index).ffill().dropna()
    if len(sm) < 63: return 0.0
    sl = df.reindex(sm.index)
    w_ = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
    s  = sum(w_[i] * p(sl, p_[i]) for i in range(4))
    m  = sum(w_[i] * p(sm, p_[i]) for i in range(4))
    return round((s - m) * 100, 1)


def calc_score(rs, vr, cd, hd):
    s_rs = 100 if rs>=25 else 80 if rs>=15 else 60 if rs>=10 else 40 if rs>=5 else 20
    s_vr = 100 if vr>=3.0 else 85 if vr>=2.5 else 70 if vr>=2.0 else 55 if vr>=1.7 else 40
    s_cd = 100 if 20<=cd<=35 else 75 if (15<=cd<20 or 35<cd<=40) else 50 if 40<cd<=50 else 30
    s_hd = 100 if 5<=hd<=10 else 75 if 10<hd<=12 else 50 if hd>12 else 60
    return round(s_rs*0.40 + s_vr*0.35 + s_cd*0.15 + s_hd*0.10)


def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period]); avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 4)


def get_market_features(kospi_idx, sig_date):
    result = {"mkt_ret_20": 0.0, "mkt_ret_60": 0.0,
              "mkt_ret_120": 0.0, "mkt_above_ma120": 0}
    if kospi_idx is None: return result
    try:
        sl = kospi_idx.loc[:sig_date]["Close"]
        if len(sl) < 5: return result
        cur = float(sl.iloc[-1])
        if len(sl) >= 20:  result["mkt_ret_20"]  = round((cur/float(sl.iloc[-20])-1)*100, 2)
        if len(sl) >= 60:  result["mkt_ret_60"]  = round((cur/float(sl.iloc[-60])-1)*100, 2)
        if len(sl) >= 120:
            result["mkt_ret_120"]     = round((cur/float(sl.iloc[-120])-1)*100, 2)
            result["mkt_above_ma120"] = 1 if cur > float(sl.iloc[-120:].mean()) else 0
    except: pass
    return result


def calc_lgbm_features(df, kospi_idx, ticker_info, pat):
    if len(df) < WINDOW + SKIP: return None
    d_idx     = len(df) - 1
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < VOL_MA: return None

    w      = df.iloc[start_idx:end_idx]
    close  = w["Close"].values
    high   = w["High"].values  if "High"  in w.columns else close
    low    = w["Low"].values   if "Low"   in w.columns else close
    volume = w["Volume"].values

    if close[-1] < MIN_PRICE: return None

    c_min, c_max = close.min(), close.max()
    if c_max == c_min: return None
    close_norm = (close - c_min) / (c_max - c_min)

    ret = np.zeros(WINDOW); ret[1:] = (close[1:] / close[:-1] - 1) * 100
    vol_ratio = np.zeros(WINDOW)
    for k in range(WINDOW):
        abs_idx  = start_idx + k
        past_vol = df["Volume"].iloc[max(0, abs_idx - VOL_MA): abs_idx]
        ma = past_vol.mean()
        vol_ratio[k] = volume[k] / ma if ma > 0 else 1.0

    feat = {}
    for k in range(WINDOW):
        feat[f"ret_{k+1}"]        = round(ret[k], 4)
        feat[f"close_norm_{k+1}"] = round(float(close_norm[k]), 4)
        feat[f"vol_ratio_{k+1}"]  = round(float(vol_ratio[k]), 4)

    # RS
    tc = df["Close"].values
    rs_at = rs_20 = rs_50 = rs_150 = 0.0
    if kospi_idx is not None and len(tc) >= 253:
        sa = kospi_idx.reindex(df.index).ffill()
        if not sa.isnull().all().any():
            sc = sa["Close"].values
            def pr(arr, n): return float(arr[-1]/arr[-n]-1) if len(arr)>=n else 0.0
            w_ = [0.4,0.2,0.2,0.2]; p_ = [63,126,189,252]
            t_rs = sum(w_[i]*pr(tc,p_[i]) for i in range(4))
            s_rs = sum(w_[i]*pr(sc,p_[i]) for i in range(4))
            rs_at  = round((t_rs-s_rs)*100, 4)
            rs_20  = round((pr(tc,20) -pr(sc,20)) *100, 4)
            rs_50  = round((pr(tc,50) -pr(sc,50)) *100, 4)
            rs_150 = round((pr(tc,150)-pr(sc,150))*100, 4)

    feat["rs_at_d2"] = rs_at; feat["rs_20"] = rs_20
    feat["rs_50"]    = rs_50; feat["rs_150"] = rs_150
    feat["rs_trend"] = round(rs_20 - rs_50, 4)
    feat["rsi_14"]   = calc_rsi(close, 14)

    ma20 = np.mean(close[-20:]); std20 = np.std(close[-20:])
    upper = ma20 + 2*std20; lower = ma20 - 2*std20
    feat["bb_pos"] = round(float(np.clip(
        (close[-1]-lower)/(upper-lower) if upper!=lower else 0.5, 0, 1)), 4)

    yc = df["Close"].iloc[max(0, d_idx-252):d_idx].values
    feat["pos_52w_high"] = round(float(close[-1]/yc.max()), 4) if len(yc)>0 else 1.0
    feat["pos_52w_low"]  = round(float(close[-1]/yc.min()), 4) if len(yc)>0 else 1.0

    tr_list = []
    h_arr = high[-15:]; l_arr = low[-15:]; c_arr = close[-15:]
    for k in range(1, len(h_arr)):
        tr = max(h_arr[k]-l_arr[k], abs(h_arr[k]-c_arr[k-1]), abs(l_arr[k]-c_arr[k-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    feat["atr_ratio"]  = round(float(atr/close[-1]) if close[-1]>0 else 0, 4)
    feat["ma20_pos"]   = round(float(close[-1]/np.mean(close[-20:])), 4)
    feat["ma50_pos"]   = round(float(close[-1]/np.mean(close[-50:])) if len(close)>=50 else 1.0, 4)

    # 거래대금
    trdval = close * volume
    trdval_20_avg = np.mean(trdval[-20:])
    feat["trdval_ratio"]  = round(float(trdval[-1]/trdval_20_avg) if trdval_20_avg>0 else 1.0, 4)
    feat["trdval_20_log"] = round(float(np.log1p(trdval_20_avg/1e8)), 4)

    # 섹터 OHE
    KR_SECTORS = ["음식료품","섬유의복","종이목재","화학","의약품",
                  "비금속광물","철강금속","기계","전기전자","의료정밀",
                  "운수장비","유통업","전기가스업","건설업","운수창고업",
                  "통신업","금융업","증권","보험","서비스업","기타"]
    sector = str(ticker_info.get("sector","기타") or "기타")
    if sector not in KR_SECTORS: sector = "기타"
    for s in KR_SECTORS:
        feat[f"sec_{s}"] = 1 if sector==s else 0

    feat["mkt_KOSPI"]  = 1 if str(ticker_info.get("market","KOSPI"))=="KOSPI" else 0
    feat["mkt_KOSDAQ"] = 1 if str(ticker_info.get("market","KOSPI"))=="KOSDAQ" else 0

    # 패턴 피처
    feat["cup_depth"]    = float(pat.get("cd", 0))
    feat["handle_depth"] = float(pat.get("hd", 0))
    feat["vol_ratio"]    = float(pat.get("vr", 0))
    feat["cup_days"]     = float(pat.get("cdays", 0))
    feat["handle_days"]  = float(pat.get("hdays", 0))
    feat["vs"]           = 1 if pat.get("vs") else 0
    feat["rs_signal"]    = float(ticker_info.get("rs", 0) or 0)
    feat["score"]        = float(ticker_info.get("score", 0) or 0)

    return feat


def predict_lgbm(feat, lgbm_model, feat_cols):
    try:
        import lightgbm as lgb
        X = np.array([[feat.get(c, 0) for c in feat_cols]], dtype=np.float32)
        return round(float(lgbm_model.predict(X)[0]), 4)
    except: return None


if __name__ == "__main__":
    if not EODHD:
        send("EODHD_TOKEN 없음!"); exit(1)

    # LightGBM 모델 로드
    lgbm_model = None; feat_cols = None
    try:
        import lightgbm as lgb
        lgbm_model = lgb.Booster(model_file="model_lgbm_kr.txt")
        with open("feat_cols_kr.pkl", "rb") as f:
            feat_cols = pickle.load(f)
        print(f"LightGBM 모델 로드 완료 (피처:{len(feat_cols)}개)")
    except Exception as e:
        print(f"모델 없음: {e}")

    # 티커 로드
    if not os.path.exists("tickers_kr.csv"):
        send("tickers_kr.csv 없음!"); exit(1)
    meta_df  = pd.read_csv("tickers_kr.csv", encoding="utf-8-sig")
    meta_map = {str(r["ticker"]).zfill(6): r.to_dict() for _, r in meta_df.iterrows()}

    # 날짜 설정
    end_date    = datetime.today().strftime("%Y-%m-%d")
    start_date  = (datetime.today() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    sig_dates   = get_recent_dates(SCAN_DAYS)
    data_cutoff = pd.Timestamp(sig_dates[0]) - timedelta(days=7)

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

    # 시장 상태
    market_ok = True; market_str = "데이터 부족"
    if kospi_idx is not None and len(kospi_idx) >= 200:
        cur   = float(kospi_idx["Close"].iloc[-1])
        ma200 = float(kospi_idx["Close"].rolling(200).mean().iloc[-1])
        if cur > ma200:
            market_ok = True; market_str = "상승장(KOSPI>200MA)"
        else:
            market_ok = False; market_str = "하락장(KOSPI<200MA)"

    send(f"🇰🇷 국장 스캐너 시작 (EODHD)\n"
         f"최근 {SCAN_DAYS}거래일 | {market_str}\n"
         f"{len(meta_map)}개 종목 수집 중...")

    # 데이터 수집
    valid_data  = {}
    ticker_list = list(meta_map.keys())

    for i, ticker in enumerate(ticker_list):
        if i % 300 == 0:
            print(f"[{i}/{len(ticker_list)}] 수신:{len(valid_data)}")
        info     = meta_map.get(ticker, {})
        exchange = str(info.get("exchange", "KO"))
        df = get_ohlcv(ticker, exchange, start=start_date, end=end_date)
        if df is None: continue
        if df.index[-1] < data_cutoff: continue
        valid_data[ticker] = df
        time.sleep(0.05)

    send(f"수집 완료: {len(valid_data)}/{len(ticker_list)}개\n패턴 분석 시작...")
    if not market_ok:
        send("⚠️ KOSPI 200MA 하방 — 시그널 신뢰도 낮음!")

    # 패턴 분석
    signals       = []
    all_scores    = []
    trend_pass    = 0
    pattern_pass  = 0
    vol_pass      = 0
    rs_pass       = 0

    for ticker in ticker_list:
        df = valid_data.get(ticker)
        if df is None: continue

        info   = meta_map.get(ticker, {})
        name   = str(info.get("name", ticker))
        market = str(info.get("market", "KOSPI"))

        for sig_str in sig_dates:
            sig_ts = pd.Timestamp(sig_str)
            if sig_ts not in df.index: continue

            pos = df.index.tolist().index(sig_ts)
            sl  = df.iloc[:pos + 1]
            cur = float(sl["Close"].iloc[-1])
            rs  = calc_rs(sl, kospi_idx) if kospi_idx is not None else 0.0

            if not check_trend(sl):
                all_scores.append({
                    "ticker": ticker, "name": name, "market": market,
                    "sector": str(info.get("sector","기타")),
                    "cur": round(cur, 0), "rs": rs,
                    "trend_ok": False, "pattern_ok": False,
                    "signal": False, "score": 0, "lgbm_score": None,
                    "reason": "트렌드 미통과",
                })
                break

            trend_pass += 1
            ok, pat = detect(sl)

            if ok:
                pattern_pass += 1
                if pat["vs"]: vol_pass += 1
                rs_ok_check = rs > 0 if "상승" in market_str else rs > -20
                if rs_ok_check: rs_pass += 1

            if not ok:
                score = calc_score(rs, 1.0, 0, 0)
                all_scores.append({
                    "ticker": ticker, "name": name, "market": market,
                    "sector": str(info.get("sector","기타")),
                    "cur": round(cur, 0), "rs": rs,
                    "trend_ok": True, "pattern_ok": False,
                    "signal": False, "score": score, "lgbm_score": None,
                    "reason": "패턴 미감지",
                })
                break

            score  = calc_score(rs, pat["vr"], pat["cd"], pat["hd"])
            pivot  = pat["pivot"]
            pct    = round((cur - pivot) / pivot * 100, 1) if pivot > 0 else 0
            safety = ("safe"    if cur >= pivot * 0.93
                      else "caution" if cur >= pivot * 0.90
                      else "danger")
            rs_ok  = rs > 0 if "상승" in market_str else rs > -20
            # 피벗 돌파 구간 (97~105%) 체크
            pivot_ok = pat["pivot"] > 0 and (pat["pivot"] * 0.97 <= pat["cur"] <= pat["pivot"] * 1.05)
            signal = pat["vs"] and rs_ok and pivot_ok
            # 대기 중 (패턴 완성, 피벗 미돌파)
            watching = (not pivot_ok) and pat["vs"] and rs_ok

            # LightGBM 점수
            lgbm_score = None
            if signal and lgbm_model and feat_cols:
                info_with = {**info, "rs": rs, "score": score}
                feat = calc_lgbm_features(sl, kospi_idx, info_with, pat)
                if feat:
                    mkt_feat = get_market_features(kospi_idx, sig_ts)
                    feat.update(mkt_feat)
                    lgbm_score = predict_lgbm(feat, lgbm_model, feat_cols)

            all_scores.append({
                "ticker": ticker, "name": name, "market": market,
                "sector": str(info.get("sector","기타")),
                "cur": pat["cur"], "rs": rs,
                "trend_ok": True, "pattern_ok": True,
                "signal": signal, "score": score, "lgbm_score": lgbm_score,
                "pivot": pivot, "cup_depth": pat["cd"],
                "handle_depth": pat["hd"], "vol_ratio": pat["vr"],
                "cup_days": pat["cdays"], "handle_days": pat["hdays"],
                "pct_from_pivot": pct, "safety": safety,
                "watching": watching,
                "reason": "시그널" if signal else (
                    "대기중(피벗미돌파)" if watching else
                    "거래량 미충족" if not pat["vs"] else "RS 미충족"),
            })

            if not signal: break

            # watching 종목도 따로 수집
            if watching and not signal:
                signals.append({
                    "sig_date":   sig_str,
                    "ticker":     ticker,
                    "name":       name,
                    "market":     market,
                    "sector":     str(info.get("sector","기타")),
                    "cur":        pat["cur"],
                    "pivot":      pivot,
                    "cd":         pat["cd"],
                    "hd":         pat["hd"],
                    "cdays":      pat["cdays"],
                    "hdays":      pat["hdays"],
                    "cup_start":  pat.get("cup_start",""),
                    "cup_end":    pat.get("cup_end",""),
                    "vr":         pat["vr"],
                    "rs":         rs,
                    "score":      score,
                    "lgbm_score": None,
                    "pct_from_pivot": pct,
                    "safety":     safety,
                    "watching":   True,
                })
                break

            if signal:
              signals.append({
                "sig_date":   sig_str,
                "ticker":     ticker,
                "name":       name,
                "market":     market,
                "sector":     str(info.get("sector","기타")),
                "cur":        pat["cur"],
                "pivot":      pivot,
                "cd":         pat["cd"],
                "hd":         pat["hd"],
                "cdays":      pat["cdays"],
                "hdays":      pat["hdays"],
                "cup_start":  pat.get("cup_start",""),
                "cup_end":    pat.get("cup_end",""),
                "vr":         pat["vr"],
                "rs":         rs,
                "score":      score,
                "lgbm_score": lgbm_score,
                "pct_from_pivot": pct,
                "safety":     safety,
                "watching":   False,
              })
            break

    # 중복 제거
    seen = set(); deduped = []
    for r in sorted(signals,
                    key=lambda x: (x["sig_date"], x.get("lgbm_score") or 0, x["score"]),
                    reverse=True):
        if r["ticker"] not in seen:
            seen.add(r["ticker"]); deduped.append(r)
    signals = deduped

    print(f"완료: {len(signals)}개 / 트렌드:{trend_pass}개")
    send(f"스캔 완료\n"
         f"트렌드: {trend_pass}개\n"
         f"패턴 감지: {pattern_pass}개\n"
         f"거래량 충족: {vol_pass}개\n"
         f"RS 충족: {rs_pass}개\n"
         f"최종 시그널: {len(signals)}개")

    # CSV 저장
    rows = []
    for r in signals:
        rows.append({
            "date": r["sig_date"], "ticker": r["ticker"],
            "name": r["name"], "market": r["market"],
            "sector": r["sector"],
            "entry": r["cur"], "pivot": r["pivot"],
            "cup_depth": r["cd"], "handle_depth": r["hd"],
            "cup_days": r["cdays"], "handle_days": r["hdays"],
            "cup_start": r.get("cup_start",""), "cup_end": r.get("cup_end",""),
            "vol_ratio": r["vr"], "rs": r["rs"],
            "score": r["score"], "lgbm_score": r.get("lgbm_score"),
            "pct_from_pivot": r["pct_from_pivot"], "safety": r["safety"],
        })

    pd.DataFrame(rows if rows else []).to_csv(
        "scanner_kr_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_scores if all_scores else []).to_csv(
        "scanner_kr_all.csv", index=False, encoding="utf-8-sig")

    if rows:
        send_file("scanner_kr_raw.csv",
                  f"🇰🇷 국장 스캐너 ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")

    # watching/signal 분리
    signal_list  = [r for r in signals if not r.get("watching")]
    watching_list = [r for r in signals if r.get("watching")]

    # 텔레그램
    if not signals:
        send(f"🇰🇷 국장 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음")
    else:
        hdr = (f"🇰🇷 미너비니 컵&핸들(국장)\n"
               f"최근 {SCAN_DAYS}거래일 | {market_str}\n"
               f"{len(signals)}개 발견\n" + "─"*24 + "\n")
        msg = hdr
        mkt_lbl    = {"KOSPI":"🔵코스피","KOSDAQ":"🟢코스닥"}
        grade_emoji = {"S":"🏆","A":"🥇","B":"🥈","C":"🥉","D":"📊"}
        grade = lambda s: "S" if s>=90 else "A" if s>=80 else "B" if s>=70 else "C" if s>=60 else "D"

        for r in signals:
            up  = round((r["pivot"] / r["cur"] - 1) * 100, 1)
            cup_date = (f"({r.get('cup_start','')}~{r.get('cup_end','')})"
                        if r.get("cup_start") else "")
            g = grade(r["score"])
            lgbm_str = (f"  🌲LGBM: {r['lgbm_score']:.3f}\n"
                        if r.get("lgbm_score") is not None else "")
            status = "⏳ 대기중" if r.get("watching") else "🎯 매수구간"
            blk = (
                f"[{r['sig_date']}] {status} {mkt_lbl.get(r['market'],r['market'])} {r['sector']}\n"
                f"◆ {r['name']}({r['ticker']})\n"
                f"  AI점수: {grade_emoji.get(g,'📊')}{r['score']}점({g}등급)\n"
                f"{lgbm_str}"
                f"  현재가: {r['cur']:,.0f}원\n"
                f"  피벗: {r['pivot']:,.0f}원({up:+.1f}%)\n"
                f"  컵:{r['cd']}%/{r['cdays']}일{cup_date} 핸들:{r['hd']}%/{r['hdays']}일\n"
                f"  거래량:{r['vr']}x{'🔥' if r['vr']>=1.40 else ''} RS:{r['rs']:+.1f}%\n\n"
            )
            if len(msg) + len(blk) > 4000:
                send(msg); msg = "(이어서)\n\n" + blk
            else:
                msg += blk
        send(msg)
