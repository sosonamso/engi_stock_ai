"""
미장 LightGBM 데이터 수집 (EODHD 버전)
- backtest_us_raw.csv 기반
- raw_data/us/ 에서 OHLCV 로드
- 라벨: r5 >= -7% AND r10 >= 8% → 1
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW     = 150
SKIP       = 2
VOL_MA     = 20
MIN_PRICE  = 5.0
MIN_VOL    = 100000
US_DIR     = "raw_data/us"


def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 4)


def calc_label(r5, r10):
    if pd.isna(r5) or pd.isna(r10): return None
    if r5 < -7.0: return 0
    if r10 >= 8.0: return 1
    return 0


def calc_features(df, spy_df, d_idx, ticker_info):
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < VOL_MA: return None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None

    close  = w["Close"].values
    high   = w["High"].values
    low    = w["Low"].values
    volume = w["Volume"].values

    if close[-1] < MIN_PRICE: return None
    if np.mean(volume[-20:]) < MIN_VOL: return None

    c_min, c_max = close.min(), close.max()
    if c_max == c_min: return None
    close_norm = (close - c_min) / (c_max - c_min)

    ret = np.zeros(WINDOW)
    ret[1:] = (close[1:] / close[:-1] - 1) * 100

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
    ref_idx = d_idx - SKIP
    td = df.index[:ref_idx + 1]
    tc = df["Close"].iloc[:ref_idx + 1].values
    rs_at = rs_20 = rs_50 = rs_150 = 0.0

    if spy_df is not None and len(tc) >= 253:
        sa = spy_df.reindex(td).ffill()
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

    feat["rs_at_d2"] = rs_at
    feat["rs_20"]    = rs_20
    feat["rs_50"]    = rs_50
    feat["rs_150"]   = rs_150
    feat["rs_trend"] = round(rs_20 - rs_50, 4)

    # 기술적 지표
    feat["rsi_14"] = calc_rsi(close, 14)

    ma20  = np.mean(close[-20:])
    std20 = np.std(close[-20:])
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
    feat["atr_ratio"] = round(float(atr/close[-1]) if close[-1]>0 else 0, 4)
    feat["ma20_pos"]  = round(float(close[-1]/np.mean(close[-20:])), 4)
    feat["ma50_pos"]  = round(float(close[-1]/np.mean(close[-50:])) if len(close)>=50 else 1.0, 4)

    # 섹터 OHE
    SECTORS = ["Technology","Healthcare","Financial","Consumer","Energy",
               "Industrial","Materials","Utilities","Real Estate","Communication","Other"]
    sector = str(ticker_info.get("sector", "") or "Other")
    if sector not in SECTORS: sector = "Other"
    for s in SECTORS:
        feat[f"sec_{s}"] = 1 if sector==s else 0

    # 시총 OHE
    CAPS = ["MegaCap","LargeCap","MidCap","SmallCap"]
    cap = str(ticker_info.get("cap", "") or "SmallCap")
    if cap not in CAPS: cap = "SmallCap"
    for c in CAPS:
        feat[f"cap_{c}"] = 1 if cap==c else 0

    return feat


if __name__ == "__main__":
    # 백테스트 로드 + 라벨
    bt = pd.read_csv("backtest_us_raw.csv", encoding="utf-8-sig")
    bt["date"]  = pd.to_datetime(bt["date"])
    bt["label"] = bt.apply(lambda r: calc_label(r.get("r5"), r.get("r10")), axis=1)
    bt = bt.dropna(subset=["label"]).copy()
    bt["label"] = bt["label"].astype(int)
    bt = bt.sort_values("date").reset_index(drop=True)

    pos = (bt.label==1).sum(); neg = (bt.label==0).sum()
    print(f"시그널: {len(bt)}건 (양성:{pos} 음성:{neg} 비율:{pos/(pos+neg)*100:.1f}%)")

    # 티커 메타
    meta_map = {}
    if os.path.exists("tickers_us.csv"):
        mdf = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        meta_map = {str(r["ticker"]): r.to_dict() for _, r in mdf.iterrows()}

    # SPY 로드
    spy_df = None
    spy_path = os.path.join(US_DIR, "SPY.csv")
    if os.path.exists(spy_path):
        spy_df = pd.read_csv(spy_path, index_col="date", parse_dates=True)
        print(f"SPY: {len(spy_df)}일치")

    # 종목별 피처 계산
    samples  = []
    grouped  = bt.groupby("ticker")
    n_tickers = len(grouped)

    for t_idx, (ticker, group) in enumerate(grouped):
        if t_idx % 100 == 0:
            print(f"[{t_idx}/{n_tickers}] {ticker} | 수집:{len(samples)}건")

        fpath = os.path.join(US_DIR, f"{ticker}.csv")
        if not os.path.exists(fpath): continue

        try:
            df = pd.read_csv(fpath, index_col="date", parse_dates=True)
            df = df[["Open","High","Low","Close","Volume"]].astype(float).dropna()
        except: continue

        if len(df) < WINDOW + SKIP + 10: continue
        idx_list = df.index.tolist()
        info     = meta_map.get(str(ticker), {})

        for _, row in group.iterrows():
            sig_date = pd.Timestamp(row["date"])
            matches  = [x for x in idx_list if x.date() == sig_date.date()]
            if not matches: continue
            d_idx = idx_list.index(matches[0])
            if d_idx < WINDOW + SKIP: continue

            feat = calc_features(df, spy_df, d_idx, info)
            if feat is None: continue

            feat["ticker"] = str(ticker)
            feat["date"]   = sig_date.strftime("%Y-%m-%d")
            feat["label"]  = int(row["label"])
            feat["entry"]  = float(row["entry"])
            feat["r5"]     = row.get("r5")
            feat["r10"]    = row.get("r10")
            samples.append(feat)

    print(f"\n수집 완료: {len(samples)}건")
    if not samples:
        print("수집 결과 없음"); exit(0)

    df_out = pd.DataFrame(samples)
    pos2 = (df_out.label==1).sum(); neg2 = (df_out.label==0).sum()
    print(f"양성:{pos2} 음성:{neg2} 비율:{pos2/(pos2+neg2)*100:.1f}%")
    df_out.to_csv("lgbm_raw_us.csv", index=False, encoding="utf-8-sig")
    print("저장: lgbm_raw_us.csv")
