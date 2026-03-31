"""
국장 LightGBM 데이터 수집 (EODHD 버전)
- backtest_kr_raw.csv 기반
- raw_data/kr/ 에서 OHLCV 로드
- 라벨: r5 >= -7% AND r10 >= 8% → 1
- 상승장 필터: KOSPI 20일 -15% OR 60일 -20% 이하면 제외
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW     = 150
SKIP       = 2
VOL_MA     = 20
MIN_PRICE  = 1000
MIN_TRDVAL = 5  # 억원 (거래대금 필터)
KR_DIR     = "raw_data/kr"


def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 4)


def calc_label(r5, r10):
    if pd.isna(r5) or pd.isna(r10): return None
    if r5 < -7.0: return 0
    if r10 >= 8.0: return 1
    return 0


def is_bull_market(idx_df, sig_date):
    """급락장 필터 (단기 낙폭 기반)"""
    if idx_df is None: return True
    try:
        spy_sl = idx_df.loc[:sig_date]["Close"]
    except: return True
    if len(spy_sl) < 20: return True
    cur = float(spy_sl.iloc[-1])
    if len(spy_sl) >= 20:
        ret_20 = (cur / float(spy_sl.iloc[-20]) - 1) * 100
        if ret_20 < -15: return False
    if len(spy_sl) >= 60:
        ret_60 = (cur / float(spy_sl.iloc[-60]) - 1) * 100
        if ret_60 < -20: return False
    return True


def calc_features(df, kospi_idx, d_idx, ticker_info):
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < VOL_MA: return None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None

    close  = w["Close"].values
    high   = w["High"].values  if "High"  in w.columns else close
    low    = w["Low"].values   if "Low"   in w.columns else close
    volume = w["Volume"].values

    if close[-1] < MIN_PRICE: return None

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

    # RS (코스피 기준)
    ref_idx = d_idx - SKIP
    tc = df["Close"].iloc[:ref_idx + 1].values
    rs_at = rs_20 = rs_50 = rs_150 = 0.0

    if kospi_idx is not None and len(tc) >= 253:
        td = df.index[:ref_idx + 1]
        sa = kospi_idx.reindex(td).ffill()
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

    # 섹터 OHE (국장)
    KR_SECTORS = ["음식료품","섬유의복","종이목재","화학","의약품",
                  "비금속광물","철강금속","기계","전기전자","의료정밀",
                  "운수장비","유통업","전기가스업","건설업","운수창고업",
                  "통신업","금융업","증권","보험","서비스업","기타"]
    sector = str(ticker_info.get("sector","기타") or "기타")
    if sector not in KR_SECTORS: sector = "기타"
    for s in KR_SECTORS:
        feat[f"sec_{s}"] = 1 if sector==s else 0

    # 시장 OHE
    market = str(ticker_info.get("market","KOSPI") or "KOSPI")
    feat["mkt_KOSPI"]  = 1 if market == "KOSPI"  else 0
    feat["mkt_KOSDAQ"] = 1 if market == "KOSDAQ" else 0

    # 패턴 피처
    feat["cup_depth"]    = float(ticker_info.get("cup_depth", 0) or 0)
    feat["handle_depth"] = float(ticker_info.get("handle_depth", 0) or 0)
    feat["vol_ratio"]    = float(ticker_info.get("vol_ratio", 0) or 0)
    feat["cup_days"]     = float(ticker_info.get("cup_days", 0) or 0)
    feat["handle_days"]  = float(ticker_info.get("handle_days", 0) or 0)
    feat["vs"]           = 1 if str(ticker_info.get("vs","")).lower() == "true" else 0
    feat["rs_signal"]    = float(ticker_info.get("rs", 0) or 0)
    feat["score"]        = float(ticker_info.get("score", 0) or 0)

    return feat


if __name__ == "__main__":
    # 백테스트 로드 + 라벨
    bt = pd.read_csv("backtest_kr_raw.csv", encoding="utf-8-sig")
    bt["date"]  = pd.to_datetime(bt["date"])
    bt["label"] = bt.apply(lambda r: calc_label(r.get("r5"), r.get("r10")), axis=1)
    bt = bt.dropna(subset=["label"]).copy()
    bt["label"] = bt["label"].astype(int)
    bt = bt.sort_values("date").reset_index(drop=True)

    pos = (bt.label==1).sum(); neg = (bt.label==0).sum()
    print(f"시그널: {len(bt)}건 (양성:{pos} 음성:{neg} 비율:{pos/(pos+neg)*100:.1f}%)")

    # KOSPI 지수 로드 (없으면 EODHD에서 수집)
    from eodhd_utils import get_ohlcv, EODHD
    kospi_idx  = None
    kospi_path = os.path.join(KR_DIR, "069500.csv")
    if os.path.exists(kospi_path):
        kospi_idx = pd.read_csv(kospi_path, index_col="date", parse_dates=True)
        print(f"KOSPI 지수 캐시 로드: {len(kospi_idx)}일치")
    elif EODHD:
        print("KOSPI 지수 없음 → EODHD에서 수집 중...")
        kospi_idx = get_ohlcv("069500", "KO", start="2000-01-01")
        if kospi_idx is not None:
            os.makedirs(KR_DIR, exist_ok=True)
            kospi_idx.to_csv(kospi_path)
            print(f"KOSPI 지수 수집 완료: {len(kospi_idx)}일치")
        else:
            print("⚠️ KOSPI 지수 수집 실패 → rs=0 처리")
    else:
        print("⚠️ EODHD 토큰 없음 → rs=0 처리")

    # 필터 테스트
    if kospi_idx is not None:
        for test_date in ["2008-10-15", "2020-03-20", "2023-06-15"]:
            result = is_bull_market(kospi_idx, pd.Timestamp(test_date))
            print(f"  is_bull_market({test_date}): {result}")

    # 종목별 피처 계산
    samples  = []
    grouped  = bt.groupby("ticker")
    n_tickers = len(grouped)

    for t_idx, (ticker, group) in enumerate(grouped):
        ticker = str(ticker).zfill(6)
        if t_idx % 50 == 0:
            print(f"[{t_idx}/{n_tickers}] {ticker} | 수집:{len(samples)}건")

        fpath = os.path.join(KR_DIR, f"{ticker}.csv")
        if not os.path.exists(fpath): continue

        try:
            df = pd.read_csv(fpath, index_col="date", parse_dates=True)
            cols = [c for c in ["Open","High","Low","Close","Adj_Close","Volume"] if c in df.columns]
            df = df[cols].astype(float).dropna()
            if "Close" not in df.columns: continue
        except: continue

        if len(df) < WINDOW + SKIP + 10: continue
        idx_list = df.index.tolist()

        for _, row in group.iterrows():
            sig_date = pd.Timestamp(row["date"])

            # 상승장 필터
            if not is_bull_market(kospi_idx, sig_date):
                continue

            matches = [x for x in idx_list if x.date() == sig_date.date()]
            if not matches: continue
            d_idx = idx_list.index(matches[0])
            if d_idx < WINDOW + SKIP: continue

            info = {
                "sector":       str(row.get("sector","기타") or "기타"),
                "market":       str(row.get("market","KOSPI") or "KOSPI"),
                "cup_depth":    row.get("cup_depth", 0),
                "handle_depth": row.get("handle_depth", 0),
                "vol_ratio":    row.get("vol_ratio", 0),
                "cup_days":     row.get("cup_days", 0),
                "handle_days":  row.get("handle_days", 0),
                "vs":           row.get("vs", False),
                "rs":           row.get("rs", 0),
                "score":        row.get("score", 0),
            }

            feat = calc_features(df, kospi_idx, d_idx, info)
            if feat is None: continue

            feat["ticker"] = ticker
            feat["name"]   = str(row.get("name", ticker))
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
    df_out.to_csv("lgbm_raw_kr.csv", index=False, encoding="utf-8-sig")
    print("저장: lgbm_raw_kr.csv")
