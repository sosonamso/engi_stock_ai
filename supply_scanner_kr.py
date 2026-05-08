"""
국장 수급 맥점 스캐너 v2
- 네이버 인기순위 30개 종목
- 최근 15일치 기관+외국인 순매수 분석
- 맥점(시총 대비 0.3% 이상) 날짜 포착
"""
import os, time, warnings, requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

warnings.filterwarnings("ignore")

TOK       = os.environ.get("TELEGRAM_TOKEN", "")
CID       = os.environ.get("TELEGRAM_CHAT_ID", "")
THRESHOLD = 0.003   # 0.3%
DAYS      = 15      # 최근 15일

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text, "parse_mode": "HTML"},
                         timeout=10)
        except: pass


def get_popular_stocks():
    try:
        resp = requests.get(
            "https://finance.naver.com/sise/lastsearch2.naver",
            headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        stocks = []
        rank = 1
        for a in soup.select("a[href*='code=']"):
            href = a["href"]
            if "code=" in href:
                code = href.split("code=")[-1].split("&")[0].strip()
                name = a.text.strip()
                if len(code)==6 and code.isdigit() and name:
                    if code not in [s["ticker"] for s in stocks]:
                        stocks.append({"ticker": code, "name": name, "rank": rank})
                        rank += 1
        print(f"인기순위: {len(stocks)}개")
        return stocks
    except Exception as e:
        print(f"인기순위 수집 실패: {e}"); return []


def get_mktcap(ticker):
    """시가총액 수집"""
    try:
        url  = f"https://finance.naver.com/item/main.naver?code={ticker}"
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(resp.text))
        for t in tables:
            vals_flat = ' '.join([str(v) for v in t.values.flatten()])
            if '시가총액' in vals_flat and '조' in vals_flat:
                for v in t.values.flatten():
                    v_str = str(v)
                    if '조' in v_str and '억' in v_str:
                        try:
                            jo = float(v_str.split('조')[0].replace(',','').strip())
                            ok = float(v_str.split('조')[1].split('억')[0].replace(',','').strip())
                            return (jo * 1_000_000_000_000) + (ok * 100_000_000)
                        except: pass
                    elif '조' in v_str and 'nan' not in v_str.lower():
                        try:
                            jo = float(v_str.split('조')[0].replace(',','').strip())
                            return jo * 1_000_000_000_000
                        except: pass
    except Exception as e:
        print(f"  [{ticker}] 시총 수집 실패: {e}")
    return None


def get_supply_history(ticker, days=15):
    """최근 N일치 기관+외국인 순매수 수집"""
    try:
        url  = f"https://finance.naver.com/item/frgn.naver?code={ticker}"
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(resp.text))

        for t in tables:
            cols_flat = ' '.join([str(c) for c in t.columns.tolist()])
            if '기관' not in cols_flat or '외국인' not in cols_flat: continue

            # 멀티인덱스 컬럼 → 단순화
            new_cols = []
            inst_idx = frgn_idx = date_idx = close_idx = None
            for i, c in enumerate(t.columns):
                c_str = str(c)
                if '날짜' in c_str:                               new_cols.append('date');  date_idx  = i
                elif '종가' in c_str:                             new_cols.append('close'); close_idx = i
                elif '기관' in c_str and '순매매' in c_str:       new_cols.append('inst');  inst_idx  = i
                elif '외국인' in c_str and '순매매' in c_str:     new_cols.append('frgn');  frgn_idx  = i
                else: new_cols.append(f'col_{i}')

            t.columns = new_cols
            if inst_idx is None or frgn_idx is None or date_idx is None: continue

            t = t.dropna(subset=['date'])
            t = t[t['date'].astype(str).str.match(r'\d{4}\.\d{2}\.\d{2}')]
            t['date']  = pd.to_datetime(t['date'], format='%Y.%m.%d')
            t['inst']  = pd.to_numeric(t['inst'],  errors='coerce')
            t['frgn']  = pd.to_numeric(t['frgn'],  errors='coerce')
            t['close'] = pd.to_numeric(t['close'], errors='coerce')
            t = t.dropna(subset=['inst','frgn']).head(days)
            return t[['date','close','inst','frgn']].reset_index(drop=True)

    except Exception as e:
        print(f"  [{ticker}] 수급 수집 실패: {e}")
    return None


if __name__ == "__main__":
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"💰 수급 맥점 스캐너 v2 | {today}")

    popular = get_popular_stocks()
    if not popular:
        send("인기순위 수집 실패"); exit(0)

    all_signals = []  # 맥점 발생 (종목, 날짜)
    summary     = []  # 전체 요약

    for s in popular:
        ticker = s["ticker"]
        name   = s["name"]
        rank   = s["rank"]
        print(f"[{rank}위] {name}({ticker}) 수집 중...")

        mktcap = get_mktcap(ticker)
        time.sleep(0.2)
        hist   = get_supply_history(ticker, DAYS)
        time.sleep(0.2)

        print(f"  시총: {mktcap} | 수급행수: {len(hist) if hist is not None else 0}")
        if hist is not None and len(hist) > 0:
            print(f"  수급샘플: inst={hist['inst'].iloc[0]} frgn={hist['frgn'].iloc[0]}")

        if mktcap is None or hist is None or len(hist) == 0:
            continue

        # 맥점 날짜 찾기
        hist["total"] = (hist["inst"] + hist["frgn"]) * hist["close"]
        hist["ratio"] = hist["total"] / mktcap * 100
        pivots = hist[hist["ratio"].abs() >= THRESHOLD * 100]

        if len(pivots) > 0:
            for _, row in pivots.iterrows():
                all_signals.append({
                    "rank":   rank,
                    "ticker": ticker,
                    "name":   name,
                    "date":   row["date"].strftime("%m/%d"),
                    "inst":   row["inst"] * row["close"],
                    "frgn":   row["frgn"] * row["close"],
                    "total":  row["total"],
                    "ratio":  round(row["ratio"], 3),
                    "close":  row["close"],
                })

        # 최근 1일 요약
        latest = hist.iloc[0]
        summary.append({
            "rank":   rank,
            "ticker": ticker,
            "name":   name,
            "ratio":  round(float(latest["ratio"]), 3),
        })

        print(f"  맥점: {len(pivots)}건 | 최근: {float(latest['ratio']):+.3f}%")

    # CSV 저장
    if all_signals:
        pd.DataFrame(all_signals).to_csv("supply_signals_kr.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv("supply_scan_kr.csv", index=False, encoding="utf-8-sig")

    # 텔레그램
    tv_url = lambda t: f"https://www.tradingview.com/chart/?symbol=KRX:{t}"

    msg  = f"💰 <b>수급 맥점 스캐너 v2</b> | {today}\n"
    msg += f"인기순위 {len(popular)}개 | 최근 {DAYS}일 분석\n"
    msg += f"기준: (기관+외국인) / 시총 ≥ 0.3%\n"
    msg += "─" * 22 + "\n\n"

    if all_signals:
        # 날짜 최신순 정렬
        sig_sorted = sorted(all_signals, key=lambda x: (x["date"], abs(x["ratio"])), reverse=True)
        msg += f"🚨 <b>맥점 포착 {len(sig_sorted)}건</b>\n\n"
        for r in sig_sorted:
            sign = "📈" if r["total"] > 0 else "📉"
            msg += (
                f"{sign} <b>{r['name']}</b>({r['ticker']}) [인기{r['rank']}위]\n"
                f"  날짜: {r['date']} | 종가: {r['close']:,.0f}원\n"
                f"  기관:{r['inst']/1e8:+.0f}억 | 외국인:{r['frgn']/1e8:+.0f}억\n"
                f"  합계:{r['total']/1e8:+.0f}억 | 시총대비:{r['ratio']:+.3f}%\n"
                f"  📊 {tv_url(r['ticker'])}\n\n"
            )
        send(msg)
    else:
        msg += "⚠️ 최근 15일 내 맥점 신호 없음\n"
        send(msg)

    # 오늘 전체 수급 요약
    sum_msg = "📊 <b>오늘 수급 현황 (인기순위)</b>\n"
    for r in sorted(summary, key=lambda x: x["ratio"], reverse=True):
        bar = "🟢" if r["ratio"] >= 0.3 else "🔴" if r["ratio"] <= -0.3 else "⚪"
        sum_msg += f"{bar} {r['name']}({r['rank']}위): {r['ratio']:+.3f}%\n"
    send(sum_msg)
    send(f"✅ 수급 스캔 완료")
