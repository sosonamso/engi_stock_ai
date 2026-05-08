"""
국장 수급 맥점 스캐너
- 네이버 인기순위 30개 종목
- 기관+외국인 순매수 / 시총 >= 0.3%
- 매일 저녁 자동 실행
"""
import os, time, warnings, requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

warnings.filterwarnings("ignore")

TOK = os.environ.get("TELEGRAM_TOKEN", "")
CID = os.environ.get("TELEGRAM_CHAT_ID", "")
THRESHOLD = 0.003  # 0.3%

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
    """네이버 인기순위 30개 수집"""
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
                if len(code) == 6 and code.isdigit() and name:
                    if code not in [s["ticker"] for s in stocks]:
                        stocks.append({"ticker": code, "name": name, "rank": rank})
                        rank += 1
        print(f"인기순위: {len(stocks)}개")
        return stocks
    except Exception as e:
        print(f"인기순위 수집 실패: {e}")
        return []


def get_supply_data(ticker):
    """기관+외국인 순매수 + 시총 수집"""
    try:
        # frgn 페이지 → 기관/외국인 순매수
        url = f"https://finance.naver.com/item/frgn.naver?code={ticker}"
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(resp.text))

        inst_buy = None   # 기관 순매수
        frgn_buy = None   # 외국인 순매수

        # 테이블 3: 날짜별 기관/외국인 순매매량
        for t in tables:
            cols = [str(c) for c in t.columns.tolist()]
            cols_flat = ' '.join(cols)
            if '기관' in cols_flat and '외국인' in cols_flat:
                # 첫 번째 데이터 행 (오늘)
                for _, row in t.iterrows():
                    if pd.isna(row.iloc[0]): continue
                    # 기관 순매매량
                    for j, col in enumerate(t.columns):
                        col_str = str(col)
                        if '기관' in col_str and '순매매' in col_str:
                            inst_buy = float(row.iloc[j]) if pd.notna(row.iloc[j]) else None
                        if '외국인' in col_str and '순매매' in col_str:
                            frgn_buy = float(row.iloc[j]) if pd.notna(row.iloc[j]) else None
                    break
                break

        time.sleep(0.3)

        # main 페이지 → 시총 + 현재가
        url2 = f"https://finance.naver.com/item/main.naver?code={ticker}"
        resp2 = requests.get(url2, headers=headers, timeout=10)
        tables2 = pd.read_html(StringIO(resp2.text))

        mktcap = None
        cur_price = None

        for t in tables2:
            vals = str(t.values.tolist())
            if '시가총액' in vals:
                for _, row in t.iterrows():
                    row_str = str(row.tolist())
                    if '시가총액' in row_str:
                        # 시총 파싱 (예: "1,543조 4,176억원")
                        for v in row.tolist():
                            v_str = str(v)
                            if '조' in v_str:
                                try:
                                    jo = float(v_str.split('조')[0].replace(',','').strip())
                                    ok = 0
                                    if '억' in v_str:
                                        ok = float(v_str.split('조')[1].split('억')[0].replace(',','').strip())
                                    mktcap = (jo * 1_000_000_000_000) + (ok * 100_000_000)
                                except: pass
                                break
                break

        # 현재가는 frgn 테이블에서
        for t in tables:
            for _, row in t.iterrows():
                if pd.isna(row.iloc[0]): continue
                try:
                    cur_price = float(str(row.iloc[1]).replace(',',''))
                    if cur_price > 100: break
                except: pass
            if cur_price: break

        return inst_buy, frgn_buy, mktcap, cur_price

    except Exception as e:
        print(f"  [{ticker}] 수급 수집 실패: {e}")
        return None, None, None, None


if __name__ == "__main__":
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"🔍 수급 맥점 스캐너 {today}")

    # 인기순위 수집
    popular = get_popular_stocks()
    if not popular:
        send("인기순위 수집 실패"); exit(0)

    results = []
    for s in popular:
        ticker = s["ticker"]
        name   = s["name"]
        rank   = s["rank"]
        print(f"[{rank}위] {name}({ticker}) 수집 중...")

        inst, frgn, mktcap, cur = get_supply_data(ticker)
        time.sleep(0.3)

        if inst is None or frgn is None or mktcap is None or mktcap == 0:
            continue

        total_buy = inst + frgn
        ratio     = total_buy / mktcap * 100  # %

        results.append({
            "rank":      rank,
            "ticker":    ticker,
            "name":      name,
            "inst":      inst,
            "frgn":      frgn,
            "total_buy": total_buy,
            "mktcap":    mktcap,
            "ratio":     round(ratio, 3),
            "cur":       cur,
        })
        print(f"  기관:{inst/1e8:.0f}억 외국인:{frgn/1e8:.0f}억 시총대비:{ratio:+.3f}%")

    print(f"\n분석 완료: {len(results)}개")

    # 맥점 필터 (0.3% 이상)
    signals = [r for r in results if r["ratio"] >= THRESHOLD * 100]
    # 음수도 참고용으로 (외국인+기관 대량 매도)
    selloffs = [r for r in results if r["ratio"] <= -THRESHOLD * 100]

    # CSV 저장
    if results:
        pd.DataFrame(results).to_csv("supply_scan_kr.csv", index=False, encoding="utf-8-sig")

    # 텔레그램
    tv_url = lambda t: f"https://www.tradingview.com/chart/?symbol=KRX:{t}"

    msg  = f"💰 <b>수급 맥점 스캐너</b> | {today}\n"
    msg += f"네이버 인기순위 {len(popular)}개 분석\n"
    msg += f"기준: (기관+외국인) / 시총 ≥ 0.3%\n"
    msg += "─" * 22 + "\n\n"

    if signals:
        msg += f"🚨 <b>맥점 포착 {len(signals)}개</b>\n\n"
        for r in sorted(signals, key=lambda x: x["ratio"], reverse=True):
            inst_str  = f"{r['inst']/1e8:+.0f}억"
            frgn_str  = f"{r['frgn']/1e8:+.0f}억"
            total_str = f"{r['total_buy']/1e8:+.0f}억"
            msg += (
                f"🔥 <b>{r['name']}</b>({r['ticker']}) [인기{r['rank']}위]\n"
                f"  기관:{inst_str} | 외국인:{frgn_str}\n"
                f"  합계:{total_str} | 시총대비:{r['ratio']:+.3f}%\n"
                f"  📊 {tv_url(r['ticker'])}\n\n"
            )
    else:
        msg += "⚠️ 맥점 신호 없음\n\n"

    # 전체 순위 요약
    msg += "─" * 22 + "\n"
    msg += "📊 <b>전체 수급 현황</b>\n"
    for r in sorted(results, key=lambda x: x["ratio"], reverse=True):
        bar = "🟢" if r["ratio"] >= 0.3 else "🔴" if r["ratio"] <= -0.3 else "⚪"
        msg += f"{bar} {r['name']}({r['rank']}위): {r['ratio']:+.3f}%\n"

    send(msg)
    send(f"✅ 수급 스캔 완료")
