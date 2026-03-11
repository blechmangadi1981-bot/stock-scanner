"""

Stock Scanner Pro - Complete Single-File Application

=====================================================

Upload this single file as app.py to Streamlit Cloud or any hosting service.

 

Deployment Instructions:

1. Go to https://share.streamlit.io

2. Sign in with GitHub

3. Create a new repo on GitHub with just this file as app.py

4. Add a requirements.txt with: streamlit yfinance pandas_ta pandas numpy lxml requests

5. Deploy!

 

Or use https://streamlit.io/cloud direct paste.

"""

 

import streamlit as st

import pandas as pd

import yfinance as yf

import pandas_ta as ta

import numpy as np

from datetime import datetime

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings

 

warnings.filterwarnings("ignore")

 

# ═══════════════════════════════════════════════════════════════

# PAGE CONFIG & STYLING

# ═══════════════════════════════════════════════════════════════

st.set_page_config(

    page_title="Stock Scanner Pro",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded",

)

 

st.markdown("""

<style>

    .main-header {

        font-size: 2.5rem;

        font-weight: 700;

        background: linear-gradient(90deg, #00C853, #00BFA5);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        text-align: center;

        padding: 0.5rem 0;

    }

    .sub-header {

        text-align: center;

        font-size: 1.05rem;

        opacity: 0.7;

        margin-bottom: 2rem;

    }

    div[data-testid="stMetric"] {

        background: rgba(255,255,255,0.05);

        border: 1px solid rgba(255,255,255,0.1);

        border-radius: 10px;

        padding: 0.8rem;

    }

    #MainMenu {visibility: hidden;}

    footer {visibility: hidden;}

    .stDeployButton {display: none;}

</style>

""", unsafe_allow_html=True)

 

 

# ═══════════════════════════════════════════════════════════════

# TICKER LISTS

# ═══════════════════════════════════════════════════════════════

 

@st.cache_data(ttl=86400, show_spinner=False)

def get_sp500_tickers():

    """Fetch S&P 500 tickers from Wikipedia."""

    try:

        url = https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

        tables = pd.read_html(url)

        df = tables[0]

        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]

        names = dict(zip(tickers, df["Security"].tolist()))

        sectors = dict(zip(tickers, df["GICS Sector"].tolist()))

        return tickers, names, sectors

    except Exception as e:

        st.error(f"Failed to fetch S&P 500 list: {e}")

        return [], {}, {}

 

 

@st.cache_data(ttl=86400, show_spinner=False)

def get_ta125_tickers():

    """TA-125 major components with .TA suffix."""

    symbols = [

        "TEVA","ICL","NICE","CHKP","PERI","ESLT","MNRT","LUMI","ORA","BEZQ",

        "CEL","DLEKG","HARL","TASE","ARPT","PHOE","MZTF","DSCT","MGDL","SPNS",

        "AMOT","ORDS","ALHE","PTCH","DANE","KRNT","SPEN","ENLT","FTAL","BRMG",

        "ISRS","ELCO","ILDC","MISH","SLARL","NAWI","RLCO","AZRG","BIGI","GZIT",

        "SKBN","BSEN","CLIS","DIMRI","ELAL","PCBT","POLY","RMLI","SANO","SHPG",

        "VILR","WLFD","YBOX","KRNV","LVPR","MCRE","NXGN","SHRA","TIGBR","AIR",

        "ALBR","AMAN","AURA","BCOM","BIG","BIRM","BRND","BWAY","CNMD","CNTG",

        "DELT","ELBT","EMTC","EVGN","FNTS","FRSX","GLTC","GNRS","HLAN","ILCO",

        "ILX","INBR","INFR","ISCD","ISHI","ISOP","ISRA","KAFR","KMDA","KONE",

        "LAHAV","LHIS","MGOR","MTRX","MVNE","NTML","OPAL","ORBI","PLSN","PNAX",

        "PRGO","RAVD","REKA","RIMO","ROBO","RVLT","SFET","SMTO","SNEL","SPRG",

        "STCM","TALD","TKUN","TMRP","TNPV","TSEM","UNIT","VNTZ","VTNA","WLINT"

    ]

    tickers = [f"{s}.TA" for s in symbols]

    names = {f"{s}.TA": s for s in symbols}

    return tickers, names

 

 

# ═══════════════════════════════════════════════════════════════

# STOCK DATA FETCHING

# ═══════════════════════════════════════════════════════════════

 

def fetch_stock_data(ticker):

    """Download 3 months of history + info for one ticker."""

    try:

        stock = yf.Ticker(ticker)

        df = stock.history(period="3mo", auto_adjust=True)

        if df is None or df.empty or len(df) < 20:

            return None, None

        info = {}

        try:

            info = stock.info or {}

        except Exception:

            pass

        return df, info

    except Exception:

        return None, None

 

 

# ═══════════════════════════════════════════════════════════════

# TECHNICAL ANALYSIS FUNCTIONS

# ═══════════════════════════════════════════════════════════════

 

def analyze_rsi(df):

    """RSI(14) crossing above 30 from below."""

    try:

        rsi = ta.rsi(df["Close"], length=14)

        if rsi is None:

            return None, False, ""

        rsi = rsi.dropna()

        if len(rsi) < 3:

            return None, False, ""

 

        curr = round(float(rsi.iloc[-1]), 2)

        prev = float(rsi.iloc[-2])

        prev2 = float(rsi.iloc[-3])

 

        crossed = (prev <= 30 and curr > 30) or (prev2 <= 30 and curr > 30)

 

        desc = f"RSI crossed above 30 → {curr}" if crossed else f"RSI: {curr}"

        return curr, crossed, desc

    except Exception:

        return None, False, ""

 

 

def analyze_volume(df):

    """Detect increasing volume over last 3 days or above-average volume."""

    try:

        if "Volume" not in df.columns or len(df) < 5:

            return None, False, ""

 

        v = df["Volume"].tail(5).values

        last3 = v[-3:]

 

        increasing = bool(last3[2] > last3[1] > last3[0])

 

        avg20 = float(df["Volume"].tail(20).mean())

        above_avg = bool(last3[-1] > avg20 * 1.1) if avg20 > 0 else False

 

        signal = increasing or above_avg

 

        if increasing:

            desc = "Volume increasing 3 consecutive days"

        elif above_avg:

            pct = ((last3[-1] / avg20) - 1) * 100 if avg20 > 0 else 0

            desc = f"Volume {pct:.0f}% above 20-day avg"

        else:

            desc = "Normal volume"

 

        return float(last3[-1]), signal, desc

    except Exception:

        return None, False, ""

 

 

def analyze_trend(df):

    """

    Bullish reversal detection:

    Bullish Engulfing, Hammer, MACD Cross, EMA breakout,

    Bullish Divergence, Morning Star.

    Returns (score, pattern_list, description).

    """

    signals = []

    score = 0

 

    if len(df) < 26:

        return 0, [], "Insufficient data"

 

    close = df["Close"]

    opn = df["Open"]

    high = df["High"]

    low = df["Low"]

 

    # 1 ── Bullish Engulfing ──────────────────────────────

    try:

        if len(df) >= 2:

            po, pc = float(opn.iloc[-2]), float(close.iloc[-2])

            co, cc = float(opn.iloc[-1]), float(close.iloc[-1])

            if pc < po and cc > co and cc > po and co < pc:

                signals.append("Bullish Engulfing")

                score += 2

    except Exception:

        pass

 

    # 2 ── Hammer ─────────────────────────────────────────

    try:

        co, cc = float(opn.iloc[-1]), float(close.iloc[-1])

        ch, cl = float(high.iloc[-1]), float(low.iloc[-1])

        body = abs(cc - co)

        lower_shadow = min(co, cc) - cl

        upper_shadow = ch - max(co, cc)

        if body > 0 and lower_shadow >= 2 * body and upper_shadow <= body * 0.5:

            signals.append("Hammer")

            score += 2

    except Exception:

        pass

 

    # 3 ── MACD Crossover ─────────────────────────────────

    try:

        m = ta.macd(close, fast=12, slow=26, signal=9)

        if m is not None and len(m.dropna()) >= 2:

            cols = m.columns.tolist()

            mc = [c for c in cols if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]

            sc = [c for c in cols if c.startswith("MACDs_")]

            if mc and sc:

                ml = m[mc[0]].dropna()

                sl = m[sc[0]].dropna()

                if len(ml) >= 2 and len(sl) >= 2:

                    pd_val = float(ml.iloc[-2]) - float(sl.iloc[-2])

                    cd_val = float(ml.iloc[-1]) - float(sl.iloc[-1])

                    if pd_val < 0 and cd_val > 0:

                        signals.append("MACD Bullish Cross")

                        score += 3

                    elif pd_val <= 0 < cd_val:

                        signals.append("MACD Turning Bullish")

                        score += 1

    except Exception:

        pass

 

    # 4 ── EMA(20) Breakout ───────────────────────────────

    try:

        ema = ta.ema(close, length=20)

        if ema is not None:

            ema = ema.dropna()

            if len(ema) >= 2:

                was_below = float(close.iloc[-2]) <= float(ema.iloc[-2])

                now_above = float(close.iloc[-1]) > float(ema.iloc[-1])

                if was_below and now_above:

                    signals.append("EMA(20) Breakout")

                    score += 2

                elif now_above:

                    score += 0.5

    except Exception:

        pass

 

    # 5 ── Bullish Divergence ─────────────────────────────

    try:

        rsi = ta.rsi(close, length=14)

        if rsi is not None:

            rsi = rsi.dropna()

            if len(rsi) >= 10:

                rc = close.tail(10)

                rr = rsi.tail(10)

                h1, h2 = rc.iloc[:5], rc.iloc[5:]

                if len(h1) > 0 and len(h2) > 0:

                    l1, l2 = float(h1.min()), float(h2.min())

                    i1, i2 = h1.idxmin(), h2.idxmin()

                    if l2 < l1 and i1 in rr.index and i2 in rr.index:

                        if float(rr.loc[i2]) > float(rr.loc[i1]):

                            signals.append("Bullish Divergence")

                            score += 3

    except Exception:

        pass

 

    # 6 ── Morning Star ───────────────────────────────────

    try:

        if len(df) >= 3:

            d1 = float(close.iloc[-3]) - float(opn.iloc[-3])

            d2 = abs(float(close.iloc[-2]) - float(opn.iloc[-2]))

            d3 = float(close.iloc[-1]) - float(opn.iloc[-1])

            if d1 < 0 and d2 < abs(d1) * 0.3 and d3 > 0 and d3 > abs(d1) * 0.5:

                signals.append("Morning Star")

                score += 3

    except Exception:

        pass

 

    desc = " | ".join(signals) if signals else "No reversal pattern"

    return score, signals, desc

 

 

# ═══════════════════════════════════════════════════════════════

# FINANCIAL HELPERS

# ═══════════════════════════════════════════════════════════════

 

def get_pe(info):

    try:

        pe = info.get("trailingPE") or info.get("forwardPE")

        if pe is not None and np.isfinite(pe):

            return round(float(pe), 2)

    except Exception:

        pass

    return None

 

 

def get_analyst_rec(info):

    try:

        r = info.get("recommendationKey")

        if r:

            return r.replace("_", " ").title()

    except Exception:

        pass

    try:

        tgt = info.get("targetMeanPrice")

        cur = info.get("currentPrice") or info.get("regularMarketPrice")

        if tgt and cur and cur > 0:

            u = ((tgt / cur) - 1) * 100

            return f"{'+'if u>=0 else ''}{u:.0f}% upside"

    except Exception:

        pass

    return None

 

 

# ═══════════════════════════════════════════════════════════════

# SINGLE STOCK SCANNER

# ═══════════════════════════════════════════════════════════════

 

def scan_stock(ticker, is_us, names, sectors):

    """Full analysis of one ticker. Returns dict or None."""

    try:

        df, info = fetch_stock_data(ticker)

        if df is None:

            return None

 

        rsi_val, rsi_sig, rsi_desc = analyze_rsi(df)

        vol_val, vol_sig, vol_desc = analyze_volume(df)

        trend_sc, trend_pats, trend_desc = analyze_trend(df)

 

        total = (3 if rsi_sig else 0) + (2 if vol_sig else 0) + trend_sc

        if total < 2:

            return None

 

        price = None

        change = None

        try:

            price = round(float(df["Close"].iloc[-1]), 2)

            if len(df) >= 2:

                prev = float(df["Close"].iloc[-2])

                if prev > 0:

                    change = round(((price / prev) - 1) * 100, 2)

        except Exception:

            pass

 

        row = {

            "Ticker": ticker,

            "Company": names.get(ticker, ticker.replace(".TA", "")),

            "Price": price,

            "Change %": change,

            "RSI(14)": rsi_val,

            "RSI Signal": "✅" if rsi_sig else "—",

            "Vol Signal": "✅" if vol_sig else "—",

            "Trend Score": trend_sc,

            "Patterns": trend_desc,

            "P/E Ratio": get_pe(info) if info else None,

            "Strength": total,

            "_rsi_desc": rsi_desc,

            "_vol_desc": vol_desc,

        }

 

        if is_us:

            row["Analyst Rec."] = get_analyst_rec(info) if info else None

            row["Sector"] = sectors.get(ticker, "—") if sectors else "—"

 

        return row

    except Exception:

        return None

 

 

# ═══════════════════════════════════════════════════════════════

# BATCH SCANNER WITH PROGRESS

# ═══════════════════════════════════════════════════════════════

 

def run_scan(tickers, is_us, names, sectors, workers):

    results = []

    total = len(tickers)

 

    progress = st.progress(0)

    status = st.empty()

    counter = st.empty()

 

    done = 0

    found = 0

 

    with ThreadPoolExecutor(max_workers=workers) as pool:

        futures = {

            pool.submit(scan_stock, t, is_us, names, sectors): t

            for t in tickers

        }

        for f in as_completed(futures):

            done += 1

            try:

                r = f.result(timeout=30)

                if r:

                    results.append(r)

                    found += 1

            except Exception:

                pass

 

            progress.progress(done / total)

            status.text(f"⏳ Scanning: {done}/{total} tickers…")

            counter.text(f"🎯 Signals found: {found}")

 

    progress.progress(1.0)

    status.text(f"✅ Done — {total} tickers scanned")

    counter.text(f"🎯 Total signals: {found}")

    time.sleep(0.5)

    progress.empty()

    status.empty()

    counter.empty()

 

    return results

 

 

# ═══════════════════════════════════════════════════════════════

# RESULTS DISPLAY

# ═══════════════════════════════════════════════════════════════

 

def show_results(results, is_us):

    if not results:

        st.warning(

            "🔍 No bullish reversal signals found right now.\n\n"

            "This is normal — the scanner looks for specific technical setups "

            "that don't occur every day. Try again after a market pullback."

        )

        return

 

    df = pd.DataFrame(results).sort_values("Strength", ascending=False).reset_index(drop=True)

 

    # ── Summary metrics ──

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📊 Signals Found", len(df))

    c2.metric("📉 RSI Crossovers", int((df["RSI Signal"] == "✅").sum()))

    c3.metric("📊 Volume Surges", int((df["Vol Signal"] == "✅").sum()))

    c4.metric("🔥 Strong (≥5)", int((df["Strength"] >= 5).sum()))

    st.markdown("---")

 

    # ── Table ──

    st.subheader("📋 Results Table")

 

    cols = [

        "Ticker", "Company", "Price", "Change %", "RSI(14)",

        "RSI Signal", "Vol Signal", "Trend Score", "Patterns",

        "P/E Ratio", "Strength",

    ]

    if is_us:

        cols.insert(-1, "Analyst Rec.")

        cols.insert(-1, "Sector")

 

    show_cols = [c for c in cols if c in df.columns]

 

    st.dataframe(

        df[show_cols],

        use_container_width=True,

        hide_index=True,

        column_config={

            "Price": st.column_config.NumberColumn("Price", format="%.2f"),

            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),

            "RSI(14)": st.column_config.NumberColumn("RSI(14)", format="%.1f"),

            "Trend Score": st.column_config.ProgressColumn(

                "Trend", min_value=0, max_value=10, format="%d"

            ),

            "Strength": st.column_config.ProgressColumn(

                "Strength", min_value=0, max_value=15, format="%d"

            ),

            "P/E Ratio": st.column_config.NumberColumn("P/E", format="%.1f"),

        },

    )

 

    # ── Detail Cards ──

    st.markdown("---")

    st.subheader("🔎 Top Signal Details")

 

    for idx, row in df.head(min(10, len(df))).iterrows():

        icon = "🔥" if row["Strength"] >= 5 else "📊"

        with st.expander(

            f"{icon} {row['Ticker']} — {row['Company']}  |  "

            f"Strength: {row['Strength']}",

            expanded=(idx < 3),

        ):

            a, b, c = st.columns(3)

            with a:

                st.markdown("**💰 Price**")

                if row.get("Price"):

                    chg = row.get("Change %")

                    color = "🟢" if (chg or 0) >= 0 else "🔴"

                    st.write(f"${row['Price']}")

                    if chg is not None:

                        st.write(f"{color} {chg:+.2f}%")

                pe = row.get("P/E Ratio")

                if pe:

                    st.write(f"P/E: {pe}")

 

            with b:

                st.markdown("**📊 Signals**")

                st.write(f"RSI(14): **{row['RSI(14)']}** {row['RSI Signal']}")

                st.write(f"Volume: {row['Vol Signal']}")

                st.write(f"Trend Score: **{row['Trend Score']}** / 10")

 

            with c:

                st.markdown("**🔍 Patterns**")

                st.write(row.get("Patterns", "—"))

                if is_us:

                    rec = row.get("Analyst Rec.")

                    if rec:

                        st.write(f"Analyst: **{rec}**")

                    sec = row.get("Sector")

                    if sec and sec != "—":

                        st.write(f"Sector: {sec}")

 

    # ── Download ──

    st.markdown("---")

    csv = df.to_csv(index=False)

    st.download_button(

        "📥 Download CSV",

        csv,

        f"scan_{datetime.now():%Y%m%d_%H%M}.csv",

        "text/csv",

        use_container_width=True,

    )

 

 

# ═══════════════════════════════════════════════════════════════

# SIDEBAR

# ═══════════════════════════════════════════════════════════════

 

def sidebar():

    with st.sidebar:

        st.markdown("## ⚙️ Settings")

        st.markdown("---")

 

        market = st.selectbox(

            "🌍 Market",

            ["S&P 500 (US)", "TA-125 (Israel)"],

        )

 

        st.markdown("---")

        workers = st.slider("Parallel Threads", 3, 20, 8,

                            help="Higher = faster but may hit rate limits")

        min_str = st.slider("Min Signal Strength", 1, 10, 2)

 

        st.markdown("---")

        st.markdown("### 📖 Signal Guide")

        st.markdown("""

| Signal | Meaning |

|--------|---------|

| **RSI ✅** | RSI(14) crossed above 30 |

| **Vol ✅** | 3-day increasing or above avg |

| **Trend** | Pattern score 0-10 |

| **Strength** | Combined score |

        """)

 

        st.markdown("---")

        st.caption(f"Stock Scanner Pro v1.0 • {datetime.now():%Y-%m-%d %H:%M}")

 

        return market, workers, min_str

 

 

# ═══════════════════════════════════════════════════════════════

# MAIN

# ═══════════════════════════════════════════════════════════════

 

def main():

    st.markdown('<h1 class="main-header">📈 Stock Scanner Pro</h1>', unsafe_allow_html=True)

    st.markdown(

        '<p class="sub-header">Bullish reversal signals · RSI · Volume · Trend Patterns</p>',

        unsafe_allow_html=True,

    )

 

    market, workers, min_str = sidebar()

    is_us = "S&P 500" in market

    flag = "🇺🇸 S&P 500" if is_us else "🇮🇱 TA-125"

 

    left, right = st.columns([3, 1])

    left.markdown(f"### {flag} Scanner")

    go = right.button("🚀 Start Scan", type="primary", use_container_width=True)

 

    if go:

        st.markdown("---")

 

        with st.spinner(f"📡 Loading {flag} tickers…"):

            if is_us:

                tickers, names, sectors = get_sp500_tickers()

            else:

                tickers, names = get_ta125_tickers()

                sectors = {}

 

        if not tickers:

            st.error("❌ Could not load tickers. Check your connection and try again.")

            return

 

        st.success(f"✅ Loaded **{len(tickers)}** tickers")

        st.markdown("### 🔍 Scanning…")

 

        t0 = time.time()

        results = run_scan(tickers, is_us, names, sectors, workers)

        results = [r for r in results if r["Strength"] >= min_str]

        elapsed = time.time() - t0

 

        st.success(f"✅ Finished in **{elapsed:.1f}s** — **{len(results)}** signals found")

 

        show_results(results, is_us)

 

    else:

        # ── Landing page ──

        st.markdown("---")

        st.markdown("""

### 🎯 How It Works

 

This scanner checks every stock in the chosen index for **bullish reversal setups**:

 

| Signal | What It Detects | Score |

|--------|----------------|-------|

| **RSI Crossover** | RSI(14) crossing above 30 from oversold | ⭐⭐⭐ |

| **Volume Surge** | 3 days of rising volume or above 20-day avg | ⭐⭐ |

| **Bullish Engulfing** | Current candle engulfs previous bearish candle | ⭐⭐ |

| **Hammer** | Long lower shadow = buying pressure | ⭐⭐ |

| **MACD Cross** | MACD line crosses above signal line | ⭐⭐⭐ |

| **EMA Breakout** | Price crosses above EMA(20) | ⭐⭐ |

| **Bullish Divergence** | Price lower low + RSI higher low | ⭐⭐⭐ |

| **Morning Star** | 3-candle bullish reversal | ⭐⭐⭐ |

        """)

 

        st.markdown("---")

        a, b, c = st.columns(3)

        a.markdown("""

#### 🇺🇸 S&P 500

- 500+ large-cap US stocks

- P/E ratio + Analyst recs

- Sector breakdown

        """)

        b.markdown("""

#### 🇮🇱 TA-125

- 125 Israeli stocks

- P/E ratio included

- Tel Aviv Exchange (.TA)

        """)

        c.markdown("""

#### ⚡ Performance

- Multi-threaded (3-20)

- ~2-5 min full scan

- CSV export

- Smart caching

        """)

 

        st.markdown("---")

        st.info("👈 Pick a market in the sidebar and hit **🚀 Start Scan**")

 

    # ── Disclaimer ──

    st.markdown("---")

    st.caption(

        "⚠️ **Disclaimer**: This tool is for educational purposes only. "

        "It is not financial advice. Always do your own research."

    )

 

 

if __name__ == "__main__":

    main()
