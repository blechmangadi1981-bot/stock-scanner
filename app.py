"""

Stock Scanner Pro v2.1 — with Historical Backtesting

=====================================================

Fixed: Removed pandas_ta dependency (numba build failure).

All indicators calculated with pure pandas/numpy.

 

requirements.txt:

    streamlit>=1.28.0

    yfinance>=0.2.31

    pandas>=2.0.0

    numpy>=1.24.0

    lxml>=4.9.0

    requests>=2.31.0

    plotly>=5.18.0

"""

 

import streamlit as st

import pandas as pd

import yfinance as yf

import numpy as np

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

from datetime import datetime

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings

 

warnings.filterwarnings("ignore")

 

# ═══════════════════════════════════════════════════════════════

# PAGE CONFIG

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

        font-size: 2.5rem; font-weight: 700;

        background: linear-gradient(90deg, #00C853, #00BFA5);

        -webkit-background-clip: text; -webkit-text-fill-color: transparent;

        text-align: center; padding: 0.5rem 0;

    }

    .sub-header {

        text-align: center; font-size: 1.05rem;

        opacity: 0.7; margin-bottom: 2rem;

    }

    div[data-testid="stMetric"] {

        background: rgba(255,255,255,0.05);

        border: 1px solid rgba(255,255,255,0.1);

        border-radius: 10px; padding: 0.8rem;

    }

    #MainMenu {visibility: hidden;}

    footer {visibility: hidden;}

    .stDeployButton {display: none;}

</style>

""", unsafe_allow_html=True)

 

 

# ═══════════════════════════════════════════════════════════════

# PURE PANDAS/NUMPY INDICATOR FUNCTIONS (no pandas_ta / numba)

# ═══════════════════════════════════════════════════════════════

 

def calc_rsi(series, length=14):

    """Calculate RSI using pure pandas — no external lib needed."""

    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)

    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()

    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

 

 

def calc_ema(series, length=20):

    """Calculate EMA using pure pandas."""

    return series.ewm(span=length, adjust=False).mean()

 

 

def calc_macd(series, fast=12, slow=26, signal=9):

    """Calculate MACD line, signal line, histogram using pure pandas."""

    ema_fast = series.ewm(span=fast, adjust=False).mean()

    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

 

 

# ═══════════════════════════════════════════════════════════════

# TICKER LISTS

# ═══════════════════════════════════════════════════════════════

 

@st.cache_data(ttl=86400, show_spinner=False)

def get_sp500_tickers():

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

# DATA FETCHING

# ═══════════════════════════════════════════════════════════════

 

def fetch_stock_data(ticker, period="3mo"):

    try:

        stock = yf.Ticker(ticker)

        df = stock.history(period=period, auto_adjust=True)

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

 

 

@st.cache_data(ttl=3600, show_spinner=False)

def fetch_backtest_data(ticker, period="2y"):

    try:

        stock = yf.Ticker(ticker)

        df = stock.history(period=period, auto_adjust=True)

        if df is None or df.empty or len(df) < 60:

            return None

        return df

    except Exception:

        return None

 

 

# ═══════════════════════════════════════════════════════════════

# TECHNICAL ANALYSIS — LIVE SCANNER

# ═══════════════════════════════════════════════════════════════

 

def analyze_rsi(df):

    try:

        rsi = calc_rsi(df["Close"], 14)

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

    signals = []

    score = 0

    if len(df) < 26:

        return 0, [], "Insufficient data"

    close = df["Close"]

    opn = df["Open"]

    high = df["High"]

    low = df["Low"]

 

    # Bullish Engulfing

    try:

        if len(df) >= 2:

            po, pc = float(opn.iloc[-2]), float(close.iloc[-2])

            co, cc = float(opn.iloc[-1]), float(close.iloc[-1])

            if pc < po and cc > co and cc > po and co < pc:

                signals.append("Bullish Engulfing"); score += 2

    except Exception:

        pass

 

    # Hammer

    try:

        co, cc = float(opn.iloc[-1]), float(close.iloc[-1])

        ch, cl = float(high.iloc[-1]), float(low.iloc[-1])

        body = abs(cc - co)

        ls = min(co, cc) - cl

        us = ch - max(co, cc)

        if body > 0 and ls >= 2 * body and us <= body * 0.5:

            signals.append("Hammer"); score += 2

    except Exception:

        pass

 

    # MACD Crossover

    try:

        macd_line, signal_line, _ = calc_macd(close)

        ml = macd_line.dropna()

        sl = signal_line.dropna()

        if len(ml) >= 2 and len(sl) >= 2:

            pd_v = float(ml.iloc[-2]) - float(sl.iloc[-2])

            cd_v = float(ml.iloc[-1]) - float(sl.iloc[-1])

            if pd_v < 0 and cd_v > 0:

                signals.append("MACD Bullish Cross"); score += 3

            elif pd_v <= 0 < cd_v:

                signals.append("MACD Turning Bullish"); score += 1

    except Exception:

        pass

 

    # EMA(20) Breakout

    try:

        ema = calc_ema(close, 20)

        if ema is not None:

            ema = ema.dropna()

            if len(ema) >= 2:

                was_below = float(close.iloc[-2]) <= float(ema.iloc[-2])

                now_above = float(close.iloc[-1]) > float(ema.iloc[-1])

                if was_below and now_above:

                    signals.append("EMA(20) Breakout"); score += 2

                elif now_above:

                    score += 0.5

    except Exception:

        pass

 

    # Bullish Divergence

    try:

        rsi = calc_rsi(close, 14)

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

                            signals.append("Bullish Divergence"); score += 3

    except Exception:

        pass

 

    # Morning Star

    try:

        if len(df) >= 3:

            d1 = float(close.iloc[-3]) - float(opn.iloc[-3])

            d2 = abs(float(close.iloc[-2]) - float(opn.iloc[-2]))

            d3 = float(close.iloc[-1]) - float(opn.iloc[-1])

            if d1 < 0 and d2 < abs(d1) * 0.3 and d3 > 0 and d3 > abs(d1) * 0.5:

                signals.append("Morning Star"); score += 3

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

 

        price = change = None

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

 

 

def run_scan(tickers, is_us, names, sectors, workers):

    results = []

    total = len(tickers)

    progress = st.progress(0)

    status = st.empty()

    counter = st.empty()

    done = found = 0

 

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

# BACKTESTING ENGINE

# ═══════════════════════════════════════════════════════════════

 

def find_historical_signals(df):

    """Walk through history, fire same signals as live scanner."""

    if df is None or len(df) < 30:

        return []

 

    close = df["Close"]

    opn = df["Open"]

    high = df["High"]

    low = df["Low"]

 

    rsi_series = calc_rsi(close, 14)

    ema_series = calc_ema(close, 20)

    macd_line, signal_line, _ = calc_macd(close)

    vol_20_avg = df["Volume"].rolling(20).mean()

 

    signals_list = []

 

    for i in range(30, len(df)):

        day_signals = []

        score = 0

        idx = df.index[i]

        entry_price = float(close.iloc[i])

 

        # RSI cross above 30

        try:

            if rsi_series is not None and i >= 1:

                curr_rsi = rsi_series.iloc[i]

                prev_rsi = rsi_series.iloc[i - 1]

                if pd.notna(curr_rsi) and pd.notna(prev_rsi):

                    if prev_rsi <= 30 and curr_rsi > 30:

                        day_signals.append("RSI Crossover")

                        score += 3

        except Exception:

            pass

 

        # Volume increasing 3 days

        try:

            if i >= 2:

                v0 = float(df["Volume"].iloc[i - 2])

                v1 = float(df["Volume"].iloc[i - 1])

                v2 = float(df["Volume"].iloc[i])

                if v2 > v1 > v0:

                    day_signals.append("Volume Surge")

                    score += 2

                elif pd.notna(vol_20_avg.iloc[i]) and vol_20_avg.iloc[i] > 0:

                    if v2 > vol_20_avg.iloc[i] * 1.1:

                        day_signals.append("Volume Above Avg")

                        score += 1

        except Exception:

            pass

 

        # Bullish Engulfing

        try:

            if i >= 1:

                po, pc = float(opn.iloc[i - 1]), float(close.iloc[i - 1])

                co, cc = float(opn.iloc[i]), float(close.iloc[i])

                if pc < po and cc > co and cc > po and co < pc:

                    day_signals.append("Bullish Engulfing")

                    score += 2

        except Exception:

            pass

 

        # Hammer

        try:

            co, cc = float(opn.iloc[i]), float(close.iloc[i])

            ch, cl = float(high.iloc[i]), float(low.iloc[i])

            body = abs(cc - co)

            ls = min(co, cc) - cl

            us = ch - max(co, cc)

            if body > 0 and ls >= 2 * body and us <= body * 0.5:

                day_signals.append("Hammer")

                score += 2

        except Exception:

            pass

 

        # MACD Cross

        try:

            if macd_line is not None and signal_line is not None and i >= 1:

                ml_c = macd_line.iloc[i]

                ml_p = macd_line.iloc[i - 1]

                sl_c = signal_line.iloc[i]

                sl_p = signal_line.iloc[i - 1]

                if pd.notna(ml_c) and pd.notna(ml_p) and pd.notna(sl_c) and pd.notna(sl_p):

                    if (ml_p - sl_p) < 0 and (ml_c - sl_c) > 0:

                        day_signals.append("MACD Cross")

                        score += 3

        except Exception:

            pass

 

        # EMA Breakout

        try:

            if ema_series is not None and i >= 1:

                e_c = ema_series.iloc[i]

                e_p = ema_series.iloc[i - 1]

                if pd.notna(e_c) and pd.notna(e_p):

                    if float(close.iloc[i - 1]) <= e_p and float(close.iloc[i]) > e_c:

                        day_signals.append("EMA Breakout")

                        score += 2

        except Exception:

            pass

 

        # Morning Star

        try:

            if i >= 2:

                d1 = float(close.iloc[i - 2]) - float(opn.iloc[i - 2])

                d2 = abs(float(close.iloc[i - 1]) - float(opn.iloc[i - 1]))

                d3 = float(close.iloc[i]) - float(opn.iloc[i])

                if d1 < 0 and d2 < abs(d1) * 0.3 and d3 > 0 and d3 > abs(d1) * 0.5:

                    day_signals.append("Morning Star")

                    score += 3

        except Exception:

            pass

 

        if score >= 2 and day_signals:

            signals_list.append({

                "date": idx,

                "signals": day_signals,

                "score": score,

                "entry_price": entry_price,

            })

 

    return signals_list

 

 

def compute_backtest_outcomes(df, signals_list, hold_days_list=[5, 10, 20]):

    if not signals_list or df is None:

        return pd.DataFrame()

 

    trades = []

    close = df["Close"]

 

    for sig in signals_list:

        dt = sig["date"]

        entry = sig["entry_price"]

        try:

            loc = df.index.get_loc(dt)

        except Exception:

            continue

 

        trade = {

            "Signal Date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt),

            "Patterns": ", ".join(sig["signals"]),

            "Score": sig["score"],

            "Entry Price": round(entry, 2),

        }

 

        for days in hold_days_list:

            exit_idx = loc + days

            if exit_idx < len(close):

                exit_price = float(close.iloc[exit_idx])

                ret = ((exit_price / entry) - 1) * 100

                trade[f"{days}D Return %"] = round(ret, 2)

                trade[f"{days}D Exit"] = round(exit_price, 2)

                trade[f"{days}D Win"] = ret > 0

            else:

                trade[f"{days}D Return %"] = None

                trade[f"{days}D Exit"] = None

                trade[f"{days}D Win"] = None

 

        trades.append(trade)

 

    return pd.DataFrame(trades)

 

 

def compute_backtest_stats(trades_df, hold_days_list=[5, 10, 20]):

    if trades_df.empty:

        return {}

    stats = {}

    for days in hold_days_list:

        col = f"{days}D Return %"

        win_col = f"{days}D Win"

        if col not in trades_df.columns:

            continue

        valid = trades_df[trades_df[col].notna()].copy()

        if valid.empty:

            continue

        total = len(valid)

        wins = int(valid[win_col].sum())

        losses = total - wins

        win_rate = (wins / total) * 100 if total > 0 else 0

        avg_ret = float(valid[col].mean())

        median_ret = float(valid[col].median())

        best = float(valid[col].max())

        worst = float(valid[col].min())

        avg_win = float(valid[valid[win_col]][col].mean()) if wins > 0 else 0

        avg_loss = float(valid[~valid[win_col]][col].mean()) if losses > 0 else 0

        pf = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')

 

        stats[f"{days}-Day"] = {

            "Total Trades": total,

            "Wins": wins,

            "Losses": losses,

            "Win Rate %": round(win_rate, 1),

            "Avg Return %": round(avg_ret, 2),

            "Median Return %": round(median_ret, 2),

            "Best Trade %": round(best, 2),

            "Worst Trade %": round(worst, 2),

            "Avg Win %": round(avg_win, 2),

            "Avg Loss %": round(avg_loss, 2),

            "Profit Factor": round(pf, 2) if pf != float('inf') else "∞",

        }

    return stats

 

 

def backtest_by_pattern(trades_df, hold_days=10):

    col = f"{hold_days}D Return %"

    win_col = f"{hold_days}D Win"

    if col not in trades_df.columns:

        return pd.DataFrame()

    valid = trades_df[trades_df[col].notna()].copy()

    if valid.empty:

        return pd.DataFrame()

 

    all_patterns = set()

    for pats in valid["Patterns"]:

        for p in str(pats).split(", "):

            p = p.strip()

            if p:

                all_patterns.add(p)

 

    rows = []

    for pat in sorted(all_patterns):

        mask = valid["Patterns"].str.contains(pat, na=False)

        subset = valid[mask]

        if len(subset) == 0:

            continue

        total = len(subset)

        wins = int(subset[win_col].sum())

        wr = (wins / total) * 100 if total > 0 else 0

        avg_r = float(subset[col].mean())

        rows.append({

            "Pattern": pat,

            "Signals": total,

            "Wins": wins,

            "Win Rate %": round(wr, 1),

            "Avg Return %": round(avg_r, 2),

        })

    return pd.DataFrame(rows).sort_values("Win Rate %", ascending=False).reset_index(drop=True)

 

 

def build_equity_curve(trades_df, hold_days=10):

    col = f"{hold_days}D Return %"

    if col not in trades_df.columns:

        return pd.DataFrame()

    valid = trades_df[trades_df[col].notna()].copy()

    if valid.empty:

        return pd.DataFrame()

    valid = valid.sort_values("Signal Date").reset_index(drop=True)

 

    capital = 10000.0

    curve = [{"Trade #": 0, "Date": valid["Signal Date"].iloc[0], "Portfolio": capital}]

    for i, row in valid.iterrows():

        ret = float(row[col]) / 100

        capital *= (1 + ret)

        curve.append({

            "Trade #": i + 1,

            "Date": row["Signal Date"],

            "Portfolio": round(capital, 2),

        })

    return pd.DataFrame(curve)

 

 

def backtest_single_ticker(ticker, period="2y"):

    try:

        df = fetch_backtest_data(ticker, period)

        if df is None:

            return pd.DataFrame()

        sigs = find_historical_signals(df)

        if not sigs:

            return pd.DataFrame()

        trades = compute_backtest_outcomes(df, sigs)

        if not trades.empty:

            trades.insert(0, "Ticker", ticker)

        return trades

    except Exception:

        return pd.DataFrame()

 

 

# ═══════════════════════════════════════════════════════════════

# DISPLAY — LIVE SCANNER RESULTS

# ═══════════════════════════════════════════════════════════════

 

def show_results(results, is_us):

    if not results:

        st.warning("🔍 No bullish reversal signals found right now. Try again after a pullback.")

        return

 

    df = pd.DataFrame(results).sort_values("Strength", ascending=False).reset_index(drop=True)

 

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📊 Signals Found", len(df))

    c2.metric("📉 RSI Crossovers", int((df["RSI Signal"] == "✅").sum()))

    c3.metric("📊 Volume Surges", int((df["Vol Signal"] == "✅").sum()))

    c4.metric("🔥 Strong (≥5)", int((df["Strength"] >= 5).sum()))

    st.markdown("---")

 

    st.subheader("📋 Results Table")

    cols = ["Ticker", "Company", "Price", "Change %", "RSI(14)", "RSI Signal",

            "Vol Signal", "Trend Score", "Patterns", "P/E Ratio", "Strength"]

    if is_us:

        cols.insert(-1, "Analyst Rec.")

        cols.insert(-1, "Sector")

    show_cols = [c for c in cols if c in df.columns]

 

    st.dataframe(

        df[show_cols], use_container_width=True, hide_index=True,

        column_config={

            "Price": st.column_config.NumberColumn("Price", format="%.2f"),

            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),

            "RSI(14)": st.column_config.NumberColumn("RSI(14)", format="%.1f"),

            "Trend Score": st.column_config.ProgressColumn("Trend", min_value=0, max_value=10, format="%d"),

            "Strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=15, format="%d"),

            "P/E Ratio": st.column_config.NumberColumn("P/E", format="%.1f"),

        },

    )

 

    st.markdown("---")

    st.subheader("🔎 Top Signal Details")

    for idx, row in df.head(min(10, len(df))).iterrows():

        icon = "🔥" if row["Strength"] >= 5 else "📊"

        with st.expander(f"{icon} {row['Ticker']} — {row['Company']}  |  Strength: {row['Strength']}", expanded=(idx < 3)):

            a, b, c = st.columns(3)

            with a:

                st.markdown("**💰 Price**")

                if row.get("Price"):

                    chg = row.get("Change %")

                    clr = "🟢" if (chg or 0) >= 0 else "🔴"

                    st.write(f"${row['Price']}")

                    if chg is not None:

                        st.write(f"{clr} {chg:+.2f}%")

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

 

    st.markdown("---")

    csv = df.to_csv(index=False)

    st.download_button("📥 Download CSV", csv,

                        f"scan_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv",

                        use_container_width=True)

 

 

# ═══════════════════════════════════════════════════════════════

# DISPLAY — BACKTEST SINGLE TICKER

# ═══════════════════════════════════════════════════════════════

 

def show_backtest_single(ticker, period):

    with st.spinner(f"📡 Downloading {period} of data for {ticker}…"):

        df = fetch_backtest_data(ticker, period)

 

    if df is None:

        st.error(f"Could not fetch data for {ticker}.")

        return

 

    st.success(f"Loaded **{len(df)} trading days** for **{ticker}**")

 

    with st.spinner("🔍 Finding historical signals…"):

        sigs = find_historical_signals(df)

 

    if not sigs:

        st.warning(f"No historical signals found for {ticker} in this period.")

        return

 

    trades = compute_backtest_outcomes(df, sigs)

    if trades.empty:

        st.warning("Could not compute trade outcomes.")

        return

 

    st.info(f"Found **{len(trades)} historical signals** for {ticker}")

 

    # Summary Stats

    st.markdown("### 📊 Performance Summary")

    stats = compute_backtest_stats(trades)

 

    if stats:

        stat_cols = st.columns(len(stats))

        for i, (period_name, s) in enumerate(stats.items()):

            with stat_cols[i]:

                wr = s["Win Rate %"]

                st.metric(

                    f"🎯 {period_name} Win Rate", f"{wr}%",

                    delta=f"Avg: {s['Avg Return %']:+.2f}%",

                    delta_color="normal" if s['Avg Return %'] >= 0 else "inverse"

                )

                st.caption(f"Trades: {s['Total Trades']} | W:{s['Wins']} L:{s['Losses']}")

                st.caption(f"Best: {s['Best Trade %']:+.2f}% | Worst: {s['Worst Trade %']:+.2f}%")

                st.caption(f"Profit Factor: {s['Profit Factor']}")

 

    if stats:

        st.markdown("### 📈 Detailed Statistics")

        stats_df = pd.DataFrame(stats).T

        stats_df.index.name = "Holding Period"

        st.dataframe(stats_df, use_container_width=True)

 

    # Pattern Breakdown

    st.markdown("### 🏷️ Performance by Pattern (10-Day Hold)")

    pat_df = backtest_by_pattern(trades, hold_days=10)

    if not pat_df.empty:

        st.dataframe(pat_df, use_container_width=True, hide_index=True,

                      column_config={

                          "Win Rate %": st.column_config.ProgressColumn("Win Rate %", min_value=0, max_value=100, format="%.1f%%"),

                          "Avg Return %": st.column_config.NumberColumn("Avg Return %", format="%.2f%%"),

                      })

        fig_pat = px.bar(pat_df, x="Pattern", y="Win Rate %", color="Avg Return %",

                         color_continuous_scale=["#FF5252", "#FFD600", "#00C853"],

                         title="Win Rate by Pattern Type", text="Win Rate %")

        fig_pat.update_layout(template="plotly_dark", height=400)

        fig_pat.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")

        st.plotly_chart(fig_pat, use_container_width=True)

 

    # Equity Curve

    st.markdown("### 💰 Equity Curve (10-Day Hold, $10K Start)")

    eq = build_equity_curve(trades, hold_days=10)

    if not eq.empty:

        final_val = eq["Portfolio"].iloc[-1]

        total_ret = ((final_val / 10000) - 1) * 100

        ec1, ec2, ec3 = st.columns(3)

        ec1.metric("Starting Capital", "$10,000")

        ec2.metric("Final Value", f"${final_val:,.2f}")

        ec3.metric("Total Return", f"{total_ret:+.2f}%")

 

        fig_eq = go.Figure()

        fig_eq.add_trace(go.Scatter(

            x=eq["Trade #"], y=eq["Portfolio"], mode="lines+markers",

            name="Portfolio Value",

            line=dict(color="#00C853" if total_ret >= 0 else "#FF5252", width=2),

            marker=dict(size=4),

            fill="tozeroy",

            fillcolor="rgba(0,200,83,0.1)" if total_ret >= 0 else "rgba(255,82,82,0.1)",

        ))

        fig_eq.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Starting Capital")

        fig_eq.update_layout(template="plotly_dark", height=400,

                              title="Portfolio Growth per Trade",

                              xaxis_title="Trade #", yaxis_title="Portfolio Value ($)")

        st.plotly_chart(fig_eq, use_container_width=True)

 

    # Return Distribution

    st.markdown("### 📊 Return Distribution (10-Day Hold)")

    if "10D Return %" in trades.columns:

        valid_rets = trades["10D Return %"].dropna()

        if len(valid_rets) > 0:

            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(x=valid_rets, nbinsx=30, name="Returns",

                                             marker_color="#00BFA5"))

            fig_dist.add_vline(x=0, line_dash="dash", line_color="white", line_width=2)

            fig_dist.add_vline(x=float(valid_rets.mean()), line_dash="dot", line_color="#FFD600",

                               annotation_text=f"Avg: {valid_rets.mean():.2f}%")

            fig_dist.update_layout(template="plotly_dark", height=350,

                                    title="Distribution of 10-Day Returns After Signal",

                                    xaxis_title="Return %", yaxis_title="Count")

            st.plotly_chart(fig_dist, use_container_width=True)

 

    # Price Chart with Signals

    st.markdown("### 📉 Price Chart with Signal Markers")

    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,

                               row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig_price.add_trace(go.Candlestick(

        x=df.index, open=df["Open"], high=df["High"],

        low=df["Low"], close=df["Close"], name="Price",

    ), row=1, col=1)

 

    sig_dates = [s["date"] for s in sigs]

    sig_prices = [s["entry_price"] for s in sigs]

    sig_labels = [", ".join(s["signals"]) for s in sigs]

    fig_price.add_trace(go.Scatter(

        x=sig_dates, y=sig_prices, mode="markers",

        marker=dict(symbol="triangle-up", size=12, color="#00C853",

                    line=dict(width=1, color="white")),

        name="Signal", text=sig_labels,

        hovertemplate="%{text}<br>Price: %{y:.2f}<extra></extra>",

    ), row=1, col=1)

 

    colors = ["#00C853" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#FF5252"

              for i in range(len(df))]

    fig_price.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",

                                marker_color=colors, opacity=0.5), row=2, col=1)

    fig_price.update_layout(template="plotly_dark", height=600,

                              title=f"{ticker} — Price & Signals",

                              xaxis_rangeslider_visible=False, showlegend=True)

    fig_price.update_yaxes(title_text="Price", row=1, col=1)

    fig_price.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig_price, use_container_width=True)

 

    # Trades Table

    st.markdown("### 📋 All Historical Trades")

    drop_cols = [c for c in trades.columns if c.endswith("D Win") or c.endswith("D Exit")]

    display_trades = trades.drop(columns=drop_cols, errors="ignore").copy()

    for days in [5, 10, 20]:

        col = f"{days}D Return %"

        if col in display_trades.columns:

            display_trades[col] = display_trades[col].apply(

                lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"

            )

    st.dataframe(display_trades, use_container_width=True, hide_index=True)

 

    csv_bt = trades.to_csv(index=False)

    st.download_button("📥 Download Backtest CSV", csv_bt,

                        f"backtest_{ticker}_{datetime.now():%Y%m%d}.csv", "text/csv",

                        use_container_width=True)

 

 

# ═══════════════════════════════════════════════════════════════

# DISPLAY — BACKTEST FULL MARKET

# ═══════════════════════════════════════════════════════════════

 

def show_backtest_market(tickers, names, workers, sample_size):

    import random

 

    if len(tickers) > sample_size:

        sample = random.sample(tickers, sample_size)

        st.info(f"Sampling **{sample_size}** tickers from {len(tickers)} for speed.")

    else:

        sample = tickers

 

    all_trades = []

    progress = st.progress(0)

    status = st.empty()

    done = 0

    total = len(sample)

 

    with ThreadPoolExecutor(max_workers=workers) as pool:

        futures = {pool.submit(backtest_single_ticker, t, "2y"): t for t in sample}

        for f in as_completed(futures):

            done += 1

            try:

                tr = f.result(timeout=60)

                if not tr.empty:

                    all_trades.append(tr)

            except Exception:

                pass

            progress.progress(done / total)

            status.text(f"⏳ Backtesting: {done}/{total} tickers…")

 

    progress.progress(1.0)

    status.text(f"✅ Backtested {total} tickers")

    time.sleep(0.5)

    progress.empty()

    status.empty()

 

    if not all_trades:

        st.warning("No historical signals found across sampled tickers.")

        return

 

    combined = pd.concat(all_trades, ignore_index=True)

    st.success(f"📊 Found **{len(combined)} total signals** across **{len(all_trades)} tickers**")

 

    # Aggregate Stats

    st.markdown("### 📊 Aggregate Backtest Performance")

    stats = compute_backtest_stats(combined)

 

    if stats:

        stat_cols = st.columns(len(stats))

        for i, (pname, s) in enumerate(stats.items()):

            with stat_cols[i]:

                wr = s["Win Rate %"]

                st.metric(f"🎯 {pname} Win Rate", f"{wr}%",

                          delta=f"Avg: {s['Avg Return %']:+.2f}%",

                          delta_color="normal" if s['Avg Return %'] >= 0 else "inverse")

                st.caption(f"Trades: {s['Total Trades']} | W:{s['Wins']} L:{s['Losses']}")

                st.caption(f"Profit Factor: {s['Profit Factor']}")

 

    if stats:

        stats_df = pd.DataFrame(stats).T

        stats_df.index.name = "Holding Period"

        st.dataframe(stats_df, use_container_width=True)

 

    # Pattern Breakdown

    st.markdown("### 🏷️ Win Rate by Pattern Type (10-Day Hold)")

    pat_df = backtest_by_pattern(combined, hold_days=10)

    if not pat_df.empty:

        st.dataframe(pat_df, use_container_width=True, hide_index=True,

                      column_config={

                          "Win Rate %": st.column_config.ProgressColumn("Win Rate %", min_value=0, max_value=100, format="%.1f%%"),

                      })

        fig_pat = px.bar(pat_df, x="Pattern", y="Win Rate %", color="Avg Return %",

                         color_continuous_scale=["#FF5252", "#FFD600", "#00C853"],

                         title="Aggregate Win Rate by Pattern", text="Win Rate %")

        fig_pat.update_layout(template="plotly_dark", height=400)

        fig_pat.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")

        st.plotly_chart(fig_pat, use_container_width=True)

 

    # Equity Curve

    st.markdown("### 💰 Aggregate Equity Curve (10-Day Hold)")

    eq = build_equity_curve(combined, hold_days=10)

    if not eq.empty:

        final_val = eq["Portfolio"].iloc[-1]

        total_ret = ((final_val / 10000) - 1) * 100

        ec1, ec2, ec3 = st.columns(3)

        ec1.metric("Starting Capital", "$10,000")

        ec2.metric("Final Value", f"${final_val:,.2f}")

        ec3.metric("Total Return", f"{total_ret:+.2f}%")

 

        fig_eq = go.Figure()

        fig_eq.add_trace(go.Scatter(

            x=eq["Trade #"], y=eq["Portfolio"], mode="lines", name="Portfolio",

            line=dict(color="#00C853" if total_ret >= 0 else "#FF5252", width=2),

            fill="tozeroy",

            fillcolor="rgba(0,200,83,0.1)" if total_ret >= 0 else "rgba(255,82,82,0.1)",

        ))

        fig_eq.add_hline(y=10000, line_dash="dash", line_color="gray")

        fig_eq.update_layout(template="plotly_dark", height=400,

                              title=f"Portfolio Growth Over {len(combined)} Trades",

                              xaxis_title="Trade #", yaxis_title="Value ($)")

        st.plotly_chart(fig_eq, use_container_width=True)

 

    # Distribution

    st.markdown("### 📊 Return Distribution")

    if "10D Return %" in combined.columns:

        rets = combined["10D Return %"].dropna()

        if len(rets) > 0:

            fig_d = go.Figure()

            fig_d.add_trace(go.Histogram(x=rets, nbinsx=40, name="10D Returns",

                                          marker_color="#00BFA5"))

            fig_d.add_vline(x=0, line_dash="dash", line_color="white")

            fig_d.add_vline(x=float(rets.mean()), line_dash="dot", line_color="#FFD600",

                            annotation_text=f"Avg: {rets.mean():.2f}%")

            fig_d.update_layout(template="plotly_dark", height=350,

                                 title="10-Day Return Distribution (All Signals)")

            st.plotly_chart(fig_d, use_container_width=True)

 

    # Top Tickers

    st.markdown("### 🏆 Top Tickers by Signal Performance")

    if "Ticker" in combined.columns and "10D Return %" in combined.columns:

        ticker_perf = combined.groupby("Ticker").agg(

            Signals=("10D Return %", "count"),

            Avg_Return=("10D Return %", "mean"),

            Win_Rate=("10D Win", lambda x: x.mean() * 100),

        ).round(2).sort_values("Avg_Return", ascending=False).head(20)

        ticker_perf.columns = ["# Signals", "Avg 10D Return %", "Win Rate %"]

        st.dataframe(ticker_perf, use_container_width=True)

 

    csv_all = combined.to_csv(index=False)

    st.download_button("📥 Download All Backtest Data", csv_all,

                        f"backtest_market_{datetime.now():%Y%m%d}.csv", "text/csv",

                        use_container_width=True)

 

 

# ═══════════════════════════════════════════════════════════════

# SIDEBAR

# ═══════════════════════════════════════════════════════════════

 

def sidebar():

    with st.sidebar:

        st.markdown("## ⚙️ Settings")

        st.markdown("---")

 

        mode = st.radio("📌 Mode", ["🔍 Live Scanner", "📊 Backtest"], horizontal=True)

        st.markdown("---")

        market = st.selectbox("🌍 Market", ["S&P 500 (US)", "TA-125 (Israel)"])

        st.markdown("---")

        workers = st.slider("Parallel Threads", 3, 20, 8)

        min_str = st.slider("Min Signal Strength", 1, 10, 2)

 

        backtest_opts = {}

        if "Backtest" in mode:

            st.markdown("---")

            st.markdown("### 📊 Backtest Settings")

            backtest_opts["type"] = st.radio(

                "Backtest Type", ["Single Ticker", "Full Market"],

                help="Single = deep dive one stock. Market = aggregate accuracy."

            )

            if backtest_opts["type"] == "Single Ticker":

                backtest_opts["ticker"] = st.text_input(

                    "Ticker Symbol", "AAPL",

                    help="Enter any ticker (e.g., AAPL, MSFT, TEVA.TA)"

                )

            backtest_opts["period"] = st.selectbox(

                "History Period", ["1y", "2y", "5y", "max"], index=1

            )

            if backtest_opts.get("type") == "Full Market":

                backtest_opts["sample_size"] = st.slider(

                    "Sample Size (tickers)", 10, 200, 50,

                    help="More tickers = slower but more reliable"

                )

 

        st.markdown("---")

        st.markdown("### 📖 Signal Guide")

        st.markdown("""

| Signal | Meaning |

|--------|---------|

| **RSI ✅** | RSI(14) crossed above 30 |

| **Vol ✅** | 3-day increasing or above avg |

| **Trend** | Pattern score 0–10 |

| **Strength** | Combined score |

        """)

        st.markdown("---")

        st.caption(f"Stock Scanner Pro v2.1 • {datetime.now():%Y-%m-%d %H:%M}")

 

        return mode, market, workers, min_str, backtest_opts

 

 

# ═══════════════════════════════════════════════════════════════

# LANDING PAGE

# ═══════════════════════════════════════════════════════════════

 

def show_landing():

    st.markdown("---")

    st.markdown("""

### 🎯 How It Works

 

| Signal | What It Detects | Score |

|--------|----------------|-------|

| **RSI Crossover** | RSI(14) crossing above 30 from oversold | ⭐⭐⭐ |

| **Volume Surge** | 3 days of rising volume or above 20-day avg | ⭐⭐ |

| **Bullish Engulfing** | Candle engulfs previous bearish candle | ⭐⭐ |

| **Hammer** | Long lower shadow = buying pressure | ⭐⭐ |

| **MACD Cross** | MACD crosses above signal line | ⭐⭐⭐ |

| **EMA Breakout** | Price crosses above EMA(20) | ⭐⭐ |

| **Bullish Divergence** | Price lower low + RSI higher low | ⭐⭐⭐ |

| **Morning Star** | 3-candle bullish reversal | ⭐⭐⭐ |

 

### 📊 Historical Backtesting

 

Switch to **Backtest mode** in the sidebar to:

- 🔬 **Single Ticker**: Deep-dive backtest on any stock

- 🌍 **Full Market**: Aggregate accuracy across 50–200 tickers

- 📈 Win rates, equity curves, and pattern breakdowns

    """)

 

    st.markdown("---")

    a, b, c = st.columns(3)

    a.markdown("#### 🇺🇸 S&P 500\n500+ US stocks · P/E · Analyst recs")

    b.markdown("#### 🇮🇱 TA-125\n125 Israeli stocks · Tel Aviv Exchange")

    c.markdown("#### ⚡ Performance\nMulti-threaded · Cached · CSV export")

    st.markdown("---")

    st.info("👈 Pick **Live Scanner** or **Backtest** mode in the sidebar to begin!")

 

 

# ═══════════════════════════════════════════════════════════════

# MAIN

# ═══════════════════════════════════════════════════════════════

 

def main():

    st.markdown('<h1 class="main-header">📈 Stock Scanner Pro</h1>', unsafe_allow_html=True)

    st.markdown(

        '<p class="sub-header">Bullish Reversal Signals · RSI · Volume · Trend · Historical Backtesting</p>',

        unsafe_allow_html=True,

    )

 

    mode, market, workers, min_str, bt_opts = sidebar()

    is_us = "S&P 500" in market

    flag = "🇺🇸 S&P 500" if is_us else "🇮🇱 TA-125"

 

    # ── LIVE SCANNER ──

    if "Live" in mode:

        left, right = st.columns([3, 1])

        left.markdown(f"### {flag} Live Scanner")

        go_btn = right.button("🚀 Start Scan", type="primary", use_container_width=True)

 

        if go_btn:

            st.markdown("---")

            with st.spinner(f"📡 Loading {flag} tickers…"):

                if is_us:

                    tickers, names, sectors = get_sp500_tickers()

                else:

                    tickers, names = get_ta125_tickers()

                    sectors = {}

            if not tickers:

                st.error("❌ Could not load tickers.")

                return

            st.success(f"✅ Loaded **{len(tickers)}** tickers")

            st.markdown("### 🔍 Scanning…")

            t0 = time.time()

            results = run_scan(tickers, is_us, names, sectors, workers)

            results = [r for r in results if r["Strength"] >= min_str]

            elapsed = time.time() - t0

            st.success(f"✅ Finished in **{elapsed:.1f}s** — **{len(results)}** signals")

            show_results(results, is_us)

        else:

            show_landing()

 

    # ── BACKTEST ──

    else:

        st.markdown(f"### 📊 Historical Backtesting — {flag}")

        st.markdown(

            "Test how accurate the scanner's signals have been historically. "

            "Every signal that **would have fired** in the past is evaluated by "

            "measuring the stock's performance 5, 10, and 20 trading days later."

        )

        st.markdown("---")

 

        if bt_opts.get("type") == "Single Ticker":

            ticker = bt_opts.get("ticker", "AAPL").strip().upper()

            bt_period = bt_opts.get("period", "2y")

            go_bt = st.button(f"🔬 Backtest {ticker}", type="primary", use_container_width=True)

            if go_bt:

                show_backtest_single(ticker, bt_period)

 

        elif bt_opts.get("type") == "Full Market":

            bt_period = bt_opts.get("period", "2y")

            sample_size = bt_opts.get("sample_size", 50)

            go_bt = st.button(f"🔬 Backtest {flag} ({sample_size} tickers)",

                              type="primary", use_container_width=True)

            if go_bt:

                with st.spinner(f"📡 Loading {flag} tickers…"):

                    if is_us:

                        tickers, names, sectors = get_sp500_tickers()

                    else:

                        tickers, names = get_ta125_tickers()

                        sectors = {}

                if not tickers:

                    st.error("❌ Could not load tickers.")

                    return

                show_backtest_market(tickers, names, workers, sample_size)

 

    # Disclaimer

    st.markdown("---")

    st.caption(

        "⚠️ **Disclaimer**: Educational purposes only. "

        "Past performance does not guarantee future results. Not financial advice."

    )

 

 

if __name__ == "__main__":

    main()
