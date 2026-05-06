"""
demo_app.py — Streamlit demo for the LMW (2000) pattern detector + CNN.

Run with:
    streamlit run demo_app.py

Pulls live OHLCV from yfinance, renders the 3-channel candlestick image used
by ChartCNNv2, scans for the 10 LMW patterns, and (if best_model_v2.pt is
present) reports the CNN's 5-day up-probability on the most recent 60-day
window.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from image_v2 import make_candlestick_image
from lmw_patterns import PATTERN_NAMES, WINDOW_LENGTH, scan_series_for_patterns


LOOKBACK = 60
HORIZON = 5
MODEL_PATH = 'best_model_v2.pt'

PATTERN_INFO = {
    'HS':   ('Head & Shoulders',
             'Three peaks; middle is highest. Classical bearish reversal.'),
    'IHS':  ('Inverse Head & Shoulders',
             'Three troughs; middle is lowest. Bullish reversal.'),
    'BTOP': ('Broadening Top',
             'Higher peaks and lower troughs from a market top.'),
    'BBOT': ('Broadening Bottom',
             'Lower troughs and higher peaks from a market bottom.'),
    'TTOP': ('Triangle Top',
             'Lower highs and higher lows converging from a peak.'),
    'TBOT': ('Triangle Bottom',
             'Higher lows and lower highs converging from a trough.'),
    'RTOP': ('Rectangle Top',
             'Flat resistance and support after an uptrend; consolidation.'),
    'RBOT': ('Rectangle Bottom',
             'Flat resistance and support after a downtrend; consolidation.'),
    'DTOP': ('Double Top',
             'Two peaks at similar height; bearish reversal candidate.'),
    'DBOT': ('Double Bottom',
             'Two troughs at similar depth; bullish reversal candidate.'),
}

PATTERN_COLORS = {
    'HS':   '#C0392B', 'IHS':  '#27AE60',
    'BTOP': '#E67E22', 'BBOT': '#16A085',
    'TTOP': '#D35400', 'TBOT': '#1ABC9C',
    'RTOP': '#8E44AD', 'RBOT': '#3498DB',
    'DTOP': '#922B21', 'DBOT': '#1E8449',
}

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
})


# ---------------------------------------------------------------------------
# Data + model helpers (testable independently of Streamlit)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance; return a DataFrame with the columns
    image_v2 / lmw_patterns expect."""
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in raw.columns for c in cols):
        return pd.DataFrame()
    df = raw[cols].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < WINDOW_LENGTH:
        return pd.DataFrame(columns=['date', 'pattern_type', 'completion_idx'])
    closes = df['Close'].copy()
    closes.index.name = 'date'
    hits = scan_series_for_patterns(closes, window=WINDOW_LENGTH, stride=1)
    if hits.empty:
        return hits
    log_close = np.log(df['Close'].to_numpy(dtype=float))
    fwd = np.full(len(df), np.nan)
    fwd[: -HORIZON] = log_close[HORIZON:] - log_close[: -HORIZON]
    fwd_series = pd.Series(fwd, index=df.index)
    hits = hits.copy()
    hits['fwd_return_5d'] = hits['date'].map(fwd_series)
    return hits


def render_chart_image(df60: pd.DataFrame) -> np.ndarray:
    """Generate the 3x96x180 image and return as (96, 180, 3) for display."""
    img = make_candlestick_image(df60)
    return np.transpose(img, (1, 2, 0))


def make_chart_image_figure(df60: pd.DataFrame, ticker: str) -> plt.Figure:
    """Wrap the 96x180 chart image in a matplotlib figure that matches the
    price chart's dimensions / typography for a unified look."""
    img = render_chart_image(df60)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(img, aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{ticker} — last {LOOKBACK} days as 3-channel CNN input "
                 f"image (96×180)", fontweight='bold')
    ax.set_xlabel("R = candles  ·  G = 20-day moving average  ·  B = volume",
                  fontsize=9, color='#555')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#999')
        spine.set_linewidth(0.6)
    fig.tight_layout()
    return fig


def cnn_predict_latest(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """Returns prediction dict, or {'status': '...'} explaining why
    inference is unavailable. Single shape so the caller branches once."""
    if not os.path.exists(model_path):
        return {'status': f"`{model_path}` not found in current directory. "
                          f"Place the trained checkpoint here to enable inference."}
    try:
        import torch
        from train_cnn import ChartCNNv2  # noqa: F401
    except ImportError:
        return {'status': "PyTorch not installed in this environment — "
                          "install with `pip install torch` (CPU-only is fine)."}
    if len(df) < LOOKBACK:
        return {'status': f'Need ≥ {LOOKBACK} days of data; got {len(df)}.'}

    window = df.iloc[-LOOKBACK:][['Open', 'High', 'Low', 'Close', 'Volume']]
    img = make_candlestick_image(window).astype(np.float32) / 255.0

    device = torch.device('cpu')
    model = ChartCNNv2(in_ch=3, h=96, w=180)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return {
        'prob_down': float(probs[0]),
        'prob_up':   float(probs[1]),
        'window_start': df.index[-LOOKBACK],
        'window_end':   df.index[-1],
    }


def make_price_with_patterns(df: pd.DataFrame, hits: pd.DataFrame,
                             ticker: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df['Close'], color='#2C3E50', lw=1.3)

    if not hits.empty:
        seen = set()
        for _, row in hits.iterrows():
            d = pd.to_datetime(row['date'])
            pat = row['pattern_type']
            color = PATTERN_COLORS.get(pat, 'gray')
            label = pat if pat not in seen else None
            seen.add(pat)
            ax.axvline(d, color=color, alpha=0.30, lw=1.1, label=label)

    ax.set_title(f"{ticker} — close with LMW pattern completion dates",
                 fontweight='bold')
    ax.set_ylabel('Price ($)')
    if not hits.empty:
        ax.legend(ncol=5, fontsize=8, loc='upper left', frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _glossary():
    st.subheader("Pattern glossary (LMW 2000)")
    st.caption(
        "Definitions follow Lo, Mamaysky, Wang (2000), Section II.A. Smoothing "
        "uses a Gaussian-kernel approximation of the original Nadaraya-Watson "
        "smoother — see lmw_patterns.find_extrema."
    )
    cols = st.columns(2)
    for i, (code, (name, desc)) in enumerate(PATTERN_INFO.items()):
        with cols[i % 2]:
            color = PATTERN_COLORS.get(code, 'gray')
            st.markdown(
                f"<span style='color:{color}; font-weight:bold; "
                f"font-family:monospace'>{code:<5}</span> — "
                f"**{name}**. {desc}",
                unsafe_allow_html=True,
            )
    st.warning(
        "Research demonstration only. Not financial advice. Patterns are "
        "noisy at the single-stock level and have small effect sizes even "
        "where statistically significant in the broader CRSP universe."
    )


def main():
    st.set_page_config(page_title="LMW Pattern Detector",
                       layout="wide", page_icon="📊")
    st.title("Chart Pattern Detector — LMW (2000) + CNN")
    st.caption("15.C51 demo · Lo–Mamaysky–Wang smoothing + ChartCNNv2 inference")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
        end_default = date.today()
        start_default = end_default - timedelta(days=180)  # ~125 trading days
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start", start_default)
        with c2:
            end_date = st.date_input("End", end_default)
        analyze = st.button("Analyze", type='primary', use_container_width=True)
        st.divider()
        st.caption(
            "Patterns are scanned over a 38-day rolling window "
            "(LMW: 35 main + 3 detection lag)."
        )

    if not analyze:
        st.info("Enter a ticker in the sidebar and click **Analyze**.")
        _glossary()
        return

    if not ticker:
        st.error("Please enter a ticker symbol.")
        return
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    with st.spinner(f"Downloading {ticker} from yfinance..."):
        df = fetch_ohlcv(ticker, start_date.isoformat(), end_date.isoformat())
    if df.empty:
        st.error(f"No OHLCV returned for {ticker} between "
                 f"{start_date} and {end_date}.")
        return
    if len(df) < WINDOW_LENGTH:
        st.error(f"Need at least {WINDOW_LENGTH} trading days; got {len(df)}.")
        return

    # ---- Headline metrics
    last_close = float(df['Close'].iloc[-1])
    period_return = float(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)
    log_returns = np.log(df['Close']).diff().dropna()
    period_vol = float(log_returns.std() * np.sqrt(252))

    m = st.columns(4)
    m[0].metric("Trading days", f"{len(df)}")
    m[1].metric("Last close", f"${last_close:,.2f}")
    m[2].metric("Period return", f"{period_return:+.1%}")
    m[3].metric("Annualized vol", f"{period_vol:.1%}")

    with st.spinner("Scanning for LMW patterns..."):
        hits = detect_patterns(df)

    # ---- Stacked figures: same width, same matplotlib style
    st.markdown("")  # spacer
    st.markdown("### Price history with detected pattern completions")
    st.pyplot(make_price_with_patterns(df, hits, ticker),
              use_container_width=True)

    if len(df) >= LOOKBACK:
        st.markdown("")
        st.markdown("### Most recent 60-day window — CNN input image")
        window60 = df.iloc[-LOOKBACK:][['Open', 'High', 'Low', 'Close', 'Volume']]
        st.pyplot(make_chart_image_figure(window60, ticker),
                  use_container_width=True)
    else:
        st.info(f"Need ≥ {LOOKBACK} days for the chart image.")

    # ---- Pattern breakdown
    st.subheader("Pattern breakdown")
    if hits.empty:
        st.info("No LMW patterns detected in this window.")
    else:
        counts = hits['pattern_type'].value_counts().reindex(PATTERN_NAMES,
                                                             fill_value=0)
        cols = st.columns(5)
        for i, (pat, n) in enumerate(counts.items()):
            cols[i % 5].metric(pat, int(n))

        st.markdown(f"**Total detections: {len(hits)}**")
        view = hits.copy()
        view['date'] = pd.to_datetime(view['date']).dt.date
        view['fwd_return_5d'] = view['fwd_return_5d'].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
        )
        view = view[['date', 'pattern_type', 'fwd_return_5d']]
        st.dataframe(view.sort_values('date', ascending=False),
                     use_container_width=True, hide_index=True)

    # ---- CNN prediction
    st.divider()
    st.subheader("CNN prediction — next 5 trading days")
    res = cnn_predict_latest(df, MODEL_PATH)
    if 'status' in res:
        st.info(res['status'])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("P(up over 5d)",   f"{res['prob_up']:.1%}")
        c2.metric("P(down over 5d)", f"{res['prob_down']:.1%}")
        c3.write(f"**Window:**  \n{res['window_start'].date()} → "
                 f"{res['window_end'].date()}")

    st.divider()
    _glossary()


if __name__ == '__main__':
    main()
