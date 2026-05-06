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
    'HS':   '#E74C3C', 'IHS':  '#2ECC71',
    'BTOP': '#F39C12', 'BBOT': '#1ABC9C',
    'TTOP': '#FF6B6B', 'TBOT': '#48DBFB',
    'RTOP': '#9B59B6', 'RBOT': '#5DADE2',
    'DTOP': '#FF7979', 'DBOT': '#26C281',
}

TOP_PATTERNS = {'HS', 'DTOP', 'RTOP', 'BTOP', 'TTOP'}     # bearish-side patterns
BOT_PATTERNS = {'IHS', 'DBOT', 'RBOT', 'BBOT', 'TBOT'}    # bullish-side patterns

DARK_BG    = '#0E1117'   # Streamlit default dark bg
DARK_PANEL = '#161A22'
DARK_FG    = '#E0E0E0'
DARK_GRID  = '#2C313A'
DARK_AXIS  = '#5A6068'

EXAMPLE_TICKERS = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
GITHUB_REPO = 'github.com/USERNAME/REPO_NAME'  # update before publishing
COURSE_TAG  = '15.C51 Spring 2026'
CNN_DISCLAIMER = (
    "CNN test accuracy is 52% out-of-sample, only marginally above the "
    "52.5% majority-class baseline. Treat probabilities as model output, "
    "not calibrated forecasts."
)

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
})


def _apply_dark_style(fig: plt.Figure, ax: plt.Axes):
    """Match the Streamlit dark theme — dark bg, light gridlines + text."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_PANEL)
    ax.tick_params(colors=DARK_FG, which='both')
    for spine in ax.spines.values():
        spine.set_color(DARK_AXIS)
    ax.grid(color=DARK_GRID, alpha=0.9, linewidth=0.5)
    ax.title.set_color(DARK_FG)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)


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
    """Wrap the 96×180 chart image in a matplotlib figure matching the price
    chart's dimensions / typography. Source image is upscaled with hard pixel
    boundaries (np.repeat) so the model's input grid is visible at display
    size, even after Streamlit re-rasterizes."""
    img = render_chart_image(df60)
    img_big = np.repeat(np.repeat(img, 6, axis=0), 6, axis=1)  # 576×1080
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    _apply_dark_style(fig, ax)
    ax.imshow(img_big, aspect='auto', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{ticker} — last {LOOKBACK} days as 3-channel CNN input "
                 f"image  (96 × 180, upscaled 6×)", fontweight='bold')
    ax.set_xlabel("R = candles  ·  G = 20-day moving average  ·  B = volume",
                  fontsize=9, color=DARK_AXIS)
    fig.tight_layout()
    return fig


def cnn_predict_latest(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """Return prediction dict, or {'status': ..., 'reason': ...} explaining why
    inference is unavailable. Reasons: 'model_missing', 'data_short',
    'torch_missing', 'inference_error'. Caller branches on `reason`."""
    if not os.path.exists(model_path):
        return {
            'status': (f"`{model_path}` not found in current directory. "
                       f"Place the trained checkpoint here to enable inference."),
            'reason': 'model_missing',
        }
    if len(df) < LOOKBACK:
        return {
            'status': f'Need ≥ {LOOKBACK} days of data; got {len(df)}.',
            'reason': 'data_short',
        }
    try:
        import torch
        from train_cnn import ChartCNNv2  # noqa: F401

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
            'prob_down':    float(probs[0]),
            'prob_up':      float(probs[1]),
            'window_start': df.index[-LOOKBACK],
            'window_end':   df.index[-1],
        }
    except ImportError:
        return {
            'status': 'PyTorch not installed in this environment.',
            'reason': 'torch_missing',
        }
    except Exception as e:
        return {
            'status': f'CNN inference failed: {e}',
            'reason': 'inference_error',
        }


# Offline CNN test-set numbers (from the cluster training run). Surfaced when
# torch can't be loaded — e.g. on the Streamlit Cloud deploy where the torch
# wheel exceeds the dependency-size budget.
CNN_OFFLINE_RESULTS = {
    'train_n':               581_008,
    'val_n':                  96_938,
    'test_n':                424_698,
    'test_accuracy':         0.520,
    'baseline_accuracy':     0.525,
    'auc_roc':               0.512,
    'long_short_sharpe_net': -7.42,
}


def render_cnn_offline_card():
    """Display the offline CNN summary used when live inference is disabled."""
    st.markdown("#### CNN inference (offline result)")
    st.info(
        "Live CNN inference is disabled in this hosted environment due to "
        "PyTorch dependency size. The CNN was trained on top-1000 CRSP "
        f"common stocks ({CNN_OFFLINE_RESULTS['train_n']:,} train images, "
        f"{CNN_OFFLINE_RESULTS['val_n']:,} val, "
        f"{CNN_OFFLINE_RESULTS['test_n']:,} test) at 96×180×3 candlestick "
        "format. Test set results below were computed offline."
    )
    cols = st.columns(3)
    cols[0].metric(
        "Test accuracy",
        f"{CNN_OFFLINE_RESULTS['test_accuracy']:.1%}",
        f"vs {CNN_OFFLINE_RESULTS['baseline_accuracy']:.1%} baseline",
        delta_color='off',
    )
    cols[1].metric("AUC-ROC", f"{CNN_OFFLINE_RESULTS['auc_roc']:.3f}")
    cols[2].metric("Long-short Sharpe (net)",
                   f"{CNN_OFFLINE_RESULTS['long_short_sharpe_net']:.2f}")
    st.markdown(
        "<i>Conclusion: the CNN extracts less out-of-sample signal than the "
        "top 7 LMW patterns. See the report for full analysis.</i>",
        unsafe_allow_html=True,
    )


def make_price_with_patterns(df: pd.DataFrame, hits: pd.DataFrame,
                             ticker: str) -> plt.Figure:
    """Price line with scatter markers ON the line at completion dates,
    triangle-down for top patterns and triangle-up for bottom patterns.
    Annotates only the most recent ~5 detections to keep things readable."""
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    _apply_dark_style(fig, ax)
    ax.plot(df.index, df['Close'], color='#FAFAFA', lw=1.3, zorder=2)

    if not hits.empty:
        seen = set()
        for _, row in hits.iterrows():
            d = pd.to_datetime(row['date'])
            pat = row['pattern_type']
            if d not in df.index:
                continue
            price = float(df.loc[d, 'Close'])
            color = PATTERN_COLORS.get(pat, '#CCC')
            marker = 'v' if pat in TOP_PATTERNS else '^'
            offset_dir = +1 if pat in TOP_PATTERNS else -1
            offset_y = price * 0.012 * offset_dir
            label = pat if pat not in seen else None
            seen.add(pat)
            ax.scatter(d, price + offset_y,
                       marker=marker, s=70, color=color,
                       edgecolor=DARK_BG, linewidth=0.6,
                       label=label, zorder=4)

        # Annotate the most-recent detection of each unique pattern type
        # (cap at 5) so labels don't stack when consecutive same-type hits
        # cluster within a few days.
        recent_by_type = (
            hits.assign(date=pd.to_datetime(hits['date']))
                .sort_values('date')
                .groupby('pattern_type').tail(1)     # latest per type
                .sort_values('date').tail(5)          # latest 5 of those
        )
        for _, row in recent_by_type.iterrows():
            d = pd.to_datetime(row['date'])
            pat = row['pattern_type']
            if d not in df.index:
                continue
            price = float(df.loc[d, 'Close'])
            offset_dir = +1 if pat in TOP_PATTERNS else -1
            ax.annotate(
                pat,
                xy=(d, price + price * 0.012 * offset_dir),
                xytext=(0, 14 * offset_dir),
                textcoords='offset points',
                fontsize=8.5, color=PATTERN_COLORS.get(pat, '#CCC'),
                fontweight='bold',
                ha='center',
                va='bottom' if offset_dir > 0 else 'top',
                zorder=5,
            )

    ax.set_title(f"{ticker} — close with LMW pattern completions",
                 fontweight='bold', fontsize=12)
    ax.set_ylabel('Price ($)')
    if not hits.empty:
        leg = ax.legend(ncol=5, fontsize=8, loc='upper left',
                        frameon=False, labelcolor=DARK_FG)
        for text in leg.get_texts():
            text.set_color(DARK_FG)
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

    if 'ticker_input' not in st.session_state:
        st.session_state['ticker_input'] = 'AAPL'

    def _set_ticker(t):
        st.session_state['ticker_input'] = t

    with st.sidebar:
        st.header("Inputs")
        st.caption("Quick picks")
        chip_cols = st.columns(len(EXAMPLE_TICKERS))
        for i, t in enumerate(EXAMPLE_TICKERS):
            chip_cols[i].button(
                t, key=f'chip_{t}',
                use_container_width=True,
                on_click=_set_ticker, args=(t,),
            )
        raw_ticker = st.text_input("Ticker symbol", key='ticker_input')
        ticker = (raw_ticker or '').upper().strip()
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

    # ---- Summary box: temporal context for the detection count
    n_hits = len(hits)
    n_days = len(df)
    months = max(n_days / 21.0, 1e-6)  # ~21 trading days/month
    rate = n_hits / months
    if n_hits:
        most_common_code = hits['pattern_type'].value_counts().idxmax()
        most_common_name = PATTERN_INFO.get(most_common_code, (most_common_code, ''))[0]
        st.success(
            f"**{n_hits}** patterns detected over **{n_days}** trading days "
            f"(~{rate:.1f}/month). Most common: **{most_common_code}** — {most_common_name}."
        )
    else:
        st.info(f"No LMW patterns detected in {n_days} trading days.")

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

    # ---- Pattern breakdown (only patterns that fired, sorted desc)
    st.subheader("Pattern breakdown")
    if hits.empty:
        st.info("No LMW patterns detected in this window.")
    else:
        counts = hits['pattern_type'].value_counts().sort_values(ascending=False)
        n_cards = len(counts)
        cols = st.columns(min(n_cards, 5))
        for i, (pat, n) in enumerate(counts.items()):
            cols[i % len(cols)].metric(pat, int(n))

        st.markdown(f"**All {len(hits)} detections**")
        all_types = list(counts.index)
        selected = st.multiselect(
            "Filter by pattern type",
            options=all_types, default=all_types,
            key='pattern_filter',
        )
        view = hits[hits['pattern_type'].isin(selected)].copy()
        view['date'] = pd.to_datetime(view['date']).dt.date
        view['fwd_return_5d'] = view['fwd_return_5d'].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
        )
        view = view[['date', 'pattern_type', 'fwd_return_5d']]
        if view.empty:
            st.caption("No detections match the current filter.")
        else:
            st.dataframe(view.sort_values('date', ascending=False),
                         use_container_width=True, hide_index=True)

    # ---- CNN prediction
    st.divider()
    st.subheader("CNN prediction — next 5 trading days")
    res = cnn_predict_latest(df, MODEL_PATH)
    if 'status' in res:
        if res.get('reason') == 'torch_missing':
            render_cnn_offline_card()
        else:
            st.info(res['status'])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("P(up over 5d)",   f"{res['prob_up']:.1%}")
        c2.metric("P(down over 5d)", f"{res['prob_down']:.1%}")
        c3.write(f"**Window:**  \n{res['window_start'].date()} → "
                 f"{res['window_end'].date()}")
        st.caption(CNN_DISCLAIMER)

    st.divider()
    _glossary()

    # ---- Footer
    st.markdown(
        f"<div style='text-align:center; color:#5A6068; font-size:0.85em; "
        f"padding-top:1em;'>"
        f"Source: <code>{GITHUB_REPO}</code>  ·  {COURSE_TAG}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
