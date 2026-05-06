"""
image_v2.py — 3-channel candlestick chart image for 15.C51 Project 2.

Output shape: (3, 96, 180) uint8 in [0, 255].
  Channel 0: candlesticks (wicks + filled-up / hollow-down bodies)
  Channel 1: 20-day moving-average line (anti-aliased)
  Channel 2: volume bars (bottom 20% of image)
"""

import numpy as np
import pandas as pd
import cv2


def make_candlestick_image(
    ohlcv_window: pd.DataFrame,
    height: int = 96,
    width: int = 180,
    ma_window: int = 20,
) -> np.ndarray:
    n_days = width // 3
    assert len(ohlcv_window) == n_days, (
        f"window must have {n_days} rows for {width}px width, got {len(ohlcv_window)}"
    )

    img = np.zeros((3, height, width), dtype=np.uint8)

    opens  = ohlcv_window['Open'].to_numpy(dtype=np.float64)
    highs  = ohlcv_window['High'].to_numpy(dtype=np.float64)
    lows   = ohlcv_window['Low'].to_numpy(dtype=np.float64)
    closes = ohlcv_window['Close'].to_numpy(dtype=np.float64)
    vols   = ohlcv_window['Volume'].to_numpy(dtype=np.float64)

    ma = pd.Series(closes).rolling(ma_window, min_periods=1).mean().to_numpy()

    # Layout: price panel rows [0, 77), volume panel rows [77, 96).
    price_top, price_bot = 0, 77
    vol_top, vol_bot = 77, height

    # Per-window min-max over {O, H, L, C, MA}
    price_stack = np.concatenate([opens, highs, lows, closes, ma])
    price_min = np.nanmin(price_stack)
    price_max = np.nanmax(price_stack)
    if not np.isfinite(price_max - price_min) or price_max == price_min:
        return img  # degenerate

    def price_to_row(p: float) -> float:
        frac = (p - price_min) / (price_max - price_min)
        return price_bot - 1 - frac * (price_bot - price_top - 1)

    # ---- Channel 0: candlesticks ----
    ch0 = img[0]
    for d in range(n_days):
        col_l, col_m, col_r = 3 * d, 3 * d + 1, 3 * d + 2

        if not (np.isnan(highs[d]) or np.isnan(lows[d])):
            r_hi = int(np.clip(round(price_to_row(highs[d])), price_top, price_bot - 1))
            r_lo = int(np.clip(round(price_to_row(lows[d])),  price_top, price_bot - 1))
            ch0[r_hi:r_lo + 1, col_m] = 255  # wick

        if not (np.isnan(opens[d]) or np.isnan(closes[d])):
            r_open  = int(np.clip(round(price_to_row(opens[d])),  price_top, price_bot - 1))
            r_close = int(np.clip(round(price_to_row(closes[d])), price_top, price_bot - 1))
            r_top, r_bot = min(r_open, r_close), max(r_open, r_close)

            if closes[d] >= opens[d]:
                ch0[r_top:r_bot + 1, col_l:col_r + 1] = 255          # solid body (up)
            else:
                if r_bot > r_top:
                    ch0[r_top,         col_l:col_r + 1] = 255         # hollow body (down)
                    ch0[r_bot,         col_l:col_r + 1] = 255
                    ch0[r_top:r_bot+1, col_l]           = 255
                    ch0[r_top:r_bot+1, col_r]           = 255
                else:
                    ch0[r_top, col_l:col_r + 1] = 255

    # ---- Channel 1: anti-aliased MA line ----
    ch1 = img[1]
    pts = []
    for d in range(n_days):
        if np.isnan(ma[d]):
            continue
        x = 3 * d + 1
        y = float(np.clip(price_to_row(ma[d]), price_top, price_bot - 1))
        pts.append((x, y))
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        cv2.line(ch1, (int(x0), int(round(y0))), (int(x1), int(round(y1))),
                 color=255, thickness=1, lineType=cv2.LINE_AA)

    # ---- Channel 2: volume bars ----
    ch2 = img[2]
    vol_max = np.nanmax(vols) if len(vols) else 0
    panel_h = vol_bot - vol_top
    if vol_max and vol_max > 0:
        for d in range(n_days):
            v = vols[d]
            if np.isnan(v) or v <= 0:
                continue
            h_pix = int(round((v / vol_max) * panel_h))
            if h_pix > 0:
                col_m = 3 * d + 1
                ch2[vol_bot - h_pix:vol_bot, col_m] = 255

    return img


def make_chart_image_legacy(ohlcv_window, height=64, width=60, ma_window=20):
    """Wrapper around the JKX-style 64x60 binary chart in utils.py for backward
    comparison. Returns a (1, H, W) uint8 array so callers can index it like
    the new 3-channel format."""
    from utils import make_chart_image
    img = make_chart_image(ohlcv_window, height=height, width=width, ma_window=ma_window)
    return img[None, :, :]
