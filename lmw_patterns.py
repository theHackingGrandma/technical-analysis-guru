"""
lmw_patterns.py — Lo, Mamaysky & Wang (2000) classical chart pattern detection.

Two halves:
  * Detector — Gaussian-smoothed extrema + 10 LMW pattern definitions.
  * Synthesizer — generate price series guaranteed to satisfy each pattern by
    construction, for sanity-checking the detector.

Pattern definitions follow LMW Section II.A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator


# Default smoothing bandwidth.  LMW use a Nadaraya-Watson kernel smoother with
# bandwidth h* set to 0.3 * h_optimal (h_optimal from cross-validation).  A
# Gaussian filter with sigma in [1.0, 2.5] is a close empirical match for daily
# data — high enough to suppress single-day noise, low enough to preserve the
# 5-extrema window patterns.  We default to sigma=1.5.
DEFAULT_SIGMA = 1.5
WINDOW_LENGTH = 38  # l = 35 main + d = 3 detection lag


# ---------------------------------------------------------------------------
# Extrema detector
# ---------------------------------------------------------------------------

def find_extrema(prices, sigma: float = DEFAULT_SIGMA):
    """Return [(idx, smoothed_price, type), ...] for local extrema of the
    Gaussian-smoothed price series.

    Extrema are detected via sign changes of the first difference of the
    smoothed series (positive→non-positive ⇒ max, negative→non-negative ⇒ min).
    """
    p = np.asarray(prices, dtype=float).ravel()
    if len(p) < 3:
        return []
    s = gaussian_filter1d(p, sigma=sigma)
    d = np.diff(s)
    out = []
    for i in range(1, len(s) - 1):
        before, after = d[i - 1], d[i]
        if before > 0 and after <= 0:
            out.append((i, float(s[i]), 'max'))
        elif before < 0 and after >= 0:
            out.append((i, float(s[i]), 'min'))
    return out


# ---------------------------------------------------------------------------
# Pattern conditions
# ---------------------------------------------------------------------------

def _all_within(values, pct):
    """True if every value is within `pct` of the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    m = arr.mean()
    if m == 0 or not np.isfinite(m):
        return False
    return bool(np.all(np.abs(arr - m) / abs(m) <= pct))


def _types(seg):
    return [e[2] for e in seg]


def _vals(seg):
    return [e[1] for e in seg]


_TYPES_MAX_FIRST = ['max', 'min', 'max', 'min', 'max']
_TYPES_MIN_FIRST = ['min', 'max', 'min', 'max', 'min']


def _hs(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MAX_FIRST:
            continue
        v = _vals(seg)
        if not (v[2] > v[0] and v[2] > v[4]):
            continue
        if not _all_within([v[0], v[4]], 0.015):
            continue
        if not _all_within([v[1], v[3]], 0.015):
            continue
        return True
    return False


def _ihs(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MIN_FIRST:
            continue
        v = _vals(seg)
        if not (v[2] < v[0] and v[2] < v[4]):
            continue
        if not _all_within([v[0], v[4]], 0.015):
            continue
        if not _all_within([v[1], v[3]], 0.015):
            continue
        return True
    return False


def _btop(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MAX_FIRST:
            continue
        v = _vals(seg)
        if v[0] < v[2] < v[4] and v[1] > v[3]:
            return True
    return False


def _bbot(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MIN_FIRST:
            continue
        v = _vals(seg)
        if v[0] > v[2] > v[4] and v[1] < v[3]:
            return True
    return False


def _ttop(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MAX_FIRST:
            continue
        v = _vals(seg)
        if v[0] > v[2] > v[4] and v[1] < v[3]:
            return True
    return False


def _tbot(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MIN_FIRST:
            continue
        v = _vals(seg)
        if v[0] < v[2] < v[4] and v[1] > v[3]:
            return True
    return False


def _rtop(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MAX_FIRST:
            continue
        v = _vals(seg)
        tops = [v[0], v[2], v[4]]
        bots = [v[1], v[3]]
        if not _all_within(tops, 0.0075):
            continue
        if not _all_within(bots, 0.0075):
            continue
        if min(tops) > max(bots):
            return True
    return False


def _rbot(extrema):
    for i in range(len(extrema) - 4):
        seg = extrema[i:i + 5]
        if _types(seg) != _TYPES_MIN_FIRST:
            continue
        v = _vals(seg)
        bots = [v[0], v[2], v[4]]
        tops = [v[1], v[3]]
        if not _all_within(tops, 0.0075):
            continue
        if not _all_within(bots, 0.0075):
            continue
        if min(tops) > max(bots):
            return True
    return False


def _dtop(extrema, min_sep: int = 22):
    maxes = [e for e in extrema if e[2] == 'max']
    if len(maxes) < 2:
        return False
    e1 = maxes[0]
    later = [m for m in maxes[1:] if m[0] - e1[0] > min_sep]
    if not later:
        return False
    ea = max(later, key=lambda m: m[1])
    return _all_within([e1[1], ea[1]], 0.015)


def _dbot(extrema, min_sep: int = 22):
    mins = [e for e in extrema if e[2] == 'min']
    if len(mins) < 2:
        return False
    e1 = mins[0]
    later = [m for m in mins[1:] if m[0] - e1[0] > min_sep]
    if not later:
        return False
    ea = min(later, key=lambda m: m[1])
    return _all_within([e1[1], ea[1]], 0.015)


_PATTERN_FNS = {
    'HS':   _hs,   'IHS':  _ihs,
    'BTOP': _btop, 'BBOT': _bbot,
    'TTOP': _ttop, 'TBOT': _tbot,
    'RTOP': _rtop, 'RBOT': _rbot,
    'DTOP': _dtop, 'DBOT': _dbot,
}

PATTERN_NAMES = list(_PATTERN_FNS.keys())


def detect_patterns_in_window(prices_38d: np.ndarray,
                              sigma: float = DEFAULT_SIGMA) -> dict:
    """Run all 10 LMW detectors on a single 38-day price window."""
    extrema = find_extrema(prices_38d, sigma=sigma)
    return {name: fn(extrema) for name, fn in _PATTERN_FNS.items()}


def scan_series_for_patterns(prices,
                             window: int = WINDOW_LENGTH,
                             stride: int = 1,
                             sigma: float = DEFAULT_SIGMA) -> pd.DataFrame:
    """Slide a `window`-day window across `prices` and record every detection.

    Returns a DataFrame with columns [date, pattern_type, completion_idx].
    `completion_idx` is the index of the last day in the window relative to
    `prices`, and `date` is `prices.index[completion_idx]` if `prices` is a
    Series, else the integer index.
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
        idx = list(prices.index)
    else:
        arr = np.asarray(prices)
        idx = list(range(len(arr)))

    rows = []
    n = len(arr)
    for start in range(0, n - window + 1, stride):
        win = arr[start:start + window]
        completion = start + window - 1
        for name, present in detect_patterns_in_window(win, sigma=sigma).items():
            if present:
                rows.append((idx[completion], name, completion))
    return pd.DataFrame(rows, columns=['date', 'pattern_type', 'completion_idx'])


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def _spaced_positions(n_days, n_pts, jitter, rng, min_gap=4):
    """Roughly evenly spaced integer positions inside [margin, n_days-1-margin],
    with a minimum gap between consecutive points so the Gaussian smoothing
    kernel can resolve them as distinct extrema."""
    margin = max(2, n_days // 12)
    base = np.linspace(margin, n_days - 1 - margin, n_pts)
    j = rng.integers(-jitter, jitter + 1, size=n_pts) if jitter else np.zeros(n_pts, int)
    pos = (base + j).astype(int)
    pos = np.clip(pos, 0, n_days - 1)
    for k in range(1, len(pos)):
        if pos[k] < pos[k - 1] + min_gap:
            pos[k] = pos[k - 1] + min_gap
    return np.clip(pos, 0, n_days - 1).tolist()


def _interpolate(positions, values, n_days, rng, noise_std_pct=0.001, pad_frac=0.3):
    """PCHIP-interpolate through the design extrema. Boundary padding sits
    `pad_frac` of the way from the first/last design extremum toward its
    opposite-type neighbor — far enough that the boundary is monotone (so the
    detector still registers a clean local extremum) but close enough that
    Gaussian smoothing doesn't drag the boundary extremum's smoothed value
    away from its interior peers."""
    pos = list(positions)
    vals = list(values)
    if pos[0] > 0:
        pos.insert(0, 0)
        vals.insert(0, values[0] + pad_frac * (values[1] - values[0]))
    if pos[-1] < n_days - 1:
        pos.append(n_days - 1)
        vals.append(values[-1] + pad_frac * (values[-2] - values[-1]))
    pchip = PchipInterpolator(pos, vals)
    y = pchip(np.arange(n_days))
    sigma_noise = float(np.mean(y)) * noise_std_pct
    if sigma_noise > 0:
        y = y + rng.normal(0, sigma_noise, n_days)
    return y


def _make_ohlcv(closes, rng):
    n = len(closes)
    base_noise = float(np.mean(closes)) * 0.002
    range_noise = float(np.mean(closes)) * 0.003
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1] + rng.normal(0, base_noise, n - 1)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, range_noise, n))
    lows  = np.minimum(opens, closes) - np.abs(rng.normal(0, range_noise, n))
    vols  = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({
        'Open':   opens,
        'High':   highs,
        'Low':    lows,
        'Close':  closes,
        'Volume': vols,
    })


def synthesize_pattern(pattern_type: str,
                       n_days: int = WINDOW_LENGTH,
                       seed: int | None = None) -> pd.DataFrame:
    """Generate a synthetic OHLCV series of length `n_days` whose smoothed
    close path satisfies the given LMW pattern by construction."""
    rng = np.random.default_rng(seed)
    base = 100.0
    p = pattern_type.upper()

    def jit(scale=0.005):
        return float(rng.uniform(-scale, scale))

    if p == 'HS':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        shoulder, head, trough = base, base * 1.07, base * 0.95
        values = [
            shoulder * (1 + jit(0.004)),  # E1 max
            trough   * (1 + jit(0.004)),  # E2 min
            head     * (1 + jit(0.003)),  # E3 max (highest)
            trough   * (1 + jit(0.004)),  # E4 min
            shoulder * (1 + jit(0.004)),  # E5 max
        ]

    elif p == 'IHS':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        shoulder, dip, peak = base, base * 0.93, base * 1.05
        values = [
            shoulder * (1 + jit(0.004)),
            peak     * (1 + jit(0.004)),
            dip      * (1 + jit(0.003)),
            peak     * (1 + jit(0.004)),
            shoulder * (1 + jit(0.004)),
        ]

    elif p == 'BTOP':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        values = [base * 1.02, base * 0.97, base * 1.05, base * 0.93, base * 1.08]

    elif p == 'BBOT':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        values = [base * 0.98, base * 1.03, base * 0.95, base * 1.07, base * 0.92]

    elif p == 'TTOP':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        values = [base * 1.08, base * 0.93, base * 1.05, base * 0.97, base * 1.02]

    elif p == 'TBOT':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        values = [base * 0.92, base * 1.07, base * 0.95, base * 1.03, base * 0.98]

    elif p == 'RTOP':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        top, bot = base * 1.05, base * 0.95
        values = [
            top * (1 + jit(0.0015)),
            bot * (1 + jit(0.0015)),
            top * (1 + jit(0.0015)),
            bot * (1 + jit(0.0015)),
            top * (1 + jit(0.0015)),
        ]

    elif p == 'RBOT':
        positions = _spaced_positions(n_days, 5, jitter=2, rng=rng)
        top, bot = base * 1.05, base * 0.95
        values = [
            bot * (1 + jit(0.0015)),
            top * (1 + jit(0.0015)),
            bot * (1 + jit(0.0015)),
            top * (1 + jit(0.0015)),
            bot * (1 + jit(0.0015)),
        ]

    elif p == 'DTOP':
        # Two peaks separated by > 22 trading days; a single trough between.
        t1 = 4 + int(rng.integers(0, 3))
        t3 = 30 + int(rng.integers(0, 4))
        if t3 - t1 <= 22:
            t3 = t1 + 24
        t2 = (t1 + t3) // 2 + int(rng.integers(-2, 3))
        positions = [t1, t2, t3]
        peak1 = base * 1.05
        peak2 = peak1 * (1 + jit(0.005))
        values = [peak1, base * 0.93, peak2]

    elif p == 'DBOT':
        t1 = 4 + int(rng.integers(0, 3))
        t3 = 30 + int(rng.integers(0, 4))
        if t3 - t1 <= 22:
            t3 = t1 + 24
        t2 = (t1 + t3) // 2 + int(rng.integers(-2, 3))
        positions = [t1, t2, t3]
        bot1 = base * 0.95
        bot2 = bot1 * (1 + jit(0.005))
        values = [bot1, base * 1.07, bot2]

    else:
        raise ValueError(f"Unknown pattern: {pattern_type}")

    closes = _interpolate(positions, values, n_days, rng)
    return _make_ohlcv(closes, rng)
