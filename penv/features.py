
import numpy as np
from scipy.special import binom
from tensortrade.feed import Stream


def fracdiff(s: Stream[float], d: float, window: int) -> Stream[float]:
    c = np.tile([1.0, -1.0], -(-window // 2))[:window]
    w = c*binom(d, np.arange(window))
    w = w[::-1]
    frac = s.rolling(window=window, min_periods=window).agg(lambda v: np.dot(w.T, v))
    return frac


def macd(s: Stream[float], fast: int, slow: int, signal: int) -> Stream[float]:
    fm = s.ewm(span=fast, adjust=False).mean()
    sm = s.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


def rsi(s: Stream[float], period: float, use_multiplier: bool = True) -> Stream[float]:
    r = s.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    v = 1 - (1 + rs)**-1
    return 100*v if use_multiplier else v
