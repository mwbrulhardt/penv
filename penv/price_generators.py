

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from tensortrade.feed import Stream


class MultiGBM(Stream[np.array]):

    def __init__(self, s0: np.array, drift: np.array, volatility: np.array, rho: np.array, n: int):
        super().__init__()
        self.n = n
        self.m = len(s0)
        self.i = 0

        self.dt = 1 / n

        self.s0 = s0.reshape(-1, 1)
        self.mu = drift.reshape(-1, 1)
        self.v = volatility.reshape(-1, 1)

        V = (self.v@self.v.T)*rho
        self.A = np.linalg.cholesky(V)
        self.x = None

    def forward(self) -> np.array:
        self.i += 1
        if self.x is None:
            self.x = self.s0.flatten().astype(float)
            return self.x
        dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=[self.m, 1])
        s = np.exp((self.mu - (1 / 2)*self.v**2)*self.dt + (self.A@dw)).T
        s = s.flatten()
        self.x *= s
        return self.x

    def has_next(self):
        return self.i < self.n

    def reset(self):
        super().reset()
        self.i = 0
        self.x = None


def multi_corr_gbm(s0: np.array, drift: np.array, volatility: np.array, rho: np.array, n: int):

    m = len(s0)
    assert drift.shape == (m,)
    assert volatility.shape == (m,)
    assert rho.shape == (m, m)

    dt = 1 / n

    s0 = s0.reshape(-1, 1) # Shape: (m, 1)
    mu = drift.reshape(-1, 1) # Shape: (m, 1)
    v = volatility.reshape(-1, 1) # Shape: (m, 1)

    V = (v@v.T)*rho
    A = np.linalg.cholesky(V) # Shape: (m, m)

    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=[m, n]) # Shape (m, n)

    S = np.exp((mu - (1 / 2)*v**2)*dt + (A@dW)).T

    S = np.vstack([np.ones(m), S])

    S = s0.T*S.cumprod(0)

    return S


def make_multi_gbm_price_curve(n: int):

    rho = np.array([
        [ 1.        , -0.34372319,  0.23809065, -0.21918481],
        [-0.34372319,  1.        , -0.07774865, -0.17430333],
        [ 0.23809065, -0.07774865,  1.        , -0.17521052],
        [-0.21918481, -0.17430333, -0.17521052,  1.        ]
    ])
    s0 = np.array([50, 48, 45, 60])
    drift = np.array([0.13, 0.16, 0.10, 0.05])
    volatility = np.array([0.25, 0.20, 0.30, 0.15])

    P = multi_corr_gbm(s0, drift, volatility, rho, n)

    prices = pd.DataFrame(P).astype(float)
    prices.columns = ["p1", "p2", "p3", "p4"]

    #prices = prices.ewm(span=50).mean()

    return prices


def make_shifting_sine_price_curves(n: int, warmup: int = 0):
    n += 1

    slide = 2*np.pi*(warmup / n)

    steps = n + warmup

    x = np.linspace(-slide, 2*np.pi, num=steps)
    x = np.repeat(x, 4).reshape(steps, 4)

    s0 = np.array([50, 48, 45, 60]).reshape(1, 4)
    shift = np.array([0, np.pi / 2, np.pi, 3*np.pi / 2]).reshape(1, 4)
    freq = np.array([1, 4, 3, 2]).reshape(1, 4)

    y = s0 + 25*np.sin(freq*(x - shift))

    prices = pd.DataFrame(y, columns=["p1", "p2", "p3", "p4"])

    return prices


class MultiSinePriceCurves(Stream[np.array]):

    def __init__(self, s0: np.array, shift: np.array, freq: np.array, n: int, warmup: int = 0):
        super().__init__()
        self.s0 = s0
        self.shift = shift
        self.freq = freq

        self.steps = n + warmup + 1
        self.i = 0
        self.m = len(s0)

        self.x = np.linspace(-2*np.pi*(warmup / n), 2*np.pi, num=self.steps)

    def forward(self) -> np.array:
        rv = truncnorm.rvs(a=-10, b=10, size=self.m)
        v = self.s0 + 25*np.sin(self.freq*(self.x[self.i] - self.shift)) + rv
        self.i += 1
        return v

    def has_next(self):
        return self.i < self.steps

    def reset(self):
        super().reset()
        self.i = 0
