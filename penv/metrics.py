
import numpy as np


def sharpe(returns: np.array, risk_free_rate: float = 0) -> float:
    return (returns.mean() - risk_free_rate) / returns.std()


def maximum_drawdown(net_worth: np.array) -> float:
    n = len(net_worth)
    nav = net_worth.copy()
    mdd = 0
    peak = -np.inf
    for i in range(n):
        if nav[i] > peak:
            peak = nav[i]
        dd = 100*(peak - nav[i]) / peak

        if dd > mdd:
            mdd = dd
    return mdd
