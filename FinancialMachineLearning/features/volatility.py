import numpy as np
import pandas as pd
import arch

def daily_volatility(close: pd.Series, lookback: int = 100) -> pd.Series:
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1],
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    df0 = df0.ewm(span = lookback).std()
    return df0

def parkinson_volatility(high: pd.Series, low : pd.Series, window : int = 20) -> pd.Series :
    ret = np.log(high / low)
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window = window).mean())

def garman_class_volatility(open : pd.Series,
                            high : pd.Series,
                            low : pd.Series,
                            close : pd.Series,
                            window : int = 20) -> pd.Series :
    ret = np.log(high / low)
    close_open_ret = np.log(close / open)
    estimator = 0.5 * ret ** 2 - (2 * np.log(2) - 1) * close_open_ret ** 2
    return np.sqrt(estimator.rolling(window = window).mean())

def yang_zhang_volatility(open : pd.Series,
                          high : pd.Series,
                          low : pd.Series,
                          close : pd.Series,
                          window : int = 20) -> pd.Series :
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret ** 2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret ** 2).rolling(window=window).sum()
    sigma_rs_sq = 1 / (window - 1) * (high_close_ret * high_open_ret + low_close_ret * low_open_ret).rolling(
        window=window).sum()

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)

class HeteroscedasticityModels:
    def __init__(self, close : pd.Series, vol : str = 'original'):
        if vol == 'original' :
            self.ret = np.log(close / close.shift(1)).dropna()
        elif vol == 'return' :
            self.ret = close
        else :
            raise ValueError('Only [original, return] can choose')
    def arch(self, p : int = 1,
             mean : str = 'Constant',
             dist : str = 'normal') :
        model = arch.arch_model(self.ret, vol = 'ARCH', p = p, mean = mean, dist = dist)
        result = model.fit()
        return result
    def garch(self, p : int = 1,
              q : int = 1,
              mean : str = 'Constant',
              dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'GARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def egarch(self, p : int = 1,
               q : int = 1,
               mean : str = 'Constant',
               dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'EGARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def garchm(self, p : int = 1,
               q : int = 1,
               mean : str = 'constant',
               dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'GARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def figarch(self,
                p : int = 1,
                q : int = 1,
                mean : str = 'constant',
                dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'FIGARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result