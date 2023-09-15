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