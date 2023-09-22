from scipy.stats import norm
import pandas as pd

def bvc_buy_volume(close: pd.Series,
                   volume: pd.Series,
                   window: int = 20) -> pd.Series:
    result = volume * norm.cdf(close.diff() / close.diff().rolling(window = window).std())
    return result