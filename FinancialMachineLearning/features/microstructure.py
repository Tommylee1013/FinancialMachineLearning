import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def tick_rule(tick_prices : pd.Series):
    price_change = tick_prices.diff()
    aggressor = pd.Series(index=tick_prices.index, data=np.nan)
    aggressor.iloc[0] = 1.
    aggressor[price_change < 0] = -1.
    aggressor[price_change > 0] = 1.
    aggressor = aggressor.fillna(method='ffill')
    return aggressor
def volume_weighted_average_price(dollar_volume: list, volume: list) -> float:
    return sum(dollar_volume) / sum(volume)
def get_avg_tick_size(tick_size_arr: list) -> float:
    return np.mean(tick_size_arr)

class RollModel:
    def __init__(self, close_prices : pd.Series, window : int = 20) -> None:
        self.close_prices = close_prices
        self.window = window
    def roll_measure(self) -> pd.Series :
        price_diff = self.close_prices.diff()
        price_diff_lag = price_diff.shift(1)
        return 2 * np.sqrt(abs(price_diff.rolling(window = self.window).cov(price_diff_lag)))
    def roll_impact(self, dollar_volume : pd.Series) -> pd.Series :
        roll_measure = self.roll_measure()
        return roll_measure / dollar_volume
def roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))
def roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    roll_measure_ = roll_measure(close_prices, window)
    return roll_measure_ / dollar_volume


class CorwinSchultz :
    def __init__(self, high : pd.Series, low : pd.Series) -> None:
        self.high = high
        self.low = low
    def beta(self, window : int) -> pd.Series:
        ret = np.log(self.high / self.low)
        high_low_ret = ret ** 2
        beta = high_low_ret.rolling(window=2).sum()
        beta = beta.rolling(window=window).mean()
        return beta
    def gamma(self) -> pd.Series:
        high_max = self.high.rolling(window = 2).max()
        low_min = self.low.rolling(window = 2).min()
        gamma = np.log(high_max / low_min) ** 2
        return gamma
    def alpha(self, window : int) -> pd.Series:
        den = 3 - 2 * 2 ** .5
        alpha = (2 ** .5 - 1) * (self.beta(window = window) ** .5) / den
        alpha -= (self.gamma() / den) ** .5
        alpha[alpha < 0] = 0
        return alpha
    def corwin_schultz_estimator(self, window : int = 20) -> pd.Series :
        alpha_ = self.alpha(window = window)
        spread = 2 * (np.exp(alpha_) - 1) / (1 + np.exp(alpha_))
        start_time = pd.Series(self.high.index[0:spread.shape[0]], index=spread.index)
        spread = pd.concat([spread, start_time], axis=1)
        spread.columns = ['Spread', 'Start_Time']
        return spread.Spread
    def becker_parkinson_vol(self, window: int = 20) -> pd.Series:
        Beta = self.beta(window = window)
        Gamma = self.gamma()
        k2 = (8 / np.pi) ** 0.5
        den = 3 - 2 * 2 ** .5
        sigma = (2 ** -0.5 - 1) * Beta ** 0.5 / (k2 * den)
        sigma += (Gamma / (k2 ** 2 * den)) ** 0.5
        sigma[sigma < 0] = 0
        return sigma
def beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta

def gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma

def alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0
    return alpha

def corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    beta_ = beta(high, low, window)
    gamma_ = gamma(high, low)
    alpha_ = alpha(beta_, gamma_)
    spread = 2 * (np.exp(alpha_) - 1) / (1 + np.exp(alpha_))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread

def becker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    Beta = beta(high, low, window)
    Gamma = gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * Beta ** 0.5 / (k2 * den)
    sigma += (Gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma

class BarbasedLambda :
    def __init__(self, close : pd.Series, volume : pd.Series,
                 dollar_volume: pd.Series, window : int = 20):
        self.close = close
        self.volume = volume
        self.window = window
        self.dollar_volume = dollar_volume
    def kyle(self) -> pd.Series :
        close_diff = self.close.diff()
        close_diff_sign = np.sign(close_diff)
        close_diff_sign.replace(0, method='pad', inplace=True)
        volume_mult_trade_signs = self.volume * close_diff_sign
        return (close_diff / volume_mult_trade_signs).rolling(window = self.window).mean()
    def amihud(self) -> pd.Series :
        returns_abs = np.log(self.close / self.close.shift(1)).abs()
        return (returns_abs / self.dollar_volume).rolling(window = self.window).mean()
    def hasbrouck(self) -> pd.Series :
        log_ret = np.log(self.close / self.close.shift(1))
        log_ret_sign = np.sign(log_ret).replace(0, method='pad')
        signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(self.dollar_volume)
        return (log_ret / signed_dollar_volume_sqrt).rolling(window = self.window).mean()
class TradebasedLambda :
    def __init__(self, price_diff : list, log_ret : list,
                 volume : list, dollar_volume : list, aggressor_flags : list) -> float:
        self.price_diff = price_diff
        self.log_ret = log_ret
        self.volume = volume
        self.dollar_volume = dollar_volume
        self.aggressor_flags = aggressor_flags
    def kyle(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        signed_volume = np.array(self.volume) * np.array(self.aggressor_flags)
        X = np.array(signed_volume).reshape(-1, 1)
        y = np.array(self.price_diff)
        model.fit(X, y)
        return model.coef_[0]
    def amihud(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        X = np.array(self.dollar_volume).reshape(-1, 1)
        y = np.abs(np.array(self.log_ret))
        model.fit(X, y)
        return model.coef_[0]
    def hasbrouck(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        X = (np.sqrt(np.array(self.dollar_volume)) * np.array(self.aggressor_flags)).reshape(-1, 1)
        y = np.abs(np.array(self.log_ret))
        model.fit(X, y)
        return model.coef_[0]
def bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    close_diff = close.diff()
    close_diff_sign = np.sign(close_diff)
    close_diff_sign.replace(0, method='pad', inplace=True)
    volume_mult_trade_signs = volume * close_diff_sign
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()
def bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    returns_abs = np.log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()
def bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = np.sign(log_ret).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()
def trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    model.fit(X, y)
    return model.coef_[0]
def trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]
def trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]
def vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 1) -> pd.Series:
    sell_volume = volume - buy_volume
    volume_imbalance = abs(buy_volume - sell_volume)
    return volume_imbalance.rolling(window=window).mean() / volume
