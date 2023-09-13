import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def TickRule(price : pd.Series) :
    price_change = price.diff()
    aggressor = pd.Series(index = price.index, data = np.nan)

    aggressor.iloc[0] = 1
    aggressor[price_change < 0] = -1
    aggressor[price_change > 0] = 1
    aggressor = aggressor.fillna(method = 'ffill')
    return aggressor
def RollModel(price : pd.Series) :
    price_change = price.diff()
    autocorr = price_change.autocorr(lag = 1)
    spread_squared = np.max([-autocorr, 0])
    spread = np.sqrt(spread_squared)
    noise = price_change.var() - 2 * (spread ** 2)
    return spread, noise
def RangeVolatility(high, low, window):
    log_high_low = np.log(high / low)
    volatility = log_high_low.rolling(window = window).mean() / np.sqrt(8. / np.pi)
    return volatility
class CorwinSchultz :
    @staticmethod
    def beta(high, low, length) :
        range_log = np.log(high / low) ** 2
        sum_neighbors = range_log.rolling(window = 2).sum()
        beta = sum_neighbors.rolling(window = length).mean()
        return beta
    @staticmethod
    def gamma(high, low):
        high_bars = high.rolling(window = 2).max()
        low_bars = low.rolling(window = 2).min()
        gamma = np.log(high_bars / low_bars) ** 2
        return gamma
    @staticmethod
    def alpha(beta, gamma):
        denominator = 3 - (2 * np.sqrt(2))
        beta_term = (np.sqrt(2) - 1) * np.sqrt(beta) / denominator
        gamma_term = np.sqrt(gamma / denominator)
        alpha = beta_term - gamma_term
        alpha[alpha < 0] = 0
        return alpha
    @staticmethod
    def BeckerParkinsonVolatility(beta, gamma):
        k2 = np.sqrt(8 / np.pi)
        denominator = 3 - 2 ** 1.5
        beta_term = (2 ** (-0.5) -1) * np.sqrt(beta) / (k2 * denominator)
        gamma_term = np.sqrt(gamma / (k2 ** 2 * denominator))
        volatility = beta_term + gamma_term
        volatility[volatility < 0] = 0
        return volatility

def CorwinSchultzSpread(high, low, sample_length=1):
    beta = CorwinSchultz.beta(high, low, sample_length)
    gamma = CorwinSchultz.gamma(high, low)
    alpha = CorwinSchultz.alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread

def BeckerParkinsonVolatility(high, low, sample_length=1):
    beta = CorwinSchultz.beta(high, low, sample_length)
    gamma = CorwinSchultz.gamma(high, low)
    volatility = CorwinSchultz.BeckerParkinsonVolatility(beta, gamma)
    return volatility
class Lambda(object) :
    def __init__(self, price, volume):
        self._price = price
        self._volume = volume
    @property
    def price(self):
        return self._price
    @price.setter
    def price(self, price):
        self._price = price
    @property
    def volume(self):
        return self._volume
    @volume.setter
    def volume(self, volume):
        self._volume
    def kyle(self, signs : pd.Series) -> pd.Series :
        price_change = self._price.diff()
        net_order_flow = signs * self._volume
        lambda_ = price_change / net_order_flow
        return lambda_
    def aminud(self) -> pd.Series :
        log_price = np.log(self._price)
        abs_diff = np.abs(log_price.diff())
        lambda_ = abs_diff / self._volume
        return lambda_
    def hasbrouck(self, signs : pd.Series ) -> pd.Series :
        log_price = np.log(self._price)
        log_diff = log_price.diff()
        net_order_flow = signs * np.sqrt(self._volume)
        lambda_ = log_diff / net_order_flow
        return lambda_


def kyleLambda(price : pd.Series,
               volume : pd.Series,
               signs,
               regressor = LinearRegression()):
    price_change = price.diff()
    net_order_flow = signs * volume
    x_val = net_order_flow.values[1:].reshape(-1, 1) # regression
    y_val = price_change.dropna().values
    lambda_ = regressor.fit(x_val, y_val)
    return lambda_.coef_[0]

def amihudLambda(price : pd.Series,
                 volume : pd.Series,
                 regressor = LinearRegression()):
    log_price = np.log(price)
    abs_diff = np.abs(log_price.diff())
    x = volume.values[1:].reshape(-1, 1)
    y = abs_diff.dropna()
    lambda_ = regressor.fit(x, y)
    return lambda_.coef_[0]

def hasbrouckLambda(price : pd.Series,
                    volume : pd.Series,
                    sign) :
    lambda_ = (np.sqrt(price * volume) * sign).sum()
    return lambda_
def hasbroucksFlow(tick_prices, tick_volumes, tick_sings):
    return (np.sqrt(tick_prices * tick_volumes) * tick_sings).sum()

def vpin(buy : pd.Series,
         sell : pd.Series,
         volume : pd.Series,
         num_bars : int) :
    abs_diff = (buy - sell).abs()
    estimated_vpin = abs_diff.rolling(window = num_bars).mean() / volume
    return estimated_vpin

def dollarVolume(price, volume) :
    dollarVol = (price * volume).sum()
    return dollarVol