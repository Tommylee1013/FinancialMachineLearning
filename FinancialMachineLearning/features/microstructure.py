import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from collections import namedtuple

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
    return volume_imbalance.rolling(window = window).mean() / volume

class microBaseBars(ABC):
    def __init__(self, file_path, metric, batch_size=2e7, additional_features=None):
        self.file_path = file_path
        self.metric = metric
        self.batch_size = batch_size
        self.prev_tick_rule = 0
        self.flag = False
        self.cache = []
        if not additional_features:
            additional_features = []
        self.additional_features = additional_features
        self.computed_additional_features = []
        self.ticks_in_current_bar = []

    def batch_run(self, verbose=True, to_csv=False, output_path=None):
        first_row = pd.read_csv(self.file_path, nrows=1)
        self._assert_csv(first_row)

        if to_csv is True:
            header = True
            open(output_path, 'w').close()

        if verbose:
            print('Reading data in batches:')

        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume'] + [feature.name for feature in self.additional_features]
        for batch in pd.read_csv(self.file_path, chunksize=self.batch_size):
            if verbose:
                print('Batch number:', count)

            list_bars = self._extract_bars(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

            self.flag = True

        if verbose:  # pragma: no cover
            print('Returning bars \n')

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df
        return None

    @abstractmethod
    def _extract_bars(self, data):
        pass

    @staticmethod
    def _assert_csv(test_batch):
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])

    @staticmethod
    def _update_high_low(high_price, low_price, price):
        if price > high_price:
            high_price = price

        if price <= low_price:
            low_price = price

        return high_price, low_price

    def _update_ticks_in_bar(self, row):
        if self.additional_features:
            self.ticks_in_current_bar.append(row)

    def _reset_ticks_in_bar(self):
        self.ticks_in_current_bar = []

    def _compute_additional_features(self):
        computed_additional_features = []

        if self.additional_features:
            tick_df = pd.DataFrame(self.ticks_in_current_bar)
            for feature in self.additional_features:
                computed_additional_features.append(feature.compute(tick_df))

        self.computed_additional_features = computed_additional_features

    def _reset_computed_additional_features(self):
        self.computed_additional_features = []

    def _create_bars(self, date_time, price, high_price, low_price, list_bars):
        open_price = self.cache[0].price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        volume = self.cache[-1].cum_volume
        additional_features = self.computed_additional_features

        # Update bars
        list_bars.append([date_time, open_price, high_price, low_price, close_price, volume] + additional_features)

    def _apply_tick_rule(self, price):
        if self.cache:
            tick_diff = price - self.cache[-1].price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        return signed_tick

    def _get_imbalance(self, price, signed_tick, volume):
        if self.metric == 'tick_imbalance' or self.metric == 'tick_run':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance' or self.metric == 'dollar_run':
            imbalance = signed_tick * volume * price
        else:  # volume imbalance or volume run
            imbalance = signed_tick * volume

        return imbalance

class microBars(microBaseBars):
    def __init__(self, file_path, metric, threshold=50000, batch_size=20000000, additional_features=None):
        microBaseBars.__init__(self, file_path, metric, batch_size, additional_features)
        self.threshold = threshold
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_dollar'])

    def _extract_bars(self, data):
        cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = self._update_counters()
        list_bars = []
        for _, row in data.iterrows():
            # Set variables
            date_time = row.iloc[0]
            price = np.float64(row.iloc[1])
            volume = row.iloc[2]
            high_price, low_price = self._update_high_low(
                high_price, low_price, price)
            cum_ticks += 1
            dollar_value = price * volume
            cum_dollar_value = cum_dollar_value + dollar_value
            cum_volume += volume
            self._update_cache(date_time, price, low_price,
                               high_price, cum_ticks, cum_volume, cum_dollar_value)
            self._update_ticks_in_bar(row)
            if eval(self.metric) >= self.threshold:
                self._compute_additional_features()
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                # Reset counters
                cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf
                self.cache = []
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()
        return list_bars

    def _update_counters(self):
        if self.flag and self.cache:
            last_entry = self.cache[-1]
            cum_ticks = int(last_entry.cum_ticks)
            cum_dollar_value = np.float64(last_entry.cum_dollar)
            cum_volume = last_entry.cum_volume
            low_price = np.float64(last_entry.low)
            high_price = np.float64(last_entry.high)
        else:
            cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf

        return cum_ticks, cum_dollar_value, cum_volume, high_price, low_price

    def _update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_dollar_value):
        cache_data = self.cache_tuple(
            date_time, price, high_price, low_price, cum_ticks, cum_volume, cum_dollar_value)
        self.cache.append(cache_data)
def vpin_volume_bars(file_path, threshold=28224, batch_size=20000000, verbose=True, to_csv=False, output_path=None, additional_features=None):
    bars = microBars(file_path=file_path, metric='cum_volume',
                     threshold=threshold, batch_size=batch_size, additional_features = additional_features)
    volume_bars = bars.batch_run(verbose=verbose, to_csv=to_csv, output_path = output_path)
    return volume_bars