from abc import ABC, abstractmethod
from collections import namedtuple
import pandas as pd
import numpy as np

class BaseBar(ABC) :
    def __init__(self, path, metric, batch_size = 2e7, additional_features = None):
        self.path = path
        self.metric = metric
        self.batch_size = batch_size
        self.tick_rule = 0
        self.flag = False
        self.cache = []
        if not additional_features : additional_features = []
        self.additional_features = additional_features
        self.computed_additional_features = []
        self.ticks_in_current_bar = []
    def batch_run(self, verbose=True, to_csv=False, output_path=None):
        first_row = pd.read_csv(self.path, nrows=1)
        self._assert_csv(first_row)
        header = False
        if to_csv is True:
            header = True
            open(output_path, 'w').close()
        if verbose: print('Reading data in batches:')
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume'] + [feature.name for feature in self.additional_features]
        for batch in pd.read_csv(self.path, chunksize=self.batch_size):
            if verbose: print('Batch number:', count)
            list_bars = self._extract_bars(data=batch)
            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else: final_bars += list_bars
            count += 1
            self.flag = True
        if verbose:
            print('Returning bars \n')
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
        try:pd.to_datetime(test_batch.iloc[0, 0])
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
        list_bars.append([date_time, open_price, high_price, low_price, close_price, volume] + additional_features)
    def _apply_tick_rule(self, price):
        if self.cache: tick_diff = price - self.cache[-1].price
        else: tick_diff = 0
        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else: signed_tick = self.prev_tick_rule
        return signed_tick
    def _get_imbalance(self, price, signed_tick, volume):
        if self.metric == 'tick_imbalance' or self.metric == 'tick_run':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance' or self.metric == 'dollar_run':
            imbalance = signed_tick * volume * price
        else: imbalance = signed_tick * volume
        return imbalance
class StandardBar(BaseBar):
    def __init__(self, file_path, metric, threshold=50000, batch_size=20000000, additional_features=None):
        BaseBar.__init__(self, file_path, metric, batch_size, additional_features)
        self.threshold = threshold
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_dollar'])
    def _extract_bars(self, data):
        cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = self._update_counters()
        list_bars = []
        for _, row in data.iterrows():
            date_time = row.iloc[0]
            price = np.float(row.iloc[1])
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
                cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf
                self.cache = []
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()
        return list_bars
    def _update_counters(self):
        if self.flag and self.cache:
            last_entry = self.cache[-1]
            cum_ticks = int(last_entry.cum_ticks)
            cum_dollar_value = np.float(last_entry.cum_dollar)
            cum_volume = last_entry.cum_volume
            low_price = np.float(last_entry.low)
            high_price = np.float(last_entry.high)
        else:
            cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = 0, 0, 0, -np.inf, np.inf
        return cum_ticks, cum_dollar_value, cum_volume, high_price, low_price
    def _update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_dollar_value):
        cache_data = self.cache_tuple(
            date_time, price, high_price, low_price, cum_ticks, cum_volume, cum_dollar_value)
        self.cache.append(cache_data)
def dollarBar(file_path : str,
              threshold : int = 1000000,
              batch_size : int = 20000000,
              verbose : bool = True,
              to_csv : bool = False,
              output_path : str = None,
              additional_features = None):
    bars = StandardBar(file_path=file_path, metric='cum_dollar_value', threshold=threshold, batch_size=batch_size, additional_features=additional_features)
    dollar_bars = bars.batch_run(verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars
def volumeBar(file_path : str,
              threshold : int = 10000,
              batch_size : int = 20000000,
              verbose : bool = True,
              to_csv : bool = False,
              output_path : str = None,
              additional_features = None):
    bars = StandardBar(file_path=file_path, metric='cum_volume',
                        threshold=threshold, batch_size=batch_size, additional_features=additional_features)
    volume_bars = bars.batch_run(verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars
def tickBar(file_path : str,
            threshold : int = 2800,
            batch_size : int = 20000000,
            verbose : bool = True,
            to_csv : bool = False,
            output_path : str = None,
            additional_features = None):
    bars = StandardBar(file_path=file_path, metric='cum_ticks',
                       threshold=threshold, batch_size=batch_size, additional_features=additional_features)
    tick_bars = bars.batch_run(verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars