from collections import namedtuple
import numpy as np
from FinancialMachineLearning.multiprocess.fast_ewma import ewma
from FinancialMachineLearning.barsampling.standard_bar_features import BaseBar

class ImbalanceBars(BaseBar):
    def __init__(self, path : str,
                 metric : str,
                 num_prev_bar : int = 3,
                 exp_num_ticks_init : int = 100000,
                 batch_size = 2e7,
                 additional_features = None):
        BaseBar.__init__(self, path, metric, batch_size, additional_features)
        self.num_prev_bar = num_prev_bar
        self.exp_num_ticks_init = exp_num_ticks_init
        self.exp_num_ticks = self.exp_num_ticks_init
        self.num_ticks_bar = []
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_theta'])
        self.expected_imbalance = np.nan
        self.imbalance_array = []
    def extract_bars(self, data):
        cum_ticks, cum_volume, cum_theta, high_price, low_price = self.update_counters()
        list_bars = []
        for _, row in data.iterrows():
            cum_ticks += 1
            date_time = row.iloc[0]
            price = np.float64(row.iloc[1])
            volume = row.iloc[2]
            cum_volume += volume
            high_price, low_price = self._update_high_low(
                high_price, low_price, price)
            signed_tick = self._apply_tick_rule(price)
            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_array.append(imbalance)
            cum_theta += imbalance
            if not list_bars and np.isnan(self.expected_imbalance):
                self.expected_imbalance = self.get_expected_imbalance(
                    self.exp_num_ticks, self.imbalance_array)
            self.update_cache(date_time, price, low_price,
                               high_price, cum_ticks, cum_volume, cum_theta)
            self._update_ticks_in_bar(row)
            if np.abs(cum_theta) > self.exp_num_ticks * np.abs(self.expected_imbalance):
                self._compute_additional_features()
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                self.num_ticks_bar.append(cum_ticks)
                self.exp_num_ticks = ewma(np.array(
                    self.num_ticks_bar[-self.num_prev_bar:], dtype=float), self.num_prev_bar)[-1]
                self.expected_imbalance = self.get_expected_imbalance(self.exp_num_ticks * self.num_prev_bar,
                                                                       self.imbalance_array)
                cum_ticks, cum_volume, cum_theta = 0, 0, 0
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()
                high_price, low_price = -np.inf, np.inf
                self.cache = []
        return list_bars
    def update_counters(self):
        if self.flag and self.cache:
            latest_entry = self.cache[-1]
            cum_ticks = int(latest_entry.cum_ticks)
            cum_volume = int(latest_entry.cum_volume)
            low_price = np.float64(latest_entry.low)
            high_price = np.float64(latest_entry.high)
            cum_theta = np.float64(latest_entry.cum_theta)
        else:
            cum_ticks, cum_theta, cum_volume = 0, 0, 0
            high_price, low_price = -np.inf, np.inf
        return cum_ticks, cum_volume, cum_theta, high_price, low_price

    def update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_theta):
        cache_data = self.cache_tuple(date_time=date_time,
                                      price=price, high=high_price,
                                      low=low_price,
                                      cum_ticks=cum_ticks,
                                      cum_volume=cum_volume,
                                      cum_theta=cum_theta)
        self.cache.append(cache_data)

    def get_expected_imbalance(self, window, imbalance_array):
        if len(imbalance_array) < self.exp_num_ticks_init: ewma_window = np.nan
        else: ewma_window = int(min(len(imbalance_array), window))
        if np.isnan(ewma_window): expected_imbalance = np.nan
        else: expected_imbalance = ewma(np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]
        return expected_imbalance
class getImbalanceBar :
    def __init__(self, path : str,
                 num_prev_bar : int,
                 exp_num_ticks_init : int =100000,
                 batch_size = 2e7,
                 verbose : bool = True,
                 to_csv : bool = False,
                 output_path : str = None,
                 additional_features = None):
        self._path = path
        self._num_prev_bar = num_prev_bar
        self._exp_num_ticks_init = exp_num_ticks_init
        self._batch_size = batch_size
        self._verbose = verbose
        self._to_csv = to_csv
        self._output_path = output_path
        self._additional_features = additional_features
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, value : str):
        self._path = value
    @property
    def num_prev_bar(self):
        return self._num_prev_bar
    @num_prev_bar.setter
    def num_prev_bar(self, value : int):
        self._num_prev_bar = value
    @property
    def exp_num_ticks_init(self):
        return self._exp_num_ticks_init
    @exp_num_ticks_init.setter
    def exp_num_ticks_init(self, value : int):
        self._exp_num_ticks_init = value
    def dollar_imbalance_bar(self):
        bars = ImbalanceBars(path = self._path,
                             metric = 'dollar_imbalance',
                             num_prev_bar = self._num_prev_bar,
                             exp_num_ticks_init = self._exp_num_ticks_init,
                             batch_size = self._batch_size,
                             additional_features = self._additional_features)
        imbalanceBar = bars.batch_run(verbose = self._verbose, to_csv = self._to_csv, output_path = self._output_path)
        return imbalanceBar
    def volume_imbalance_bar(self):
        bars = ImbalanceBars(path = self._path,
                             metric = 'volume_imbalance',
                             num_prev_bar = self._num_prev_bar,
                             exp_num_ticks_init = self._exp_num_ticks_init,
                             batch_size = self._batch_size,
                             additional_features = self._additional_features)
        imbalanceBar = bars.batch_run(verbose = self._verbose, to_csv = self._to_csv, output_path = self._output_path)
        return imbalanceBar
    def tick_imbalance_bar(self):
        bars = ImbalanceBars(path = self._path,
                             metric = 'tick_imbalance',
                             num_prev_bar = self._num_prev_bar,
                             exp_num_ticks_init = self._exp_num_ticks_init,
                             batch_size = self._batch_size,
                             additional_features = self._additional_features)
        imbalanceBar = bars.batch_run(verbose = self._verbose, to_csv = self._to_csv, output_path = self._output_path)

        return imbalanceBar