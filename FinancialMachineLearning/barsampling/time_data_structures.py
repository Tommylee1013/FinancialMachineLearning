from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.base_bars import BaseBars

class TimeBars(BaseBars):
    def __init__(self, resolution: str, num_units: int, batch_size: int = 20000000):

        BaseBars.__init__(self, metric=None, batch_size=batch_size)

        self.time_bar_thresh_mapping = {'D': 86400, 'H': 3600, 'MIN': 60, 'S': 1}
        assert resolution in self.time_bar_thresh_mapping, "{} resolution is not implemented".format(resolution)
        self.resolution = resolution
        self.num_units = num_units
        self.threshold = self.num_units * self.time_bar_thresh_mapping[self.resolution]
        self.timestamp = None

    def _reset_cache(self):
        self.open_price = None
        self.close_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        list_bars = []

        for row in data:
            date_time = row[0].timestamp()
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            timestamp_threshold = (int(
                float(date_time)) // self.threshold + 1) * self.threshold
            if self.timestamp is None:
                self.timestamp = timestamp_threshold
            elif self.timestamp < timestamp_threshold:
                self._create_bars(self.timestamp, self.close_price,
                                  self.high_price, self.low_price, list_bars)

                self._reset_cache()
                self.timestamp = timestamp_threshold

            if self.open_price is None:
                self.open_price = price

            self.high_price, self.low_price = self._update_high_low(price)

            self.close_price = price

            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

        return list_bars

def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = 'D', num_units: int = 1, batch_size: int = 20000000,
                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = TimeBars(resolution=resolution, num_units=num_units, batch_size=batch_size)
    time_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return time_bars
