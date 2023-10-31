from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.base_bars import BaseBars

class StandardBars(BaseBars):
    def __init__(self, metric: str,
                 threshold: int = 50000,
                 batch_size: int = 20000000):

        BaseBars.__init__(self, metric, batch_size)
        self.threshold = threshold

    def _reset_cache(self):
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        list_bars = []

        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = np.float64(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            self.high_price, self.low_price = self._update_high_low(price)

            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            if self.cum_statistics[self.metric] >= self.threshold:
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)
                self._reset_cache()
        return list_bars

def dollar_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
               threshold: float = 1000000,
               batch_size: int = 1000000,
               verbose: bool = True,
               to_csv: bool = False,
               output_path: Optional[str] = None):
    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars

def volume_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
               threshold: float = 10000,
               batch_size: int = 1000000,
               verbose: bool = True,
               to_csv: bool = False,
               output_path: Optional[str] = None):
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars

def tick_bar(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
             threshold: float = 600,
             batch_size: int = 1000000,
             verbose: bool = True,
             to_csv: bool = False,
             output_path: Optional[str] = None):
    bars = StandardBars(metric='cum_ticks',
                        threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars
