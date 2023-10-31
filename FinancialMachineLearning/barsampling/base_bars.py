from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.multiprocess.fast_ewma import ewma

def _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object
class BaseBars(ABC):
    def __init__(self, metric: str, batch_size: int = 2e7):
        self.metric = metric
        self.batch_size = batch_size
        self.prev_tick_rule = 0
        self.open_price, self.prev_price, self.close_price = None, None, None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.tick_num = 0
        self.flag = False
    def batch_run(self,
                  file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                  verbose: bool = True,
                  to_csv: bool = False,
                  output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        if to_csv is True:
            header = True
            open(output_path, 'w').close()

        if verbose:
            print('Reading data in batches:')

        count = 0
        final_bars = []
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume', 'cum_ticks',
                'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:
                print('Batch number:', count)

            list_bars = self.run(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                final_bars += list_bars
            count += 1

        if verbose:
            print('Returning bars \n')

        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df
        return None

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        if isinstance(file_path_or_df, (list, tuple)):
            for file_path in file_path_or_df:
                self._read_first_row(file_path)
            for file_path in file_path_or_df:
                for batch in pd.read_csv(file_path, chunksize=self.batch_size, parse_dates=[0]):
                    yield batch

        elif isinstance(file_path_or_df, str):
            self._read_first_row(file_path_or_df)
            for batch in pd.read_csv(file_path_or_df, chunksize=self.batch_size, parse_dates=[0]):
                yield batch

        elif isinstance(file_path_or_df, pd.DataFrame):
            for batch in _crop_data_frame_in_batches(file_path_or_df, self.batch_size):
                yield batch

        else:
            raise ValueError('file_path_or_df is neither string(path to a csv file), iterable of strings, nor pd.DataFrame')

    def _read_first_row(self, file_path: str):
        first_row = pd.read_csv(file_path, nrows=1)
        self._assert_csv(first_row)

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        if isinstance(data, (list, tuple)):
            values = data

        elif isinstance(data, pd.DataFrame):
            values = data.values

        else:
            raise ValueError('data is neither list nor tuple nor pd.DataFrame')

        list_bars = self._extract_bars(data=values)

        # Set flag to True: notify function to use cache
        self.flag = True

        return list_bars

    @abstractmethod
    def _extract_bars(self, data: pd.DataFrame) -> list:
        pass
    @abstractmethod
    def _reset_cache(self):
        pass

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])

    def _update_high_low(self, price: float) -> Union[float, float]:
        if price > self.high_price:
            high_price = price
        else:
            high_price = self.high_price

        if price < self.low_price:
            low_price = price
        else:
            low_price = self.low_price

        return high_price, low_price

    def _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) -> None:
        open_price = self.open_price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        volume = self.cum_statistics['cum_volume']
        cum_buy_volume = self.cum_statistics['cum_buy_volume']
        cum_ticks = self.cum_statistics['cum_ticks']
        cum_dollar_value = self.cum_statistics['cum_dollar_value']

        list_bars.append(
            [date_time, self.tick_num, open_price, high_price, low_price, close_price, volume, cum_buy_volume,
             cum_ticks,
             cum_dollar_value])

    def _apply_tick_rule(self, price: float) -> int:
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        self.prev_price = price
        return signed_tick

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        if self.metric == 'tick_imbalance' or self.metric == 'tick_run':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance' or self.metric == 'dollar_run':
            imbalance = signed_tick * volume * price
        elif self.metric == 'volume_imbalance' or self.metric == 'volume_run':
            imbalance = signed_tick * volume
        else:
            raise ValueError('Unknown imbalance metric, possible values are tick/dollar/volume imbalance/run')
        return imbalance


class BaseImbalanceBars(BaseBars):
    def __init__(self, metric: str,
                 batch_size: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int,
                 analyse_thresholds: bool):
        BaseBars.__init__(self, metric, batch_size)

        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta': 0, 'expected_imbalance': np.nan, 'exp_num_ticks': exp_num_ticks_init}

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array': []}

        if analyse_thresholds is True:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

    def _reset_cache(self):
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta'] = 0

    def _extract_bars(self, data: Tuple[dict, pd.DataFrame]) -> list:
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

            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_tick_statistics['imbalance_array'].append(imbalance)
            self.thresholds['cum_theta'] += imbalance

            if not list_bars and np.isnan(self.thresholds['expected_imbalance']):
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            if np.abs(self.thresholds['cum_theta']) > self.thresholds['exp_num_ticks'] * np.abs(
                    self.thresholds['expected_imbalance']):
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                # Expected number of ticks based on formed bars
                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()
                # Get expected imbalance
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)
                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, window: int):
        if len(self.imbalance_tick_statistics['imbalance_array']) < self.thresholds['exp_num_ticks']:
            ewma_window = np.nan
        else:
            ewma_window = int(min(len(self.imbalance_tick_statistics['imbalance_array']), window))

        if np.isnan(ewma_window):
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(self.imbalance_tick_statistics['imbalance_array'][-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        pass


class BaseRunBars(BaseBars):
    def __init__(self, metric: str, batch_size: int, num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int, analyse_thresholds: bool):
        BaseBars.__init__(self, metric, batch_size)

        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta_buy': 0, 'cum_theta_sell': 0, 'exp_imbalance_buy': np.nan,
                           'exp_imbalance_sell': np.nan, 'exp_num_ticks': exp_num_ticks_init,
                           'exp_buy_ticks_proportion': np.nan, 'buy_ticks_num': 0}

        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array_buy': [], 'imbalance_array_sell': [],
                                          'buy_ticks_proportion': []}

        if analyse_thresholds: self.bars_thresholds = []
        else: self.bars_thresholds = None
        self.warm_up_flag = False

    def _reset_cache(self):
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'], self.thresholds['buy_ticks_num'] = 0, 0, 0

    def _extract_bars(self, data: Tuple[list, np.ndarray]) -> list:
        list_bars = []
        for row in data:
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
            imbalance = self._get_imbalance(price, signed_tick, volume)

            if imbalance > 0:
                self.imbalance_tick_statistics['imbalance_array_buy'].append(imbalance)
                self.thresholds['cum_theta_buy'] += imbalance
                self.thresholds['buy_ticks_num'] += 1
            elif imbalance < 0:
                self.imbalance_tick_statistics['imbalance_array_sell'].append(abs(imbalance))
                self.thresholds['cum_theta_sell'] += abs(imbalance)

            self.warm_up_flag = np.isnan([self.thresholds['exp_imbalance_buy'], self.thresholds[
                'exp_imbalance_sell']]).any()
            if not list_bars and self.warm_up_flag:
                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window, warm_up=True)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window,
                    warm_up=True)

                if bool(np.isnan([self.thresholds['exp_imbalance_buy'],
                                  self.thresholds['exp_imbalance_sell']]).any()) is False:
                    self.thresholds['exp_buy_ticks_proportion'] = self.thresholds['buy_ticks_num'] / \
                                                                  self.cum_statistics['cum_ticks']

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            max_proportion = max(
                self.thresholds['exp_imbalance_buy'] * self.thresholds['exp_buy_ticks_proportion'],
                self.thresholds['exp_imbalance_sell'] * (1 - self.thresholds['exp_buy_ticks_proportion']))
            max_theta = max(self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'])
            if max_theta > self.thresholds['exp_num_ticks'] * max_proportion and not np.isnan(max_proportion):
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                self.imbalance_tick_statistics['buy_ticks_proportion'].append(
                    self.thresholds['buy_ticks_num'] / self.cum_statistics['cum_ticks'])

                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()

                exp_buy_ticks_proportion = ewma(
                    np.array(self.imbalance_tick_statistics['buy_ticks_proportion'][-self.num_prev_bars:], dtype=float),
                    self.num_prev_bars)[-1]
                self.thresholds['exp_buy_ticks_proportion'] = exp_buy_ticks_proportion

                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window)

                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        if len(array) < self.thresholds['exp_num_ticks'] and warm_up is True:
            ewma_window = np.nan
        else: ewma_window = int(min(len(array), window))

        if np.isnan(ewma_window):
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(array[-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        pass
