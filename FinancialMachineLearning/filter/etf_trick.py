import pandas as pd
import numpy as np

class etfTrick :
    def __init__(self, open, close, alloc, costs, rates, index_col = 0):
        self.index_col = index_col
        self.prev_k = 1.0
        self.prev_allocs_change = False
        self.prev_h = None
        self.data_dict = {}
        self.iter_dict = None
        self.init_fields = None

        if isinstance(alloc, str):
            self.init_fields = {'open_df': open, 'close_df': close, 'alloc_df': alloc, 'costs_df': costs,
                                'rates_df': rates, 'index_col': index_col}
            self.iter_dict = dict.fromkeys(['open', 'close', 'alloc', 'costs', 'rates'], None)
            self.iter_dict['open'] = pd.read_csv(open,
                                                 iterator=True,
                                                 index_col=self.index_col,
                                                 parse_dates=[self.index_col])
            self.iter_dict['close'] = pd.read_csv(close,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])
            self.iter_dict['alloc'] = pd.read_csv(alloc,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])
            self.iter_dict['costs'] = pd.read_csv(costs,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])
            if rates is not None:
                self.iter_dict['rates'] = pd.read_csv(rates,
                                                      iterator=True,
                                                      index_col=self.index_col,
                                                      parse_dates=[self.index_col])
            self.securities = list(pd.read_csv(alloc, nrows=0, header=0, index_col=self.index_col))

        elif isinstance(alloc, pd.DataFrame):
            self.data_dict['open'] = open
            self.data_dict['close'] = close
            self.data_dict['alloc'] = alloc
            self.data_dict['costs'] = costs
            self.data_dict['rates'] = rates
            self.securities = self.data_dict['alloc'].columns

            if rates is None:
                self.data_dict['rates'] = open.copy()
                self.data_dict['rates'][self.securities] = 1.0
            for df_name in self.data_dict:
                self.data_dict[df_name] = self.data_dict[df_name][self.securities]

            self._index_check()
        else:
            raise TypeError('Wrong input to ETFTrick class. Either strings with paths to csv files, or pd.DataFrames')

        self.prev_allocs = np.array([np.nan for _ in range(0, len(self.securities))])  # Init weights with nan values

    def _append_previous_rows(self, cache):
        max_prev_index = cache['open'].index.max()
        second_max_prev_index = cache['open'].index[-2]
        for df_name in self.data_dict:
            temp_df = self.data_dict[df_name]
            temp_df.loc[max_prev_index, :] = cache[df_name].iloc[-1]
            self.data_dict[df_name] = temp_df
        self.data_dict['close'].loc[second_max_prev_index, :] = cache['close'].loc[second_max_prev_index, :]
        for df_name in self.data_dict:
            self.data_dict[df_name].sort_index(inplace=True)
            self.data_dict[df_name] = self.data_dict[df_name][self.securities]
        price_diff = self.data_dict['close'].diff().iloc[1:]
        self.data_dict['close'] = self.data_dict['close'].iloc[1:]
        return price_diff

    def generate_trick_components(self, cache=None):
        if cache:
            price_diff = self._append_previous_rows(cache)
        else:
            price_diff = self.data_dict['close'].diff()
        next_open_df = self.data_dict['open'].shift(-1)
        close_open_diff = self.data_dict['close'].sub(self.data_dict['open'])
        self.data_dict['alloc']['abs_w_sum'] = self.data_dict['alloc'].abs().sum(
            axis=1)
        delever_df = self.data_dict['alloc'].div(self.data_dict['alloc']['abs_w_sum'], axis='index')
        next_open_mul_rates_df = next_open_df.mul(self.data_dict['rates'], axis='index')
        h_without_k = delever_df.div(next_open_mul_rates_df)
        weights_df = self.data_dict['alloc'][self.securities]
        h_without_k = h_without_k[self.securities]
        close_open_diff = close_open_diff[self.securities]
        price_diff = price_diff[self.securities]
        return pd.concat(
            [weights_df, h_without_k, close_open_diff, price_diff, self.data_dict['costs'], self.data_dict['rates']],
            axis=1,
            keys=[
                'w', 'h_t', 'close_open', 'price_diff', 'costs',
                'rate'])
    def _update_cache(self):
        cache_dict = {'open': self.data_dict['open'].iloc[-2:], 'close': self.data_dict['close'].iloc[-2:],
                      'alloc': self.data_dict['alloc'].iloc[-2:], 'costs': self.data_dict['costs'].iloc[-2:],
                      'rates': self.data_dict['rates'].iloc[-2:]}
        return cache_dict
    def _chunk_loop(self, data_df):
        etf_series = pd.Series()
        for index, row in zip(data_df.index, data_df.values):
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(row,6)
            weights_arr = np.nan_to_num(weights_arr)
            allocs_change = bool(~(self.prev_allocs == weights_arr).all())
            if self.prev_allocs_change is True:
                delta = close_open
            else:
                delta = price_diff
            if self.prev_h is None:
                self.prev_h = h_t * self.prev_k
                etf_series[index] = self.prev_k
            else:
                if self.prev_allocs_change is True:
                    self.prev_h = h_t * self.prev_k

                k = self.prev_k + \
                    np.nansum(self.prev_h * rate * (delta + costs))
                etf_series[index] = k

                self.prev_k = k
                self.prev_allocs_change = allocs_change
                self.prev_allocs = weights_arr
        return etf_series

    def _index_check(self):
        for temp_df in self.data_dict.values():
            if self.data_dict['open'].index.difference(temp_df.index).shape[0] != 0 or \
                    self.data_dict['open'].shape != temp_df.shape:
                raise ValueError('DataFrames indices are different')

    def _get_batch_from_csv(self, batch_size):
        self.data_dict['open'] = self.iter_dict['open'].get_chunk(batch_size)
        self.data_dict['close'] = self.iter_dict['close'].get_chunk(batch_size)
        self.data_dict['alloc'] = self.iter_dict['alloc'].get_chunk(batch_size)
        self.data_dict['costs'] = self.iter_dict['costs'].get_chunk(batch_size)

        if self.iter_dict['rates'] is not None:
            self.data_dict['rates'] = self.iter_dict['rates'].get_chunk(batch_size)
        else:
            self.data_dict['rates'] = self.data_dict['open'].copy()
            self.data_dict['rates'][self.securities] = 1.0
        for df_name in self.data_dict:
            self.data_dict[df_name] = self.data_dict[df_name][self.securities]

        self._index_check()

    def _rewind_etf_trick(self, alloc_df, etf_series):
        self.prev_k = etf_series.iloc[-2]
        self.prev_allocs = alloc_df.iloc[-2]
        self.prev_allocs_change = bool(~(self.prev_allocs == alloc_df.iloc[-3]).all())

    def _csv_file_etf_series(self, batch_size):
        etf_series = pd.Series()
        self._get_batch_from_csv(batch_size)
        data_df = self.generate_trick_components(cache=None)
        cache = self._update_cache()
        data_df = data_df.iloc[1:]
        omit_last_row = False
        while True:
            try:
                chunk_etf_series = self._chunk_loop(data_df)
                if omit_last_row is True:
                    etf_series = etf_series.iloc[:-1]
                etf_series = etf_series.append(chunk_etf_series)
                self._get_batch_from_csv(batch_size)
                self._rewind_etf_trick(data_df['w'], etf_series)
                data_df = self.generate_trick_components(cache)
                cache = self._update_cache()
                omit_last_row = True
            except StopIteration:
                return etf_series

    def _in_memory_etf_series(self):
        data_df = self.generate_trick_components()
        data_df = data_df.iloc[1:]
        return self._chunk_loop(data_df)

    def get_etf_series(self, batch_size=1e5):
        if self.iter_dict is None:
            etf_trick_series = self._in_memory_etf_series()
        else:
            if batch_size < 3:
                raise ValueError('Batch size should be >= 3')
            etf_trick_series = self._csv_file_etf_series(batch_size)
        return etf_trick_series

    def reset(self):
        self.__init__(**self.init_fields)

    def get_futures_roll_series(data_df, open_col, close_col, sec_col, current_sec_col, roll_backward=False):
        filtered_df = data_df[data_df[sec_col] == data_df[current_sec_col]]
        filtered_df.sort_index(inplace=True)
        roll_dates = filtered_df[current_sec_col].drop_duplicates(keep='first').index
        gaps = filtered_df[close_col] * 0
        gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] - filtered_df[close_col].loc[
            roll_dates[1:]]
        gaps = gaps.cumsum()
        if roll_backward:
            gaps -= gaps.iloc[-1]

        return gaps
