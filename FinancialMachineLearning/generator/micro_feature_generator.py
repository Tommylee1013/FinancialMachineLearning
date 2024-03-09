import pandas as pd
import numpy as np
from FinancialMachineLearning.features.entropy import shannon_entropy, plug_in_entropy, lempel_ziv_entropy, konto_entropy
from FinancialMachineLearning.features.encoding import encode_array
from FinancialMachineLearning.features.microstructure import trades_based_kyle_lambda, trades_based_amihud_lambda, trades_based_hasbrouck_lambda, get_avg_tick_size, volume_weighted_average_price
from FinancialMachineLearning.features.encoding import encode_tick_rule_array
from FinancialMachineLearning.utils.misc import crop_data_frame_in_batches

class MicrostructuralFeaturesGenerator:
    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None):
        if isinstance(trades_input, str):
            self.generator_object = pd.read_csv(trades_input, chunksize=batch_size, parse_dates=[0])
            first_row = pd.read_csv(trades_input, nrows=1)
            self._assert_csv(first_row)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = crop_data_frame_in_batches(trades_input, batch_size)
        else:
            raise ValueError('trades_input is neither string(path to a csv file) nor pd.DataFrame')

        self.tick_num_generator = iter(tick_num_series)
        self.current_bar_tick_num = self.tick_num_generator.__next__()

        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv', 'konto']

        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        if to_csv is True:
            header = True
            open(output_path, 'w').close()
        count = 0
        final_bars = []
        cols = ['date_time', 'avg_tick_size', 'tick_rule_sum', 'vwap', 'kyle_lambda', 'amihud_lambda',
                'hasbrouck_lambda']

        for en_type in self.entropy_types:
            cols += ['tick_rule_entropy_' + en_type]

        if self.volume_encoding is not None:
            for en_type in self.entropy_types:
                cols += ['volume_entropy_' + en_type]

        if self.pct_encoding is not None:
            for en_type in self.entropy_types:
                cols += ['pct_entropy_' + en_type]

        for batch in self.generator_object:
            if verbose:
                print('Batch number:', count)

            list_bars, stop_flag = self._extract_bars(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header = header, index=False, mode='a')
                header = False
            else: final_bars += list_bars
            count += 1

            if stop_flag is True:
                break

        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        return None

    def _reset_cache(self):
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

    def _extract_bars(self, data):
        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            self.tick_num += 1

            # Derivative variables
            price_diff = self._get_price_diff(price)
            log_ret = self._get_log_ret(price)

            self.price_diff.append(price_diff)
            self.trade_size.append(volume)
            self.tick_rule.append(signed_tick)
            self.dollar_size.append(dollar_value)
            self.log_ret.append(log_ret)

            self.prev_price = price

            # If date_time reached bar index
            if self.tick_num >= self.current_bar_tick_num:
                self._get_bar_features(date_time, list_bars)

                # Take the next tick number
                try:
                    self.current_bar_tick_num = self.tick_num_generator.__next__()
                except StopIteration:
                    return list_bars, True  # Looped through all bar index
                # Reset cache
                self._reset_cache()
        return list_bars, False

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) -> list:
        features = [date_time]
        features.append(get_avg_tick_size(self.trade_size))
        features.append(sum(self.tick_rule))
        features.append(volume_weighted_average_price(self.dollar_size, self.trade_size))

        features.append(trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))
        features.append(trades_based_amihud_lambda(self.log_ret, self.dollar_size))
        features.append(
            trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))

        encoded_tick_rule_message = encode_tick_rule_array(self.tick_rule)
        features.append(shannon_entropy(encoded_tick_rule_message))
        features.append(plug_in_entropy(encoded_tick_rule_message))
        features.append(lempel_ziv_entropy(encoded_tick_rule_message))
        features.append(konto_entropy(encoded_tick_rule_message))

        if self.volume_encoding is not None:
            message = encode_array(self.trade_size, self.volume_encoding)
            features.append(shannon_entropy(message))
            features.append(plug_in_entropy(message))
            features.append(lempel_ziv_entropy(message))
            features.append(konto_entropy(message))

        if self.pct_encoding is not None:
            message = encode_array(self.log_ret, self.pct_encoding)
            features.append(shannon_entropy(message))
            features.append(plug_in_entropy(message))
            features.append(lempel_ziv_entropy(message))
            features.append(konto_entropy(message))

        list_bars.append(features)

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

        return signed_tick

    def _get_price_diff(self, price: float) -> float:
        if self.prev_price is not None:
            price_diff = price - self.prev_price
        else:
            price_diff = 0
        return price_diff

    def _get_log_ret(self, price: float) -> float:
        if self.prev_price is not None:
            log_ret = np.log(price / self.prev_price)
        else:
            log_ret = 0
        return log_ret

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