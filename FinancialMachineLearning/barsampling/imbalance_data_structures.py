from typing import Union, Iterable, List, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.base_bars import BaseImbalanceBars
from FinancialMachineLearning.multiprocess.fast_ewma import ewma
class EMAImbalanceBars(BaseImbalanceBars):
    def __init__(self, metric: str,
                 num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int,
                 exp_num_ticks_constraints: List,
                 batch_size: int,
                 analyse_thresholds: bool):
        BaseImbalanceBars.__init__(self, metric, batch_size, expected_imbalance_window,
                                   exp_num_ticks_init, analyse_thresholds)

        self.num_prev_bars = num_prev_bars
        if exp_num_ticks_constraints is None:
            self.min_exp_num_ticks = 0
            self.max_exp_num_ticks = np.inf
        else:
            self.min_exp_num_ticks = exp_num_ticks_constraints[0]
            self.max_exp_num_ticks = exp_num_ticks_constraints[1]

    def _get_exp_num_ticks(self):
        prev_num_of_ticks = self.imbalance_tick_statistics['num_ticks_bar']
        exp_num_ticks = ewma(np.array(
            prev_num_of_ticks[-self.num_prev_bars:], dtype=float), self.num_prev_bars)[-1]
        return min(max(exp_num_ticks, self.min_exp_num_ticks), self.max_exp_num_ticks)


class ConstImbalanceBars(BaseImbalanceBars):
    def __init__(self, metric: str,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int,
                 batch_size: int,
                 analyse_thresholds: bool):
        BaseImbalanceBars.__init__(self, metric, batch_size,
                                   expected_imbalance_window,
                                   exp_num_ticks_init,
                                   analyse_thresholds)

    def _get_exp_num_ticks(self):
        return self.thresholds['exp_num_ticks']


def ema_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                              num_prev_bars: int = 3,
                              expected_imbalance_window: int = 10000,
                              exp_num_ticks_init: int = 20000,
                              exp_num_ticks_constraints: List[float] = None,
                              batch_size: int = 2e7,
                              analyse_thresholds: bool = False,
                              verbose: bool = True,
                              to_csv: bool = False,
                              output_path: Optional[str] = None):
    bars = EMAImbalanceBars(metric='dollar_imbalance', num_prev_bars=num_prev_bars,
                            expected_imbalance_window=expected_imbalance_window,
                            exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                            batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)
    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)
def ema_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                                  expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                                  exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                                  analyse_thresholds: bool = False,
                                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = EMAImbalanceBars(metric='volume_imbalance', num_prev_bars=num_prev_bars,
                            expected_imbalance_window=expected_imbalance_window,
                            exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                            batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)

    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

def ema_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                                expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                                exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                                analyse_thresholds: bool = False,
                                verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):

    bars = EMAImbalanceBars(metric='tick_imbalance', num_prev_bars=num_prev_bars,
                            expected_imbalance_window=expected_imbalance_window,
                            exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                            batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)

    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)


def const_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int = 10000,
                                    exp_num_ticks_init: int = 20000,
                                    batch_size: int = 2e7, analyse_thresholds: bool = False,
                                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = ConstImbalanceBars(metric='dollar_imbalance',
                              expected_imbalance_window=expected_imbalance_window,
                              exp_num_ticks_init=exp_num_ticks_init,
                              batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)

    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)


def const_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int = 10000,
                                    exp_num_ticks_init: int = 20000,
                                    batch_size: int = 2e7, analyse_thresholds: bool = False,
                                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = ConstImbalanceBars(metric='volume_imbalance',
                              expected_imbalance_window=expected_imbalance_window,
                              exp_num_ticks_init=exp_num_ticks_init,
                              batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)

    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)


def const_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int = 10000,
                                  exp_num_ticks_init: int = 20000,
                                  batch_size: int = 2e7, analyse_thresholds: bool = False,
                                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = ConstImbalanceBars(metric='tick_imbalance',
                              expected_imbalance_window=expected_imbalance_window,
                              exp_num_ticks_init=exp_num_ticks_init,
                              batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    imbalance_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                                    verbose=verbose, to_csv=to_csv, output_path=output_path)

    return imbalance_bars, pd.DataFrame(bars.bars_thresholds)
