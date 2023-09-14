from typing import Union, Iterable, List, Optional
import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.base_bars import BaseRunBars
from FinancialMachineLearning.multiprocess.fast_ewma import ewma

class EMARunBars(BaseRunBars):
    def __init__(self, metric: str,
                 num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int,
                 exp_num_ticks_constraints: List[float],
                 batch_size: int,
                 analyse_thresholds: bool):
        BaseRunBars.__init__(self, metric, batch_size, num_prev_bars, expected_imbalance_window,
                             exp_num_ticks_init, analyse_thresholds)

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


class ConstRunBars(BaseRunBars):
    def __init__(self, metric: str,
                 num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int,
                 batch_size: int,
                 analyse_thresholds: bool):
        BaseRunBars.__init__(self, metric, batch_size, num_prev_bars, expected_imbalance_window,
                             exp_num_ticks_init,
                             analyse_thresholds)

    def _get_exp_num_ticks(self):
        return self.thresholds['exp_num_ticks']


def get_ema_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                            expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                            exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                            analyse_thresholds: bool = False,
                            verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = EMARunBars(metric='dollar_run', num_prev_bars=num_prev_bars,
                      expected_imbalance_window=expected_imbalance_window,
                      exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                      batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_ema_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                            expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                            exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                            analyse_thresholds: bool = False,
                            verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    bars = EMARunBars(metric='volume_run', num_prev_bars=num_prev_bars,
                      expected_imbalance_window=expected_imbalance_window,
                      exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                      batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)
def get_ema_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                          num_prev_bars: int = 3,
                          expected_imbalance_window: int = 10000,
                          exp_num_ticks_init: int = 20000,
                          exp_num_ticks_constraints: List[float] = None,
                          batch_size: int = 2e7,
                          analyse_thresholds: bool = False,
                          verbose: bool = True,
                          to_csv: bool = False,
                          output_path: Optional[str] = None):
    bars = EMARunBars(metric='tick_run', num_prev_bars=num_prev_bars,
                      expected_imbalance_window=expected_imbalance_window,
                      exp_num_ticks_init=exp_num_ticks_init, exp_num_ticks_constraints=exp_num_ticks_constraints,
                      batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)
def get_const_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                              num_prev_bars: int,
                              expected_imbalance_window: int = 10000,
                              exp_num_ticks_init: int = 20000,
                              batch_size: int = 2e7,
                              analyse_thresholds: bool = False,
                              verbose: bool = True,
                              to_csv: bool = False,
                              output_path: Optional[str] = None):
    bars = ConstRunBars(metric='dollar_run', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)


def get_const_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                              num_prev_bars: int,
                              expected_imbalance_window: int = 10000,
                              exp_num_ticks_init: int = 20000,
                              batch_size: int = 2e7,
                              analyse_thresholds: bool = False,
                              verbose: bool = True,
                              to_csv: bool = False,
                              output_path: Optional[str] = None):
    bars = ConstRunBars(metric='volume_run', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)

def get_const_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                            num_prev_bars: int,
                            expected_imbalance_window: int = 10000,
                            exp_num_ticks_init: int = 20000,
                            batch_size: int = 2e7,
                            analyse_thresholds: bool = False,
                            verbose: bool = True,
                            to_csv: bool = False,
                            output_path: Optional[str] = None):
    bars = ConstRunBars(metric='tick_run', num_prev_bars=num_prev_bars,
                        expected_imbalance_window=expected_imbalance_window,
                        exp_num_ticks_init=exp_num_ticks_init,
                        batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    run_bars = bars.batch_run(file_path_or_df=file_path_or_df,
                              verbose=verbose, to_csv=to_csv, output_path=output_path)

    return run_bars, pd.DataFrame(bars.bars_thresholds)
