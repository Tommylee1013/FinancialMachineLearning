import numpy as np
import pandas as pd
from FinancialMachineLearning.barsampling.imbalance_data_structures import *
from FinancialMachineLearning.barsampling.standard_data_structures import *
from FinancialMachineLearning.barsampling.run_data_structures import *
from FinancialMachineLearning.barsampling.time_data_structures import *

class StandardBarFeatures :
    def __init__(self,
                 file_path_or_df : Union[str, Iterable[str], pd.DataFrame],
                 batch_size : int = 1000000,
                 verbose : bool = True,
                 to_csv: bool = False):
        self.file_path_or_df = file_path_or_df
        self.batch_size = batch_size
        self.verbose = verbose
        self.to_csv = to_csv
    def dollar_bar(self, threshold: float = 1000000,
                   output_path: Optional[str] = None):
        bars = StandardBars(metric='cum_dollar_value',
                            threshold = threshold,
                            batch_size = self.batch_size)
        dollar_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                     verbose = self.verbose,
                                     to_csv = self.to_csv,
                                     output_path = output_path)
        return dollar_bars
    def volume_bar(self, threshold: float = 10000,
                   output_path: Optional[str] = None):
        bars = StandardBars(metric='cum_volume',
                            threshold = threshold,
                            batch_size = self.batch_size)
        volume_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                     verbose = self.verbose,
                                     to_csv = self.to_csv,
                                     output_path = output_path)
        return volume_bars

    def tick_bar(self, threshold: float = 600,
                 output_path: Optional[str] = None):
        bars = StandardBars(metric = 'cum_ticks',
                            threshold = threshold,
                            batch_size = self.batch_size)
        tick_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                   verbose = self.verbose,
                                   to_csv = self.to_csv,
                                   output_path = output_path)
        return tick_bars

class ImbalanceBarFeatures :
    def __init__(self, file_path_or_df : Union[str, Iterable[str], pd.DataFrame],
                 num_prev_bars : int = 3,
                 exp_num_ticks_constraints: List[float] = None,
                 batch_size : int = 10000000,
                 analyse_thresholds: bool = False,
                 verbose: bool = True,
                 to_csv: bool = False,
                 output_path: Optional[str] = None):
        self.file_path_or_df = file_path_or_df
        self.num_prev_bars = num_prev_bars
        self.exp_num_ticks_constraints = exp_num_ticks_constraints
        self.batch_size = batch_size
        self.analyse_thresholds = analyse_thresholds
        self.verbose = verbose
        self.to_csv = to_csv
        self.output_path = output_path

    def ema_dollar_imbalance_bar(self, expected_imbalance_window: int = 50,
                                  exp_num_ticks_init: int = 100):
        bars = EMAImbalanceBars(metric='dollar_imbalance', num_prev_bars = self.num_prev_bars,
                                expected_imbalance_window = expected_imbalance_window,
                                exp_num_ticks_init = exp_num_ticks_init,
                                exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                                batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)
        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

    def ema_volume_imbalance_bar(self, expected_imbalance_window: int = 50,
                                 exp_num_ticks_init: int = 100):
        bars = EMAImbalanceBars(metric='volume_imbalance', num_prev_bars = self.num_prev_bars,
                                expected_imbalance_window = expected_imbalance_window,
                                exp_num_ticks_init = exp_num_ticks_init,
                                exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                                batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv,
                                        output_path = self.output_path)

        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

    def ema_tick_imbalance_bar(self, expected_imbalance_window: int = 100,
                               exp_num_ticks_init: int = 200):
        bars = EMAImbalanceBars(metric='tick_imbalance', num_prev_bars = self.num_prev_bars,
                                expected_imbalance_window = expected_imbalance_window,
                                exp_num_ticks_init = exp_num_ticks_init,
                                exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                                batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

    def const_dollar_imbalance_bar(self, expected_imbalance_window: int = 50,
                                   exp_num_ticks_init: int = 100):
        bars = ConstImbalanceBars(metric = 'dollar_imbalance',
                                  expected_imbalance_window = expected_imbalance_window,
                                  exp_num_ticks_init = exp_num_ticks_init,
                                  batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

    def const_volume_imbalance_bar(self, expected_imbalance_window: int = 50,
                                   exp_num_ticks_init: int = 100):
        bars = ConstImbalanceBars(metric = 'volume_imbalance',
                                  expected_imbalance_window = expected_imbalance_window,
                                  exp_num_ticks_init = exp_num_ticks_init,
                                  batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

    def const_tick_imbalance_bar(self, expected_imbalance_window: int = 100,
                                 exp_num_ticks_init: int = 200):
        bars = ConstImbalanceBars(metric = 'tick_imbalance',
                                  expected_imbalance_window = expected_imbalance_window,
                                  exp_num_ticks_init = exp_num_ticks_init,
                                  batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        imbalance_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                        verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)
        return imbalance_bars, pd.DataFrame(bars.bars_thresholds)

class RunBarFeatures :
    def __init__(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                 num_prev_bars: int = 3,
                 exp_num_ticks_constraints: List[float] = None,
                 batch_size: int = 10000000,
                 analyse_thresholds: bool = False,
                 verbose: bool = True,
                 to_csv: bool = False,
                 output_path: Optional[str] = None):
        self.file_path_or_df = file_path_or_df
        self.num_prev_bars = num_prev_bars
        self.exp_num_ticks_constraints = exp_num_ticks_constraints
        self.batch_size = batch_size
        self.analyse_thresholds = analyse_thresholds
        self.verbose = verbose
        self.to_csv = to_csv
        self.output_path = output_path

    def ema_dollar_run_bar(self, expected_imbalance_window: int = 50,
                           exp_num_ticks_init: int = 100):
        bars = EMARunBars(metric = 'dollar_run', num_prev_bars = self.num_prev_bars,
                          expected_imbalance_window = expected_imbalance_window,
                          exp_num_ticks_init = exp_num_ticks_init,
                          exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                          batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return run_bars, pd.DataFrame(bars.bars_thresholds)

    def ema_volume_run_bar(self, expected_imbalance_window: int = 50,
                           exp_num_ticks_init: int = 100):
        bars = EMARunBars(metric = 'volume_run', num_prev_bars = self.num_prev_bars,
                          expected_imbalance_window = expected_imbalance_window,
                          exp_num_ticks_init = exp_num_ticks_init,
                          exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                          batch_size = self.batch_size,
                          analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return run_bars, pd.DataFrame(bars.bars_thresholds)

    def ema_tick_run_bar(self, expected_imbalance_window: int = 100,
                         exp_num_ticks_init: int = 200):
        bars = EMARunBars(metric = 'tick_run', num_prev_bars = self.num_prev_bars,
                          expected_imbalance_window = expected_imbalance_window,
                          exp_num_ticks_init = exp_num_ticks_init,
                          exp_num_ticks_constraints = self.exp_num_ticks_constraints,
                          batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)

        return run_bars, pd.DataFrame(bars.bars_thresholds)

    def const_dollar_run_bar(self, expected_imbalance_window: int = 50,
                             exp_num_ticks_init: int = 100):
        bars = ConstRunBars(metric = 'dollar_run', num_prev_bars = self.num_prev_bars,
                            expected_imbalance_window = expected_imbalance_window,
                            exp_num_ticks_init = exp_num_ticks_init,
                            batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)
        return run_bars, pd.DataFrame(bars.bars_thresholds)

    def const_volume_run_bar(self, expected_imbalance_window: int = 50,
                             exp_num_ticks_init: int = 100):
        bars = ConstRunBars(metric = 'volume_run', num_prev_bars = self.num_prev_bars,
                            expected_imbalance_window = expected_imbalance_window,
                            exp_num_ticks_init = exp_num_ticks_init,
                            batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)
        return run_bars, pd.DataFrame(bars.bars_thresholds)

    def const_tick_run_bar(self, expected_imbalance_window: int = 10000,
                           exp_num_ticks_init: int = 20000):
        bars = ConstRunBars(metric = 'tick_run', num_prev_bars = self.num_prev_bars,
                            expected_imbalance_window = expected_imbalance_window,
                            exp_num_ticks_init = exp_num_ticks_init,
                            batch_size = self.batch_size, analyse_thresholds = self.analyse_thresholds)
        run_bars = bars.batch_run(file_path_or_df = self.file_path_or_df,
                                  verbose = self.verbose, to_csv = self.to_csv, output_path = self.output_path)
        return run_bars, pd.DataFrame(bars.bars_thresholds)