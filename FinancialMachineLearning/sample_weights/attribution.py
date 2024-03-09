import numpy as np
import pandas as pd

from FinancialMachineLearning.features.concurrency import concurrent_events, average_uniqueness_triple_barrier
from FinancialMachineLearning.utils.multiprocess import mp_pandas_obj

def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    ret = np.log(close_series).diff()
    weights = pd.Series(index=molecule)

    for t_in, t_out in label_endtime.loc[weights.index].iteritems():
        weights.loc[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()
    return weights.abs()

def weights_by_return(triple_barrier_events, close_series, num_threads=5):
    has_null_events = bool(triple_barrier_events.isnull().values.any())
    has_null_index = bool(triple_barrier_events.index.isnull().any())
    assert has_null_events is False and has_null_index is False, 'NaN values in triple_barrier_events, delete nans'

    num_conc_events = mp_pandas_obj(concurrent_events, ('molecule', triple_barrier_events.index), num_threads,
                                    close_series_index=close_series.index, label_endtime=triple_barrier_events['t1'])
    num_conc_events = num_conc_events.loc[~num_conc_events.index.duplicated(keep='last')]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
    weights = mp_pandas_obj(_apply_weight_by_return, ('molecule', triple_barrier_events.index), num_threads,
                            label_endtime=triple_barrier_events['t1'], num_conc_events=num_conc_events,
                            close_series=close_series)
    weights *= weights.shape[0] / weights.sum()
    return weights

def weights_by_time_decay(triple_barrier_events, close_series, num_threads=5, decay=1):
    assert bool(triple_barrier_events.isnull().values.any()) is False and bool(
        triple_barrier_events.index.isnull().any()) is False, 'NaN values in triple_barrier_events, delete nans'
    av_uniqueness = average_uniqueness_triple_barrier(triple_barrier_events, close_series, num_threads)
    decay_w = av_uniqueness['tW'].sort_index().cumsum()
    if decay >= 0:
        slope = (1 - decay) / decay_w.iloc[-1]
    else:
        slope = 1 / ((decay + 1) * decay_w.iloc[-1])
    const = 1 - slope * decay_w.iloc[-1]
    decay_w = const + slope * decay_w
    decay_w[decay_w < 0] = 0
    return decay_w
