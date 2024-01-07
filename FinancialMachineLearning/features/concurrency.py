import numpy as np
import pandas as pd
from FinancialMachineLearning.multiprocess.multiprocess import mp_pandas_obj
def concurrent_events(close_series_index, label_endtime, molecule):
    label_endtime = label_endtime.fillna(close_series_index[-1])
    label_endtime = label_endtime[label_endtime >= molecule[0]]
    label_endtime = label_endtime.loc[:label_endtime[molecule].max()]

    nearest_index = close_series_index.searchsorted(np.array([label_endtime.index[0], label_endtime.max()]))
    count = pd.Series(0, index=close_series_index[nearest_index[0]:nearest_index[1] + 1])
    for t_in, t_out in label_endtime.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0]:label_endtime[molecule].max()]
def average_uniqueness(label_endtime, num_conc_events, molecule):
    wght = pd.Series(index=molecule)
    for t_in, t_out in label_endtime.loc[wght.index].items():
        wght.loc[t_in] = (1. / num_conc_events.loc[t_in:t_out]).mean()
    return wght
def average_uniqueness_triple_barrier(triple_barrier_events, close_series, num_threads):
    out = pd.DataFrame()
    num_conc_events = mp_pandas_obj(concurrent_events, ('molecule', triple_barrier_events.index), num_threads,
                                    close_series_index=close_series.index, label_endtime=triple_barrier_events['t1'])
    num_conc_events = num_conc_events.loc[~num_conc_events.index.duplicated(keep='last')]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
    out['tW'] = mp_pandas_obj(average_uniqueness, ('molecule', triple_barrier_events.index), num_threads,
                              label_endtime=triple_barrier_events['t1'], num_conc_events=num_conc_events)
    return out
