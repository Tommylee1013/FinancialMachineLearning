import numpy as np
import pandas as pd
from FinancialMachineLearning.multiprocess.multiprocess import mp_pandas_obj
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
        closing_prices = close[loc: vertical_barrier]
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()
        out.loc[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()
    return out
def add_vertical_barrier(t_events, close, num_days = 0, num_hours = 0, num_minutes = 0, num_seconds = 0):
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
    nearest_index = close.index.searchsorted(t_events + timedelta)
    nearest_index = nearest_index[nearest_index < close.shape[0]]
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]
    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers
def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times = False,
               side_prediction=None):
    target = target.loc[t_events]
    target = target[target > min_ret]
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.loc[target.index]
        pt_sl_ = pt_sl[:2]
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      close=close,
                                      events=events,
                                      pt_sl=pt_sl_)
    for ind in events.index:
        events.loc[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()
    if side_prediction is None:
        events = events.drop('side', axis=1)
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]
    return events
def barrier_touched(out_df, events):
    store = []
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']

        pt_level_reached = ret > target * events.loc[date_time, 'pt']
        sl_level_reached = ret < -target * events.loc[date_time, 'sl']

        if ret > 0.0 and pt_level_reached:
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            store.append(-1)
        else:
            store.append(0)
    out_df['bin'] = store
    return out_df
def meta_labeling(triple_barrier_events, close):
    events_ = triple_barrier_events.dropna(subset = ['t1'])
    all_dates = events_.index.union(other = events_['t1'].values).drop_duplicates()
    prices = close.reindex(all_dates, method = 'bfill')

    out_df = pd.DataFrame(index = events_.index)
    out_df['ret'] = np.log(prices.loc[events_['t1'].values].values / prices.loc[events_.index])
    out_df['trgt'] = events_['trgt']

    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']
    out_df = barrier_touched(out_df, triple_barrier_events)
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0
    out_df['ret'] = np.exp(out_df['ret']) - 1
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']
    return out_df
def drop_labels(events, min_pct = 0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events