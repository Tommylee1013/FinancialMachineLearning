import numpy as np
import pandas as pd
from scipy.stats import norm, moment

from FinancialMachineLearning.bet_sizing.ef3m import M2N, raw_moment, most_likely_parameters
from FinancialMachineLearning.multiprocess.multiprocess import mp_pandas_obj
import warnings

def get_signal(prob, num_classes, pred = None):
    if prob.shape[0] == 0:
        return pd.Series()
    bet_sizes = (prob - 1/num_classes) / (prob * (1 - prob))**0.5
    if not isinstance(pred, type(None)): bet_sizes = pred * (2 * norm.cdf(bet_sizes) - 1)
    else: bet_sizes = bet_sizes.apply(lambda s: 2 * norm.cdf(s) - 1)
    return bet_sizes

def avg_active_signals(signals, num_threads=1):
    t_pnts = set(signals['t1'].dropna().to_numpy())
    t_pnts = t_pnts.union(signals.index.to_numpy())
    t_pnts = list(t_pnts)
    t_pnts.sort()
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', t_pnts), num_threads, signals=signals)
    return out

def mp_avg_active_signals(signals, molecule):
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.to_numpy() <= loc)&((loc < signals['t1'])|pd.isnull(signals['t1']))
        act = signals[df0].index
        if act.size > 0: out[loc] = signals.loc[act, 'signal'].mean()
        else: out[loc] = 0
    return out

def discrete_signal(signal0, step_size):
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor
    return signal1

def bet_size_sigmoid(w_param, price_div):
    return price_div * ((w_param + price_div**2)**(-0.5))


def get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos):
    return int(bet_size_sigmoid(w_param, forecast_price-market_price) * max_pos)


def inv_price_sigmoid(forecast_price, w_param, m_bet_size):
    return forecast_price - m_bet_size * (w_param / (1 - m_bet_size**2))**0.5


def limit_price_sigmoid(target_pos, pos, forecast_price, w_param, max_pos):
    if target_pos == pos:
        return np.nan
    sgn = np.sign(target_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(target_pos+1)):
        l_p += inv_price_sigmoid(forecast_price, w_param, j/float(max_pos))
    l_p = l_p / abs(target_pos-pos)
    return l_p

def get_w_sigmoid(price_div, m_bet_size):
    return (price_div**2) * ((m_bet_size**(-2)) - 1)

def bet_size_power(w_param, price_div):
    if not (-1 <= price_div <= 1):
        raise ValueError(f"Price divergence must be between -1 and 1, inclusive. Found price divergence value:"
                         f" {price_div}")
    if price_div == 0.0:
        return 0.0

    return np.sign(price_div) * abs(price_div)**w_param


def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    return int(bet_size_power(w_param, forecast_price-market_price) * max_pos)


def inv_price_power(forecast_price, w_param, m_bet_size):
    if m_bet_size == 0.0:
        return forecast_price
    return forecast_price - np.sign(m_bet_size) * abs(m_bet_size)**(1/w_param)

def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    sgn = np.sign(target_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(target_pos+1)):
        l_p += inv_price_power(forecast_price, w_param, j/float(max_pos))

    l_p = l_p / abs(target_pos-pos)
    return l_p

def get_w_power(price_div, m_bet_size):
    if not -1 <= price_div <= 1:
        raise ValueError("Price divergence argument 'x' must be between -1 and 1,"
                         " inclusive when using function 'power'.")

    w_calc = np.log(m_bet_size/np.sign(price_div)) / np.log(abs(price_div))
    if w_calc < 0:
        warnings.warn("'w' parameter evaluates to less than zero. Zero is returned.", UserWarning)

    return max(0, w_calc)
def bet_size(w_param, price_div, func):
    return {'sigmoid': bet_size_sigmoid,
            'power': bet_size_power}[func](w_param, price_div)

def get_target_pos(w_param, forecast_price, market_price, max_pos, func):
    return {'sigmoid': get_target_pos_sigmoid,
            'power': get_target_pos_power}[func](w_param, forecast_price, market_price, max_pos)

def inv_price(forecast_price, w_param, m_bet_size, func):
    return {'sigmoid': inv_price_sigmoid,
            'power': inv_price_power}[func](forecast_price, w_param, m_bet_size)

def limit_price(target_pos, pos, forecast_price, w_param, max_pos, func):
    return {'sigmoid': limit_price_sigmoid,
            'power': limit_price_power}[func](int(target_pos), int(pos), forecast_price, w_param, max_pos)

def get_w(price_div, m_bet_size, func):
    return {'sigmoid': get_w_sigmoid,
            'power': get_w_power}[func](price_div, m_bet_size)

def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    signal_0 = get_signal(prob, num_classes, pred)
    events_0 = signal_0.to_frame('signal').join(events['t1'], how='left')
    if average_active:
        signal_1 = avg_active_signals(events_0, num_threads)
    else:
        signal_1 = events_0.signal

    if abs(step_size) > 0:
        signal_1 = discrete_signal(signal0=signal_1, step_size=abs(step_size))

    return signal_1

def bet_size_dynamic(current_pos,
                     max_pos, market_price,
                     forecast_price,
                     cal_divergence=10,
                     cal_bet_size=0.95,
                     func='sigmoid'):
    d_vars = {'pos': current_pos, 'max_pos': max_pos, 'm_p': market_price, 'f': forecast_price}
    events_0 = confirm_and_cast_to_df(d_vars)
    w_param = get_w(cal_divergence, cal_bet_size, func)
    events_0['t_pos'] = events_0.apply(lambda x: get_target_pos(w_param, x.f, x.m_p, x.max_pos, func), axis=1)
    events_0['l_p'] = events_0.apply(lambda x: limit_price(x.t_pos, x.pos, x.f, w_param, x.max_pos, func), axis=1)
    events_0['bet_size'] = events_0.apply(lambda x: bet_size(w_param, x.f-x.m_p, func), axis=1)

    return events_0[['bet_size', 't_pos', 'l_p']]

def bet_size_budget(events_t1, sides):
    events_1 = get_concurrent_sides(events_t1, sides)
    active_long_max, active_short_max = events_1['active_long'].max(), events_1['active_short'].max()
    frac_active_long = events_1['active_long'] / active_long_max if active_long_max > 0 else 0
    frac_active_short = events_1['active_short'] / active_short_max if active_short_max > 0 else 0
    events_1['bet_size'] = frac_active_long - frac_active_short
    return events_1

def bet_size_reserve(events_t1, sides,
                     fit_runs=100,
                     epsilon=1e-5,
                     factor=5,
                     variant=2,
                     max_iter=10_000,
                     num_workers=1,
                     return_parameters=False):
    events_active = get_concurrent_sides(events_t1, sides)
    events_active['c_t'] = events_active['active_long'] - events_active['active_short']
    central_mmnts = [moment(events_active['c_t'].to_numpy(), moment=i) for i in range(1, 6)]
    raw_mmnts = raw_moment(central_moments=central_mmnts, dist_mean=events_active['c_t'].mean())
    m2n = M2N(raw_mmnts, epsilon=epsilon, factor=factor, n_runs=fit_runs,
              variant=variant, max_iter=max_iter, num_workers=num_workers)
    df_fit_results = m2n.mp_fit()
    fit_params = most_likely_parameters(df_fit_results)
    params_list = [fit_params[key] for key in ['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1']]
    events_active['bet_size'] = events_active['c_t'].apply(lambda c: single_bet_size_mixed(c, params_list))

    if return_parameters:
        return events_active, fit_params
    return events_active

def confirm_and_cast_to_df(d_vars):
    any_series = False
    all_series = True
    ser_len = 0
    for var in d_vars.values():
        any_series = any_series or isinstance(var, pd.Series)
        all_series = all_series and isinstance(var, pd.Series)

        if isinstance(var, pd.Series):
            ser_len = var.size
            idx = var.index
    if not any_series:
        for k in d_vars:
            d_vars[k] = pd.Series(data=[d_vars[k]], index=[0])

    if any_series and not all_series:
        for k in d_vars:
            if not isinstance(d_vars[k], pd.Series):
                d_vars[k] = pd.Series(data=np.array([d_vars[k] for i in range(ser_len)]), index=idx)

    events = pd.concat(list(d_vars.values()), axis=1)
    events.columns = list(d_vars.keys())

    return events

def get_concurrent_sides(events_t1, sides):
    events_0 = pd.DataFrame({'t1':events_t1, 'side':sides})
    events_0['active_long'] = 0
    events_0['active_short'] = 0

    for idx in events_0.index:
        df_long_active_idx = set(events_0[(events_0.index <= idx) & (events_0['t1'] > idx) & (events_0['side'] > 0)].index)
        events_0.loc[idx, 'active_long'] = len(df_long_active_idx)
        df_short_active_idx = set(events_0[(events_0.index <= idx) & (events_0['t1'] > idx) & (events_0['side'] < 0)].index)
        events_0.loc[idx, 'active_short'] = len(df_short_active_idx)

    return events_0

def cdf_mixture(x_val, parameters):
    mu_1, mu_2, sigma_1, sigma_2, p_1 = parameters  # Parameters reassigned for clarity.
    return p_1 * norm.cdf(x_val, mu_1, sigma_1) + (1-p_1) * norm.cdf(x_val, mu_2, sigma_2)

def single_bet_size_mixed(c_t, parameters):
    if c_t >= 0:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / (1 - cdf_mixture(0, parameters))
    else:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / cdf_mixture(0, parameters)
    return single_bet_size
