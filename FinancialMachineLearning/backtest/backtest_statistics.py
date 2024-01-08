import numpy as np
import pandas as pd
import scipy.stats as ss

def timing_of_flattening_and_flips(target_positions : pd.Series) -> pd.DatetimeIndex:
    '''
    get betting timing
    :param target_positions: pd.Series
    :return: pd.DatetimeIndex
    '''
    empty_positions = target_positions[(target_positions == 0)].index
    previous_positions = target_positions.shift(1)
    previous_positions = previous_positions[(previous_positions != 0)].index
    flattening = empty_positions.intersection(previous_positions)
    multiplied_posions = target_positions.iloc[1:] * target_positions.iloc[:-1].values
    flips = multiplied_posions[(multiplied_posions < 0)].index
    flips_and_flattenings = flattening.union(flips).sort_values()

    if target_positions.index[-1] not in flips_and_flattenings:
        flips_and_flattenings = flips_and_flattenings.append(target_positions.index[-1:])

    return flips_and_flattenings

def average_holding_period(target_positions : pd.Series) -> float :
    holding_period = pd.DataFrame(columns=['holding_time', 'weight'])
    entry_time = 0
    position_difference = target_positions.diff()
    time_difference = (target_positions.index - target_positions.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, target_positions.shape[0]):
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) >= 0:
            if float(target_positions.iloc[i]) != 0:
                entry_time = (entry_time * target_positions.iloc[i - 1] +
                              time_difference[i] * position_difference.iloc[i]) / \
                             target_positions.iloc[i]
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) < 0:
            hold_time = time_difference[i] - entry_time
            if float(target_positions.iloc[i] * target_positions.iloc[i - 1]) < 0:
                weight = abs(target_positions.iloc[i - 1])
                holding_period.loc[target_positions.index[i],
                                   ['holding_time', 'weight']] = (hold_time, weight)
                entry_time = time_difference[i]
            else:
                weight = abs(position_difference.iloc[i])
                holding_period.loc[target_positions.index[i],
                                   ['holding_time', 'weight']] = (hold_time, weight)
    if float(holding_period['weight'].sum()) > 0:
        avg_holding_period = float((holding_period['holding_time'] * holding_period['weight']).sum() / holding_period['weight'].sum())
    else:
        avg_holding_period = float('nan')
    return avg_holding_period

def bets_concentration(returns: pd.Series) -> float:
    '''
    get concentrations derived from HHI(Herfindahl - Hirschman Index)
    :param returns: pd.Series
    :return: float
    '''
    if returns.shape[0] <= 2:
        return float('nan')
    weights = returns / returns.sum()
    hhi = (weights ** 2).sum()
    hhi = float((hhi - returns.shape[0] ** (-1)) / (1 - returns.shape[0] ** (-1)))
    return hhi

def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    positive_concentration = bets_concentration(returns[returns >= 0])
    negative_concentration = bets_concentration(returns[returns < 0])
    time_concentration = \
        bets_concentration(returns.groupby(pd.Grouper(freq=frequency)).count())
    return (positive_concentration, negative_concentration, time_concentration)


def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    frame = returns.to_frame('pnl')
    frame['hwm'] = returns.expanding().max()
    high_watermarks = frame.groupby('hwm').min().reset_index()
    high_watermarks.columns = ['hwm', 'min']
    high_watermarks.index = frame['hwm'].drop_duplicates(keep='first').index
    high_watermarks = high_watermarks[high_watermarks['hwm'] > high_watermarks['min']]
    if dollars:
        drawdown = high_watermarks['hwm'] - high_watermarks['min']
    else:
        drawdown = 1 - high_watermarks['min'] / high_watermarks['hwm']
    time_under_water = ((high_watermarks.index[1:] - high_watermarks.index[:-1]).days / 365.25)
    time_under_water = pd.Series(time_under_water, index=high_watermarks.index[:-1])
    return drawdown, time_under_water


def sharpe_ratio(returns: pd.Series, cumulative: bool = False,
                 entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    if cumulative:
        returns = returns / returns.shift(1) - 1
        returns = returns[1:]
    sharpe_r = (returns.mean() - risk_free_rate) / returns.std() * \
               (entries_per_year) ** (1 / 2)

    return sharpe_r


def probabalistic_sharpe_ratio(observed_sr: float, benchmark_sr: float,
                               number_of_returns: int, skewness_of_returns: float = 0,
                               kurtosis_of_returns: float = 3) -> float:
    probab_sr = ss.norm.cdf(((observed_sr - benchmark_sr) * (number_of_returns - 1) ** (1 / 2)) / \
                            (1 - skewness_of_returns * observed_sr +
                             (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) ** (1 / 2))

    return probab_sr


def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list,
                          number_of_returns: int, skewness_of_returns: float = 0,
                          kurtosis_of_returns: float = 3) -> float:
    benchmark_sr = np.array(sr_estimates).std() * \
                   ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / len(sr_estimates)) +
                    np.euler_gamma * ss.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

    deflated_sr = probabalistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns,
                                             skewness_of_returns, kurtosis_of_returns)

    return deflated_sr


def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    track_rec_length = 1 + (1 - skewness_of_returns * observed_sr +
                            (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) * \
                       (ss.norm.ppf(1 - alpha) / (observed_sr - benchmark_sr)) ** (2)

    return track_rec_length