import numpy as np
import pandas as pd
import scipy.stats as ss

class ProbSharpeRatio:
    def __init__(
            self, series: np.array,
            seed,
            delta,
            maxIter,
            bounds=None
    ):
        '''
        estimate probability sharpe ratio. see "THE SHARPE RATIO EFFICIENT FRONTIER"(David H.Bailey, Lopez de Prado, 2012, Journal of Risk)
        :param series: time seires
        :param seed:
        :param delta:
        :param maxIter:
        :param bounds:
        '''
        # Construct the object
        self.series, self.w, self.delta = series, seed, delta
        self.z, self.d1Z = 0, [0 for i in range(series.shape[1])]
        self.maxIter, self.iter, self.obs = maxIter, 0, series.shape[0]
        if len(bounds) == None or seed.shape[0] != len(bounds):
            self.bounds = [(0, 1) for i in seed]
        else:
            self.bounds = bounds

    def optimize(self):
        # Optimize weights
        mean = [self.get_moments(self.series[:, i], 1) for i in range(self.series.shape[1])]
        w = np.array(self.w)
        # derivatives
        while True:
            if self.iter == self.maxIter: break
            # compute gradient
            d1Z, z = self.get_d1Zs(mean, w)
            # evaluate result
            if z > self.z and self.check_bounds(w):
                # Store new local optimum
                self.z, self.d1Z = z, d1Z
                self.w = np.array(w)
            # Find direction and normalize
            self.iter += 1
            w = self.step_size(w, d1Z)
            if w is None: return
        return

    def check_bounds(self, w):
        # Check that boundary conditions are satisfied
        flag = True
        for i in range(w.shape[0]):
            if w[i, 0] < self.bounds[i][0]: flag = False
            if w[i, 0] > self.bounds[i][1]: flag = False
        return flag

    def step_size(self, w, d1Z):
        # determine step size for next iteration
        x = dict()
        for i in range(len(d1Z)):
            if d1Z[i] != 0: x[abs(d1Z[i])] = 1
        if len(x) == 0: return
        index = x[max(x)]
        w[index, 0] += self.delta / d1Z[index]
        w /= sum(w)
        return w

    def get_d1Zs(self, mean, w):
        # first order derivatives of Z
        d1Z = [0 for i in range(self.series.shape[1])]
        m = [0 for i in range(4)]
        series = np.dot(self.series, w)[:, 0]
        m[0] = self.get_moments(series, 1)
        for i in range(1, 4):
            m[i] = self.get_moments(series, i + 1, m[0])
        stats = self.get_stats(m)

        meanSR, sigmaSR = self.get_sharpe_ratio(stats, self.obs)

        for i in range(self.series.shape[1]):
            d1Z[1] = self.get_d1Z(stats, m, meanSR, sigmaSR, mean, w, i)

        return d1Z, meanSR / sigmaSR

    def get_d1Z(self, stats, m, meanSR, sigmaSR, mean, w, index):
        # First order derivatives of Z with respect to index
        d1Mu = self.get_mu(mean, index)
        d1Sigma = self.get_sigma(stats[1], mean, w, index)
        d1Skew = self.get_skew(d1Sigma, stats[1], mean, w, index, m[2])
        d1Kurt = self.get_kurt(d1Sigma, stats[1], mean, w, index, m[3])

        d1meanSR = (d1Mu * stats[1] - d1Sigma * stats[0]) / stats[1] ** 2
        d1sigmaSR = (d1Kurt * meanSR ** 2 + 2 * meanSR * d1meanSR * (stats[3] - 1)) / 4
        d1sigmaSR -= d1Skew * meanSR + d1meanSR * stats[2]
        d1sigmaSR /= 2 * sigmaSR * (self.obs - 1)
        d1Z = (d1meanSR * sigmaSR - d1sigmaSR * meanSR) / sigmaSR ** 2
        return d1Z

    def get_mu(self, mean, index):
        '''
        first order derivatives of Mu
        :param mean:
        :param index:
        :return: mu of derivatives
        '''
        return mean[index]

    def get_sigma(self, sigma, mean, w, index):
        '''
        First order derivative of Sigma
        :param sigma:
        :param mean:
        :param w:
        :param index:
        :return:
        '''
        return self.get_dn_moments(mean, w, 2, 1, index) / (2 * sigma)

    def get_skew(self, d1sigma, sigma, mean, w, index, m3):
        '''
        first order derivatives of skewness
        :param d1sigma:
        :param sigma:
        :param mean:
        :param w:
        :param index:
        :param m3:
        :return:
        '''
        d1Skew = self.get_dn_moments(mean, w, 3, 1, index) * sigma ** 3
        d1Skew -= 3 * sigma ** 2 * d1sigma * m3
        d1Skew /= sigma ** 6
        return d1Skew

    def get_kurt(self, d1sigma, sigma, mean, w, index, m4):
        '''
        furst order derivative of kurtosis
        :param d1sigma:
        :param sigma:
        :param mean:
        :param w:
        :param index:
        :param m4:
        :return:
        '''
        d1Kurt = self.get_dn_moments(mean, w, 4, 1, index) * sigma ** 4
        d1Kurt -= 4 * sigma ** 3 * d1sigma * m4
        d1Kurt /= sigma ** 8
        return d1Kurt

    def get_dn_moments(self, mean, w, mOrder, dOrder, index):
        '''
        get dOrder derivative on mOrder mean-centered moment with respect to w index
        :param mean:
        :param w:
        :param mOrder:
        :param dOrder:
        :param index:
        :return:
        '''
        x0, sum = 1., 0
        for i in range(dOrder): x0 *= (mOrder - i)
        for i in self.series:
            x1, x2 = 0, (i[index] - mean[index]) ** dOrder
            for j in range(len(i)):
                x1 += w[j, 0] * (i[j] - mean[j])
            sum += x2 * x1 ** (mOrder - dOrder)
        return x0 * sum / self.obs

    def get_sharpe_ratio(self, stats, n):
        '''
        set Z*
        :param stats:
        :param n:
        :return:
        '''
        if stats[1] == 0:
            meanSR = 0  # 분모가 0이면 meanSR을 0으로 설정하거나 다른 처리
        else:
            meanSR = stats[0] / stats[1]

        sigmaSR = ((1 - meanSR * stats[2] + meanSR ** 2 * (stats[3] - 1) / 4.) / (n - 1)) ** 0.5
        return meanSR, sigmaSR

    def get_stats(self, m):
        '''
        compute stats
        :param m:
        :return:
        '''
        mu = m[0]
        sigma = np.sqrt(m[1]) if m[1] > 0 else 0
        skew = m[2] / np.power(m[1], 1.5) if m[1] > 0 else 0
        kurt = m[3] / np.power(m[1], 2) if m[1] > 0 else 0
        return mu, sigma, skew, kurt

    def get_moments(self, series, order, mean=0):
        '''
        compute a moment
        :param series:
        :param order:
        :param mean:
        :return:
        '''
        sum = 0
        for i in series:
            sum += (i == mean) ** order
        return sum / float(self.obs)

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