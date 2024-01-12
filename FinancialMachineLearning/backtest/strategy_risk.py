import numpy as np
import pandas as pd
import scipy.stats as ss

def binHR(stop_loss, profit_taking, freq, t_sharpe_ratio):
    a = (freq + t_sharpe_ratio ** 2) * (profit_taking - stop_loss) ** 2
    b = (2 * freq * stop_loss - t_sharpe_ratio ** 2 * (profit_taking - stop_loss)) * (profit_taking - stop_loss)
    c = freq * stop_loss ** 2
    p = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2.0 * a)
    return p

def get_grid_precision(
        stop_loss: np.array,
        profit_taking: float,
        observation: np.array,
        expected_sharpe: float
) -> pd.DataFrame():
    values = np.ones(len(stop_loss) * len(observation)).reshape(len(stop_loss), len(observation))
    for i in range(len(stop_loss)):
        for j in range(len(observation)):
            values[i][j] = binHR(stop_loss[i], profit_taking, observation[j], expected_sharpe)
    values = pd.DataFrame(
        values, columns=observation, index=stop_loss
    )
    return values
def mix_gaussians(mu1, mu2, sigma1, sigma2, prob1, nObs) :
    '''
    gaussian simulation
    :param mu1: return rate of regime 1
    :param mu2: return rate of regime 2
    :param sigma1: market risk of regime 1
    :param sigma2: market risk of regime 2
    :param prob1:
    :param nObs: number of observation
    :return:
    '''
    ret1 = np.random.normal(mu1, sigma1, size = int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size = int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis = 0)
    np.random.shuffle(ret)
    return ret

def prob_failure(ret, freq, t_sharpe_ratio) :
    '''
    calculate failure proability of strategy given parameters
    :param ret: time series of such equities
    :param freq: betting frequency for unit periods
    :param t_sharpe_ratio: target sharpe ratio
    :return: strategy risk
    '''
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    threshold_p = binHR(rNeg, rPos, freq, t_sharpe_ratio)
    risk = ss.norm.cdf(threshold_p, p, p * (1 - p))
    return risk