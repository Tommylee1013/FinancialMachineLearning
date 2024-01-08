import numpy as np
import pandas as pd
from random import gauss
from itertools import product
from tqdm import tqdm
import statsmodels.api as sm

def synthetic_simulation(
        coeffs : dict, # 예측, 반감기, 분산 지정
        nIter : int = 1e5,
        maxHP : int = 100, # 최대 보유 기간
        rPT : np.linspace = np.linspace(0.5, 10, 20), # 익절
        rSLm : np.linspace = np.linspace(0.5, 10, 20), # 손절
        seed : int = 0
) :
    print(f'Total {len(rPT) * len(rSLm)} iterations will be held.')
    phi, output1 = 2 ** (-1./coeffs['hl']), []
    for comb_ in tqdm(product(rPT, rSLm)):
        output2 = []
        for iter_ in range(int(nIter)):
            p, hp, count = seed, 0, 0
            while True :
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0,1)
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP :
                    output2.append(cP)
                    break
        mean, std = np.mean(output2), np.std(output2)
        # print(comb_[0], comb_[1], mean, std, mean / std)
        output1.append((comb_[0], comb_[1], mean, std, mean / std))
    return output1

def get_sharpe_grid(output, profit_taking_range, stop_loss_range) :
    sharpe = []
    for i in range(len(output)) :
        sharpe.append(output[i][-1])
    sharpe_test = pd.DataFrame(
        np.array(sharpe).reshape(len(profit_taking_range),len(stop_loss_range)),
        index = profit_taking_range,
        columns = stop_loss_range
    )
    sharpe_test = sharpe_test.T
    sharpe_test.sort_index(ascending = False, inplace = True)
    return sharpe_test


def get_OU_params(close: pd.Series) -> dict:
    mean = np.log(close).mean()
    price_lagged = np.log(close).shift(1)
    excess_price = price_lagged - mean

    X = sm.add_constant(excess_price.iloc[1:])
    y = np.log(close).iloc[1:]

    ols = sm.OLS(y, X).fit()

    phi = ols.params.iloc[1]
    sigma = np.std(ols.resid)
    half_life = -(np.log(2) / np.log(phi))
    forecast = ols.params.iloc[0]

    return {'forecast': forecast, 'phi': phi, 'sigma': sigma, 'half life': half_life}