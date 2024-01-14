import numpy as np
import pandas as pd
from typing import Union
from sklearn.datasets import make_classification
import datetime

class RegimeGenerator(object):
    def __init__(self, init_ar: tuple, inner_steps: int, phi_positive: tuple, phi_negative: tuple,
                 standard_deviation: float):
        '''
        generate bi-regime market time series
        :param init_ar: tuple, set initial ar1, ar2, ar3 values
        :param inner_steps: number of observations which is negative regime
        :param phi_positive: coefficient of AR models in positive regime
        :param phi_negative: coefficient of AR models in negative regime
        :param standard_deviation: standard deviation of error term
        '''
        self.r1 = init_ar[0]
        self.r2 = init_ar[1]
        self.r3 = init_ar[2]
        self.inner_steps = inner_steps
        self.p1 = phi_positive[0]
        self.p2 = phi_positive[1]
        self.p3 = phi_positive[2]
        self.pn1 = phi_negative[0]
        self.pn2 = phi_negative[1]
        self.pn3 = phi_negative[2]
        self.stdev = standard_deviation
        self.data = None

    def _gen_data(self, phi1, phi2, phi3, flag, drift, steps):
        r1, r2, r3 = self.r1, self.r2, self.r3  # initialization

        rets, flags = [], []
        for _ in range(0, steps):
            a = np.random.normal(loc=0, scale=self.stdev, size=1)  # white noise component using IBM weekly std
            rt = drift + phi1 * r1 + phi2 * r2 + phi3 * r3 + a
            flags.append(flag)
            rets.append(float(rt))

            # Update lagged returns
            r3, r2, r1 = r2, r1, rt

        return rets, flags

    def _gen_dual_regime(self, steps, inner_steps, prob_switch, stdev):
        rets, flags = [], []
        for _ in range(0, steps):

            rand = np.random.uniform()
            is_regime_two = rand < prob_switch

            if is_regime_two:
                # This negative regime has negative sign coefficients to the original
                rets_regime, flags_regime = self._gen_data(phi1=self.pn1, phi2=self.pn2, phi3=self.pn3,
                                                           flag=1, steps=inner_steps, drift=-0.0001)
            else:
                # Original Regime
                rets_regime, flags_regime = self._gen_data(phi1=self.p1, phi2=self.p2, phi3=self.p3,
                                                           flag=0, steps=inner_steps, drift=0.000)

            # Add to store
            rets.extend(rets_regime)
            flags.extend(flags_regime)
        return rets, flags

    def single_regime(self, steps, drift):
        # Generate returns
        rets, _ = self._gen_data(phi1=self.p1, phi2=self.p2, phi3=self.p3, flag=0, steps=steps, drift=drift)

        # Convert to DF and add dates
        self.data = pd.DataFrame({'rets': np.array(rets).flatten()})  # initialization
        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=steps,
            freq='d',
            normalize=True
        )
        self.data.index = dates

        return self.data

    def dual_regime(self, total_steps, prob_switch):
        # Params
        steps = int(total_steps / self.inner_steps)  # Set steps so that total steps is reached

        # Gen dual regime data
        rets, flags = self._gen_dual_regime(
            steps=steps,
            inner_steps=self.inner_steps,
            prob_switch=prob_switch,
            stdev=self.stdev
        )
        # Convert to DF
        date_range = pd.date_range(
            end=datetime.datetime.now(),
            periods=steps * self.inner_steps,
            freq='d',
            normalize=True
        )
        self.data = pd.DataFrame({'rets': np.array(rets).flatten(), 'flags': flags}, index=date_range)

        return self.data

    def prep_data(self, with_flags: bool = True):

        # Set target variable
        self.data['target'] = self.data['rets'].apply(lambda x: 0 if x < 0 else 1).shift(-1)  # Binary classification

        # Create data set
        self.data['target_rets'] = self.data['rets'].shift(-1)  # Add target rets for debugging
        self.data.dropna(inplace=True)

        # Auto-correlation trading rule: trade sign of previous day.
        self.data['pmodel'] = self.data['rets'].apply(lambda x: 1 if x > 0.0 else 0)
        # primary model

        # Strategy daily returns
        self.data['prets'] = (self.data['pmodel'] * self.data['target_rets']).shift(
            1)  # Lag by 1 to remove look ahead and align dates
        self.data.dropna(inplace=True)

        # Add lag rets 2 and 3 for Logistic regression
        self.data['rets2'] = self.data['rets'].shift(1)
        self.data['rets3'] = self.data['rets'].shift(2)

        # Add Regime indicator if with_flags is on
        if with_flags:
            # Add Regime features, lagged by 5 days.
            # We lag it to imitate the lagging nature of rolling statistics.
            self.data['regime'] = self.data['flags'].shift(5)

        # Data used to train model
        model_data = self.data[self.data['pmodel'] == 1].copy()

        # Apply labels to total data set
        # In this setting the target for the pmodel is the meta_labels when you filter by only pmodel=1
        model_data.dropna(inplace=True)

        return model_data, self.data

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

class GeometricBrownianMotion :
    def __init__(self, mu : Union[int, float], sigma : float, n_paths : int,
                 n_steps : int, start , end, initial_price : Union[int, float]) -> None:
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = start
        self.T = end
        self.S_0 = initial_price

    def get_paths(self):
        dt = self.T / self.n_steps
        dW = np.sqrt(dt) * np.random.randn(self.n_paths, self.n_steps)
        dS = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * dW

        dS = np.insert(dS, 0, 0, axis=1)
        S = np.cumsum(dS, axis=1)

        S = self.S_0 * np.exp(S)
        return S

    def mean(self) -> float:
        mean = self.S_0 * np.exp(self.mu * self.t)
        return mean

    def var(self) -> float:
        variance = (self.S_0 ** 2) * np.exp(2 * self.mu * self.t) * (np.exp(self.t * self.sigma ** 2) - 1)
        return variance

    def simulate(self) -> pd.DataFrame:
        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_steps+1)
        df0 = pd.date_range(start=start, end=end, freq='B')
        simulation = pd.DataFrame(self.get_paths().transpose(), index = df0)
        return simulation


class OrnsteinUhlenbeckProcess:
    def __init__(self, alpha, mu, sigma, n_paths : int, n_steps : int,
                 start, end, initial_price : Union[int, float]) -> None:
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = start
        self.T = end
        self.S_0 = initial_price
    def get_paths(self, analytic_EM : bool = False):
        dt = self.T / self.n_steps
        N = np.random.randn(self.n_steps, self.n_paths)
        S = np.concatenate((self.S_0 * np.ones((1, self.n_paths)), np.zeros((self.n_steps, self.n_paths))), axis=0)

        if analytic_EM == True:
            sdev = self.sigma * np.sqrt((1 - np.exp(-2 * self.alpha * dt)) / (2 * self.alpha))
            for i in range(0, self.n_steps):
                S[i + 1, :] = self.mu + (S[i, :] - self.mu) * np.exp(-self.alpha * dt) + sdev * N[i, :]
        else:
            sdev = self.sigma * np.sqrt(dt)
            for i in range(0, self.n_steps):
                S[i + 1, :] = S[i, :] + self.alpha * (self.mu - S[i, :]) * dt + sdev * N[i, :]
        return S
    def mean(self) -> float:
        mean = self.mu + (self.S_0 - self.mu) * np.exp(-self.alpha * self.t)
        return mean
    def var(self) -> float:
        variance = (1 - np.exp(-2 * self.alpha * self.t)) * (self.sigma ** 2) / (2 * self.alpha)
        return variance
    def simulate(self, analytic_EM = False) -> pd.DataFrame:
        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_steps+1)
        df0 = pd.date_range(start=start, end=end, freq='B')
        simulation = pd.DataFrame(self.get_paths(analytic_EM), index = df0)
        return simulation

class AutoRegressiveProcess:
    def __init__(self, p : int, n_paths : int ,
                 n_steps : int, start : int,
                 end : int, initial_price : Union[int, float],
                 coefficients = None):
        self.p = p
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = start
        self.T = end
        self.S_0 = initial_price

        if coefficients is None:
            self.coefficients = np.random.randn(p) / 100
        else:
            if len(coefficients) != p:
                raise ValueError(f"coefficients must have elements {p}")
            self.coefficients = np.array(coefficients) / 100

    def mean(self) -> float:
        mean = self.coefficients.mean()
        return mean

    def var(self) -> float:
        return np.var(self.coefficients) / (1 - np.sum(self.coefficients) ** 2)

    def simulate(self) -> pd.DataFrame:
        data = np.zeros((self.n_steps + 1, self.n_paths))
        for j in range(self.n_paths):
            for i in range(self.p, self.n_steps + 1):
                ar_term = np.sum(self.coefficients * data[i - self.p:i, j])
                new_value = ar_term + np.random.randn()
                data[i, j] = new_value

        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_steps+1)
        df0 = pd.date_range(start = start, end = end, freq='B')

        simulation = pd.DataFrame(data, columns = [f'Path_{i}' for i in range(self.n_paths)], index = df0)

        return simulation

class JumpDiffusionProcess :
    def __init__(self, mu : Union[int, float], sigma : float, lambdaN : float, eta1 : float, eta2 : float, p : float,
                 n_paths : int, n_steps : int, start, end, initial_price : Union[int, float]) -> None:
        self.mu = mu
        self.sigma = sigma
        self.lambdaN = lambdaN
        self.eta1 = eta1
        self.eta2 = eta2
        self.p = p
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = start
        self.T = end
        self.S_0 = initial_price

    def get_paths(self):
        dt = self.T / self.n_steps
        dX = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * np.random.randn(self.n_steps,
                                                                                                 self.n_paths)
        dP = np.random.poisson(self.lambdaN * dt, (self.n_steps, self.n_paths))
        U = np.random.uniform(0, 1, (self.n_steps, self.n_paths))
        Z = np.zeros((self.n_steps, self.n_paths))
        for i in range(0, len(U[0])):
            for j in range(0, len(U)):
                if U[j, i] >= self.p:
                    Z[j, i] = (-1 / self.eta1) * np.log((1 - U[j, i]) / self.p)
                elif U[j, i] < self.p:
                    Z[j, i] = (1 / self.eta2) * np.log(U[j, i] / (1 - self.p))

        dJ = (np.exp(Z) - 1) * dP
        dS = dX + dJ

        dS = np.insert(dS, 0, self.S_0, axis=0)
        S = np.cumsum(dS, axis=0)
        return S

    def mean(self) -> float:
        mean = (self.mu + self.lambdaN * (self.p / self.eta1 - (1 - self.p) / self.eta2)) * self.t + self.S_0
        return mean

    def var(self) -> float:
        variance = (self.sigma ** 2 + 2 * self.lambdaN * (self.p / (self.eta1 ** 2) + (1 - self.p) / (self.eta2 ** 2))) * self.t
        return variance

    def simulate(self) -> pd.DataFrame:
        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_steps+1)
        df0 = pd.date_range(start=start, end=end, freq='B')
        simulation = pd.DataFrame(self.get_paths(), index = df0)
        return simulation

class PradoSyntheticProcess :
    def __init__(self, n_features : int, n_informative : int, n_redundant : int, n_samples : int = 1000) -> None:
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_samples = n_samples
    def simulate(self):
        trnsX, _ = make_classification(n_samples = self.n_samples,
                                       n_features = self.n_features,
                                       n_informative = self.n_informative,
                                       n_redundant = self.n_redundant,
                                       shuffle = False)
        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_samples)
        df0 = pd.date_range(start=start, end=end, freq='B')
        trnsX = pd.DataFrame(trnsX, index=df0)
        trnsX = trnsX / 100
        return trnsX
    def mean(self) :
        return self.simulate().mean().mean()
    def var(self) :
        return self.simulate().var().mean()
    def skew(self) :
        return self.simulate().skew().mean()
    def kurt(self) :
        return  self.simulate().kurt().mean()