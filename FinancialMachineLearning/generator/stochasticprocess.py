import numpy as np
import pandas as pd
from typing import Union
from sklearn.datasets import make_classification
import datetime
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
        simulation = pd.DataFrame(self.get_paths().transpose())
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
    def simulate(self, analytic_EM=False) -> pd.DataFrame:
        simulation = pd.DataFrame(self.get_paths(analytic_EM))
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
            self.coefficients = np.random.randn(p)
        else:
            if len(coefficients) != p:
                raise ValueError(f"coefficients must have elements {p}")
            self.coefficients = np.array(coefficients)

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

        simulation = pd.DataFrame(data, columns=[f'Path_{i}' for i in range(self.n_paths)])
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
        simulation = pd.DataFrame(self.get_paths())
        return simulation

class PradoSyntheticProcess :
    def __init__(self, n_features : int, n_informative : int, n_redundant : int, n_samples : int = 1000) -> None:
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_samples = n_samples
    def simulator(self):
        trnsX, _ = make_classification(n_samples = self.n_samples,
                                       n_features = self.n_features,
                                       n_informative = self.n_informative,
                                       n_redundant = self.n_redundant,
                                       shuffle = False)
        start = datetime.datetime.today()
        end = start + pd.offsets.BDay(self.n_samples - 1)
        df0 = pd.date_range(start=start, end=end, freq='B')
        trnsX = pd.DataFrame(trnsX, index=df0)
        trnsX = trnsX / 100
        return trnsX
    def mean(self) :
        return self.simulator().mean().mean()
    def var(self) :
        return self.simulator().var().mean()