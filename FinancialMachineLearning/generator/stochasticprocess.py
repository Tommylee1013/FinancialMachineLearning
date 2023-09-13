import numpy as np
import pandas as pd
import scipy.stats as ss
class GeometricBrownianMotion :
    pass
class OrnsteinUhlenbeckProcess:
    def __init__(self, sigma : float = 0.2,
                 theta : float = -0.1,
                 kappa : float = 0.1,
                 T : int = 1,
                 N : int = 10000) :
        self._theta = theta
        self._T = T
        self._N = N
        if sigma < 0 or kappa < 0 :
            raise ValueError("sigma, theta, kappa must be positive.")
        else :
            self._sigma = sigma
            self._kappa = kappa
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        self._theta = value
    @property
    def kappa(self):
        return self._kappa
    @kappa.setter
    def kappa(self, value):
        self._kappa = value
    def generator(self, X0 : int = 0,
                  paths : int = 1) -> np.array :
        T_vec, dt = np.linspace(0, self._T, self._N, retstep = True)
        X = np.zeros((paths, self._N))
        X[:, 0] = X0
        W = ss.norm.rvs(loc = 0, scale = 1, size = (paths, self._N-1))
        std_dt = np.sqrt(self._sigma**2 / (2 * self._kappa) * (1 - np.exp(-2 * self._kappa * dt)))
        for i in range(0, self._N-1):
            X[:, i + 1] = self._theta + np.exp(-self._kappa * dt) * (X[:, i] - self._theta) + std_dt * W[:, i]
        return X
    def var(self):
        dt = self._T / self._N
        out = self._sigma**2 / (2 * self._kappa) * (1 - np.exp(-2 * self._kappa * dt))
        return out
    def std(self):
        dt = self._T / self._N
        out = np.sqrt(self._sigma**2 / (2 * self._kappa) * (1 - np.exp(-2 * self._kappa * dt)))
        return out
    def mean(self, s0 : int = 1, t : int = 1):
        out = self._theta + (s0 - self._theta) * np.exp(-self.kappa * t)
        return out

class AutoRegressiveProcess :
    pass