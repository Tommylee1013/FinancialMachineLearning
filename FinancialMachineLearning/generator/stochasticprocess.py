import numpy as np
import pandas as pd
from typing import Union
from sklearn.datasets import make_classification
import datetime
import scipy.stats as ss

class MonteCarloSimulation:
    def __init__(self, interest_rate: float, initial_price: float, maturity: float, sigma: float,
                 dividend_yield: float, nObs: int, slices: int, random_state: Union[bool, int] = False):
        np.random.seed(random_state if isinstance(random_state, int) else None)
        self.S0 = initial_price
        self.T = maturity
        self.dividend_yield = dividend_yield
        self.nObs = nObs
        self.slices = slices
        self.dt = self.T / self.slices
        self.mu = interest_rate
        self.sigma = sigma
        self.paths = np.zeros((nObs, slices + 1))
        self.paths[:, 0] = self.S0

    def _generate_standard_normal_random_variables(self, correlation_matrix=None):
        if correlation_matrix is None:
            z = np.random.standard_normal((self.nObs, self.slices))
        else:
            z = np.random.multivariate_normal(np.zeros(2), correlation_matrix, (self.nObs, self.slices))
        return z

    def geometric_brownian_motion(self):
        z = self._generate_standard_normal_random_variables()
        drift = (self.mu - self.dividend_yield - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        self.paths[:, 1:] = self.S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
        return self.paths

    def vasicek_model(self, kappa: float, theta: float, sigma_r: float):
        r = np.zeros((self.nObs, self.slices + 1))
        r[:, 0] = self.mu
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            dr = kappa * (theta - r[:, t-1]) * self.dt + sigma_r * z[:, t-1] * np.sqrt(self.dt)
            r[:, t] = r[:, t-1] + dr
        return r

    def cox_ingersoll_ross_model(self, kappa: float, theta: float, sigma_r: float):
        if not 2 * kappa * theta > sigma_r ** 2:
            raise ValueError("2 * kappa * theta must be greater than sigma_r ** 2 for CIR model.")
        r = np.zeros((self.nObs, self.slices + 1))
        r[:, 0] = self.mu
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            dr = kappa * (theta - r[:, t-1]) * self.dt + sigma_r * np.sqrt(r[:, t-1]) * z[:, t-1] * np.sqrt(self.dt)
            r[:, t] = np.maximum(r[:, t-1] + dr, 0)  # Ensure non-negative rates
        return r

    def heston_model(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0):
        correlation_matrix = np.array([[1, rho], [rho, 1]])
        z = self._generate_standard_normal_random_variables(correlation_matrix)
        s = np.zeros((self.nObs, self.slices + 1))
        v = np.zeros((self.nObs, self.slices + 1))
        s[:, 0] = self.S0
        v[:, 0] = self.sigma ** 2
        for t in range(1, self.slices + 1):
            dv = kappa * (theta - v[:, t-1]) * self.dt + sigma_v * np.sqrt(v[:, t-1]) * z[:, t-1, 1] * np.sqrt(self.dt)
            ds = s[:, t-1] * (self.mu * self.dt + np.sqrt(np.maximum(v[:, t-1], 0)) * z[:, t-1, 0] * np.sqrt(self.dt))
            v[:, t] = np.maximum(v[:, t-1] + dv, 0)  # Ensure non-negative variance
            s[:, t] = s[:, t-1] + ds
        return s

    def ornstein_uhlenbeck(self, kappa: float, theta: float, sigma: float):
        ou_paths = np.zeros((self.nObs, self.slices + 1))
        ou_paths[:, 0] = self.S0
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            ou_paths[:, t] = ou_paths[:, t - 1] + kappa * (theta - ou_paths[:, t - 1]) * self.dt + sigma * z[:,
                                                                                                           t - 1] * np.sqrt(
                self.dt)
        return ou_paths

    def jump_diffusion_model(self, jump_intensity: float, jump_mean: float, jump_std: float):
        jd_paths = np.zeros((self.nObs, self.slices + 1))
        jd_paths[:, 0] = self.S0
        z = self._generate_standard_normal_random_variables()
        jumps = np.random.poisson(lam=jump_intensity * self.dt, size=(self.nObs, self.slices))
        for t in range(1, self.slices + 1):
            jump_sizes = np.random.normal(loc=jump_mean, scale=jump_std, size=(self.nObs, jumps[:, t - 1].max()))
            total_jump = np.array([jump_sizes[i, :jumps[i, t - 1]].sum() for i in range(self.nObs)])
            jd_paths[:, t] = jd_paths[:, t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * z[:, t - 1] * np.sqrt(self.dt)) + total_jump
        return jd_paths

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