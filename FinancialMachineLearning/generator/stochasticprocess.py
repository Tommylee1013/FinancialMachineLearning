import numpy as np
import pandas as pd
from typing import Union
from sklearn.datasets import make_classification
import datetime
import scipy.stats as ss

class MonteCarloSimulation:
    def __init__(self, interest_rate: float, initial_price: float, maturity: float, sigma: float,
                 dividend_yield: float, nObs: int, slices: int, random_state: bool or int = False):
        self.S0 = float(initial_price)
        self.T = float(maturity)
        self.dividend_yield = float(dividend_yield)
        self.slices = int(slices)
        self.nObs = int(nObs)
        self.dt = self.T / self.slices
        self.mu = interest_rate
        self.r = np.full((self.nObs, self.slices), interest_rate * self.dt)
        self.discount_table = np.exp(np.cumsum(-self.r, axis=1))
        self.sigma = np.full((self.nObs, self.slices), sigma)
        self.terminal_prices = []
        self.z_t = np.random.standard_normal((self.nObs, self.slices))

        if type(random_state) is bool:
            if random_state: np.random.seed(0)
        elif type(random_state) is int:
            np.random.seed(random_state)

    def vasicek_model(self, kappa: float, theta: float, sigma_r: float) -> np.ndarray:
        interest_z_t = np.random.standard_normal((self.nObs, self.slices))
        interest_array = np.full(
            (self.nObs, self.slices), self.r[0, 0] * self.dt
        )

        for i in range(1, self.slices):
            interest_array[:, i] = theta + np.exp(- kappa / self.slices) * (interest_array[:, i - 1] - theta) + np.sqrt(
                sigma_r ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa / self.slices))
            ) * interest_z_t[:, i]

        self.r = interest_array
        return self.r

    def cox_ingersoll_ross_model(self, kappa: float, theta: float, sigma_r: float) -> np.ndarray:
        if 2 * kappa * theta > sigma_r ** 2:
            raise ValueError(
                "Simulation Error. to ensure the interest rate positive, you need to set environment to '2 * kappa * theta < sigma ** 2'.")
        interest_array = np.full(
            (self.nObs, self.slices), self.r[0, 0] * self.dt
        )
        df = 4 * theta * kappa / sigma_r ** 2

        for i in range(1, self.slices):
            _lambda = (4 * kappa * np.exp(-kappa / self.slices) * interest_array[:, i - 1] / (
                    sigma_r ** 2 * (1 - np.exp(- kappa / self.slices))))
            _chi_square_factor = np.random.noncentral_chisquare(
                df=df, nonc=_lambda, size=self.nObs
            )
            interest_array[:, i] = sigma_r ** 2 * (1 - np.exp(-kappa / self.slices)) / (4 * kappa) * _chi_square_factor

        self.r = interest_array
        return self.r

    def heston_model(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0) -> np.ndarray:
        variance_v = sigma_v ** 2
        if 2 * kappa * theta > variance_v:
            raise ValueError(
                "Simulation Error. to ensure the interest rate positive, you need to set environment to '2 * kappa * theta < sigma_v ** 2'.")

        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])

        zt = ss.multivariate_normal.rvs(
            mean=mu, cov=cov, size=(self.nObs, self.slices))
        variance_array = np.full(
            (self.nObs, self.slices), self.sigma[0, 0] ** 2
        )
        self.z_t = zt[:, :, 0]
        zt_v = zt[:, :, 1]

        for i in range(1, self.slices):
            previous_slice_variance = np.maximum(variance_array[:, i - 1], 0)
            drift = kappa * (theta - previous_slice_variance) * self.dt
            diffusion = sigma_v * np.sqrt(previous_slice_variance) * zt_v[:, i - 1] * np.sqrt(self.dt)
            delta_vt = drift + diffusion
            variance_array[:, i] = variance_array[:, i - 1] + delta_vt

        self.sigma = np.sqrt(np.maximum(variance_array, 0))
        return self.sigma

    def geometric_brownian_motion(self) -> np.ndarray:
        self.exp_mean = (self.mu - self.dividend_yield - (self.sigma ** 2.0) * 0.5) * self.dt
        self.exp_diffusion = self.sigma * np.sqrt(self.dt)

        self.price_array = np.zeros((self.nObs, self.slices))
        self.price_array[:, 0] = self.S0

        for i in range(1, self.slices):
            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean[:, i - 1] + self.exp_diffusion[:, i - 1] * self.z_t[:, i - 1]
            )

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.average(self.terminal_prices)

        result = (self.price_array - self.price_array.mean()).cumsum(axis = 0)
        result = result + self.S0

        return result

    def ornstein_uhlenbeck(self, kappa: float, theta: float) -> np.ndarray:
        self.price_array = np.zeros((self.nObs, self.slices))
        self.price_array[:, 0] = self.S0
        sigma = self.sigma[0, 0]

        for i in range(1, self.slices):
            dX = kappa * (theta - self.price_array[:, i - 1]) * self.dt + sigma * np.sqrt(self.dt) * self.z_t[:, i - 1]
            self.price_array[:, i] = self.price_array[:, i - 1] + dX

        self.terminal_prices = self.price_array[:, -1]
        self.process_expectation = np.average(self.terminal_prices)

        result = (self.price_array - self.price_array.mean()).cumsum(axis=0)
        result = result + self.S0

        return result

    def jump_diffusion_model(self, jump_alpha: float, jump_std: float, poisson_lambda: float) -> np.ndarray:
        self.z_t_stock = np.random.standard_normal((self.nObs, self.slices))
        self.price_array = np.zeros((self.nObs, self.slices))
        self.price_array[:, 0] = self.S0

        self.k = np.exp(jump_alpha) - 1

        self.exp_mean = (self.mu - self.dividend_yield - poisson_lambda * self.k - (self.sigma ** 2.0) * 0.5) * self.dt
        self.exp_diffusion = self.sigma * np.sqrt(self.dt)

        for i in range(1, self.slices):
            self.sum_w = []
            self.m = np.random.poisson(lam=poisson_lambda, size=self.nObs)

            for j in self.m:
                self.sum_w.append(np.sum(np.random.standard_normal(j)))

            self.poisson_jump_factor = np.exp(
                self.m * (jump_alpha - 0.5 * jump_std ** 2) + jump_std * np.array(self.sum_w)
            )

            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean[:, i] + self.exp_diffusion[:, i]
                * self.z_t_stock[:, i]
            ) * self.poisson_jump_factor

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.average(self.terminal_prices)

        result = (self.price_array - self.price_array.mean()).cumsum(axis=0)
        result = result + self.S0

        return result

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