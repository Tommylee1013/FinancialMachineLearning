import numbers
from math import log, ceil
import numpy as np
import pandas as pd
from FinancialMachineLearning.portfolio_optimization.return_estimators import ReturnEstimation

class CriticalLineAlgorithm:
    def __init__(self, weight_bounds = (0, 1), calculate_expected_returns = "mean"):
        self.weight_bounds = weight_bounds
        self.calculate_expected_returns = calculate_expected_returns
        self.weights = list()
        self.lambdas = list()
        self.gammas = list()
        self.free_weights = list()
        self.expected_returns = None
        self.cov_matrix = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.max_sharpe = None
        self.min_var = None
        self.efficient_frontier_means = None
        self.efficient_frontier_sigma = None
        self.returns_estimator = ReturnEstimation()

    @staticmethod
    def _infnone(number):
        return float("-inf") if number is None else number

    def _init_algo(self):
        structured_array = np.zeros((self.expected_returns.shape[0]), dtype=[("id", int), ("mu", float)])
        expected_returns = [self.expected_returns[i][0] for i in
                            range(self.expected_returns.shape[0])]
        structured_array[:] = list(zip(list(range(self.expected_returns.shape[0])), expected_returns))
        expected_returns = np.sort(structured_array, order="mu")
        index, weights = expected_returns.shape[0], np.copy(self.lower_bounds)
        while np.sum(weights) < 1:
            index -= 1
            weights[expected_returns[index][0]] = self.upper_bounds[expected_returns[index][0]]
        weights[expected_returns[index][0]] += 1 - np.sum(weights)
        return [expected_returns[index][0]], weights

    @staticmethod
    def _compute_bi(c_final, asset_bounds_i):
        if c_final > 0:
            return asset_bounds_i[1][0]
        return asset_bounds_i[0][0]

    def _compute_w(self, covar_f_inv, covar_fb, mean_f, w_b):
        ones_f = np.ones(mean_f.shape)
        g_1 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        g_2 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        if w_b is None:
            g_final, w_1 = float(-self.lambdas[-1] * g_1 / g_2 + 1 / g_2), 0
        else:
            ones_b = np.ones(w_b.shape)
            g_3 = np.dot(ones_b.T, w_b)
            g_4 = np.dot(covar_f_inv, covar_fb)
            w_1 = np.dot(g_4, w_b)
            g_4 = np.dot(ones_f.T, w_1)
            g_final = float(-self.lambdas[-1] * g_1 / g_2 + (1 - g_3 + g_4) / g_2)

        w_2 = np.dot(covar_f_inv, ones_f)
        w_3 = np.dot(covar_f_inv, mean_f)
        free_asset_weights = -1 * w_1 + g_final * w_2 + self.lambdas[-1] * w_3
        return free_asset_weights, g_final

    def _compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, asset_index, b_i):
        ones_f = np.ones(mean_f.shape)
        c_1 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        c_2 = np.dot(covar_f_inv, mean_f)
        c_3 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        c_4 = np.dot(covar_f_inv, ones_f)
        c_final = -1 * c_1 * c_2[asset_index] + c_3 * c_4[asset_index]
        if c_final == 0:
            return None, None

        if isinstance(b_i, list):
            b_i = self._compute_bi(c_final, b_i)

        if w_b is None:
            return float((c_4[asset_index] - c_1 * b_i) / c_final), b_i

        ones_b = np.ones(w_b.shape)
        l_1 = np.dot(ones_b.T, w_b)
        l_2 = np.dot(covar_f_inv, covar_fb)
        l_3 = np.dot(l_2, w_b)
        l_2 = np.dot(ones_f.T, l_3)
        lambda_value = float(((1 - l_1 + l_2) * c_4[asset_index] - c_1 * (b_i + l_3[asset_index])) / c_final)
        return lambda_value, b_i

    def _get_matrices(self, free_weights):
        covar_f = self._reduce_matrix(self.cov_matrix, free_weights, free_weights)
        mean_f = self._reduce_matrix(self.expected_returns, free_weights, [0])
        bounded_weights = self._get_bounded_weights(free_weights)
        covar_fb = self._reduce_matrix(self.cov_matrix, free_weights, bounded_weights)
        w_b = self._reduce_matrix(self.weights[-1], bounded_weights, [0])
        return covar_f, covar_fb, mean_f, w_b

    def _get_bounded_weights(self, free_weights):
        return self._diff_lists(list(range(self.expected_returns.shape[0])), free_weights)

    @staticmethod
    def _diff_lists(list_1, list_2):
        return list(set(list_1) - set(list_2))

    @staticmethod
    def _reduce_matrix(matrix, row_indices, col_indices):
        return matrix[np.ix_(row_indices, col_indices)]

    def _purge_num_err(self, tol):
        index_1 = 0
        while True:
            flag = False
            if index_1 == len(self.weights):
                break
            if abs(sum(self.weights[index_1]) - 1) > tol:
                flag = True
            else:
                for index_2 in range(len(self.weights[index_1])):
                    if (
                            self.weights[index_1][index_2] - self.lower_bounds[index_2] < -tol
                            or self.weights[index_1][index_2] - self.upper_bounds[index_2] > tol
                    ):
                        flag = True
                        break
            if flag is True:
                del self.weights[index_1]
                del self.lambdas[index_1]
                del self.gammas[index_1]
                del self.free_weights[index_1]
            else:
                index_1 += 1

    def _purge_excess(self):
        index_1, repeat = 0, False
        while True:
            if repeat is False:
                index_1 += 1
            if index_1 >= len(self.weights) - 1:
                break
            weights = self.weights[index_1]
            mean = np.dot(weights.T, self.expected_returns)[0, 0]
            index_2, repeat = index_1 + 1, False
            while True:
                if index_2 == len(self.weights):
                    break
                weights = self.weights[index_2]
                mean_ = np.dot(weights.T, self.expected_returns)[0, 0]
                if mean < mean_:
                    del self.weights[index_1]
                    del self.lambdas[index_1]
                    del self.gammas[index_1]
                    del self.free_weights[index_1]
                    repeat = True
                    break
                index_2 += 1

    @staticmethod
    def _golden_section(obj, left, right, **kwargs):
        tol, sign, args = 1.0e-9, -1, None
        args = kwargs.get("args", None)
        num_iterations = int(ceil(-2.078087 * log(tol / abs(right - left))))
        gs_ratio = 0.618033989
        complementary_gs_ratio = 1.0 - gs_ratio
        x_1 = gs_ratio * left + complementary_gs_ratio * right
        x_2 = complementary_gs_ratio * left + gs_ratio * right
        f_1 = sign * obj(x_1, *args)
        f_2 = sign * obj(x_2, *args)

        for _ in range(num_iterations):
            if f_1 > f_2:
                left = x_1
                x_1 = x_2
                f_1 = f_2
                x_2 = complementary_gs_ratio * left + gs_ratio * right
                f_2 = sign * obj(x_2, *args)
            else:
                right = x_2
                x_2 = x_1
                f_2 = f_1
                x_1 = gs_ratio * left + complementary_gs_ratio * right
                f_1 = sign * obj(x_1, *args)

        if f_1 < f_2:
            return x_1, sign * f_1
        return x_2, sign * f_2

    def _eval_sr(self, alpha, w_0, w_1):
        weights = alpha * w_0 + (1 - alpha) * w_1
        returns = np.dot(weights.T, self.expected_returns)[0, 0]
        volatility = np.dot(np.dot(weights.T, self.cov_matrix), weights)[0, 0] ** 0.5
        return returns / volatility

    def _bound_free_weight(self, free_weights):
        lambda_in = None
        i_in = None
        bi_in = None
        if len(free_weights) > 1:
            covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
            covar_f_inv = np.linalg.inv(covar_f)
            j = 0
            for i in free_weights:
                lambda_i, b_i = self._compute_lambda(
                    covar_f_inv, covar_fb, mean_f, w_b, j, [self.lower_bounds[i], self.upper_bounds[i]]
                )
                if self._infnone(lambda_i) > self._infnone(lambda_in):
                    lambda_in, i_in, bi_in = lambda_i, i, b_i
                j += 1
        return lambda_in, i_in, bi_in

    def _free_bound_weight(self, free_weights):
        lambda_out = None
        i_out = None
        if len(free_weights) < self.expected_returns.shape[0]:
            bounded_weight_indices = self._get_bounded_weights(free_weights)
            for i in bounded_weight_indices:
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights + [i])
                covar_f_inv = np.linalg.inv(covar_f)
                lambda_i, _ = self._compute_lambda(
                    covar_f_inv,
                    covar_fb,
                    mean_f,
                    w_b,
                    mean_f.shape[0] - 1,
                    self.weights[-1][i],
                )
                if (self.lambdas[-1] is None or lambda_i < self.lambdas[-1]) and lambda_i > self._infnone(lambda_out):
                    lambda_out, i_out = lambda_i, i
        return lambda_out, i_out

    def _initialise(self, asset_prices, expected_asset_returns, covariance_matrix, resample_by):
        self.expected_returns = expected_asset_returns
        if expected_asset_returns is None:
            if self.calculate_expected_returns == "mean":
                self.expected_returns = self.returns_estimator.mean_historical_return_calculator(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                self.expected_returns = self.returns_estimator.exponential_historical_return_calculator(asset_prices = asset_prices,
                    resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
        self.expected_returns = np.array(self.expected_returns).reshape((len(self.expected_returns), 1))
        if (self.expected_returns == np.ones(self.expected_returns.shape) * self.expected_returns.mean()).all():
            self.expected_returns[-1, 0] += 1e-5

        if covariance_matrix is None:
            returns = self.returns_estimator.return_calculator(asset_prices=asset_prices, resample_by=resample_by)
            covariance_matrix = returns.cov()
        self.cov_matrix = np.asarray(covariance_matrix)

        if isinstance(self.weight_bounds[0], numbers.Real):
            self.lower_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[0]
        else:
            self.lower_bounds = np.array(self.weight_bounds[0]).reshape(self.expected_returns.shape)

        if isinstance(self.weight_bounds[0], numbers.Real):
            self.upper_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[1]
        else:
            self.upper_bounds = np.array(self.weight_bounds[1]).reshape(self.expected_returns.shape)

        self.weights = []
        self.lambdas = []
        self.gammas = []
        self.free_weights = []

    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 expected_asset_returns=None,
                 covariance_matrix=None,
                 solution="cla_turning_points",
                 resample_by=None):
        if asset_prices is None and (expected_asset_returns is None or covariance_matrix is None):
            raise ValueError("Either supply your own asset returns matrix or pass the asset prices as input")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        self._initialise(asset_prices=asset_prices,
                         resample_by=resample_by,
                         expected_asset_returns=expected_asset_returns,
                         covariance_matrix=covariance_matrix)

        free_weights, weights = self._init_algo()
        self.weights.append(np.copy(weights))
        self.lambdas.append(None)
        self.gammas.append(None)
        self.free_weights.append(free_weights[:])
        while True:
            lambda_in, i_in, bi_in = self._bound_free_weight(free_weights)
            lambda_out, i_out = self._free_bound_weight(free_weights)
            if (lambda_in is None or lambda_in < 0) and (lambda_out is None or lambda_out < 0):
                self.lambdas.append(0)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)
                mean_f = np.zeros(mean_f.shape)
            else:
                if self._infnone(lambda_in) > self._infnone(lambda_out):
                    self.lambdas.append(lambda_in)
                    free_weights.remove(i_in)
                    weights[i_in] = bi_in
                else:
                    self.lambdas.append(lambda_out)
                    free_weights.append(i_out)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)
            w_f, gamma = self._compute_w(covar_f_inv, covar_fb, mean_f, w_b)
            for i in range(len(free_weights)):
                weights[free_weights[i]] = w_f[i]
            self.weights.append(np.copy(weights))
            self.gammas.append(gamma)
            self.free_weights.append(free_weights[:])
            if self.lambdas[-1] == 0:
                break
        self._purge_num_err(10e-10)
        self._purge_excess()
        self._compute_solution(assets=asset_names, solution=solution)

    def _compute_solution(self, assets, solution):
        if solution == "max_sharpe":
            self.max_sharpe, self.weights = self._max_sharpe()
            self.weights = pd.DataFrame(self.weights)
            self.weights.index = assets
            self.weights = self.weights.T
        elif solution == "min_volatility":
            self.min_var, self.weights = self._min_volatility()
            self.weights = pd.DataFrame(self.weights)
            self.weights.index = assets
            self.weights = self.weights.T
        elif solution == "efficient_frontier":
            self.efficient_frontier_means, self.efficient_frontier_sigma, self.weights = self._efficient_frontier()
            weights_copy = self.weights.copy()
            for i, turning_point in enumerate(weights_copy):
                self.weights[i] = turning_point.reshape(1, -1)[0]
            self.weights = pd.DataFrame(self.weights, columns=assets)
        elif solution == "cla_turning_points":
            # Reshape the weight matrix
            weights_copy = self.weights.copy()
            for i, turning_point in enumerate(weights_copy):
                self.weights[i] = turning_point.reshape(1, -1)[0]
            self.weights = pd.DataFrame(self.weights, columns=assets)
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - cla_turning_points, "
                             "efficient_frontier, min_volatility, max_sharpe")

    def _max_sharpe(self):
        w_sr, sharpe_ratios = [], []
        for i in range(len(self.weights) - 1):
            w_0 = np.copy(self.weights[i])
            w_1 = np.copy(self.weights[i + 1])
            kwargs = {"minimum": False, "args": (w_0, w_1)}
            alpha, sharpe_ratio = self._golden_section(self._eval_sr, 0, 1, **kwargs)
            w_sr.append(alpha * w_0 + (1 - alpha) * w_1)
            sharpe_ratios.append(sharpe_ratio)

        maximum_sharp_ratio = max(sharpe_ratios)
        weights_with_max_sharpe_ratio = w_sr[sharpe_ratios.index(maximum_sharp_ratio)]
        return maximum_sharp_ratio, weights_with_max_sharpe_ratio

    def _min_volatility(self):
        var = []
        for weights in self.weights:
            volatility = np.dot(np.dot(weights.T, self.cov_matrix), weights)
            var.append(volatility)
        min_var = min(var)
        return min_var ** .5, self.weights[var.index(min_var)]

    def _efficient_frontier(self, points=100):
        means, sigma, weights = [], [], []
        partitions = np.linspace(0, 1, points // len(self.weights))[:-1]
        b = list(range(len(self.weights) - 1))
        for i in b:
            w_0, w_1 = self.weights[i], self.weights[i + 1]

            if i == b[-1]:
                partitions = np.linspace(0, 1, points // len(self.weights))

            for j in partitions:
                w = w_1 * j + (1 - j) * w_0
                weights.append(np.copy(w))
                means.append(np.dot(w.T, self.expected_returns)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5)
        return means, sigma, weights