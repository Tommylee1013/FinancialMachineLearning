import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from FinancialMachineLearning.portfolio_optimization.return_estimators import ReturnEstimation

class MeanVarianceOptimisation:
    def __init__(self, calculate_expected_returns='mean'):
        self.weights = list()
        self.portfolio_risk = None
        self.portfolio_return = None
        self.portfolio_sharpe_ratio = None
        self.calculate_expected_returns = calculate_expected_returns
        self.returns_estimator = ReturnEstimation()
        self.weight_bounds = None

    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 expected_asset_returns=None,
                 covariance_matrix=None,
                 solution='inverse_variance',
                 risk_free_rate=0.05,
                 target_return=0.2,
                 weight_bounds=(0, 1),
                 resample_by=None):
        if asset_prices is None and expected_asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or expected returns "
                             "and a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")
        self.weight_bounds = weight_bounds
        if expected_asset_returns is None:
            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.mean_historical_return_calculator(
                                                                        asset_prices=asset_prices,
                                                                        resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.exponential_historical_return_calculator(
                                                                        asset_prices=asset_prices,
                                                                        resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
        expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            returns = self.returns_estimator.return_calculator(asset_prices=asset_prices, resample_by=resample_by)
            covariance_matrix = returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(covariance=cov)
        elif solution == 'min_volatility':
            self.weights, self.portfolio_risk = self._min_volatility(covariance=cov, num_assets=len(asset_names))
        elif solution == 'max_sharpe':
            self.weights, self.portfolio_risk, self.portfolio_return = self._max_sharpe(
                                                                                    covariance=cov,
                                                                                    expected_returns=expected_asset_returns,
                                                                                    risk_free_rate=risk_free_rate,
                                                                                    num_assets=len(asset_names))
        elif solution == 'efficient_risk':
            self.weights, self.portfolio_risk, self.portfolio_return = self._min_volatility_for_target_return(
                                                                                    covariance=cov,
                                                                                    expected_returns=expected_asset_returns,
                                                                                    target_return=target_return,
                                                                                    num_assets=len(asset_names))
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - "
                             "inverse_variance, min_volatility, max_sharpe and efficient_risk.")

        negative_weight_indices = np.argwhere(self.weights < 0)
        self.weights[negative_weight_indices] = np.round(self.weights[negative_weight_indices], 3)

        if self.portfolio_risk is None:
            self.portfolio_risk = self.weights @ cov @ self.weights.T
        if self.portfolio_return is None:
            self.portfolio_return = self.weights @ expected_asset_returns
        self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))

        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T

    @staticmethod
    def _inverse_variance(covariance):
        ivp = 1. / np.diag(covariance)
        ivp /= ivp.sum()
        return ivp

    def _min_volatility(self, covariance, num_assets):
        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        risk = cp.quad_form(weights, covariance)
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum(weights) == 1,
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        if isinstance(self.weight_bounds, dict):
            asset_indices = list(range(num_assets))
            for asset_index in asset_indices:
                lower_bound, upper_bound = self.weight_bounds.get(asset_index, (0, 1))
                allocation_constraints.extend(
                    [
                        weights[asset_index] >= lower_bound,
                        weights[asset_index] <= min(upper_bound, 1)
                    ]
                )

        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')
        return weights.value, risk.value ** 0.5

    def _max_sharpe(self, covariance, expected_returns, risk_free_rate, num_assets):
        y = cp.Variable(num_assets)
        y.value = np.array([1 / num_assets] * num_assets)
        kappa = cp.Variable(1)
        risk = cp.quad_form(y, covariance)

        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum((expected_returns - risk_free_rate).T @ y) == 1,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    y >= kappa * self.weight_bounds[0],
                    y <= kappa * self.weight_bounds[1]
                ]
            )
        if isinstance(self.weight_bounds, dict):
            asset_indices = list(range(num_assets))
            for asset_index in asset_indices:
                lower_bound, upper_bound = self.weight_bounds.get(asset_index, (0, 1))
                allocation_constraints.extend(
                    [
                        y[asset_index] >= kappa * lower_bound,
                        y[asset_index] <= kappa * upper_bound
                    ]
                )

        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if y.value is None or kappa.value is None:
            raise ValueError('No optimal set of weights found.')
        weights = y.value / kappa.value
        portfolio_return = (expected_returns.T @ weights)[0]
        return weights, risk.value ** 0.5, portfolio_return

    def _min_volatility_for_target_return(self, covariance, expected_returns, target_return, num_assets):
        weights = cp.Variable(num_assets)
        risk = cp.quad_form(weights, covariance)
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum(weights) == 1,
            (expected_returns.T @ weights)[0] == target_return,
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        if isinstance(self.weight_bounds, dict):
            asset_indices = list(range(num_assets))
            for asset_index in asset_indices:
                lower_bound, upper_bound = self.weight_bounds.get(asset_index, (0, 1))
                allocation_constraints.extend(
                    [
                        weights[asset_index] >= lower_bound,
                        weights[asset_index] <= min(upper_bound, 1)
                    ]
                )
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve()
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')
        return weights.value, risk.value ** 0.5, target_return

    def plot_efficient_frontier(self,
                                covariance,
                                expected_asset_returns,
                                num_assets,
                                min_return=0,
                                max_return=0.4,
                                risk_free_rate=0.05):
        expected_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))
        volatilities = []
        returns = []
        sharpe_ratios = []
        for portfolio_return in np.linspace(min_return, max_return, 100):
            _, risk, _ = self._min_volatility_for_target_return(covariance=covariance,
                                                   expected_returns=expected_returns,
                                                   target_return=portfolio_return,
                                                   num_assets=num_assets)
            volatilities.append(risk)
            returns.append(portfolio_return)
            sharpe_ratios.append((portfolio_return - risk_free_rate) / (risk ** 0.5 + 1e-16))
        max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))
        min_volatility_index = volatilities.index(min(volatilities))
        figure = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(volatilities[max_sharpe_ratio_index], returns[max_sharpe_ratio_index], marker='*', color='g', s=400, label='Maximum Sharpe Ratio')
        plt.scatter(volatilities[min_volatility_index], returns[min_volatility_index], marker='*', color='r', s=400, label='Minimum Volatility')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend(loc='upper left')
        return figure