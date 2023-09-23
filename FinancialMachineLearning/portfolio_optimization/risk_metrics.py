# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd

class RiskMetrics:
    def __init__(self):
        return

    @staticmethod
    def calculate_variance(covariance, weights):
        return np.dot(weights, np.dot(covariance, weights))

    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        return returns.quantile(confidence_level, interpolation='higher')[0]

    def calculate_expected_shortfall(self, returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        value_at_risk = self.calculate_value_at_risk(returns, confidence_level)
        expected_shortfall = np.nanmean(returns[returns < value_at_risk])
        return expected_shortfall

    @staticmethod
    def calculate_conditional_drawdown_risk(returns, confidence_level=0.05):
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        drawdown = returns.expanding().max() - returns
        max_drawdown = drawdown.expanding().max()
        max_drawdown_at_confidence_level = max_drawdown.quantile(confidence_level, interpolation='higher')
        conditional_drawdown = np.nanmean(max_drawdown[max_drawdown > max_drawdown_at_confidence_level])
        return conditional_drawdown
