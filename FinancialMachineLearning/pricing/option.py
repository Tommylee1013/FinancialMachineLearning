import numpy as np
import scipy.stats as ss

class OptionPricing:
    def __init__(self, stock_price_paths: np.ndarray, interest_rate: float, maturity: float):
        self.stock_price_paths = stock_price_paths
        self.interest_rate = interest_rate
        self.maturity = maturity
        self.discount_factor = np.exp(-interest_rate * maturity)

        self.time_steps = stock_price_paths.shape[-1]
        self.dt = maturity / (self.time_steps - 1)
        self.discount_table = np.exp(-interest_rate * np.arange(self.time_steps) * self.dt)

    def european_call_option(self, exercise_price: float) -> np.ndarray:
        call_payoffs = np.maximum(self.stock_price_paths - exercise_price, 0)
        discounted_call_payoffs = call_payoffs * self.discount_factor
        return discounted_call_payoffs

    def european_put_option(self, exercise_price: float) -> np.ndarray:
        put_payoffs = np.maximum(exercise_price - self.stock_price_paths, 0)
        discounted_put_payoffs = put_payoffs * self.discount_factor
        return discounted_put_payoffs

    def asian_call_option(self, exercise_price: float, average_method: str = 'arithmetic') -> np.ndarray:
        if average_method == 'arithmetic':
            average_price = self.stock_price_paths.mean(axis=0)
        elif average_method == 'geometric':
            average_price = ss.gmean(self.stock_price_paths, axis=0)
        else:
            raise ValueError("avg_method must be either 'arithmetic' or 'geometric'")

        payoffs = np.maximum(average_price - exercise_price, 0)
        discounted_payoffs = payoffs * self.discount_factor

        return discounted_payoffs

    def asian_put_option(self, exercise_price: float, average_method: str = 'arithmetic') -> np.ndarray:
        if average_method == 'arithmetic':
            average_price = self.stock_price_paths.mean(axis=0)
        elif average_method == 'geometric':
            average_price = ss.gmean(self.stock_price_paths, axis=0)
        else:
            raise ValueError("avg_method must be either 'arithmetic' or 'geometric'")

        payoffs = np.maximum(exercise_price - average_price, 0)
        discounted_payoffs = payoffs * self.discount_factor

        return discounted_payoffs

    def american_call_option(self, exercise_price: float, poly_degree: int = 2) -> np.ndarray:
        """
        American call option pricing using Longstaff-Schwartz method.
        """
        payoffs = np.maximum(self.stock_price_paths - exercise_price, 0)
        cash_flows = np.copy(payoffs)

        for t in reversed(range(1, self.time_steps - 1)):
            in_the_money = self.stock_price_paths[:, t] > exercise_price
            if np.any(in_the_money):
                regression_x = self.stock_price_paths[in_the_money, t]
                regression_y = cash_flows[in_the_money, t + 1] * np.exp(-self.interest_rate * self.dt)
                if len(regression_x) > 0:
                    coefficients = np.polyfit(regression_x, regression_y, poly_degree)
                    continuation_values = np.polyval(coefficients, self.stock_price_paths[in_the_money, t])
                    exercise_decision = payoffs[in_the_money, t] > continuation_values
                    cash_flows[in_the_money, t] = np.where(exercise_decision, payoffs[in_the_money, t],
                                                           cash_flows[in_the_money, t + 1] * np.exp(
                                                               -self.interest_rate * self.dt))
                cash_flows[~in_the_money, t] = cash_flows[~in_the_money, t + 1] * np.exp(-self.interest_rate * self.dt)

        discounted_cash_flows = cash_flows * np.exp(-self.interest_rate * self.dt)
        return discounted_cash_flows

    def american_put_option(self, exercise_price: float, poly_degree: int = 2) -> np.ndarray:
        """
        American put option pricing using longstaff schwartz methods
        """
        payoffs = np.maximum(exercise_price - self.stock_price_paths, 0)
        cash_flows = np.copy(payoffs)

        for t in reversed(range(1, self.time_steps - 1)):
            in_the_money = self.stock_price_paths[:, t] > exercise_price
            if np.any(in_the_money):
                regression_x = self.stock_price_paths[in_the_money, t]
                regression_y = cash_flows[in_the_money, t + 1] * np.exp(-self.interest_rate * self.dt)
                if len(regression_x) > 0:
                    coefficients = np.polyfit(regression_x, regression_y, poly_degree)
                    continuation_values = np.polyval(coefficients, self.stock_price_paths[in_the_money, t])
                    exercise_decision = payoffs[in_the_money, t] > continuation_values
                    cash_flows[in_the_money, t] = np.where(exercise_decision, payoffs[in_the_money, t],
                                                           cash_flows[in_the_money, t + 1] * np.exp(
                                                               -self.interest_rate * self.dt))
                cash_flows[~in_the_money, t] = cash_flows[~in_the_money, t + 1] * np.exp(-self.interest_rate * self.dt)

        discounted_cash_flows = cash_flows * np.exp(-self.interest_rate * self.dt)
        return discounted_cash_flows