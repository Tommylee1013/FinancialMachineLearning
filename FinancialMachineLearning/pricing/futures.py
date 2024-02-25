import numpy as np
class FutureForwardPricing:
    def __init__(self, stock_paths, interest_rate, dividend_yield, maturity):
        self.stock_paths = stock_paths
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield
        self.maturity = maturity
    def futures(self):
        """
        Calculate the future price as the average of the simulated stock prices at maturity.
        """
        future_prices = self.stock_paths[:, -1]  # Prices at maturity
        future_price = np.mean(future_prices)
        return future_price
    def forwards(self):
        """
        Calculate the forward price using the formula: S0 * exp((r - q) * T),
        where S0 is the initial stock price, r is the risk-free interest rate,
        q is the dividend yield, and T is the time to maturity.
        """
        S0 = self.stock_paths[0, 0]  # Initial stock price
        forward_price = S0 * np.exp((self.interest_rate - self.dividend_yield) * self.maturity)
        return forward_price