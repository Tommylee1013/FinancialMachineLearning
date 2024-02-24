import numpy as np
import scipy.stats as ss

class OptionPricing:
    def __init__(self, stock_price_paths: np.ndarray, interest_rate: float, maturity: float):
        self.stock_price_paths = stock_price_paths
        self.interest_rate = interest_rate
        self.maturity = maturity
        self.discount_factor = np.exp(-interest_rate * maturity)

    def european_call_option(self, strike_price: float) -> np.ndarray:
        call_payoffs = np.maximum(self.stock_price_paths - strike_price, 0)
        discounted_call_payoffs = call_payoffs * self.discount_factor
        return discounted_call_payoffs

    def european_put_option(self, strike_price: float) -> np.ndarray:
        put_payoffs = np.maximum(strike_price - self.stock_price_paths, 0)
        discounted_put_payoffs = put_payoffs * self.discount_factor
        return discounted_put_payoffs