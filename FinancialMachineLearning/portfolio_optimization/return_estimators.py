class ReturnEstimation:
    def __init__(self) -> None :
        return
    @staticmethod
    def mean_historical_return_calculator(asset_prices, resample_by = None, frequency : int = 252):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns
    @staticmethod
    def exponential_historical_return_calculator(asset_prices, resample_by = None,
                                                 frequency : int = 252, span : int = 500):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span = span).mean().iloc[-1] * frequency
        return returns
    @staticmethod
    def return_calculator(asset_prices, resample_by = None):
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns