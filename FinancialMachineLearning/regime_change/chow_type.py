import pandas as pd
from FinancialMachineLearning.regime_change.supremum_adf import beta
from FinancialMachineLearning.utils.multiprocess import mp_pandas_obj

def chow_type_adf(series: pd.Series, molecule: list) -> pd.Series:
    dfc_series = pd.Series(index=molecule)

    for index in molecule:
        series_diff = series.diff().dropna()
        series_lag = series.shift(1).dropna()
        series_lag[:index] = 0

        y = series_diff.loc[series_lag.index].values
        x = series_lag.values
        coefs, coef_vars = beta(x.reshape(-1, 1), y)
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series[index] = b_estimate / (b_var ** 0.5)
    return dfc_series

def chow_type_stat(series: pd.Series, min_length: int = 20, num_threads: int = 8) -> pd.Series:
    molecule = series.index[min_length:series.shape[0] - min_length]
    dfc_series = mp_pandas_obj(func=chow_type_adf,
                               pd_obj=('molecule', molecule),
                               series=series,
                               num_threads=num_threads,
                               )
    return dfc_series
