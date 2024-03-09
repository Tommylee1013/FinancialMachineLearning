from typing import Union, Tuple
import pandas as pd
import numpy as np
from FinancialMachineLearning.utils.multiprocess import mp_pandas_obj

def supremum_adf_test(X: pd.DataFrame, y: pd.DataFrame, min_length: int) -> float:
    start_points, bsadf = range(0, y.shape[0] - min_length + 1), -np.inf
    for start in start_points:
        y_, X_ = y[start:], X[start:]
        b_mean_, b_std_ = beta(X_, y_)
        if not np.isnan(b_mean_[0]):
            b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
            all_adf = b_mean_ / b_std_
            if all_adf > bsadf:
                bsadf = all_adf
    return bsadf

def set_sadf_data(series: pd.Series, model: str, lags: Union[int, list],
             add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = lag_df(series_diff, lags).dropna()
    x['y_lagged'] = series.shift(1).loc[x.index]
    y = series_diff.loc[x.index]

    if add_const is True:
        x['const'] = 1

    if model == 'linear':
        x['trend'] = np.arange(x.shape[0])
        beta_column = 'y_lagged'
    elif model == 'quadratic':
        x['trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'y_lagged'
    elif model == 'sm_poly_1':
        y = series.loc[y.index]
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_poly_2':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_exp':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        beta_column = 'trend'
    elif model == 'sm_power':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['log_trend'] = np.log(np.arange(x.shape[0]))
        beta_column = 'log_trend'
    else:
        raise ValueError('Unknown model')

    columns = list(x.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    x = x[columns]
    return x, y


def lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how='outer')
    return df_lagged


def beta(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.array, np.array]:
    xy = np.dot(X.T, y)
    xx = np.dot(X.T, X)
    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return [np.nan], [[np.nan, np.nan]]
    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(X, b_mean)
    b_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * xx_inv
    return b_mean, b_var

def sadf_outer_loop(X: pd.DataFrame, y: pd.DataFrame, min_length: int, molecule: list) -> pd.Series:
    sadf_series = pd.Series(index=molecule)
    for index in molecule:
        X_subset = X.loc[:index].values
        y_subset = y.loc[:index].values.reshape(-1, 1)
        value = supremum_adf_test(X_subset, y_subset, min_length)
        sadf_series[index] = value
    return sadf_series


def get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
             num_threads: int = 8) -> pd.Series:
    X, y = set_sadf_data(series, model, lags, add_const)
    molecule = y.index[min_length:y.shape[0]]

    sadf_series = mp_pandas_obj(
        func = sadf_outer_loop,
        pd_obj = ('molecule', molecule),
        X = X,
        y = y,
        min_length = min_length,
        num_threads = num_threads,
    )
    return sadf_series
