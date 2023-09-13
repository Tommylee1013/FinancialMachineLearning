import pandas as pd
import numpy as np
from FinancialMachineLearning.multiprocess.multiprocess import mp_pandas_obj
def get_values_diff(test_type, series, index, ind):
    if test_type == 'one_sided':
        values_diff = series.loc[index] - series.loc[ind]
    elif test_type == 'two_sided':
        values_diff = abs(series.loc[index] - series.loc[ind])
    else:
        raise ValueError('Test type is unknown: can be either one_sided or two_sided')

    return values_diff
def get_s_n_for_t(series: pd.Series, test_type: str, molecule: list) -> pd.DataFrame:
    s_n_t_series = pd.DataFrame(index=molecule, columns=['stat', 'critical_value'])
    for index in molecule:

        series_t = series.loc[:index]
        squared_diff = series_t.diff().dropna() ** 2
        integer_index = series_t.index.get_loc(index)
        sigma_sq_t = 1 / (integer_index - 1) * sum(squared_diff)

        max_s_n_value = -np.inf
        max_s_n_critical_value = None
        for ind in series_t.index[:-1]:
            values_diff = get_values_diff(test_type, series, index, ind)
            temp_integer_index = series_t.index.get_loc(ind)
            s_n_t = 1 / (sigma_sq_t * np.sqrt(integer_index - temp_integer_index)) * values_diff
            if s_n_t > max_s_n_value:
                max_s_n_value = s_n_t
                max_s_n_critical_value = np.sqrt(
                    4.6 + np.log(integer_index - temp_integer_index))
        s_n_t_series.loc[index, ['stat', 'critical_value']] = max_s_n_value, max_s_n_critical_value
    return s_n_t_series


def get_chu_stinchcombe_white_statistics(series: pd.Series, test_type: str = 'one_sided',
                                         num_threads: int = 8) -> pd.Series:
    molecule = series.index[2:series.shape[0]]
    s_n_t_series = mp_pandas_obj(func=get_s_n_for_t,
                                 pd_obj=('molecule', molecule),
                                 series=series,
                                 test_type=test_type,
                                 num_threads=num_threads,
                                 )
    return s_n_t_series
