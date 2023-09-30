import statsmodels.api as sm
import pandas as pd
import numpy as np

def linear_trend_t_values(close : np.array) :
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm.OLS(close, x).fit()
    return ols.tvalues[1]

def trend_labeling(molecule, close : pd.Series, span : list) :
    out = pd.DataFrame(index = molecule, columns = ['t1','tVal','bin'])
    horizons = range(*span)
    for dt0 in molecule :
        df0 = pd.Series()
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(horizons) > close.shape[0] : continue
        for horizon in horizons :
            dt1 = close.index[iloc0 + horizon -1]
            df1 = close.loc[dt0 : dt1]
            df0.loc[dt1] = linear_trend_t_values(df1.values)
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ['t1','tVal','bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast = 'signed')
    return out.dropna(subset = ['bin'])