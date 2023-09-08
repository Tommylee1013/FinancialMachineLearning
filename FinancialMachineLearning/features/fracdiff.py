import numpy as np
import pandas as pd

class FractionalDifferentiatedFeatures :
    @staticmethod
    def getWeights(diff_amt, size):
        weights = [1]
        for k in range(1, size) :
            weights_ = -weights[-1] * (diff_amt - k + 1) / k
            weights.append(weights_)
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights
    @staticmethod
    def fracDiff(price : pd.Series, diff_amt, threshold = 0.01):
        weights = FractionalDifferentiatedFeatures.getWeights(diff_amt, price.shape[0])
        weights_ = np.cumsum(abs(weights))
        weights_ /= weights_[-1]
        skip = weights_[weights_ > threshold].shape[0]
        output_df = {}
        for name in price.columns:
            series_f = price[[name]].fillna(method = 'ffill').dropna()
            output_df_ = pd.Series(index = price.index)

            for iloc in range(skip, series_f.shape[0]):
                loc = series_f.index[iloc]
                output_df_[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]

            output_df[name] = output_df_.copy(deep=True)
        output_df = pd.concat(output_df, axis=1)
        return output_df
    @staticmethod
    def getWeightsFFD(diff_amt, threshold, lim):
        weights = [1.]
        k = 1
        ctr = 0
        while True:
            weights_ = -weights[-1] * (diff_amt - k + 1) / k
            if abs(weights_) < threshold:
                break
            weights.append(weights_)
            k += 1
            ctr += 1
            if ctr == lim - 1:
                break
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights
    @staticmethod
    def fracDiffFFD(series, diff_amt, threshold = 1e-5):
        weights = FractionalDifferentiatedFeatures.getWeightsFFD(diff_amt, threshold, series.shape[0])
        width = len(weights) - 1
        output_df = {}
        for name in series.columns:
            series_f = series[[name]].fillna(method='ffill').dropna()
            temp_df_ = pd.Series(index=series.index)
            for iloc1 in range(width, series_f.shape[0]):
                loc0 = series_f.index[iloc1 - width]
                loc1 = series.index[iloc1]
                temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]
            output_df[name] = temp_df_.copy(deep=True)
        output_df = pd.concat(output_df, axis=1)
        return output_df