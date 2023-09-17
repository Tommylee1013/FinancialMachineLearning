from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class AbstractModelFingerprint(ABC):
    def __init__(self):
        self.linear_effect = None
        self.non_linear_effect = None
        self.pair_wise_effect = None

        self.ind_partial_dep_functions = None
        self.feature_column_position_mapping = None
        self.feature_values = None

    def fit(self, model: object, X: pd.DataFrame, num_values: int = 50, pairwise_combinations: list = None) -> None:
        self._get_feature_values(X, num_values)
        self._get_individual_partial_dependence(model, X)

        linear_effect = self._get_linear_effect(X)
        non_linear_effect = self._get_non_linear_effect(X)

        if pairwise_combinations is not None:
            pairwise_effect = self._get_pairwise_effect(pairwise_combinations, model, X, num_values)
            self.pair_wise_effect = {'raw': pairwise_effect, 'norm': self._normalize(pairwise_effect)}
        self.linear_effect = {'raw': linear_effect, 'norm': self._normalize(linear_effect)}
        self.non_linear_effect = {'raw': non_linear_effect, 'norm': self._normalize(non_linear_effect)}

    def get_effects(self) -> Tuple:
        return self.linear_effect, self.non_linear_effect, self.pair_wise_effect

    def plot_effects(self) -> plt.figure:
        if self.pair_wise_effect is None:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax3.set_title('Pair-wise effect')
            ax3.bar(*zip(*self.pair_wise_effect['norm'].items()))

        ax1.set_title('Linear effect')
        ax1.bar(*zip(*self.linear_effect['norm'].items()))

        ax2.set_title('Non-Linear effect')
        ax2.bar(*zip(*self.non_linear_effect['norm'].items()))

        fig.tight_layout()
        return fig

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> None:
        self.feature_values = {}
        for feature in X.columns:
            values = []
            for q in np.linspace(0, 1, num_values):
                values.append(np.quantile(X[feature], q=q))
            self.feature_values[feature] = np.array(values)

        self.ind_partial_dep_functions = pd.DataFrame(index=list(range(num_values)), columns=X.columns)

        self.feature_column_position_mapping = dict(zip(X.columns, range(0, X.shape[1])))

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> None:
        for col in X.columns:
            y_mean_arr = []
            for x_k in self.feature_values[col]:
                col_k_position = self.feature_column_position_mapping[col]
                X_ = X.values.copy()
                X_[:, col_k_position] = x_k
                y_pred = self._get_model_predictions(model, X_)
                y_pred_mean = np.mean(y_pred)

                y_mean_arr.append(y_pred_mean)

            self.ind_partial_dep_functions[col] = y_mean_arr

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        store = {}
        for col in X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            y_mean = np.mean(y)
            linear_effect = np.mean(np.abs(lmodel.predict(x) - y_mean))
            store[col] = linear_effect
        return store

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        store = {}
        for col in X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            nonlinear_effect = np.mean(np.abs(lmodel.predict(x) - y.values))
            store[col] = nonlinear_effect
        return store

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values) -> dict:
        store = {}

        for pair in pairwise_combinations:
            function_values = []
            col_k = pair[0]
            col_l = pair[1]

            y_cdf_k_centered = self.ind_partial_dep_functions[col_k] - np.mean(self.ind_partial_dep_functions[col_k])
            y_cdf_l_centered = self.ind_partial_dep_functions[col_l] - np.mean(self.ind_partial_dep_functions[col_l])

            for x_k, y_cdf_k in zip(self.feature_values[col_k], y_cdf_k_centered):
                for x_l, y_cdf_l in zip(self.feature_values[col_l], y_cdf_l_centered):
                    col_k_position = self.feature_column_position_mapping[col_k]
                    col_l_position = self.feature_column_position_mapping[col_l]
                    X_ = X.values.copy()
                    X_[:, col_k_position] = x_k
                    X_[:, col_l_position] = x_l

                    y_cdf_k_l = self._get_model_predictions(model, X_).mean()

                    function_values.append([y_cdf_k_l, y_cdf_k, y_cdf_l])

            function_values = np.array(function_values)
            centered_y_cdf_k_l = function_values[:, 0] - np.mean(function_values[:, 0])

            f_k = function_values[:, 1]
            f_l = function_values[:, 2]
            func_value = sum(abs((centered_y_cdf_k_l - f_k - f_l)))

            store[str(pair)] = func_value / (num_values ** 2)

        return store

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        raise NotImplementedError('Must implement _get_model_predictions')

    @staticmethod
    def _normalize(effect: dict) -> dict:
        values_sum = sum(effect.values())
        updated_effect = {}

        for k, v in effect.items():
            updated_effect[k] = v / values_sum
        return updated_effect
class RegressionModelFingerprint(AbstractModelFingerprint):
    def __init__(self):
        AbstractModelFingerprint.__init__(self)

    def _get_model_predictions(self, model, X_):
        return model.predict(X_)
class ClassificationModelFingerprint(AbstractModelFingerprint):
    def __init__(self):
        AbstractModelFingerprint.__init__(self)
    def _get_model_predictions(self, model, X_):
        return model.predict_proba(X_)[:, 1]