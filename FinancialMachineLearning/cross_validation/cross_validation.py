from typing import Callable
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

from scipy.stats import rv_continuous, kstest

def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index
        df1 = train[(start_ix <= train) & (train <= end_ix)].index
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups = None):
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(
                index=[self.samples_info_sets.iloc[start_ix]],
                data=[self.samples_info_sets.iloc[end_ix-1]]
            )
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices

class SampledPipeline(Pipeline) :
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None :
            fit_params[self.steps[-1][0] + ' sample_weight'] = sample_weight
        return super(SampledPipeline, self).fit(X, y, **fit_params)

def ml_cross_val_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        cv_gen: BaseCrossValidator,
        sample_weight: np.ndarray = None,
        scoring: Callable[[np.array, np.array], float] = log_loss):
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1 * scoring(y.iloc[test], prob, sample_weight=sample_weight[test], labels=classifier.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
        ret_scores.append(score)
    return np.array(ret_scores)

def grid_search_cross_validation(
        feat,
        label : pd.Series,
        samples_info_sets : pd.Series,
        pipe_clf,
        param_grid,
        cv : int = 3,
        bagging : list = [0, None, 1],
        random_search_iterator : int = 0,
        n_jobs : int = -1,
        pct_embargo : float = 0.0,
        **fit_params) :
    if set(label.values) == {0,1} : scoring = 'f1'
    else : scoring = 'neg_log_loss'

    inner_cv = PurgedKFold(n_splits = cv, samples_info_sets = samples_info_sets, pct_embargo = pct_embargo)
    if random_search_iterator == 0:
        grid_search = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                                    scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    else :
        grid_search = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid,
                                         scoring = scoring, cv = inner_cv, n_jobs = n_jobs,
                                         iid = False, n_iter = random_search_iterator)
    grid_search = grid_search.fit(feat, label, **fit_params).best_extimator_

    if bagging[1] > 0 :
        grid_search = BaggingClassifier(estimator = SampledPipeline(grid_search.steps),
                                        n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                        max_features = float(bagging[2]), n_jobs = n_jobs)
        grid_search = grid_search.fit(feat, label,
                                      sample_weight = fit_params[grid_search.base_estimator.steps[-1][0]+' sample_weight'])
        grid_search = Pipeline([('bag', grid_search)])
    return grid_search

class logUniform_gen(rv_continuous) :
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)
def log_uniform(a = 1, b = np.exp(1)) :
    return logUniform_gen(a = a, b = b, name = 'logUniform')