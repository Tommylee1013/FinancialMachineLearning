import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from FinancialMachineLearning.cross_validation.cross_validation import ml_cross_val_score

def mean_decrease_impurity(model, feature_names):
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names
    feature_imp_df = feature_imp_df.replace(0, np.nan)

    importance = pd.concat({'mean': feature_imp_df.mean(),
                            'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5},
                           axis=1)
    importance /= importance['mean'].sum()
    return importance

def mean_decrease_accuracy(model, X, y, cv_gen, sample_weight=None, scoring=log_loss):
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    fold_metrics_values, features_metrics_values = pd.Series(dtype = float), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight = sample_weight[train])
        pred = fit.predict(X.iloc[test, :])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring(y.iloc[test], prob, sample_weight = sample_weight[test],
                                                  labels = model.classes_)
        else:
            fold_metrics_values.loc[i] = scoring(y.iloc[test], pred, sample_weight = sample_weight[test])
        for j in X.columns:
            X1_ = X.iloc[test, :].copy(deep=True)
            np.random.shuffle(X1_[j].values)
            if scoring == log_loss:
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                             labels=model.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = scoring(y.iloc[test], pred,
                                                            sample_weight=sample_weight[test])

    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == log_loss:
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values)
    importance = pd.concat({'mean': importance.mean(), 'std': importance.std() * importance.shape[0] ** -.5}, axis=1)
    importance.replace([-np.inf, np.nan], 0, inplace=True)

    return importance

def single_feature_importance(clf, X, y, cv_gen, sample_weight = None, scoring = log_loss):
    feature_names = X.columns
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        feat_cross_val_scores = ml_cross_val_score(clf, X=X[[feat]], y=y, sample_weight=sample_weight,
                                                   scoring=scoring, cv_gen=cv_gen)
        imp.loc[feat, 'mean'] = feat_cross_val_scores.mean()
        imp.loc[feat, 'std'] = feat_cross_val_scores.std() * feat_cross_val_scores.shape[0] ** -.5
    return imp

def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
    plt.figure(figsize=(10, importance_df.shape[0] / 5))
    importance_df.sort_values('mean', ascending=True, inplace=True)
    importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
    plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))

    if save_fig is True:
        plt.savefig(output_path)
    else:
        plt.show()