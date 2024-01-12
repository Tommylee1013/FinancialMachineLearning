import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def classification_stats(actual, predicted, prefix, get_specificity):
    # Create Report
    report = classification_report(actual, predicted, output_dict=True,
                                   labels=[0, 1], zero_division=0)
    # Extract (long only) metrics
    report['1'][prefix + '_accuracy'] = report['accuracy']
    report['1'][prefix + '_auc'] = roc_auc_score(actual, predicted)
    report['1'][prefix + '_macro_avg_f1'] = report['macro avg']['f1-score']
    report['1'][prefix + '_weighted_avg'] = report['weighted avg']['f1-score']

    # To DataFrame
    row = pd.DataFrame.from_dict(report['1'], orient='index').T
    row.columns = [prefix + '_precision', prefix + '_recall', prefix + '_f1_score',
                   prefix + '_support', prefix + '_accuracy', prefix + '_auc',
                   prefix + '_macro_avg_f1', prefix + '_weighted_avg_f1']

    # Add Specificity
    if get_specificity:
        row[prefix + '_specificity'] = report['0']['recall']
    else:
        row[prefix + '_specificity'] = 0

    return row

def strat_metrics(rets):
    avg = rets.mean()
    stdev = rets.std()
    sharpe_ratio = avg / stdev * np.sqrt(252)
    return avg, stdev, sharpe_ratio

def add_strat_metrics(row, rets, prefix):
    # Get metrics
    avg, stdev, sharpe_ratio = strat_metrics(rets)
    # Save to row
    row[prefix + '_mean'] = avg
    row[prefix + '_stdev'] = stdev
    row[prefix + '_sr'] = sharpe_ratio

def mean_abs_error(y_true, y_predict):
    return np.abs(np.array(y_true) - np.array(y_predict)).mean()