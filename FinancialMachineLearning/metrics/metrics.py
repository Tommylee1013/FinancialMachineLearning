import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import norm, percentileofscore
import scipy.stats as ss

def getExpectedMaxSR(nTrials, meanSR, stdSR):
    # Expected max SR, controlling for SBuMT
    emc = 0.477215664901532860606512090082402431042159336  # Euler-Mascheronis Constant
    sr0 = (1 - emc) * norm.ppf(1 - 1. / nTrials) + emc * norm.ppf(1 - (nTrials * np.e) ** -1)
    sr0 = meanSR + stdSR * sr0
    return sr0

def getDistMaxSR(nSims, nTrials, stdSR, meanSR):
    # Monte carlo of max{SR} on nTrials, from nSims simulations
    rng = np.random.RandomState()
    out = pd.DataFrame()
    for nTrials_ in nTrials:
        # 1) simulated sharpe ratios
        sr = pd.DataFrame(
            rng.randn(nSims, nTrials_))  # Return a sample (or samples) from the “standard normal” distribution.
        sr = sr.sub(sr.mean(axis=1), axis=0)  # center
        sr = sr.div(sr.std(axis=1), axis=0)  # scale
        sr = meanSR + sr * stdSR
        # 2) store output
        out_ = sr.max(axis=1).to_frame('max{SR}')
        out_['nTrials'] = nTrials_
        out = pd.concat([out, out_], axis=0)
    return out

# code snippet 8.2 - mean and standard deviation of the prediction errors
def getMeanStdError(nSims0, nSims1, nTrials, stdSR=1, meanSR=0):
    # compute standard deviation of errors per nTrials
    # nTrials: [number of SR used to derive max{SR}]
    # nSims0: number of max{SR} u{sed to estimate E[max{SR}]
    # nSims1: number of errors on which std is computed
    sr0 = pd.Series({i: getExpectedMaxSR(i, meanSR, stdSR) for i in nTrials})
    sr0 = sr0.to_frame('E[max{SR}]')
    sr0.index.name = 'nTrials'
    err = pd.DataFrame()
    for i in range(0, int(nSims1)):
        # sr1 = getDistDSR(nSims=1000, nTrials=nTrials, meanSR=0, stdSR=1)
        sr1 = getDistMaxSR(nSims=100, nTrials=nTrials, meanSR=0, stdSR=1)
        sr1 = sr1.groupby('nTrials').mean()
        err_ = sr0.join(sr1).reset_index()
        err_['err'] = err_['max{SR}'] / err_['E[max{SR}]'] - 1.
        err = pd.concat([err, err_])
    out = {'meanErr': err.groupby('nTrials')['err'].mean()}
    out['stdErr'] = err.groupby('nTrials')['err'].std()
    out = pd.DataFrame.from_dict(out, orient='columns')
    return out

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

def getZStat(sr, t, sr_=0, skew=0, kurt=3):
    z = (sr - sr_) * (t - 1) ** .5
    z /= (1 - skew * sr + (kurt - 1) / 4. * sr ** 2) ** .5
    return z

def type1Err(z, k=1):
    # false positive rate
    alpha = ss.norm.cdf(-z)
    alpha_k = 1 - (1 - alpha) ** k  # multi-testing correction
    return alpha_k

# code snippet 8.4 - Type II error (false negative) - with numerical example
def getTheta(sr, t, sr_=0., skew=0., kurt=3):
    theta = sr_ * (t - 1) ** .5
    theta /= (1 - skew * sr + (kurt - 1) / .4 * sr ** 2) ** .5
    return theta

def type2Err(alpha_k, k, theta):
    # false negative rate
    z = ss.norm.ppf((1 - alpha_k) ** (1. / k))  # Sidak's correction
    beta = ss.norm.cdf(z - theta)
    return beta
