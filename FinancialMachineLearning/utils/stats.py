import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

def covariance_to_correlation(cov) :
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr

def correlation_to_covariance(corr, std):
    cov = corr * np.outer(std, std)
    return cov

def marchenko_pastur_prob_distribution(var, q, pts) :
    eMin, eMax = var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index = eVal)
    return pdf

def fit_kde(obs, bWidth = 0.25, kernel = 'gaussian', x = None) :
    if len(obs.shape) == 1: obs = obs.reshape(-1, 1)
    kde = KernelDensity(
        kernel = kernel,
        bandwidth = bWidth
    ).fit(obs)

    if x is None: x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1: x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(
        np.exp(logProb),
        index = x.flatten()
    )
    return pdf

def pdf_error(var, eVal, q, bWidth, pts = 1000, verbose = False) :
    var = var[0]
    pdf0 = marchenko_pastur_prob_distribution(var, q, pts)  # theoretical pdf
    pdf1 = fit_kde(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    if verbose : print("sse:" + str(sse))
    return sse

def find_max_eval(eVal, q, bWidth, verbose=False):
    out = minimize(
        lambda *x: pdf_error(*x),
        x0=np.array(0.5),
        args=(eVal, q, bWidth),
        bounds=((1E-5, 1 - 1E-5),)
    )

    if verbose: print("found errPDFs" + str(out['x'][0]))

    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2

    return eMax, var

def optimizing_portfolio(cov, mu = None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape = (inv.shape[0],1))
    if mu is None : mu = ones
    w = np.dot(inv,mu)
    w /= np.dot(ones.T,w)
    return w