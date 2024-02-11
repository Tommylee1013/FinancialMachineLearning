import pandas as pd
import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

class DistanceDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _corr_dist(x, y):
        return np.sqrt(2 * len(x) * (1 - np.corrcoef(x, y)[0, 1]))

    @staticmethod
    def _js_divergence(x, y):
        kde1, kde2 = gaussian_kde(x), gaussian_kde(y)
        estimate = np.linspace(
            min(np.min(x), np.min(y)),
            max(np.max(x), np.max(y)),
            len(x)
        )
        pdf1, pdf2 = kde1(estimate), kde2(estimate)
        m = 0.5 * (pdf1 + pdf2)
        return 0.5 * np.sum(rel_entr(pdf1, m)) + 0.5 * np.sum(rel_entr(pdf2, m))

    @staticmethod
    def num_bins(nObs, corr=None):
        if corr is None:
            z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2) ** 0.5) ** (1 / 3)
            b = round(z / 6 + 2 / (3 * z) + 1 / 3)
        else:
            b = round(2 ** 0.5 * (1 + (1 + 24 * nObs * (1 - corr ** 2)) ** 0.5) ** 0.5)
        return int(b)

    @staticmethod
    def _var_info(x, y, norm=False):
        b_xy = DistanceDataFrame.num_bins(len(x), np.corrcoef(x, y)[0, 1])
        c_xy = np.histogram2d(x, y, bins=b_xy)[0]
        i_xy = mutual_info_score(None, None, contingency=c_xy)
        hx = ss.entropy(np.histogram(x, bins=b_xy)[0])
        hy = ss.entropy(np.histogram(y, bins=b_xy)[0])
        v_xy = hx + hy - (2 * i_xy)
        if norm:
            h_xy = hx + hy - i_xy
            v_xy /= h_xy
        return v_xy

    @staticmethod
    def _mutual_info(x, y, norm=False):
        b_xy = DistanceDataFrame.num_bins(len(x), np.corrcoef(x, y)[0, 1])
        c_xy = np.histogram2d(x, y, bins=b_xy)[0]
        i_xy = mutual_info_score(None, None, contingency=c_xy)
        if norm:
            hx = ss.entropy(np.histogram(x, bins=b_xy)[0])
            hy = ss.entropy(np.histogram(y, bins=b_xy)[0])
            i_xy /= (hx + hy)
        return i_xy

    def corr_based_distance(self):
        n = len(self.columns)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = DistanceDataFrame._corr_dist(self.iloc[:, i], self.iloc[:, j])
        dist = pd.DataFrame(dist, columns=self.columns, index=self.columns)
        return dist

    def jensen_shannon_divergence(self):
        n = len(self.columns)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = DistanceDataFrame._js_divergence(self.iloc[:, i], self.iloc[:, j])
        dist = pd.DataFrame(dist, columns=self.columns, index=self.columns)
        return dist

    def variational_information(self, norm=False):
        n = len(self.columns)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = DistanceDataFrame._var_info(self.iloc[:, i], self.iloc[:, j], norm)
        dist = pd.DataFrame(dist, columns=self.columns, index=self.columns)
        return dist

    def mutual_information(self, norm=False):
        n = len(self.columns)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = DistanceDataFrame._mutual_info(self.iloc[:, i], self.iloc[:, j], norm)
        dist = pd.DataFrame(dist, columns=self.columns, index=self.columns)
        return dist

def _corr_dist(x, y) :
    return np.sqrt(2 * len(x) * (1 - np.corrcoef(x, y)[0,1]))

def corr_based_distance(ret) :
    n = len(ret.columns)
    dist = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            dist[i,j] = _corr_dist(ret.iloc[:,i], ret.iloc[:,j])
    dist = pd.DataFrame(dist, columns = ret.columns, index = ret.columns)
    return dist

def _js_divergence(x, y) :
    kde1, kde2 = gaussian_kde(x), gaussian_kde(y)
    estimate = np.linspace(
        min(np.min(x), np.min(y)),
        max(np.max(x), np.max(y)),
        len(x)
    )
    pdf1, pdf2 = kde1(estimate), kde2(estimate)
    m = 0.5 * (pdf1 + pdf2)
    return 0.5 * np.sum(rel_entr(pdf1, m)) + 0.5 * np.sum(rel_entr(pdf2, m))

def jensen_shannon_divergence(ret) :
    n = len(ret.columns)
    dist = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            dist[i,j] = _js_divergence(ret.iloc[:,i], ret.iloc[:,j])
    dist = pd.DataFrame(dist, columns = ret.columns, index = ret.columns)
    return dist

def num_bins(nObs, corr = None) :
    if corr is None :
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2) ** 0.5) ** (1/3)
        b = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else :
        b = round(2 ** 0.5 * (1 + (1 + 24 * nObs * (1 - corr ** 2)) ** 0.5) ** 0.5)
    return int(b)

def _var_info(x, y, norm = False) :
    b_xy = num_bins(len(x), np.corrcoef(x, y)[0,1])
    c_xy = np.histogram2d(x, y, bins = b_xy)[0]
    i_xy = mutual_info_score(None, None, contingency = c_xy)
    hx = ss.entropy(np.histogram(x, bins = b_xy)[0])
    hy = ss.entropy(np.histogram(y, bins = b_xy)[0])
    v_xy = hx + hy - (2 * i_xy)
    if norm :
        h_xy = hx + hy - i_xy
        v_xy /= h_xy
    return v_xy

def variational_information(ret, norm = False) :
    n = len(ret.columns)
    dist = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            dist[i,j] = _var_info(ret.iloc[:,i], ret.iloc[:,j], norm)
    dist = pd.DataFrame(dist, columns = ret.columns, index = ret.columns)
    return dist

def _mutual_info(x, y, norm = False) :
    b_xy = num_bins(len(x), np.corrcoef(x, y)[0,1])
    c_xy = np.histogram2d(x, y, bins = b_xy)[0]
    i_xy = mutual_info_score(None, None, contingency = c_xy)
    if norm :
        hx = ss.entropy(np.histogram(x, bins = b_xy)[0])
        hy = ss.entropy(np.histogram(y, bins = b_xy)[0])
        i_xy /= (hx + hy)
    return i_xy

def mutual_information(ret, norm = False) :
    n = len(ret.columns)
    dist = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            dist[i,j] = _mutual_info(ret.iloc[:,i], ret.iloc[:,j], norm)
    dist = pd.DataFrame(dist, columns = ret.columns, index = ret.columns)
    return dist