import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from FinancialMachineLearning.filter.denoising import *

def clusterKMeansBase(corr0, maxNumClusters = 10, n_init = 10):
    x, silh = ((1 - corr0.fillna(0))/2)**0.5, pd.Series()
    for init in range(n_init) :
        for i in range(2, maxNumClusters+1) :
            kmeans_ = KMeans(n_clusters = i, n_init = 1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
            if np.isnan(stat[1]) or stat[0] > stat[1] : silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]
    corr1 = corr1.iloc[:, newIdx]
    clusters = {i : corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index = x.index)
    return corr1, clusters, silh

def nested_clustered_optimization(cov : np.array, mu = None, maxNumClusters = None) :
    cov = pd.DataFrame(cov)
    if mu is not None : mu = pd.Series(mu[:, 0])
    corr1 = covariance_to_correlation(cov)
    corr1, clusters, _ = clusterKMeansBase(corr1, maxNumClusters, n_init = 10)
    wIntra = pd.DataFrame(0, index = cov.index, columns = clusters.keys())
    for i in clusters :
        cov_ = cov.loc[clusters[i], clusters[i]].values
        if mu is None : mu_ = None
        else : mu_ = mu.loc[clusters[i]].values.reshape(-1, 1)
        wIntra.loc[clusters[i], i] = optimizing_portfolio(cov_, mu_).flatten()
    cov_ = wIntra.T.dot(np.dot(cov, wIntra))
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = pd.Series(optimizing_portfolio(cov_, mu_).flatten(), index = cov_.index)
    nco = wIntra.mul(wInter, axis = 1).sum(axis = 1).values.reshape(-1, 1)
    return nco