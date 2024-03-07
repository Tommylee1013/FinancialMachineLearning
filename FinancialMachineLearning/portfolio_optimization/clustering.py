import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from FinancialMachineLearning.filter.denoising import *

def clusterKMeansBase(
        corr0, maxNumClusters=10, n_init=10, verbose=False
):
    corr0[corr0 > 1] = 1
    dist_matrix = ((1 - corr0.fillna(0)) / 2.) ** .5
    silh_coef_optimal = pd.Series(dtype='float64')  # observations matrixs
    kmeans, stat = None, None
    maxNumClusters = min(maxNumClusters, int(np.floor(dist_matrix.shape[0] / 2)))

    for init in range(0, n_init):
        for num_clusters in range(2, maxNumClusters + 1):
            # (maxNumClusters + 2 - num_clusters) # go in reverse order to view more sub-optimal solutions
            kmeans_ = KMeans(n_clusters=num_clusters,
                             n_init=10)  # , random_state=3425) #n_jobs=None #n_jobs=None - use all CPUs
            kmeans_ = kmeans_.fit(dist_matrix)
            silh_coef = silhouette_samples(dist_matrix, kmeans_.labels_)
            stat = (silh_coef.mean() / silh_coef.std(), silh_coef_optimal.mean() / silh_coef_optimal.std())

            # If this metric better than the previous set as the optimal number of clusters
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh_coef_optimal = silh_coef
                kmeans = kmeans_
                if verbose == True:
                    print(kmeans)
                    print(stat)
                    silhouette_avg = silhouette_score(dist_matrix, kmeans_.labels_)
                    print("For n_clusters =" + str(num_clusters) + "The average silhouette_score is :" + str(
                        silhouette_avg))
                    print("********")

    newIdx = np.argsort(kmeans.labels_)
    # print(newIdx)

    corr1 = corr0.iloc[newIdx]  # reorder rows
    corr1 = corr1.iloc[:, newIdx]  # reorder columns

    clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
              np.unique(kmeans.labels_)}  # cluster members
    silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)

    return corr1, clstrs, silh_coef_optimal


def makeNewOutputs(
        corr0, clusters, clusters2
):
    clstrsNew, newIdx = {}, []
    for i in clusters.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clusters[i])

    for i in clusters2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clusters2[i])

    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]

    dist = ((1 - corr0.fillna(0)) / 2.) ** .5
    kmeans_labels = np.zeros(len(dist.columns))
    for i in clstrsNew.keys():
        idxs = [dist.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i

    silhNew = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)

    return corrNew, clstrsNew, silhNew


def clusterKMeansTop(
        corr0, maxNumClusters=None, n_init=10
):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1] - 1

    corr1, clstrs, silh = clusterKMeansBase(
        corr0,
        maxNumClusters=min(maxNumClusters, corr0.shape[1] - 1),
        n_init=10
    )
    print("clstrs length:" + str(len(clstrs.keys())))
    print("best clustr:" + str(len(clstrs.keys())))

    # for i in clstrs.keys():
    #    print("std:"+str(np.std(silh[clstrs[i]])))

    clusterTstats = {
        i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()
    }
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]
    # print("redo cluster:"+str(redoClusters))

    if len(redoClusters) <= 2:
        print("If 2 or less clusters have a quality rating less than the average then stop.")
        print("redoCluster <=1:" + str(redoClusters) + " clstrs len:" + str(len(clstrs.keys())))
        return corr1, clstrs, silh

    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]

        _, clstrs2, _ = clusterKMeansTop(
            corrTmp,
            maxNumClusters=min(maxNumClusters, corrTmp.shape[1] - 1),
            n_init=n_init
        )

        print("clstrs2.len, stat:" + str(len(clstrs2.keys())))

        # Make new outputs, if necessary
        dict_redo_clstrs = {i: clstrs[i] for i in clstrs.keys() if i not in redoClusters}
        corrNew, clstrsNew, silhNew = makeNewOutputs(
            corr0,
            dict_redo_clstrs,
            clstrs2
        )
        newTstatMean = np.mean(
            [np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()]
        )

        if newTstatMean <= tStatMean:
            print("newTstatMean <= tStatMean" + str(newTstatMean) + " (len:newClst)" + str(
                len(clstrsNew.keys())) + " <= " + str(tStatMean) + " (len:Clst)" + str(len(clstrs.keys())))
            return corr1, clstrs, silh
        else:
            print("newTstatMean > tStatMean" + str(newTstatMean) + " (len:newClst)" + str(len(clstrsNew.keys()))
                  + " > " + str(tStatMean) + " (len:Clst)" + str(len(clstrs.keys())))
            return corrNew, clstrsNew, silhNew

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

def getPCA(matrix):
    eVal, eVec = np.linalg.eig(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec