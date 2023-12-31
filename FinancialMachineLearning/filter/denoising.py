import numpy as np, pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf

class DenoiseCorrelation:
    def __init__(self, eVal, eVec, nFacts):
        self.eVal = eVal
        self.eVec = eVec
        self.nFacts = nFacts

    def constant_residual_eigenvalue(self):
        eVal_ = np.diag(self.eVal).copy()
        eVal_[self.nFacts:] = eVal_[self.nFacts:].sum() / float(
            eVal_.shape[0] - self.nFacts)
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(self.eVec, eVal_).dot(self.eVec.T)
        corr1 = covariance_to_correlation(corr1)
        return corr1
    def target_shrink(self, alpha : int = 0):
        eValL, eVecL = self.eVal[:self.nFacts, :self.nFacts], self.eVec[:, :self.nFacts]
        eValR, eVecR = self.eVal[self.nFacts:, self.nFacts:], self.eVec[:, self.nFacts:]
        corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
        corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
        corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
        return corr2


def detoning(corr, eigenvalues, eigenvectors, market_component = 1):
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)
    corr = corr - corr_mark
    corr = covariance_to_correlation(corr)
    return corr

def optimizing_portfolio(cov, mu = None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape = (inv.shape[0],1))
    if mu is None : mu = ones
    w = np.dot(inv,mu)
    w /= np.dot(ones.T,w)
    return w

def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr
def correlation_to_covariance(corr, std):
    cov = corr * np.outer(std, std)
    return cov


class GenerateSamples:
    @staticmethod
    def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
        if len(obs.shape) == 1: obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
        if x is None: x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1: x = x.reshape(-1, 1)
        logProb = kde.score_samples(x)
        pdf = pd.Series(np.exp(logProb), index=x.flatten())
        return pdf

    @staticmethod
    def errPDFs(var, eVal, q, bWidth, pts=1000):
        var = var[0]
        pdf0 = GenerateSamples.mpPDF(var, q, pts)  # theoretical pdf
        pdf1 = GenerateSamples.fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        # print("sse:" + str(sse))
        return sse

    @staticmethod
    def getRndCov(nCols, nFacts):
        w = np.random.normal(size=(nCols, nFacts))
        cov = np.dot(w, w.T)
        cov += np.diag(np.random.uniform(size=nCols))
        return cov

    @staticmethod
    def findMaxEval(eVal, q, bWidth):
        out = minimize(lambda *x: GenerateSamples.errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth), bounds=((1E-5, 1 - 1E-5),))
        # print("found errPDFs" + str(out['x'][0]))
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eMax = var * (1 + (1. / q) ** .5) ** 2
        return eMax, var

    @staticmethod
    def formBlockMatrix(nBlocks,bSize,bCorr):
        block=np.ones((bSize,bSize))*bCorr
        block[range(bSize),range(bSize)]=1
        corr = block_diag(*([block]*nBlocks))
        return corr

    @staticmethod
    def formTrueMatrix(nBlocks,bSize,bCorr):
        corr0 = GenerateSamples.formBlockMatrix(nBlocks,bSize,bCorr)
        corr0=pd.DataFrame(corr0)
        cols=corr0.columns.tolist()
        np.random.shuffle(cols)
        corr0=corr0[cols].loc[cols].copy(deep=True)
        std0=np.random.uniform(.05,.2,corr0.shape[0])
        cov0=correlation_to_covariance(corr0,std0)
        mu0=np.random.normal(std0,std0,cov0.shape[0]).reshape(-1,1)
        return mu0,cov0

    @staticmethod
    def simCovMu(mu0,cov0,nObs,shrink = False):
        x=np.random.multivariate_normal(mu0.flatten(),cov0,size=nObs)
        mu1 = x.mean(axis=0).reshape(-1,1)
        if shrink: cov1=LedoitWolf().fit(x).covariance_
        else: cov1=np.cov(x,rowvar=0)
        return mu1,cov1

    @staticmethod
    def mpPDF(var, q, pts):
        eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eVal = np.linspace(eMin, eMax, pts)
        pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
        pdf = pd.Series(pdf, index = eVal)
        return pdf
    @staticmethod
    def denoise_covariance(cov0, q, bWidth):
        corr0 = covariance_to_correlation(cov0)
        eVal0, eVec0 = GenerateSamples.getPCA(corr0)
        eMax0, var0 = GenerateSamples.findMaxEval(np.diag(eVal0), q, bWidth)
        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
        temp = DenoiseCorrelation(eVal0, eVec0, nFacts0)
        corr1 = temp.constant_residual_eigenvalue()
        cov1 = correlation_to_covariance(corr1, np.diag(cov0) ** .5)
        return cov1