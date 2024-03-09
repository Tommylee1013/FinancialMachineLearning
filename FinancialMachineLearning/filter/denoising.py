import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from FinancialMachineLearning.utils.stats import *

class Denoise:
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

def denoise_constant_residual_eigenvalue(eVal, eVec, nFacts):
    """
    denoising to fix random eigen value from correlation matrix
    """
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = covariance_to_correlation(corr1)
    return corr1

def denoise_target_shrinkage(eVal, eVec, nFacts, alpha = 0) :
    """
    denoising to use target shrinkage method from correlation matrix
    """
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2

def signal_detoning(corr, eigenvalues, eigenvectors, market_component = 1):
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]

    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)
    corr = corr - corr_mark
    corr = covariance_to_correlation(corr)
    return corr