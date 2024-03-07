import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.utils import check_random_state


def get_covariance_sub(
        nObs, nCols, sigma, random_state=None
):
    rng = check_random_state(random_state)
    if nCols == 1:
        return np.ones((1, 1))

    ar0 = rng.normal(size=(nObs, 1))
    ar0 = np.repeat(ar0, nCols, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)
    return ar0


def get_random_block_covariance(
        nCols, nBlocks, minBlockSize=1, sigma=1.0, random_state=None
):
    rng = check_random_state(random_state)
    parts = rng.choice(
        range(1, nCols - (minBlockSize - 1) * nBlocks),
        nBlocks - 1,
        replace=False
    )
    parts.sort()
    parts = np.append(parts, nCols - (minBlockSize - 1) * nBlocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + minBlockSize
    cov = None

    for nCols_ in parts:
        cov_ = get_covariance_sub(
            int(max(nCols_ * (nCols_ + 1) / 2.0, 100)),
            nCols_,
            sigma,
            random_state=rng
        )
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)
    return cov


def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def get_random_block_correlation(
        nCols, nBlocks, random_state=None, minBlockSize=1
):
    rng = check_random_state(random_state)

    cov0 = get_random_block_covariance(
        nCols, nBlocks,
        minBlockSize=minBlockSize,
        sigma=0.5,
        random_state=rng
    )
    cov1 = get_random_block_covariance(
        nCols, 1,
        minBlockSize=minBlockSize,
        sigma=1.0,
        random_state=rng
    )

    cov0 += cov1
    corr0 = covariance_to_correlation(cov0)
    corr0 = pd.DataFrame(corr0)
    return corr0