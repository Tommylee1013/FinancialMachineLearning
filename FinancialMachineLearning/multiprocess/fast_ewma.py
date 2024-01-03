import numpy as np
from numba import jit
from numba import float64
from numba import int64

@jit((float64[:], int64), nopython = True, nogil = True)
def ewma(arr_in, window):
    arr_length = arr_in.shape[0]
    ewma_arr = np.empty(arr_length, dtype=float64)
    alpha = 2 / (window + 1)
    weight = 1
    ewma_old = arr_in[0]
    ewma_arr[0] = ewma_old
    for i in range(1, arr_length):
        weight += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight

    return ewma_arr
