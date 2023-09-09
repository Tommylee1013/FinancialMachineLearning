import numpy as np
import pandas as pd

class BarFeature:
    def __init__(self, name, function):
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        if not callable(function):
            raise ValueError('function must be a callable')
        self.name = name
        self.function = function
    def compute(self, tick_df):
        return self.function(tick_df)

def madOutlier(y, thresh = 3.):
    median = np.median(y)
    print(median)
    diff = np.sum((y - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    print(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    print(modified_z_score)
    return modified_z_score > thresh