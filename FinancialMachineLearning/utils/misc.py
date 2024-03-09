import pandas as pd
import numpy as np

def crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int):
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object
