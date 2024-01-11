import numpy as np
import pandas as pd

def binHR(stop_loss, profit_taking, freq, t_sharpe_ratio):
    a = (freq + t_sharpe_ratio ** 2) * (profit_taking - stop_loss) ** 2
    b = (2 * freq * stop_loss - t_sharpe_ratio ** 2 * (profit_taking - stop_loss)) * (profit_taking - stop_loss)
    c = freq * stop_loss ** 2
    p = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2.0 * a)
    return p

def get_grid_precision(
        stop_loss: np.array,
        profit_taking: float,
        observation: np.array,
        expected_sharpe: float
) -> pd.DataFrame():
    values = np.ones(len(stop_loss) * len(observation)).reshape(len(stop_loss), len(observation))
    for i in range(len(stop_loss)):
        for j in range(len(observation)):
            values[i][j] = binHR(stop_loss[i], profit_taking, observation[j], expected_sharpe)
    values = pd.DataFrame(
        values, columns=observation, index=stop_loss
    )
    return values
