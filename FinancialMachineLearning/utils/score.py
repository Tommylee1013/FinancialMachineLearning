from sklearn.metrics import make_scorer
import numpy as np

def probability_weighted_accuracy(y_true, y_pred):
    """
    Compute the Probability Weighted Accuracy.

    Parameters:
    - y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
              True labels or binary label indicators.
    - y_pred: array-like of shape (n_samples,) or (n_samples, n_classes)
              Target scores, can either be probability estimates of the positive class,
              confidence values, or non-thresholded measure of decisions.

    Returns:
    - pwa: float
           Probability Weighted Accuracy.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Handling binary classification
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        K = 2
        y_pred = np.column_stack([1 - y_pred, y_pred])
    else:
        K = y_pred.shape[1]

    # Compute the Probability Weighted Accuracy
    pwa_numerator = np.sum(y_true * (y_pred - 1 / K))
    pwa_denominator = np.sum(y_pred - 1 / K)

    # Avoid division by zero
    if pwa_denominator == 0:
        return 0

    pwa = pwa_numerator / pwa_denominator
    return pwa

def profit_maximization_score(y_true, y_pred_proba, threshold=0.5, profit_per_trade=100, loss_per_trade=-100):
    y_pred = (y_pred_proba > threshold).astype(int)
    correct_trades = (y_pred == y_true).astype(int)
    total_profit = (correct_trades * profit_per_trade + (1 - correct_trades) * loss_per_trade).sum()
    return total_profit
