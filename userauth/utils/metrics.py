import numpy as np


def custom_roc_auc(y_true, y_scores):
    """
    Compute ROC and AUC.

    Parameters:
    - y_true: Truth binary labels (0 or 1).
    - y_scores: Predicted probabilities for the positive class.

    Returns:
    - FALSE_POSITIVES: False Positive Rates
    - TRUE_POSITIVE_RATE: True Positive Rates
    - AREA_UNDER_CURVE: Area Under the ROC curve.
    """
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    TRUE_POSITIVES = np.cumsum(y_true_sorted)
    FALSE_POSITIVES = np.cumsum(1 - y_true_sorted)
    POSITIVES = np.sum(y_true)
    NEGATIVES = len(y_true) - POSITIVES

    TRUE_POSITIVE_RATE = TRUE_POSITIVES / POSITIVES
    FALSE_POSITIVE_RATE = FALSE_POSITIVES / NEGATIVES

    TRUE_POSITIVE_RATE = np.concatenate(([0], TRUE_POSITIVE_RATE))
    FALSE_POSITIVE_RATE = np.concatenate(([0], FALSE_POSITIVE_RATE))

    AREA_UNDER_CURVE = np.trapezoid(TRUE_POSITIVE_RATE, FALSE_POSITIVE_RATE)
    return FALSE_POSITIVE_RATE, TRUE_POSITIVE_RATE, AREA_UNDER_CURVE
