import numpy as np


def custom_roc_auc(y_true, y_scores):
    """
    Compute ROC and AUC.

    Parameters:
    - y_true: Truth binary labels (0 or 1).
    - y_scores: Predicted probabilities for the positive class.

    Returns:
    - fpr: false positive rates
    - tpr: true positive rates
    - auc: area under the roc curve.
    - tp: true positives
    - fp: false positives
    - p: positives
    - n: negatives
    """
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    p = np.sum(y_true)
    n = len(y_true) - p

    tpr = tp / p
    fpr = fp / n

    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))

    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc, tp, fp, p, n
