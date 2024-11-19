from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    """Load MNIST dataset and preprocess it."""
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Relabel target 3 as an anomaly (1) and all others as normal (0)
    y_relabel = np.where(y == 3, 1, 0)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_relabel, train_size=0.8, stratify=y_relabel, random_state=1234
    )

    return X_train, X_test, y_train, y_test
