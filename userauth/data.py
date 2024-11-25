import os
from PIL import Image
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


def load_data_SOCOfing():
    """Load SOCOFing dataset."""
    X_train, y_train = socofing_data_loader_util(dataset_paths=[
        "./SOCOFing/Real",
        "./SOCOFing/Altered/Altered-Hard",
        "./SOCOFing/Altered/Altered-Medium",
    ])

    X_test, y_test = socofing_data_loader_util(
        dataset_paths=["./SOCOFing/Altered/Altered-Easy"])

    return X_train, X_test, y_train, y_test


def socofing_data_loader_util(dataset_paths=["./SOCOFing/Real"],
                              anomaly_finger='Left_index_finger'):
    """
    Load and preprocess the SOCOFing dataset.
    Parameters:
        dataset_path (str): Path to the SOCOFing dataset folder (Real images).

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed and split dataset.
    """
    X = []
    y = []

    for dataset_path in dataset_paths:
        image_files = os.listdir(dataset_path)

        for file in image_files:
            img_path = os.path.join(dataset_path, file)
            with Image.open(img_path) as img:
                img = img.convert('L')
                img = img.resize((128, 128))
            img = np.array(img) / 255.0
            img = img.flatten()

            _, finger_type = file.split('__')

            if anomaly_finger in finger_type:
                label = 1
            else:
                label = 0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)
