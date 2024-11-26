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
    X, y = socofing_data_loader_util(
        anomaly_fingers=None,
        dataset_paths=[
            "./SOCOFing/Real/",
            "./SOCOFing/Altered/Altered-Easy/",
            "./SOCOFing/Altered/Altered-Medium/",
            "./SOCOFing/Altered/Altered-Hard/",
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=1234
    )

    return X_train, X_test, y_train, y_test


def socofing_data_loader_util(
        dataset_paths,
        anomaly_fingers=[
            "Left_index_finger",
            "Left_little_finger",
            "Left_middle_finger",
            "Left_ring_finger",
            "Left_thumb_finger",
        ],
        anomaly_userids=[
            str(number) for number in range(20)
        ],
        dimensionality_reduction=True,
        components=100):
    """
    Load and preprocess the SOCOFing dataset.
    Returns:
        X, y: Preprocessed dataset.
    """
    X = []
    y = []

    for dataset_path in dataset_paths:
        image_files = os.listdir(dataset_path)

        for file in image_files:
            img_path = os.path.join(dataset_path, file)
            img = Image.open(img_path)
            img = img.convert('L')
            img = img.resize((128, 128))
            image_array = np.array(img).flatten()
            img.close()

            user_id, rest_of_string = file.split('__')
            finger_type = "_".join(
                rest_of_string.split('.')[0].split('_')[1:4])

            label = int(
                (user_id in anomaly_userids)
                and (
                    (anomaly_fingers is None) or
                    (finger_type in anomaly_fingers)
                )
            )

            X.append(image_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    if dimensionality_reduction:
        from sklearn.manifold import MDS
        mds = MDS(n_components=components, random_state=1234)
        X = mds.fit_transform(X)

    return X, y

def load_data_from_npz(filename):
    """Load data from a .npz file if it exists."""
    if os.path.exists(filename):
        print(f"Loading data from {filename}")
        data = np.load(filename)
        X = data["X"]
        y = data["y"]
        return X, y
    else:
        print(f"{filename} not found. Returning None.")
        return None, None


def save_data_to_npz(X, y, filename):
    """Save the dataset to a .npz file."""
    np.savez_compressed(filename, X=X, y=y)
    print(f"Data saved to {filename}")
