import os
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
from skimage import exposure
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
    train_filename = "SOCOFing_train_data.npz"
    test_filename = "SOCOFing_test_data.npz"

    train_paths = [
        "./SOCOFing/Real/",
    ]
    test_paths = [
        "./SOCOFing/Altered/Altered-Easy/",
        "./SOCOFing/Altered/Altered-Hard/",
        "./SOCOFing/Altered/Altered-Medium/",
    ]

    X_train, y_train = socofing_data_loader_util(
        anomaly_fingers=None,
        dataset_paths=train_paths,
        filename=train_filename,
    )
    X_test, y_test = socofing_data_loader_util(
        anomaly_fingers=None,
        dataset_paths=test_paths,
        filename=test_filename,
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
            str(number) for number in range(60)
        ],
        dimensionality_reduction=True,
        num_threads=8,
        filename=None,
):
    """
    Load and preprocess the SOCOFing dataset.
    Returns:
        X, y: Preprocessed dataset.
    """
    if filename:
        X, y = load_data_from_npz(filename)
        if X is not None and y is not None:
            return X, y

    X = []
    y = []

    # Thread pool executor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for dataset_path in dataset_paths:
            image_files = os.listdir(dataset_path)

            for file in image_files:
                futures.append(executor.submit(
                    process_image,
                    file,
                    dataset_path,
                    anomaly_userids,
                    anomaly_fingers,
                    dimensionality_reduction
                ))

        # Collect results from the futures
        for future in futures:
            image_array, label = future.result()
            X.append(image_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    if filename:
        save_data_to_npz(X, y, filename)

    return X, y


def process_image(file, dataset_path, anomaly_userids, anomaly_fingers,
                  dimensionality_reduction):
    """Helper function to process each image."""
    img_path = os.path.join(dataset_path, file)
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((128, 128))
    image_array = np.array(img)
    image_array -= 255
    img.close()

    user_id, rest_of_string = file.split('__')
    finger_type = "_".join(rest_of_string.split('.')[0].split('_')[1:4])

    label = int(
        (user_id in anomaly_userids)
        and (
            (anomaly_fingers is None) or
            (finger_type in anomaly_fingers)
        )
    )
    if dimensionality_reduction:
        radius = 1
        n_points = 8 * radius
        image_array = local_binary_pattern(
            image_array, n_points, radius, method="uniform")
        image_array = exposure.equalize_hist(image_array)

    image_array = image_array.flatten()

    return image_array, label


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
