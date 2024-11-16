import pickle
import argparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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


def load_data():
    """Load MNIST dataset and preprocess it."""
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Relabel target 3 as an anomaly (1) and all others as normal (0)
    y_relabel = np.where(y == 3, 1, 0)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_relabel, test_size=0.8, random_state=1234
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(models, X_train, X_test, y_train, y_test,
                       from_loaded=False):
    """Train models, evaluate accuracy, ROC, and AUC."""
    results = {}
    trained_models = {}

    for name, model in models.items():
        if not from_loaded:
            model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": acc}

        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred))

        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, auc_value = custom_roc_auc(y_test, y_scores)

        results[name]["fpr"] = fpr
        results[name]["tpr"] = tpr
        results[name]["auc"] = auc_value

        plt.plot(fpr, tpr, label=f'{name} (AUC = {
                 auc_value:.2f})', linewidth=2)

    return results, trained_models


def plot_roc_curve():
    """Plot the ROC curve."""
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", linewidth=2)

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)


def plot_performance(results):
    """Plot accuracy and AUC performance for all models."""
    accuracy_values = [result["accuracy"] for result in results.values()]
    auc_values = [result["auc"] for result in results.values()]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=list(results.keys()), y=accuracy_values,
                palette="viridis", hue=list(results.keys()))
    plt.title("Model Performance (Accuracy)")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    sns.barplot(x=list(results.keys()), y=auc_values,
                palette="viridis", hue=list(results.keys()))
    plt.title("Model Performance (AUC)")
    plt.xlabel("Model")
    plt.ylabel("AUC")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()


def save_models(trained_models, cache_file):
    """Save the trained models to a cache file using pickle."""
    with open(cache_file, 'wb') as f:
        pickle.dump(trained_models, f)


def load_models(cache_file):
    """Load trained models from a cache file."""
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def delete_cache_file(cache_file):
    """Delete the cache file if exists."""
    import os
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Cache file {cache_file} deleted.")


def main(remove_cache=False):
    cache_file = 'trained_models.pkl'
    if remove_cache:
        delete_cache_file(cache_file)

    X_train, X_test, y_train, y_test = load_data()

    results = {}
    trained_models = load_models(cache_file)
    if trained_models is None:
        print("No cached models found. Training new models.")

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=10,
                max_depth=10,
                random_state=42),
            "KNN": KNeighborsClassifier(
                n_neighbors=10,
                algorithm='auto',
                weights='uniform'),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                random_state=42, solver='adam',
                activation='relu'),
        }

        results, trained_models = train_and_evaluate(
            models, X_train, X_test, y_train, y_test)

        save_models(trained_models, cache_file)
    else:
        print("Loaded cached models.")
        results, _ = train_and_evaluate(
            trained_models, X_train, X_test, y_train, y_test, from_loaded=True)

    plot_roc_curve()
    plot_performance(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models.")
    parser.add_argument(
        "--remove-cache", action="store_true",
        help="Remove the cached models file."
    )
    args = parser.parse_args()

    main(remove_cache=args.remove_cache)
