from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utils.metrics import custom_roc_auc
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_curve, auc, roc_auc_score)


def define_models():
    """Return a dictionary of models."""
    return {
        "Random Forest": RandomForestClassifier(
            random_state=1234,
            max_depth=20,
            max_features="sqrt",
            n_estimators=100,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            algorithm="auto",
        ),
        "MLP": MLPClassifier(
            random_state=1234,
            hidden_layer_sizes=(100,),
            activation="relu",
            hidden_layer_sizes=(100, 50,),
            max_iter=500,
            solver="sgd",
    }


def define_param_grid():
    """Return a dictionary of model names and their parameter grids."""
    return {
        "Random Forest": {
            "n_estimators": [10, 50, 100],
            "max_depth": [5, 10, 20],
            "max_features": ["sqrt", "log2"],
        },
        "KNN": {
            "n_neighbors": [5, 10, 15],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
        },
        "MLP": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "solver": ["adam", "sgd"],
            "activation": ["relu", "tanh"],
            "max_iter": [500, 1000],
        },
    }


def train_model(model, X_train, y_train):
    """Trains a single model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model on test data.

    Returns:
    - results: Dictionary with accuracy, ROC, and Precision-Recall metrics.
    """
    results = {}

    y_pred = model.predict(X_test)
    results["accuracy"] = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    # ROC metrics
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, auc_roc = custom_roc_auc(y_test, y_scores)
    results["fpr"] = fpr
    results["tpr"] = tpr
    results["auc"] = auc_roc
    results["metrics_roc_auc"] = roc_auc_score(y_test, y_scores)

    # Precision-Recall metrics
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auc_pr = auc(recall, precision)
    results["precision"] = precision
    results["recall"] = recall
    results["auc_pr"] = auc_pr

    return results


def train_models(models, X_train, y_train):
    """Train models."""
    trained_models = {}
    for name, model in models.items():
        print(f"--- Training {name} ---")
        trained_models[name] = train_model(model, X_train, y_train)
    return trained_models


def evaluate_models(models, X_test, y_test):
    """Evaluate models."""
    results = {}
    for name, model in models.items():
        print(f"--- Evaluating {name} ---")
        results[name] = evaluate_model(model, X_test, y_test)
    return results


def tune_models(models, param_grids, X_train, y_train):
    """
    Tune models using GridSearchCV and return the best models with their
    parameters.

    Parameters:
    - models: Dictionary of model names and instances.
    - param_grids: Dictionary of model names and their parameter grids.
    - X_train: Training data features.
    - y_train: Training data labels.

    Returns:
    - best_models: Dictionary of model names and their tuned instances.
    """
    best_models = {}

    for name, model in models.items():
        if name in param_grids:
            print(f"Tuning {name}...")
            grid_search = GridSearchCV(
                model, param_grids[name], scoring='roc_auc', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {
                grid_search.best_params_}")
        else:
            # No tuning grid provided, use the original model
            best_models[name] = model

    return best_models
