from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utils.metrics import custom_roc_auc
from sklearn.metrics import (
    accuracy_score,
    classification_report, precision_recall_curve,
    auc)


def define_models():
    """Return a dictionary of models."""
    return {
        "Random Forest": RandomForestClassifier(
            random_state=1234,
        ),
        "KNN": KNeighborsClassifier(
        ),
        "MLP": MLPClassifier(
            random_state=1234,
        ),
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
        fpr, tpr, auc_roc = custom_roc_auc(y_test, y_scores)

        results[name]["fpr"] = fpr
        results[name]["tpr"] = tpr
        results[name]["auc"] = auc_roc

        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auc_pr = auc(recall, precision)
        results[name]["precision"] = precision
        results[name]["recall"] = recall
        results[name]["auc_pr"] = auc_pr

    return results, trained_models


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
