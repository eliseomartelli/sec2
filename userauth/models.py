from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utils.metrics import custom_roc_auc
from sklearn.metrics import accuracy_score, classification_report


def define_models():
    """Return a dictionary of models."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=10, max_depth=10, random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=10, algorithm='auto', weights='uniform'
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=1000, random_state=42,
            solver='adam', activation='relu'
        ),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test, from_loaded=False):
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

    return results, trained_models
