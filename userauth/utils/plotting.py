import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve(results):
    """
    Plot the ROC curves for all models based on their results.

    Parameters:
    - results: Dictionary containing FPR, TPR, and AUC for each model.
    """
    for model_name, metrics in results.items():
        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        auc = metrics["auc"]
        library_auc = metrics.get("metrics_roc_auc", None)

        label = f'{model_name} (Custom AUC = {auc:.2f})'
        label += f', Library AUC = {library_auc:.2f}'

        plt.plot(fpr, tpr, label=label, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", linewidth=2)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_performance(results):
    """Plot accuracy and AUC performance for all models."""
    accuracy_values = [result["accuracy"] for result in results.values()]
    auc_values = [result["auc"] for result in results.values()]
    library_auc_values = [result.get("metrics_roc_auc", 0)
                          for result in results.values()]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    sns.barplot(x=list(results.keys()), y=accuracy_values,
                palette="viridis", hue=list(results.keys()))
    plt.title("Model Performance (Accuracy)")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    sns.barplot(x=list(results.keys()), y=auc_values,
                palette="viridis", hue=list(results.keys()))
    plt.title("Model Performance (AUC)")
    plt.xlabel("Model")
    plt.ylabel("AUC")
    plt.ylim(0, 1)

    plt.subplot(1, 3, 3)
    sns.barplot(x=list(results.keys()), y=library_auc_values,
                palette="viridis", hue=list(results.keys()))
    plt.title("Library AUC Performance")
    plt.xlabel("Model")
    plt.ylabel("Library AUC")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(results):
    """
    Plot the Precision-Recall curve for all models.

    Parameters:
    - results: Dictionary containing 'precision', 'recall', and 'auc_pr' for
    each model.
    """
    plt.figure(figsize=(8, 6))

    for name, result in results.items():
        precision = result.get("precision", [])
        recall = result.get("recall", [])
        auc_pr = result.get("auc_pr", 0)

        if (precision is not None and recall is not None
                and precision.size > 0 and recall.size > 0):
            plt.plot(recall, precision, label=f'{
                     name} (AUC = {auc_pr:.2f})', linewidth=2)

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
