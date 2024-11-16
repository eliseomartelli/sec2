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
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {
                 auc:.2f})', linewidth=2)

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
