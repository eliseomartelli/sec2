from utils.plotting import (
    plot_roc_curve, plot_performance, plot_precision_recall_curve)
from utils.cache import load_models, save_models, delete_cache_file
import argparse
from data import load_data
from models import (define_models, train_models, evaluate_models, tune_models,
                    define_param_grid)


def main(remove_cache=False, tune=False):
    cache_file = 'trained_models.pkl'

    if remove_cache:
        delete_cache_file(cache_file)

    X_train, X_test, y_train, y_test = load_data()

    models = load_models(cache_file)
    if models is None:
        print("No cached models found. Training new models.")
        models, param_grids = define_models(), define_param_grid()
        if tune:
            models = tune_models(models, param_grids, X_train, y_train)
        else:
            models = train_models(models, X_train, y_train)

        save_models(models, cache_file)
    else:
        print("Loaded cached models.")

    results = evaluate_models(
        models, X_test, y_test
    )

    plot_roc_curve(results)
    plot_performance(results)
    plot_precision_recall_curve(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models.")
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="Remove the cached models file."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning."
    )
    args = parser.parse_args()
    main(remove_cache=args.remove_cache, tune=args.tune)
