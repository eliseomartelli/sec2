import argparse
from data import load_data
from models import define_models, train_and_evaluate
from utils.cache import load_models, save_models, delete_cache_file
from utils.plotting import plot_roc_curve, plot_performance


def main(remove_cache=False):
    cache_file = 'trained_models.pkl'

    if remove_cache:
        delete_cache_file(cache_file)

    X_train, X_test, y_train, y_test = load_data()

    trained_models = load_models(cache_file)
    if trained_models is None:
        print("No cached models found. Training new models.")
        models = define_models()
        results, trained_models = train_and_evaluate(
            models, X_train, X_test, y_train, y_test
        )
        save_models(trained_models, cache_file)
    else:
        print("Loaded cached models.")
        results, _ = train_and_evaluate(
            trained_models, X_train, X_test, y_train, y_test, from_loaded=True
        )

    plot_roc_curve(results)
    plot_performance(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models.")
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="Remove the cached models file."
    )
    args = parser.parse_args()
    main(remove_cache=args.remove_cache)
