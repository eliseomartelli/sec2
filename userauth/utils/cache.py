import pickle
import os


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
    """Delete the cache file if it exists."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Cache file {cache_file} deleted.")
