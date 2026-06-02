import pickle
import numpy as np


def save_predictions(predictions, path):
    with open(path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {path}")


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_file(path, **kwargs):
    return np.load(path, allow_pickle=True, **kwargs)
