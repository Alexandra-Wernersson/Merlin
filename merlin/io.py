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


def load_array(value):
    """
    Resolve a config entry that may be either a file path (str) to a .npy
    file, or an inline list/array of values already parsed from YAML.
    """
    if isinstance(value, str):
        return np.load(value)
    return np.asarray(value, dtype=float)


def format_duration(seconds):
    """Format a wall-clock duration for a progress print: hours if >=1h, minutes if >=1min, else seconds."""
    if seconds >= 3600:
        return f"{seconds/3600:.2f} hours"
    if seconds >= 60:
        return f"{seconds/60:.1f} minutes"
    return f"{seconds:.1f} seconds"
