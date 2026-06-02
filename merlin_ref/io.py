"""
I/O helpers for saving and loading predictions and auxiliary files.
"""

import pickle
import numpy as np


def save_predictions(predictions, save_path):
    """Pickle *predictions* to *save_path*."""
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {save_path}")


def load_file(file_path):
    """Load a ``.npy`` / ``.npz`` file (with pickle support)."""
    return np.load(file_path, allow_pickle=True)
