import os
import numpy as np


def load(file: str) -> np.ndarray:
    """
    Loads the given file from this directory and returns its contents as a numpy array.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assert file in os.listdir(dir_path)

    file_path = os.path.join(dir_path, file)

    return np.load(file_path)
