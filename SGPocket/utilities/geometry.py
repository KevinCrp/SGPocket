import numpy as np


def geometric_center(coords: np.array) -> np.array:
    """Compute the center of coordinates

    Args:
        coords (np.array): coordinates

    Returns:
        np.array: center
    """
    if coords.shape[0] == 0:
        return None
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = ((max_coords - min_coords) / 2) + min_coords
    return center
