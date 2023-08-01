import numpy as np

from SGPocket.utilities.geometry import geometric_center


def test_geometric_center():
    coords = np.array([[1.0, 2.0, 3.0],
                       [3.0, 1.0, 2.0],
                       [2.0, 5.0, 7.0]])
    center = geometric_center(coords)
    assert (center[0] == 2.0)
    assert (center[1] == 3.0)
    assert (center[2] == 4.5)


def test_geometric_center_none():
    coords = np.array([])
    center = geometric_center(coords)
    assert (center is None)
