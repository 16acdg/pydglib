from typing import Union
import numpy as np

from pydglib.utils.geometry import is_valid_triangle


def computational_to_physical_1d(
    r: Union[float, np.ndarray], xl: float, xr: float
) -> Union[float, np.ndarray]:
    """
    Maps the point r in the computational domain [-1,1] to the physical domain [`xl`, `xr`]
        using an affine transformation such that -1 -> `xl` and +1 -> `xr`.

    The input `r` can be a single point in [-1, 1] or a numpy array of multiple points in [-1, 1].

    Args:
        `r` (Union[float, np.ndarray]): Point(s) in the computational domain [-1, 1].
        `xl` (float): Left boundary of the physical domain.
        `xr` (float): Right boundary of the physical domain.

    Returns:
        Union[float, np.ndarray]: A float if `r` is a float, or a numpy array of the same shape as `r` if `r` is a numpy array.
    """
    assert xl < xr, f"Invalid physical domain [{xl}, {xr}]."

    m = (xr - xl) / 2
    b = (xr + xl) / 2

    return m * r + b


def physical_to_computational_1d(
    x: Union[float, np.ndarray], xl: float, xr: float
) -> float:
    """
    Maps the point x in the physical domain [`xl`, `xr`] to the computational domain [-1, 1]
        using an affine transformation such that `xl` -> -1 and `xr` -> +1.

    The input `x` can be a single point in [`xl`, `xr`] or a numpy array of multiple points in [`xl`, `xr`].

    Args:
        `x` (Union[float, np.ndarray]): Point(s) in the physical domain [`xl`, `xr`].
        `xl` (float): Left boundary of the physical domain.
        `xr` (float): Right boundary of the physical domain.

    Returns:
        Union[float, np.ndarray]: A float if `x` is a float, or a numpy array of the same shape as `x` if `x` is a numpy array.
    """
    assert xl < xr, f"Invalid physical domain [{xl}, {xr}]."

    m = 2 / (xr - xl)
    b = (xr + xl) / (xl - xr)

    return m * x + b


def computational_to_physical_2d(
    r: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> np.ndarray:
    """
    Maps the point `r` in computational coordinates to the physical coordinates of the triangle defined by the vertices `v1`, `v2`, `v3`.

    Args:
        r (np.ndarray): Point(s) in computational domain.
            1d numpy array if a single point, 2d if multiple points (zeroth axis is batch size).
        v1 (np.ndarray): First vertex of physical domain triangle.
        v2 (np.ndarray): Second vertex of physical domain triangle.
        v3 (np.ndarray): Third vertex of physical domain triangle.

    Returns:
        np.ndarray: Points in physical domain. 1d array if `r` is 1d, 2d if `r` is 2d.
    """
    assert isinstance(r, np.ndarray)
    assert len(r.shape) in [1, 2]
    assert is_valid_triangle(v1, v2, v3)

    is_batch_input = len(r.shape) == 2

    if not is_batch_input:
        r = r.reshape(1, -1)

    a1 = -(r[:, 0] + r[:, 1]) / 2
    a2 = (r[:, 0] + 1) / 2
    a3 = (r[:, 1] + 1) / 2

    x = np.outer(a1, v1) + np.outer(a2, v2) + np.outer(a3, v3)

    if not is_batch_input:
        x = x.reshape(-1)

    return x


def physical_to_computational_2d(
    x: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> np.ndarray:
    """
    Maps the point `x` in physical coordinates of the triangle defined by the vertices `v1`, `v2`, `v3` to computational coordinates`.

    Args:
        x (np.ndarray): Point(s) in physical domain.
            1d numpy array if a single point, 2d if multiple points (zeroth axis is batch size).
        v1 (np.ndarray): First vertex of physical domain triangle.
        v2 (np.ndarray): Second vertex of physical domain triangle.
        v3 (np.ndarray): Third vertex of physical domain triangle.

    Returns:
        np.ndarray: Points in physical domain. 1d array if `r` is 1d, 2d if `r` is 2d.
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) in [1, 2]
    assert is_valid_triangle(v1, v2, v3)

    is_batch_input = len(x.shape) == 2

    if not is_batch_input:
        x = x.reshape(1, -1)

    r = np.zeros_like(x)

    # Solve linear system Ar = b, where A = A(v1, v2, v3), b = b(x, v1, v2, v3)
    A = np.vstack((-0.5 * v1 + 0.5 * v2, -0.5 * v1 + 0.5 * v3)).T
    A_inv = np.linalg.inv(A)
    for i in range(x.shape[0]):
        b = x[i] - 0.5 * v2 - 0.5 * v3
        r[i] = A_inv @ b

    if not is_batch_input:
        r = r.reshape(-1)

    return r
