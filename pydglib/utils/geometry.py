from typing import Union, List, Tuple
import numpy as np


def _is_valid_vertex(v: np.ndarray) -> bool:
    return isinstance(v, np.ndarray) and len(v.shape) == 1 and v.shape[0] == 2


def is_valid_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
    """
    Returns True if the triangle with vertices `v1`, `v2`, `v3` is valid.

    A valid triangle must have valid vertices and the area of the convex hull of  `v1`, `v2`, `v3` must be positive.

    Args:
        v1 (np.ndarray): First vertex of the triangle.
        v2 (np.ndarray): Second vertex of the triangle.
        v3 (np.ndarray): Third vertex of the triangle.

    Returns:
        bool: True if valid, False otherwise.
    """
    return get_area_of_triangle(v1, v2, v3) > 0


def get_area_of_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Returns the surface area of the triangle with vertices `v1`, `v2`, `v3`.

    The first coordinate of each vertex is the x position, the second is the y position.

    Args:
        v1 (np.ndarray): First vertex of the triangle.
        v2 (np.ndarray): Second vertex of the triangle.
        v3 (np.ndarray): Third vertex of the triangle.

    Returns:
        float: Surface area of the triangle.
    """
    for v in [v1, v2, v3]:
        assert _is_valid_vertex(v)

    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return area


def get_perimeter_of_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Returns the perimeter of the triangle with vertices `v1`, `v2`, `v3`.

    The first coordinate of each vertex is the x position, the second is the y position.

    Args:
        v1 (np.ndarray): First vertex of a triangle.
        v2 (np.ndarray): Second vertex of a triangle.
        v3 (np.ndarray): Third vertex of a triangle.

    Returns:
        float: Perimeter of the triangle.
    """
    for v in [v1, v2, v3]:
        assert _is_valid_vertex(v)

    perimeter = sum([np.linalg.norm(e) for e in [v1 - v2, v2 - v3, v3 - v1]])

    return perimeter


def get_outward_unit_normals_of_triangle(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the outward unit normals of the triangle with vertices `v1`, `v2`, `v3`.

    The first coordinate of each vertex is the x position, the second is the y position.
    Assumes that the vertices are ordered counter-clockwise.

    Args:
        v1 (np.ndarray): First vertex of a triangle.
        v2 (np.ndarray): Second vertex of a triangle.
        v3 (np.ndarray): Third vertex of a triangle.

    Returns:
        np.ndarray: The outward unit normal of the edge `v1`-`v2`.
        np.ndarray: The outward unit normal of the edge `v2`-`v3`.
        np.ndarray: The outward unit normal of the edge `v3`-`v1`.
    """

    # Make vertices 3d
    v1 = np.hstack((v1, 0))
    v2 = np.hstack((v2, 0))
    v3 = np.hstack((v3, 0))

    # Define edges of the triangle
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v1 - v3

    # Find outward normals via cross products (in 3-space)
    z = np.array([0, 0, 1])
    n1 = np.cross(e1, z)[:2]
    n2 = np.cross(e2, z)[:2]
    n3 = np.cross(e3, z)[:2]

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    n3 = n3 / np.linalg.norm(n3)

    return n1, n2, n3
