import numpy as np


def meshgen1d(xl: float, xr: float, n_elements: int):
    """
    Creates a mesh for the physical domain [`xl`, `xr`] consisting of `n_elements` evenly spaced elements.

    Args:
        xl (float): Left boundary of the physical domain.
        xr (float): Right boundary of the physical domain.
        n_elements (int): Number of elements to partition the physical domain into.

    Returns:
        np.ndarray: VX, a 1D numpy array of the vertices of the mesh in the physical domain.
        np.ndarray: EToV, a 2D numpy array mapping elements to the global vertex number of the vertex at its left and right faces.
    """
    VX = np.linspace(xl, xr, n_elements + 1)

    EToV = np.zeros((n_elements, 2), dtype=np.int32)
    for i in range(n_elements):
        EToV[i, 0] = i
        EToV[i, 1] = i + 1

    return VX, EToV


def meshgen2d(x1: float, x2: float, y1: float, y2: float, nx: int, ny: int):
    """
    Creates a mesh for the domain [`x1`, `x2`] x [`y1`, `y2`] consisting of 2*`nx`*`ny` triangular elements.

    Triangular elements are oriented counter-clockwise.

    Args:
        x1 (float): Left x boundary of the domain.
        x2 (float): Right x boundary of the domain.
        y1 (float): Bottom y boundary of the domain.
        y2 (float): Top y boundary of the domain.
        nx (int): Partition [`x1`, `x2`] into this many evenly spaced partitions.
        ny (int): Partition [`y1`, `y2`] into this many evenly spaced partitions.

    Returns:
        np.ndarray: `VX`, the x positions of mesh vertices.
        np.ndarray: `VY`, the y positions of mesh vertices.
        np.ndarray: `EToV`, element-to-vertex map.
    """
    assert x1 < x2
    assert y1 < y2

    assert nx > 0
    assert ny > 0

    assert isinstance(nx, int)
    assert isinstance(ny, int)

    # Vertex x and y coordinates
    x = np.linspace(x1, x2, nx + 1)
    y = np.linspace(y1, y2, ny + 1)
    VX, VY = np.meshgrid(x, y)
    VX = VX.reshape(-1)
    VY = VY.reshape(-1)

    # Create element-to-vertex map
    n_elements = 2 * nx * ny
    EToV = np.zeros((n_elements, 3), dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            k = 2 * (i * nx + j)

            EToV[k, 0] = i * (nx + 1) + j
            EToV[k, 1] = i * (nx + 1) + j + 1
            EToV[k, 2] = i * (nx + 1) + j + 1 + nx + 1

            EToV[k + 1, 0] = i * (nx + 1) + j
            EToV[k + 1, 1] = i * (nx + 1) + j + 1 + nx + 1
            EToV[k + 1, 2] = i * (nx + 1) + j + 1 + nx

    return VX, VY, EToV
