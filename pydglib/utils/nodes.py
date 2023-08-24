from typing import Tuple
import numpy as np
from scipy.special import legendre as legendre_unscaled

from pydglib.utils.polynomials import legendre
from pydglib.utils.transformations import physical_to_computational_2d


# Maximum number of nodes that will be returned by get_nodes
NODES_MAX_1D = 20
NODES_MAX_2D = 250


def get_nodes_1d(degree: int) -> np.ndarray:
    """
    Returns the Legendre-Gauss-Lobatto (LGL) nodes for the computational domain [-1, 1].

    Args:
        degree (int): The degree of the local polynomial approximation.

    Returns:
        np.ndarray: `degree` + 1 nodes in increasing order as a 1d numpy array.
    """
    n_nodes = degree + 1
    assert n_nodes > 1
    assert n_nodes <= NODES_MAX_1D

    q = np.poly1d([-1, 0, 1])  # 1 - x**2
    p = np.polymul(q, legendre_unscaled(degree).deriv())

    nodes = np.roots(p)
    nodes.sort()

    # Overwrite boundary nodes because these are always +/- 1
    nodes[0] = -1
    nodes[-1] = 1

    # If length of nodes is odd, overwrite middle node because it is always 0
    if n_nodes % 2 == 1:
        nodes[int(n_nodes / 2)] = 0

    return nodes


def _get_alpha_opt(degree: int) -> float:
    """
    Returns the optimum value of alpha for shifting the the equidistant nodes.

    Args:
        degree (int): Degree of the local polynomial approximation.

    Returns:
        float: The optimal value of alpha for the given `degree`.
    """
    assert 1 <= degree
    assert isinstance(degree, int)

    ALPHA_OPT = [
        0.0,
        0.0,
        1.4152,
        0.1001,
        0.2751,
        0.9808,
        1.0999,
        1.2832,
        1.3648,
        1.4773,
        1.4959,
        1.5743,
        1.5770,
        1.6223,
        1.6258,
    ]
    return ALPHA_OPT[degree - 1] if degree <= 15 else 5 / 3


def warpfactor(degree: int, rout: np.ndarray) -> np.ndarray:
    # Compute LGL and equidistant node distribution
    LGLr = get_nodes_1d(degree)
    req = np.linspace(-1, 1, degree + 1)

    # Compute V based on req
    Veq = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        Veq[:, i] = legendre(i)(req)

    # Evaluate Lagrange polynomial at rout
    Nr = len(rout)
    Pmat = np.zeros((degree + 1, Nr))
    for i in range(degree + 1):
        Pmat[i] = legendre(i)(rout)
    Lmat = np.linalg.pinv(Veq.T) @ Pmat

    # Compute warp factor
    warp = Lmat.T @ (LGLr - req)

    # Scale factor
    zerof = np.abs(rout) < 1 - 1e-10
    sf = 1 - np.multiply(zerof, rout) ** 2
    warp = np.divide(warp, sf) + np.multiply(warp, zerof - 1)

    return warp


def get_nodes_2d(
    degree: int, include_boundary: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns nodes for the 2d reference element I = ConvexHull{(-1,-1), (1,-1), (-1,1)}.

    The nodes are derived from equidistant nodes on I with warping derived from the 1d LGL nodes.

    Args:
        degree (int): Degree of the local polynomial approximation.
        include_boundary (bool, optional): If True, include numpy arrays specifying indices of nodes that are on the boundary of I. Defaults to False.

    Returns:
        np.darray: Returns Np := (`degree` + 1) * (`degree` + 2) / 2 nodes.
            The nodes are returned as 2d numpy array of size Np-by-2.
        If `include_boundary` = True, then the following is also returned:
        np.ndarray: 1d numpy array of indices of nodes on first edge of I.
        np.ndarray: 1d numpy array of indices of nodes on second edge of I.
        np.ndarray: 1d numpy array of indices of nodes on third edge of I.
    """
    alpha = _get_alpha_opt(degree)

    n_nodes = int(0.5 * (degree + 1) * (degree + 2))

    # Create equidistributed nodes on equilateral triangle
    nodes = np.zeros((n_nodes, 2))
    L1 = np.zeros(n_nodes)
    L2 = np.zeros(n_nodes)
    L3 = np.zeros(n_nodes)
    node_idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            L1[node_idx] = i / degree
            L3[node_idx] = j / degree
            node_idx += 1
    L2 = 1 - L1 - L3
    nodes[:, 0] = L3 - L2  # x positions
    nodes[:, 1] = (2 * L1 - L2 - L3) / np.sqrt(3)  # y positions

    # Compute blending function at each node for each edge
    blend1 = 4 * np.multiply(L2, L3)
    blend2 = 4 * np.multiply(L1, L3)
    blend3 = 4 * np.multiply(L1, L2)

    # Amount of warp for each node, for each edge
    warpf1 = warpfactor(degree, L3 - L2)
    warpf2 = warpfactor(degree, L1 - L3)
    warpf3 = warpfactor(degree, L2 - L1)

    # Combine blend & warp
    warp1 = np.multiply(np.multiply(blend1, warpf1), 1 + (alpha * L1) ** 2)
    warp2 = np.multiply(np.multiply(blend2, warpf2), 1 + (alpha * L2) ** 2)
    warp3 = np.multiply(np.multiply(blend3, warpf3), 1 + (alpha * L3) ** 2)

    # Accumulate deformations associated with each edge
    nodes[:, 0] += warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    nodes[:, 1] += np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3

    # Get indicies of nodes that are on boundaries
    if include_boundary:
        b1 = []
        b2 = []
        b3 = []
        for i in range(n_nodes):
            if np.isclose(L1[i], 0):
                b1.append(i)
            if np.isclose(L2[i], 0):
                b2.append(i)
            if np.isclose(L3[i], 0):
                b3.append(i)
        b3.reverse()
        b1 = np.array(b1)
        b2 = np.array(b2)
        b3 = np.array(b3)

    # Vertices of the equilateral triangle
    v1 = np.array([-1, -1 / np.sqrt(3)])
    v2 = np.array([1, -1 / np.sqrt(3)])
    v3 = np.array([0, 2 / np.sqrt(3)])

    # Map nodes to computational domain
    nodes = physical_to_computational_2d(nodes, v1, v2, v3)

    if include_boundary:
        return nodes, b1, b2, b3
    else:
        return nodes
