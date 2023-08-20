from typing import List, Callable, Tuple
import numpy as np

from pydglib.utils.nodes import get_nodes_1d, get_nodes_2d
from pydglib.utils.polynomials import (
    jacobi,
    legendre,
    jacobi_deriv,
    legendre_deriv,
    Polynomial,
)

from pydglib.element import get_reference_triangle


def _get_orthonormal_poly_basis_1d(
    degree: int,
) -> Tuple[List[Polynomial], List[Polynomial]]:
    """
    Returns the orthonormal Legendre polynomial basis up to the given degree `degree`.

    Args:
        degree (int): Maximum degree of the basis functions.

    Returns:
        List[Polynomial]: Legendre polynomials ordered by increasing degree.
        List[Polynomial]: Derivatives of the legendre polynomials ordered by increasing degree.
    """
    assert isinstance(degree, int)
    assert degree >= 0

    P = [legendre(i) for i in range(degree + 1)]
    Pr = [legendre_deriv(i) for i in range(degree + 1)]

    return P, Pr


def _make_poly_2d(Legendre, Legendre_deriv, Jacobi, Jacobi_deriv, i):
    def a(r, s):
        if s == 1:
            return -1
        else:
            return 2 * (1 + r) / (1 - s) - 1

    def _as(r, s):
        if s == 1:
            return 0
        else:
            return 2 * (1 + r) / ((1 - s) ** 2)

    def p(r, s):
        # TODO: I don't think this should be necessary...
        # if s == 1:
        #     s -= 0.000001
        return np.sqrt(2) * Legendre(a(r, s)) * Jacobi(s) * (1 - s) ** i

    # Partial of p wrt r
    def pr(r, s):
        out = 2 * np.sqrt(2) * Legendre_deriv(a(r, s)) * Jacobi(s)
        if i > 0:
            out *= (1 - s) ** (i - 1)
        return out

    # Partial of p wrt s
    def ps(r, s):
        out = (
            _as(r, s) * np.sqrt(2) * Legendre_deriv(a(r, s)) * Jacobi(s) * (1 - s) ** i
            + np.sqrt(2) * Legendre(a(r, s)) * Jacobi_deriv(s) * (1 - s) ** i
        )
        if i > 0:
            out += (
                np.sqrt(2)
                * Legendre(a(r, s))
                * i
                * Jacobi(s)
                * (1 - s) ** (i - 1)
                * (-1)
            )
        return out

    return p, pr, ps


def _get_orthonormal_poly_basis_2d(
    degree: int,
) -> Tuple[List[Polynomial], List[Polynomial], List[Polynomial]]:
    """
    Returns a 2d polynomial basis that is orthonormal wrt the reference triangle.

    Args:
        degree (int): Maximum degree of the basis functions.

    Returns:
        List[Polynomial]: Orthonormal polynomials.
        List[Polynomial]: Derivative of polynomials wrt the r direction in computational space.
        List[Polynomial]: Derivative of polynomials wrt the s direction in computational space.
    """
    assert isinstance(degree, int)
    assert degree >= 0

    P = []
    Pr = []
    Ps = []

    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j <= degree:
                L = legendre(i)
                dL = legendre_deriv(i)
                J = jacobi(2 * i + 1, 0, j)
                dJ = jacobi_deriv(2 * i + 1, 0, j)
                p, pr, ps = _make_poly_2d(L, dL, J, dJ, i)

                P.append(p)
                Pr.append(pr)
                Ps.append(ps)

    return P, Pr, Ps


def _get_V(nodes: np.ndarray, P: List[Callable]) -> np.ndarray:
    """
    Returns a Vandermonde matrix defined by V[i,j] = `P`[j](`nodes`[i]).

    Args:
        nodes (np.ndarray): Nodes for a computational domain. Can be a 1d or 2d numpy array.
        P (List[Callable]): Polynomial basis. Functions must accept scalars if `nodes` is 1d,
            or functions must have `nodes.shape[1]` inputs if `nodes` is 2d. Functions must return scalars.

    Returns:
        np.ndarray: Vandermonde matrix as a square 2d numpy array.
    """
    assert isinstance(nodes, np.ndarray)

    if len(nodes.shape) == 1:
        nodes = nodes.reshape(-1, 1)

    assert len(nodes.shape) == 2
    n_nodes = nodes.shape[0]

    V = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            V[i, j] = P[j](*nodes[i])
    return V


def get_derivative_operator_1d(degree: int) -> np.ndarray:
    """
    Returns the derivative operator Dr.

    The operator is polynomially exact wrt the reference interval [-1, 1] for
    polynomial inputs with degree up to and including `degree`.

    Args:
        degree (int): Degree of the local polynomial approximation.

    Returns:
        np.ndarray: Derivative operator as a 2d square numpy array.
    """
    reference_nodes = get_nodes_1d(degree)
    P, Pr = _get_orthonormal_poly_basis_1d(degree)

    V = _get_V(reference_nodes, P)
    Vr = _get_V(reference_nodes, Pr)

    V_inv = np.linalg.inv(V)
    Dr = Vr @ V_inv

    return Dr


def get_LIFT_1d(degree: int) -> np.ndarray:
    """
    Returns the LIFT matrix defined by M^-1 @ Emat, where Emat = [1 0; 0 0; ...; 0 0; 0 1].

    Args:
        degree (int): Degree of the local polynomial approximation

    Returns:
        np.ndarray: LIFT matrix as a 2d numpy array.
    """
    n_nodes = degree + 1
    reference_nodes = get_nodes_1d(degree)
    P, _ = _get_orthonormal_poly_basis_1d(degree)

    V = _get_V(reference_nodes, P)

    Emat = np.zeros((n_nodes, 2), dtype=np.float64)
    Emat[0, 0] = 1
    Emat[-1, -1] = 1

    # M^-1 = V @ V.T
    LIFT = np.matmul(V, np.matmul(V.T, Emat))

    return LIFT


def get_derivative_operators_2d(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the derivative operators `Dr` and `Ds` for the given degree of the local polynomial approximation.

    FIXME: This function is very slow for large degree. Try to speed it up by reducing redundant calls.

    Args:
        degree (int): Degree of the local polynomial approximation.

    Returns:
        np.ndarray: The derivative operator `Dr` for the r coordinate in computational space as a 2d array.
        np.ndarray: The derivative operator `Ds` for the s coordinate in computational space as a 2d array.
    """
    reference_nodes = get_nodes_2d(degree)
    P, Pr, Ps = _get_orthonormal_poly_basis_2d(degree)

    V = _get_V(reference_nodes, P)
    Vr = _get_V(reference_nodes, Pr)
    Vs = _get_V(reference_nodes, Ps)

    V_inv = np.linalg.inv(V)
    Dr = Vr @ V_inv
    Ds = Vs @ V_inv

    return Dr, Ds


def get_edge_mass_matrix_2d(degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates and returns mass matrices for each edge of a triangular 2d element.

    The mass matrices allow computation of surface integral terms along the edges of the element.

    Args:
        degree (int): Degree of the local polynomial approximation

    Returns:
        np.ndarray: Edge mass matrix as a square 2d numpy array of size `degree`+1.
    """
    edge_nodes = get_nodes_1d(degree)
    P, _ = _get_orthonormal_poly_basis_1d(degree)
    V1D = _get_V(edge_nodes, P)
    M = np.linalg.inv(V1D @ V1D.T)
    return M


def get_LIFT_2d(degree: int) -> np.ndarray:
    n_nodes = int(0.5 * (degree + 1) * (degree + 2))
    n_edge_nodes = degree + 1

    nodes = get_nodes_2d(degree)
    P, _, _ = _get_orthonormal_poly_basis_2d(degree)
    V = _get_V(nodes, P)
    M_inv = V @ V.T

    Emat = np.zeros((n_nodes, 3 * n_edge_nodes))
    M_edge = get_edge_mass_matrix_2d(degree)

    reference_element = get_reference_triangle(degree)

    Emat[reference_element._edge_node_indicies[0], :n_edge_nodes] = M_edge
    Emat[
        reference_element._edge_node_indicies[1], n_edge_nodes : 2 * n_edge_nodes
    ] = M_edge
    Emat[np.flip(reference_element._edge_node_indicies[2]), 2 * n_edge_nodes :] = M_edge

    LIFT = M_inv @ Emat

    return LIFT
