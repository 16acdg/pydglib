from abc import ABC, abstractmethod
from typing import List, Callable, Tuple
import numpy as np
from pydglib.element import Element2D

from pydglib.utils.nodes import get_nodes_1d, get_nodes_2d
from pydglib.utils.polynomials import (
    jacobi,
    legendre,
    jacobi_deriv,
    legendre_deriv,
    Polynomial,
)
from .element import Element1D, Element2D


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


def rstoab(r, s):
    n_nodes = len(r)
    a = np.zeros(n_nodes)
    for i in range(n_nodes):
        if s[i] != 1:
            a[i] = 2 * (1 + r[i]) / (1 - s[i]) - 1
        else:
            a[i] = -1
    b = s
    return a, b


def Simplex2DP(a, b, i, j):
    h1 = jacobi(0, 0, i)(a)
    h2 = jacobi(2 * i + 1, 0, j)(b)
    P = np.sqrt(2) * np.multiply(h1, np.multiply(h2, (1 - b) ** i))
    return P


def Vandermonde2D(N, r, s):
    n_nodes = int(0.5 * (N + 1) * (N + 2))
    V2D = np.zeros((len(r), n_nodes))
    a, b = rstoab(r, s)
    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk += 1
    return V2D


def GradSimplex2DP(a, b, i, j):
    fa = jacobi(0, 0, i)(a)
    gb = jacobi(2 * i + 1, 0, j)(b)
    dfa = jacobi_deriv(0, 0, i)(a)
    dgb = jacobi_deriv(2 * i + 1, 0, j)(b)

    # r derivative
    dmodedr = np.multiply(dfa, gb)
    if i > 0:
        dmodedr = np.multiply(dmodedr, (0.5 * (1 - b)) ** (i - 1))

    # s derivative
    dmodeds = np.multiply(dfa, np.multiply(gb, 0.5 * (1 + a)))
    if i > 0:
        dmodeds = np.multiply(dmodeds, (0.5 * (1 - b)) ** (i - 1))

    tmp = np.multiply(dgb, (0.5 * (1 - b)) ** i)
    if i > 0:
        tmp -= 0.5 * i * np.multiply(gb, (0.5 * (1 - b)) ** (i - 1))
    dmodeds += np.multiply(fa, tmp)

    # normalize
    dmodedr *= 2 ** (i + 0.5)
    dmodeds *= 2 ** (i + 0.5)

    return dmodedr, dmodeds


def GradVandermonde2D(N, r, s):
    n_nodes = int(0.5 * (N + 1) * (N + 2))
    V2Dr = np.zeros((len(r), n_nodes))
    V2Ds = np.zeros((len(r), n_nodes))
    a, b = rstoab(r, s)
    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk += 1
    return V2Dr, V2Ds


def get_derivative_operators_2d(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the derivative operators `Dr` and `Ds` for the given degree of the local polynomial approximation.

    Args:
        degree (int): Degree of the local polynomial approximation.

    Returns:
        np.ndarray: The derivative operator `Dr` for the r coordinate in computational space as a 2d array.
        np.ndarray: The derivative operator `Ds` for the s coordinate in computational space as a 2d array.
    """
    reference_nodes = get_nodes_2d(degree)
    r = reference_nodes[:, 0]
    s = reference_nodes[:, 1]
    V = Vandermonde2D(degree, r, s)
    Vr, Vs = GradVandermonde2D(degree, r, s)

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


def get_LIFT_2d(degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = int(0.5 * (degree + 1) * (degree + 2))
    n_edge_nodes = degree + 1

    nodes, *edge_node_indices = get_nodes_2d(degree, include_boundary=True)
    V = Vandermonde2D(degree, nodes[:, 0], nodes[:, 1])
    M_inv = V @ V.T

    Emat = [np.zeros((n_nodes, n_edge_nodes)) for _ in range(3)]
    M_edge = get_edge_mass_matrix_2d(degree)

    for i in range(3):
        Emat[i][edge_node_indices[i]] = M_edge

    LIFT = [M_inv @ E for E in Emat]

    return LIFT


class DerivativeOperator2D(ABC):
    def __init__(self, degree: int):
        """
        Creates a new DerivativeOperator2D instance.

        Args:
            degree (int): Degree of the local polynomial approximation used by all elements in a grid.
        """
        self.degree = degree

    @abstractmethod
    def _get_Dr_and_Ds_for_element(
        self, element: Element2D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the derivative operators Dr and Ds for the given element.

        Args:
            element (Element2D): Returns derivative operators specific to this element.

        Returns:
            np.ndarray: The derivative operator `Dr` for the r coordinate in computational space as a 2d array.
            np.ndarray: The derivative operator `Ds` for the s coordinate in computational space as a 2d array.
        """
        pass

    def grad(self, u: np.ndarray, element: Element2D) -> np.ndarray:
        """
        Computes the gradient of the state variable `u` for the element `element`.

        Args:
            u (np.ndarray): 1d state vector for a single component of the `element`'s state.
            element (Element2D): Element who has the state `u`.

        Returns:
            np.ndarray: The gradient of the vector `u` as a 1d numpy array.
        """
        Dr, Ds = self._get_Dr_and_Ds_for_element(element)

        ur = Dr @ u
        us = Ds @ u

        ux = element.rx * ur + element.sx * us
        uy = element.ry * ur + element.sy * us

        return ux, uy

    def curl(self, u: np.ndarray, v: np.ndarray, element: Element2D) -> np.ndarray:
        """
        Compute the 2d curl operator in the x-y plane.

        Args:
            u (np.ndarray): x component of state as a 1d numpy array.
            v (np.ndarray): y component of state as a 1d numpy array.

        Returns:
            np.ndarray: The curl of the 2d vector field (`ux`, `uy`) in the z direction as a 1d numpy array.
        """
        Dr, Ds = self._get_Dr_and_Ds_for_element(element)

        ur = Dr @ u
        us = Ds @ u
        vr = Dr @ v
        vs = Ds @ v

        vz = element.rx * vr + element.sx * vs - element.ry * ur - element.sy * us

        return vz

    def div(self, u: np.ndarray, v: np.ndarray, element: Element2D) -> np.ndarray:
        """
        Computes the divergence of the 2d vector field (`u`, `v`).

        Args:
            u (np.ndarray): First components of the vector field to get divergence of, as a 1d numpy array.
            v (np.ndarray): Second components of the vector field to get diviergence of, as a 1d numpy array.
            element (Element2D): The element who has state `u` and `v`.

        Returns:
            np.ndarray: The 1d numpy array `out`, where `out`[i] is the divergence of the 2d vector field (`u`[i], `v`[i]).
        """
        Dr, Ds = self._get_Dr_and_Ds_for_element(element)

        ur = Dr @ u
        us = Ds @ u
        vr = Dr @ v
        vs = Ds @ v

        ux = element.rx * ur + element.sx * us
        vy = element.ry * vr + element.sy * vs

        return ux + vy


class DerivativeOperatorDG2D(DerivativeOperator2D):
    def __init__(self, degree: int):
        super().__init__(degree)
        self._Dr, self._Ds = get_derivative_operators_2d(self.degree)

    def _get_Dr_and_Ds_for_element(self, _) -> Tuple[np.ndarray, np.ndarray]:
        return self._Dr, self._Ds
