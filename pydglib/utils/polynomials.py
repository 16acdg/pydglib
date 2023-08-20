from typing import Callable, List, Sequence, Union, Tuple
import numpy as np
from scipy.special import gamma


"""
Returns a float if the input is a float.
If the input is array like, then the polynomial is applied pointwise.
"""
Polynomial = Callable[[float | np.ndarray], float | np.ndarray]


def lagrange(nodes: np.ndarray) -> List[Polynomial]:
    """
    Returns the Lagrange interpolating polynomials for the given nodes.

    The given `nodes` must be a sequence of floats. The returned list has the same length as this array.
    The polynomial at position i in the output takes value 1 at `nodes`[i] and 0 at `nodes`[j] for j != i.

    Args:
        nodes (np.ndarray): Nodes to interpolate as a 1d numpy array.

    Returns:
        List[Polynomial]: Interpolating polynomials.
    """
    assert len(nodes) > 0
    assert len(nodes.shape) == 1
    assert np.unique(nodes).size == nodes.size

    def get_ith_polynomial(i: int, nodes: np.ndarray) -> Polynomial:
        @np.vectorize
        def p(x):
            out = 1
            for j in range(nodes.size):
                if i != j:
                    out *= (x - nodes[j]) / (nodes[i] - nodes[j])

            return out

        return p

    return [get_ith_polynomial(i, nodes) for i in range(nodes.size)]


def _eval_jacobi(x: np.ndarray, alpha: float, beta: float, degree: int) -> np.ndarray:
    """
    Evaluates the Jacobi polynomial of degree `degree` and parameters `alpha` and `beta` at `x`.

    This implementation is adopted from the Matlab implementation in Appendix A of Nodal DG (Hesthaven).

    Args:
        x (np.ndarray): Evalaute the Jacobi polynomial at these positions. Must be a 1d array.
        alpha (float): Alpha parameter of the Jacobi polynomial.
        beta (float): Beta parameter of the Jacobi polynomial.
        degree (int): The degree of the Jacobi polynomial.

    Returns:
        np.ndarray: The Jacobi polynomial evaluated at each `x` position.
            This is also a 1d numpy array with same length as `x`.
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1

    dims = x.size
    N = degree

    PL = np.zeros((N + 1, dims))

    # Initial values P_0(x) and P_1(x)
    gamma0 = (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    PL[0] = 1 / np.sqrt(gamma0) * np.ones(dims)

    if N == 0:
        return PL[0]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1] = (0.5 * (alpha + beta + 2) * x + 0.5 * (alpha - beta)) / np.sqrt(gamma1)

    if N == 1:
        return PL[1]

    # Repeat value in recurrence.
    aold = (
        2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    )

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        bnew = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        PL[i + 1] = 1 / anew * (-aold * PL[i - 1] + np.multiply(x - bnew, PL[i]))
        aold = anew

    P = PL[-1]

    return P


def _eval_jacobi_deriv(
    x: np.ndarray, alpha: float, beta: float, degree: int
) -> np.ndarray:
    """
    Evaluates the derivative of the Jacobi polynomial of degree `degree` and parameters `alpha` and `beta` at `x`.

    This implementation is adopted from the Matlab implementation in Appendix A of Nodal DG (Hesthaven).

    Args:
        x (np.ndarray): Evalaute the Jacobi polynomial's derivative at these positions. Must be a 1d array.
        alpha (float): Alpha parameter of the Jacobi polynomial.
        beta (float): Beta parameter of the Jacobi polynomial.
        degree (int): The degree of the Jacobi polynomial.

    Returns:
        np.ndarray: The derivative of the Jacobi polynomial evaluated at each `x` position.
            This is also a 1d numpy array with same length as `x`.
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1

    dP = np.zeros(x.size)

    if degree > 0:
        dP = np.sqrt(degree * (degree + alpha + beta + 1)) * _eval_jacobi(
            x, alpha + 1, beta + 1, degree - 1
        )

    return dP


def jacobi(alpha: float, beta: float, degree: int) -> Polynomial:
    """
    Returns the Jacobi polynomial of given degree with parameters `alpha` and `beta`.

    Args:
        alpha (float): Alpha parameter of the Jacobi polynomial.
        beta (float): Beta parameter of the Jacobi polynomial.
        degree (int): The degree of the Jacobi polynomial.

    Returns:
        Polynomial: The Jacobi polynomial as a python function.
    """
    assert alpha > -1
    assert beta > -1
    assert degree >= 0
    assert int(degree) - degree == 0

    @np.vectorize
    def p(x):
        if isinstance(x, np.ndarray):
            x_shape = x.shape
            x = x.reshape(-1)
            y = _eval_jacobi(x, alpha, beta, degree)
            y = y.reshape(x_shape)
            return y
        else:  # x is assumed to be float
            x = np.array([x])
            y = _eval_jacobi(x, alpha, beta, degree)
            return float(y[0])

    return p


def legendre(degree: int) -> Polynomial:
    """
    Returns the Legendre polynomial of given degree.

    Args:
        degree (int): The degree of the Legendre polynomial.

    Returns:
        Polynomial: The Legendre polynomial as a python function.
    """
    return jacobi(0, 0, degree)


def jacobi_deriv(alpha: float, beta: float, degree: int) -> Polynomial:
    """
    Returns the derivative of the Jacobi polynomial of given degree with parameters `alpha` and `beta`.

    Args:
        alpha (float): Alpha parameter of the Jacobi polynomial.
        beta (float): Beta parameter of the Jacobi polynomial.
        degree (int): The degree of the Jacobi polynomial.

    Returns:
        Polynomial: The derivative of the Jacobi polynomial as a python function.
    """
    assert alpha > -1
    assert beta > -1
    assert degree >= 0
    assert int(degree) - degree == 0

    def dp(x):
        if isinstance(x, np.ndarray):
            x_shape = x.shape
            x = x.reshape(-1)
            y = _eval_jacobi_deriv(x, alpha, beta, degree)
            y = y.reshape(x_shape)
            return y
        else:  # x is assumed to be float
            x = np.array([x])
            y = _eval_jacobi_deriv(x, alpha, beta, degree)
            return float(y[0])

    return dp


def legendre_deriv(degree: int) -> Polynomial:
    """
    Returns the derivative of the Legendre polynomial of given degree.

    Args:
        degree (int): The degree of the Legendre polynomial.

    Returns:
        Polynomial: The derivative of the Legendre polynomial as a python function.
    """
    return jacobi_deriv(0, 0, degree)
