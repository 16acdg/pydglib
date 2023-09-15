from typing import Callable
import math
import numpy as np

from pydglib.element import Element1D, InitialConditions
from pydglib.grid import Grid1D
from pydglib.mesh import meshgen1d
from pydglib.odeint import odeint
from pydglib.operators import DerivativeOperator1D, DerivativeOperatorDG1D, get_LIFT_1d
from pydglib.utils.nodes import get_nodes_1d
from pydglib.utils.transformations import computational_to_physical_1d


def get_time_step(epsilon, xl, xr, n_elements, degree, u, final_time):
    r = get_nodes_1d(degree)
    h = (xr - xl) / n_elements
    x = computational_to_physical_1d(r, xl, xl + h)
    xmin = abs(x[0] - x[1])
    CFL = 0.25
    umax = np.max(np.abs(u))
    dt = CFL * min(xmin / umax, xmin**2 / np.sqrt(epsilon))
    n_steps = math.ceil(final_time / dt)
    dt = final_time / n_steps
    return dt


def get_du_left(u: Element1D, t, BC):
    internal_state = u[0]
    external_state = BC(u.nodes[0], t) if u.left is None else u.left[-1]
    return internal_state - external_state


def get_du_right(u: Element1D, t, BC):
    internal_state = u[-1]
    external_state = BC(u.nodes[-1], t) if u.right is None else u.right[0]
    return internal_state - external_state


def get_du(element: Element1D, t, BC):
    return np.array([-get_du_left(element, t, BC), get_du_right(element, t, BC)])


def compute_q(element: Element1D, time, do: DerivativeOperator1D, LIFT, epsilon, BC):
    du = get_du(element, time, BC)
    surface_terms = (1 / element.h) * LIFT @ du

    u = element.state
    ux = do(u, element)

    q = np.sqrt(epsilon) * (ux - surface_terms)

    return q


def get_dq_left(element: Element1D):
    if element.left is None:
        return 0
    else:
        return (element.q[0] - element.left.q[-1]) / 2


def get_dq_right(element: Element1D):
    if element.right is None:
        return 0
    else:
        return (element.q[-1] - element.right.q[0]) / 2


def get_dq(element: Element1D):
    return np.array([-get_dq_left(element), get_dq_right(element)])


def get_du2_left(u: Element1D, t, BC):
    internal_state = u[0]
    external_state = BC(u.nodes[0], t) if u.left is None else u.left[-1]
    return (internal_state**2 - external_state**2) / 2


def get_du2_right(u: Element1D, t, BC):
    internal_state = u[-1]
    external_state = BC(u.nodes[-1], t) if u.right is None else u.right[0]
    return (internal_state**2 - external_state**2) / 2


def get_du2(element: Element1D, t, BC):
    return np.array([-get_du2_left(element, t, BC), get_du2_right(element, t, BC)])


def compute_gradient(
    element: Element1D, time, do: DerivativeOperator1D, LIFT, epsilon, f, BC
):
    u = element.state
    q = element.q

    dq = get_dq(element)
    du2 = get_du2(element, time, BC)

    # flux term
    dflux = 0.5 * du2 - np.sqrt(epsilon) * dq

    maxvel = np.max(np.abs(u))

    du = get_du(element, time, BC)
    du[0] = -du[0]

    # flux term
    flux = dflux - 0.5 * maxvel * du

    # local derivatives of field
    dfdx = do(0.5 * u**2 - np.sqrt(epsilon) * q, element)

    # compute right hand sides of the semi-discrete PDE
    dudt = -dfdx + (2 / element.h) * LIFT @ flux

    # Apply source term if it is given
    if f is not None:
        dudt += f(element.nodes, time)

    return dudt


def BurgersRHS1D(grid: Grid1D, time, do: DerivativeOperator1D, LIFT, epsilon, f, BC):
    # Compute q and save to each element
    for element in grid.elements:
        q = compute_q(element, time, do, LIFT, epsilon, BC)
        element.q = q

    # Compute gradients and save gradient for each element
    for element in grid.elements:
        dudt = compute_gradient(element, time, do, LIFT, epsilon, f, BC)
        element.update_gradients(dudt)


def solve(
    diffusion: float = None,
    xl: float = None,
    xr: float = None,
    f: Callable[[np.ndarray, float], np.ndarray] = None,
    IC: InitialConditions = None,
    BC: Callable[[float, float], float] = None,
    final_time: float = None,
    n_elements: int = None,
    degree: int = None,
    do: DerivativeOperator1D = None,
    dt: float = None,
):
    """
    Solves the 1d Burgers' equation using nodal DG:

        u_t + uu_x - diffusion * u_xx = f(x,t)`

    Args:
        diffusion (float): Coefficient for the diffusion term. Must be positive.
        xl (float): Left boundary of the physical domain.
        xr (float): Right boundary of the physical domain.
        f (Callable[[np.ndarray, float], np.ndarray], optional): Source function f = f(x, t) for the Burgers' equation. Defaults to 0.
        IC (Callable[[np.ndarray], np.ndarray]): Initial conditions for the state.
        BC (Callable[[float, float], float], optional): Boundary conditions BC = BC(x, t). Defaults to periodic boundary conditions.
        final_time (float): Compute solution up to this time.
        n_elements (int): Number of uniformly sized elements to partition domain [`xl`, `xr`] into.
        degree (int): Degree of the local polynomial approximations for all elements.
        do (DerivativeOperator1D, optional): Derivative operator that predicts spatial derivatives. Defaults to using the DG derivative operator.
        dt (float, optional): Time step for time integration. Defaults to a good guess.

    Returns:
        np.ndarray: `soln`, a 3d numpy array of size (num time steps, `n_elements`, n_nodes) containing the solution.
        np.ndarray: `nodes`, a 2d numpy array of size (`n_elements`, n_nodes) containing the node positions.
    """
    # TODO: Write separate solver for the inviscid Burgers' equation
    assert diffusion > 0

    VX, EToV = meshgen1d(xl, xr, n_elements)

    if do is None:
        do = DerivativeOperatorDG1D(degree)

    LIFT = get_LIFT_1d(degree)

    grid = Grid1D(VX, EToV, degree, IC)

    # Impose periodic boundary conditions
    if BC is None:
        grid.elements[0].left = grid.elements[-1]
        grid.elements[-1].right = grid.elements[0]

    # Estimate good time step
    if dt is None:
        dt = get_time_step(
            diffusion, xl, xr, n_elements, degree, grid.state, final_time
        )

    soln = odeint(
        BurgersRHS1D,
        grid,
        final_time,
        dt,
        args=(do, LIFT, diffusion, f, BC),
    )

    return soln, grid.nodes
