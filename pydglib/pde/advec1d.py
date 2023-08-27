from typing import Callable, Tuple
import numpy as np

from pydglib.element import Element1D, InitialConditions
from pydglib.grid import Grid1D
from pydglib.mesh import meshgen1d
from pydglib.odeint import odeint
from pydglib.operators import DerivativeOperator1D, DerivativeOperatorDG1D, get_LIFT_1d


BoundaryConditions = Callable[[float], float]


def numerical_flux(
    interior: float, exterior: float, a: float, normal: int, alpha: float = 1
) -> float:
    assert normal in [-1, 1]
    assert 0 <= alpha <= 1

    avg = (interior + exterior) / 2
    jump_along_normal = normal * (exterior - interior)

    return a * avg + a * ((1 - alpha) / 2) * jump_along_normal


def get_state_at_left_edge(
    element: Element1D, time: float, a: float, BC
) -> Tuple[float, float]:
    internal_state = element.state[0]

    edge_is_boundary = element.is_leftmost

    if edge_is_boundary:
        inflow_boundary = a > 0

        if inflow_boundary:
            external_state = BC(element.nodes[0], time)
        else:
            external_state = internal_state
    else:
        external_state = element.left.state[-1]

    return internal_state, external_state


def get_state_at_right_edge(
    element: Element1D, time: float, a: float, BC
) -> Tuple[float, float]:
    internal_state = element.state[-1]

    edge_is_boundary = element.is_rightmost

    if edge_is_boundary:
        inflow_boundary = a < 0

        if inflow_boundary:
            external_state = BC(element.nodes[-1], time)
        else:
            external_state = internal_state
    else:
        external_state = element.right.state[0]

    return internal_state, external_state


def compute_surface_terms(u: Element1D, time, LIFT, a, BC) -> np.ndarray:
    internal_state, external_state = get_state_at_left_edge(u, time, a, BC)
    flux_l = numerical_flux(internal_state, external_state, a, -1)

    internal_state, external_state = get_state_at_right_edge(u, time, a, BC)
    flux_r = numerical_flux(internal_state, external_state, a, 1)

    du = np.array([-(a * u[0] - flux_l), a * u[-1] - flux_r])

    surface_terms = (2 / u.h) * LIFT @ du

    return surface_terms


def compute_gradient(
    element: Element1D,
    time: float,
    do: DerivativeOperator1D,
    LIFT: np.ndarray,
    a: float,
    BC: BoundaryConditions,
) -> np.ndarray:
    u = element.state
    ux = do(u, element)  # du/dx

    surface_terms = compute_surface_terms(element, time, LIFT, a, BC)

    dudt = -a * ux + surface_terms

    return dudt


def AdvecRHS1D(grid: Grid1D, time, do: DerivativeOperator1D, LIFT, a, BC):
    for element in grid.elements:
        dudt = compute_gradient(element, time, do, LIFT, a, BC)

        element.update_gradients(dudt)


def solve(
    xl: float,
    xr: float,
    a: float,
    IC: InitialConditions,
    BC: BoundaryConditions,
    final_time: float,
    n_elements: int,
    degree: int,
):
    VX, EToV = meshgen1d(xl, xr, n_elements)

    do = DerivativeOperatorDG1D(degree)
    LIFT = get_LIFT_1d(degree)

    grid = Grid1D(VX, EToV, degree, IC)

    dt = grid.get_time_step()

    soln = odeint(
        AdvecRHS1D,
        grid,
        final_time,
        dt,
        args=(do, LIFT, a, BC),
    )

    return soln, grid.nodes
