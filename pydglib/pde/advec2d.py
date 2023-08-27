import numpy as np
import pdb
from pydglib.element import Element2D, ElementEdge2D
from pydglib.grid import Grid2D
from pydglib.mesh import meshgen2d
from pydglib.odeint import odeint
from pydglib.operators import DerivativeOperator2D, DerivativeOperatorDG2D, get_LIFT_2d


def get_state_at_edge(edge: ElementEdge2D, time, a, BC):
    internal_state = edge.state

    if edge.is_boundary:
        inflow_boundary = np.dot(a, edge.normal) < 0

        if inflow_boundary:
            external_state = BC(edge.nodes, time)
        else:
            external_state = internal_state

    else:
        external_state = edge.external_state

    return internal_state, external_state


def compute_surface_terms(element: Element2D, time, LIFT, a, BC):
    surface_terms = np.zeros(element.n_nodes)

    for i, edge in enumerate(element.edges):
        internal_state, external_state = get_state_at_edge(edge, time, a, BC)
        g = (
            0.5
            * element.Fscale[i]
            * np.dot(a, edge.normal)
            * (internal_state - external_state)
        )
        surface_terms += LIFT[i] @ g

    return surface_terms


def compute_gradient(
    element: Element2D, time, do: DerivativeOperator2D, LIFT, a, BC
) -> np.ndarray:
    """
    Computes the gradient du/dt for the given element.
    """
    u = element.state
    ax, ay = a
    ux, uy = do.grad(u, element)

    surface_terms = compute_surface_terms(element, time, LIFT, a, BC)

    dudt = -ax * ux - ay * uy + surface_terms

    return dudt


def AdvecRHS2D(grid: Grid2D, time, do, LIFT, a, BC):
    for element in grid.elements:
        dudt = compute_gradient(element, time, do, LIFT, a, BC)

        element.update_gradients(dudt)


def solve(x0, x1, y0, y1, ax, ay, IC, BC, final_time, n_elements, degree):
    nx = int(np.sqrt(n_elements / 2))
    ny = int((n_elements / 2) / nx)

    VX, VY, EToV = meshgen2d(x0, x1, y0, y1, nx, ny)

    do = DerivativeOperatorDG2D(degree)
    LIFT = get_LIFT_2d(degree)

    grid = Grid2D(VX, VY, EToV, degree, IC)

    x = grid.nodes

    dt = grid.get_time_step()

    a = np.array([ax, ay])  # wave velocity

    sol = odeint(
        AdvecRHS2D,
        grid,
        final_time,
        dt,
        args=(do, LIFT, a, BC),
    )

    return sol, x
