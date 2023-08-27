import numpy as np
from typing import Tuple

from pydglib.grid import Grid2D
from pydglib.mesh import meshgen2d
from pydglib.odeint import odeint
from pydglib.operators import (
    get_LIFT_2d,
    DerivativeOperator2D,
    DerivativeOperatorDG2D,
)
from pydglib.element import Element2D
from pydglib.utils.nodes import get_nodes_1d


def compute_gradients(element: Element2D, do: DerivativeOperator2D, LIFT):
    n_edge_nodes = element.degree + 1

    # Extract state variables
    Hx = element.state[:, 0]
    Hy = element.state[:, 1]
    Ez = element.state[:, 2]

    fluxHx = np.zeros((element.n_faces, n_edge_nodes))
    fluxHy = np.zeros((element.n_faces, n_edge_nodes))
    fluxEz = np.zeros((element.n_faces, n_edge_nodes))

    # Define field differences at faces
    for i, edge in enumerate(element.edges):
        if edge.is_boundary:
            dHx = np.zeros(n_edge_nodes)
            dHy = np.zeros(n_edge_nodes)
            dEz = 2 * edge.state[:, 2]
        else:
            internal_state = edge.state
            external_state = edge.external_state
            dHx = internal_state[:, 0] - external_state[:, 0]
            dHy = internal_state[:, 1] - external_state[:, 1]
            dEz = internal_state[:, 2] - external_state[:, 2]

        # Evaluate upwind fluxes
        nx, ny = edge.normal
        ndotdH = nx * dHx + ny * dHy
        fluxHx[i] = element.Fscale[i] * (ny * dEz + nx * ndotdH - dHx)
        fluxHy[i] = element.Fscale[i] * (-nx * dEz + ny * ndotdH - dHy)
        fluxEz[i] = element.Fscale[i] * (-nx * dHy + ny * dHx - dEz)

    # local derivatives of fields
    Ezx, Ezy = do.grad(Ez, element)
    CuHz = do.curl(Hx, Hy, element)

    # Reshape flux vectors
    fluxHx = fluxHx.reshape(-1)
    fluxHy = fluxHy.reshape(-1)
    fluxEz = fluxEz.reshape(-1)

    # compute right hand sides of the PDE’s
    rhsHx = -Ezy + LIFT @ fluxHx / 2
    rhsHy = Ezx + LIFT @ fluxHy / 2
    rhsEz = CuHz + LIFT @ fluxEz / 2

    return rhsHx, rhsHy, rhsEz


def MaxwellRHS2D(grid: Grid2D, time, do, LIFT):
    for element in grid.elements:
        dHxdt, dHydt, dEzdt = compute_gradients(element, do, LIFT)

        element.update_gradients(dHxdt, dHydt, dEzdt)


def get_time_step(grid: Grid2D) -> float:
    # TODO: Move to abstract method on Grid
    rLGL = get_nodes_1d(grid.degree)
    rmin = abs(rLGL[0] - rLGL[1])

    dtscale = np.zeros(len(grid.elements))
    for i, element in enumerate(grid.elements):
        inscribed_radius = 2 * element.area / element.perimeter
        dtscale[i] = inscribed_radius

    dt = np.min(dtscale) * rmin * 2 / 3

    return dt


def solve(x0, x1, y0, y1, IC, final_time, n_elements, degree):
    nx = int(np.sqrt(n_elements / 2))
    ny = int((n_elements / 2) / nx)

    VX, VY, EToV = meshgen2d(x0, x1, y0, y1, nx, ny)

    do = DerivativeOperatorDG2D(degree)
    LIFT = get_LIFT_2d(degree)

    grid = Grid2D(VX, VY, EToV, degree, IC)

    x = grid.nodes

    dt = get_time_step(grid)

    sol = odeint(
        MaxwellRHS2D,
        grid,
        final_time,
        dt,
        args=(do, LIFT),
    )

    return sol, x
