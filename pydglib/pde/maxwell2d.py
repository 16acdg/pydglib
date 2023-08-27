import numpy as np

from pydglib.grid import Grid2D
from pydglib.mesh import meshgen2d
from pydglib.odeint import odeint
from pydglib.operators import (
    get_LIFT_2d,
    DerivativeOperator2D,
    DerivativeOperatorDG2D,
)
from pydglib.element import Element2D


def compute_surface_terms(element: Element2D, LIFT):
    surfaceHx = np.zeros((element.n_faces, element.n_nodes))
    surfaceHy = np.zeros((element.n_faces, element.n_nodes))
    surfaceEz = np.zeros((element.n_faces, element.n_nodes))

    for i, edge in enumerate(element.edges):
        if edge.is_boundary:
            dHx = np.zeros(edge.n_nodes)
            dHy = np.zeros(edge.n_nodes)
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
        fluxHx = element.Fscale[i] * (ny * dEz + nx * ndotdH - dHx) / 2
        fluxHy = element.Fscale[i] * (-nx * dEz + ny * ndotdH - dHy) / 2
        fluxEz = element.Fscale[i] * (-nx * dHy + ny * dHx - dEz) / 2

        surfaceHx[i] = LIFT[i] @ fluxHx
        surfaceHy[i] = LIFT[i] @ fluxHy
        surfaceEz[i] = LIFT[i] @ fluxEz

    # Sum contributions from all 3 faces
    surfaceHx = np.sum(surfaceHx, axis=0)
    surfaceHy = np.sum(surfaceHy, axis=0)
    surfaceEz = np.sum(surfaceEz, axis=0)

    return surfaceHx, surfaceHy, surfaceEz


def compute_gradients(element: Element2D, do: DerivativeOperator2D, LIFT):
    # Extract state variables
    Hx = element.state[:, 0]
    Hy = element.state[:, 1]
    Ez = element.state[:, 2]

    surfaceHx, surfaceHy, surfaceEz = compute_surface_terms(element, LIFT)

    # local derivatives of fields
    Ezx, Ezy = do.grad(Ez, element)
    CuHz = do.curl(Hx, Hy, element)

    # compute right hand sides of the PDE’s
    rhsHx = -Ezy + surfaceHx
    rhsHy = Ezx + surfaceHy
    rhsEz = CuHz + surfaceEz

    return rhsHx, rhsHy, rhsEz


def MaxwellRHS2D(grid: Grid2D, time, do, LIFT):
    for element in grid.elements:
        dHxdt, dHydt, dEzdt = compute_gradients(element, do, LIFT)

        element.update_gradients(dHxdt, dHydt, dEzdt)


def solve(x0, x1, y0, y1, IC, final_time, n_elements, degree):
    nx = int(np.sqrt(n_elements / 2))
    ny = int((n_elements / 2) / nx)

    VX, VY, EToV = meshgen2d(x0, x1, y0, y1, nx, ny)

    do = DerivativeOperatorDG2D(degree)
    LIFT = get_LIFT_2d(degree)

    grid = Grid2D(VX, VY, EToV, degree, IC)

    x = grid.nodes

    dt = grid.get_time_step()

    sol = odeint(
        MaxwellRHS2D,
        grid,
        final_time,
        dt,
        args=(do, LIFT),
    )

    return sol, x
