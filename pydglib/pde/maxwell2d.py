import numpy as np
from typing import Tuple

from pydglib.grid import Grid2D
from pydglib.mesh import meshgen2d
from pydglib.odeint import odeint
from pydglib.operators import get_derivative_operators_2d, get_LIFT_2d
from pydglib.element import Element2D
from pydglib.utils.nodes import get_nodes_1d


def Curl2D(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    rx: float,
    sx: float,
    ry: float,
    sy: float,
    Dr,
    Ds,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2D curl-operator in the x-y plane.
    """
    uxr = Dr @ ux
    uxs = Ds @ ux
    uyr = Dr @ uy
    uys = Ds @ uy

    vz = rx * uyr + sx * uys - ry * uxr - sy * uxs

    if len(uz) > 0:
        uzr = Dr @ uz
        uzs = Ds @ uz
        vx = ry * uzr + sy * uzs
        vy = -rx * uzr - sx * uzs
    else:
        vx = []
        vy = []

    return vx, vy, vz


def Grad2D(
    u: np.ndarray, rx: float, sx: float, ry: float, sy: float, Dr, Ds
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the gradient [ux, uy] of the scalar field u.
    """
    ur = Dr @ u
    us = Ds @ u

    ux = rx * ur + sx * us
    uy = ry * ur + sy * us

    return ux, uy


def compute_gradients(element: Element2D, LIFT, Dr, Ds):
    n_edge_nodes = element.degree + 1
    n_nodes = element.n_nodes

    # Extract state variables
    Hx = element.state[:, 0]
    Hy = element.state[:, 1]
    Ez = element.state[:, 2]

    fluxHx = np.zeros((3, n_edge_nodes))
    fluxHy = np.zeros((3, n_edge_nodes))
    fluxEz = np.zeros((3, n_edge_nodes))

    # Define field differences at faces
    for i, edge in enumerate(element.edges):
        if edge is None:  # physical boundary
            dHx = np.zeros(n_edge_nodes)
            dHy = np.zeros(n_edge_nodes)
            dEz = 2 * element.get_edge(i)[:, 2]
        else:
            internal_state = element.get_edge(i)
            external_state = edge.get_external_state()
            dHx = internal_state[:, 0] - external_state[:, 0]
            dHy = internal_state[:, 1] - external_state[:, 1]
            dEz = internal_state[:, 2] - external_state[:, 2]

        # Evaluate upwind fluxes
        nx, ny = element.normals[i]
        ndotdH = nx * dHx + ny * dHy
        fluxHx[i] = element.Fscale[i] * (ny * dEz + nx * ndotdH - dHx)
        fluxHy[i] = element.Fscale[i] * (-nx * dEz + ny * ndotdH - dHy)
        fluxEz[i] = element.Fscale[i] * (-nx * dHy + ny * dHx - dEz)

    # local derivatives of fields
    Ezx, Ezy = Grad2D(Ez, element.rx, element.sx, element.ry, element.sy, Dr, Ds)
    _, _, CuHz = Curl2D(
        Hx, Hy, [], element.rx, element.sx, element.ry, element.sy, Dr, Ds
    )

    # Reshape flux vectors
    fluxHx = fluxHx.reshape(-1)
    fluxHy = fluxHy.reshape(-1)
    fluxEz = fluxEz.reshape(-1)

    # compute right hand sides of the PDE’s
    rhsHx = -Ezy + LIFT @ fluxHx / 2
    rhsHy = Ezx + LIFT @ fluxHy / 2
    rhsEz = CuHz + LIFT @ fluxEz / 2

    return rhsHx, rhsHy, rhsEz


def MaxwellRHS2D(grid: Grid2D, time, Dr, Ds, LIFT):
    for element in grid.elements:
        dHxdt, dHydt, dEzdt = compute_gradients(element, LIFT, Dr, Ds)
        element.update_gradients(dHxdt, dHydt, dEzdt)


def get_time_step(grid: Grid2D) -> float:
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

    Dr, Ds = get_derivative_operators_2d(degree)
    LIFT = get_LIFT_2d(degree)

    grid = Grid2D(VX, VY, EToV, degree, IC)

    x = grid.nodes

    dt = get_time_step(grid)

    sol = odeint(
        MaxwellRHS2D,
        grid,
        final_time,
        dt,
        args=(Dr, Ds, LIFT),
    )

    return sol, x
