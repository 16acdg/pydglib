import numpy as np
import pdb
from typing import Tuple

from pydglib.grid import Grid2D
from pydglib.mesh import meshgen2d
from pydglib.odeint import odeint
from pydglib.operators import get_derivative_operators_2d, get_LIFT_2d
from pydglib.element import Element2D


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


def GeometricFactors2D(element: Element2D):
    v1, v2, v3 = element.vertices
    xr, yr = (v2 - v1) / 2
    xs, ys = (v3 - v1) / 2
    J = xr * ys - xs * yr
    rx = ys / J
    ry = -xs / J
    sx = -yr / J
    sy = xr / J
    return rx, sx, ry, sy, J


def compute_surface_terms(element, LIFT, Dr, Ds, IDX):
    n_edge_nodes = element.degree + 1
    n_nodes = element.n_nodes

    # Extract state variables
    Hx = element.state[0]
    Hy = element.state[1]
    Ez = element.state[2]

    fluxHx = np.zeros((3, n_edge_nodes))
    fluxHy = np.zeros((3, n_edge_nodes))
    fluxEz = np.zeros((3, n_edge_nodes))

    if IDX % 2 == 0:
        Fscale = np.array([8, 8, 8 * np.sqrt(2)])
    else:
        Fscale = np.array([8 * np.sqrt(2), 8, 8])

    # Define field differences at faces
    for i, edge in enumerate(element.edges):
        if edge is None:  # physical boundary
            dHx = np.zeros(n_edge_nodes)
            dHy = np.zeros(n_edge_nodes)
            dEz = 2 * element.get_edge(i)[2]
        else:
            internal_state = element.get_edge(i)
            external_state = edge.get_external_state()
            dHx = internal_state[0] - external_state[0]
            dHy = internal_state[1] - external_state[1]
            dEz = internal_state[2] - external_state[2]

        ###############
        if i == 2:
            dHx = np.flip(dHx)
            dHy = np.flip(dHy)
            dEz = np.flip(dEz)
        ###############

        # Evaluate upwind fluxes
        nx, ny = element.normals[i]
        ndotdH = nx * dHx + ny * dHy
        fluxHx[i] = Fscale[i] * (ny * dEz + nx * ndotdH - dHx)
        fluxHy[i] = Fscale[i] * (-nx * dEz + ny * ndotdH - dHy)
        fluxEz[i] = Fscale[i] * (-nx * dHy + ny * dHx - dEz)

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
    for IDX, element in enumerate(grid.elements):
        rhsHx, rhsHy, rhsEz = compute_surface_terms(element, LIFT, Dr, Ds, IDX)
        element.grad[0] = rhsHx  # dHx/dt
        element.grad[1] = rhsHy  # dHy/dt
        element.grad[2] = rhsEz  # dEz/dt


def solve(x0, x1, y0, y1, IC, final_time, n_elements, degree):
    nx = int(np.sqrt(n_elements / 2))
    ny = int((n_elements / 2) / nx)

    VX, VY, EToV = meshgen2d(x0, x1, y0, y1, nx, ny)

    Dr, Ds = get_derivative_operators_2d(degree)
    LIFT = get_LIFT_2d(degree)

    grid = Grid2D(VX, VY, EToV, degree, IC)

    # Set geometric factors on elements
    for element in grid.elements:
        rx, sx, ry, sy, _ = GeometricFactors2D(element)
        element.rx = rx
        element.sx = sx
        element.ry = ry
        element.sy = sy

    x = grid.nodes

    # TODO: Determine time step from grid size
    dt = 0.0045

    sol = odeint(
        MaxwellRHS2D,
        grid,
        final_time,
        dt,
        args=(Dr, Ds, LIFT),
    )

    return sol, x
