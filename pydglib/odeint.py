from typing import Tuple
import numpy as np
from tqdm import tqdm

from .grid import Grid1D

# Low storage RK coefficients
rk4a = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
]
rk4b = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
]
rk4c = [
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
]


def odeint(
    sys, grid: Grid1D, final_time: float, dt: float, args: Tuple[any] = ()
) -> np.ndarray:
    """
    Integrates the ODE system dy/dt = sys(y,t).

    Uses a Runge-Kutta method

    Args:
        sys (function): Right-hand-side of the ODE.
        grid (Grid1D): Grid on which to compute and update solution
        final_time (int): Rime to integrate until.
        dt (float): Time step.
        args (Tuple[any], optional): Arguments to pass to sys. Defaults to ().

    Returns:
        np.ndarray: 3d numpy array of the solution at each time step.
    """
    time = 0  # running time
    nt = int(final_time / dt) + 1

    # Set shape of storage arrays based on dimensionality of grid state
    if grid.state_dimension == 1:
        resu_shape = (grid.n_elements, grid.n_nodes)
        soln_shape = (nt, grid.n_nodes, grid.n_elements)
    else:
        resu_shape = (grid.n_elements, grid.state_dimension, grid.n_nodes)
        soln_shape = (nt, grid.state_dimension, grid.n_nodes, grid.n_elements)

    # RK residual storage
    resu = np.zeros(resu_shape, dtype=np.float64)

    # save solution for each time step
    soln = np.zeros(soln_shape, dtype=np.float64)

    # Save initial conditions to solution array
    soln[0] = grid.state

    # outer time loop
    for tstep in tqdm(range(1, nt)):
        for INTRK in range(5):
            time_local = time + rk4c[INTRK] * dt

            # Update gradients
            sys(grid, time_local, *args)

            # Update state
            for k, u in enumerate(grid.elements):
                resu[k] = rk4a[INTRK] * resu[k] + dt * u.grad
                u += rk4b[INTRK] * resu[k]

        time += dt
        soln[tstep] = grid.state

    return soln
