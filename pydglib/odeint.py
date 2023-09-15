from typing import Tuple, Callable
import numpy as np
from tqdm import tqdm
import math

from pydglib.grid import Grid

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
    sys: Callable,
    grid: Grid,
    final_time: float,
    dt: float,
    args: Tuple[any] = (),
    cache_time_derivatives: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Integrates the ODE system dy/dt = sys(y,t).

    Uses RK4 integration.

    Args:
        sys (Callable): Right-hand-side of the ODE.
        grid (Grid): Grid on which to compute and update solution
        final_time (float): Time to integrate until.
        dt (float): Time step.
        args (Tuple[any], optional): Arguments to pass to `sys`. Defaults to ().
        cache_time_derivatives (bool, optional): If True, this function will also return time derivatives for each time step. Defaults to False.

    Returns:
        np.ndarray: 3d or 4d numpy array of the solution at each time step.
            If the `state_dimension` = 1, then shape = (num time steps, n_elements, n_nodes`, `state_dimension`).
            If `state_dimension` > 1, then shape = (num time steps, `n_elements`, `n_nodes`, `state_dimension`).
    """
    time = 0  # running time
    nt = math.ceil(final_time / dt)

    # RK residual storage
    resu = np.zeros(grid.shape)

    # save solution for each time step
    soln = np.zeros((nt + 1, *grid.shape))

    # Save initial conditions to solution array
    soln[0] = grid.state

    # Storage for time derivatives
    if cache_time_derivatives:
        dudt = np.zeros((nt, *grid.shape))

    for tstep in tqdm(range(1, nt + 1)):
        # Shrink the final time step to match the time interval
        if 0 < final_time - (dt * (tstep - 1)) < dt:
            dt = final_time - (dt * (tstep - 1))

        for INTRK in range(5):
            time_local = time + rk4c[INTRK] * dt

            # Update gradients
            sys(grid, time_local, *args)

            # Cache time derivatives on first RK4 step
            if cache_time_derivatives and INTRK == 0:
                dudt[tstep - 1] = grid.grad

            # Update state
            for k, u in enumerate(grid.elements):
                resu[k] = rk4a[INTRK] * resu[k] + dt * u.grad
                u += rk4b[INTRK] * resu[k]

        time += dt
        soln[tstep] = grid.state

    if cache_time_derivatives:
        return soln, dudt

    return soln
