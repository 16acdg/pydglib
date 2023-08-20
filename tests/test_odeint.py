import numpy as np
from scipy.integrate import odeint as scipy_odeint

from pydglib.grid import Grid1D
from pydglib.mesh import meshgen1d
from pydglib.odeint import odeint


def mock_system(grid: Grid1D, time: float):
    pass


def test_odeint_returns_3d_numpy_array_when_state_is_1d():
    xl = 0
    xr = 1
    n_elements = 3
    n_nodes = 4
    IC = lambda x: np.zeros_like(x)
    VX, EToV = meshgen1d(xl, xr, n_elements)
    grid = Grid1D(VX, EToV, n_nodes, IC)

    final_time = 1
    dt = 0.01
    soln = odeint(mock_system, grid, final_time, dt)

    assert isinstance(soln, np.ndarray)
    assert len(soln.shape) == 3
    assert soln.shape[1] == n_nodes
    assert soln.shape[2] == n_elements


def test_odeint_returns_4d_numpy_array_when_state_is_2d():
    xl = 0
    xr = 1
    n_elements = 3
    n_nodes = 4
    ICs = [
        lambda x: np.zeros_like(x),
        lambda x: np.zeros_like(x),
    ]
    VX, EToV = meshgen1d(xl, xr, n_elements)
    grid = Grid1D(VX, EToV, n_nodes, ICs)

    final_time = 1
    dt = 0.01
    soln = odeint(mock_system, grid, final_time, dt)

    assert isinstance(soln, np.ndarray)
    assert len(soln.shape) == 4
    assert soln.shape[1] == len(ICs)
    assert soln.shape[2] == n_nodes
    assert soln.shape[3] == n_elements


# From https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def scipy_pend(y, t, b, c):
    theta, omega = y
    dydt = np.array([omega, -b * omega - c * np.sin(theta)])
    return dydt


def pend1d(grid: Grid1D, t, b, c):
    u = grid.elements[0]  # Only one element needed for ODE
    theta, omega = u
    u.grad = np.array([[omega, -b * omega - c * np.sin(theta)]])


def test_odeint_solves_1d_pendulum_system():
    b = 0.25
    c = 5.0
    y0 = np.array([np.pi - 0.1, 0.0])
    final_time = 10
    dt = 0.1
    t = np.linspace(0, 10, int(final_time / dt) + 1)
    scipy_soln = scipy_odeint(scipy_pend, y0, t, args=(b, c))

    VX = np.array([0, 1])
    EToV = np.array([[0, 1]])
    n_nodes = 2
    IC = lambda _: y0
    grid = Grid1D(VX, EToV, n_nodes, IC)
    soln = odeint(pend1d, grid, final_time, dt, args=(b, c))[:, :, 0]

    assert scipy_soln.shape == soln.shape
    assert np.allclose(scipy_soln, soln, atol=1e-3, rtol=0)


def pend2d(grid: Grid1D, t, b0, c0, b1, c1):
    u = grid.elements[0]  # Only one element needed for ODE
    theta0, omega0 = u[0]
    theta1, omega1 = u[1]
    u.grad[0] = np.array([omega0, -b0 * omega0 - c0 * np.sin(theta0)])
    u.grad[1] = np.array([omega1, -b1 * omega1 - c1 * np.sin(theta1)])


def test_odeint_solves_multid_pendulum_system():
    # test that odeint works when the state dimension of a grid and its elements is greater than 1
    # (eg for maxwell's equations when there is a state vector for E and a state vector for H)
    b0 = 0.25
    c0 = 5.0
    b1 = 0.5
    c1 = 3
    y0 = np.array([np.pi - 0.1, 0.0])
    final_time = 10
    dt = 0.1
    t = np.linspace(0, 10, int(final_time / dt) + 1)
    scipy_soln0 = scipy_odeint(scipy_pend, y0, t, args=(b0, c0))
    scipy_soln1 = scipy_odeint(scipy_pend, y0, t, args=(b1, c1))

    VX = np.array([0, 1])
    EToV = np.array([[0, 1]])
    n_nodes = 2
    ICs = [
        lambda _: y0,
        lambda _: y0,
    ]
    grid = Grid1D(VX, EToV, n_nodes, ICs)
    soln = odeint(pend2d, grid, final_time, dt, args=(b0, c0, b1, c1))[:, :, :, 0]

    assert np.allclose(scipy_soln0, soln[:, 0, :], atol=1e-3, rtol=0)
    assert np.allclose(scipy_soln1, soln[:, 1, :], atol=1e-3, rtol=0)
