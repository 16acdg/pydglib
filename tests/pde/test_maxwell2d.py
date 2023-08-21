import numpy as np

from pydglib.pde.maxwell2d import solve

from ..data import load


class TestSolve:
    def test_matches_correct_solution(self):
        # Physical domain: [x0, y0] x [x1, y1]
        x0 = y0 = -1
        x1 = y1 = 1

        # Initial conditions
        IC_Hx = lambda x: np.zeros(x.shape[0])
        IC_Hy = lambda x: np.zeros(x.shape[0])
        IC_Ez = lambda x: np.multiply(
            np.sin(np.pi * x[:, 0]), np.sin(np.pi * x[:, 1])
        )  # (x,y) -> sin(pi*x) * sin(pi*y)
        IC = [IC_Hx, IC_Hy, IC_Ez]

        final_time = 1

        # Discretization parameters
        n_elements = 128
        degree = 10

        # Solve using solver and extract final state
        soln, nodes = solve(x0, x1, y0, y1, IC, final_time, n_elements, degree)
        Hx_pred = soln[-1, 0]
        Hy_pred = soln[-1, 1]
        Ez_pred = soln[-1, 2]

        Hx_correct = load("maxwell2d_Hx_t1.npy")
        Hy_correct = load("maxwell2d_Hy_t1.npy")
        Ez_correct = load("maxwell2d_Ez_t1.npy")

        assert np.allclose(Hx_pred, Hx_correct)
        assert np.allclose(Hy_pred, Hy_correct)
        assert np.allclose(Ez_pred, Ez_correct)
