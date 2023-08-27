import numpy as np

from pydglib.pde.advec2d import solve


class TestSolveAdvec2D:
    def test_matches_correct_solution(self):
        x0 = y0 = -1
        x1 = y1 = 1
        ax = 1
        ay = 1

        IC = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        def BC(x, t):
            x[:, 0] -= ax * t
            x[:, 1] -= ay * t
            return IC(x)

        final_time = 0.65
        n_elements = 8 * 8 * 2
        degree = 10

        soln, nodes = solve(
            x0, x1, y0, y1, ax, ay, IC, BC, final_time, n_elements, degree
        )

        def analytic_soln(x, t):
            x[:, 0] -= ax * t
            x[:, 1] -= ay * t
            return IC(x)

        X = nodes[:, :, 0].reshape(-1)
        Y = nodes[:, :, 1].reshape(-1)
        Z = soln[-1].reshape(-1)
        Z_actual = analytic_soln(np.vstack((X, Y)).T, final_time).reshape(-1)

        assert np.allclose(Z, Z_actual)
