import numpy as np

from pydglib.pde.advec1d import solve


class TestSolveAdvec1D:
    def test_numerical_solution_matches_analytic_solution_for_positive_wave_speed(self):
        a = 2 * np.pi
        IC = lambda x: np.sin(x)
        BC = lambda x, t: -np.sin(2 * np.pi * t)
        xl = 0
        xr = 2 * np.pi
        final_time = 0.7
        n_elements = 10
        degree = 9

        numerical_soln, x = solve(xl, xr, a, IC, BC, final_time, n_elements, degree)

        analytic_soln = np.sin(x - a * final_time)

        assert np.allclose(numerical_soln[-1], analytic_soln, atol=1e-2, rtol=0)

    def test_numerical_solution_matches_analytic_solution_for_negative_wave_speed(self):
        a = -2 * np.pi
        IC = lambda x: np.sin(x)
        BC = lambda x, t: np.sin(2 * np.pi * t)
        xl = 0
        xr = 2 * np.pi
        final_time = 0.7
        n_elements = 10
        degree = 9

        numerical_soln, x = solve(xl, xr, a, IC, BC, final_time, n_elements, degree)

        analytic_soln = np.sin(x - a * final_time)

        assert np.allclose(numerical_soln[-1], analytic_soln, atol=1e-2, rtol=0)
