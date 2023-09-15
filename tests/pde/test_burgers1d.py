import numpy as np
import matplotlib.pyplot as plt


from pydglib.pde.burgers1d import solve


class TestSolverBurgers1D:
    def test_computes_travelling_wave_solution(self):
        epsilon = 0.1
        xl = -1
        xr = 1
        analytic_solution = lambda x, t: -np.tanh((x + 0.5 - t) / (2 * epsilon)) + 1
        IC = lambda x: analytic_solution(x, 0)
        BC = analytic_solution
        final_time = 1

        soln, nodes = solve(
            diffusion=epsilon,
            xl=xl,
            xr=xr,
            IC=IC,
            BC=BC,
            final_time=final_time,
            n_elements=8,
            degree=4,
        )

        true_soln = analytic_solution(nodes, final_time)

        assert np.allclose(soln[-1], true_soln, rtol=0, atol=1e-3)
