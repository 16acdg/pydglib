from typing import Callable
import pytest
import numpy as np

from pydglib.utils.polynomials import (
    lagrange,
    jacobi,
    legendre,
    jacobi_deriv,
    legendre_deriv,
)


def is_zero(p) -> bool:
    dx = 0.1
    for x in np.arange(-1, 1, dx):
        if p(x) != 0:
            return False
    return True


def is_constant(p) -> bool:
    dx = 0.1
    for x in np.arange(-1, 1, dx):
        if p(x) != p(x + dx):
            return False
    return True


def is_linear(p) -> bool:
    dx = 0.1
    for x in np.arange(-1, 1, dx):
        if not np.isclose(p(x + dx) - p(x), p(x) - p(x - dx)):
            return False
    return True


class TestLagrange:
    valid_node_inputs = [
        np.array([0, 0.5, 1]),
        np.array([1, 0, -1.5]),
        np.linspace(3, 5, 7),
        np.array([4]),
    ]

    def test_raises_exception_when_input_is_empty(self):
        nodes = np.array([])
        with pytest.raises(Exception):
            _ = lagrange(nodes)

    @pytest.mark.parametrize(
        "nodes", [np.array([[0, 1, 2]]), np.array([[0, 1, 2], [3, 4, 5]])]
    )
    def test_raises_exception_when_input_is_multid_numpy_array(self, nodes):
        with pytest.raises(Exception):
            _ = lagrange(nodes)

    def test_raises_exception_when_input_nodes_are_repeated(self):
        nodes = np.array([0, 1, 1])
        with pytest.raises(Exception):
            _ = lagrange(nodes)

    @pytest.mark.parametrize("nodes", [(), []])
    def test_raises_exception_when_input_sequence_is_empty(self, nodes):
        with pytest.raises(Exception):
            _ = lagrange(nodes)

    @pytest.mark.parametrize("nodes", valid_node_inputs)
    def test_returns_list_of_functions(self, nodes):
        polynomials = lagrange(nodes)
        for p in polynomials:
            assert callable(p)

    @pytest.mark.parametrize("nodes", valid_node_inputs)
    def test_returned_functions_return_scalar_when_input_is_scalar(self, nodes):
        polynomials = lagrange(nodes)
        for p in polynomials:
            for x in [-1, 0, 4.3]:
                y = p(x)
                # Need to multiply by 1 because this transforms y from a 0d numpy array to a scalar.
                assert np.isscalar(y * 1)

    @pytest.mark.parametrize("nodes", valid_node_inputs)
    def test_returned_functions_apply_pointwise_when_input_is_numpy_array(self, nodes):
        polynomials = lagrange(nodes)
        for p in polynomials:
            for x in [np.random.random(4), np.random.random((3, 3, 3))]:
                y = p(x)
                assert isinstance(y, np.ndarray)
                assert y.shape == x.shape

    @pytest.mark.parametrize("nodes", valid_node_inputs)
    def test_returned_functions_interpolate_nodes(self, nodes):
        polynomials = lagrange(nodes)
        for i, p in enumerate(polynomials):
            for j, x in enumerate(nodes):
                assert p(x) == (1 if i == j else 0)


def l2_inner_product(f, g, alpha=0, beta=0) -> float:
    """Returns the L2 inner product over the interval [-1, 1] with weighting (1-x)^alpha(1+x)^beta."""
    xl, xr = -1, 1
    dx = 0.01
    out = 0
    for i in range(int((xr - xl) / dx)):
        x = xl + (i + 0.5) * dx
        out += (1 - x) ** alpha * (1 + x) ** beta * f(x) * g(x) * dx
    return out


class TestJacobi:
    @pytest.mark.parametrize("alpha", [-5, -1])
    def test_raises_exception_for_invalid_alpha(self, alpha):
        beta = 0
        degree = 2
        with pytest.raises(Exception):
            _ = jacobi(alpha, beta, degree)

    @pytest.mark.parametrize("beta", [-3, -1])
    def test_raises_exception_for_invalid_beta(self, beta):
        alpha = 0
        degree = 2
        with pytest.raises(Exception):
            _ = jacobi(alpha, beta, degree)

    @pytest.mark.parametrize("degree", [-1, -0.4, 0.3, 2.999])
    def test_raises_exception_for_invalid_degree(self, degree):
        alpha = 1
        beta = 1.3
        with pytest.raises(Exception):
            _ = jacobi(alpha, beta, degree)

    def test_returns_function(self):
        alpha = 1
        beta = 2
        degree = 3
        p = jacobi(alpha, beta, degree)
        assert callable(p)

    def test_returned_functions_return_scalar_when_input_is_scalar(self):
        alpha = 1
        beta = 2
        degree = 3
        p = jacobi(alpha, beta, degree)
        x = 4.5
        assert np.isscalar(p(x) * 1)

    def test_returned_functions_apply_pointwise_when_input_is_numpy_array(self):
        alpha = 1
        beta = 2
        degree = 3
        p = jacobi(alpha, beta, degree)
        x = np.random.random((3, 2, 1))
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert y[0, 0, 0] == p(x[0, 0, 0])

    def test_returned_functions_are_mutually_orthogonal(self):
        alpha = 2
        beta = 1
        polynomials = [jacobi(alpha, beta, degree) for degree in range(3)]
        for i, p in enumerate(polynomials):
            for j, q in enumerate(polynomials):
                if i != j:
                    assert np.isclose(
                        l2_inner_product(p, q, alpha, beta), 0, rtol=0, atol=1e-3
                    )

    def test_returned_functions_have_unit_norm(self):
        alpha = 1
        beta = 2
        polynomials = [jacobi(alpha, beta, degree) for degree in range(5)]
        for p in polynomials:
            assert np.isclose(l2_inner_product(p, p, alpha, beta), 1, rtol=0, atol=1e-3)

    @pytest.mark.parametrize("alpha,beta", [(1, 1), (1.3, -0.2), (0, 0.4), (0.6, 0)])
    def test_zeroth_degree_polynomial_is_constant(self, alpha, beta):
        degree = 0
        p = jacobi(alpha, beta, degree)
        assert is_constant(p)

    @pytest.mark.parametrize("alpha,beta", [(1, 1), (1.3, -0.2), (0, 0.4), (0.6, 0)])
    def test_first_degree_polynomial_is_linear(self, alpha, beta):
        degree = 1
        p = jacobi(alpha, beta, degree)
        assert is_linear(p)

    @pytest.mark.parametrize(
        "x,alpha,beta,degree,expected",
        [
            (0, 1, 0, 1, 0.5),
            (0, 0, 1, 1, -0.5),
            (0.92, 1.3, 1.2, 0, 0.894305587779304),
            (1.08, -0.43, 0.76, 4, 1.836771044721720),
            (0.05, 1, 2, 2, -0.741137085624178),
            (-0.75, 0.5, -0.5, 3, 0.493665885604287),
        ],
    )
    def test_returned_functions_output_correct_values(
        self, x, alpha, beta, degree, expected
    ):
        p = jacobi(alpha, beta, degree)
        assert np.isclose(p(x), expected)


class TestLegendre:
    @pytest.mark.parametrize("degree", [-1, -0.4, 0.3, 2.999])
    def test_raises_exception_for_invalid_degree(self, degree):
        with pytest.raises(Exception):
            _ = legendre(degree)

    def test_returns_function(self):
        degree = 3
        p = legendre(degree)
        assert callable(p)

    def test_returned_functions_return_scalar_when_input_is_scalar(self):
        degree = 3
        p = legendre(degree)
        x = 4.5
        assert np.isscalar(p(x) * 1)

    def test_returned_functions_apply_pointwise_when_input_is_numpy_array(self):
        degree = 3
        p = legendre(degree)
        x = np.random.random((3, 2, 1))
        y = p(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert y[0, 0, 0] == p(x[0, 0, 0])

    def test_returned_functions_are_mutually_orthogonal(self):
        polynomials = [legendre(degree) for degree in range(3)]
        for i, p in enumerate(polynomials):
            for j, q in enumerate(polynomials):
                if i != j:
                    assert np.isclose(l2_inner_product(p, q), 0, rtol=0, atol=1e-3)

    def test_returned_functions_have_unit_norm(self):
        polynomials = [legendre(degree) for degree in range(5)]
        for p in polynomials:
            assert np.isclose(l2_inner_product(p, p), 1, rtol=0, atol=1e-3)

    def test_zeroth_degree_polynomial_is_constant(self):
        degree = 0
        p = legendre(degree)
        assert is_constant(p)

    def test_first_degree_polynomial_is_linear(self):
        degree = 1
        p = legendre(degree)
        assert is_linear(p)

    @pytest.mark.parametrize(
        "x,degree,expected",
        [
            (1.04, 0, 1 / np.sqrt(2)),
            (0, 1, 0),
            (1, 1, 1.224744871391589),
            (2, 1, 2.449489742783),
            (-1, 1, -1.224744871391589),
            (-2, 1, -2.449489742783178),
            (0, 2, -0.790569415042095),
            (1.4, 3, 8.905144580521981),
            (-0.6, 4, -0.865498700172334),
            (0.99, 5, 2.005575796321654),
            (0, 6, -0.796721798998873),
        ],
    )
    def test_returned_functions_output_correct_values(self, x, degree, expected):
        p = legendre(degree)
        assert np.isclose(p(x), expected)


class TestJacobiDeriv:
    @pytest.mark.parametrize("alpha", [-5, -1])
    def test_raises_exception_for_invalid_alpha(self, alpha):
        beta = 0
        degree = 2
        with pytest.raises(Exception):
            _ = jacobi_deriv(alpha, beta, degree)

    @pytest.mark.parametrize("beta", [-3, -1])
    def test_raises_exception_for_invalid_beta(self, beta):
        alpha = 0
        degree = 2
        with pytest.raises(Exception):
            _ = jacobi_deriv(alpha, beta, degree)

    @pytest.mark.parametrize("degree", [-1, -0.4, 0.3, 2.999])
    def test_raises_exception_for_invalid_degree(self, degree):
        alpha = 1
        beta = 1.3
        with pytest.raises(Exception):
            _ = jacobi_deriv(alpha, beta, degree)

    def test_returns_function(self):
        alpha = 1
        beta = 2
        degree = 3
        dp = jacobi_deriv(alpha, beta, degree)
        assert callable(dp)

    def test_returned_functions_return_scalar_when_input_is_scalar(self):
        alpha = 1
        beta = 2
        degree = 3
        dp = jacobi_deriv(alpha, beta, degree)
        x = 4.5
        assert np.isscalar(dp(x) * 1)

    def test_returned_functions_apply_pointwise_when_input_is_numpy_array(self):
        alpha = 1
        beta = 2
        degree = 3
        dp = jacobi_deriv(alpha, beta, degree)
        x = np.random.random((3, 2, 1))
        y = dp(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert y[0, 0, 0] == dp(x[0, 0, 0])

    def test_derivative_of_zeroth_degree_polynomial_is_zero(self):
        alpha = 0
        beta = 1
        degree = 0
        dp = jacobi_deriv(alpha, beta, degree)
        assert is_zero(dp)

    def test_derivative_of_first_degree_polynomial_is_constant(self):
        alpha = 1
        beta = 1
        degree = 1
        dp = jacobi_deriv(alpha, beta, degree)
        assert is_constant(dp)

    def test_derivative_of_second_degree_polynomial_is_linear(self):
        alpha = 1.7
        beta = np.pi / 2
        degree = 2
        dp = jacobi_deriv(alpha, beta, degree)
        assert is_linear(dp)

    @pytest.mark.parametrize(
        "x,alpha,beta,degree,expected",
        [
            (0.9, 0.8, 0.01, 0, 0),
            (0.4, -0.5, 0.6, 0, 0),
            (0, 1, 0, 1, 1.5),
            (0, 0, 1, 1, 1.5),
            (1.08, -0.43, 0.76, 4, 18.337188342480289),
            (0.05, 1, 2, 2, -0.890049155945895),
            (-0.75, 0.5, -0.5, 3, 1.974663542417146),
        ],
    )
    def test_returned_functions_output_correct_values(
        self, x, alpha, beta, degree, expected
    ):
        dp = jacobi_deriv(alpha, beta, degree)
        assert np.isclose(dp(x), expected)


class TestLegendreDeriv:
    @pytest.mark.parametrize("degree", [-1, -0.4, 0.3, 2.999])
    def test_raises_exception_for_invalid_degree(self, degree):
        with pytest.raises(Exception):
            _ = legendre_deriv(degree)

    def test_returns_function(self):
        degree = 3
        dp = legendre_deriv(degree)
        assert callable(dp)

    def test_returned_functions_return_scalar_when_input_is_scalar(self):
        degree = 3
        dp = legendre_deriv(degree)
        x = 4.5
        assert np.isscalar(dp(x) * 1)

    def test_returned_functions_apply_pointwise_when_input_is_numpy_array(self):
        degree = 3
        dp = legendre_deriv(degree)
        x = np.random.random((3, 2, 1))
        y = dp(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
        assert y[0, 0, 0] == dp(x[0, 0, 0])

    def test_derivative_of_zeroth_degree_polynomial_is_zero(self):
        degree = 0
        dp = legendre_deriv(degree)
        assert is_zero(dp)

    def test_derivative_of_first_degree_polynomial_is_constant(self):
        degree = 1
        dp = legendre_deriv(degree)
        assert is_constant(dp)

    def test_derivative_of_second_degree_polynomial_is_linear(self):
        degree = 2
        dp = legendre_deriv(degree)
        assert is_linear(dp)

    @pytest.mark.parametrize(
        "x,degree,expected",
        [
            (-0.543, 0, 0),
            (0.15, 1, 1.224744871391589),
            (0.9, 2, 4.269074841227313),
            (1.4, 3, 24.694938752708012),
            (-0.6, 4, 1.527350647362943),
            (0.99, 5, 32.764530843431075),
            (0, 6, 0),
        ],
    )
    def test_returned_functions_output_correct_values(self, x, degree, expected):
        dp = legendre_deriv(degree)
        assert np.isclose(dp(x), expected)
