import os
import numpy as np
import pytest

from pydglib.operators import (
    _get_V,
    get_derivative_operator_1d,
    get_LIFT_1d,
    get_derivative_operators_2d,
    get_edge_mass_matrix_2d,
    get_LIFT_2d,
    _get_orthonormal_poly_basis_1d,
    _get_orthonormal_poly_basis_2d,
)
from pydglib.utils.nodes import get_nodes_1d, get_nodes_2d
from .data import load


class TestGetV:
    def test_returns_square_numpy_array_for_1d_polynomials(self):
        degree = 7
        n_nodes = degree + 1
        nodes = get_nodes_1d(degree)
        P, _ = _get_orthonormal_poly_basis_1d(degree)
        V = _get_V(nodes, P)
        assert isinstance(V, np.ndarray)
        assert len(V.shape) == 2
        assert V.shape[0] == n_nodes
        assert V.shape[1] == n_nodes

    def test_returns_square_numpy_array_for_2d_polynomials(self):
        degree = 4
        nodes = get_nodes_2d(degree)
        n_nodes = nodes.shape[0]
        P, _, _ = _get_orthonormal_poly_basis_2d(degree)
        V = _get_V(nodes, P)
        assert isinstance(V, np.ndarray)
        assert len(V.shape) == 2
        assert V.shape[0] == n_nodes
        assert V.shape[1] == n_nodes

    def test_returns_correct_V_array_for_2d_polynomials_of_degree_2(self):
        degree = 2
        nodes = get_nodes_2d(degree)
        P, _, _ = _get_orthonormal_poly_basis_2d(degree)
        V = _get_V(nodes, P)
        V_actual = np.array(
            [
                [
                    np.sqrt(2) / 2,
                    -1,
                    np.sqrt(3 / 2),
                    -np.sqrt(3),
                    np.sqrt(9 / 2),
                    np.sqrt(15 / 2),
                ],
                [np.sqrt(2) / 2, -1, np.sqrt(3 / 2), 0, 0, -np.sqrt(15 / 2) / 2],
                [
                    np.sqrt(2) / 2,
                    -1,
                    np.sqrt(3 / 2),
                    np.sqrt(3),
                    -np.sqrt(9 / 2),
                    np.sqrt(15 / 2),
                ],
                [
                    np.sqrt(2) / 2,
                    0.5,
                    -np.sqrt(3 / 2) / 2,
                    -np.sqrt(3) / 2,
                    -np.sqrt(9 / 2) * 0.75,
                    np.sqrt(15 / 2) / 4,
                ],
                [
                    np.sqrt(2) / 2,
                    0.5,
                    -np.sqrt(3 / 2) / 2,
                    np.sqrt(3) / 2,
                    np.sqrt(9 / 2) * 0.75,
                    np.sqrt(15 / 2) / 4,
                ],
                [np.sqrt(2) / 2, 2, 3 * np.sqrt(3 / 2), 0, 0, 0],
            ]
        )
        assert np.allclose(V, V_actual)

    def test_returns_correct_Vr_array_for_2d_polynomials_of_degree_2(self):
        degree = 2
        nodes = get_nodes_2d(degree)
        _, Pr, _ = _get_orthonormal_poly_basis_2d(degree)
        Vr = _get_V(nodes, Pr)
        Vr_actual = np.array(
            [
                [0, 0, 0, np.sqrt(3), -np.sqrt(9 / 2), -np.sqrt(135 / 2)],
                [0, 0, 0, np.sqrt(3), -np.sqrt(9 / 2), 0],
                [0, 0, 0, np.sqrt(3), -np.sqrt(9 / 2), np.sqrt(135 / 2)],
                [0, 0, 0, np.sqrt(3), np.sqrt(9 / 2) * 3 / 2, -np.sqrt(135 / 2) / 2],
                [0, 0, 0, np.sqrt(3), np.sqrt(9 / 2) * 3 / 2, np.sqrt(135 / 2) / 2],
                [0, 0, 0, np.sqrt(3), np.sqrt(9 / 2) * 4, 0],
            ]
        )
        assert np.allclose(Vr, Vr_actual)

    def test_returns_correct_Vs_array_for_2d_polynomials_of_degree_2(self):
        degree = 2
        nodes = get_nodes_2d(degree)
        _, _, Ps = _get_orthonormal_poly_basis_2d(degree)
        Vs = _get_V(nodes, Ps)
        Vs_actual = np.array(
            [
                [
                    0,
                    1.5,
                    -4 * np.sqrt(3 / 2),
                    np.sqrt(3 / 4),
                    -6 * np.sqrt(9 / 8),
                    -2 * np.sqrt(15 / 8),
                ],
                [
                    0,
                    1.5,
                    -4 * np.sqrt(3 / 2),
                    np.sqrt(3 / 4),
                    -np.sqrt(9 / 8),
                    np.sqrt(15 / 8),
                ],
                [
                    0,
                    1.5,
                    -4 * np.sqrt(3 / 2),
                    np.sqrt(3 / 4),
                    4 * np.sqrt(9 / 8),
                    4 * np.sqrt(15 / 8),
                ],
                [
                    0,
                    1.5,
                    np.sqrt(3 / 2),
                    np.sqrt(3 / 4),
                    -np.sqrt(9 / 8),
                    -np.sqrt(15 / 8),
                ],
                [
                    0,
                    1.5,
                    np.sqrt(3 / 2),
                    np.sqrt(3 / 4),
                    4 * np.sqrt(9 / 8),
                    2 * np.sqrt(15 / 8),
                ],
                [0, 1.5, 6 * np.sqrt(3 / 2), np.sqrt(3 / 4), 4 * np.sqrt(9 / 8), 0],
            ]
        )
        assert np.allclose(Vs, Vs_actual)


class TestGetDerivativeOperator1D:
    def test_returns_square_numpy_array(self):
        degree = 2
        n_nodes = degree + 1
        Dr = get_derivative_operator_1d(degree)
        assert isinstance(Dr, np.ndarray)
        assert len(Dr.shape) == 2
        assert Dr.shape[0] == n_nodes
        assert Dr.shape[1] == n_nodes

    def test_returns_correct_Dr_for_degree_2(self):
        degree = 2
        Dr_actual = np.array([[-1.5, 2, -0.5], [-0.5, 0, 0.5], [0.5, -2, 1.5]])
        Dr_pred = get_derivative_operator_1d(degree)
        assert np.allclose(Dr_pred, Dr_actual)

    def test_returns_correct_Dr_for_degree_10(self):
        degree = 10
        Dr_actual = load("Dr_1d_degree_10.npy")
        Dr_pred = get_derivative_operator_1d(degree)
        assert np.allclose(Dr_actual, Dr_pred)


class TestGetLIFT1D:
    def test_returns_2d_numpy_array(self):
        degree = 2
        n_nodes = degree + 1
        LIFT = get_LIFT_1d(degree)
        assert isinstance(LIFT, np.ndarray)
        assert len(LIFT.shape) == 2
        assert LIFT.shape[0] == n_nodes
        assert LIFT.shape[1] == 2

    def test_returns_correct_matrix_for_degree_10(self):
        degree = 10
        LIFT_actual = load("LIFT_1d_degree_10.npy")
        LIFT_pred = get_LIFT_1d(degree)
        assert np.allclose(LIFT_actual, LIFT_pred)


class TestGetDerivativeOperator2D:
    def test_returns_square_numpy_array(self):
        degree = 5
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        Dr, Ds = get_derivative_operators_2d(degree)
        for D in [Dr, Ds]:
            assert isinstance(D, np.ndarray)
            assert len(D.shape) == 2
            assert D.shape[0] == n_nodes
            assert D.shape[1] == n_nodes

    def test_returns_correct_Dr_when_degree_is_2(self):
        degree = 2
        Dr, _ = get_derivative_operators_2d(degree)
        Dr_actual = np.array(
            [
                [-1.5, 2, -0.5, 0, 0, 0],
                [-0.5, 0, 0.5, 0, 0, 0],
                [0.5, -2, 1.5, 0, 0, 0],
                [-0.5, 1, -0.5, -1, 1, 0],
                [0.5, -1, 0.5, -1, 1, 0],
                [0.5, 0, -0.5, -2, 2, 0],
            ]
        )
        assert np.allclose(Dr, Dr_actual)

    def test_returns_correct_Ds_when_degree_is_2(self):
        degree = 2
        _, Ds = get_derivative_operators_2d(degree)
        Ds_actual = np.array(
            [
                [-1.5, 0, 0, 2, 0, -0.5],
                [-0.5, -1, 0, 1, 1, -0.5],
                [0.5, -2, 0, 0, 2, -0.5],
                [-0.5, 0, 0, 0, 0, 0.5],
                [0.5, -1, 0, -1, 1, 0.5],
                [0.5, 0, 0, -2, 0, 1.5],
            ]
        )
        assert np.allclose(Ds, Ds_actual)

    def test_returns_correct_Dr_when_degree_is_10(self):
        degree = 10
        Dr, _ = get_derivative_operators_2d(degree)
        Dr_actual = load("Dr_2d_degree_10.npy")
        assert np.allclose(Dr, Dr_actual)

    def test_returns_correct_Ds_when_degree_is_10(self):
        degree = 10
        _, Ds = get_derivative_operators_2d(degree)
        Ds_actual = load("Ds_2d_degree_10.npy")
        assert np.allclose(Ds, Ds_actual)


class TestGetEdgeMassMatrices2D:
    def test_returns_square_numpy_array(self):
        degree = 3
        n_edge_nodes = degree + 1
        M = get_edge_mass_matrix_2d(degree)
        assert isinstance(M, np.ndarray)
        assert len(M.shape) == 2
        assert M.shape[0] == n_edge_nodes
        assert M.shape[1] == n_edge_nodes

    def test_returns_correct_matrix_when_degree_is_2(self):
        degree = 2
        M_pred = get_edge_mass_matrix_2d(degree)
        M_actual = np.array(
            [
                [4 / 15, 2 / 15, -1 / 15],
                [2 / 15, 16 / 15, 2 / 15],
                [-1 / 15, 2 / 15, 4 / 15],
            ]
        )
        assert np.allclose(M_pred, M_actual, rtol=0, atol=1e-3)


class TestGetLIFT2D:
    def test_returns_numpy_array(self):
        degree = 5
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        n_edge_nodes = degree + 1
        LIFT = get_LIFT_2d(degree)
        assert isinstance(LIFT, np.ndarray)
        assert len(LIFT.shape) == 2
        assert LIFT.shape[0] == n_nodes
        assert LIFT.shape[1] == 3 * n_edge_nodes

    def test_returns_correct_array_for_degree_2(self):
        degree = 2
        LIFT = get_LIFT_2d(degree)
        LIFT_actual = np.array(
            [
                [4.5, 2, -0.5, 1, 4, 1, 4.5, 2, -0.5],
                [0.5, 5, 0.5, -0.625, -1.5, 0.625, -0.625, -1.5, 0.625],
                [-0.5, 2, 4.5, 4.5, 2, -0.5, 1, 4, 1],
                [-0.625, -1.5, 0.625, 0.625, -1.5, -0.625, 0.5, 5, 0.5],
                [0.625, -1.5, -0.625, 0.5, 5, 0.5, 0.625, -1.5, -0.625],
                [1, 4, 1, -0.5, 2, 4.5, -0.5, 2, 4.5],
            ]
        )
        assert np.allclose(LIFT, LIFT_actual)

    def test_returns_correct_array_for_degree_10(self):
        degree = 10
        LIFT = get_LIFT_2d(degree)
        LIFT_actual = load("LIFT_2d_degree_10.npy")
        assert np.allclose(LIFT_actual, LIFT)
