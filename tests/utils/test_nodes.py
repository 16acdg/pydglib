import pytest
import numpy as np

from pydglib.utils.nodes import get_nodes_1d, NODES_MAX_1D, _get_alpha_opt, get_nodes_2d


class TestGetNodes1D:
    @pytest.mark.parametrize("degree", [-1, 0, 2.1, "3"])
    def test_fails_if_degree_not_an_integer_greater_than_0(self, degree):
        with pytest.raises(Exception):
            _ = get_nodes_1d(degree)

    @pytest.mark.parametrize("degree", [NODES_MAX_1D, 2 * NODES_MAX_1D])
    def test_fails_if_request_too_many_nodes(self, degree):
        with pytest.raises(Exception):
            _ = get_nodes_1d(degree)

    @pytest.mark.parametrize("degree", [2, 6, 9, 13, 18])
    def test_returns_nodes_as_1d_numpy_array(self, degree):
        n_nodes = degree + 1
        nodes = get_nodes_1d(degree)
        assert isinstance(nodes, np.ndarray)
        assert len(nodes.shape) == 1
        assert nodes.shape[0] == n_nodes

    @pytest.mark.parametrize("degree", [2, 6, 9, 13, 18])
    def test_returned_nodes_are_in_increasing_order(self, degree):
        nodes = get_nodes_1d(degree)
        for i in range(len(nodes) - 1):
            assert nodes[i] < nodes[i + 1]

    @pytest.mark.parametrize("degree", [2, 6, 9, 13, 18])
    def test_boundary_nodes_are_plus_and_minus_1(self, degree):
        nodes = get_nodes_1d(degree)
        assert nodes[0] == -1
        assert nodes[-1] == 1

    def test_returns_correct_nodes_when_degree_is_1(self):
        correct_nodes = np.array([-1, 1])
        nodes = get_nodes_1d(1)
        assert np.allclose(nodes, correct_nodes)

    def test_returns_correct_nodes_when_degree_is_2(self):
        correct_nodes = np.array([-1, 0, 1])
        nodes = get_nodes_1d(2)
        assert np.allclose(nodes, correct_nodes)

    def test_returns_correct_nodes_when_degree_is_3(self):
        correct_nodes = np.array([-1, -np.sqrt(5) / 5, np.sqrt(5) / 5, 1])
        nodes = get_nodes_1d(3)
        assert np.allclose(nodes, correct_nodes)

    def test_returns_correct_nodes_when_degree_is_4(self):
        correct_nodes = np.array([-1, -np.sqrt(21) / 7, 0, np.sqrt(21) / 7, 1])
        nodes = get_nodes_1d(4)
        assert np.allclose(nodes, correct_nodes)

    def test_returns_correct_nodes_when_degree_is_5(self):
        correct_nodes = np.array(
            [
                -1,
                -np.sqrt(147 + 42 * np.sqrt(7)) / 21,
                -np.sqrt(147 - 42 * np.sqrt(7)) / 21,
                np.sqrt(147 - 42 * np.sqrt(7)) / 21,
                np.sqrt(147 + 42 * np.sqrt(7)) / 21,
                1,
            ]
        )
        nodes = get_nodes_1d(5)
        assert np.allclose(nodes, correct_nodes)

    def test_returns_correct_nodes_when_degree_is_10(self):
        correct_nodes = np.array(
            [
                -1,
                -0.934001430408059,
                -0.784483473663144,
                -0.565235326996205,
                -0.295758135586939,
                0,
                0.295758135586939,
                0.565235326996205,
                0.784483473663144,
                0.934001430408059,
                1,
            ]
        )
        nodes = get_nodes_1d(10)
        assert np.allclose(nodes, correct_nodes)


class TestGetAlphaOpt:
    @pytest.mark.parametrize("degree", [-1, 0, 0.5, "2"])
    def test_raises_exception_for_invalid_degree(self, degree):
        with pytest.raises(Exception):
            _ = _get_alpha_opt(degree)

    @pytest.mark.parametrize("degree", [1, 4, 8, 15, 16, 21])
    def test_returns_valid_float(self, degree):
        alpha = _get_alpha_opt(degree)
        assert isinstance(alpha, float)
        assert 0 <= alpha < 2


class TestGetNodes2D:
    @pytest.mark.parametrize("degree", [-1, 2.1, "3"])
    def test_fails_if_input_not_an_integer_greater_than_1(self, degree):
        with pytest.raises(Exception):
            _ = get_nodes_2d(degree)

    @pytest.mark.parametrize("degree", [20, 46, 1020])
    def test_fails_if_request_too_many_nodes(self, degree):
        with pytest.raises(Exception):
            _ = get_nodes_2d(degree)

    def test_returns_nodes_as_2d_numpy_array(self):
        degree = 6
        nodes = get_nodes_2d(degree)
        assert isinstance(nodes, np.ndarray)
        assert len(nodes.shape) == 2
        assert nodes.shape[0] == int(0.5 * (degree + 1) * (degree + 2))
        assert nodes.shape[1] == 2

    def test_returned_nodes_have_no_repeats(self):
        degree = 4
        nodes = get_nodes_2d(degree)
        for i in range(nodes.shape[0]):
            for j in range(nodes.shape[0]):
                if i != j:
                    assert np.linalg.norm(nodes[i] - nodes[j]) > 0

    def test_returned_nodes_are_inside_reference_triangle(self):
        degree = 4
        nodes = get_nodes_2d(degree)
        for i in range(nodes.shape[0]):
            x, y = nodes[i]
            assert x + y < 1e-10
            assert -1 <= x <= 1
            assert -1 <= y <= 1

    def test_returns_correct_nodes_for_2nd_degree_approximation(self):
        degree = 2
        nodes = get_nodes_2d(degree)
        nodes_correct = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [-1, 1]])
        assert np.allclose(nodes, nodes_correct)

    def test_returns_correct_nodes_when_degree_is_10(self):
        degree = 10
        nodes = get_nodes_2d(degree)
        nodes_correct = np.array(
            [
                [-1.0, -1.0],
                [-0.934001430408059, -1.0],
                [-0.784483473663144, -1.0],
                [-0.565235326996205, -1.0],
                [-0.295758135586939, -1.0],
                [2.77555756156289e-16, -1.0],
                [0.295758135586939, -1.0],
                [0.565235326996205, -1.0],
                [0.784483473663144, -1.0],
                [0.934001430408059, -1.0],
                [1.0, -1.0],
                [-1.0, -0.934001430408059],
                [-0.908326844622856, -0.908326844622856],
                [-0.730941443379584, -0.896043136459892],
                [-0.488061300910199, -0.891238121900236],
                [-0.204128963938669, -0.889864041132599],
                [0.0939930050712676, -0.889864041132598],
                [0.379299422810433, -0.891238121900235],
                [0.626984579839477, -0.896043136459892],
                [0.816653689245712, -0.908326844622856],
                [0.934001430408059, -0.934001430408059],
                [-1.0, -0.784483473663144],
                [-0.896043136459892, -0.730941443379585],
                [-0.702662749234053, -0.702662749234052],
                [-0.446507724921086, -0.689751231477825],
                [-0.156934807111189, -0.686130385777621],
                [0.136258956398911, -0.689751231477826],
                [0.405325498468105, -0.702662749234052],
                [0.626984579839476, -0.730941443379584],
                [0.784483473663144, -0.784483473663144],
                [-1.0, -0.565235326996205],
                [-0.891238121900235, -0.488061300910198],
                [-0.689751231477826, -0.446507724921085],
                [-0.428742602834574, -0.428742602834574],
                [-0.142514794330852, -0.428742602834574],
                [0.136258956398911, -0.446507724921086],
                [0.379299422810434, -0.488061300910199],
                [0.565235326996205, -0.565235326996205],
                [-1.0, -0.295758135586939],
                [-0.889864041132599, -0.204128963938669],
                [-0.686130385777621, -0.15693480711119],
                [-0.428742602834574, -0.142514794330852],
                [-0.156934807111189, -0.156934807111189],
                [0.0939930050712674, -0.204128963938669],
                [0.295758135586939, -0.295758135586939],
                [-1.0, -2.77555756156289e-16],
                [-0.889864041132599, 0.0939930050712676],
                [-0.689751231477825, 0.136258956398911],
                [-0.446507724921085, 0.136258956398911],
                [-0.204128963938669, 0.0939930050712678],
                [-2.77555756156289e-16, 2.77555756156289e-16],
                [-1.0, 0.295758135586939],
                [-0.891238121900235, 0.379299422810434],
                [-0.702662749234052, 0.405325498468105],
                [-0.488061300910198, 0.379299422810433],
                [-0.295758135586939, 0.295758135586939],
                [-1.0, 0.565235326996205],
                [-0.896043136459892, 0.626984579839476],
                [-0.730941443379585, 0.626984579839477],
                [-0.565235326996205, 0.565235326996205],
                [-1.0, 0.784483473663144],
                [-0.908326844622856, 0.816653689245712],
                [-0.784483473663144, 0.784483473663144],
                [-1.0, 0.934001430408059],
                [-0.934001430408059, 0.934001430408059],
                [-1.0, 1.0],
            ]
        )
        assert np.allclose(nodes, nodes_correct)
