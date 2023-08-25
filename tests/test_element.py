from typing import Tuple
import pytest
import numpy as np

from pydglib.element import Element1D, Element2D, Element2DInterface, ElementGeometry2D
import pydglib.utils.nodes as pydglib_nodes_module


class TestElement1D:
    def test_raises_exception_for_invalid_physical_domain(self):
        id = 0
        n_nodes = 3
        xl = -1
        xr = -2
        IC = lambda x: np.zeros_like(x)
        with pytest.raises(Exception):
            _ = Element1D(id, n_nodes, xl, xr, IC)

    def test_state_dimension_set_correctly_1d(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 2
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert element.state_dimension == 1

    def test_state_dimension_set_correctly_multid(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 2
        ICs = [
            lambda x: 0 * np.ones_like(x),
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        assert element.state_dimension == len(ICs)

    def test_nodes_are_set(self):
        id = 0
        n_nodes = 7
        xl = 0
        xr = 1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert isinstance(element.nodes, np.ndarray)
        assert len(element.nodes.shape) == 1
        assert element.nodes.shape[0] == n_nodes
        assert element.nodes[0] == xl
        assert element.nodes[-1] == xr
        if n_nodes % 2 == 1:
            assert element.nodes[int(n_nodes / 2)] == (xl + xr) / 2
        for i in range(n_nodes - 1):
            assert element.nodes[i] < element.nodes[i + 1]

    def test_state_correct_shape_when_state_is_1d(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert isinstance(element.state, np.ndarray)
        assert len(element.state.shape) == 1
        assert element.state.shape[0] == n_nodes

    def test_state_correct_shape_when_state_is_multid(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1
        ICs = [
            lambda x: 0 * np.ones_like(x),
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        assert isinstance(element.state, np.ndarray)
        assert len(element.state.shape) == 2
        assert element.state.shape[0] == len(ICs)
        assert element.state.shape[1] == n_nodes

    def test_grad_correct_shape_when_state_is_1d(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert isinstance(element.grad, np.ndarray)
        assert len(element.grad.shape) == 1
        assert element.grad.shape[0] == n_nodes

    def test_grad_correct_shape_when_state_is_multid(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1
        ICs = [
            lambda x: 0 * np.ones_like(x),
            lambda x: 1 * np.ones_like(x),
        ]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        assert isinstance(element.grad, np.ndarray)
        assert len(element.grad.shape) == 2
        assert element.grad.shape[0] == len(ICs)
        assert element.grad.shape[1] == n_nodes

    def test_initial_conditions_applied_when_state_is_1d(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1
        y0 = np.linspace(xl, xr, n_nodes)
        IC = lambda _: y0
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert np.allclose(element.state, y0)

    def test_initial_conditions_applied_when_state_is_multid(self):
        id = 0
        n_nodes = 4
        xl = 0.4
        xr = 1.1
        y0 = np.linspace(xl, xr, n_nodes)
        y1 = 2 * y0
        ICs = [lambda _: y0, lambda _: y1]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        assert np.allclose(element.state[0], y0)
        assert np.allclose(element.state[1], y1)

    def test_left_reference_None_by_default(self):
        id = 0
        n_nodes = 4
        xl = 0.4
        xr = 1.1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert element.left is None

    def test_right_reference_None_by_default(self):
        id = 0
        n_nodes = 4
        xl = 0.4
        xr = 1.1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert element.right is None

    def test_h_property(self):
        id = 0
        n_nodes = 4
        xl = 0.4
        xr = 1.1
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert element.h == xr - xl

    def test_iadd_adds_to_state(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1.101
        IC = lambda x: np.zeros_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)

        element += 1
        assert np.allclose(element.state, np.ones_like(element.nodes))

        element += np.linspace(1, n_nodes, n_nodes)
        assert np.allclose(element.state, np.linspace(2, n_nodes + 1, n_nodes))

    def test_imul_multiplies_state(self):
        id = 0
        n_nodes = 3
        xl = 0.1
        xr = 1.101
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)

        element *= 0.5
        assert np.allclose(element.state, np.ones_like(element.nodes))

        element *= np.linspace(1, n_nodes, n_nodes)
        assert np.allclose(element.state, np.linspace(1, n_nodes, n_nodes))

    def test_getitem_retrieves_state_when_state_is_1d(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert element[n_nodes - 1] == IC(xr)

    def test_getitem_retrieves_state_when_state_is_multid(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 1
        ICs = [
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        assert element[0, n_nodes - 1] == ICs[0](xr)
        assert element[1, n_nodes - 1] == ICs[1](xr)
        assert np.allclose(element[0], np.ones(n_nodes))

    def test_setitem_sets_state_when_state_is_1d(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        element[1] = 5.123
        assert element[1] == 5.123

    def test_setitem_sets_state_when_state_is_multid(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 1
        ICs = [
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        element = Element1D(id, n_nodes, xl, xr, ICs)
        element[0, 1] = 7
        element[1] = 3 * np.linspace(1, n_nodes, n_nodes)
        assert element[0, 1] == 7
        assert np.allclose(element[1], 3 * np.linspace(1, n_nodes, n_nodes))

    def test_conversion_to_numpy_array(self):
        id = 0
        n_nodes = 3
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        array = np.array(element)
        assert isinstance(array, np.ndarray)
        assert array.shape == element.state.shape
        assert np.allclose(array, element.state)

    def test_len_returns_number_of_nodes(self):
        id = 0
        n_nodes = 9
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(id, n_nodes, xl, xr, IC)
        assert len(element) == n_nodes

    def test_is_leftmost(self):
        n_nodes = 3
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(0, n_nodes, xl, xr, IC)
        assert not element.is_leftmost()

        element.right = Element1D(1, n_nodes, xl, xr, IC)
        assert element.is_leftmost()

        element.left = Element1D(2, n_nodes, xl, xr, IC)
        assert not element.is_leftmost()

        element.right = None
        assert not element.is_leftmost()

    def test_is_rightmost(self):
        n_nodes = 3
        xl = 0
        xr = 1
        IC = lambda x: 2 * np.ones_like(x)
        element = Element1D(0, n_nodes, xl, xr, IC)
        assert not element.is_rightmost()

        element.left = Element1D(1, n_nodes, xl, xr, IC)
        assert element.is_rightmost()

        element.right = Element1D(2, n_nodes, xl, xr, IC)
        assert not element.is_rightmost()

        element.left = None
        assert not element.is_rightmost()


class TestElementGeometry2D:
    def get_geometric_factors():
        return [
            (
                np.array([-1, -1]),
                np.array([-0.75, -1]),
                np.array([-0.75, -0.75]),
                8,
                -8,
                0,
                8,
                np.array([8, 8, np.sqrt(2) * 8]),
            ),
            (
                np.array([-1, -1]),
                np.array([-0.75, -0.75]),
                np.array([-1, -0.75]),
                8,
                0,
                -8,
                8,
                np.array([np.sqrt(2) * 8, 8, 8]),
            ),
        ]

    @pytest.mark.parametrize("v1,v2,v3,rx,ry,sx,sy,Fscale", get_geometric_factors())
    def test_correct_values_of_geometric_factors_set(
        self, v1, v2, v3, rx, ry, sx, sy, Fscale
    ):
        geometry = ElementGeometry2D(v1, v2, v3)
        assert np.isclose(geometry.rx, rx)
        assert np.isclose(geometry.ry, ry)
        assert np.isclose(geometry.sx, sx)
        assert np.isclose(geometry.sy, sy)
        assert np.allclose(geometry.Fscale, Fscale)


class TestElement2D:
    def test_raises_exception_for_invalid_physical_domain(self):
        degree = 2
        v1 = np.ones(2)
        v2 = 2 * v1
        v3 = 3 * v1
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        with pytest.raises(Exception):
            _ = Element2D(degree, v1, v2, v3, IC)

    def test_nodes_are_set(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert isinstance(element.nodes, np.ndarray)
        assert len(element.nodes.shape) == 2
        assert element.n_nodes == int(0.5 * (degree + 1) * (degree + 2))
        assert element.nodes.shape[0] == element.n_nodes
        assert element.nodes.shape[1] == 2

    def test_state_correct_shape(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert isinstance(element.state, np.ndarray)
        assert len(element.state.shape) == 1
        assert element.state.shape[0] == element.n_nodes

    def test_grad_correct_shape(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert isinstance(element.grad, np.ndarray)
        assert len(element.grad.shape) == 1
        assert element.grad.shape[0] == element.n_nodes

    def test_initial_conditions_applied(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert element.state[4] == 5

    def test_interface_references_are_None_by_default(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        for edge in element.edges:
            assert edge is None

    def test_area_is_set(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert element.area == 0.5

    def test_perimeter_is_set(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert element.perimeter == 2 + np.sqrt(2)

    def test_iadd_adds_to_state(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        element += 2
        assert element.state[3] == 7

    def test_imul_multiplies_state(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        element *= 2
        assert element.state[2] == 10

    def test_getitem_retrieves_state(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert element[4] == 5

    def test_setitem_sets_state(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        element[3] = 8
        assert element[3] == 8

    def test_conversion_to_numpy_array(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert np.allclose(np.array(element), 5 * np.ones(element.n_nodes))

    def test_len_returns_number_of_nodes(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        assert len(element) == int(0.5 * (degree + 1) * (degree + 2))

    def test_vertices_are_set(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        for vertex, vi in zip(element.vertices, [v1, v2, v3]):
            assert isinstance(vertex, np.ndarray)
            assert len(vertex.shape) == 1
            assert np.allclose(vertex, vi)

    def test_normals_are_set(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        for normal, edge in zip(element.normals, [v2 - v1, v3 - v2, v1 - v3]):
            assert isinstance(normal, np.ndarray)
            assert len(normal.shape) == 1
            assert normal.size == 2
            assert np.isclose(np.linalg.norm(normal), 1)

    def test_normals_are_orthogonal_to_edges(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        for normal, edge in zip(element.normals, [v2 - v1, v3 - v2, v1 - v3]):
            assert np.isclose(np.dot(normal, edge), 0)

    def test_get_edge_nodes(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: 5 if len(x.shape) == 1 else 5 * np.ones(x.shape[0])
        element = Element2D(degree, v1, v2, v3, IC)
        edge1_nodes = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        edge2_nodes = np.array([[1, 1], [0.5, 1], [0, 1]])
        edge3_nodes = np.array([[0, 1], [0, 0.5], [0, 0]])
        assert np.allclose(element.get_edge_nodes(0), edge1_nodes)
        assert np.allclose(element.get_edge_nodes(1), edge2_nodes)
        assert np.allclose(element.get_edge_nodes(2), edge3_nodes)

    def test_get_edge(self):
        degree = 2
        v1 = np.zeros(2)
        v2 = np.ones(2)
        v3 = np.array([0, 1])
        IC = lambda x: x[:, 0] * x[:, 1]  # IC1(x,y) = x * y
        element = Element2D(degree, v1, v2, v3, IC)
        assert np.allclose(element.get_edge(0), np.array([0, 0.25, 1]))
        assert np.allclose(element.get_edge(1), np.array([1, 0.5, 0]))
        assert np.allclose(element.get_edge(2), np.array([0, 0, 0]))

    def test_returns_correct_internal_and_external_state_when_degree_is_2(self):
        degree = 2
        v1 = np.array([0, 0])
        v2 = np.array([1, 0])
        v3 = np.array([1, 1])
        v4 = np.array([0, 1])
        zero_IC = lambda x: np.zeros(x.shape[0])

        element1 = Element2D(degree, v1, v2, v4, zero_IC)
        element2 = Element2D(degree, v2, v3, v4, zero_IC)

        element1.state = np.array([1, 2, 3, 4, 5, 6])
        element2.state = np.array([1, 2, 3, 4, 5, 6])

        element1.edges[1] = Element2DInterface(element1, element2, 1, 2)
        element2.edges[2] = Element2DInterface(element2, element1, 2, 1)

        assert np.allclose(element1.get_edge(0), np.array([1, 2, 3]))
        assert np.allclose(element1.get_edge(1), np.array([3, 5, 6]))
        assert np.allclose(element1.get_edge(2), np.array([6, 4, 1]))
        assert np.allclose(element1.edges[1].get_internal_state(), element1.get_edge(1))
        assert np.allclose(
            element2.edges[2].get_external_state(), np.flip(element1.get_edge(1))
        )

        assert np.allclose(element2.get_edge(0), np.array([1, 2, 3]))
        assert np.allclose(element2.get_edge(1), np.array([3, 5, 6]))
        assert np.allclose(element2.get_edge(2), np.array([6, 4, 1]))
        assert np.allclose(element2.edges[2].get_internal_state(), element2.get_edge(2))
        assert np.allclose(
            element1.edges[1].get_external_state(), np.flip(element2.get_edge(2))
        )

    def test_doesnt_call_get_nodes_when_nodes_are_supplied_in_constructor(
        self, monkeypatch
    ):
        # Define arguments for Element2D instance
        degree = 8
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        n_edge_nodes = degree + 1
        v1 = np.array([0, 0])
        v2 = np.array([1, 1])
        v3 = np.array([0, 1])
        IC = lambda x: np.zeros(x.shape[0])
        nodes, b1, b2, b3 = pydglib_nodes_module.get_nodes_2d(
            degree, include_boundary=True
        )

        # Patch get_nodes_2d so we know when if get_nodes_2d is called by the element
        self.invocation_counter = 0

        def mock_get_nodes_2d(*args, **kwargs):
            self.invocation_counter += 1
            return nodes, b1, b2, b3

        monkeypatch.setattr(pydglib_nodes_module, "get_nodes_2d", mock_get_nodes_2d)

        element = Element2D(degree, v1, v2, v3, IC, nodes, b1, b2, b3)

        assert self.invocation_counter == 0
        assert isinstance(element.nodes, np.ndarray)
        assert len(element.nodes.shape) == 2
        assert element.nodes.shape[0] == n_nodes
        assert element.nodes.shape[1] == 2
        assert len(element._edge_node_indicies) == 3
        for indicies in element._edge_node_indicies:
            assert isinstance(indicies, np.ndarray)
            assert len(indicies.shape) == 1
            assert indicies.shape[0] == n_edge_nodes

    def test_geometric_factors_are_accessible_as_instance_properties(self):
        v1 = np.array([0, 0])
        v2 = np.array([1, 0])
        v3 = np.array([1, 1])
        element = Element2D(3, v1, v2, v3, lambda x: np.zeros(x.shape[0]))
        assert np.isscalar(element.rx)
        assert np.isscalar(element.ry)
        assert np.isscalar(element.sx)
        assert np.isscalar(element.sy)

    def test_Fscale_is_accessible_as_an_instance_property(self):
        v1 = np.array([0, 0])
        v2 = np.array([1, 0])
        v3 = np.array([1, 1])
        element = Element2D(3, v1, v2, v3, lambda x: np.zeros(x.shape[0]))
        assert len(element.Fscale) == 3
        for f in element.Fscale:
            assert np.isscalar(f)


class TestElement2DInterface:
    def get_elements(self, degree: int) -> Tuple[Element2D, Element2D]:
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.2, 0.1])
        v3 = np.array([0.1, 0.3])
        v4 = np.array([0.3, 0.25])
        IC = lambda x: 0 if len(x.shape) == 1 else np.zeros(x.shape[0])
        element1 = Element2D(degree, v1, v2, v3, IC)
        element2 = Element2D(degree, v2, v4, v3, IC)
        return element1, element2

    def test_raises_exception_when_supplied_with_invalid_face_id(self):
        degree = 2
        element1, element2 = self.get_elements(degree)
        for int, ext in [(0, 3), (3, 2), (-1, 3)]:
            with pytest.raises(Exception):
                _ = Element2DInterface(element1, element2, int, ext)
            with pytest.raises(Exception):
                _ = Element2DInterface(element2, element1, ext, int)

    def test_create_node_map_for_interface_creates_correct_map(self):
        interior_nodes = np.array([[0, 0], [0.1, 0.05], [0.2, 0.1]])
        interior_node_indices = np.array([1, 7, 16])
        exterior_nodes = np.array([[0.2, 0.09998], [0, 0], [0.1, 0.04999]])
        exterior_node_indicies = np.array([2, 4, 9])
        lut = Element2DInterface._create_node_map_for_interface(
            interior_nodes,
            interior_node_indices,
            exterior_nodes,
            exterior_node_indicies,
        )
        correct_nodal_mapping = np.array([[1, 4], [7, 9], [16, 2]])
        assert np.allclose(lut, correct_nodal_mapping)

    def test_create_node_map_creates_2d_numpy_array(self):
        degree = 5
        n_edge_nodes = degree + 1
        e1, e2 = self.get_elements(degree)
        int_face_idx = 1
        ext_face_idx = 2
        interface = Element2DInterface(e1, e2, int_face_idx, ext_face_idx)
        assert isinstance(interface._node_map, np.ndarray)
        assert len(interface._node_map.shape) == 2
        assert interface._node_map.shape[0] == n_edge_nodes
        assert interface._node_map.shape[1] == 2

    def test_entries_in_node_map_are_unique_node_ids(self):
        degree = 4
        n_edge_nodes = degree + 1
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        e1, e2 = self.get_elements(degree)
        int_face_idx = 1
        ext_face_idx = 2
        interface = Element2DInterface(e1, e2, int_face_idx, ext_face_idx)
        for i in range(n_edge_nodes):
            for j in [0, 1]:
                assert isinstance(interface._node_map[i, j], np.integer)
                assert 0 <= interface._node_map[i, j] < n_nodes
        for i in [0, 1]:
            assert np.unique(interface._node_map[:, i]).size == n_edge_nodes

    def test_node_map_maps_adjacent_nodes_across_interface(self):
        degree = 2
        e1, e2 = self.get_elements(degree)
        int_face_idx = 1
        ext_face_idx = 2
        interface = Element2DInterface(e1, e2, int_face_idx, ext_face_idx)
        correct_node_map = np.array([[2, 0], [4, 3], [5, 5]])
        assert np.allclose(interface._node_map, correct_node_map)

    def test_get_external_state(self):
        degree = 2
        n_edge_nodes = degree + 1
        e1, e2 = self.get_elements(degree)
        int_face_idx = 1
        ext_face_idx = 2
        interface = Element2DInterface(e1, e2, int_face_idx, ext_face_idx)
        ext_state = interface.get_external_state()
        assert isinstance(ext_state, np.ndarray)
        assert len(ext_state.shape) == 1
        assert ext_state.shape[0] == n_edge_nodes

    def test_get_internal_state(self):
        degree = 2
        n_edge_nodes = degree + 1
        e1, e2 = self.get_elements(degree)
        int_face_idx = 1
        ext_face_idx = 2
        interface = Element2DInterface(e1, e2, int_face_idx, ext_face_idx)
        int_state = interface.get_internal_state()
        assert isinstance(int_state, np.ndarray)
        assert len(int_state.shape) == 1
        assert int_state.shape[0] == n_edge_nodes
