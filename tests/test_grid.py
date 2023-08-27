from typing import List
import numpy as np

from pydglib.grid import Grid1D, create_elements_2d, connect_elements_2d, Grid2D
from pydglib.element import Element1D, Element2D, Element2DInterface, ElementEdge2D
from pydglib.mesh import meshgen1d, meshgen2d

from .data import load


class TestGrid1D:
    def test_n_elements_is_set(self):
        xl = 0
        xr = 1
        n_elements = 3
        degree = 4
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.zeros_like(x)
        grid = Grid1D(VX, EToV, degree, IC)
        assert grid.n_elements == n_elements

    def test_xl_and_xr_are_set(self):
        xl = 0
        xr = 1.25
        n_elements = 5
        degree = 4
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.zeros_like(x)
        grid = Grid1D(VX, EToV, degree, IC)
        assert grid.xl == xl
        assert grid.xr == xr

    def test_state_dimension_set_correctly_1d(self):
        xl = 0
        xr = 1.25
        n_elements = 5
        degree = 3
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.zeros_like(x)
        grid = Grid1D(VX, EToV, degree, IC)
        assert grid.state_dimension == 1

    def test_state_dimension_set_correctly_multid(self):
        xl = 0
        xr = 1.25
        n_elements = 5
        degree = 3
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda x: 0 * np.ones_like(x),
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)
        assert grid.state_dimension == len(ICs)

    def test_elements_are_created(self):
        xl = 0
        xr = 1.25
        n_elements = 5
        degree = 3
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda x: 0 * np.ones_like(x),
            lambda x: 1 * np.ones_like(x),
            lambda x: 2 * np.ones_like(x),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)
        assert len(grid.elements) == n_elements
        for element in grid.elements:
            assert isinstance(element, Element1D)
        assert grid.elements[0].nodes[0] == xl
        assert grid.elements[0][0, 0] == 0
        assert grid.elements[-1].nodes[-1] == xr
        assert grid.elements[-1][2, -1] == 2

    def test_references_between_elements_are_correct(self):
        xl = 1.2
        xr = 1.25
        n_elements = 4
        degree = 3
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.ones_like(x)
        grid = Grid1D(VX, EToV, degree, IC)

        assert grid.elements[0].left is None

        for i in range(n_elements - 1):
            assert grid.elements[i] == grid.elements[i + 1].left
            assert grid.elements[i].right == grid.elements[i + 1]

        assert grid.elements[-1].right is None

    def test_nodes_property_returns_matrix_of_all_elements_nodes(self):
        xl = 0.1
        xr = 0.9
        n_elements = 4
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.zeros_like(x)
        grid = Grid1D(VX, EToV, degree, IC)

        assert isinstance(grid.nodes, np.ndarray)
        assert len(grid.nodes.shape) == 2
        assert grid.nodes.shape[0] == n_elements
        assert grid.nodes.shape[1] == n_nodes
        for i in range(n_elements):
            assert np.allclose(grid.nodes[i], grid.elements[i].nodes)

    def test_state_property_returns_matrix_of_all_elements_states_1d(self):
        xl = 0.1
        xr = 0.9
        n_elements = 4
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda _: np.linspace(10, 12, n_nodes)
        grid = Grid1D(VX, EToV, degree, IC)

        assert isinstance(grid.state, np.ndarray)
        assert len(grid.state.shape) == 2
        assert grid.state.shape[0] == n_elements
        assert grid.state.shape[1] == n_nodes
        for i in range(n_elements):
            assert np.allclose(grid.state[i], grid.elements[i])

    def test_state_property_returns_matrix_of_all_elements_states_multid(self):
        xl = 0.1
        xr = 0.9
        n_elements = 4
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda _: np.linspace(9, 10, n_nodes),
            lambda _: np.linspace(99, 100, n_nodes),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)

        assert isinstance(grid.state, np.ndarray)
        assert len(grid.state.shape) == 3
        assert grid.state.shape[0] == n_elements
        assert grid.state.shape[1] == n_nodes
        assert grid.state.shape[2] == len(ICs)
        for i in range(n_elements):
            for j in range(len(ICs)):
                assert np.allclose(grid.state[i, :, j], grid.elements[i][:, j])

    def test_grad_property_returns_matrix_of_all_elements_gradients_1d(self):
        xl = 0.1
        xr = 0.9
        n_elements = 3
        degree = 6
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda x: np.zeros_like(x)
        grid = Grid1D(VX, EToV, degree, IC)

        assert isinstance(grid.grad, np.ndarray)
        assert len(grid.grad.shape) == 2
        assert grid.grad.shape[0] == n_elements
        assert grid.grad.shape[1] == n_nodes
        for i in range(n_elements):
            assert np.allclose(grid.grad[i], grid.elements[i].grad)

    def test_grad_property_returns_matrix_of_all_elements_gradients_multid(self):
        xl = 0.1
        xr = 0.9
        n_elements = 2
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda _: np.linspace(9, 10, n_nodes),
            lambda _: np.linspace(99, 100, n_nodes),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)

        assert isinstance(grid.grad, np.ndarray)
        assert len(grid.grad.shape) == 3
        assert grid.grad.shape[0] == n_elements
        assert grid.grad.shape[1] == n_nodes
        assert grid.grad.shape[2] == len(ICs)
        for i in range(n_elements):
            for j in range(len(ICs)):
                assert np.allclose(grid.grad[i, :, j], grid.elements[i].grad[:, j])

    def test_shape_property_matches_state_shape_1d(self):
        xl = 0.1
        xr = 0.9
        n_elements = 4
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda _: np.linspace(9, 10, n_nodes)
        grid = Grid1D(VX, EToV, degree, IC)
        assert len(grid.shape) == 2
        assert grid.shape[0] == n_elements
        assert grid.shape[1] == n_nodes

    def test_shape_property_matches_state_shape_multid(self):
        xl = 0.1
        xr = 0.9
        n_elements = 4
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda _: np.linspace(9, 10, n_nodes),
            lambda _: np.linspace(99, 100, n_nodes),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)
        assert len(grid.shape) == 3
        assert grid.shape[0] == n_elements
        assert grid.shape[1] == n_nodes
        assert grid.shape[2] == len(ICs)

    def test_conversion_to_numpy_array_when_state_is_1d(self):
        xl = 0.1
        xr = 0.9
        n_elements = 5
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        IC = lambda _: np.linspace(3, 100, n_nodes)
        grid = Grid1D(VX, EToV, degree, IC)
        array = np.array(grid)
        assert isinstance(array, np.ndarray)
        assert array.shape == grid.state.shape
        assert np.allclose(array, grid.state)

    def test_conversion_to_numpy_array_when_state_is_multid(self):
        xl = 0.1
        xr = 0.9
        n_elements = 2
        degree = 2
        n_nodes = degree + 1
        VX, EToV = meshgen1d(xl, xr, n_elements)
        ICs = [
            lambda _: np.linspace(9, 10, n_nodes),
            lambda _: np.linspace(99, 100, n_nodes),
        ]
        grid = Grid1D(VX, EToV, degree, ICs)
        array = np.array(grid)
        assert isinstance(array, np.ndarray)
        assert array.shape == grid.state.shape
        assert np.allclose(array, grid.state)

    def test_get_time_step(self):
        xl, xr = 0, 1
        n_elements = 10
        degree = 8
        VX, EToV = meshgen1d(xl, xr, n_elements)
        grid = Grid1D(VX, EToV, degree, lambda x: np.zeros_like(x))
        dt = grid.get_time_step()
        assert np.isclose(dt, 0.00596831)


class TestCreateElements2D:
    def test_returns_list_of_elements(self):
        x1 = y1 = 0
        x2 = y2 = 2
        nx = ny = 2
        n_elements = 2 * nx * ny
        VX, VY, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        degree = 3
        IC = lambda x: 1 if len(x.shape) == 1 else np.ones(x.shape[0])
        elements = create_elements_2d(degree, VX, VY, EToV, IC)
        assert len(elements) == n_elements
        for element in elements:
            assert isinstance(element, Element2D)

    def test_returned_elements_are_not_connected(self):
        x1 = y1 = 0
        x2 = y2 = 2
        nx = ny = 2
        VX, VY, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        degree = 3
        IC = lambda x: 1 if len(x.shape) == 1 else np.ones(x.shape[0])
        elements = create_elements_2d(degree, VX, VY, EToV, IC)
        for element in elements:
            for edge in element.edges:
                assert edge._neighbour is None

    def test_elements_have_correct_vertices(self):
        degree = 2
        VX = np.array([0.0, 1.0, 1.0, 0.0])
        VY = np.array([0.0, 0.0, 1.0, 1.0])
        EToV = np.array([[0, 1, 3], [1, 2, 3]])
        IC = lambda x: 1 if len(x.shape) == 1 else np.ones(x.shape[0])
        e1, e2 = create_elements_2d(degree, VX, VY, EToV, IC)
        assert np.allclose(e1.vertices[0], np.array([0, 0]))
        assert np.allclose(e1.vertices[1], np.array([1, 0]))
        assert np.allclose(e1.vertices[2], np.array([0, 1]))
        assert np.allclose(e2.vertices[0], np.array([1, 0]))
        assert np.allclose(e2.vertices[1], np.array([1, 1]))
        assert np.allclose(e2.vertices[2], np.array([0, 1]))


class TestConnectGrid2D:
    def create_and_connect_elements(self) -> List[Element2D]:
        x1 = y1 = 0
        x2 = y2 = 1
        nx = ny = 3
        VX, VY, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        degree = 3
        IC = lambda x: np.ones(x.shape[0])
        elements = create_elements_2d(degree, VX, VY, EToV, IC)
        connect_elements_2d(elements, EToV)
        return elements

    def test_neighbour_exists_if_is_not_boundary(self):
        elements = self.create_and_connect_elements()
        for element in elements:
            for edge in element.edges:
                if not edge.is_boundary:
                    assert isinstance(edge._neighbour, ElementEdge2D)

    def test_edge_not_boundary_if_neighbour_exists(self):
        elements = self.create_and_connect_elements()
        for element in elements:
            for edge in element.edges:
                if isinstance(edge._neighbour, ElementEdge2D):
                    assert not edge.is_boundary

    def test_edge_is_boundary_if_boundary_type_exists(self):
        elements = self.create_and_connect_elements()
        for element in elements:
            for edge in element.edges:
                if edge.boundary_type is not None:
                    assert edge.is_boundary

    def test_boundary_type_exists_if_edge_is_boundary(self):
        elements = self.create_and_connect_elements()
        for element in elements:
            for edge in element.edges:
                if edge.is_boundary:
                    assert edge.boundary_type is not None

    def test_all_edges_are_either_boundary_or_interior(self):
        elements = self.create_and_connect_elements()
        for element in elements:
            for edge in element.edges:
                assert edge.is_boundary or isinstance(edge._neighbour, ElementEdge2D)

    def test_correctly_connects_two_right_angled_triangles_forming_a_square(self):
        degree = 2
        VX = np.array([0.0, 1.0, 1.0, 0.0])
        VY = np.array([0.0, 0.0, 1.0, 1.0])
        EToV = np.array([[0, 1, 3], [1, 2, 3]])
        IC = lambda x: 1 if len(x.shape) == 1 else np.ones(x.shape[0])
        elements = create_elements_2d(degree, VX, VY, EToV, IC)
        connect_elements_2d(elements, EToV)

        assert elements[0].edges[0].is_boundary
        assert not elements[0].edges[1].is_boundary
        assert elements[0].edges[2].is_boundary
        assert elements[1].edges[0].is_boundary
        assert elements[1].edges[1].is_boundary
        assert not elements[1].edges[2].is_boundary

        assert elements[0].edges[1]._neighbour == elements[1].edges[2]
        assert elements[1].edges[2]._neighbour == elements[0].edges[1]

        interior_nodes = elements[0].edges[1].nodes
        exterior_nodes = np.flip(elements[1].edges[2].nodes, axis=0)
        for i in range(degree + 1):
            distance = np.linalg.norm(interior_nodes[i] - exterior_nodes[i])
            assert np.isclose(distance, 0)

        interior_nodes = elements[1].edges[2].nodes
        exterior_nodes = np.flip(elements[0].edges[1].nodes, axis=0)
        for i in range(degree + 1):
            distance = np.linalg.norm(interior_nodes[i] - exterior_nodes[i])
            assert np.isclose(distance, 0)

    def test_correct_number_of_physical_boundaries_are_set_for_square_domain(self):
        n_elements = 128
        nx = ny = int(np.sqrt(n_elements / 2))
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, nx, ny)
        IC = lambda x: np.zeros(x.shape[0])
        elements = create_elements_2d(2, VX, VY, EToV, IC)
        connect_elements_2d(elements, EToV)
        n_physical_boundaries = 0
        for element in elements:
            for edge in element.edges:
                if edge.is_boundary:
                    n_physical_boundaries += 1
        assert n_physical_boundaries == 2 * nx + 2 * ny


class TestGrid2D:
    def test_n_elements_is_set(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        grid = Grid2D(VX, VY, EToV, 3, lambda x: np.zeros(x.shape[0]))
        assert grid.n_elements == 8

    def test_n_nodes_is_set(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        degree = 3
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.zeros(x.shape[0]))
        assert grid.n_nodes == n_nodes

    def test_state_dimension_set_correctly(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        grid = Grid2D(VX, VY, EToV, 3, lambda x: np.zeros(x.shape[0]))
        assert grid.state_dimension == 1

    def test_elements_are_created(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        grid = Grid2D(VX, VY, EToV, 3, lambda x: np.zeros(x.shape[0]))
        for element in grid.elements:
            assert isinstance(element, Element2D)
            assert not all(edge is None for edge in element.edges)

    def test_nodes_property_returns_matrix_of_all_elements_nodes(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        n_elements = EToV.shape[0]
        degree = 3
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.ones(x.shape[0]))
        assert isinstance(grid.nodes, np.ndarray)
        assert len(grid.nodes.shape) == 3
        assert grid.nodes.shape[0] == n_elements
        assert grid.nodes.shape[1] == n_nodes
        assert grid.nodes.shape[2] == 2

    def test_state_property_returns_matrix_of_all_elements_states(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        n_elements = EToV.shape[0]
        degree = 3
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.ones(x.shape[0]))
        assert isinstance(grid.state, np.ndarray)
        assert len(grid.state.shape) == 2
        assert grid.state.shape[0] == n_elements
        assert grid.state.shape[1] == n_nodes
        assert grid.state[5, 4] == 1

    def test_grad_property_returns_matrix_of_all_elements_gradients(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        n_elements = EToV.shape[0]
        degree = 3
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.ones(x.shape[0]))
        assert isinstance(grid.grad, np.ndarray)
        assert len(grid.grad.shape) == 2
        assert grid.grad.shape[0] == n_elements
        assert grid.grad.shape[1] == n_nodes
        assert grid.grad[3, 4] == 0

    def test_conversion_to_numpy_array(self):
        VX, VY, EToV = meshgen2d(0, 1, 0, 1, 2, 2)
        n_elements = EToV.shape[0]
        degree = 3
        n_nodes = int(0.5 * (degree + 1) * (degree + 2))
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.ones(x.shape[0]))
        arr = np.array(grid)
        assert isinstance(arr, np.ndarray)
        assert len(arr.shape) == 2
        assert arr.shape[0] == n_elements
        assert arr.shape[1] == n_nodes
        assert arr[5, 4] == 1

    def test_get_time_step(self):
        x0 = y0 = -1
        x1 = y1 = 1
        nx = ny = 8
        degree = 10
        VX, VY, EToV = meshgen2d(x0, x1, y0, y1, nx, ny)
        grid = Grid2D(VX, VY, EToV, degree, lambda x: np.zeros(x.shape[0]))
        dt = grid.get_time_step()
        assert np.isclose(dt, 0.0032217555)
