from enum import Enum
from typing import Callable, Tuple, List, Union
import numpy as np

from .utils.nodes import get_nodes_1d, get_nodes_2d
from .utils.transformations import (
    computational_to_physical_1d,
    computational_to_physical_2d,
)
from .utils.geometry import (
    get_area_of_triangle,
    get_perimeter_of_triangle,
    get_outward_unit_normals_of_triangle,
)


class Element1D:
    def __init__(
        self,
        id,
        Np,
        xl,
        xr,
        IC,
        left=None,
        right=None,
        dtype=np.float64,
    ):
        """
        Create a new Element1D instance.

        Args:
            id (int): Global identifier for this element.
            Np (int): Number of nodes for this element.
            xl (float): Left position of element in physical domain.
            xr (float): Right position of element in physical domain.
            IC (Callable or List[Callable]): Initial conditions for the state of this element.
                If state's dimension = 1, then IC must be a function.
                If state's dimension > 1, then IC must be a list of functions, one for each dimension of state.
            left (Element1D, optionl): Reference to the element's left neighbour. Defaults to None.
            right (Element1D, optionl): Reference to the element's right neighbour. Defaults to None.
            dtype (np.dtype, optional): Data type of this element's state. Defaults to np.float64.
        """
        assert Np > 0
        assert xl < xr
        assert isinstance(IC, Callable) or isinstance(IC, List)

        self.id = id
        self.Np = Np
        self.xl = xl
        self.xr = xr
        self.IC = IC
        self.dtype = dtype

        self.state_dimension = 1 if isinstance(self.IC, Callable) else len(self.IC)
        self.nodes = computational_to_physical_1d(
            get_nodes_1d(self.Np - 1), self.xl, self.xr
        )
        state_shape = (
            self.Np if self.state_dimension == 1 else (self.state_dimension, self.Np)
        )
        self.state = np.zeros(state_shape, dtype=self.dtype)
        self.grad = np.zeros_like(self.state)

        # Apply initial conditions to state
        if self.state_dimension == 1:
            self.state = self.IC(self.nodes)
        elif self.state_dimension > 1:
            for i in range(self.state_dimension):
                self.state[i] = self.IC[i](self.nodes)

        # Reference to element at the left and right boundaries of this element
        self.left = left
        self.right = right

    @property
    def h(self) -> float:
        return self.xr - self.xl

    def __repr__(self) -> str:
        return f"Element({self.state}, grad={self.grad})"

    def __iadd__(self, other):
        self.state += other
        return self

    def __imul__(self, other):
        self.state *= other
        return self

    def __getitem__(self, index):
        return self.state[index]

    def __setitem__(self, index, new_value):
        self.state[index] = new_value

    def __array__(self, dtype=None) -> np.ndarray:
        return self.state

    def __len__(self) -> int:
        return self.Np

    def is_leftmost(self) -> bool:
        return self.left is None and self.right is not None

    def is_rightmost(self) -> bool:
        return self.left is not None and self.right is None


class Element2DInterface:
    pass


class Element2D:
    def __init__(
        self,
        degree: int,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        IC: Callable[[np.ndarray], np.ndarray],
        nodes: np.ndarray = None,
        b1_nodes=None,
        b2_nodes=None,
        b3_nodes=None,
    ):
        """
        Create a new Element2D instance.

        The vertices `v1`, `v2`, `v3` must be 1d numpy arrays.
        Each array must be length 2; the first entry is the x position, the second the y position.

        Note that it is not strictly necessary for the nodes to be an argument for initializing an Element2D instance,
        as the only argument needed to produce nodes for the computational domain is the degree of the polynomial approximation.
        However, supplying the nodes is much faster when a large number of elements must be created.

        Args:
            degree (int): The degree of the local polynomial approximation.
            v1 (np.ndarray): First vertex of a triangle.
            v2 (np.ndarray): Second vertex of a triangle.
            v3 (np.ndarray): Third vertex of a triangle.
            IC (Callable[[np.ndarray], np.ndarray]): Initial conditions. Input array must be 1d or 2d.
                If input array is 1d, then the array is a single node position and `IC` must return the initial condition at this position.
                If input array is 2d, then the array is a batch of node positions; 0th axis is the batch size, 1st axis is the physical dimension.
            nodes (np.ndarray, optional): Nodes for the computational domain.
            b1_nodes (np.ndarray, optional): Inidicies of nodes that are on the first edge.
            b2_nodes (np.ndarray, optional): Inidicies of nodes that are on the second edge.
            b3_nodes (np.ndarray, optional): Inidicies of nodes that are on the third edge.
        """
        self.degree = degree
        self.vertices = [v1, v2, v3]

        # Calculate the surface area and perimeter of the triangle v1, v2, v3.
        self.perimeter = get_perimeter_of_triangle(v1, v2, v3)
        self.area = get_area_of_triangle(v1, v2, v3)

        # Radius of the inscribed circle for the triangle v1, v2, v3.
        self.inscribed_radius = 2 * self.area / self.perimeter

        # Set outward unit normals of triangle v1, v2, v3.
        n1, n2, n3 = get_outward_unit_normals_of_triangle(v1, v2, v3)
        self.normals = [n1, n2, n3]

        # Boundaries either save reference to an adjacent element or None, if edge is a physical boundary.
        self.edges: List[Element2DInterface | None] = [None, None, None]

        # Create nodes for this element
        if nodes is None:
            self.nodes, b1_nodes, b2_nodes, b3_nodes = get_nodes_2d(
                self.degree, include_boundary=True
            )
        else:
            self.nodes = nodes
        self.nodes = computational_to_physical_2d(self.nodes, v1, v2, v3)
        self.n_nodes = self.nodes.shape[0]
        self._edge_node_indicies: List[np.ndarray] = [b1_nodes, b2_nodes, b3_nodes]

        # Initialize dimension of the state of this element
        self.state_dimension = 1 if isinstance(IC, Callable) else len(IC)
        state_shape = (
            self.n_nodes
            if self.state_dimension == 1
            else (self.state_dimension, self.n_nodes)
        )
        self.state = np.zeros(state_shape)
        self.grad = np.zeros_like(self.state)

        # Apply initial condition(s)
        if self.state_dimension == 1:
            self.state = IC(self.nodes)
        elif self.state_dimension > 1:
            for i in range(self.state_dimension):
                self.state[i] = IC[i](self.nodes)

    def get_edge_nodes(self, edge: int) -> np.ndarray:
        assert edge in [0, 1, 2]
        return self.nodes[self._edge_node_indicies[edge]]

    def get_edge(self, edge: int) -> np.ndarray:
        """Returns the state of this element on the specified edge."""
        assert edge in [0, 1, 2]
        if self.state_dimension == 1:
            return self.state[self._edge_node_indicies[edge]]
        else:
            return self.state[:, self._edge_node_indicies[edge]]

    # def get_edge_external(self, edge: int) -> np.ndarray:
    #     """Returns the neighbouring element's state along the given edge, if such an element exist. Otherwise raises exception."""
    #     assert edge in [0, 1, 2]
    #     edge_is_boundary = self.edges[edge] is None
    #     if edge_is_boundary:
    #         raise Exception("Attempting to access adjacent element that doesn't exist.")
    #     else:
    #         return self.edges[edge].get_external_state()

    def __iadd__(self, other):
        self.state += other
        return self

    def __imul__(self, other):
        self.state *= other
        return self

    def __getitem__(self, index) -> np.ndarray:
        return self.state[index]

    def __setitem__(self, index, new_value):
        self.state[index] = new_value

    def __array__(self, dtype=None) -> np.ndarray:
        return self.state

    def __len__(self) -> int:
        return self.n_nodes

    def __repr__(self) -> str:
        return f"Element2D(n_nodes={self.n_nodes})"


class Element2DInterface:
    def __init__(
        self,
        interior_element: Element2D,
        exterior_element: Element2D,
        interior_edge: int,
        exterior_edge: int,
    ):
        """
        Create an Element2DInterface instance.

        Args:
            interior_element (Element2D): Reference to the interior element.
            exterior_element (Element2D): Reference to the neighbouring element.
            interior_edge ([0, 1, 2]): Interior element's local ID for the edge at this interface.
            exterior_edge ([0, 1, 2]): Neighbour's local ID for the edge at this interface.
        """
        assert interior_edge in [0, 1, 2]
        assert exterior_edge in [0, 1, 2]

        self.interior_element = interior_element
        self.exterior_element = exterior_element
        self.interior_edge = interior_edge
        self.exterior_edge = exterior_edge

        self._node_map = self._create_node_map()

    @staticmethod
    def _create_node_map_for_interface(
        interior_nodes: np.ndarray,
        interior_node_indicies: np.ndarray,
        exterior_nodes: np.ndarray,
        exterior_node_indicies: np.ndarray,
    ) -> np.ndarray:
        """
        Creates a 2d numpy array that maps indicies of interior nodes to the index of the closest exterior node.
        """
        n_nodes = interior_nodes.shape[0]
        node_map = np.zeros((n_nodes, 2), dtype=np.int32)
        node_map[:, 0] = interior_node_indicies

        # Create matrix of distances between all nodes on boundary
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                distances[i, j] = np.linalg.norm(interior_nodes[i] - exterior_nodes[j])

        # Match interior nodes with closest exterior node
        closest = np.argmin(distances, axis=1)
        node_map[:, 1] = exterior_node_indicies[closest]

        return node_map

    def _create_node_map(self) -> np.ndarray:
        """
        Creates 2d numpy array that maps nodes that pairs adjacent nodes from neighbouring elements.

        First dimension is the number of nodes along a boundary. Second dimension equals 2.
        First column contains the local indicies of the nodes that appear along the interior of the interface.
        Second column contains the local indicies of the neighbour's nodes that appear along the exterior of the interface.
        The two indicies that appear in the same row correspond with nodes that are adjacent.

        Returns:
            np.ndarray: Node map as a 2d numpy array.
        """
        i_nodes = self.interior_element.get_edge_nodes(self.interior_edge)
        e_nodes = self.exterior_element.get_edge_nodes(self.exterior_edge)

        i_node_indicies = self.interior_element._edge_node_indicies[self.interior_edge]
        e_node_indicies = self.exterior_element._edge_node_indicies[self.exterior_edge]

        node_map = Element2DInterface._create_node_map_for_interface(
            i_nodes, i_node_indicies, e_nodes, e_node_indicies
        )

        return node_map

    def get_external_state(self) -> np.ndarray:
        node_indicies = self._node_map[:, 1]
        if self.exterior_element.state_dimension == 1:
            return self.exterior_element.state[node_indicies]
        else:
            return self.exterior_element.state[:, node_indicies]

    def get_internal_state(self) -> np.ndarray:
        node_indicies = self._node_map[:, 0]
        if self.exterior_element.state_dimension == 1:
            return self.interior_element.state[node_indicies]
        else:
            return self.interior_element.state[:, node_indicies]


def get_reference_triangle(degree: int) -> Element2D:
    """
    Returns the reference triangle ConvHull{(-1, -1), (1, -1), (-1, 1)} as an Element2D instance.

    Args:
        degree (int): Degree of the local polynomial approximation.

    Returns:
        Element2D: The reference triangle.
    """
    return Element2D(
        degree,
        np.array([-1, -1]),
        np.array([1, -1]),
        np.array([-1, 1]),
        lambda x: np.zeros(x.shape[0]),
    )
