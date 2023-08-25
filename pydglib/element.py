from abc import ABC, abstractmethod
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
    get_outward_unit_normals_of_triangle,
)

# If initial condition is a function, then it is assumed the state is unidimensional and this function is the initial conditions for the state.
# If initial condition is a list of functions, then it is assumed that function at index i is the initial condtion for the ith component of the state vector.
# If physical dimension is 1, each function must accept a 1d numpy array of positions and return the initial condition applied pointwise.
# If physical dimension is greater than 1, each function must accept a 2d numpy array of positions and return a 1d numpy array of the initial condition applied pointwise along the zeroth axis.
InitialConditions = Union[
    Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]
]


class ElementStateContainer:
    """
    This class is for any element to store its state (ie, the nodal values) and gradients.

    This class is dimension agnostic, because elements in every dimension store their state and gradients in a 1d or 2d numpy array.
    Along the zeroth axis is the node index. If the state is multidimensional, then the first axis is the state index.
    If the state is unidimensional, then the state and gradient storage arrays are 1d.

    Attributes:
        dimension (int): The dimension of the state.
        state (np.ndarray): The state at the nodal values as a 1d or 2d numpy array.
        grad (np.ndarray): The gradients at the nodal values as a 1d or 2d numpy array.
    """

    def __init__(self, IC: InitialConditions, nodes: np.ndarray):
        """
        Creates a new ElementStateContainer instance.

        Args:
            IC (InitialConditions): Initial conditions for all state variables.
            nodes (np.ndarray): Node positions as a 2d numpy array. Zeroth axis is batch size, first axis is physical dimension.
        """
        if callable(IC):
            self.dimension = 1
            self.state = IC(nodes)
            self.grad = np.zeros_like(self.state)

        else:
            self.dimension = len(IC)
            n_nodes = nodes.shape[0]
            self.state = np.zeros((n_nodes, self.dimension))
            self.grad = np.zeros((n_nodes, self.dimension))

            for i in range(self.dimension):
                self.state[:, i] = IC[i](nodes)


class Element(ABC):
    def __init__(self, degree: int, nodes: np.ndarray, IC: InitialConditions):
        self.degree = degree
        self.nodes = nodes
        self._state_container = ElementStateContainer(IC, nodes)

    def __iadd__(self, other):
        self._state_container.state += other
        return self

    def __imul__(self, other):
        self._state_container.state *= other
        return self

    def __getitem__(self, index):
        return self._state_container.state[index]

    def __setitem__(self, index, new_value):
        self._state_container.state[index] = new_value

    def __array__(self, dtype=None) -> np.ndarray:
        return self._state_container.state

    def __len__(self) -> int:
        return self.n_nodes

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def state_dimension(self) -> int:
        return self._state_container.dimension

    @property
    def state(self) -> np.ndarray:
        return self._state_container.state

    @property
    def grad(self) -> np.ndarray:
        return self._state_container.grad

    def update_gradients(self, *gradients):
        """
        Updates the gradients of this element.

        Args:
            gradients (np.ndarray): Updated gradient vectors.
                Must supply gradients for all state dimensions.
                Each gradient vector must be a 1d numpy array and contain the derivative of the nodal value at each respective index.
        """
        assert len(gradients) == self.state_dimension

        for gradient in gradients:
            assert isinstance(gradient, np.ndarray)
            assert gradient.size == self.n_nodes

        if len(gradients) == 1:
            self._state_container.grad = gradients[0]

        else:
            for i, gradient in enumerate(gradients):
                self._state_container.grad[:, i] = gradient


class Element1D(Element):
    def __init__(
        self,
        id,
        Np,
        xl,
        xr,
        IC: InitialConditions,
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
            IC (InitialConditions): Initial conditions for the state of this element.
            left (Element1D, optional): Reference to the element's left neighbour. Defaults to None.
            right (Element1D, optional): Reference to the element's right neighbour. Defaults to None.
            dtype (np.dtype, optional): Data type of this element's state. Defaults to np.float64.
        """
        assert Np > 0
        assert xl < xr
        assert isinstance(IC, Callable) or isinstance(IC, List)

        # Create nodes
        degree = Np - 1
        nodes = computational_to_physical_1d(get_nodes_1d(degree), xl, xr)

        super().__init__(degree, nodes, IC)

        self.id = id
        self.Np = Np
        self.xl = xl
        self.xr = xr
        self.IC = IC
        self.dtype = dtype

        # Reference to element at the left and right boundaries of this element
        self.left = left
        self.right = right

    @property
    def h(self) -> float:
        return self.xr - self.xl

    def is_leftmost(self) -> bool:
        return self.left is None and self.right is not None

    def is_rightmost(self) -> bool:
        return self.left is not None and self.right is None


class Element2DInterface:
    pass


class ElementGeometry2D:
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
        self.vertices = [v1, v2, v3]
        self.area = get_area_of_triangle(v1, v2, v3)
        assert self.area > 0

        self.edge_lengths = [np.linalg.norm(e) for e in [v2 - v1, v3 - v2, v1 - v3]]
        self.perimeter = sum(self.edge_lengths)
        self.normals = get_outward_unit_normals_of_triangle(v1, v2, v3)

        # Partials of map from computational domain to physical domain
        self.xr = (v2[0] - v1[0]) / 2
        self.yr = (v2[1] - v1[1]) / 2
        self.xs = (v3[0] - v1[0]) / 2
        self.ys = (v3[1] - v1[1]) / 2

        # Determinant of the Jacobian of the transformation from computational domain to physical domain
        self.J = self.xr * self.ys - self.xs * self.yr

        assert np.isclose(self.J, self.area / 2)

        # Partials of map from physical domain to computational domain
        self.rx = self.ys / self.J
        self.ry = -self.xs / self.J
        self.sx = -self.yr / self.J
        self.sy = self.xr / self.J

        # Ratio of surface Jacobian (along edges) to the Jacobian for the 2d mapping
        self.Fscale = [l / self.area for l in self.edge_lengths]


class Element2D(Element):
    def __init__(
        self,
        degree: int,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        IC: InitialConditions,
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
            IC (InitialConditions): Initial condition(s) for the element's state.
            nodes (np.ndarray, optional): Nodes for the computational domain.
            b1_nodes (np.ndarray, optional): Inidicies of nodes that are on the first edge.
            b2_nodes (np.ndarray, optional): Inidicies of nodes that are on the second edge.
            b3_nodes (np.ndarray, optional): Inidicies of nodes that are on the third edge.
        """
        # Create nodes
        if nodes is None:
            nodes, *edge_node_indices = get_nodes_2d(degree, include_boundary=True)
        else:
            edge_node_indices = [b1_nodes, b2_nodes, b3_nodes]
        nodes = computational_to_physical_2d(nodes, v1, v2, v3)

        super().__init__(degree, nodes, IC)

        self._edge_node_indices: List[np.ndarray] = edge_node_indices

        self._geometry = ElementGeometry2D(v1, v2, v3)

        # Boundaries either save reference to an adjacent element or None, if edge is a physical boundary.
        self.edges: List[Element2DInterface | None] = [None, None, None]

    @property
    def area(self) -> float:
        return self._geometry.area

    @property
    def normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._geometry.normals

    @property
    def perimeter(self) -> float:
        return self._geometry.perimeter

    @property
    def vertices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._geometry.vertices

    @property
    def rx(self) -> float:
        return self._geometry.rx

    @property
    def ry(self) -> float:
        return self._geometry.ry

    @property
    def sx(self) -> float:
        return self._geometry.sx

    @property
    def sy(self) -> float:
        return self._geometry.sy

    @property
    def Fscale(self) -> np.ndarray:
        return self._geometry.Fscale

    def get_edge_nodes(self, edge: int) -> np.ndarray:
        assert edge in [0, 1, 2]
        return self.nodes[self._edge_node_indices[edge]]

    def get_edge(self, edge: int) -> np.ndarray:
        """Returns the state of this element on the specified edge."""
        assert edge in [0, 1, 2]
        return self.state[self._edge_node_indices[edge]]


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
        interior_node_indices: np.ndarray,
        exterior_nodes: np.ndarray,
        exterior_node_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Creates a 2d numpy array that maps indicies of interior nodes to the index of the closest exterior node.
        """
        n_nodes = interior_nodes.shape[0]
        node_map = np.zeros((n_nodes, 2), dtype=np.int32)
        node_map[:, 0] = interior_node_indices

        # Create matrix of distances between all nodes on boundary
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                distances[i, j] = np.linalg.norm(interior_nodes[i] - exterior_nodes[j])

        # Match interior nodes with closest exterior node
        closest = np.argmin(distances, axis=1)
        node_map[:, 1] = exterior_node_indices[closest]

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

        i_node_indicies = self.interior_element._edge_node_indices[self.interior_edge]
        e_node_indicies = self.exterior_element._edge_node_indices[self.exterior_edge]

        node_map = Element2DInterface._create_node_map_for_interface(
            i_nodes, i_node_indicies, e_nodes, e_node_indicies
        )

        return node_map

    def get_external_state(self) -> np.ndarray:
        node_indices = self._node_map[:, 1]
        return self.exterior_element.state[node_indices]

    def get_internal_state(self) -> np.ndarray:
        node_indices = self._node_map[:, 0]
        return self.exterior_element.state[node_indices]


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
