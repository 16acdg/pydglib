from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple, List, Union
import numpy as np

from .boundary_type import BoundaryType
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
        """Returns the length of this element"""
        return self.xr - self.xl

    @property
    def rx(self) -> float:
        return 2 / self.h

    @property
    def is_leftmost(self) -> bool:
        return self.left is None and self.right is not None

    @property
    def is_rightmost(self) -> bool:
        return self.left is not None and self.right is None


class Element2DInterface:
    pass


class ElementEdge2D:
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
        # self.edges: List[Element2DInterface | None] = [None, None, None]
        self.n_faces = 3
        self.edges = [
            ElementEdge2D(
                i,
                self._state_container,
                self._geometry,
                self.nodes,
                self._edge_node_indices[i],
            )
            for i in range(self.n_faces)
        ]

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


class ElementEdge2D:
    """
    This class represents one of 3 straight edges of an Element2D instance.

    This class provides an API for accessing variables related to this edge.
    All ElementEdge2D instances are assumed to have counterclockwise orientation, meaning that all nodes and states are returned in this order.

    Attributes:
        index (int): Local id of this edge within an Element2D instance (either 0, 1, or 2).
        is_boundary (bool): True if this edge is a physical boundary, False otherwise.
        boundary_type (BoundaryType | None): If this edge is a physical boundary, then this is the type of physical boundary.
        n_nodes (int): The number of nodes along this edge. For an Element2D instance with degree of local polynomial approximation `d`, `n_nodes = d + 1`.
        nodes (np.ndarray): The nodes that are on this edge, in counterclockwise orientation.
        state (np.ndarray): The nodal values along this edge, in counterclockwise orientation.
        external_state (np.ndarray): The nodal values at nodes of an element adjacent to this edge, in counterclockwise orientation.
        normal (np.ndarray): The outward unit normal vector to this edge.
    """

    def __init__(
        self,
        index: int,
        state: ElementStateContainer,
        geometry: ElementGeometry2D,
        nodes: np.ndarray,
        edge_node_indices: np.ndarray,
        boundary_type: BoundaryType = None,
    ):
        """
        Creates a new ElementEdge2D.

        This class represents one of the 3 edges of a 2d triangular element.
        This class keeps a reference to the neighbouring element along this edge, if such an element exists, and provides an API to access it.

        Args:
            index (int): Parent Element2D's local id for this edge. Must be in [0, 1, 2].
            state (ElementStateContainer): Reference to the parent Element2D's state container.
            geometry (ElementGeometry2D): Reference to the parent Element2D's geometry.
            nodes (np.ndarray): Parent element's nodes.
            edge_node_indices (np.ndarray): 1d array of indices in `nodes` that are on this edge, in counterclockwise orientation.
            boundary_type (BoundaryType, optional): If this edge is a physical boundary, this is the type of physical boundary. Defaults to None.
        """
        self.index = index
        self._state = state
        self._geometry = geometry
        self._nodes = nodes
        self._edge_node_indices = edge_node_indices
        self.is_boundary: bool = boundary_type is not None
        self.boundary_type: BoundaryType | None = boundary_type
        self._neighbour: ElementEdge2D | None = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes along this edge."""
        return len(self._edge_node_indices)

    @property
    def nodes(self) -> np.ndarray:
        """Returns the nodes along this edge, in counterclockwise orientation."""
        return self._nodes[self._edge_node_indices]

    @property
    def state(self) -> np.ndarray:
        """Returns the state of the nodes along this edge, in counterclockwise orientation."""
        return self._state.state[self._edge_node_indices]

    @property
    def external_state(self) -> np.ndarray:
        """Returns the state of the neighbouring element, if such an element exists, in counterclockwise orientation."""
        if self._neighbour is None:
            raise Exception("No element is adjacent to this element along this edge.")

        # NOTE: Because all Element2Ds are assumed to be oriented counterclockwise, the state along any neighbouring edge will be backwards, so flip it.
        return np.flip(self._neighbour.state, axis=0)

    @property
    def normal(self) -> np.ndarray:
        return self._geometry.normals[self.index]


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
