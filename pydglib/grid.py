from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Union
import numpy as np

from pydglib.element import (
    Element,
    Element1D,
    Element2D,
    InitialConditions,
)
from pydglib.utils.nodes import get_nodes_1d, get_nodes_2d
from .boundary_type import BoundaryType


class Grid(ABC):
    def __init__(self, elements: List[Element], physical_dimension: int):
        self.elements = elements
        self.physical_dimension = physical_dimension

    @property
    def degree(self) -> int:
        return self.elements[0].degree if len(self.elements) > 0 else 0

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_nodes(self) -> int:
        return self.elements[0].n_nodes if len(self.elements) > 0 else 0

    @property
    def state_dimension(self) -> int:
        return self.elements[0].state_dimension if len(self.elements) > 0 else 0

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the grid's state."""
        if self.state_dimension == 1:
            return (self.n_elements, self.n_nodes)
        else:
            return (self.n_elements, self.n_nodes, self.state_dimension)

    @property
    def nodes(self) -> np.ndarray:
        """Returns the node positions of all elements in this grid as a (`n_elements`, `n_nodes`, `physical_dimension`) shaped numpy array."""
        if self.physical_dimension == 1:
            out_shape = (self.n_elements, self.n_nodes)
        else:
            out_shape = (self.n_elements, self.n_nodes, self.physical_dimension)
        out = np.zeros(out_shape)
        for i, element in enumerate(self.elements):
            out[i] = element.nodes
        return out

    @property
    def state(self) -> np.ndarray:
        """Returns the state of all elements in this grid as a (`n_elements`, `n_nodes`, state_dimension) sized numpy array."""
        return self.__array__()

    @property
    def grad(self) -> np.ndarray:
        """Returns the gradients of all elements in this grid as a (`n_elements`, `n_nodes`, state_dimension) sized numpy array."""
        out = np.zeros(self.shape)
        for i, element in enumerate(self.elements):
            out[i] = element.grad
        return out

    def __array__(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts this Grid instance into a numpy array containing the grid's state.

        Args:
            dtype (np.dtype, optional): Data type of the returned numpy array. Defaults to float32.

        Returns:
            np.ndarray: This Grid instance as a numpy array.
        """
        out = np.zeros(self.shape, dtype=dtype)
        for i, element in enumerate(self.elements):
            out[i] = element.state
        return out

    @abstractmethod
    def get_time_step(self) -> float:
        """Returns an appropriate time step for the given grid."""
        pass


def create_elements_1d(
    degree: int, VX: np.ndarray, EToV: np.ndarray, IC: InitialConditions
) -> List[Element1D]:
    elements = []
    n_nodes = degree + 1
    n_elements = EToV.shape[0]
    for i in range(n_elements):
        xl = VX[i]
        xr = VX[i + 1]
        element = Element1D(i, n_nodes, xl, xr, IC)
        elements.append(element)
    return elements


def connect_elements_1d(elements: List[Element1D], EToV: np.ndarray):
    k = 1
    el_index = np.where(EToV[:, 0] == 0)[0][0]
    next_v_index = EToV[el_index, 1]
    prev_el = elements[el_index]
    while k < len(elements):
        el_index = np.where(EToV[:, 0] == next_v_index)[0][0]
        elements[el_index].left = prev_el
        prev_el.right = elements[el_index]
        prev_el = elements[el_index]
        next_v_index = EToV[el_index, 1]
        k += 1


class Grid1D(Grid):
    def __init__(
        self,
        VX: np.ndarray,
        EToV: np.ndarray,
        degree: int,
        IC: InitialConditions,
    ):
        """
        Creates a new Grid1D instance.

        Args:
            VX (np.ndarray): Positions of vertices in the physical domain.
            EToV (np.ndarray): Element-to-vertex map.
            degree (int): Degree of the local polynomial approximation.
            IC (InitialConditions): Initial conditions for state.
        """
        elements = create_elements_1d(degree, VX, EToV, IC)

        super().__init__(elements, 1)

        self.xl = VX[0]
        self.xr = VX[-1]
        self.VX = VX  # TODO: Keep as instance attribute?
        self.EToV = EToV  # TODO: Remove as instance attribute
        self.IC = IC  # TODO: Remove as instance attribute

        connect_elements_1d(self.elements, self.EToV)

    def get_time_step(self) -> float:
        x = self.nodes
        xmin = np.min(np.abs(x[:, 0] - x[:, 1]))
        CFL = 0.75
        dt = xmin * CFL / (4 * np.pi)
        return dt


def create_elements_2d(degree, VX, VY, EToV, IC) -> List[Element2D]:
    """
    Creates Element2D instances from the given mesh specification.

    Args:
        degree (int): Degree of the local polynomial approximations.
        VX (np.ndarray): 1d numpy array containing the x coordinates of mesh vertices.
        VY (np.ndarray): 1d numpy array containing the y coordinates of mesh vertices.
        EToV (np.ndarray): Table mapping global element ids to global ids of vertices that compose the element.
        IC (Callable): Initial condition.

    Returns:
        List[Element2D]: All elements for the specified mesh. Returned elements are disconnected,
            in the sense that local information has been set in each Element2D instance,
            but no references have been made between adjacent elements in the mesh.
    """
    nodes, b1_nodes, b2_nodes, b3_nodes = get_nodes_2d(degree, include_boundary=True)

    n_elements = EToV.shape[0]
    elements = []
    for i in range(n_elements):
        vertex_indices = EToV[i]
        vertices = [np.array([VX[j], VY[j]]) for j in vertex_indices]
        element = Element2D(degree, *vertices, IC, nodes, b1_nodes, b2_nodes, b3_nodes)
        elements.append(element)
    return elements


def _get_adjacent_element(element_index, vi, vf, EToV) -> Tuple[int, int]:
    """
    Returns the global element index and local face index of an adjacent element.

    The adjacent element is with respect to the element with index `element_index` and adjacency
    is along the edge `v1`->`v2`, where `v1` and `v2` are local vertex indicies.

    Args:
        element_index (int): Index of the element.
        vi (int): Local vertex index of the edge `vi`->`vf`. Short for "vertex initial".
        vf (int): Local vertex index of the edge `vi`->`vf`. Short for "vertex final".
        EToV (np.ndarray): Edge to vertex map.

    Returns:
        int: The global element id of the element adjacent along `v1`->`v2`, if such an element exists. Otherwise -1.
        int: The neighbouring element's local face index, if such an element exists. Otherwise -1.
    """
    n_elements = EToV.shape[0]
    edge = set([vi, vf])
    for i in range(n_elements):
        if i != element_index:
            if set([EToV[i, 0], EToV[i, 1]]) == edge:
                face_index = 0
                return i, face_index
            elif set([EToV[i, 1], EToV[i, 2]]) == edge:
                face_index = 1
                return i, face_index
            elif set([EToV[i, 2], EToV[i, 0]]) == edge:
                face_index = 2
                return i, face_index
    return -1, -1


def connect_elements_2d(elements: List[Element2D], EToV: np.ndarray):
    """
    Makes connections between Element2D instances via Element2DInterface objects.

    Args:
        elements (List[Element2D]): Unconnected elements. Index in this list corresponds with index in `EToV`.
        EToV (np.ndarray): Element to vertex map.
    """
    for i, element in enumerate(elements):
        # Extract global vertex numbers for element i
        v1, v2, v3 = EToV[i]

        for edge, (vi, vf) in zip(element.edges, [(v1, v2), (v2, v3), (v3, v1)]):
            ext_element_idx, ext_edge_idx = _get_adjacent_element(i, vi, vf, EToV)
            edge_is_physical_boundary = ext_element_idx == -1

            if edge_is_physical_boundary:
                edge.is_boundary = True
                edge.boundary_type = (
                    BoundaryType.DIRICHLET
                )  # TODO: Set this based on a new argument to this function
            else:
                # Get reference to exterior element (ie the element along the current edge)
                exterior_element = elements[ext_element_idx]

                # Save reference to the exterior element's edge that coincides with the current edge
                edge._neighbour = exterior_element.edges[ext_edge_idx]


class Grid2D(Grid):
    def __init__(
        self,
        VX: np.ndarray,
        VY: np.ndarray,
        EToV: np.ndarray,
        degree: int,
        IC: InitialConditions,
    ):
        """
        Creates a new Grid2D instance.

        Args:
            VX (np.ndarray): x coordinates of grid vertices.
            VY (np.ndarray): y coordinates of grid vertices.
            EToV (np.ndarray): Element-to-vertex map.
            degree (int): Degree of the local polynomial approximation.
            IC (InitialConditions): Initial conditions for state.
        """
        # Create elements
        elements = create_elements_2d(degree, VX, VY, EToV, IC)

        super().__init__(elements, 2)

        self.VX = VX
        self.VY = VY
        self.EToV = EToV
        self.IC = IC

        # Create graph of elements (ie set references between adjacent elements)
        connect_elements_2d(self.elements, self.EToV)

    def get_time_step(self) -> float:
        r = get_nodes_1d(self.degree)
        rmin = abs(r[0] - r[1])

        dtscale = np.zeros(len(self.elements))
        for i, element in enumerate(self.elements):
            inscribed_radius = 2 * element.area / element.perimeter
            dtscale[i] = inscribed_radius

        dt = np.min(dtscale) * rmin * 2 / 3

        return dt
