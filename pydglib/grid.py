from typing import Tuple, Callable, List, Union
import numpy as np

from pydglib.element import Element1D, Element2D, Element2DInterface
from pydglib.utils.nodes import get_nodes_2d


class Grid1D:
    def __init__(
        self,
        VX: np.ndarray,
        EToV: np.ndarray,
        n_nodes: int,
        IC: Union[Callable, List[Callable]],
        dtype=np.float64,
    ):
        """
        Creates a new Grid1D instance.

        The state of the grid is stored in elements that are attached to this grid.
        The grid assumes the state of all elements is 1D if `IC` is functions.
        If instead `IC` is a List of functions, then it is assumed that each function is for a different
        dimension of the state, and hence the dimension of the state is assumed to be `len(IC)`.

        Args:
            VX (np.ndarray): Positions of vertices in the physical domain.
            EToV (np.ndarray): Element-to-vertex map.
            n_nodes (int): Number of nodes per element.
            IC (Union[Callable, List[Callable]]): Initial conditions for state.
                If state's dimension is 1, then IC must be a function.
                If state's dimension is greater than 1, then IC must be a list of functions, one for each state dimension.
            dtype (np.dtype, optional): Data type of the state. Defaults to np.float64.
        """
        assert isinstance(VX, np.ndarray)
        assert isinstance(EToV, np.ndarray)
        assert isinstance(n_nodes, int) and n_nodes > 0
        assert len(VX.shape) == 1
        assert len(EToV.shape) == 2
        assert VX.shape[0] == EToV.shape[0] + 1
        assert EToV.shape[1] == 2
        # Assume VX is given in increasing order
        for i in range(VX.size - 1):
            assert VX[i] < VX[i + 1]

        self.n_nodes = n_nodes
        self.n_elements = EToV.shape[0]
        self.xl = VX[0]
        self.xr = VX[-1]
        self.dtype = dtype
        self.VX = VX
        self.EToV = EToV
        self.IC = IC
        self.state_dimension = 1 if isinstance(self.IC, Callable) else len(self.IC)

        self.elements: List[Element1D] = []
        for k in range(self.n_elements):
            xl = VX[k]
            xr = VX[k + 1]
            element = Element1D(k, n_nodes, xl, xr, IC, dtype=dtype)
            self.elements.append(element)
        self._create_references_between_elements()

    def _create_references_between_elements(self):
        """
        Sets left and right pointers of elements that are adjacent in the `self.elements` list.
        """
        k = 1
        el_index = np.where(self.EToV[:, 0] == 0)[0][0]
        next_v_index = self.EToV[el_index, 1]
        prev_el = self.elements[el_index]
        while k < self.n_elements:
            el_index = np.where(self.EToV[:, 0] == next_v_index)[0][0]
            self.elements[el_index].left = prev_el
            prev_el.right = self.elements[el_index]
            prev_el = self.elements[el_index]
            next_v_index = self.EToV[el_index, 1]
            k += 1

    @property
    def nodes(self) -> np.ndarray:
        """Returns the node positions of all elements in this grid as a 2d numpy array."""
        out = np.zeros((self.n_nodes, self.n_elements), dtype=self.dtype)
        for k in range(self.n_elements):
            out[:, k] = self.elements[k].nodes
        return out

    @property
    def state(self) -> np.ndarray:
        """Returns the state of all elements in this grid as a 2d numpy array."""
        return self.__array__()

    @property
    def grad(self) -> np.ndarray:
        """Returns the gradients of all elements in this grid as a 2d numpy array."""
        if self.state_dimension == 1:
            out = np.zeros((self.n_nodes, self.n_elements), dtype=self.dtype)
            for k in range(self.n_elements):
                out[:, k] = self.elements[k].grad
        else:
            out = np.zeros(
                (self.state_dimension, self.n_nodes, self.n_elements), dtype=self.dtype
            )
            for k in range(self.n_elements):
                for i in range(self.state_dimension):
                    out[i, :, k] = self.elements[k].grad[i]
        return out

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the grid's state."""
        if self.state_dimension == 1:
            return (self.n_nodes, self.n_elements)
        else:
            return (self.state_dimension, self.n_nodes, self.n_elements)

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Converts this Grid1D instance into a numpy array containing the grid's state.

        Args:
            dtype (np.dtype, optional): Data type of the returned numpy array

        Returns:
            np.ndarray: This Grid1D instance as a numpy array.
        """
        if dtype is None:
            dtype = self.dtype
        if self.state_dimension == 1:
            out = np.zeros((self.n_nodes, self.n_elements), dtype=dtype)
            for k in range(self.n_elements):
                out[:, k] = self.elements[k].state
        else:
            out = np.zeros(
                (self.state_dimension, self.n_nodes, self.n_elements), dtype=dtype
            )
            for k in range(self.n_elements):
                for i in range(self.state_dimension):
                    out[i, :, k] = self.elements[k].state[i]
        return out

    def __repr__(self) -> str:
        return f"Grid1D({self.state}, grad={self.grad})"


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


def _get_index_of_adjacent_element(element_index, vi, vf, EToV) -> Tuple[int, int]:
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
        # Get global vertex numbers for element i
        v1, v2, v3 = EToV[i]

        edges = [(v1, v2), (v2, v3), (v3, v1)]

        for j, edge in enumerate(edges):
            # Get id of element that is adjacent to the current edge.
            # Also get the local id of the adjacent element's edge.
            ext_element_idx, ext_edge_idx = _get_index_of_adjacent_element(
                i, edge[0], edge[1], EToV
            )
            edge_is_physical_boundary = ext_element_idx == -1

            if not edge_is_physical_boundary:
                ext_element = elements[ext_element_idx]

                # Create Element2DInterface instance for the interior element's edge
                if element.edges[j] is None:
                    element.edges[j] = Element2DInterface(
                        element, ext_element, j, ext_edge_idx
                    )

                # Create Element2DInterface instance for the exterior element's edge
                if ext_element.edges[ext_edge_idx] is None:
                    ext_element.edges[ext_edge_idx] = Element2DInterface(
                        ext_element, element, ext_edge_idx, j
                    )


class Grid2D:
    def __init__(
        self,
        VX: np.ndarray,
        VY: np.ndarray,
        EToV: np.ndarray,
        degree: int,
        IC: Union[Callable, List[Callable]],
    ):
        """
        Creates a new Grid2D instance.

        Args:
            VX (np.ndarray): x coordinates of grid vertices.
            VY (np.ndarray): y coordinates of grid vertices.
            EToV (np.ndarray): Element-to-vertex map.
            degree (int): Degree of the local polynomial approximation.
            IC (Callable): Initial conditions. Must support 1d and 2d array inputs.
                If input to `IC` is 1d, then the array must be length 2 and the function must return a float.
                If input to `IC` is 2d, the second dimension of the array must be length 2 and it must return a 1d numpy array,
                where the length of the 1d array equals the size of the input array's first dimension.
        """
        self.VX = VX
        self.VY = VY
        self.EToV = EToV
        self.degree = degree
        self.IC = IC

        self.n_elements = EToV.shape[0]
        self.n_edge_nodes = degree + 1  # per element
        self.n_nodes = int(0.5 * (degree + 1) * (degree + 2))  # per element
        self.state_dimension = 1 if isinstance(self.IC, Callable) else len(self.IC)

        # Create graph of elements (ie set references between adjacent elements)
        self.elements: List[Element2D] = create_elements_2d(degree, VX, VY, EToV, IC)
        connect_elements_2d(self.elements, self.EToV)

    @property
    def nodes(self) -> np.ndarray:
        """Returns the positions of all nodes of all elements in this grid as a 2d numpy array."""
        out = np.zeros((self.n_nodes, self.n_elements, 2))
        for i, element in enumerate(self.elements):
            out[:, i, :] = element.nodes
        return out

    @property
    def state(self) -> np.ndarray:
        """Returns the state of all elements in this grid as a 2d numpy array."""
        return self.__array__()

    @property
    def grad(self) -> np.ndarray:
        """Returns the gradients of all elements in this grid as a 2d numpy array."""
        if self.state_dimension == 1:
            out = np.zeros((self.n_nodes, self.n_elements))
            for i, element in enumerate(self.elements):
                out[:, i] = element.grad
        else:
            out = np.zeros((self.state_dimension, self.n_nodes, self.n_elements))
            for k in range(self.n_elements):
                for i in range(self.state_dimension):
                    out[i, :, k] = self.elements[k].grad[i]
        return out

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the grid's state."""
        if self.state_dimension == 1:
            return (self.n_nodes, self.n_elements)
        else:
            return (self.state_dimension, self.n_nodes, self.n_elements)

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        """
        Converts this Grid2D instance into a numpy array containing the grid's state.

        Args:
            dtype (np.dtype, optional): Data type of the returned numpy array

        Returns:
            np.ndarray: This Grid2D instance as a numpy array.
        """
        if self.state_dimension == 1:
            out = np.zeros((self.n_nodes, self.n_elements))
            for i, element in enumerate(self.elements):
                out[:, i] = element.state
        else:
            out = np.zeros((self.state_dimension, self.n_nodes, self.n_elements))
            for k in range(self.n_elements):
                for i in range(self.state_dimension):
                    out[i, :, k] = self.elements[k].state[i]
        return out

    def __repr__(self) -> str:
        return f"Grid2D({self.state}, grad={self.grad})"
