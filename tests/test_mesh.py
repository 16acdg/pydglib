import numpy as np
import pytest

from pydglib.mesh import meshgen1d, meshgen2d


class TestMeshgen1d:
    def test_VX_is_a_1d_numpy_array(self):
        xl = 0
        xr = 1
        n_elements = 10
        VX, _ = meshgen1d(xl, xr, n_elements)
        assert isinstance(VX, np.ndarray)
        assert len(VX.shape) == 1
        assert VX.shape[0] == n_elements + 1

    def test_VX_is_evenly_spaced(self):
        xl = 4
        xr = 7
        n_elements = 8
        VX, _ = meshgen1d(xl, xr, n_elements)
        h = VX[1] - VX[0]
        for i in range(1, n_elements):
            assert np.isclose(VX[i + 1] - VX[i], h)

    def test_VX_is_in_increasing_order(self):
        xl = 1
        xr = 3
        n_elements = 11
        VX, _ = meshgen1d(xl, xr, n_elements)
        for i in range(n_elements):
            assert VX[i] < VX[i + 1]

    def test_EToV_is_a_2d_numpy_array(self):
        xl = 0
        xr = 1
        n_elements = 10
        _, EToV = meshgen1d(xl, xr, n_elements)
        assert isinstance(EToV, np.ndarray)
        assert len(EToV.shape) == 2
        assert EToV.shape[0] == n_elements
        assert EToV.shape[1] == 2  # each element has 2 faces in 1D

    def test_EToV_array_is_type_integer(self):
        xl = 0
        xr = 1
        n_elements = 10
        _, EToV = meshgen1d(xl, xr, n_elements)
        for i in range(n_elements):
            for j in range(2):
                assert isinstance(EToV[i, j], np.integer)

    def test_vertex_numbers_for_a_single_element_differ_by_one_in_EToV(self):
        # Since 1D elements only have a left and right face and the elements are non-overlapping,
        # the id of the vertex at the element's left face must be exactly one less than the id of the vertex at the right face.
        xl = 0
        xr = 1
        n_elements = 10
        _, EToV = meshgen1d(xl, xr, n_elements)
        for i in range(n_elements):
            left_vertex = EToV[i, 0]
            right_vertex = EToV[i, 1]
            assert right_vertex - left_vertex == 1

    def test_boundary_vertices_appear_once_in_EToV(self):
        xl = 0
        xr = 2
        n_elements = 14
        _, EToV = meshgen1d(xl, xr, n_elements)
        left_boundary_vertex = 0
        right_boundary_vertex = n_elements
        assert np.count_nonzero(EToV - left_boundary_vertex) == 2 * n_elements - 1
        assert np.count_nonzero(EToV - right_boundary_vertex) == 2 * n_elements - 1

    def test_nonboundary_vertices_appear_twice_in_EToV(self):
        xl = 0
        xr = 2
        n_elements = 14
        VX, EToV = meshgen1d(xl, xr, n_elements)
        for i in range(1, n_elements):
            assert np.count_nonzero(EToV - i) == 2 * n_elements - 2


class TestMeshGen2D:
    def test_raises_exception_for_invalid_domain(self):
        INVALID_DOMAINS = [((1, 2), (1, 0)), ((2, 1), (0, 1)), ((1, 1), (0, 1))]
        for (x1, x2), (y1, y2) in INVALID_DOMAINS:
            nx = ny = 10
            with pytest.raises(Exception):
                _ = meshgen2d(x1, x2, y1, y2, nx, ny)

    def test_raises_exception_for_invalid_discretization(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        INVALID_DISCRETIZATIONS = [(10, 0), (0, 1), (-1, 4)]
        for nx, ny in INVALID_DISCRETIZATIONS:
            with pytest.raises(Exception):
                _ = meshgen2d(x1, x2, y1, y2, nx, ny)

    def test_VX_is_a_1d_numpy_array(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_vertices = (nx + 1) * (ny + 1)
        VX, _, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        assert isinstance(VX, np.ndarray)
        assert len(VX.shape) == 1
        assert VX.shape[0] == n_vertices

    def test_VY_is_a_1d_numpy_array(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_vertices = (nx + 1) * (ny + 1)
        _, VY, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        assert isinstance(VY, np.ndarray)
        assert len(VY.shape) == 1
        assert VY.shape[0] == n_vertices

    def test_VX_and_VY_have_same_size(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        VX, VY, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        assert VX.size == VY.size

    def test_VX_is_inside_physical_domain(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        VX, _, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        for x in VX:
            assert x1 <= x <= x2

    def test_VY_is_inside_physical_domain(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        _, VY, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        for y in VY:
            assert y1 <= y <= y2

    def test_corners_of_physical_domain_are_vertices_in_VX_and_VY(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        VX, VY, _ = meshgen2d(x1, x2, y1, y2, nx, ny)
        vertices = list(zip(VX, VY))
        assert (x1, y1) in vertices
        assert (x1, y2) in vertices
        assert (x2, y1) in vertices
        assert (x2, y2) in vertices

    def test_EToV_is_a_2d_numpy_array(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_elements = 2 * nx * ny
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        assert isinstance(EToV, np.ndarray)
        assert len(EToV.shape) == 2
        assert EToV.shape[0] == n_elements
        assert EToV.shape[1] == 3

    def test_EToV_array_is_type_integer(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_elements = 2 * nx * ny
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        for i in range(n_elements):
            for j in range(3):
                assert isinstance(EToV[i, j], np.integer)

    def test_EToV_contains_indicies_of_vertices(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_elements = 2 * nx * ny
        n_vertices = 2 * (nx + 1) * (ny + 1)
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        for i in range(n_elements):
            for j in range(3):
                assert 0 <= EToV[i, j] < n_vertices

    def test_each_element_in_EToV_has_distinct_vertices(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        n_elements = 2 * nx * ny
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        for i in range(n_elements):
            assert len(set(EToV[i])) == 3

    def test_top_left_and_bottom_right_vertices_appear_once_in_EToV(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        _, counts = np.unique(EToV, return_counts=True)
        assert np.count_nonzero(counts - 1) == counts.size - 2

    def test_bottom_left_and_top_right_vertices_appear_twice_in_EToV(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        _, counts = np.unique(EToV, return_counts=True)
        assert np.count_nonzero(counts - 2) == counts.size - 2

    def test_noncorner_boundary_vertices_appear_three_times_in_EToV(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        _, counts = np.unique(EToV, return_counts=True)
        assert np.count_nonzero(counts - 3) == counts.size - 2 * (nx - 1) - 2 * (ny - 1)

    def test_interior_vertices_appear_six_times_in_EToV(self):
        x1, x2 = 1, 2
        y1, y2 = -1, 1
        nx = ny = 10
        _, _, EToV = meshgen2d(x1, x2, y1, y2, nx, ny)
        _, counts = np.unique(EToV, return_counts=True)
        assert np.count_nonzero(counts - 6) == counts.size - (nx - 1) * (ny - 1)
