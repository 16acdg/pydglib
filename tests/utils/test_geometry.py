import numpy as np
from scipy.spatial import Delaunay
import pytest

from pydglib.utils.geometry import (
    is_valid_triangle,
    get_area_of_triangle,
    get_perimeter_of_triangle,
    get_outward_unit_normals_of_triangle,
)


# Valid triangles labeled with their correct surface area and perimeter
VALID_TRIANGLES = [
    # v1, v2, v3, area, perimeter, n1, n2, n3
    (
        np.array([-1, -1]),
        np.array([1, -1]),
        np.array([-1, 1]),
        2,
        4 + 2 * np.sqrt(2),
        np.array([0, -1]),
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
        np.array([-1, 0]),
    ),
    (
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([1, 1]),
        0.5,
        2 + np.sqrt(2),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ),
    (
        np.array([1.5, 1.5]),
        np.array([2, 2.2]),
        np.array([0.9, 2.4]),
        0.435,
        3.05993189809,
        np.array([1, -5 / 7]) / np.linalg.norm(np.array([1, -5 / 7])),
        np.array([1, 5.5]) / np.linalg.norm(np.array([1, 5.5])),
        np.array([-1, -2 / 3]) / np.linalg.norm(np.array([1, 2 / 3])),
    ),
]


class TestIsValidTriangle:
    triangles_with_overlapping_vertices = [
        (np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.5, 1.0])),
        (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])),
        (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])),
    ]

    @pytest.mark.parametrize("v1,v2,v3", triangles_with_overlapping_vertices)
    def test_returns_false_when_vertices_overlap(self, v1, v2, v3):
        assert not is_valid_triangle(v1, v2, v3)
        assert not is_valid_triangle(v1, v3, v2)
        assert not is_valid_triangle(v2, v1, v3)
        assert not is_valid_triangle(v3, v1, v2)
        assert not is_valid_triangle(v2, v3, v1)
        assert not is_valid_triangle(v3, v2, v1)

    triangles_with_colinear_vertices = [
        (np.array([-0.5, -0.5]), np.array([0.5, 0.5]), np.array([1.5, 1.5]))
    ]

    @pytest.mark.parametrize("v1,v2,v3", triangles_with_colinear_vertices)
    def test_returns_false_when_vertices_distinct_but_colinear(self, v1, v2, v3):
        assert not is_valid_triangle(v1, v2, v3)
        assert not is_valid_triangle(v1, v3, v2)
        assert not is_valid_triangle(v2, v1, v3)
        assert not is_valid_triangle(v3, v1, v2)
        assert not is_valid_triangle(v2, v3, v1)
        assert not is_valid_triangle(v3, v2, v1)

    @pytest.mark.parametrize("v1,v2,v3,area,perimeter,n1,n2,n3", VALID_TRIANGLES)
    def test_returns_true_when_vertices_distinct_and_not_colinear(
        self, v1, v2, v3, area, perimeter, n1, n2, n3
    ):
        assert is_valid_triangle(v1, v2, v3)
        assert is_valid_triangle(v1, v3, v2)
        assert is_valid_triangle(v2, v1, v3)
        assert is_valid_triangle(v3, v1, v2)
        assert is_valid_triangle(v2, v3, v1)
        assert is_valid_triangle(v3, v2, v1)


class TestGetAreaOfTriangle:
    def test_accepts_1d_numpy_arrays(self):
        v1 = np.array([-1, -1])
        v2 = np.array([0, 0])
        v3 = np.array([-1, 0])
        assert get_area_of_triangle(v1, v2, v3) == 0.5

    def test_returns_zero_if_vertices_colinear(self):
        v1 = np.array([-1, -1])
        v2 = np.array([0, 0])
        v3 = np.array([1, 1])
        assert np.isclose(get_area_of_triangle(v1, v2, v3), 0)

    @pytest.mark.parametrize("v1,v2,v3,area,perimeter,n1,n2,n3", VALID_TRIANGLES)
    def test_returns_correct_areas(self, v1, v2, v3, area, perimeter, n1, n2, n3):
        assert np.isclose(get_area_of_triangle(v1, v2, v3), area)


class TestGetPerimeterOfTriangle:
    def test_accepts_1d_numpy_arrays(self):
        v1 = np.array([-1, -1])
        v2 = np.array([0, 0])
        v3 = np.array([-1, 0])
        assert np.isclose(get_perimeter_of_triangle(v1, v2, v3), 2 + np.sqrt(2))

    @pytest.mark.parametrize("v1,v2,v3,area,perimeter,n1,n2,n3", VALID_TRIANGLES)
    def test_returns_correct_perimeters(self, v1, v2, v3, area, perimeter, n1, n2, n3):
        assert np.isclose(get_perimeter_of_triangle(v1, v2, v3), perimeter)


class TestGetOutwardUnitNormalsOfTriangle:
    def test_returns_three_numpy_arrays(self):
        v1 = np.array([-1, -1])
        v2 = np.array([0, 0])
        v3 = np.array([-1, 0])
        n1, n2, n3 = get_outward_unit_normals_of_triangle(v1, v2, v3)
        for n in [n1, n2, n3]:
            assert isinstance(n, np.ndarray)
            assert len(n.shape) == 1
            assert n.shape[0] == 2

    def test_returned_normals_are_orthogonal_to_edges(self):
        v1 = np.array([-1, -1])
        v2 = np.array([0, 0])
        v3 = np.array([-1, 0])
        n1, n2, n3 = get_outward_unit_normals_of_triangle(v1, v2, v3)
        assert np.isclose(np.dot(n1, v2 - v1), 0)
        assert np.isclose(np.dot(n2, v3 - v2), 0)
        assert np.isclose(np.dot(n3, v1 - v3), 0)

    def test_returned_normals_are_unit_length(self):
        v1 = np.array([-0.9, -1])
        v2 = np.array([0, 0])
        v3 = np.array([-1, 0])
        n1, n2, n3 = get_outward_unit_normals_of_triangle(v1, v2, v3)
        assert np.isclose(np.linalg.norm(n1), 1)
        assert np.isclose(np.linalg.norm(n2), 1)
        assert np.isclose(np.linalg.norm(n3), 1)

    def test_returned_normals_point_outwards_from_the_triangle(self):
        v1 = np.array([-11, -12])
        v2 = np.array([1, 0.5])
        v3 = np.array([-9, 2])
        n1, n2, n3 = get_outward_unit_normals_of_triangle(v1, v2, v3)
        tri = Delaunay(np.vstack((v1, v2, v3)))
        assert tri.find_simplex(0.5 * (v1 + v2) + n1) == -1
        assert tri.find_simplex(0.5 * (v2 + v3) + n2) == -1
        assert tri.find_simplex(0.5 * (v3 + v1) + n3) == -1

    @pytest.mark.parametrize("v1,v2,v3,area,perimeter,n1,n2,n3", VALID_TRIANGLES)
    def test_returns_correct_outward_unit_normals(
        self, v1, v2, v3, area, perimeter, n1, n2, n3
    ):
        normals = get_outward_unit_normals_of_triangle(v1, v2, v3)
        assert np.allclose(normals[0], n1)
        assert np.allclose(normals[1], n2)
        assert np.allclose(normals[2], n3)
