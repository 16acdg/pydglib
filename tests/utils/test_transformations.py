import pytest
import numpy as np

from pydglib.utils.transformations import (
    computational_to_physical_1d,
    physical_to_computational_1d,
    computational_to_physical_2d,
    physical_to_computational_2d,
)


# Samples of physical domains, a point in the physical domain, and the corresponding point in the
# computational domain under the affine map defined by -1 <-> xl and +1 <-> xr.
AFFINE_SAMPLES_1D = [
    # (r, x, xl, xr)
    (0.33, 0.33, -1, 1),
    (0, 1.5, 1, 2),
    (0.5, -1, -10, 2),
    (-0.5, -7, -10, 2),
    (0.7, 8.5, 0, 10),
]
AFFINE_SAMPLES_2D = [
    # (r, x, v1, v2, v3)
    (
        np.array([-0.5, -0.5]),
        np.array([0.375, 0.25]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 1.0]),
    ),
    (
        np.array([-1, -1]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 1.0]),
    ),
    (
        np.array([1, -1]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 1.0]),
    ),
    (
        np.array([-1, 1]),
        np.array([0.5, 1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 1.0]),
    ),
    (
        np.array([-1 / 3, -1 / 3]),  # centroid of reference triangle
        np.array([22 / 15, 61 / 30]),  # centroid of other triangle
        np.array([1.5, 1.5]),
        np.array([2, 2.2]),
        np.array([0.9, 2.4]),
    ),
]


class TestComputationalToPhysical1D:
    def test_raises_exception_for_invalid_physical_domain(self):
        xl = 2
        xr = 1
        with pytest.raises(Exception):
            _ = computational_to_physical_1d(0, xl, xr)

    def test_returns_float_when_inputs_is_float(self):
        xl = 3
        xr = 4
        r = -0.5
        x = computational_to_physical_1d(r, xl, xr)
        assert isinstance(x, (int, float))

    def test_returns_numpy_array_when_input_is_numpy_array(self):
        xl = 0.8
        xr = 0.9
        r = np.linspace(-1, 1, 7)
        x = computational_to_physical_1d(r, xl, xr)
        assert isinstance(x, np.ndarray)
        assert x.shape == r.shape

    @pytest.mark.parametrize("r,x,xl,xr", AFFINE_SAMPLES_1D)
    def test_affine_mapping_is_correct(self, r, x, xl, xr):
        assert np.isclose(computational_to_physical_1d(r, xl, xr), x)


class TestPhysicalToComputational1D:
    def test_raises_exception_for_invalid_physical_domain(self):
        xl = 0
        xr = -1
        with pytest.raises(Exception):
            _ = physical_to_computational_1d(0, xl, xr)

    def test_returns_float_when_input_is_float(self):
        xl = 0
        xr = 1
        x = 0.5
        r = physical_to_computational_1d(x, xl, xr)
        assert isinstance(r, (int, float))

    def test_returns_numpy_array_when_input_is_numpy_array(self):
        xl = 0.8
        xr = 0.9
        x = np.linspace(xl, xr, 7)
        r = physical_to_computational_1d(x, xl, xr)
        assert isinstance(r, np.ndarray)
        assert x.shape == r.shape

    @pytest.mark.parametrize("r,x,xl,xr", AFFINE_SAMPLES_1D)
    def test_affine_mapping_is_correct(self, r, x, xl, xr):
        assert np.isclose(physical_to_computational_1d(x, xl, xr), r)


class TestComputationalToPhysical2D:
    def test_raises_exception_for_invalid_physical_domain(self):
        v1 = np.array([1.0, 1.0])
        v2 = 2 * v1
        v3 = 3 * v1
        r = np.array([-0.5, -0.5])
        with pytest.raises(Exception):
            _ = computational_to_physical_2d(r, v1, v2, v3)

    def test_returns_1d_array_when_input_is_1d_array(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, 1.0])
        r = np.array([0.0, 0.0])
        x = computational_to_physical_2d(r, v1, v2, v3)
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 1
        assert x.shape[0] == 2

    def test_returns_2d_array_when_input_is_2d_array(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, 1.0])
        r = np.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5]])
        batch_size = r.shape[0]
        x = computational_to_physical_2d(r, v1, v2, v3)
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2
        assert x.shape[0] == batch_size
        assert x.shape[1] == 2

    @pytest.mark.parametrize("r,x,v1,v2,v3", AFFINE_SAMPLES_2D)
    def test_affine_mapping_is_correct(self, r, x, v1, v2, v3):
        assert np.allclose(computational_to_physical_2d(r, v1, v2, v3), x)


class TestPhysicalToComputational2D:
    def test_raises_exception_for_invalid_physical_domain(self):
        v1 = np.array([1.0, 1.0])
        v2 = 2 * v1
        v3 = 3 * v1
        x = np.array([1.5, 1.5])
        with pytest.raises(Exception):
            _ = physical_to_computational_2d(x, v1, v2, v3)

    def test_returns_1d_array_when_input_is_1d_array(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, 1.0])
        x = np.array([0.5, 0.5])
        r = physical_to_computational_2d(x, v1, v2, v3)
        assert isinstance(r, np.ndarray)
        assert len(r.shape) == 1
        assert r.shape[0] == 2

    def test_returns_2d_array_when_input_is_2d_array(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, 1.0])
        x = np.array([[0.5, 0.5], [0.25, 0.1], [0.5, 0.2]])
        batch_size = x.shape[0]
        r = physical_to_computational_2d(x, v1, v2, v3)
        assert isinstance(r, np.ndarray)
        assert len(r.shape) == 2
        assert r.shape[0] == batch_size
        assert r.shape[1] == 2

    @pytest.mark.parametrize("r,x,v1,v2,v3", AFFINE_SAMPLES_2D)
    def test_affine_mapping_is_correct(self, r, x, v1, v2, v3):
        assert np.allclose(physical_to_computational_2d(x, v1, v2, v3), r)
