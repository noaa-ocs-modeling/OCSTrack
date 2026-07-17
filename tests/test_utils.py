"""Tests for ocstrack.utils.convert_longitude."""

import numpy as np
import pytest

from ocstrack.utils import convert_longitude


class TestConvertLongitude:
    def test_mode_1_neg180_180_to_0_360(self):
        """Mode 1: [-180, 180] -> [0, 360]."""
        lon = np.array([-180.0, -90.0, 0.0, 90.0, 179.0])
        out = convert_longitude(lon, mode=1)
        np.testing.assert_allclose(out, [180.0, 270.0, 0.0, 90.0, 179.0])

    def test_mode_2_0_360_to_neg180_180(self):
        """Mode 2: [0, 360] -> [-180, 180]."""
        lon = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
        out = convert_longitude(lon, mode=2)
        np.testing.assert_allclose(out, [0.0, 90.0, 180.0, -90.0, -1.0])

    def test_mode_3_shift_by_180(self):
        """Mode 3: [-180, 180] (Greenwich at 0) -> [0, 360] (Greenwich at 180)."""
        lon = np.array([-180.0, 0.0, 180.0])
        out = convert_longitude(lon, mode=3)
        np.testing.assert_allclose(out, [0.0, 180.0, 360.0])

    def test_mode_4(self):
        """Mode 4: [0, 360] (Greenwich at 0) -> [0, 360] (Greenwich at 180)."""
        lon = np.array([0.0, 180.0, 360.0])
        out = convert_longitude(lon, mode=4)
        np.testing.assert_allclose(out, [180.0, 0.0, 180.0])

    def test_mode_5(self):
        """Mode 5: [0, 360] (Greenwich at 180) -> [0, 360] (Greenwich at 0)."""
        lon = np.array([0.0, 180.0, 360.0])
        out = convert_longitude(lon, mode=5)
        np.testing.assert_allclose(out, [180.0, 0.0, 180.0])

    def test_accepts_scalar(self):
        """A scalar input is converted and returned as an ndarray."""
        out = convert_longitude(-90.0, mode=1)
        np.testing.assert_allclose(out, 270.0)

    def test_returns_ndarray(self):
        """Output is always a numpy array."""
        out = convert_longitude([10.0, 20.0], mode=1)
        assert isinstance(out, np.ndarray)

    def test_invalid_mode_raises(self):
        """An unsupported mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            convert_longitude(np.array([0.0]), mode=99)
