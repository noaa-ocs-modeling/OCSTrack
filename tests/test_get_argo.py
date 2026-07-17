
from ocstrack.Observation.get_argo import generate_monthly_dates

def test_generate_monthly_dates():
    """
    Test the generate_monthly_dates function for a multi-month range.
    """
    start_date = "2023-01-15"
    end_date = "2023-03-20"
    
    expected_months = [
        ("2023", "01"),
        ("2023", "02"),
        ("2023", "03"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_single_month():
    """
    Test the function for a date range within a single month.
    """
    start_date = "2023-01-10"
    end_date = "2023-01-20"
    
    expected_months = [("2023", "01")]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_year_boundary():
    """
    Test the function across a year boundary.
    """
    start_date = "2022-12-01"
    end_date = "2023-02-01"
    
    expected_months = [
        ("2022", "12"),
        ("2023", "01"),
        ("2023", "02"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months


from ocstrack.Observation.get_argo import _download_file
from unittest.mock import patch, mock_open, MagicMock
import requests

@patch('requests.get')
def test_download_file_success(mock_get):
    """
    Test the _download_file helper for a successful download.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake data']
    mock_get.return_value.__enter__.return_value = mock_response

    m_open = mock_open()
    with patch('builtins.open', m_open):
        success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

        assert success is True
        mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True, timeout=60)
        m_open.assert_called_once_with('/fake/path/file.nc', 'wb')
        m_open().write.assert_called_once_with(b'fake data')

@patch('requests.get')
@patch('os.path.exists', return_value=True)
@patch('os.remove')
def test_download_file_failure(mock_remove, mock_exists, mock_get):
    """
    Test the _download_file helper for a failed download.
    It should return False and clean up partially downloaded files.
    """
    mock_get.side_effect = requests.exceptions.RequestException('Test Error')

    # The function should catch the exception and return False
    success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

    assert success is False
    mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True, timeout=60)
    # Check that it attempts to clean up the file
    mock_remove.assert_called_once_with('/fake/path/file.nc')


# ---------------------------------------------------------------------------
# crop_by_shape_argo
# ---------------------------------------------------------------------------

import os
import pytest
import numpy as np
import xarray as xr
from ocstrack.Observation import get_argo as get_argo_mod
from ocstrack.Observation.get_argo import (
    crop_by_shape_argo,
    crop_argo_data_by_shape,
    get_argo,
)


def _write_argo_nc(path, n=10, lat0=0.0, lat1=9.0, lon0=0.0, lon1=9.0,
                   t0="2023-01-05"):
    """Write a minimal Argo-style per-profile NetCDF file to *path*."""
    juld = np.array(
        [np.datetime64(t0) + np.timedelta64(i, "D") for i in range(n)],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {
            "JULD":      ("N_PROF", juld),
            "LATITUDE":  ("N_PROF", np.linspace(lat0, lat1, n)),
            "LONGITUDE": ("N_PROF", np.linspace(lon0, lon1, n)),
            "PRES":      (("N_PROF", "N_LEVELS"), np.ones((n, 3))),
            "TEMP":      (("N_PROF", "N_LEVELS"), np.ones((n, 3))),
        },
    )
    ds.to_netcdf(path)
    return path


def _make_argo_ds(n=10):
    """Build a minimal Argo-like dataset with N_PROF profiles."""
    juld = np.array(
        [np.datetime64("2023-01-01") + np.timedelta64(i, "h") for i in range(n)],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "JULD":      ("N_PROF", juld),
            "LATITUDE":  ("N_PROF", np.linspace(0.0, 9.0, n)),
            "LONGITUDE": ("N_PROF", np.linspace(0.0, 9.0, n)),
            "PRES":      (("N_PROF", "N_LEVELS"), np.ones((n, 5))),
        },
        coords={"N_PROF": np.arange(n)},
    )


class TestCropByShapeArgo:
    def test_polygon_keeps_correct_profiles(self):
        """Profiles inside the polygon are kept; outside are dropped."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds(n=10)
        # Box covering lon/lat in [0, 4.5] -> first 5 profiles should survive
        bbox = box(-0.5, -0.5, 4.5, 4.5)
        cropped = crop_by_shape_argo(ds, bbox)
        assert cropped.sizes["N_PROF"] == 5
        assert float(cropped["LONGITUDE"].max()) <= 4.5

    def test_polygon_drops_all_outside(self):
        """A polygon with no profiles inside returns an empty dataset."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds(n=10)
        # Box far outside the data range
        empty_box = box(50.0, 50.0, 60.0, 60.0)
        cropped = crop_by_shape_argo(ds, empty_box)
        assert cropped.sizes["N_PROF"] == 0

    def test_geojson_dict(self):
        """Accepts a bare GeoJSON geometry dict."""
        shapely = pytest.importorskip("shapely")

        ds = _make_argo_ds(n=10)
        geojson = {
            "type": "Polygon",
            "coordinates": [[[-0.5, -0.5], [4.5, -0.5], [4.5, 4.5], [-0.5, 4.5], [-0.5, -0.5]]],
        }
        cropped = crop_by_shape_argo(ds, geojson)
        assert cropped.sizes["N_PROF"] == 5

    def test_missing_coords_raises(self):
        """Raises ValueError when LONGITUDE is missing from the dataset."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds().drop_vars("LONGITUDE")
        with pytest.raises(ValueError, match="does not contain"):
            crop_by_shape_argo(ds, box(0, 0, 5, 5))

    def test_invalid_shape_raises(self):
        """Raises ValueError for an unrecognised shape type."""
        shapely = pytest.importorskip("shapely")

        ds = _make_argo_ds()
        with pytest.raises(ValueError):
            crop_by_shape_argo(ds, {"bad": "dict"})


class TestGetArgoShapeBoxMutuallyExclusive:
    def test_both_box_and_shape_raises(self):
        """Providing both bounding-box and shape arguments raises ValueError."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        with pytest.raises(ValueError, match="not both"):
            get_argo(
                start_date="2023-01-01",
                end_date="2023-01-31",
                region="atlantic_ocean",
                output_dir="/tmp/test_argo",
                lat_min=0, lat_max=10, lon_min=0, lon_max=10,
                shape=box(0, 0, 5, 5),
            )


# ---------------------------------------------------------------------------
# crop_argo_data_by_shape / crop_argo_data / clean_argo_data
# ---------------------------------------------------------------------------

class TestCropArgoDataByShape:
    def test_writes_cropped_file(self, tmp_path):
        pytest.importorskip("shapely")
        from shapely.geometry import box

        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, lat0=0.0, lat1=9.0, lon0=0.0, lon1=9.0)
        cropped_dir = str(tmp_path / "cropped")

        crop_argo_data_by_shape(
            [raw], cropped_dir, box(-0.5, -0.5, 4.5, 4.5),
            start_date="2023-01-01", end_date="2023-12-31",
        )
        out_files = os.listdir(cropped_dir)
        assert len(out_files) == 1
        with xr.open_dataset(str(tmp_path / "cropped" / out_files[0])) as ds:
            assert float(ds["LATITUDE"].max()) <= 4.5

    def test_empty_crop_is_skipped(self, tmp_path):
        pytest.importorskip("shapely")
        from shapely.geometry import box

        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, lat0=0.0, lat1=9.0)
        cropped_dir = str(tmp_path / "cropped")

        crop_argo_data_by_shape(
            [raw], cropped_dir, box(50, 50, 60, 60),
            start_date="2023-01-01", end_date="2023-12-31",
        )
        assert os.path.isdir(cropped_dir)
        assert os.listdir(cropped_dir) == []

    def test_bad_file_is_skipped(self, tmp_path):
        pytest.importorskip("shapely")
        from shapely.geometry import box

        cropped_dir = str(tmp_path / "cropped")
        # Should not raise even though the file does not exist
        crop_argo_data_by_shape(
            [str(tmp_path / "missing.nc")], cropped_dir, box(0, 0, 5, 5),
            start_date="2023-01-01", end_date="2023-12-31",
        )
        assert os.listdir(cropped_dir) == []


class TestCropArgoData:
    def test_writes_cropped_file(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, lat0=0.0, lat1=9.0, lon0=0.0, lon1=9.0)
        cropped_dir = str(tmp_path / "cropped")

        get_argo_mod.crop_argo_data(
            [raw], cropped_dir, lat_min=0, lat_max=4.5,
            lon_min=0, lon_max=4.5, start_date="2023-01-01", end_date="2023-12-31",
        )
        out_files = os.listdir(cropped_dir)
        assert len(out_files) == 1


class TestCleanArgoData:
    def test_writes_time_filtered_file(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, t0="2023-01-05")
        clean_dir = str(tmp_path / "clean")

        get_argo_mod.clean_argo_data(
            [raw], clean_dir, start_date="2023-01-01", end_date="2023-12-31",
        )
        assert len(os.listdir(clean_dir)) == 1

    def test_empty_time_filter_is_skipped(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, t0="2023-01-05")
        clean_dir = str(tmp_path / "clean")

        # Date range with no overlapping profiles
        get_argo_mod.clean_argo_data(
            [raw], clean_dir, start_date="2020-01-01", end_date="2020-01-02",
        )
        assert os.listdir(clean_dir) == []


# ---------------------------------------------------------------------------
# get_argo orchestration (download mocked)
# ---------------------------------------------------------------------------

class TestGetArgoPipeline:
    def test_no_raw_files_returns_none(self, tmp_path):
        with patch.object(get_argo_mod, "download_argo_data", return_value=[]):
            result = get_argo(
                "2023-01-01", "2023-01-31", "atlantic_ocean", str(tmp_path),
            )
        assert result is None

    def test_box_crop_pipeline(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, lat0=0.0, lat1=9.0, lon0=0.0, lon1=9.0)

        with patch.object(get_argo_mod, "download_argo_data", return_value=[raw]):
            result = get_argo(
                "2023-01-01", "2023-01-31", "atlantic_ocean", str(tmp_path),
                lat_min=0, lat_max=4.5, lon_min=0, lon_max=4.5,
            )
        assert result is not None
        assert os.path.isdir(result)
        assert len(os.listdir(result)) == 1

    def test_shape_crop_pipeline(self, tmp_path):
        pytest.importorskip("shapely")
        from shapely.geometry import box

        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, lat0=0.0, lat1=9.0, lon0=0.0, lon1=9.0)

        with patch.object(get_argo_mod, "download_argo_data", return_value=[raw]):
            result = get_argo(
                "2023-01-01", "2023-01-31", "atlantic_ocean", str(tmp_path),
                shape=box(-0.5, -0.5, 4.5, 4.5),
            )
        assert result is not None
        assert len(os.listdir(result)) == 1

    def test_no_crop_pipeline(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, t0="2023-01-05")

        with patch.object(get_argo_mod, "download_argo_data", return_value=[raw]):
            result = get_argo(
                "2023-01-01", "2023-01-31", "atlantic_ocean", str(tmp_path),
            )
        assert result is not None
        assert len(os.listdir(result)) == 1

    def test_clean_raw_removes_raw_dir(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _write_argo_nc(raw, t0="2023-01-05")

        with patch.object(get_argo_mod, "download_argo_data", return_value=[raw]):
            get_argo(
                "2023-01-01", "2023-01-31", "atlantic_ocean", str(tmp_path),
                clean_raw=True,
            )
        assert not os.path.exists(os.path.join(str(tmp_path), "atlantic_ocean", "raw"))


