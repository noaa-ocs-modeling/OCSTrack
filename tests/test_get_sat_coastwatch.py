
import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import numpy as np
import xarray as xr
from ocstrack.Observation import get_sat_coastwatch
from ocstrack.Observation.get_sat_coastwatch import (
    download_sat_data,
    crop_by_shape,
    crop_by_box,
    crop_sat_data,
    concat_sat_data,
    generate_daily_dates,
    get_per_sat_coastwatch,
    get_multi_sat_coastwatch,
)
from ocstrack.Observation.urls import (
    URL_TEMPLATES,
    CCI_ALTIMETERS,
    resolve_sat_key,
    _canonical_sat_key,
)
import requests

@pytest.fixture
def mock_requests_get():
    """Fixture to mock requests.get."""
    with patch('requests.get') as mock_get:
        yield mock_get

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_success(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test successful download of a single file.
    """
    # Configure the mock for a successful response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake', b'file', b'content']
    mock_requests_get.return_value.__enter__.return_value = mock_response

    # Use mock_open to simulate file writing
    m_open = mock_open()
    with patch('builtins.open', m_open):
        result = download_sat_data(
            dates_str=['20230101'],
            url_template='http://fake.url/',
            raw_dir='/fake/dir',
            sat='test_sat'
        )

        # Assertions
        mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
        mock_requests_get.assert_called_once_with('http://fake.url/20230101.nc', stream=True, timeout=60)
        m_open.assert_called_once_with('/fake/dir/20230101.nc', 'wb')
        handle = m_open()
        handle.write.assert_any_call(b'fake')
        handle.write.assert_any_call(b'content')
        assert result == ['/fake/dir/20230101.nc']

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_failure_and_retry(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test that the function retries on a request failure.
    """
    # Configure the mock to raise an exception
    mock_requests_get.side_effect = requests.exceptions.RequestException("Test Error")

    m_open = mock_open()
    with patch('builtins.open', m_open):
        with patch('time.sleep', return_value=None) as mock_sleep: # Mock time.sleep
            result = download_sat_data(
                dates_str=['20230101'],
                url_template='http://fake.url/',
                raw_dir='/fake/dir',
                sat='test_sat',
                retries=3,
                delay=1
            )

            # Assertions
            mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
            assert mock_requests_get.call_count == 3
            assert mock_sleep.call_count == 3 # Sleeps after each of the 3 failures
            m_open.assert_not_called() # File should not be opened on failure
            assert result == ['/fake/dir/20230101.nc'] # Still returns the expected path


# ---------------------------------------------------------------------------
# Satellite key normalization
# ---------------------------------------------------------------------------

class TestResolveSatKey:
    def test_canonical_strips_punctuation_and_case(self):
        assert _canonical_sat_key("Jason-3") == "jason3"
        assert _canonical_sat_key("JASON_3") == "jason3"
        assert _canonical_sat_key("jason 3") == "jason3"
        assert _canonical_sat_key("sentinel-3_a") == "sentinel3a"

    def test_exact_match_coastwatch(self):
        assert resolve_sat_key("jason3", URL_TEMPLATES) == "jason3"

    def test_variant_resolves_to_coastwatch_key(self):
        # CCI-style 'jason-3' should resolve to CoastWatch 'jason3'
        assert resolve_sat_key("jason-3", URL_TEMPLATES) == "jason3"
        assert resolve_sat_key("Sentinel-3A", URL_TEMPLATES) == "sentinel3a"

    def test_exact_match_cci(self):
        assert resolve_sat_key("jason-3", CCI_ALTIMETERS) == "jason-3"

    def test_variant_resolves_to_cci_key(self):
        # CoastWatch-style 'jason3' should resolve to CCI 'jason-3'
        assert resolve_sat_key("jason3", CCI_ALTIMETERS) == "jason-3"
        assert resolve_sat_key("sentinel3a", CCI_ALTIMETERS) == "sentinel-3a"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite key"):
            resolve_sat_key("not-a-satellite", URL_TEMPLATES)


# ---------------------------------------------------------------------------
# crop_by_shape (regression test: numpy must be imported)
# ---------------------------------------------------------------------------

class TestCropByShape:
    def _make_ds(self):
        n = 10
        times = np.arange(
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")
        return xr.Dataset(
            {
                "swh": ("time", np.random.rand(n)),
                "lat": ("time", np.linspace(0, 9, n)),
                "lon": ("time", np.linspace(0, 9, n)),
            },
            coords={"time": times},
        )

    def test_crop_by_shape_polygon(self):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = self._make_ds()
        # Box covering lon/lat in [0, 4.5] -> should keep first 5 points
        bbox = box(-0.5, -0.5, 4.5, 4.5)
        cropped = crop_by_shape(ds, bbox)
        assert cropped.sizes["time"] == 5
        assert float(cropped.lon.max()) <= 4.5

    def test_crop_by_shape_missing_coords_raises(self):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = self._make_ds().drop_vars("lon")
        with pytest.raises(ValueError, match="does not contain"):
            crop_by_shape(ds, box(0, 0, 1, 1))


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------

class TestGenerateDailyDates:
    def test_inclusive_range(self):
        result = generate_daily_dates("2023-01-16", "2023-01-18")
        assert result == ["20230116", "20230117", "20230118"]

    def test_single_day(self):
        assert generate_daily_dates("2023-01-16", "2023-01-16") == ["20230116"]


# ---------------------------------------------------------------------------
# Box cropping
# ---------------------------------------------------------------------------

def _make_cw_pass(path, n=10, lat0=30.0, lat1=40.0, lon0=-80.0, lon1=-70.0,
                  t0="2023-01-16"):
    """Write a minimal CoastWatch-style NetCDF file."""
    times = np.arange(
        np.datetime64(t0),
        np.datetime64(t0) + np.timedelta64(n, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "swh": ("time", np.random.rand(n)),
            "sla": ("time", np.random.rand(n)),
            "lat": ("time", np.linspace(lat0, lat1, n)),
            "lon": ("time", np.linspace(lon0, lon1, n)),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)
    return path


class TestCropByBox:
    def _make_ds(self):
        n = 10
        times = np.arange(
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")
        return xr.Dataset(
            {
                "swh": ("time", np.random.rand(n)),
                "lat": ("time", np.linspace(0, 9, n)),
                "lon": ("time", np.linspace(0, 9, n)),
            },
            coords={"time": times},
        )

    def test_box_keeps_in_range(self):
        ds = self._make_ds()
        cropped = crop_by_box(ds, lat_min=2, lat_max=6, lon_min=2, lon_max=6)
        assert float(cropped.lat.min()) >= 2
        assert float(cropped.lat.max()) <= 6

    def test_box_missing_coords_raises(self):
        ds = self._make_ds().drop_vars("lon")
        with pytest.raises(ValueError, match="does not contain"):
            crop_by_box(ds, 0, 1, 0, 1)

    def test_box_antimeridian_crossing(self):
        # lon_min > lon_max triggers the OR branch (dateline crossing)
        n = 10
        times = np.arange(
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")
        ds = xr.Dataset(
            {
                "swh": ("time", np.random.rand(n)),
                "lat": ("time", np.linspace(50, 60, n)),
                "lon": ("time", np.linspace(160, -160, n)),
            },
            coords={"time": times},
        )
        cropped = crop_by_box(ds, lat_min=50, lat_max=60,
                              lon_min=165, lon_max=-165)
        # Only points with lon >= 165 or lon <= -165 survive
        lons = cropped.lon.values
        assert np.all((lons >= 165) | (lons <= -165))


# ---------------------------------------------------------------------------
# Crop + concat file helpers
# ---------------------------------------------------------------------------

class TestCropSatData:
    def test_crop_writes_files(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _make_cw_pass(raw)
        cropped_dir = str(tmp_path / "cropped")
        result = crop_sat_data([raw], cropped_dir, 32, 38, -78, -72)
        assert len(result) == 1
        assert os.path.isdir(cropped_dir)
        # Filename should be clean (no stray whitespace/newlines)
        files = os.listdir(cropped_dir)
        assert len(files) == 1
        assert files[0] == "cropped_raw.nc"

    def test_crop_empty_skipped(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _make_cw_pass(raw, lat0=30, lat1=40)
        cropped_dir = str(tmp_path / "cropped")
        result = crop_sat_data([raw], cropped_dir, -10, -5, 0, 10)
        assert result == []


class TestConcatSatData:
    def test_concat_sets_source(self, tmp_path):
        ds_list = []
        for i in range(2):
            p = str(tmp_path / f"p{i}.nc")
            _make_cw_pass(p, t0=f"2023-01-1{6 + i}")
            with xr.open_dataset(p) as ds:
                ds_list.append(ds.load())
        out = str(tmp_path / "merged.nc")
        merged = concat_sat_data(ds_list, out, sat="jason3")
        assert merged is not None
        assert os.path.exists(out)
        assert str(merged.source.values) == "jason3"

    def test_concat_empty_returns_none(self, tmp_path):
        out = str(tmp_path / "merged.nc")
        assert concat_sat_data([], out, sat="jason3") is None


# ---------------------------------------------------------------------------
# High-level orchestration (download mocked)
# ---------------------------------------------------------------------------

class TestGetPerSatCoastwatch:
    def test_unknown_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown satellite key"):
            get_per_sat_coastwatch(
                "2023-01-16", "2023-01-16", "not-a-sat", str(tmp_path),
            )

    def test_pre_2020_warns(self, tmp_path, caplog):
        raw = str(tmp_path / "raw.nc")
        _make_cw_pass(raw)
        with patch.object(get_sat_coastwatch, "download_sat_data",
                          return_value=[raw]):
            with caplog.at_level("WARNING"):
                get_per_sat_coastwatch(
                    "2019-07-30", "2019-07-30", "jason3", str(tmp_path),
                    lat_min=32, lat_max=38, lon_min=-78, lon_max=-72,
                )
        assert any("prior to 2020" in r.message for r in caplog.records)

    def test_full_pipeline_with_crop(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _make_cw_pass(raw)
        with patch.object(get_sat_coastwatch, "download_sat_data",
                          return_value=[raw]):
            result = get_per_sat_coastwatch(
                "2023-01-16", "2023-01-16", "jason3", str(tmp_path),
                lat_min=32, lat_max=38, lon_min=-78, lon_max=-72,
            )
        assert result is not None
        expected = tmp_path / "jason3" / "concat_cropped_jason3_2023-01-16_2023-01-16.nc"
        assert expected.exists()

    def test_key_normalization(self, tmp_path):
        """A CCI-style key ('jason-3') should resolve to CoastWatch 'jason3'."""
        raw = str(tmp_path / "raw.nc")
        _make_cw_pass(raw)
        with patch.object(get_sat_coastwatch, "download_sat_data",
                          return_value=[raw]):
            result = get_per_sat_coastwatch(
                "2023-01-16", "2023-01-16", "jason-3", str(tmp_path),
                lat_min=32, lat_max=38, lon_min=-78, lon_max=-72,
            )
        assert result is not None
        assert (tmp_path / "jason3").is_dir()


class TestGetMultiSatCoastwatch:
    def test_no_datasets_returns_none(self, tmp_path):
        with patch.object(get_sat_coastwatch, "get_per_sat_coastwatch",
                          return_value=None):
            result = get_multi_sat_coastwatch(
                "2023-01-16", "2023-01-16", ["jason3", "saral"], str(tmp_path),
            )
        assert result is None

    def test_multi_merge(self, tmp_path):
        def fake_download(dates_str, url_template, raw_dir, sat, **kwargs):
            os.makedirs(raw_dir, exist_ok=True)
            p = os.path.join(raw_dir, "raw.nc")
            _make_cw_pass(p)
            return [p]

        with patch.object(get_sat_coastwatch, "download_sat_data",
                          side_effect=fake_download):
            result = get_multi_sat_coastwatch(
                "2023-01-16", "2023-01-16", ["jason3", "saral"], str(tmp_path),
                lat_min=32, lat_max=38, lon_min=-78, lon_max=-72,
                clean_raw=False, clean_cropped=False,
            )
        assert result is not None
        expected = tmp_path / "multisat_cropped_2023-01-16_2023-01-16.nc"
        assert expected.exists()
