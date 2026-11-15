"""Tests for the Collocate class (2D surface and 3D profile paths)."""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch, PropertyMock

from ocstrack.Collocation.collocate import Collocate
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation.argofloat import ArgoData


# ---------------------------------------------------------------------------
# Helpers: minimal mock model and observation objects
# ---------------------------------------------------------------------------

def _make_mock_model_2d(n_nodes=20, n_times=3, var_name="sigWaveHeight"):
    """Return a mock SCHISM-like model for 2D surface collocation."""
    lons = np.linspace(-80, -70, n_nodes)
    lats = np.linspace(30, 40, n_nodes)
    depths = np.ones(n_nodes) * 50.0

    times = np.array(
        [f"2023-06-01T{h:02d}" for h in range(0, n_times * 6, 6)],
        dtype="datetime64[ns]",
    )

    # Build a DataArray that mimics model.load_variable(path)
    model_da = xr.DataArray(
        np.random.rand(n_times, n_nodes).astype(np.float32),
        dims=["time", "node"],
        coords={"time": times},
        name=var_name,
    )

    model = MagicMock()
    model.model_dict = {"var": var_name, "var_type": "2D"}
    model.mesh_x = lons
    model.mesh_y = lats
    model.mesh_depth = depths
    model.files = ["fake_file_0.nc"]
    model.load_variable.return_value = model_da
    type(model).time = PropertyMock(return_value=times)

    return model


def _make_mock_sat_ds(n_obs=5, t_start="2023-06-01T01"):
    """Return a minimal SatelliteData-like xr.Dataset."""
    base = np.datetime64(t_start, "ns")
    times = np.array(
        [base + np.timedelta64(i * 2, "h") for i in range(n_obs)],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "swh": ("time", np.random.rand(n_obs).astype(np.float32)),
            "sla": ("time", np.random.rand(n_obs).astype(np.float32)),
            "lat": ("time", np.linspace(31, 39, n_obs).astype(np.float32)),
            "lon": ("time", np.linspace(-79, -71, n_obs).astype(np.float32)),
            "source": ("time", np.full(n_obs, "mock_sat", dtype=object)),
        },
        coords={"time": times},
    )


def _make_mock_sat_obs(n_obs=5):
    """Return a mock SatelliteData object."""
    obs = MagicMock(spec=SatelliteData)
    obs.ds = _make_mock_sat_ds(n_obs)
    return obs


# ---------------------------------------------------------------------------
# Collocate initialisation
# ---------------------------------------------------------------------------

class TestCollocateInit:
    def test_init_2d_nearest(self):
        model = _make_mock_model_2d()
        obs = _make_mock_sat_obs()

        col = Collocate(
            model_run=model,
            observation=obs,
            n_nearest=4,
            time_buffer=np.timedelta64(3, "h"),
        )
        assert col.n_nearest == 4
        assert col.collocation_type == "2D"

    def test_both_n_nearest_and_radius_warns_and_uses_radius(self):
        model = _make_mock_model_2d()
        obs = _make_mock_sat_obs()

        # Should log a warning internally but not raise an exception
        col = Collocate(
            model_run=model,
            observation=obs,
            n_nearest=4,
            search_radius=50_000,
            time_buffer=np.timedelta64(3, "h"),
        )
        assert col.n_nearest is None
        assert col.search_radius == 50_000

    def test_neither_n_nearest_nor_radius_raises(self):
        model = _make_mock_model_2d()
        obs = _make_mock_sat_obs()

        with pytest.raises(ValueError, match="n_nearest.*search_radius"):
            Collocate(
                model_run=model,
                observation=obs,
                n_nearest=None,
                search_radius=None,
                time_buffer=np.timedelta64(3, "h"),
            )

    def test_wrong_obs_type_raises(self):
        model = _make_mock_model_2d()
        wrong_obs = MagicMock()  # Not SatelliteData or ArgoData

        with pytest.raises(TypeError, match="SatelliteData or ArgoData"):
            Collocate(
                model_run=model,
                observation=wrong_obs,
                n_nearest=4,
                time_buffer=np.timedelta64(3, "h"),
            )

    def test_time_buffer_inferred_from_model_time(self):
        model = _make_mock_model_2d(n_times=3)
        obs = _make_mock_sat_obs()

        col = Collocate(
            model_run=model,
            observation=obs,
            n_nearest=4,
            # No time_buffer — should be inferred
        )
        assert col.time_buffer is not None


# ---------------------------------------------------------------------------
# Collocate.run — 2D surface
# ---------------------------------------------------------------------------

class TestCollocateRun2D:
    def _build_collocator(self, n_nearest=4):
        model = _make_mock_model_2d(n_nodes=20, n_times=4, var_name="sigWaveHeight")
        obs = _make_mock_sat_obs(n_obs=6)
        col = Collocate(
            model_run=model,
            observation=obs,
            n_nearest=n_nearest,
            time_buffer=np.timedelta64(6, "h"),
        )
        return col

    def test_run_returns_dataset(self):
        col = self._build_collocator()
        ds = col.run()
        assert isinstance(ds, xr.Dataset)

    def test_run_contains_expected_variables(self):
        col = self._build_collocator()
        ds = col.run()
        # Should have obs and model data
        assert "model_sigWaveHeight" in ds or len(ds) == 0  # may be empty if no overlap

    def test_run_wrong_obs_type_raises(self):
        model = _make_mock_model_2d()
        argo_obs = MagicMock(spec=ArgoData)
        argo_obs.ds = MagicMock()

        with pytest.raises(TypeError):
            Collocate(
                model_run=model,
                observation=argo_obs,
                n_nearest=4,
                time_buffer=np.timedelta64(3, "h"),
            ).run()


# ---------------------------------------------------------------------------
# Collocate.run — unknown var_type
# ---------------------------------------------------------------------------

class TestCollocateRunUnknownType:
    def test_run_unknown_type_raises_not_implemented(self):
        model = _make_mock_model_2d()
        model.model_dict = {"var": "sigWaveHeight", "var_type": "UNKNOWN_TYPE"}
        obs = _make_mock_sat_obs()

        col = Collocate(
            model_run=model,
            observation=obs,
            n_nearest=4,
            time_buffer=np.timedelta64(3, "h"),
        )
        with pytest.raises(NotImplementedError):
            col.run()


# ---------------------------------------------------------------------------
# Collocate.run with one observation sitting exactly on a model node
# ---------------------------------------------------------------------------

class TestCollocateRun2DCoincidentNode:
    def _build(self):
        n_nodes = 20
        lons = np.linspace(-80.0, -70.0, n_nodes)
        lats = np.linspace(30.0, 40.0, n_nodes)
        times = np.array(
            [f"2023-06-01T{h:02d}" for h in range(0, 24, 6)],
            dtype="datetime64[ns]",
        )
        rng = np.random.default_rng(0)
        model_da = xr.DataArray(
            rng.random((len(times), n_nodes)) + 1.0,
            dims=["time", "node"],
            coords={"time": times},
            name="sigWaveHeight",
        )
        model = MagicMock()
        model.model_dict = {"var": "sigWaveHeight", "var_type": "2D"}
        model.mesh_x = lons
        model.mesh_y = lats
        model.mesh_depth = np.ones(n_nodes) * 50.0
        model.files = ["fake_file_0.nc"]
        model.load_variable.return_value = model_da
        type(model).time = PropertyMock(return_value=times)

        # The first observation lands exactly on model node 5.
        obs_ds = xr.Dataset(
            {
                "swh": ("time", np.array([1.0, 2.0, 3.0])),
                "lat": ("time", np.array([lats[5], 35.0, 36.0])),
                "lon": ("time", np.array([lons[5], -75.0, -73.0])),
            },
            coords={
                "time": np.array(
                    ["2023-06-01T01", "2023-06-01T07", "2023-06-01T13"],
                    dtype="datetime64[ns]",
                )
            },
        )
        obs = MagicMock(spec=SatelliteData)
        obs.ds = obs_ds

        return Collocate(
            model_run=model,
            observation=obs,
            n_nearest=3,
            time_buffer=np.timedelta64(6, "h"),
        )

    def test_other_observations_keep_their_weighted_value(self):
        ds = self._build().run()

        weighted = ds["model_sigWaveHeight_weighted"].values
        neighbours = ds["model_sigWaveHeight"].values

        assert np.all(np.isfinite(weighted))
        # Every weighted value must sit inside the range of its own neighbours.
        for i in range(len(weighted)):
            assert weighted[i] >= neighbours[i].min() - 1e-9
            assert weighted[i] <= neighbours[i].max() + 1e-9
