import os
import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from ocstrack.Model.model import WW3

@pytest.fixture(scope="module")
def ww3_test_data(tmpdir_factory):
    """Create a dummy WW3 run directory with NetCDF files."""
    rundir = tmpdir_factory.mktemp("ww3_run")
    nx = 10
    ny = 1
    lon = np.linspace(0, 9, nx)
    lat = np.linspace(50, 59, nx)

    start_date = datetime(2023, 1, 1)
    for i in range(5):
        date = start_date + timedelta(days=i)
        time = [date]
        filepath = rundir.join(f"{date.strftime('%Y%m%d')}.000000.out_grd.ww3.nc")

        ds = xr.Dataset(
            {
                "time": (("time",), time),
                "lon": (("ny", "nx"), [lon]),
                "lat": (("ny", "nx"), [lat]),
                "HS": (("time", "ny", "nx"), [[[np.random.rand(nx)]]]),
            },
            coords={
                "time": time,
                "ny": [0],
                "nx": np.arange(nx),
            },
        )
        ds.to_netcdf(filepath)

    return str(rundir)

def test_ww3_init(ww3_test_data):
    """Test the initialization of the WW3 class."""
    model = WW3(
        rundir=ww3_test_data,
        model_dict={"var": "HS"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-05"),
    )
    assert model.rundir == ww3_test_data
    assert len(model.files) == 5

def test_ww3_select_files(ww3_test_data):
    """Test the file selection logic."""
    model = WW3(
        rundir=ww3_test_data,
        model_dict={"var": "HS"},
        start_date=np.datetime64("2023-01-02"),
        end_date=np.datetime64("2023-01-03"),
    )
    assert len(model.files) == 2
    assert "20230102" in model.files[0]
    assert "20230103" in model.files[1]

def test_ww3_load_mesh(ww3_test_data):
    """Test loading of the mesh data."""
    model = WW3(
        rundir=ww3_test_data,
        model_dict={"var": "HS"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-01"),
    )
    assert model.mesh_x.shape == (10,)
    assert model.mesh_y.shape == (10,)
    assert np.all(np.isnan(model.mesh_depth))

def test_ww3_load_variable(ww3_test_data):
    """Test loading a variable."""
    model = WW3(
        rundir=ww3_test_data,
        model_dict={"var": "HS"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-01"),
    )
    var = model.load_variable(model.files[0])
    assert var.name == "HS"
    assert var.shape == (1, 10)  # ny dimension should be squeezed

def test_ww3_time_property(ww3_test_data):
    """Test the time property."""
    model = WW3(
        rundir=ww3_test_data,
        model_dict={"var": "HS"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-03"),
    )
    assert len(model.time) == 3
    assert model.time[0] == np.datetime64("2023-01-01")
    assert model.time[1] == np.datetime64("2023-01-02")
    assert model.time[2] == np.datetime64("2023-01-03")
