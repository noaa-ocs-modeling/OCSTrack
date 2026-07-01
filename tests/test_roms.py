import os
import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from ocstrack.Model.model import ROMS

@pytest.fixture(scope="module")
def roms_test_data(tmpdir_factory):
    """Create a dummy ROMS run directory with an ocean.in file and NetCDF files."""
    rundir = tmpdir_factory.mktemp("roms_run")
    his_dir = rundir.mkdir("HIS")

    # Create a dummy ocean.in file
    ocean_in_content = """
    Vtransform == 2
    Vstretching == 4
    THETA_S == 7.0d0
    THETA_B == 4.0d0
    TCLINE == 50.0d0
    N == 40
    """
    rundir.join("ocean.in").write(ocean_in_content)

    # Create dummy NetCDF files
    eta_rho = 20
    xi_rho = 10
    s_rho = 40
    lon_rho = np.linspace(-75, -70, xi_rho)
    lat_rho = np.linspace(35, 40, eta_rho)
    lon_rho_2d, lat_rho_2d = np.meshgrid(lon_rho, lat_rho)
    h = np.random.rand(eta_rho, xi_rho) * 100

    start_date = datetime(2023, 1, 1)
    for i in range(3):
        date = start_date + timedelta(hours=i)
        time_val = [date]
        filepath = his_dir.join(f"roms_his_{i:04d}.nc")

        ds = xr.Dataset(
            {
                "lon_rho": (( "eta_rho","xi_rho"), lon_rho_2d),
                "lat_rho": (("eta_rho", "xi_rho"), lat_rho_2d),
                "h": (("eta_rho", "xi_rho"), h),
                "zeta": (("ocean_time", "eta_rho", "xi_rho"), np.random.rand(1, eta_rho, xi_rho)),
                "temp": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.random.rand(1, s_rho, eta_rho, xi_rho)),
            },
            coords={
                "ocean_time": ("ocean_time", time_val),
                "s_rho": ("s_rho", np.linspace(-1, 0, s_rho)),
                "eta_rho": ("eta_rho", np.arange(eta_rho)),
                "xi_rho": ("xi_rho", np.arange(xi_rho)),
            },
        )
        ds.to_netcdf(filepath)

    return str(rundir)

def test_roms_init(roms_test_data):
    """Test the initialization of the ROMS class."""
    model = ROMS(
        rundir=roms_test_data,
        model_dict={"var": "temp", "var_type": "3D_Profile", "zcor_var": "z_rho"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-01T02:00:00"),
    )
    assert model.rundir == roms_test_data
    assert len(model.files) == 3
    assert model.Vtransform == 2
    assert model.Vstretching == 4
    assert model.theta_s == 7.0
    assert model.theta_b == 4.0
    assert model.hc == 50.0
    assert model.N == 40

def test_roms_load_mesh(roms_test_data):
    """Test loading of the mesh data."""
    model = ROMS(
        rundir=roms_test_data,
        model_dict={"var": "temp", "var_type": "3D_Profile", "zcor_var": "z_rho"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-01T02:00:00"),
    )
    assert model.mesh_x.shape == (200,)
    assert model.mesh_y.shape == (200,)
    assert model.h.shape == (20, 10)

def test_roms_load_3d_file_pair(roms_test_data):
    """Test the load_3d_file_pair method."""
    model = ROMS(
        rundir=roms_test_data,
        model_dict={"var": "temp", "var_type": "3D_Profile", "zcor_var": "z_rho"},
        start_date=np.datetime64("2023-01-01"),
        end_date=np.datetime64("2023-01-01T02:00:00"),
    )
    ds = model.load_3d_file_pair(model.files[0])
    assert "temp" in ds
    assert "z_rho" in ds
    assert "time" in ds.dims
    assert "nSCHISM_hgrid_node" in ds.dims
    assert ds["z_rho"].shape == (1, 40, 200)
    assert ds["time"].shape == (1,)
