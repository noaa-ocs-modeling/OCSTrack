import logging
import os
import re
from typing import List, Tuple, Union

import numpy as np
import xarray as xr


_logger = logging.getLogger(__name__)


def natural_sort_key(filename: str) -> List[Union[int, str]]:
    """
    Generate a key for natural sorting of filenames (e.g., file10 comes after file2).

    Parameters
    ----------
    filename : str
        Filename to generate sorting key for

    Returns
    -------
    List[Union[int, str]]
        List of numeric and string parts to be used for sorting
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', filename)]

def _parse_gr3_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a SCHISM hgrid.gr3 mesh file to extract node coordinates and depth.

    Parameters
    ----------
    filepath : str
        Path to the hgrid.gr3 mesh file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (lon, lat, depth) arrays for each mesh node

    Notes
    -----
    Assumes the hgrid.gr3 file contains node-based data with the expected format.
    This was added so we don't need OCSMesh as a requirement anymore.
    """
    with open(filepath, 'r') as f:
        _ = f.readline()  # mesh name
        ne_np_line = f.readline()
        n_elements, n_nodes = map(int, ne_np_line.strip().split())

        lons = np.empty(n_nodes)
        lats = np.empty(n_nodes)
        depths = np.empty(n_nodes)

        for i in range(n_nodes):
            parts = f.readline().strip().split()
            lons[i] = float(parts[1])
            lats[i] = float(parts[2])
            depths[i] = float(parts[3])

    return lons, lats, depths

class SCHISM:
    """
    SCHISM model interface

    Handles selection, filtering, and loading of model outputs from a SCHISM run directory.
    Also parses the model mesh (hgrid.gr3) for spatial queries.
    This assumes a run directory structure where:
    .
    ├── RunDir
        ├── hgrid.gr3
        ├── ...
        ├── outputs
            ├── out2d_*.nc
            └── *.nc

    Methods
    -------
    load_variable(path)
        Load model variable from a NetCDF file and extract surface layer if 3D
    """
    def __init__(self, rundir: str,
                 model_dict: dict,
                 start_date: np.datetime64,
                 end_date: np.datetime64,
                 output_subdir: str = "outputs"):
        """
        Initialize a SCHISM model run

        Parameters
        ----------
        rundir : str
            Path to the SCHISM model run directory
        model_dict : dict
            Dictionary with keys: 'startswith', 'var', 'var_type'
        start_date : np.datetime64
            Start of the time range for selecting model files
        end_date : np.datetime64
            End of the time range for selecting model files
        output_subdir : str, optional
            Name of the subdirectory containing output NetCDF files (default: "outputs")
        """
        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.output_dir = os.path.join(self.rundir, output_subdir)

        self._validate_model_dict()
        self._files = self._select_model_files()

        self._mesh_path = os.path.join(self.rundir, 'hgrid.gr3')
        self._mesh_x, self._mesh_y, self._mesh_depth = _parse_gr3_mesh(self._mesh_path)

    def _validate_model_dict(self) -> None:
        """
        Ensure the model_dict contains all required keys.

        Raises
        ------
        ValueError
            If required keys are missing from model_dict
        """
        required_keys = ['startswith', 'var', 'var_type']
        missing = [k for k in required_keys if k not in self.model_dict]
        if missing:
            raise ValueError(f"Missing keys in model_dict: {missing}")

    def _select_model_files(self) -> List[str]:
        """
        Select NetCDF output files within the specified time range.

        Returns
        -------
        List[str]
            List of file paths to model outputs that overlap with the requested time window

        Notes
        -----
        Only files that contain a 'time' variable and overlap the specified time window
        are selected.
        Time decoding is limited to the 'time' variable for performance and robustness.
        """
        if not os.path.isdir(self.output_dir):
            _logger.warning(f"Output directory {self.output_dir} does not exist.")
            return []

        all_files = [f for f in os.listdir(self.output_dir)
                     if os.path.isfile(os.path.join(self.output_dir, f))]
        all_files.sort(key=natural_sort_key)

        selected = []
        for fname in all_files:
            if not fname.startswith(self.model_dict['startswith']) or not fname.endswith(".nc"):
                continue

            fpath = os.path.join(self.output_dir, fname)
            try:
                with xr.open_dataset(fpath, decode_times=False) as ds:
                    if 'time' not in ds.variables:
                        continue
                    times = ds['time'].values
                    times = xr.decode_cf(ds[['time']])['time'].values  # decode only time

                    if times[-1] >= self.start_date and times[0] <= self.end_date:
                        selected.append(fpath)
            except Exception as e:
                _logger.warning(f"Error reading {fpath}: {e}")
                continue
            # selected.append(os.path.join(self.output_dir, fname))
        if not selected:
            _logger.warning(f"No files matched pattern in {self.output_dir}.\n"
            f"Make sure the model files fall within {self.start_date} and {self.end_date} ")
        return selected

    def load_variable(self, path: str) -> xr.DataArray:
        """
        Load the specified variable from a model NetCDF file.

        Parameters
        ----------
        path : str
            Path to the NetCDF file to open

        Returns
        -------
        xr.DataArray
            The requested variable, surface-only if 3D

        Notes
        -----
        For 3D variables, this method extracts the surface layer (last index of vertical layers).
        """
        _logger.info("Opening model file: %s", path)
        with xr.open_dataset(path) as ds:
            var = ds[self.model_dict['var']]
            if self.model_dict['var_type'] == '3D':
                var = var.isel(nSCHISM_vgrid_layers=-1)
        return var


    @property
    def mesh_x(self) -> np.ndarray:
        """ return mesh_x """
        return self._mesh_x
    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        """ set mesh_y """
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = new_mesh_x

    @property
    def mesh_y(self) -> np.ndarray:
        """ return mesh_y """
        return self._mesh_y
    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        """ set mesh_y """
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_y = new_mesh_y

    @property
    def mesh_depth(self) -> np.ndarray:
        """ return mesh_depth """
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        """ return file list """
        return self._files
