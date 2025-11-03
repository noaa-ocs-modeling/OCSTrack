""" Module for handling the Model data """

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

        valid_types = ['2D', '3D_Surface', '3D_Profile']
        var_type = self.model_dict['var_type']

        if var_type not in valid_types:
            raise ValueError(
                f"var_type must be one of {valid_types}, "
                f"but got '{var_type}'"
            )

        if var_type == '3D_Profile':
            profile_keys = ['zcor_var', 'zcor_startswith']
            missing_profile = [k for k in profile_keys if k not in self.model_dict]
            if missing_profile:
                raise ValueError(
                    f"For '3D_Profile', model_dict must also include: {missing_profile}"
                )

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
            The requested variable, , surface-only if var_type is '3D_Surface'

        Notes
        -----
        For 3D variables, this method extracts the surface layer (last index of vertical layers).
        """
        _logger.info("Opening model file: %s", path)
        with xr.open_dataset(path) as ds:
            var = ds[self.model_dict['var']]

            # Check for the new '3D_Surface' type
            if self.model_dict.get('var_type') == '3D_Surface':
                _logger.info("Extracting surface layer from 3D variable.")
                var = var.isel(nSCHISM_vgrid_layers=-1)
        return var

    def load_3d_file_pair(self, f_main_path: str) -> xr.Dataset:
        """
        Loads a single 3D variable file and its matching z-coordinate file.

        This is the memory-efficient method for 3D collocation.

        Parameters
        ----------
        f_main_path : str
            The full path to the main variable file (e.g., "temperature_84.nc").

        Returns
        -------
        xr.Dataset
            A single, in-memory dataset containing the 3D variable
            and its 'zcor' variable for the time steps in that file.
        """

        main_var = self.model_dict['var']
        main_startswith = self.model_dict['startswith']
        zcor_var = self.model_dict['zcor_var']
        zcor_startswith = self.model_dict['zcor_startswith']
        f_main_name = os.path.basename(f_main_path)

        # Construct the zcor filename
        file_suffix = f_main_name[len(main_startswith):]
        f_zcor_name = f"{zcor_startswith}{file_suffix}"
        f_zcor_path = os.path.join(self.output_dir, f_zcor_name)

        if not os.path.exists(f_zcor_path):
            _logger.error(f"Cannot find matching zcor file for {f_main_path}")
            _logger.error(f"Looked for: {f_zcor_path}")
            raise ValueError(f"Missing zcor file: {f_zcor_name}")

        try:
            ds_main = xr.open_dataset(f_main_path, engine='netcdf4')
            ds_zcor = xr.open_dataset(f_zcor_path, engine='netcdf4')

            # Keep only the essential variables
            ds_main = ds_main[[main_var]]
            ds_zcor = ds_zcor[[zcor_var]]

            # Merge
            ds_merged = xr.merge([ds_main, ds_zcor])

            # Slice by time *before* loading, just in case
            time_slice = slice(self.start_date, self.end_date)
            ds_sliced = ds_merged.sel(time=time_slice)

            # Load this small chunk into memory
            ds_sliced.load()
            ds_main.close()
            ds_zcor.close()

            return ds_sliced

        except Exception as e:
            _logger.error(f"Error opening/merging {f_main_path} and {f_zcor_path}: {e}")
            raise

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


class ADCSWAN:
    """
    ADCIRC+SWAN model interface

    Handles selection and loading of model outputs from a single ADCIRC+SWAN NetCDF file.
    The class locates a single file based on 'startswith' and validates its
    time range against the requested start/end dates.
    It reads mesh coordinates (x, y, depth) directly from this file.

    This class mimics the SCHISM interface for compatibility.
    """
    def __init__(self, rundir: str,
                 model_dict: dict,
                 start_date: np.datetime64,
                 end_date: np.datetime64,
                 **kwargs):
        """
        Initialize an ADCIRC+SWAN model run

        Parameters
        ----------
        rundir : str
            Path to the directory containing the model output NetCDF file
        model_dict : dict
            Dictionary with keys: 'startswith', 'var'.
            'startswith' is the prefix of the NetCDF file (e.g., "swan_HS.63")
            'var' is the variable to be loaded (e.g., "swan_HS")
        start_date : np.datetime64
            Start of the time range for validation and slicing (if needed)
        end_date : np.datetime64
            End of the time range for validation and slicing (if needed)
        **kwargs :
            Ignored. Added for interface compatibility with SCHISM (e.g., output_subdir).
        """
        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)

        # Note: self.output_dir is kept for SCHISM compatibility but points to rundir
        self.output_dir = self.rundir

        self._validate_model_dict()
        self._files = self._select_model_files()

        if self._files:
            self._mesh_path = self._files[0]
            self._mesh_x, self._mesh_y, self._mesh_depth = self._load_mesh_data(self._mesh_path)
            _logger.info(f"ADC+SWAN mesh loaded from {self._mesh_path}")
        else:
            self._mesh_path = None
            self._mesh_x, self._mesh_y, self._mesh_depth = (np.array([]),
                                                            np.array([]),
                                                            np.array([]))
            _logger.warning("No ADC+SWAN file found, mesh could not be loaded.")

    def _validate_model_dict(self) -> None:
        """
        Ensure the model_dict contains all required keys.
        Raises
        ------
        ValueError
            If required keys are missing from model_dict
        """
        required_keys = ['startswith', 'var'] # 'var_type' is not required for ADCSWAN
        missing = [k for k in required_keys if k not in self.model_dict]
        if missing:
            raise ValueError(f"Missing keys in model_dict: {missing}")

    def _load_mesh_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse the ADC+SWAN NetCDF file to extract node coordinates and depth.
        """
        _logger.debug(f"Loading mesh data from {filepath}")
        try:
            with xr.open_dataset(filepath, drop_variables=['neta','nvel']) as ds:
                # Use .load() to read data into memory and close the file
                lons = ds['x'].load().values
                lats = ds['y'].load().values
                depths = ds['depth'].load().values
            return lons, lats, depths
        except Exception as e:
            _logger.error(f"Failed to load mesh data from {filepath}: {e}")
            return np.array([]), np.array([]), np.array([])


    def _select_model_files(self) -> List[str]:
        """
        Select the ADCIRC+SWAN NetCDF output file and validate its time range.

        Returns
        -------
        List[str]
            A list containing the path to the model file, if found and valid.
            Otherwise, an empty list.
        """
        if not os.path.isdir(self.rundir):
            _logger.warning(f"Run directory {self.rundir} does not exist.")
            return []

        all_files = [f for f in os.listdir(self.rundir)
                     if os.path.isfile(os.path.join(self.rundir, f))]
        all_files.sort(key=natural_sort_key)

        selected = []
        file_pattern = self.model_dict['startswith']

        found_files = [f for f in all_files if f.startswith(file_pattern) and f.endswith(".nc")]

        if not found_files:
            _logger.warning(f"No file found in {self.rundir} starting with '{file_pattern}'")
            return []

        if len(found_files) > 1:
            _logger.warning(f"Multiple files found matching '{file_pattern}'. "
                            f"Using the first one: {found_files[0]}")

        fpath = os.path.join(self.rundir, found_files[0])

        try:
            # Check time range for overlap
            with xr.open_dataset(fpath, decode_times=False, drop_variables=['neta','nvel']) as ds:
                if 'time' not in ds.variables:
                    _logger.warning(f"File {fpath} has no 'time' variable. Skipping.")
                    return []

                # Decode only time for validation
                times = xr.decode_cf(ds[['time']])['time'].values

                if times[-1] >= self.start_date and times[0] <= self.end_date:
                    selected.append(fpath)
                else:
                    _logger.warning(f"File {fpath} time range ({times[0]} to {times[-1]}) "
                                    f"does not overlap with requested range "
                                    f"({self.start_date} to {self.end_date}).")
        except Exception as e:
            _logger.warning(f"Error reading {fpath}: {e}")
            return []

        return selected

    def load_variable(self, path: str) -> xr.DataArray:
        """
        Load the specified variable from the model NetCDF file.

        Parameters
        ----------
        path : str
            Path to the NetCDF file to open (should be the one in self.files)

        Returns
        -------
        xr.DataArray
            The requested variable, sliced by time.
        
        Notes
        -----
        For compatibility with the SCHISM class pattern, this method loads
        the variable from the *given path*.
        """
        _logger.info("Opening model file: %s", path)
        try:
            # Xarray will open the file, slice, and then load.
            ds = xr.open_dataset(path, drop_variables=['neta','nvel'])
            var = ds[self.model_dict['var']]

            time_slice = slice(self.start_date, self.end_date)
            var_sliced = var.sel(time=time_slice)

            var_loaded = var_sliced.load()
            ds.close()

            return var_loaded

        except KeyError:
            _logger.error(f"Variable '{self.model_dict['var']}' not found in {path}")
            ds.close()
            raise
        except Exception as e:
            _logger.error(f"Error loading variable from {path}: {e}")
            if 'ds' in locals():
                ds.close()
            raise

    @property
    def mesh_x(self) -> np.ndarray:
        """ return mesh_x (longitude) """
        return self._mesh_x
    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        """ set mesh_x """
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = np.asarray(new_mesh_x)

    @property
    def mesh_y(self) -> np.ndarray:
        """ return mesh_y (latitude) """
        return self._mesh_y
    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        """ set mesh_y """
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New latitude array must match existing size.")
        self._mesh_y = np.asarray(new_mesh_y)

    @property
    def mesh_depth(self) -> np.ndarray:
        """ return mesh_depth """
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        """ return file list (will contain 0 or 1 file) """
        return self._files
