''' Module for handling the Model data '''

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

        self._time = None

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
            _logger.warning(f"No files matched pattern in {self.output_dir}.
"
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
        """Return mesh longitudes."""
        return self._mesh_x

    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        """Set mesh longitudes."""
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = new_mesh_x

    @property
    def mesh_y(self) -> np.ndarray:
        """Return mesh latitudes."""
        return self._mesh_y

    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        """Set mesh latitudes."""
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New latitude array must match existing size.")
        self._mesh_y = new_mesh_y

    @property
    def mesh_depth(self) -> np.ndarray:
        """Return mesh node depths."""
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        """Return the list of selected model output files."""
        return self._files

    @property
    def time(self) -> np.ndarray:
        """
        Return the concatenated time array for all selected files.

        The time array is cached after the first call.
        """
        if self._time is not None:
            return self._time
        if not self.files:
            return np.array([])

        all_times = []
        # print("Generating global time array from files...") # Optional debug print
        for fpath in self.files:
            try:
                # Open strictly to read the time variable
                with xr.open_dataset(fpath) as ds:
                    # Ensure we get datetime64 objects
                    if 'time' in ds:
                        t = ds['time'].values
                        # If simple float/int, try to decode. If already datetime, use as is.
                        # (SCHISM usually needs decoding if the file wasn't saved with CF conventions)
                        if not np.issubdtype(t.dtype, np.datetime64):
                             t = xr.decode_cf(ds[['time']])['time'].values
                        all_times.append(t)
            except Exception as e:
                print(f"Warning: Could not read time from {fpath}: {e}")

        if all_times:
            self._time = np.concatenate(all_times)
            # Ensure it is sorted, just in case files were out of order
            self._time.sort()
        else:
            self._time = np.array([])

        return self._time

class ADCSWAN:
    """
    ADCIRC+SWAN model interface.

    Handles selection and loading of model outputs from a single ADCIRC+SWAN
    NetCDF file. This class locates a single file based on 'startswith' and
    validates its time range against the requested start/end dates.
    It reads mesh coordinates (x, y, depth) directly from this file.

    This class mimics the SCHISM interface for compatibility in the collocation
    workflow.
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
        """Return mesh longitudes."""
        return self._mesh_x

    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        """Set mesh longitudes."""
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = np.asarray(new_mesh_x)

    @property
    def mesh_y(self) -> np.ndarray:
        """Return mesh latitudes."""
        return self._mesh_y

    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        """Set mesh latitudes."""
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New latitude array must match existing size.")
        self._mesh_y = np.asarray(new_mesh_y)

    @property
    def mesh_depth(self) -> np.ndarray:
        """Return mesh node depths."""
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        """Return the model file path (a list with 0 or 1 item)."""
        return self._files


class WW3:
    """
    WaveWatchIII (WW3) model interface.

    Handles selection, filtering, and loading of model outputs from a WW3 run directory.
    This assumes a run directory structure where:
    .
    ├── RunDir
        ├── yyyymmdd.hhmmss.out_grd.ww3.nc
        └── ...

    Methods
    -------
    load_variable(path)
        Load model variable from a NetCDF file
    """
    def __init__(self, rundir: str,
                 model_dict: dict,
                 start_date: np.datetime64,
                 end_date: np.datetime64):
        """
        Initialize a WW3 model run

        Parameters
        ----------
        rundir : str
            Path to the WW3 model run directory
        model_dict : dict
            Dictionary with keys: 'var', 'var_type'
        start_date : np.datetime64
            Start of the time range for selecting model files
        end_date : np.datetime64
            End of the time range for selecting model files
        """
        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.output_dir = self.rundir

        self._time = None

        self._validate_model_dict()
        self._files = self._select_model_files()

        if self._files:
            self._mesh_path = self._files[0]
            self._mesh_x, self._mesh_y, self._mesh_depth = self._load_mesh_data(self._mesh_path)
            _logger.info(f"WW3 mesh loaded from {self._mesh_path}")
        else:
            self._mesh_path = None
            self._mesh_x, self._mesh_y, self._mesh_depth = (np.array([]),
                                                            np.array([]),
                                                            np.array([]))
            _logger.warning("No WW3 file found, mesh could not be loaded.")

    def _validate_model_dict(self) -> None:
        """
        Ensure the model_dict contains all required keys.

        Raises
        -------
        ValueError
            If required keys are missing from model_dict
        """
        required_keys = ['var']
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
        """
        if not os.path.isdir(self.output_dir):
            _logger.warning(f"Output directory {self.output_dir} does not exist.")
            return []

        all_files = [f for f in os.listdir(self.output_dir)
                     if f.endswith(".out_grd.ww3.nc")]
        all_files.sort()

        selected = []
        for fname in all_files:
            fpath = os.path.join(self.output_dir, fname)
            try:
                with xr.open_dataset(fpath, decode_times=False) as ds:
                    if 'time' not in ds.variables:
                        continue
                    times = ds['time'].values
                    times = xr.decode_cf(ds[['time']])['time'].values

                    if times[-1] >= self.start_date and times[0] <= self.end_date:
                        selected.append(fpath)
            except Exception as e:
                _logger.warning(f"Error reading {fpath}: {e}")
                continue

        if not selected:
            _logger.warning(f"No files matched pattern in {self.output_dir}.
"
            f"Make sure the model files fall within {self.start_date} and {self.end_date} ")
        return selected

    def _load_mesh_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse the WW3 NetCDF file to extract node coordinates.
        It is assumed that WW3 is an unstructured grid with 'longitude' and 'latitude' variables.
        Depth is not available in WW3 output, so it is returned as an array of NaNs.
        """
        _logger.debug(f"Loading mesh data from {filepath}")
        try:
            with xr.open_dataset(filepath) as ds:
                lons = ds['lon'].load().values.squeeze()
                lats = ds['lat'].load().values.squeeze()
                depths = np.full_like(lons, np.nan)
            return lons, lats, depths
        except Exception as e:
            _logger.error(f"Failed to load mesh data from {filepath}: {e}")
            return np.array([]), np.array([]), np.array([])

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
            The requested variable, sliced by time.
        """
        _logger.info("Opening model file: %s", path)
        try:
            ds = xr.open_dataset(path)
            var = ds[self.model_dict['var']]
            if 'ny' in var.dims:
                var = var.squeeze('ny')
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
        """Return mesh longitudes."""
        return self._mesh_x

    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        """Set mesh longitudes."""
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = np.asarray(new_mesh_x)

    @property
    def mesh_y(self) -> np.ndarray:
        """Return mesh latitudes."""
        return self._mesh_y

    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        """Set mesh latitudes."""
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New latitude array must match existing size.")
        self._mesh_y = np.asarray(new_mesh_y)

    @property
    def mesh_depth(self) -> np.ndarray:
        """Return mesh node depths."""
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        """Return the list of selected model output files."""
        return self._files

    @property
    def time(self) -> np.ndarray:
        """
        Return the concatenated time array for all selected files.

        The time array is cached after the first call.
        """
        if self._time is not None:
            return self._time
        if not self.files:
            return np.array([])

        all_times = []
        for fpath in self.files:
            try:
                with xr.open_dataset(fpath) as ds:
                    if 'time' in ds:
                        t = ds['time'].values
                        if not np.issubdtype(t.dtype, np.datetime64):
                             t = xr.decode_cf(ds[['time']])['time'].values
                        all_times.append(t)
            except Exception as e:
                print(f"Warning: Could not read time from {fpath}: {e}")

        if all_times:
            self._time = np.concatenate(all_times)
            self._time.sort()
        else:
            self._time = np.array([])

        return self._time

def stretching(Vstr, thts, thtb, hc, N, kgrid):
    """
     STRETCHING:  Compute ROMS vertical coordinate stretching function

     [s,C]=stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, report)

     Given vertical terrain-following vertical stretching parameters, this
     routine computes the vertical stretching function used in ROMS vertical
     coordinate transformation. Check the following link for details:

        https://www.myroms.org/wiki/index.php/Vertical_S-coordinate

     On Input:

        Vstretching   Vertical stretching function:
                        Vstretching = 1,  original (Song and Haidvogel, 1994)
                        Vstretching = 2,  A. Shchepetkin (UCLA-ROMS, 2005)
                        Vstretching = 3,  R. Geyer BBL refinement
                        Vstretching = 4,  A. Shchepetkin (UCLA-ROMS, 2010)
        theta_s       S-coordinate surface control parameter (scalar)
        theta_b       S-coordinate bottom control parameter (scalar)
        hc            Width (m) of surface or bottom boundary layer in which
                        higher vertical resolution is required during
                        stretching (scalar)
        N             Number of vertical levels (scalar)
        kgrid         Depth grid type logical switch:
                        kgrid = 0,        function at vertical RHO-points
                        kgrid = 1,        function at vertical W-points
     On Output:

        s             S-coordinate independent variable, [-1 <= s <= 0] at
                        vertical RHO- or W-points (vector)
        C             Nondimensional, monotonic, vertical stretching function,
                        C(s), 1D array, [-1 <= C(s) <= 0]

    """
    s=[]
    C=[]

    Np=N+1

    #-----------------------------------------------------------------
    # Compute ROMS S-coordinates vertical stretching function
    #-----------------------------------------------------------------

    # Original vertical stretching function (Song and Haidvogel, 1994).
    if (Vstr == 1):
        ds = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Ptheta = np.sinh(thts*s)/np.sinh(thts)
            Rtheta = np.tanh(thts*(s+0.5))/(2.0*np.tanh(0.5*thts))-0.5
            C      = (1.0-thtb)*Ptheta+thtb*Rtheta
        else:
            C=s

    # A. Shchepetkin (UCLA-ROMS, 2005) vertical stretching function.
    if (Vstr==2):
        alfa = 1.0
        beta = 1.0
        ds   = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Csur = (1.0-np.cosh(thts*s))/(np.cosh(thts)-1.0)
            if (thtb > 0):
                Cbot   = -1.0+np.sinh(thtb*(s+1.0))/np.sinh(thtb)
                weigth = (s+1.0)**alfa*(1.0+(alfa/beta)*(1.0-(s+1.0)**beta))
                C      = weigth*Csur+(1.0-weigth)*Cbot
            else:
                C=Csur
        else:
            C=s

    # R. Geyer BBL vertical stretching function.
    if (Vstr==3):
        ds   = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            exp_s = thts   # surface stretching exponent
            exp_b = thtb   # bottom  stretching exponent
            alpha = 3      # scale factor for all hyperbolic functions
            Cbot  = np.log(np.cosh(alpha*(s+1.0)**exp_b))/np.log(np.cosh(alpha))-1.0
            Csur  = -np.log(cosh(alpha*abs(s)**exp_s))/log(cosh(alpha))
            weight= (1-np.tanh( alpha*(s+0.5)))/2.0
            C     = weight*Cbot+(1.0-weight)*Csur
        else:
            C=s

    # A. Shchepetkin (UCLA-ROMS, 2010) double vertical stretching function
    # with bottom refinement
    if (Vstr == 4):
        ds   = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Csur = (1.0-np.cosh(thts*s))/(np.cosh(thts)-1.0)
        else:
            Csur = -s**2

        if (thtb > 0):
            Cbot = (np.exp(thtb*Csur)-1.0)/(1.0-np.exp(-thtb))
            C    = Cbot
        else:
            C    = Csur

    return (s,C)


def set_depth( Vtr, Vstr, thts, thtb, hc, N, igrid, h, zeta ):
    """
     Given a batymetry (h), free-surface (zeta) and terrain-following
     parameters, this function computes the 3D depths for the requested
     C-grid location. If the free-surface is not provided, a zero value
     is assumed resulting in unperturb depths.  This function can be
     used when generating initial conditions or climatology data for
     an application. Check the following link for details:

        https://www.myroms.org/wiki/index.php/Vertical_S-coordinate

     On Input:

        Vtransform    Vertical transformation equation:

                        Vtransform = 1,   original transformation

                        z(x,y,s,t)=Zo(x,y,s)+zeta(x,y,t)*[1+Zo(x,y,s)/h(x,y)]

                        Zo(x,y,s)=hc*s+[h(x,y)-hc]*C(s)

                        Vtransform = 2,   new transformation

                        z(x,y,s,t)=zeta(x,y,t)+[zeta(x,y,t)+h(x,y)]*Zo(x,y,s)

                        Zo(x,y,s)=[hc*s(k)+h(x,y)*C(k)]/[hc+h(x,y)]

        Vstretching   Vertical stretching function:
                        Vstretching = 1,  original (Song and Haidvogel, 1994)
                        Vstretching = 2,  A. Shchepetkin (UCLA-ROMS, 2005)
                        Vstretching = 3,  R. Geyer BBL refinement
                        Vstretching = 4,  A. Shchepetkin (UCLA-ROMS, 2010)

        theta_s       S-coordinate surface control parameter (scalar)

        theta_b       S-coordinate bottom control parameter (scalar)

        hc            Width (m) of surface or bottom boundary layer in which
                        higher vertical resolution is required during
                        stretching (scalar)

        N             Number of vertical levels (scalar)

        igrid         Staggered grid C-type (integer):
                        igrid=1  => density points
                        igrid=2  => streamfunction points
                        igrid=3  => u-velocity points
                        igrid=4  => v-velocity points
                        igrid=5  => w-velocity points

        h             Bottom depth, 2D array at RHO-points (m, positive),
                        h(1:Lp+1,1:Mp+1)

        zeta          Free-surface, 2D array at RHO-points (m), OPTIONAL,
                        zeta(1:Lp+1,1:Mp+1)

     On Output:

        z             Depths (m, negative), 3D array
    """

    Np      = N+1
    Lp,Mp   = np.shape(h)
    L       = Lp-1
    M       = Mp-1
    if (igrid==5):
        z   = np.empty((Lp,Mp,Np))
    else:
        z   = np.empty((Lp,Mp,N))

    hmin    = np.min(h)
    hmax    = np.max(h)

    if (igrid == 5):
        kgrid=1
    else:
        kgrid=0

    s,C = stretching(Vstr, thts, thtb, hc, N, kgrid);
    #-----------------------------------------------------------------------
    #  Average bathymetry and free-surface at requested C-grid type.
    #-----------------------------------------------------------------------

    if (igrid==1):
        hr    = h
        zetar = zeta
    elif (igrid==2):
        hp    = 0.25*(h[0:L,0:M]+h[1:Lp,0:M]+h[0:L,1:Mp]+h[1:Lp,1:Mp])
        zetap = 0.25*(zeta[0:L,0:M]+zeta[1:Lp,0:M]+zeta[0:L,1:Mp]+zeta[1:Lp,1:Mp])
    elif (igrid==3):
        hu    = 0.5*(h[0:L,0:Mp]+h[1:Lp,0:Mp])
        zetau = 0.5*(zeta[0:L,0:Mp]+zeta[1:Lp,0:Mp])
    elif (igrid==4):
        hv    = 0.5*(h[0:Lp,0:M]+h[0:Lp,1:Mp])
        zetav = 0.5*(zeta[0:Lp,0:M]+zeta[0:Lp,1:Mp])
    elif (igrid==5):
        hr    = h
        zetar = zeta

    #----------------------------------------------------------------------
    # Compute depths (m) at requested C-grid location.
    #----------------------------------------------------------------------
    if (Vtr == 1):
        if (igrid==1):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hr
                z[:,:,k] = z0 + zetar*(1.0 + z0/hr)
        elif (igrid==2):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hp
                z[:,:,k] = z0 + zetap*(1.0 + z0/hp)
        elif (igrid==3):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hu
                z[:,:,k] = z0 + zetau*(1.0 + z0/hu)
        elif (igrid==4):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hv
                z[:,:,k] = z0 + zetav*(1.0 + z0/hv)
        elif (igrid==5):
            z[:,:,0] = -hr
            for k in range (0,Np):
                z0 = (s[k]-C[k])*hc + C[k]*hr
                z[:,:,k] = z0 + zetar*(1.0 + z0/hr)
    elif (Vtr==2):
        if (igrid==1):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hr)/(hc+hr)
                z[:,:,k] = zetar+(zeta+hr)*z0
        elif (igrid==2):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hp)/(hc+hp)
                z[:,:,k] = zetap+(zetap+hp)*z0
        elif (igrid==3):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hu)/(hc+hu)
                z[:,:,k] = zetau+(zetau+hu)*z0
        elif (igrid==4):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hv)/(hc+hv)
                z[:,:,k] = zetav+(zetav+hv)*z0
        elif (igrid==5):
            for k in range (0,Np):
                z0 = (hc*s[k]+C[k]*hr)/(hc+hr)
                z[:,:,k] = zetar+(zetar+hr)*z0

    return z

class ROMS:
    """
    ROMS model interface.
    """
    def __init__(self, rundir: str, model_dict: dict, start_date: np.datetime64, end_date: np.datetime64):
        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.output_dir = os.path.join(self.rundir, "HIS")

        self._time = None
        self.Vtransform = None
        self.Vstretching = None
        self.theta_s = None
        self.theta_b = None
        self.hc = None
        self.N = None

        self._parse_ocean_in()
        self._validate_model_dict()
        self._files = self._select_model_files()
        self._load_mesh_data()

    def _parse_ocean_in(self):
        """Parse the ocean.in file to get vertical coordinate parameters."""
        ocean_in_path = os.path.join(self.rundir, "ocean.in")
        if not os.path.exists(ocean_in_path):
            raise FileNotFoundError("ocean.in not found in the run directory.")

        with open(ocean_in_path, "r") as f:
            for line in f:
                if "Vtransform" in line:
                    self.Vtransform = int(line.split("==")[1].strip())
                elif "Vstretching" in line:
                    self.Vstretching = int(line.split("==")[1].strip())
                elif "THETA_S" in line:
                    self.theta_s = float(line.split("==")[1].split("d")[0].strip())
                elif "THETA_B" in line:
                    self.theta_b = float(line.split("==")[1].split("d")[0].strip())
                elif "TCLINE" in line:
                    self.hc = float(line.split("==")[1].split("d")[0].strip())
                elif "N ==" in line:
                    self.N = int(line.split("==")[1].strip().split("!")[0])


    def _validate_model_dict(self):
        """Validate the model_dict."""
        required_keys = ['var', 'var_type']
        missing = [k for k in required_keys if k not in self.model_dict]
        if missing:
            raise ValueError(f"Missing keys in model_dict: {missing}")

    def _select_model_files(self):
        """Select model files."""
        if not os.path.isdir(self.output_dir):
            _logger.warning(f"Output directory {self.output_dir} does not exist.")
            return []

        all_files = [f for f in os.listdir(self.output_dir) if f.startswith("roms_his_") and f.endswith(".nc")]
        all_files.sort(key=natural_sort_key)

        selected = []
        for fname in all_files:
            fpath = os.path.join(self.output_dir, fname)
            try:
                with xr.open_dataset(fpath, decode_times=False) as ds:
                    if 'ocean_time' not in ds.variables:
                        continue
                    times = ds['ocean_time'].values
                    times = xr.decode_cf(ds[['ocean_time']])['ocean_time'].values

                    if times[-1] >= self.start_date and times[0] <= self.end_date:
                        selected.append(fpath)
            except Exception as e:
                _logger.warning(f"Error reading {fpath}: {e}")
                continue
        return selected

    def _load_mesh_data(self):
        """Load mesh data."""
        if not self.files:
            self._mesh_x, self._mesh_y, self.h = None, None, None
            return

        grid_file = None
        with xr.open_dataset(self.files[0]) as ds:
            if 'grd_file' in ds.attrs:
                grid_file = ds.attrs['grd_file']

        if grid_file and os.path.exists(grid_file):
            with xr.open_dataset(grid_file) as ds:
                self._mesh_x = ds['lon_rho'].values.flatten()
                self._mesh_y = ds['lat_rho'].values.flatten()
                self.h = ds['h'].values
        else:
            with xr.open_dataset(self.files[0]) as ds:
                self._mesh_x = ds['lon_rho'].values.flatten()
                self._mesh_y = ds['lat_rho'].values.flatten()
                self.h = ds['h'].values

    def load_3d_file_pair(self, f_main_path: str):
        """Load 3D file pair."""
        main_var = self.model_dict['var']
        zcor_var = self.model_dict.get('zcor_var', 'z_rho')

        with xr.open_dataset(f_main_path) as ds:
            main_var_data = ds[main_var]
            zeta = ds['zeta']

            z_rho = set_depth(self.Vtransform, self.Vstretching, self.theta_s, self.theta_b, self.hc, self.N, 1, self.h, zeta.values)
            
            ds_out = xr.Dataset(
                {
                    main_var: main_var_data,
                    zcor_var: (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), z_rho)
                },
                coords=ds.coords
            )
            return ds_out

    @property
    def mesh_x(self):
        return self._mesh_x

    @property
    def mesh_y(self):
        return self._mesh_y

    @property
    def files(self):
        return self._files

    @property
    def time(self):
        if self._time is not None:
            return self._time
        if not self.files:
            return np.array([])

        all_times = []
        for fpath in self.files:
            try:
                with xr.open_dataset(fpath) as ds:
                    if 'ocean_time' in ds:
                        t = ds['ocean_time'].values
                        if not np.issubdtype(t.dtype, np.datetime64):
                             t = xr.decode_cf(ds[['ocean_time']])['ocean_time'].values
                        all_times.append(t)
            except Exception as e:
                print(f"Warning: Could not read time from {fpath}: {e}")

        if all_times:
            self._time = np.concatenate(all_times)
            self._time.sort()
        else:
            self._time = np.array([])

        return self._time
