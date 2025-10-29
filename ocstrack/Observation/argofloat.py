""" Module for handling the Argo Float data """

import os
import glob
import logging
from typing import Union

import xarray as xr
import numpy as np

_logger = logging.getLogger(__name__)


class ArgoData:
    """
    Argo Float profile data handler.

    Loads, preprocesses, and concatenates multiple Argo NetCDF profile files
    from a specified directory. It is designed to handle the common issue of
    varying vertical levels (N_LEVELS) across different files.
    
    This class implements a robust manual loading strategy:
    1. Preprocesses each file to remove conflicting coordinates/variables.
    2. Finds the maximum N_LEVELS size across all files.
    3. Manually pads all smaller datasets with NaN to match this maximum size.
    4. Concatenates all processed, uniform datasets into one `xarray.Dataset`.

    Provides accessor properties for key variables (time, lon, lat, pres, temp, psal)
    and a time filtering method.

    Methods
    -------
    filter_by_time(start_date, end_date)
        Restrict the dataset to a specific time range.
    """

    def __init__(self, directory_path: str):
        """
        Initialize the ArgoData object by loading and concatenating
        all NetCDF datasets from a directory.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing processed Argo .nc files.

        Raises
        ------
        ValueError
            If no .nc files are found in the directory or if
            required variables (JULD, LONGITUDE, LATITUDE, etc.)
            are missing from the final combined dataset.
        RuntimeError
            If the concatenation of files fails for an unexpected reason.
        """

        search_path = os.path.join(directory_path, "*.nc")
        files = sorted(glob.glob(search_path))

        if not files:
            raise ValueError(f"No .nc files found in directory: {directory_path}")


        # Preprocessing function
        def preprocess_argo(ds):
            """
            Clean up unnecessary variables and coordinates.

            This inner function is applied to each dataset as it is loaded.
            It demotes all coordinates (solving index conflicts) and
            removes known problematic 'HISTORY_*' variables.
            """
            vars_to_drop = []
            # Nuke all coordinates to solve index conflicts
            ds = ds.reset_coords(drop=True)
            ds = ds.drop_vars(vars_to_drop, errors='ignore')

            # Ensure N_LEVELS is just a dimension (not a coordinate)
            if 'N_LEVELS' in ds.coords:
                ds = ds.reset_coords('N_LEVELS', drop=True)

            return ds


        # Load all files
        datasets = []
        max_levels = 0

        for f in files:
            print(f"Loading {f} ...")
            ds = xr.open_dataset(f, engine="netcdf4")
            ds = preprocess_argo(ds)

            # Track largest N_LEVELS
            if 'N_LEVELS' in ds.dims:
                max_levels = max(max_levels, ds.sizes['N_LEVELS'])

            datasets.append(ds)

        print(f"Maximum N_LEVELS detected: {max_levels}")


        # Pad smaller datasets with NaNs along N_LEVELS
        def pad_to_max_levels(ds, max_levels):
            """
            Manually pad a dataset to the maximum N_LEVELS size.

            This bypasses `xarray.reindex()`, which can fail due to
            stubborn index conflicts. It creates a new NaN-filled
            array and copies the existing data into it.

            Parameters
            ----------
            ds : xr.Dataset
                The dataset to pad (must be preprocessed).
            max_levels : int
                The target size for the 'N_LEVELS' dimension.

            Returns
            -------
            xr.Dataset
                A new, padded dataset.
            """
            if 'N_LEVELS' not in ds.dims:
                return ds  # skip if no depth dimension

            current_levels = ds.sizes['N_LEVELS']
            if current_levels == max_levels:
                return ds

            # Create a padded dataset with NaNs for all N_LEVELS variables
            padded = {}
            for var in ds.data_vars:
                dims = ds[var].dims
                data = ds[var].values

                if 'N_LEVELS' in dims:
                    # Create the new shape, assuming N_PROF is the first dim
                    new_shape = list(ds[var].shape)
                    n_levels_dim_index = dims.index('N_LEVELS')
                    new_shape[n_levels_dim_index] = max_levels
                    
                    new_data = np.full(
                        new_shape,
                        np.nan,
                        dtype=ds[var].dtype
                    )
                    
                    # Create a slice object to copy data into the correct position
                    slicer = [slice(None)] * ds[var].ndim
                    slicer[n_levels_dim_index] = slice(0, current_levels)
                    new_data[tuple(slicer)] = data
                    
                    padded[var] = (dims, new_data)
                else:
                    padded[var] = (dims, data)
            
            # Rebuild the dataset
            ds_new = xr.Dataset(padded, attrs=ds.attrs)

            # Copy over key data variables that became coordinates
            for coord in ['JULD', 'LATITUDE', 'LONGITUDE']:
                if coord in ds:
                    ds_new[coord] = ds[coord]

            return ds_new

        datasets = [pad_to_max_levels(ds, max_levels) for ds in datasets]


        # Concatenate all datasets
        try:
            self.ds = xr.concat(datasets, dim="N_PROF", combine_attrs="override")
            _logger.info("Concatenation successful.")

            # Restore coordinates
            coords_to_restore = ['JULD', 'LATITUDE', 'LONGITUDE']
            existing_coords = [c for c in coords_to_restore if c in self.ds]
            if existing_coords:
                self.ds = self.ds.set_coords(existing_coords)

            # Load coordinates into memory
            for c in self.ds.coords:
                if c in coords_to_restore:
                    self.ds.coords[c].load()

            _logger.info("Dataset loaded successfully.")

        except Exception as e:
            _logger.error(f"Failed to open/combine files in {directory_path}: {e}")
            raise RuntimeError(
                f"Failed to open/combine files in {directory_path}: {e}"
            )


        # Check for required Argo variables
        required_vars = ['JULD', 'LONGITUDE', 'LATITUDE', 'PRES', 'TEMP', 'PSAL']
        missing_check = [v for v in required_vars if v not in self.ds]
        if missing_check:
            self.ds.close()
            raise ValueError(
                f"Missing required Argo variables in combined dataset: {missing_check}"
            )


    @property
    def time(self):
        """Return time (JULD) as a numpy array."""
        return self.ds.JULD.values

    @property
    def lon(self):
        """Return longitudes as a numpy array."""
        return self.ds.LONGITUDE.values

    @lon.setter
    def lon(self, new_lon: Union[np.ndarray, list]):
        """
        Set new values for longitude.

        Parameters
        ----------
        new_lon : np.ndarray or list
            New longitude values to assign.

        Raises
        ------
        ValueError
            If the length of new_lon does not match N_PROF.
        """
        if len(new_lon) != self.ds.sizes['N_PROF']:
            raise ValueError("New longitude array must match existing size (N_PROF).")
        self.ds['LONGITUDE'] = ('N_PROF', np.array(new_lon))

    @property
    def lat(self):
        """Return latitudes as a numpy array."""
        return self.ds.LATITUDE.values

    @lat.setter
    def lat(self, new_lat: Union[np.ndarray, list]):
        """
        Set new values for latitude.

        Parameters
        ----------
        new_lat : np.ndarray or list
            New latitude values to assign.

        Raises
        ------
        ValueError
            If the length of new_lat does not match N_PROF.
        """
        if len(new_lat) != self.ds.sizes['N_PROF']:
            raise ValueError("New latitude array must match existing size (N_PROF).")
        self.ds['LATITUDE'] = ('N_PROF', np.array(new_lat))

    @property
    def pres(self):
        """
        Return pressure (PRES) as a numpy array.
        
        Prioritizes 'PRES_ADJUSTED' if available, falls back to 'PRES'.
        """
        return self.ds.get('PRES_ADJUSTED', self.ds['PRES']).values

    @property
    def temp(self):
        """
        Return temperature (TEMP) as a numpy array.
        
        Prioritizes 'TEMP_ADJUSTED' if available, falls back to 'TEMP'.
        """
        return self.ds.get('TEMP_ADJUSTED', self.ds['TEMP']).values

    @property
    def psal(self):
        """
        Return salinity (PSAL) as a numpy array.
        
        Prioritizes 'PSAL_ADJUSTED' if available, falls back to 'PSAL'.
        """
        return self.ds.get('PSAL_ADJUSTED', self.ds['PSAL']).values

    # ---------------------------------------------------------------------
    # FILTER METHODS
    # ---------------------------------------------------------------------

    def filter_by_time(self, start_date: str, end_date: str) -> None:
        """
        Filter the dataset by time range.

        Parameters
        ----------
        start_date : str
            ISO 8601 string representing the start date (e.g., "2020-01-01").
        end_date : str
            ISO 8601 string representing the end date (e.g., "2020-02-01").

        Notes
        -----
        This method modifies the internal `self.ds` dataset in-place.
        It ensures the 'JULD' time coordinate is correctly decoded
        and sorted before applying the time slice.
        """
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        if not np.issubdtype(self.ds['JULD'].dtype, np.datetime64):
            self.ds['JULD'] = xr.decode_cf(self.ds).JULD

        self.ds = self.ds.sortby('JULD')
        time_mask = (self.ds.JULD >= start) & (self.ds.JULD <= end)
        self.ds = self.ds.sel(N_PROF=time_mask)
