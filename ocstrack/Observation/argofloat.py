""" 
Module for handling Argo Float data.

This module provides the ArgoData class for loading, preprocessing, 
and concatenating Argo NetCDF profile files. It handles varying vertical levels,
ensures coordinates exist, and prepares the dataset for use in collocation with models.
"""

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
    from a specified directory. Handles varying vertical levels (N_LEVELS)
    by padding smaller datasets with NaNs. Ensures key coordinates exist.

    Attributes
    ----------
    ds : xarray.Dataset
        The concatenated Argo dataset, indexed by 'JULD', ready for collocation.

    Properties
    ----------
    time : numpy.ndarray
        Array of JULD times.
    lon : numpy.ndarray
        Array of longitudes.
    lat : numpy.ndarray
        Array of latitudes.
    pres : numpy.ndarray
        Array of pressures (dbar).
    temp : numpy.ndarray
        Array of temperatures (°C).
    psal : numpy.ndarray
        Array of salinities (PSU).
    depth : numpy.ndarray
        Depths (meters), approximate from pressure.

    Methods
    -------
    filter_by_time(start_date, end_date)
        Restrict dataset to a specific time range.
    """

    def __init__(self, directory_path: str):
        """
        Initialize the ArgoData object by loading all NetCDF datasets in a directory.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing processed Argo .nc files.
        """
        search_path = os.path.join(directory_path, "*.nc")
        files = sorted(glob.glob(search_path))

        if not files:
            raise ValueError(f"No .nc files found in directory: {directory_path}")

        datasets = []
        max_levels = 0

        for i, f in enumerate(files):
            print(f"Loading file {i}: {f}")
            ds = xr.open_dataset(f, engine="netcdf4")
            ds = ds.reset_coords(drop=True)

            if 'N_LEVELS' in ds.dims:
                max_levels = max(max_levels, ds.sizes['N_LEVELS'])
            
            # --- START FIX: Rename JULD_LOCATION to JULD immediately ---
            # This ensures all datasets are consistent *before* concatenation
            if 'JULD' not in ds.variables and 'JULD_LOCATION' in ds.variables:
                _logger.info(f"Renaming 'JULD_LOCATION' to 'JULD' in {f}")
                ds = ds.rename({'JULD_LOCATION': 'JULD'})
            elif 'JULD' not in ds.variables:
                 _logger.warning(f"No JULD or JULD_LOCATION found in file: {f}. Skipping this file.")
                 continue # Skip this bad file
            # --- END FIX ---

            datasets.append(ds)

        print(f"Maximum N_LEVELS detected: {max_levels}")

        # --- (pad_to_max_levels function is unchanged) ---
        def pad_to_max_levels(ds, max_levels):
            if 'N_LEVELS' not in ds.dims:
                return ds
            current_levels = ds.sizes['N_LEVELS']
            if current_levels == max_levels:
                return ds

            padded_vars = {}
            for var in ds.data_vars:
                dims = ds[var].dims
                data = ds[var].values
                if 'N_LEVELS' in dims:
                    shape = list(data.shape)
                    n_levels_index = dims.index('N_LEVELS')
                    shape[n_levels_index] = max_levels
                    new_data = np.full(shape, np.nan, dtype=data.dtype)
                    slicer = [slice(None)] * data.ndim
                    slicer[n_levels_index] = slice(0, current_levels)
                    new_data[tuple(slicer)] = data
                    padded_vars[var] = (dims, new_data)
                else:
                    padded_vars[var] = (dims, data)

            ds_new = xr.Dataset(padded_vars, attrs=ds.attrs)
            # Copy over key variables that will become coordinates
            for coord in ['JULD', 'LATITUDE', 'LONGITUDE']:
                if coord in ds:
                    ds_new[coord] = ds[coord]
            return ds_new

        datasets = [pad_to_max_levels(ds, max_levels) for ds in datasets]

        # Concatenate along N_PROF
        self.ds = xr.concat(
            datasets, 
            dim='N_PROF', 
            combine_attrs="override",
            join="outer"  # <-- ADDED: Ensures all vars (like JULD) are kept
        )

        # Ensure key variables are coordinates
        for coord in ['JULD', 'LATITUDE', 'LONGITUDE']:
            if coord in self.ds.data_vars and coord not in self.ds.coords:
                self.ds = self.ds.set_coords(coord)

        # --- RE-ORDERED LOGIC FOR ROBUSTNESS ---
        
        # 1. Sort by JULD first
        self.ds = self.ds.sortby('JULD')
        
        # 2. Find and remove duplicate times
        _, unique_idx = np.unique(self.ds['JULD'], return_index=True)
        if len(unique_idx) < self.ds.sizes['N_PROF']:
            _logger.warning(f"Found and removed {self.ds.sizes['N_PROF'] - len(unique_idx)} duplicate time profiles.")
            self.ds = self.ds.isel(N_PROF=unique_idx)

        # 3. Now it is safe to set JULD as the index
        self.ds = self.ds.set_index(N_PROF='JULD')
        
        # 4. (Optional but clean) Rename the dimension itself
        self.ds = self.ds.rename({'N_PROF': 'JULD'})

        # --- END RE-ORDERED LOGIC ---

        print("Argo dataset loaded successfully.")
        print(f"Dataset dims: {self.ds.dims}")
        print(f"Dataset coords: {list(self.ds.coords)}")
        print(f"Dataset variables: {list(self.ds.data_vars)}")

    @property
    def time(self):
        """Return time (JULD) as a numpy array."""
        return self.ds['JULD'].values

    @property
    def lon(self):
        """Return longitudes as a numpy array."""
        return self.ds.LONGITUDE.values

    @lon.setter
    def lon(self, new_lon: Union[np.ndarray, list]):
        """
        Set new values for longitude.
        (FIXED to use the new 'JULD' dimension)
        """
        if len(new_lon) != self.ds.sizes['JULD']:
             raise ValueError("New longitude array must match existing size (JULD).")
        # Assign new values to the coordinate, using the correct dimension name
        self.ds['LONGITUDE'] = (self.ds.JULD.name, np.array(new_lon))

    @property
    def lat(self):
        """Return latitudes as a numpy array."""
        return self.ds.LATITUDE.values

    @lat.setter
    def lat(self, new_lat: Union[np.ndarray, list]):
        """
        Set new values for latitude.
        (FIXED to use the new 'JULD' dimension)
        """
        if len(new_lat) != self.ds.sizes['JULD']:
             raise ValueError("New latitude array must match existing size (JULD).")
        # Assign new values to the coordinate, using the correct dimension name
        self.ds['LATITUDE'] = (self.ds.JULD.name, np.array(new_lat))

    @property
    def pres(self):
        """Return pressure (dbar) as a numpy array."""
        return self.ds.get('PRES_ADJUSTED', self.ds['PRES']).values

    @property
    def temp(self):
        """Return temperature (°C) as a numpy array."""
        return self.ds.get('TEMP_ADJUSTED', self.ds['TEMP']).values

    @property
    def psal(self):
        """Return salinity (PSU) as a numpy array."""
        return self.ds.get('PSAL_ADJUSTED', self.ds['PSAL']).values

    @property
    def depth(self):
        """Return depth (meters) from pressure. Simple approximation: depth ≈ -1.0197 * pres."""
        return self.pres * -1.0197

    def filter_by_time(self, start_date: str, end_date: str) -> None:
        """
        Filter the dataset by a time range.

        Parameters
        ----------
        start_date : str
            Start date (ISO 8601 string, e.g., "2020-01-01").
        end_date : str
            End date (ISO 8601 string, e.g., "2020-02-01").

        Notes
        -----
        Modifies the internal dataset in-place.
        (FIXED to use .sel() on the new 'JULD' index)
        """
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        # Use .sel() for index-based selection, which is much faster
        self.ds = self.ds.sel(JULD=slice(start, end))
