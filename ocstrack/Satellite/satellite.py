""" Module for handling the Satellite data """

from typing import Union

import xarray as xr
import numpy as np

class SatelliteData:
    """
    Satellite altimetry data handler.

    Loads a NetCDF file containing satellite-derived variables such as
    significant wave height (SWH), sea level anomaly (SLA), and metadata.
    
    Provides accessor properties and time filtering functionality.

    Methods
    -------
    filter_by_time(start_date, end_date)
        Restrict the dataset to a specific time range
    """
    def __init__(self, filepath: str):
        """
        Initialize the SatelliteData object by loading a NetCDF dataset.

        Parameters
        ----------
        filepath : str
            Path to the satellite NetCDF file

        Raises
        ------
        ValueError
            If required variables are missing from the dataset
        """
        self.ds = xr.open_dataset(filepath)

        # Check essential variables exist
        required_vars = ['time', 'lon', 'lat', 'swh', 'sla', 'source']
        missing = [v for v in required_vars if v not in self.ds.variables]
        if missing:
            raise ValueError(f"Missing required variables in dataset: {missing}")

    @property
    def time(self):
        """ """
        return self.ds.time.values

    @property
    def lon(self):
        """ """
        return self.ds.lon.values
    @lon.setter
    def lon(self, new_lon: Union[np.ndarray, list]):
        """ """
        if len(new_lon) != len(self.ds.lon):
            raise ValueError("New longitude array must match existing size.")
        self.ds['lon'] = ('time', np.array(new_lon))

    @property
    def lat(self):
        """ """
        return self.ds.lat.values
    @lat.setter
    def lat(self, new_lat: Union[np.ndarray, list]):
        """ """
        if len(new_lat) != len(self.ds.lat):
            raise ValueError("New latitude array must match existing size.")
        self.ds['lat'] = ('time', np.array(new_lat))

    @property
    def swh(self):
        """ """
        return self.ds.swh.values

    @property
    def sla(self):
        """ """
        return self.ds.sla.values

    @property
    def source(self):
        """ """
        return self.ds.source.values

    def filter_by_time(self,
                       start_date: str,
                       end_date: str) -> None:
        """
        Filter the dataset by time range.

        Parameters
        ----------
        start_date : str
            ISO 8601 string representing the start date
        end_date : str
            ISO 8601 string representing the end date

        Notes
        -----
        The method converts time variables to datetime64 and ensures the time
        dimension is sorted before filtering.
        """
        # Convert to datetime for safety
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        # Ensure time is datetime64 and sorted
        if not np.issubdtype(self.ds['time'].dtype, np.datetime64):
            self.ds['time'] = xr.decode_cf(self.ds).time

        self.ds = self.ds.sortby('time')  # Ensure sorted time axis
        self.ds = self.ds.sel(time=slice(start, end))
