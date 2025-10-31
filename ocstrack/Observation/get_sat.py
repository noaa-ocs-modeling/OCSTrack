""" Downlaod and pre-process Satellite Altimetry data """

import os
import requests
import logging
import shutil
import time

import xarray as xr

from datetime import datetime, timedelta
from typing import Union, Optional, List
from tqdm import tqdm

from .urls import URL_TEMPLATES


_logger = logging.getLogger(__name__)


def generate_daily_dates(start_date_str: str,
                         end_date_str: str) -> List[str]:
    """
    This function generates a list of formated
    dates between the start and end dates.

    Args:
        start_date_str: String with dates as 'YYYY-MM-DD'
        end_date_str: String with dates as 'YYYY-MM-DD'

    Returns:
        List of formated dates (daily).
    """

    start_date = datetime.strptime(start_date_str,
                                   '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str,
                                 '%Y-%m-%d')

    return [(start_date + timedelta(days=i)).strftime('%Y%m%d')
            for i in range((end_date - start_date).days + 1)]


def download_sat_data(dates_str: List[str],
                      url_template: str,
                      raw_dir: str,
                      sat: str,
                      retries: int = 1,
                      delay: int = 5) -> List[str]:
    """
    This function downloads the satellite data

    Parameters
    ----------
        dates_str: List with all the dates data will be downloaded for
        url_template: from urls.py
        raw_dir: path to where the raw sat data will be saved
        retries: how many times will it try to download the data
        delay: how long will it wait to try the download again

    Returns
    -------
        List of paths to the downlaoded satellite files.
    """
    os.makedirs(raw_dir, exist_ok=True)
    raw_files = []

    for date_str in tqdm(dates_str, desc=f"Downloading {sat} data..."):
        url = f"{url_template}{date_str}.nc"
        filename = os.path.basename(url)
        raw_path = os.path.join(raw_dir, filename)

        if not os.path.exists(raw_path):
            for attempt in range(retries):
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(raw_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    _logger.info(f"Downloaded {filename}")
                    break
                except requests.RequestException as e:
                    _logger.warning(f"Attempt {attempt+1}/{retries} \
                                    failed for {url}: {e}")
                    time.sleep(delay)
            else:
                _logger.error(f"Failed to download {filename} after \
                              {retries} attempts.")
        else:
            _logger.info(f"File already exists: {filename}")
        raw_files.append(raw_path)

    return raw_files

def crop_by_box(dataset: xr.Dataset,
              lat_min: float,
              lat_max: float,
              lon_min: float,
              lon_max: float) -> xr.Dataset:
    """
    Crops xarray data based on lats and lons

    Parameters
    ----------
        lat_min: float/int of mininum latitude
        lat_max: float/int of maximum latitude
        lon_min: float/int of mininum longitude
        lon_max: float/int of maximum latitude
    Returns
    -------
        xarray object of the cropped data

    Notes
    -----
        Satellite data uses the -180 to 180 standard
        If you want to cross the meridian, then pass a lon_min > lon_max
        
    """
    # Check if latitude and longitude coordinates are in the dataset
    if 'lat' not in dataset or 'lon' not in dataset:
        raise ValueError("Dataset does not contain lat or lon dimensions")

    if lon_min < lon_max:
        lon_mask = (dataset.lon >= lon_min) & (dataset.lon <= lon_max)
    else:
        lon_mask = (dataset.lon >= lon_min) | (dataset.lon <= lon_max)

    lat_mask = (dataset.lat >= lat_min) & (dataset.lat <= lat_max)
    cropped = dataset.where(lat_mask & lon_mask, drop=True)

    return cropped

def crop_by_shape():
    """
    To be implemented
    """
    pass

def crop_sat_data(file_paths: List[str],
                  cropped_dir: str,
                  lat_min: float,
                  lat_max: float,
                  lon_min: float,
                  lon_max: float) -> List[xr.Dataset]:
    """
    Handles the satellite data cropping
    Crops and saves all the satellite data

    Parameters
    ----------
        lat_min: float/int of mininum latitude
        lat_max: float/int of maximum latitude
        lon_min: float/int of mininum longitude
        lon_max: float/int of maximum latitude

    Returns
    -------
        xarray object of the cropped data

    Notes
    -----
        Satellite data uses the -180 to 180 standard
        If you want to change the longitudes, use util.convert_longitude(sat_data.lon,mode=?)
    """
    os.makedirs(cropped_dir, exist_ok=True)
    cropped_datasets = []

    for file_path in tqdm(file_paths, desc="Cropping"):
        try:
            with xr.open_dataset(file_path) as ds:
                ds.load()
                cropped = crop_by_box(ds, lat_min, lat_max, lon_min, lon_max)
                if cropped.dims and all(size > 0 for size in cropped.sizes.values()):
                    out_path = os.path.join(cropped_dir,
                                            f"cropped_\
                                                {os.path.basename(file_path)}")
                    cropped.to_netcdf(out_path)
                    _logger.info(f"Saved {out_path}")
                    cropped_datasets.append(cropped)
                else:
                    _logger.warning(f"Skipping empty cropped dataset: \
                                    {file_path}")
        except Exception as e:
            _logger.warning(f"Failed to crop \
                            {file_path}: {type(e).__name__} - {e}")

    return cropped_datasets


def concat_sat_data(datasets: List[xr.Dataset],
                    output_path: str,
                    sat: str) -> Optional[xr.Dataset]:
    """
    Handles the satellite data concatenation
    Concat and saves all the satellite data on the datasets list

    Parameters
    ----------
        datasets: List xr satellite data to be concatenated
        output_path: path to where the concatenated data will be saved
        sat: name of the satellite

    Returns
    ----------
        xarray object of the concatenated data
    """
    if not datasets:
        _logger.warning("No datasets to concatenate.")
        return None

    try:
        concat_ds = xr.concat(datasets, dim='time')
        concat_ds = concat_ds.assign_coords(source=sat)
        concat_ds.to_netcdf(output_path)
        _logger.info(f"Concatenated dataset saved to {output_path}")
        return concat_ds
    except Exception as e:
        _logger.warning(f"Failed to concatenate datasets: \
                        {type(e).__name__} - {e}")
        return None


def get_per_sat(start_date: str,
         end_date: str,
         sat: str,
         output_dir: Union[str, os.PathLike],
         lat_min: Optional[float] = None,
         lat_max: Optional[float] = None,
         lon_min: Optional[float] = None,
         lon_max: Optional[float] = None,
         concat: bool = True,
         clean_raw: bool = False,
         clean_cropped: bool = False) -> Optional[xr.Dataset]:
    """
    Download, crop, and optionally concatenate satellite data.

    Parameters
    ----------
        start_date: Start date in 'YYYY-MM-DD'
        end_date: End date in 'YYYY-MM-DD'
        sat: Satellite key for URL_TEMPLATES
        output_dir: Directory to save files
        lat_min, lat_max, lon_min, lon_max: Optional cropping bounds
        concat: Save a single concatenated output
        clean_raw: Delete raw files after processing
        clean_cropped: Delete cropped files after processing

    Returns
    ----------
        xarray.Dataset if concatenated, otherwise None
    """
    try:
        url_template = URL_TEMPLATES[sat]
    except KeyError:
        raise ValueError(f"Unknown satellite key: {sat}")

    output_dir = os.path.join(output_dir, sat)
    raw_dir = os.path.join(output_dir, "raw")
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    cropping_enabled = None not in (lat_min, lat_max, lon_min, lon_max)
    dates_str = generate_daily_dates(start_date, end_date)

    raw_files = download_sat_data(dates_str, url_template, raw_dir, sat)

    datasets_to_concat = []
    if cropping_enabled:
        datasets_to_concat = crop_sat_data(raw_files,
                                           cropped_dir,
                                           lat_min,
                                           lat_max,
                                           lon_min,
                                           lon_max)
    else:
        for file_path in raw_files:
            try:
                with xr.open_dataset(file_path) as ds:
                    ds.load()
                    datasets_to_concat.append(ds)
            except Exception as e:
                _logger.warning(f"Failed to load \
                                {file_path}: {type(e).__name__} - {e}")

    final_dataset = None
    if concat and datasets_to_concat:
        concat_filename = f"concat_{'cropped_' if cropping_enabled else ''}{sat}_{start_date}_{end_date}.nc"
        concat_path = os.path.join(output_dir, concat_filename)
        final_dataset = concat_sat_data(datasets_to_concat, concat_path, sat)

    if clean_raw and os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        _logger.info("Raw files removed.")

    if clean_cropped and os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
        _logger.info("Cropped files removed.")

    return final_dataset

def get_multi_sat(start_date: str,
         end_date: str,
         sat_list: List,
         output_dir: Union[str, os.PathLike],
         lat_min: Optional[float] = None,
         lat_max: Optional[float] = None,
         lon_min: Optional[float] = None,
         lon_max: Optional[float] = None,
         concat: bool = True,
         clean_raw: bool = True,
         clean_cropped: bool = True) -> Optional[xr.Dataset]:
    """
    Run download and processing for multiple satellites.

    Parameters
    ----------
        start_date: Start date in 'YYYY-MM-DD'
        end_date: End date in 'YYYY-MM-DD'
        sat: Satellite key for URL_TEMPLATES
        output_dir: Directory to save files
        lat_min, lat_max, lon_min, lon_max: Optional cropping bounds
        concat: Save a single concatenated output
        clean_raw: Delete raw files after processing

    Returns
    ----------
        xarray.Dataset if concatenated, otherwise None
    """

    all_sat = []
    for sat in sat_list:
        concat_ds = get_per_sat(start_date,
                            end_date,
                            sat,
                            output_dir,
                            lat_min,
                            lat_max,
                            lon_min,
                            lon_max,
                            concat=concat,
                            clean_raw=clean_raw,
                            clean_cropped=clean_cropped)
        if concat_ds is not None:
            all_sat.append(concat_ds)

    if all_sat:
        try:
            multisat_filename = f"multisat_{'cropped_' if  lat_min is not None else ''}{start_date}_{end_date}.nc"
            multisat_path = os.path.join(output_dir, multisat_filename)
            all_sat_ds = xr.concat(all_sat, dim='time')
            all_sat_ds.to_netcdf(multisat_path)
            _logger.info(f"Concatenated multisat dataset saved to \
                         {multisat_path}")
            return all_sat_ds
        except Exception as e:
            _logger.warning(f"Failed to concatenate all satellite datasets: \
                            {type(e).__name__} - {e}")
    else:
        _logger.warning("No satellite datasets were successfully processed.")

    return None
