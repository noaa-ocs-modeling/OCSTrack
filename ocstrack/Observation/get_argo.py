""" Download and pre-process Argo Float data """

import os
import requests
import logging
import shutil
import time
import xarray as xr
import re
import numpy as np

from urllib.parse import urljoin
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Union, Optional, List
from tqdm import tqdm

# Assuming urls.py is in the same module
from .urls import ARGO_BASE_URL


_logger = logging.getLogger(__name__)

def generate_monthly_dates(start_date_str: str,
                           end_date_str: str) -> List[tuple[str, str]]:
    """
    Generates a list of (year, month) tuples between start and end dates.

    Args:
        start_date_str: String with dates as 'YYYY-MM-DD'
        end_date_str: String with dates as 'YYYY-MM-DD'

    Returns:
        List of tuples, e.g., [('2019', '08'), ('2019', '09')]
    """

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(day=1)
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    months = []
    current_date = start_date
    while current_date <= end_date:
        months.append((current_date.strftime('%Y'), current_date.strftime('%m')))
        current_date += relativedelta(months=1)

    return list(sorted(set(months)))

def _download_file(url: str, target_path: str) -> bool:
    """
    Helper to download a single file with requests.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    target_path : str
        The local path to save the file.

    Returns
    -------
    bool
        True if download was successful, False otherwise.
    """

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        _logger.info(f"Successfully downloaded {os.path.basename(target_path)}")
        return True
    except requests.RequestException as e:
        _logger.warning(f"Failed to download {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

def download_argo_data(year: str,
                       month: str,
                       region: str,
                       base_url: str,
                       raw_dir: str,
                       start_date: str,
                       end_date: str
                       ) -> List[str]:
    """
    Downloads Argo data for a given year/month using requests + re.
    This function recursively finds and downloads all .nc files from the
    Ifremer FTP-like HTML directory.

    Parameters
    ----------
    year : str
        Year string (e.g., "2019")
    month : str
        Month string (e.g., "08")
    region : str
        Geo region (e.g., "pacific_ocean")
    base_url : str
        Base URL for Argo (from urls.py)
    raw_dir : str
        Path to where the raw data will be saved
    start_date : str
        ISO 8601 string for start date.
    end_date : str
        ISO 8601 string for end date.

    Returns
    -------
    List[str]
        List of paths to the downloaded .nc files.
    """

    target_dir = os.path.join(raw_dir, year, month)
    os.makedirs(target_dir, exist_ok=True)

    start_url = f"{base_url}/{region}/{year}/{month}/"

    # Create datetime objects for comparison
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    downloaded_files = []
    visited_urls = set()
    urls_to_scrape = [start_url]

    link_finder = re.compile(r'href="([^"]+)"')

    _logger.info(f"Scanning {start_url} for files between {start_date} and {end_date}...")

    while urls_to_scrape:
        current_url = urls_to_scrape.pop(0)
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
            response.raise_for_status()
        except requests.RequestException as e:
            _logger.warning(f"Failed to access URL {current_url}: {e}")
            continue

        links = link_finder.findall(response.text)

        for href in links:
            if not href or href.startswith('?') or href.startswith('../'):
                continue

            absolute_url = urljoin(current_url, href)

            if href.endswith('.nc'):
                filename = os.path.basename(href)

                try:
                    # Parse date from filename, e.g., "20190829"
                    date_str = filename.split('_')[0]
                    file_date = datetime.strptime(date_str, '%Y%m%d')

                    # Check if the file's date is in our range
                    if not start_dt <= file_date <= end_dt:
                        _logger.info(f"Skipping download (out of date range): {filename}")
                        continue # Skip this file
                except Exception:
                    _logger.warning(f"Could not parse date from {filename}. Skipping.")
                    continue

                target_path = os.path.join(target_dir, filename)

                if not os.path.exists(target_path):
                    if _download_file(absolute_url, target_path):
                        downloaded_files.append(target_path)
                else:
                    _logger.info(f"File already exists: {filename}")
                    if target_path not in downloaded_files:
                        downloaded_files.append(target_path)

            elif href.endswith('/'):
                if absolute_url not in visited_urls and absolute_url.startswith(start_url):
                    urls_to_scrape.append(absolute_url)

    _logger.info(f"Downloaded {len(downloaded_files)} files for {year}-{month}")
    return downloaded_files


def crop_by_box_argo(dataset: xr.Dataset,
                     lat_min: float,
                     lat_max: float,
                     lon_min: float,
                     lon_max: float) -> xr.Dataset:
    """
    Crops xarray data based on lats and lons (Argo variable names).

    Parameters
    ----------
    dataset : xr.Dataset
        The Argo dataset to crop.
    lat_min : float
        Mininum latitude.
    lat_max : float
        Maximum latitude.
    lon_min : float
        Mininum longitude.
    lon_max : float
        Maximum longitude.

    Returns
    -------
    xr.Dataset
        xarray object of the cropped data.

    Notes
    -----
    Argo data uses the -180 to 180 longitude standard.
    If you want to cross the meridian, then pass a lon_min > lon_max.
    """
    if 'LATITUDE' not in dataset or 'LONGITUDE' not in dataset:
        raise ValueError("Dataset does not contain LATITUDE or LONGITUDE dimensions")

    if lon_min < lon_max:
        lon_mask = (dataset.LONGITUDE >= lon_min) & (dataset.LONGITUDE <= lon_max)
    else: # Handle dateline crossing
        lon_mask = (dataset.LONGITUDE >= lon_min) | (dataset.LONGITUDE <= lon_max)

    lat_mask = (dataset.LATITUDE >= lat_min) & (dataset.LATITUDE <= lat_max)

    # Combine the masks
    mask = lat_mask & lon_mask

    # Use .sel() to explicitly select along the N_PROF dimension
    cropped = dataset.sel(N_PROF=mask)

    return cropped


def crop_argo_data(file_paths: List[str],
                   cropped_dir: str,
                   lat_min: float,
                   lat_max: float,
                   lon_min: float,
                   lon_max: float,
                   start_date: str,
                   end_date: str
                   ) -> None:
    """
    Handles the Argo data loading, time filtering, and spatial cropping.

    This function uses a robust loading strategy to avoid
    decoding errors with metadata variables.

    Parameters
    ----------
    file_paths : List[str]
        List of raw .nc files to process.
    cropped_dir : str
        Directory to save the new cropped .nc files.
    lat_min : float
        Mininum latitude.
    lat_max : float
        Maximum latitude.
    lon_min : float
        Mininum longitude.
    lon_max : float
        Maximum longitude.
    start_date : str
        ISO 8601 string for start date.
    end_date : str
        ISO 8601 string for end date.
    """

    os.makedirs(cropped_dir, exist_ok=True)

    DROP_VARS = [
        'DATA_TYPE', 'FORMAT_VERSION', 'HANDBOOK_VERSION', 'REFERENCE_DATE_TIME',
        'DATE_CREATION', 'DATE_UPDATE', 'PLATFORM_NUMBER', 'PROJECT_NAME',
        'PI_NAME', 'STATION_PARAMETERS', 'DIRECTION', 'DATA_CENTRE',
        'DC_REFERENCE', 'DATA_STATE_INDICATOR', 'DATA_MODE', 'PLATFORM_TYPE',
        'FLOAT_SERIAL_NO', 'FIRMWARE_VERSION', 'WMO_INST_TYPE', 'JULD_QC',
        'POSITION_QC', 'POSITIONING_SYSTEM', 'PROFILE_PRES_QC',
        'PROFILE_TEMP_QC', 'PROFILE_PSAL_QC', 'VERTICAL_SAMPLING_SCHEME',
        'PRES_QC', 'PRES_ADJUSTED_QC', 'TEMP_QC', 'TEMP_ADJUSTED_QC',
        'PSAL_QC', 'PSAL_ADJUSTED_QC', 'PARAMETER', 'SCIENTIFIC_CALIB_EQUATION',
        'SCIENTIFIC_CALIB_COEFFICIENT', 'SCIENTIFIC_CALIB_COMMENT',
        'SCIENTIFIC_CALIB_DATE', 'HISTORY_INSTITUTION', 'HISTORY_STEP',
        'HISTORY_SOFTWARE', 'HISTORY_SOFTWARE_RELEASE', 'HISTORY_REFERENCE',
        'HISTORY_DATE', 'HISTORY_ACTION', 'HISTORY_PARAMETER', 'HISTORY_QCTEST',
        'HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_PREVIOUS_VALUE',
        'HISTORY_DATE', 'HISTORY_ACTION', 'HISTORY_PARAMETER','HISTORY_QCTEST',
        'HISTORY_INSTITUTION','N_HISTORY','CONFIG_MISSION_NUMBER','PRES_ADJUSTED_ERROR',
        'TEMP_ADJUSTED_ERROR','PSAL_ADJUSTED_ERROR','CYCLE_NUMBER'
    ]

    start_dt = np.datetime64(start_date)
    end_dt = np.datetime64(end_date)

    for file_path in tqdm(file_paths, desc="Cropping Argo data"):
        try:
            with xr.open_dataset(
                file_path,
                engine="netcdf4",
                decode_cf=False,
                drop_variables=DROP_VARS
            ) as ds_raw:
                ds = xr.decode_cf(ds_raw, decode_coords=False)

                # Time filtering
                if not np.issubdtype(ds['JULD'].dtype, np.datetime64):
                    ds['JULD'] = xr.decode_cf(ds).JULD

                time_mask = (ds.JULD >= start_dt) & (ds.JULD <= end_dt)
                ds_time_filtered = ds.sel(N_PROF=time_mask)

                # Pass the time-filtered dataset to the spatial crop
                cropped = crop_by_box_argo(ds_time_filtered, lat_min, lat_max, lon_min, lon_max)
                cropped.load()

            # Use .sizes to avoid FutureWarning
            if 'N_PROF' in cropped.sizes and cropped.sizes['N_PROF'] > 0:
                out_path = os.path.join(cropped_dir,
                                        f"cropped_{os.path.basename(file_path)}")
                cropped.to_netcdf(out_path)
                _logger.info(f"Saved {out_path}")
            else:
                _logger.warning(f"Skipping empty cropped dataset: {file_path}")

        except Exception as e:
            _logger.warning(f"Failed to process {file_path}: {type(e).__name__} - {e}")

def clean_argo_data(file_paths: List[str],
                    clean_dir: str,
                    start_date: str,
                    end_date: str
                    ) -> None:
    """
    Loads, time-filters, and re-saves Argo data to the clean_dir.
    This is used when cropping is disabled.

    Parameters
    ----------
    file_paths : List[str]
        List of raw .nc files to process.
    clean_dir : str
        Directory to save the new cleaned .nc files.
    start_date : str
        ISO 8601 string for start date.
    end_date : str
        ISO 8601 string for end date.
    """

    os.makedirs(clean_dir, exist_ok=True)

    DROP_VARS = [
        'DATA_TYPE', 'FORMAT_VERSION', 'HANDBOOK_VERSION', 'REFERENCE_DATE_TIME',
        'DATE_CREATION', 'DATE_UPDATE', 'PLATFORM_NUMBER', 'PROJECT_NAME',
        'PI_NAME', 'STATION_PARAMETERS', 'DIRECTION', 'DATA_CENTRE',
        'DC_REFERENCE', 'DATA_STATE_INDICATOR', 'DATA_MODE', 'PLATFORM_TYPE',
        'FLOAT_SERIAL_NO', 'FIRMWARE_VERSION', 'WMO_INST_TYPE', 'JULD_QC',
        'POSITION_QC', 'POSITIONING_SYSTEM', 'PROFILE_PRES_QC',
        'PROFILE_TEMP_QC', 'PROFILE_PSAL_QC', 'VERTICAL_SAMPLING_SCHEME',
        'PRES_QC', 'PRES_ADJUSTED_QC', 'TEMP_QC', 'TEMP_ADJUSTED_QC',
        'PSAL_QC', 'PSAL_ADJUSTED_QC', 'PARAMETER', 'SCIENTIFIC_CALIB_EQUATION',
        'SCIENTIFIC_CALIB_COEFFICIENT', 'SCIENTIFIC_CALIB_COMMENT',
        'SCIENTIFIC_CALIB_DATE', 'HISTORY_INSTITUTION', 'HISTORY_STEP',
        'HISTORY_SOFTWARE', 'HISTORY_SOFTWARE_RELEASE', 'HISTORY_REFERENCE',
        'HISTORY_DATE', 'HISTORY_ACTION', 'HISTORY_PARAMETER', 'HISTORY_QCTEST',
        'HISTORY_START_PRES', 'HISTORY_STOP_PRES', 'HISTORY_PREVIOUS_VALUE',
        'HISTORY_DATE', 'HISTORY_ACTION', 'HISTORY_PARAMETER','HISTORY_QCTEST',
        'HISTORY_INSTITUTION','N_HISTORY','CONFIG_MISSION_NUMBER','PRES_ADJUSTED_ERROR',
        'TEMP_ADJUSTED_ERROR','PSAL_ADJUSTED_ERROR','CYCLE_NUMBER'
    ]

    start_dt = np.datetime64(start_date)
    end_dt = np.datetime64(end_date)

    for file_path in tqdm(file_paths, desc="Cleaning Argo data"):
        try:
            with xr.open_dataset(
                file_path,
                engine="netcdf4",
                decode_cf=False,
                drop_variables=DROP_VARS
            ) as ds_raw:
                ds = xr.decode_cf(ds_raw, decode_coords=False)
                ds.load()

            # Time filtering step
            if not np.issubdtype(ds['JULD'].dtype, np.datetime64):
                ds['JULD'] = xr.decode_cf(ds).JULD

            time_mask = (ds.JULD >= start_dt) & (ds.JULD <= end_dt)
            ds_time_filtered = ds.sel(N_PROF=time_mask)

            # Save the time-filtered (but not spatially cropped) data
            if 'N_PROF' in ds_time_filtered.sizes and ds_time_filtered.sizes['N_PROF'] > 0:
                out_path = os.path.join(clean_dir, os.path.basename(file_path))
                ds_time_filtered.to_netcdf(out_path)
                _logger.info(f"Saved cleaned file: {out_path}")
            else:
                _logger.warning(f"Skipping empty time-filtered dataset: {file_path}")

        except Exception as e:
            _logger.warning(f"Failed to clean {file_path}: {type(e).__name__} - {e}")


def get_argo(start_date: str,
             end_date: str,
             region: str,
             output_dir: Union[str, os.PathLike],
             lat_min: Optional[float] = None,
             lat_max: Optional[float] = None,
             lon_min: Optional[float] = None,
             lon_max: Optional[float] = None,
             clean_raw: bool = False) -> Optional[str]:
    """
    Download, clean, and optionally crop Argo data for a given region
    within a specific date range.

    This is the main entry-point function. It will download all raw data
    and create a 'processed' directory containing cleaned, time-filtered,
    and (optionally) spatially-cropped individual .nc files.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD'. Data from this date will be INCLUDED.
    end_date : str
        End date in 'YYYY-MM-DD'. Data from this date will be INCLUDED.
    region : str
        Argo geo region (e.g., 'pacific_ocean', 'atlantic_ocean')
    output_dir : Union[str, os.PathLike]
        Base directory to save files. A 'raw' and 'processed' folder
        will be created inside a sub-directory named after the region.
    lat_min, lat_max, lon_min, lon_max : Optional[float]
        Optional cropping bounds
    clean_raw : bool
        Delete raw files after processing is complete

    Returns
    -------
    Optional[str]
        The path to the 'processed' directory containing the final
        .nc files, or None if the process failed.
    """

    output_dir = os.path.join(output_dir, region)
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    cropping_enabled = None not in (lat_min, lat_max, lon_min, lon_max)

    months_to_download = generate_monthly_dates(start_date, end_date)

    all_raw_files = []
    for year, month in tqdm(months_to_download, desc=f"Downloading {region} data"):
        # Pass dates down to the downloader
        raw_files = download_argo_data(year,
                                       month,
                                       region,
                                       ARGO_BASE_URL,
                                       raw_dir,
                                       start_date,
                                       end_date
                                       )
        all_raw_files.extend(raw_files)

    if not all_raw_files:
        _logger.warning(f"No raw files found for {region}. Exiting.")
        return None

    if cropping_enabled:
        _logger.info("Cropping enabled. Time-filtering and cropping files...")
        crop_argo_data(all_raw_files,
                       processed_dir,
                       lat_min,
                       lat_max,
                       lon_min,
                       lon_max,
                       start_date,
                       end_date
                       )
    else:
        _logger.info("Cropping disabled. Time-filtering all raw files...")
        clean_argo_data(all_raw_files,
                        processed_dir,
                        start_date,
                        end_date
                        )

    if clean_raw and os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        _logger.info("Raw files removed.")

    _logger.info(f"Processing complete. Output directory: {processed_dir}")
    return processed_dir
