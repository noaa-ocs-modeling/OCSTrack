"""Download and pre-process ESA CCI Sea State satellite altimetry data from IFREMER FTP.

Data source: ESA Climate Change Initiative (CCI) Sea State v5
FTP server:  eftp.ifremer.fr

Credentials are required. Register at https://eftp.ifremer.fr to obtain a
username and password. Pass them directly to the download functions via the
``ftp_user`` and ``ftp_pass`` arguments.

Example registration and credential format
------------------------------------------
1. Go to https://eftp.ifremer.fr and register for a free account.
2. You will receive a username (e.g. ``pe31b4c``) and a password.
3. Pass them as::

       get_per_sat_cci(..., ftp_user='pe31b4c', ftp_pass='your-password')

Notes
-----
- The pipeline mirrors the CoastWatch workflow: download per-pass files,
  crop spatially, then merge into a single NetCDF per satellite.
- Only a subset of variables is retained in the merged output (see
  ``CCI_KEEP_VARS`` in ``urls.py``). This keeps file sizes manageable and
  removes variables not relevant to wave height collocation.
- SAR data (Sentinel-1) contains 2D wave spectra, not along-track SWH, and
  is therefore not directly compatible with the collocation engine. It is
  supported by the downloader for completeness but is excluded from the
  default ``get_multi_sat_cci`` call.
"""

import ftplib
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import urllib.request

import xarray as xr
from tqdm import tqdm

from .get_sat_coastwatch import crop_by_box, crop_by_shape
from .urls import (
    CCI_ALTIMETERS,
    CCI_FTP_BASE_PATH,
    CCI_FTP_HOST,
    CCI_FTP_VERSION,
    CCI_KEEP_VARS,
    CCI_SARS,
    resolve_sat_key,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_credentials(ftp_user: Optional[str], ftp_pass: Optional[str]) -> None:
    """Raise a descriptive error if FTP credentials are missing."""
    if not ftp_user or not ftp_pass:
        raise ValueError(
            "FTP credentials are required to download ESA CCI Sea State data.\n\n"
            "How to obtain credentials:\n"
            "  1. Go to https://eftp.ifremer.fr and register for a free account.\n"
            "  2. You will receive a username and password by e-mail.\n\n"
            "How to use them:\n"
            "  Pass the credentials directly to the download function:\n\n"
            "      get_per_sat_cci(..., ftp_user='your_username', ftp_pass='your_password')\n"
            "      get_multi_sat_cci(..., ftp_user='your_username', ftp_pass='your_password')\n\n"
            "Security note:\n"
            "  Do NOT hard-code credentials in scripts. Instead, read them from\n"
            "  environment variables or a config file:\n\n"
            "      import os\n"
            "      ftp_user = os.environ['CCI_FTP_USER']\n"
            "      ftp_pass = os.environ['CCI_FTP_PASS']\n"
        )


def _generate_dates(start_date_str: str, end_date_str: str) -> List[str]:
    """Return a list of 'YYYYMMDD' strings between start and end dates (inclusive)."""
    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    return [
        (start + timedelta(days=i)).strftime('%Y%m%d')
        for i in range((end - start).days + 1)
    ]


def _list_ftp_directory(ftp: ftplib.FTP, path: str) -> List[str]:
    """Return directory listing for *path*, or empty list on error."""
    try:
        ftp.cwd(path)
        return ftp.nlst()
    except ftplib.error_perm:
        return []


def _download_file(file_url: str, local_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Download a single file. Returns (filename, error_message) on failure, else (None, None)."""
    filename = os.path.basename(local_path)
    if os.path.exists(local_path):
        return None, None
    try:
        urllib.request.urlretrieve(file_url, local_path)
        return None, None
    except Exception as exc:  # pylint: disable=broad-except
        return filename, str(exc)


def _subset_vars(ds: xr.Dataset, keep_vars: List[str]) -> xr.Dataset:
    """Drop variables not in *keep_vars*, keeping all that are present."""
    present = [v for v in keep_vars if v in ds]
    # Always keep coordinates
    coords = list(ds.coords)
    to_keep = list(set(present + coords))
    drop = [v for v in ds.data_vars if v not in to_keep]
    return ds.drop_vars(drop) if drop else ds


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------

def download_cci_data(
    start_date: str,
    end_date: str,
    sat_ftp_name: str,
    raw_dir: str,
    ftp_user: str,
    ftp_pass: str,
    sensor_type: str = "altimeter",
    max_parallel: int = 4,
    version: str = CCI_FTP_VERSION,
) -> List[str]:
    """
    Download ESA CCI Sea State per-pass NetCDF files from the IFREMER FTP server.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    sat_ftp_name : str
        The FTP directory name for the satellite (e.g. ``'jason-3'``).
        Use values from ``CCI_ALTIMETERS`` or ``CCI_SARS`` in ``urls.py``.
    raw_dir : str
        Local directory where raw files will be saved.
    ftp_user : str
        IFREMER FTP username. Register at https://eftp.ifremer.fr.
    ftp_pass : str
        IFREMER FTP password.
    sensor_type : str, optional
        ``'altimeter'`` (default) or ``'sar'``.
    max_parallel : int, optional
        Maximum simultaneous downloads (keep <= 5 to avoid server limits).
    version : str, optional
        CCI product version (default ``'5'``).

    Returns
    -------
    list of str
        Paths to all successfully downloaded (or already existing) local files.

    Raises
    ------
    ValueError
        If FTP credentials are not provided.
    """
    _check_credentials(ftp_user, ftp_pass)
    os.makedirs(raw_dir, exist_ok=True)

    valid_dates = set(_generate_dates(start_date, end_date))
    base_path = CCI_FTP_BASE_PATH.format(version=version)

    if sensor_type == "altimeter":
        year_base = f"{base_path}/altimeter/l2p-swh/{sat_ftp_name}"
    elif sensor_type == "sar":
        year_base = f"{base_path}/sar/{sat_ftp_name}/l2p"
    else:
        raise ValueError(f"Unknown sensor_type: '{sensor_type}'. Use 'altimeter' or 'sar'.")

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    all_local_files: List[str] = []

    for year in range(start_year, end_year + 1):
        year_path = f"{year_base}/{year}"

        try:
            ftp = ftplib.FTP(CCI_FTP_HOST)
            ftp.login(ftp_user, ftp_pass)
        except ftplib.all_errors as exc:
            _logger.error("FTP connection failed for %s (%s): %s", sat_ftp_name, year, exc)
            continue

        cycles = _list_ftp_directory(ftp, year_path)
        cycles = [c for c in cycles if re.match(r'^[0-9a-zA-Z_-]+$', c)]

        if not cycles:
            _logger.debug("No cycle directories found for %s (%s).", sat_ftp_name, year)
            ftp.quit()
            continue

        _logger.info("Scanning %s for %s — found %d cycle(s).", sat_ftp_name, year, len(cycles))

        files_to_download: List[Tuple[str, str]] = []

        for cycle in cycles:
            cycle_path = f"{year_path}/{cycle}"
            all_files = _list_ftp_directory(ftp, cycle_path)

            for filename in all_files:
                if not filename.endswith('.nc'):
                    continue
                match = re.search(r'([0-9]{8})', filename)
                if not match:
                    continue
                file_date = match.group(1)
                if file_date not in valid_dates:
                    continue

                local_path = os.path.join(raw_dir, filename)
                auth_url = (
                    f"ftp://{ftp_user}:{ftp_pass}@{CCI_FTP_HOST}{cycle_path}/{filename}"
                )
                files_to_download.append((auth_url, local_path))

        ftp.quit()

        if not files_to_download:
            continue

        _logger.info(
            "Found %d file(s) to download for %s (%s).",
            len(files_to_download), sat_ftp_name, year,
        )

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(_download_file, url, lp): os.path.basename(lp)
                for url, lp in files_to_download
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Downloading {sat_ftp_name} ({year})",
            ):
                err_file, err_msg = future.result()
                if err_file:
                    _logger.warning("Failed to download %s: %s", err_file, err_msg)
                else:
                    local_path = [lp for _, lp in files_to_download
                                  if os.path.basename(lp) == futures[future]]
                    if local_path:
                        all_local_files.append(local_path[0])

        # Collect any pre-existing files
        for _, lp in files_to_download:
            if os.path.exists(lp) and lp not in all_local_files:
                all_local_files.append(lp)

    return sorted(set(all_local_files))


# ---------------------------------------------------------------------------
# Crop and merge helpers
# ---------------------------------------------------------------------------

def crop_cci_data(
    file_paths: List[str],
    cropped_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    keep_vars: Optional[List[str]] = None,
) -> List[xr.Dataset]:
    """
    Crop a list of CCI per-pass NetCDF files to a bounding box and retain
    only the variables needed for collocation.

    Parameters
    ----------
    file_paths : list of str
        Paths to raw CCI NetCDF files.
    cropped_dir : str
        Directory to save cropped files.
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box for spatial cropping.
    keep_vars : list of str, optional
        Variables to retain. Defaults to ``CCI_KEEP_VARS``.

    Returns
    -------
    list of xr.Dataset
        Non-empty cropped datasets.
    """
    if keep_vars is None:
        keep_vars = CCI_KEEP_VARS

    os.makedirs(cropped_dir, exist_ok=True)
    cropped_datasets: List[xr.Dataset] = []

    for file_path in tqdm(file_paths, desc="Cropping CCI"):
        try:
            with xr.open_dataset(file_path) as ds:
                ds = ds.load()
                cropped = crop_by_box(ds, lat_min, lat_max, lon_min, lon_max)

                if not cropped.dims or not all(s > 0 for s in cropped.sizes.values()):
                    _logger.debug("Skipping empty crop: %s", file_path)
                    continue

                cropped = _subset_vars(cropped, keep_vars)
                out_path = os.path.join(cropped_dir, f"cropped_{os.path.basename(file_path)}")
                cropped.to_netcdf(out_path)
                _logger.info("Saved %s", out_path)
                cropped_datasets.append(cropped)

        except (OSError, ValueError) as exc:
            _logger.warning("Failed to crop %s: %s - %s", file_path, type(exc).__name__, exc)

    return cropped_datasets


def crop_cci_data_by_shape(
    file_paths: List[str],
    cropped_dir: str,
    shape,
    keep_vars: Optional[List[str]] = None,
) -> List[xr.Dataset]:
    """
    Crop a list of CCI per-pass NetCDF files to a polygon / shapefile.

    Parameters
    ----------
    file_paths : list of str
        Paths to raw CCI NetCDF files.
    cropped_dir : str
        Directory to save cropped files.
    shape : str, dict, or shapely geometry
        Bounding shape. See :func:`ocstrack.Observation.get_sat_coastwatch.crop_by_shape`
        for accepted formats.
    keep_vars : list of str, optional
        Variables to retain. Defaults to ``CCI_KEEP_VARS``.

    Returns
    -------
    list of xr.Dataset
        Non-empty cropped datasets.
    """
    if keep_vars is None:
        keep_vars = CCI_KEEP_VARS

    os.makedirs(cropped_dir, exist_ok=True)
    cropped_datasets: List[xr.Dataset] = []

    for file_path in tqdm(file_paths, desc="Cropping CCI by shape"):
        try:
            with xr.open_dataset(file_path) as ds:
                ds = ds.load()
                cropped = crop_by_shape(ds, shape)

                if not cropped.dims or not all(s > 0 for s in cropped.sizes.values()):
                    _logger.debug("Skipping empty crop: %s", file_path)
                    continue

                cropped = _subset_vars(cropped, keep_vars)
                out_path = os.path.join(cropped_dir, f"cropped_{os.path.basename(file_path)}")
                cropped.to_netcdf(out_path)
                _logger.info("Saved %s", out_path)
                cropped_datasets.append(cropped)

        except (OSError, ValueError) as exc:
            _logger.warning(
                "Failed to crop %s: %s - %s", file_path, type(exc).__name__, exc
            )

    return cropped_datasets


def concat_cci_data(
    datasets: List[xr.Dataset],
    output_path: str,
    sat: str,
) -> Optional[xr.Dataset]:
    """
    Concatenate a list of CCI datasets along the ``time`` dimension.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Cropped (or raw) CCI datasets for a single satellite.
    output_path : str
        Path where the merged NetCDF file will be saved.
    sat : str
        Satellite name used to set the ``source`` coordinate.

    Returns
    -------
    xr.Dataset or None
        The merged dataset, or ``None`` if concatenation fails.
    """
    if not datasets:
        _logger.warning("No CCI datasets to concatenate for %s.", sat)
        return None

    try:
        merged = xr.concat(datasets, dim='time')
        merged = merged.sortby('time')
        merged = merged.assign_coords(source=sat)
        merged.to_netcdf(output_path)
        _logger.info("Merged CCI dataset saved to %s", output_path)
        return merged
    except (ValueError, OSError) as exc:
        _logger.warning(
            "Failed to concatenate CCI datasets for %s: %s - %s",
            sat, type(exc).__name__, exc,
        )
        return None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def get_per_sat_cci(
    start_date: str,
    end_date: str,
    sat: str,
    output_dir: Union[str, os.PathLike],
    ftp_user: str,
    ftp_pass: str,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    shape=None,
    keep_vars: Optional[List[str]] = None,
    sensor_type: str = "altimeter",
    max_parallel: int = 4,
    concat: bool = True,
    clean_raw: bool = False,
    clean_cropped: bool = False,
    version: str = CCI_FTP_VERSION,
) -> Optional[xr.Dataset]:
    """
    Download, crop, and optionally merge ESA CCI Sea State data for one satellite.

    This mirrors the ``get_per_sat_coastwatch`` workflow for CoastWatch data.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    sat : str
        Satellite key. For altimeters use keys from ``CCI_ALTIMETERS``
        (e.g. ``'jason-3'``, ``'sentinel-3a'``). For SAR use keys from
        ``CCI_SARS`` (e.g. ``'sentinel-1a'``).
    output_dir : str or path-like
        Root directory for all output files.
    ftp_user : str
        IFREMER FTP username. Register at https://eftp.ifremer.fr.
    ftp_pass : str
        IFREMER FTP password.
    lat_min, lat_max, lon_min, lon_max : float, optional
        Bounding box for spatial cropping. If all four are provided, a
        box crop is performed. Mutually exclusive with ``shape``.
    shape : str, dict, or shapely geometry, optional
        Alternative polygon/shapefile for spatial cropping. Mutually
        exclusive with the bounding box arguments. Requires ``shapely``
        (and ``geopandas`` for shapefile paths).
    keep_vars : list of str, optional
        Variables to retain in merged output. Defaults to ``CCI_KEEP_VARS``.
    sensor_type : str, optional
        ``'altimeter'`` (default) or ``'sar'``.
    max_parallel : int, optional
        Maximum simultaneous FTP downloads (keep <= 5).
    concat : bool, optional
        If ``True`` (default), save a single merged NetCDF file.
    clean_raw : bool, optional
        Delete raw per-pass files after processing (default ``False``).
    clean_cropped : bool, optional
        Delete cropped files after merging (default ``False``).
    version : str, optional
        CCI product version (default ``'5'``).

    Returns
    -------
    xr.Dataset or None
        Merged dataset if ``concat=True``, otherwise ``None``.

    Raises
    ------
    ValueError
        If FTP credentials are missing or the satellite key is not recognised.
    """
    _check_credentials(ftp_user, ftp_pass)

    # Resolve FTP directory name (punctuation/case-insensitive lookup)
    sat_lookup: Dict[str, str] = {**CCI_ALTIMETERS, **CCI_SARS}
    sat = resolve_sat_key(sat, sat_lookup)
    sat_ftp_name = sat_lookup[sat]

    sat_dir = os.path.join(output_dir, sat)
    raw_dir = os.path.join(sat_dir, "raw")
    cropped_dir = os.path.join(sat_dir, "cropped")
    os.makedirs(sat_dir, exist_ok=True)

    box_crop = None not in (lat_min, lat_max, lon_min, lon_max)
    shape_crop = shape is not None

    if box_crop and shape_crop:
        raise ValueError(
            "Provide either a bounding box (lat_min/lat_max/lon_min/lon_max) "
            "or a shape, not both."
        )

    # --- Step 1: Download ---
    raw_files = download_cci_data(
        start_date=start_date,
        end_date=end_date,
        sat_ftp_name=sat_ftp_name,
        raw_dir=raw_dir,
        ftp_user=ftp_user,
        ftp_pass=ftp_pass,
        sensor_type=sensor_type,
        max_parallel=max_parallel,
        version=version,
    )

    if not raw_files:
        _logger.warning("No files downloaded for %s (%s – %s).", sat, start_date, end_date)
        return None

    # --- Step 2: Crop ---
    datasets_to_merge: List[xr.Dataset] = []

    if box_crop:
        datasets_to_merge = crop_cci_data(
            raw_files, cropped_dir, lat_min, lat_max, lon_min, lon_max, keep_vars
        )
    elif shape_crop:
        datasets_to_merge = crop_cci_data_by_shape(raw_files, cropped_dir, shape, keep_vars)
    else:
        # No cropping — just subset variables
        _keep = keep_vars if keep_vars is not None else CCI_KEEP_VARS
        for fp in raw_files:
            try:
                with xr.open_dataset(fp) as ds:
                    ds = ds.load()
                    datasets_to_merge.append(_subset_vars(ds, _keep))
            except (OSError, ValueError) as exc:
                _logger.warning("Failed to load %s: %s - %s", fp, type(exc).__name__, exc)

    # --- Step 3: Merge ---
    final_dataset = None
    if concat and datasets_to_merge:
        crop_tag = "cropped_" if (box_crop or shape_crop) else ""
        concat_filename = (
            f"concat_cci_{crop_tag}{sat}_{start_date}_{end_date}.nc"
        )
        concat_path = os.path.join(sat_dir, concat_filename)
        final_dataset = concat_cci_data(datasets_to_merge, concat_path, sat)

    # --- Cleanup ---
    if clean_raw and os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        _logger.info("Raw CCI files removed for %s.", sat)

    if clean_cropped and os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
        _logger.info("Cropped CCI files removed for %s.", sat)

    return final_dataset


def get_multi_sat_cci(
    start_date: str,
    end_date: str,
    sat_list: List[str],
    output_dir: Union[str, os.PathLike],
    ftp_user: str,
    ftp_pass: str,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    shape=None,
    keep_vars: Optional[List[str]] = None,
    sensor_type: str = "altimeter",
    max_parallel: int = 4,
    concat: bool = True,
    clean_raw: bool = True,
    clean_cropped: bool = True,
    version: str = CCI_FTP_VERSION,
) -> Optional[xr.Dataset]:
    """
    Download, crop, and merge ESA CCI Sea State data for multiple satellites.

    This mirrors the ``get_multi_sat_coastwatch`` workflow for CoastWatch data.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    sat_list : list of str
        List of satellite keys (see ``get_per_sat_cci`` for valid values).
    output_dir : str or path-like
        Root directory for all output files.
    ftp_user : str
        IFREMER FTP username. Register at https://eftp.ifremer.fr.
    ftp_pass : str
        IFREMER FTP password.
    lat_min, lat_max, lon_min, lon_max : float, optional
        Bounding box for spatial cropping.
    shape : str, dict, or shapely geometry, optional
        Alternative polygon/shapefile for spatial cropping.
    keep_vars : list of str, optional
        Variables to retain. Defaults to ``CCI_KEEP_VARS``.
    sensor_type : str, optional
        ``'altimeter'`` (default) or ``'sar'``.
    max_parallel : int, optional
        Maximum simultaneous FTP downloads (keep <= 5).
    concat : bool, optional
        If ``True`` (default), save a merged multi-satellite NetCDF file.
    clean_raw : bool, optional
        Delete raw per-pass files after processing (default ``True``).
    clean_cropped : bool, optional
        Delete per-satellite cropped files after merging (default ``True``).
    version : str, optional
        CCI product version (default ``'5'``).

    Returns
    -------
    xr.Dataset or None
        Merged multi-satellite dataset, or ``None`` if no data was processed.

    Raises
    ------
    ValueError
        If FTP credentials are missing.
    """
    _check_credentials(ftp_user, ftp_pass)

    all_datasets: List[xr.Dataset] = []

    for sat in sat_list:
        ds = get_per_sat_cci(
            start_date=start_date,
            end_date=end_date,
            sat=sat,
            output_dir=output_dir,
            ftp_user=ftp_user,
            ftp_pass=ftp_pass,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            shape=shape,
            keep_vars=keep_vars,
            sensor_type=sensor_type,
            max_parallel=max_parallel,
            concat=concat,
            clean_raw=clean_raw,
            clean_cropped=clean_cropped,
            version=version,
        )
        if ds is not None:
            all_datasets.append(ds)

    if not all_datasets:
        _logger.warning("No CCI satellite datasets were successfully processed.")
        return None

    try:
        crop_tag = "cropped_" if None not in (lat_min, lat_max, lon_min, lon_max) else ""
        multisat_filename = (
            f"multisat_cci_{crop_tag}{start_date}_{end_date}.nc"
        )
        multisat_path = os.path.join(output_dir, multisat_filename)
        merged = xr.concat(all_datasets, dim='time')
        merged = merged.sortby('time')
        merged.to_netcdf(multisat_path)
        _logger.info("Multi-satellite CCI dataset saved to %s", multisat_path)
        return merged
    except (ValueError, OSError) as exc:
        _logger.warning(
            "Failed to merge multi-satellite CCI datasets: %s - %s",
            type(exc).__name__, exc,
        )
        return None
