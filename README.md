# OCSTrack: Ocean-Model-Data Collocation Tools

[![CI](https://github.com/noaa-ocs-modeling/OCSTrack/actions/workflows/ci.yml/badge.svg)](https://github.com/noaa-ocs-modeling/OCSTrack/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/ocstrack.svg)](https://badge.fury.io/py/ocstrack)
[![codecov](https://codecov.io/gh/noaa-ocs-modeling/OCSTrack/graph/badge.svg?token=YOUR_CODECOV_TOKEN_IF_PRIVATE)](https://codecov.io/gh/noaa-ocs-modeling/OCSTrack)
[![Pylint Score](https://noaa-ocs-modeling.github.io/OCSTrack/pylint.svg?raw=1)](https://github.com/noaa-ocs-modeling/OCSTrack/actions/workflows/ci.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

**OCSTrack** is an object-oriented Python package for the along-track collocation of satellite (2D) and Argo Float (3D) data with ocean circulation and wave model outputs. It simplifies the process of aligning diverse datasets, making it easier to compare and validate model simulations against observational data.

---

## Key Features

- **Automated Data Fetching**: Downloads satellite altimetry data from two sources — CoastWatch (NOAA STAR) and ESA CCI Sea State (IFREMER) — and Argo float data from IFREMER.
- **Model Support**: Natively handles outputs from SCHISM (and WWM), ADCIRC, SWAN, and WW3 models.
- **Flexible Collocation**: Performs temporal and spatial collocation for both 2D surface tracks and 3D profiles.
- **Efficient & Scalable**: Uses `xarray` and `dask` for efficient, out-of-core computations on large datasets.
- **Customizable**: Object-oriented design makes it easy to extend support for new models or observational data types.

## Installation

You can install OCSTrack directly from PyPI:

```bash
pip install ocstrack
```

To install the latest development version directly from this repository:

```bash
pip install git+https://github.com/noaa-ocs-modeling/OCSTrack.git
```

### Optional dependencies

OCSTrack has two sets of optional extras:

| Extra | Packages installed | When you need it |
|---|---|---|
| `gsw` | `gsw` | Accurate pressure-to-depth conversion for Argo 3D collocation. Without it, a simple linear approximation (`dbar × -1.0197`) is used instead. |
| `geo` | `shapely`, `geopandas` | Required only if you use `crop_by_shape()` to mask satellite data with a polygon or shapefile. |
| `all` | all of the above | Install everything at once. |

```bash
# Accurate depth conversion for Argo data
pip install "ocstrack[gsw]"

# Shapefile/polygon cropping support
pip install "ocstrack[geo]"

# Everything
pip install "ocstrack[all]"
```

## Profiling Float Data Sources

OCSTrack supports Argo Float profile data from the IFREMER GDAC. Argo data is used for 3D collocation against ocean circulation model outputs (SCHISM, ROMS), comparing model temperature and salinity profiles against observed vertical profiles. No credentials are required.

### Argo Floats (IFREMER GDAC)

Profile data is downloaded by ocean region and date range. The `get_argo` function scrapes the IFREMER GDAC HTTP server, applies time filtering and optional spatial cropping, and saves individual cleaned `.nc` files to a `processed/` directory. The `ArgoData` class then loads and concatenates all processed files into a single dataset, padding profiles to a uniform number of vertical levels.

```python
from ocstrack.Observation.get_argo import get_argo
from ocstrack.Observation.argofloat import ArgoData

# Download and pre-process Argo data for a region, cropped to a shapefile
# shape also accepts a shapely geometry or a GeoJSON dict; requires ocstrack[geo]
get_argo(
    start_date="2019-08-29",
    end_date="2019-10-05",
    region="pacific_ocean",
    output_dir="./argo_data/",
    shape="/path/to/domain.shp",
)

# Load the processed profiles
argo_data = ArgoData("./argo_data/pacific_ocean/processed/")
print(f"Profiles loaded : {argo_data.ds.sizes['JULD']}")
print(f"Vertical levels : {argo_data.ds.sizes['N_LEVELS']}")
print(f"Time range      : {argo_data.time.min()} to {argo_data.time.max()}")
```

**Available regions:** `atlantic_ocean`, `pacific_ocean`, `indian_ocean`, `mediterranean_sea`, `black_sea`, `caspian_sea`, `red_sea`, `baltic_sea`, and others as listed on the [IFREMER GDAC](https://data-argo.ifremer.fr/geo).

**Key variables:** `PRES` / `PRES_ADJUSTED` (pressure, dbar), `TEMP` / `TEMP_ADJUSTED` (temperature, °C), `PSAL` / `PSAL_ADJUSTED` (salinity, PSU), `LATITUDE`, `LONGITUDE`, `JULD` (time). Adjusted variables are preferred automatically when available.

**Depth conversion:** If the optional `gsw` package is installed (`pip install "ocstrack[gsw]"`), pressure is converted to depth using the Gibbs SeaWater toolbox for full accuracy. Otherwise a linear approximation (`depth ≈ pressure × -1.0197`) is used.

---

## Satellite Data Sources

OCSTrack supports two satellite altimetry data sources. Both are handled by the same `SatelliteData` class, which auto-detects the format.

### CoastWatch (NOAA STAR)

Daily merged NetCDF files from the NOAA STAR CoastWatch program. No credentials required.

```python
from ocstrack.Observation.get_sat_coastwatch import get_multi_sat_coastwatch

get_multi_sat_coastwatch(
    start_date="2023-01-16", end_date="2023-01-31",
    sat_list=['sentinel3a', 'sentinel3b', 'jason3'],
    output_dir="./sat_data/",
    lat_min=18, lat_max=31, lon_min=-98, lon_max=-80,
)
```

**Limitations:** Old data may be deleted without notice; SWH is capped at 8 m.

### ESA CCI Sea State v5 (IFREMER)

Along-track per-pass files from the ESA Climate Change Initiative Sea State project, hosted on the IFREMER FTP server. Suitable for extreme wave analysis (no SWH cap).

**Credentials required.** Register at [https://eftp.ifremer.fr](https://eftp.ifremer.fr) to obtain a free username and password. Store them as environment variables:

```bash
export CCI_FTP_USER="your_username"
export CCI_FTP_PASS="your_password"
```

```python
import os
from ocstrack.Observation.get_sat_cci import get_multi_sat_cci

get_multi_sat_cci(
    start_date="2023-01-16", end_date="2023-01-31",
    sat_list=['jason-3', 'sentinel-3a', 'sentinel-3b'],
    output_dir="./cci_sat_data/",
    ftp_user=os.environ["CCI_FTP_USER"],
    ftp_pass=os.environ["CCI_FTP_PASS"],
    lat_min=18, lat_max=31, lon_min=-98, lon_max=-80,
)
```

**Advantages over CoastWatch:** Stable long-term archive; no SWH cap; includes `swh_with_8m_offset_correction`, `swh_quality_level`, `swh_uncertainty`, `bathymetry`, and `distance_to_coast`.

## Quick Start

Here is a minimal example of how to collocate satellite altimetry data with a WW3 model run.

```python
import numpy as np
from ocstrack.Model.model import WW3
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Collocation.collocate import Collocate

# Load satellite data — SatelliteData auto-detects CoastWatch or CCI format
sat_data = SatelliteData("/path/to/merged_satellite_data.nc")
print(sat_data.data_source)  # 'coastwatch' or 'cci'

# Load WW3 model
model_run = WW3(
    rundir="/path/to/ww3/run/",
    model_dict={"var": "hs"},
    start_date=np.datetime64("2023-01-16"),
    end_date=np.datetime64("2023-01-31"),
)

# Collocate
coll = Collocate(model_run=model_run, observation=sat_data, n_nearest=3)
ds_coll = coll.run(output_path="collocated.nc")
print("Collocation complete!")
```

## Documentation

For more detailed examples and the full API reference, please see our documentation website:

[**https://noaa-ocs-modeling.github.io/OCSTrack/**](https://noaa-ocs-modeling.github.io/OCSTrack/)

## Contributing

We welcome contributions! If you have ideas for new features, find a bug, or would like to improve the documentation, please open an issue or submit a pull request. Developers, please follow the guidelines [here](https://github.com/noaa-ocs-modeling/OCSTrack/blob/main/docs/contributing.md).

## License

This project is licensed under the terms of the CC0 1.0 Universal license. See the `LICENSE.txt` file for details.

## How to Cite

If you use OCSTrack for 3D collocation with Argo floats, please cite:

Cassalho, F., S. Mani, S. Moghimi, F. Ye, and Y. J. Zhang. "OCSMesh and an automated creek-to-ocean mesh generation workflow." *Ocean Modelling* 203 (2026): 102774. https://doi.org/10.1016/j.ocemod.2026.102774.

If you use OCSTrack for 2D collocation with satellite altimetry, please cite:

Cassalho, F., A. L. Kurapov, S. Moghimi, S. M. Durski, J. Y. Zhang, A. Abdolali, B. Khazaei, Y. Sun, F. Ye, E. Myers. "Tidal modulation of waves around the Aleutian Islands." *Journal of Geophysical Research: Oceans* 131 (2026): e2025JC023780. https://doi.org/10.1029/2025JC023780.

## Contributors

The Satellite Altimetry capabilities within the Observation Module as well as the WAVEWATCH III class in the Model Module were adapted from [Ali Abdolali](https://github.com/aliabdolali)'s [wave-tools](https://github.com/erdc/wave-tools) package developed under the [US Army Engineer Research and Development Center](https://github.com/erdc). Please cite:

Abdolali A., A. Roland, A. Van Der Westhuysen, J. Meixner, A. Chawla, T. Hesser,  J. M. Smith, and M. Dutour Sikiric, "Large-scale Hurricane Modeling Using Domain Decomposition Parallelization and Implicit Scheme Implemented in WAVEWATCH III Wave Model." *Coastal Engineering* 157 (2020): 103656. https://doi.org/10.1016/j.coastaleng.2020.10365.

---
  
#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

