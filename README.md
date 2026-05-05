# OCSTrack: Ocean-Model-Satellite Collocation Tools

[![CI/CD](https://github.com/noaa-ocs-modeling/OCSTrack/actions/workflows/lint.yml/badge.svg)](https://github.com/noaa-ocs-modeling/OCSTrack/actions/workflows/lint.yml)
[![PyPI version](https://badge.fury.io/py/ocstrack.svg)](https://badge.fury.io/py/ocstrack)
[![codecov](https://codecov.io/gh/noaa-ocs-modeling/OCSTrack/graph/badge.svg?token=YOUR_CODECOV_TOKEN_IF_PRIVATE)](https://codecov.io/gh/noaa-ocs-modeling/OCSTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OCSTrack** is an object-oriented Python package for the along-track collocation of satellite (2D) and Argo Float (3D) data with ocean circulation and wave model outputs. It simplifies the process of aligning diverse datasets, making it easier to compare and validate model simulations against observational data.

---

## Key Features

- **Automated Data Fetching**: Downloads satellite altimetry and Argo float data from public repositories.
- **Model Support**: Natively handles outputs from SCHISM, ADCIRC, and SWAN models.
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

## Quick Start

Here is a minimal example of how to collocate satellite altimetry data with a SCHISM model run.

```python
from ocstrack import Collocate
from ocstrack.Model import SCHISM
from ocstrack.Observation import Satellite

# 1. Define time range and region of interest
start_date = '2021-05-01'
end_date = '2021-05-05'
bbox = [-76, 34, -72, 38]  # [lon_min, lat_min, lon_max, lat_max]

# 2. Initialize the Model object
# This assumes a SCHISM run directory with standard outputs
schism_run_dir = '/path/to/your/schism/run/'
model = SCHISM(schism_run_dir, start_date=start_date, end_date=end_date)

# 3. Initialize the Observation object
sat_name = 'sentinel-3a' # Example satellite
observation = Satellite(sat_name, start_date=start_date, end_date=end_date, bbox=bbox)

# 4. Create a Collocation object and run the analysis
collocator = Collocate(model, observation)
collocated_dataset = collocator.run(output_path='collocated_data.nc')

print("Collocation complete!")
print(collocated_dataset)
```

## Documentation

For more detailed examples and the full API reference, please see our documentation website (coming soon).

**Note on Badges**: The PyPI version badge is active. The CI/CD badge will become active once you commit the `.github/workflows/lint.yml` file and push it to your `main` branch.

## Contributing

We welcome contributions! If you have ideas for new features, find a bug, or would like to improve the documentation, please open an issue or submit a pull request.

## License

This project is licensed under the terms of the MIT license. See the `LICENSE.txt` file for details.

---

#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
