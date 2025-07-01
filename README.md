# OCSTrack
OCSTrack is an object-oriented Python package for the along-track collocation of satellite data with ocean circulation and wave model outputs.
It simplifies the process of aligning diverse datasets, making it easier to compare and analyze satellite observations against model simulations.

## Key Features
### Satellite Altimetry Data Support

Seamlessly integrates with NOAA [CoastalWatch](https://coastwatch.noaa.gov/cwn/products/along-track-significant-wave-height-wind-speed-and-sea-level-anomaly-multiple-altimeters.html) altimetry data, providing access to a wide range of missions:
 * Jason-2
 * Jason-3
 * Sentinel-3A
 * Sentinel-3B
 * Sentinel-6A
 * CryoSat-2
 * SARAL
 * SWOT

### Ocean Model Data Support
 Supports outputs from various ocean circulation and wave models:
 * [SCHISM+WWM](https://github.com/schism-dev/schism)
 * WaveWatch3 (to be implemented)
 * ADCIRC+SWAN (to be implemented)


## Installation

1. Create a Conda Environment:
```
conda create --name ocstrack python=3.11
conda activate ocstrack
```

2. Install the Package:
```
pip install "git+https://github.com/noaa-ocs-modeling/OCSTrack.git"
```

## Usage
Here's a typical workflow demonstrating how to use OCSTrack to download satellite data, load model outputs, and perform collocation.
```
import numpy as np
import xarray as xr
# Assuming ocstrack is installed and available in your environment
from ocstrack.Model.model import SCHISM
from ocstrack.Satellite.satellite import SatelliteData
from ocstrack.Satellite import get_sat
from ocstrack.Collocation.collocate import Collocate
from ocstrack.utils import convert_longitude


# 1. Download Satellite Data
#    Specify your desired date range, list of satellites, output directory, and geographical bounding box.
get_sat.get_multi_sat(start_date="2019-07-30",
                      end_date="2019-08-04",
                      sat_list=['sentinel3a','sentinel3b','jason2','jason3','cryosat2','saral'],
                      output_dir=r"Your/Path/Here/",
                      lat_min=49.109,
                      lat_max=66.304309,
                      lon_min=156.6854,
                      lon_max=-156.864,
                     )

# 2. Define File Paths
#    Set the paths for your downloaded satellite data, model run, and where you want to save the collocated output.
sat_path = r"Path/to/Satellite/Data/multisat_cropped_2019-07-30_2019-08-04.nc"
model_path = r"Path/to/Model/Run/R__/"
output_path =  r"Path/to/CollocatedFile/collocated.nc"
s_time,e_time = "2019-08-01", "2019-08-03"

# 3. Load Satellite Data
#    Initialize the SatelliteData object with your satellite data file.
sat_data = SatelliteData(sat_path)
#    It's crucial to ensure longitude conventions match between satellite and model data.
#    Use convert_longitude if needed (mode=1 for converting to 0-360 degrees).
sat_data.lon = convert_longitude(sat_data.lon, mode=1)

# 4. Load Model Data
#    Instantiate the SCHISM model object, specifying the run directory and model variable details.
model_run = SCHISM(
                    rundir=model_path,
                    model_dict={'var': 'sigWaveHeight',
                                'startswith': 'out2d_', # File name prefix for 2D outputs
                                'var_type': '2D',
                                'model': 'SCHISM'},
                    start_date=np.datetime64(s_time),
                    end_date=np.datetime64(e_time)
                  )

# 5. Perform Collocation
#    Create a Collocate object, providing the loaded model and satellite data.
coll = Collocate(
                 model_run=model_run,
                 satellite=sat_data,
                 dist_coast=dist_coast,
                 n_nearest=3,
                 # search_radius = 3000,
                 temporal_interp=True
                 )
ds_coll = coll.run(output_path=output_path) # Execute the collocation and save the results
```

## Contributing
We welcome contributions to OCSTrack! If you have ideas for improvements, new features, or find a bug, please don't hesitate to open an issue or submit a pull request on our GitHub repository. Your input helps make OCSTrack better for everyone.

#### Contact
<sub><sup>Contact: felicio.cassalho@noaa.gov </sup></sub>

![NOAA logo](https://user-images.githubusercontent.com/72229285/216712553-c1e4b2fa-4b6d-4eab-be0f-f7075b6151d1.png)


#### Acknowledgements:
*OCSTrack was inspired by the MATLAB-based [WW3-tools](https://github.com/NOAA-EMC/WW3-tools) and [wave-tools](https://github.com/NOAA-EMC/WW3-tools) collocation tools developed for WaveWatch3.*
