# OCSTrack
OCSTrack is an object-oriented Python package for the along-track collocation of satellite (2D) and ArgoFloat (3D) data with ocean circulation and wave model outputs.
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


 ### Satellite Altimetry Data Support

It also integrates the Euro Argo dataset [ifremmer](https://fleetmonitoring.euro-argo.eu/dashboard?Status=Active&Country=France) for 3D temperature and salinity collocation.

### Ocean Model Data Support
 Supports outputs from various ocean circulation and wave models:
 * [SCHISM](https://github.com/schism-dev/schism) and its coupled verison with WWMIII
 * [ADCIRC+SWAN](https://adcirc.org/)
 * WaveWatch3 (to be implemented)


## Installation

1.  **Create new conda environment:**
    This command creates an environment named `ocstrack` and installs all dependencies from `conda-forge`.
    ```bash
    conda create -n ocstrack -c conda-forge python=3.10 numpy xarray scipy tqdm requests netcdf4 h5netcdf dask
    conda activate ocstrack
    ```

2.  **Install `ocstrack`:**
    Finally, install this package using `pip`.
    ```bash
    pip install ocstrack
    ```

    If you want to install the latest dev version, using this instead:
    ```bash
    pip install "git+[https://github.com/noaa-ocs-modeling/OCSTrack.git](https://github.com/noaa-ocs-modeling/OCSTrack.git)"
    ```

## Usage
See examples directory.

## Contributing
We welcome contributions to OCSTrack! If you have ideas for improvements, new features, or find a bug, please don't hesitate to open an issue or submit a pull request on our GitHub repository. Your input helps make OCSTrack better for everyone.

### Contact
<sup>Contact: felicio.cassalho@noaa.gov </sup>

![NOAA logo](https://user-images.githubusercontent.com/72229285/216712553-c1e4b2fa-4b6d-4eab-be0f-f7075b6151d1.png)


#### Acknowledgements:
*OCSTrack was inspired by the MATLAB-based [WW3-tools](https://github.com/NOAA-EMC/WW3-tools) and [wave-tools](https://github.com/erdc/wave-tools) collocation tools developed for WaveWatch3.*
