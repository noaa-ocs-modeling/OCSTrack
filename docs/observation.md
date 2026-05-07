# Observation Data

This section covers tools for downloading, processing, and handling observational data from satellite altimetry and Argo floats.

## Data Handlers

These classes are used to interact with processed observational data within the collocation workflow.

### Argo Floats

::: ocstrack.Observation.argofloat.ArgoData

### Satellite Altimetry

::: ocstrack.Observation.satellite.SatelliteData

---

## Data Acquisition

These functions are high-level entry points for downloading and pre-processing raw data from public repositories.

### Argo Data Acquisition

Use these functions to download and prepare Argo float data.

!!! tip
    The main function to use here is `get_argo`. It orchestrates the entire download and processing pipeline.

::: ocstrack.Observation.get_argo.get_argo
::: ocstrack.Observation.get_argo.download_argo_data
::: ocstrack.Observation.get_argo.crop_argo_data
::: ocstrack.Observation.get_argo.clean_argo_data
::: ocstrack.Observation.get_argo.generate_monthly_dates
::: ocstrack.Observation.get_argo.crop_by_box_argo

### Satellite Data Acquisition

Use these functions to download and prepare satellite altimetry data.

!!! tip
    The main functions to use here are `get_per_sat` for a single satellite and `get_multi_sat` for multiple satellites.

::: ocstrack.Observation.get_sat.get_per_sat
::: ocstrack.Observation.get_sat.get_multi_sat
::: ocstrack.Observation.get_sat.download_sat_data
::: ocstrack.Observation.get_sat.crop_sat_data
::: ocstrack.Observation.get_sat.concat_sat_data
::: ocstrack.Observation.get_sat.generate_daily_dates
::: ocstrack.Observation.get_sat.crop_by_box

---

## Data URLs

This module contains the base URLs for the data sources.

::: ocstrack.Observation.urls
