"""Functions for temporal collocation"""

import numpy as np
import xarray as xr


def temporal_nearest(ds_sat: xr.Dataset,
                     model_times: np.ndarray,
                     buffer: np.timedelta64
                     ) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """
    Match satellite observations to the nearest model timestamps.

    Parameters
    ----------
    ds_sat : xr.Dataset
        Satellite dataset with a 'time' dimension
    model_times : np.ndarray
        Array of model time values (np.datetime64)
    buffer : np.timedelta64
        Time buffer to include satellite points near the model time window

    Returns
    -------
    tuple
        sat_sub : xr.Dataset
            Subset of satellite data within the buffered time range
        nearest_inds : np.ndarray
            Index of nearest model time for each satellite time
        time_deltas : np.ndarray
            Difference (in seconds) between satellite and matched model time
    """
    start = model_times.min() - buffer
    end = model_times.max() + buffer
    sat_sub = ds_sat.sortby("time").sel(time=slice(start, end))
    sat_times = sat_sub["time"].values

    nearest_inds = np.abs(sat_times[:, None] - model_times[None, :]).argmin(axis=1)
    nearest_model_times = model_times[nearest_inds]
    time_deltas = (sat_times - nearest_model_times).astype("timedelta64[s]").astype(int)

    return sat_sub, nearest_inds, time_deltas


def temporal_interpolated(ds_sat: xr.Dataset,
                          model_times: np.ndarray,
                          buffer: np.timedelta64
                          ) -> tuple[xr.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear time interpolation using surrounding model timestamps.

    Parameters
    ----------
    ds_sat : xr.Dataset
        Satellite dataset with a 'time' dimension
    model_times : np.ndarray
        Array of model time values (np.datetime64)
    buffer : np.timedelta64
        Time buffer to include satellite points near the model time window

    Returns
    -------
    tuple
        sat_sub : xr.Dataset
            Subset of satellite data used in interpolation
        ib : np.ndarray
            Index of earlier model time
        ia : np.ndarray
            Index of later model time
        weights : np.ndarray
            Linear interpolation weights for each satellite point
        time_deltas : np.ndarray
            Difference (in seconds) between satellite and closest model time
    """
    start = model_times.min() - buffer
    end = model_times.max() + buffer
    sat_sorted = ds_sat.sortby("time").sel(time=slice(start, end))
    sat_times = sat_sorted["time"].values
    model_times_s = model_times.astype("datetime64[s]")

    ib, ia, weights, valid_idx = [], [], [], []

    for i, t in enumerate(sat_times):
        idx = np.searchsorted(model_times_s, t)
        i0 = max(0, idx - 1)
        i1 = min(len(model_times_s) - 1, idx)

        if i0 == i1:
            continue  # No valid interval for interpolation

        dt = model_times_s[i1] - model_times_s[i0]
        if dt == np.timedelta64(0, "s"):
            continue  # Avoid divide-by-zero or duplicate timestamps

        w = (t - model_times_s[i0]) / dt
        ib.append(i0)
        ia.append(i1)
        weights.append(w)
        valid_idx.append(i)

    sat_sub = sat_sorted.isel(time=valid_idx)
    ib = np.array(ib, dtype=int)
    ia = np.array(ia, dtype=int)
    weights = np.array(weights)

    # For metadata: calculate time delta to the nearest of the two model timestamps
    t0 = model_times_s[ib]
    t1 = model_times_s[ia]
    dt0 = np.abs(sat_sub["time"].values - t0)
    dt1 = np.abs(sat_sub["time"].values - t1)
    nearest_model_times = np.where(dt0 <= dt1, t0, t1)
    time_deltas = (
        sat_sub["time"].values - nearest_model_times
        ).astype("timedelta64[s]").astype(int)

    return sat_sub, ib, ia, weights, time_deltas
