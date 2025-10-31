"""Functions for temporal collocation"""

import numpy as np
import xarray as xr


def temporal_nearest(ds_obs: xr.Dataset,
                     model_times: np.ndarray,
                     buffer: np.timedelta64,
                     time_coord_name: str = 'time'
                     ) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """
    Match observations observations to the nearest model timestamps.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset with a 'time' dimension
    model_times : np.ndarray
        Array of model time values (np.datetime64)
    buffer : np.timedelta64
        Time buffer to include obs points near the model time window

    Returns
    -------
    tuple
        obs_sub : xr.Dataset
            Subset of obs data within the buffered time range
        nearest_inds : np.ndarray
            Index of nearest model time for each obs time
        time_deltas : np.ndarray
            Difference (in seconds) between obs and matched model time
    """
    start = model_times.min() - buffer
    end = model_times.max() + buffer

    obs_sub = ds_obs.sortby(time_coord_name).sel({time_coord_name: slice(start, end)})
    obs_times = obs_sub[time_coord_name].values

    nearest_inds = np.abs(obs_times[:, None] - model_times[None, :]).argmin(axis=1)
    nearest_model_times = model_times[nearest_inds]
    time_deltas = (obs_times - nearest_model_times).astype("timedelta64[s]").astype(int)

    return obs_sub, nearest_inds, time_deltas


def temporal_interpolated(ds_obs: xr.Dataset,
                          model_times: np.ndarray,
                          buffer: np.timedelta64,
                          time_coord_name: str = 'time'
                          ) -> tuple[xr.Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear time interpolation using surrounding model timestamps.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset with a 'time' dimension
    model_times : np.ndarray
        Array of model time values (np.datetime64)
    buffer : np.timedelta64
        Time buffer to include obs points near the model time window

    Returns
    -------
    tuple
        obs_sub : xr.Dataset
            Subset of obs data used in interpolation
        ib : np.ndarray
            Index of earlier model time
        ia : np.ndarray
            Index of later model time
        weights : np.ndarray
            Linear interpolation weights for each obs point
        time_deltas : np.ndarray
            Difference (in seconds) between obs and closest model time
    """
    start = model_times.min() - buffer
    end = model_times.max() + buffer

    obs_sorted = ds_obs.sortby(time_coord_name).sel({time_coord_name: slice(start, end)})
    obs_times = obs_sorted[time_coord_name].values

    model_times_s = model_times.astype("datetime64[s]")

    ib, ia, weights, valid_idx = [], [], [], []

    for i, t in enumerate(obs_times):
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

    obs_sub = obs_sorted.isel({time_coord_name: valid_idx})
    ib = np.array(ib, dtype=int)
    ia = np.array(ia, dtype=int)
    weights = np.array(weights)

    # For metadata: calculate time delta to the nearest of the two model timestamps
    t0 = model_times_s[ib]
    t1 = model_times_s[ia]

    dt0 = np.abs(obs_sub[time_coord_name].values - t0)
    dt1 = np.abs(obs_sub[time_coord_name].values - t1)
    nearest_model_times = np.where(dt0 <= dt1, t0, t1)
    time_deltas = (
        obs_sub[time_coord_name].values - nearest_model_times
        ).astype("timedelta64[s]").astype(int)

    return obs_sub, ib, ia, weights, time_deltas
