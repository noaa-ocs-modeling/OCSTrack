""" Functions for writing collocated outputs file """

import xarray as xr
import numpy as np
import logging

_logger = logging.getLogger(__name__)

def get_max_neighbors(result_list):
    """
    Get the maximum number of neighbors from a list of 2D arrays.
    
    Used to handle ragged results from radius search.
    
    Parameters
    ----------
    result_list : list
        A list of 2D numpy arrays (n_obs, n_neighbors)

    Returns
    -------
    int
        The maximum number of neighbors (max columns) found in the list.
    """
    if not result_list:
        return 1
    
    # Filter for valid arrays
    valid_arrs = [arr for arr in result_list if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.size > 0]
    if not valid_arrs:
        # This can happen if radius search finds 0 neighbors for all points
        return 1
        
    return max((arr.shape[1] for arr in valid_arrs), default=1)


def pad_arrays_to_max(arrays, max_cols):
    """
    Pad 2D arrays with NaNs to have the same number of columns.
    
    Used to stack ragged results from radius search into a single 2D array.

    Parameters
    ----------
    arrays : list
        List of 2D numpy arrays (n_obs_i, n_neighbors_i).
    max_cols : int
        The target number of columns (max neighbors) to pad to.

    Returns
    -------
    np.ndarray
        A single, stacked 2D array of shape (N_total_obs, max_cols).
    """
    padded = []
    for arr in arrays:
        # Handle empty lists or other non-array data
        if not hasattr(arr, 'ndim') or arr.size == 0:
             # Create an empty array with the correct 2nd dim shape
             if hasattr(arr, 'shape') and arr.ndim == 2:
                 padded.append(np.full((arr.shape[0], max_cols), np.nan))
             else:
                 padded.append(np.full((0, max_cols), np.nan))
             continue
        
        if arr.ndim == 1: 
            # Ensure it's a 2D array, (n_obs, 1)
            arr = arr.reshape(-1, 1)
            
        if arr.shape[1] < max_cols:
            pad_width = max_cols - arr.shape[1]
            pad_arr = np.pad(arr, ((0, 0), (0, pad_width)), constant_values=np.nan)
            padded.append(pad_arr)
        else:
            padded.append(arr[:, :max_cols])  # Truncate if larger
    
    if not padded:
        return np.empty((0, max_cols))
    
    try:
        return np.vstack(padded)
    except ValueError as e:
        _logger.error(f"Error vstacking padded arrays: {e}")
        for i, p in enumerate(padded):
            _logger.error(f"Array {i} shape: {p.shape}")
        raise


def make_collocated_nc_2d(results: dict, 
                          n_nearest: int = None,
                          model_var_name: str = 'model_var',
                          obs_var_name: str = 'obs_var'
                          ) -> xr.Dataset:
    """ 
    Format the 2D collocated data into a CF-compliant xarray Dataset.

    Parameters
    ----------
    results : dict
        Dictionary containing the concatenated lists of collocated data.
    n_nearest : int, optional
        The number of nearest neighbors used (k). If None, inferred
        from the radius search results.
    model_var_name : str, default='model_var'
        The name of the model variable (e.g., 'sigWaveHeight').
    obs_var_name : str, default='obs_var'
        The name of the observation variable (e.g., 'swh').

    Returns
    -------
    xarray.Dataset
        A CF-compliant dataset of the collocated 2D surface data.
    """
    
    # Check for empty results
    if not results['time_obs']:
        _logger.warning("No collocated 2D data found. Returning empty dataset.")
        return xr.Dataset()
    
    model_var_key = f"model_{model_var_name}"
    model_weighted_key = f"model_{model_var_name}_weighted"
    obs_var_key = f"obs_{obs_var_name}"

    # Determine max neighbors from actual data
    max_neighbors = get_max_neighbors(results[model_var_key]) if n_nearest is None else n_nearest
    
    data_vars = {}
    
    # Concatenate 1D arrays
    for key in ["time_obs", "lat_obs", "lon_obs", "time_deltas", 
                "bias_raw", "bias_weighted", model_weighted_key]:
        if key in results:
            data_vars[key] = (["time"], np.concatenate(results[key]))

    # Add optional 1D arrays
    if obs_var_key in results:
        data_vars[obs_var_key] = (["time"], np.concatenate(results[obs_var_key]))
    if "obs_sla" in results:
        data_vars["obs_sla"] = (["time"], np.concatenate(results["obs_sla"]))
    if "source_obs" in results:
         data_vars["source_obs"] = (["time"], np.concatenate(results["source_obs"]))
    if "dist_coast" in results:
        data_vars["dist_coast"] = (["time"], np.concatenate(results["dist_coast"]))

    # Pad and concatenate 2D arrays
    data_vars[model_var_key] = (["time", "nearest_nodes"], pad_arrays_to_max(results[model_var_key], max_neighbors))
    data_vars["model_dpt"] = (["time", "nearest_nodes"], pad_arrays_to_max(results["model_dpt"], max_neighbors))
    data_vars["dist_deltas"] = (["time", "nearest_nodes"], pad_arrays_to_max(results["dist_deltas"], max_neighbors))
    data_vars["node_ids"] = (["time", "nearest_nodes"], pad_arrays_to_max(results["node_ids"], max_neighbors))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": data_vars["time_obs"][1], # Use the concatenated time as coord
            "nearest_nodes": np.arange(max_neighbors),
        },
        attrs={
            "Conventions": "CF-1.7",
            "title": "CF-compliant Satellite vs Model Collocation Dataset",
            "description": f"Collocation of {obs_var_name} vs {model_var_name}"
        }
    )
    return ds

def make_collocated_nc_3d(results: dict, max_levels: int) -> xr.Dataset:
    """
    Format the 3D collocated data into a CF-compliant xarray Dataset.
    
    Parameters
    ----------
    results : dict
        Dictionary containing the final 3D profile arrays.
    max_levels : int
        The size of the padded vertical dimension ('n_levels').

    Returns
    -------
    xarray.Dataset
        A CF-compliant dataset of the collocated 3D profile data.
    """
    if results["time"].size == 0:
        _logger.warning("No collocated 3D data found. Returning empty dataset.")
        return xr.Dataset()

    # Get the variable names from the results dict
    obs_var_name = [k for k in results.keys() if k.startswith('argo_') and k != 'argo_depth'][0]
    model_var_name = [k for k in results.keys() if k.startswith('model_')][0]
    
    data_vars = {
        "lon": (["time"], results["lon"]),
        "lat": (["time"], results["lat"]),
        
        # Profile data
        "depth": (["time", "n_levels"], results["argo_depth"]),
        obs_var_name: (["time", "n_levels"], results[obs_var_name]),
        model_var_name: (["time", "n_levels"], results[model_var_name]),
        
        # Collocation metadata
        "time_deltas": (["time"], results["time_deltas"]),
        "dist_deltas": (["time", "nearest_nodes"], results["dist_deltas"]),
        "node_ids": (["time", "nearest_nodes"], results["node_ids"]),
    }
    if "dist_coast" in results:
        data_vars["dist_coast"] = (["time"], results["dist_coast"])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": results["time"],
            "n_levels": np.arange(max_levels),
            "nearest_nodes": np.arange(results["node_ids"].shape[1]),
        },
        attrs={
            "Conventions": "CF-1.7",
            "title": "CF-compliant Profile (Argo) vs Model (SCHISM) Dataset",
            "description": f"Collocation of {obs_var_name} vs {model_var_name}"
        }
    )
    return ds
