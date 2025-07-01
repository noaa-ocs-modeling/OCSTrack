import xarray as xr
import numpy as np


def get_max_neighbors(result_list):
    """Get the maximum number of neighbors from a list of 2D arrays."""
    return max(arr.shape[1] for arr in result_list if arr.ndim == 2)


def pad_arrays_to_max(arrays, max_cols):
    """Pad 2D arrays with NaNs to have the same number of columns."""
    padded = []
    for arr in arrays:
        if arr.shape[1] < max_cols:
            pad_width = max_cols - arr.shape[1]
            pad_arr = np.pad(arr, ((0, 0), (0, pad_width)), constant_values=np.nan)
            padded.append(pad_arr)
        else:
            padded.append(arr[:, :max_cols])  # Optionally truncate
    return padded


def make_collocated_nc(results: dict, n_nearest: int = None) -> xr.Dataset:
    # Determine max neighbors from actual data if not explicitly provided
    max_neighbors = get_max_neighbors(results["model_swh"]) if n_nearest is None else n_nearest

    # Pad all neighbor-dependent arrays
    model_swh = pad_arrays_to_max(results["model_swh"], max_neighbors)
    model_dpt = pad_arrays_to_max(results["model_dpt"], max_neighbors)
    dist_deltas = pad_arrays_to_max(results["dist_deltas"], max_neighbors)
    node_ids = pad_arrays_to_max(results["node_ids"], max_neighbors)

    data_vars = {
        "lon": (["time"], np.concatenate(results["lon_sat"])),
        "lat": (["time"], np.concatenate(results["lat_sat"])),
        "sat_swh": (["time"], np.concatenate(results["sat_swh"])),
        "sat_sla": (["time"], np.concatenate(results["sat_sla"])),
        "model_swh": (["time", "nearest_nodes"], np.vstack(model_swh)),
        "model_swh_weighted": (["time"], np.concatenate(results["model_swh_weighted"])),
        "model_dpt": (["time", "nearest_nodes"], np.vstack(model_dpt)),
        "dist_deltas": (["time", "nearest_nodes"], np.vstack(dist_deltas)),
        "node_ids": (["time", "nearest_nodes"], np.vstack(node_ids)),
        "time_deltas": (["time"], np.concatenate(results["time_deltas"])),
        "bias_raw": (["time"], np.concatenate(results["bias_raw"])),
        "bias_weighted": (["time"], np.concatenate(results["bias_weighted"])),
        "source_sat": (["time"], np.concatenate(results["source_sat"])),
    }

    if "dist_coast" in results:
        data_vars["dist_coast"] = (["time"], np.concatenate(results["dist_coast"]))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.concatenate(results["time_sat"]),
            "nearest_nodes": np.arange(max_neighbors),
        },
        attrs={
            "Conventions": "CF-1.7",
            "title": "CF-compliant Satellite vs Model SWH Dataset",
        }
    )
    return ds
