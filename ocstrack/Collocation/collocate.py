"""Collocate Module"""

from typing import Optional, Tuple, Union
import logging
import numpy as np
import xarray as xr
from tqdm import tqdm

from ocstrack.Model.model import SCHISM
from ocstrack.Satellite.satellite import SatelliteData
from ocstrack.Collocation.temporal import temporal_nearest, temporal_interpolated
from ocstrack.Collocation.spatial import GeocentricSpatialLocator, inverse_distance_weights
from ocstrack.Collocation.output import make_collocated_nc

_logger = logging.getLogger(__name__)


class Collocate:
    """Model–satellite collocation engine

    This is the mains class. 
    It handles the spatial and temporal collocation of satellite
    altimetry data (e.g., significant wave height, sea level anomaly (TBD))
    with unstructured model outputs (e.g., SCHISM). It supports both
    nearest-neighbor (in time) and temporally interpolated collocation strategies.

    Methods
    -------
    run(output_path=None)
        Run the collocation over all model files and return a combined
        xarray.Dataset.

    Notes
    -----
    Collocation is performed using:
    - Nearest N spatial nodes (with inverse distance weighting)
    - Radius (meters) based search
    - Nearest or interpolated temporal matching
    - Optional distance-to-coast dataset for filtering/post-processing

    Automatically infers time_buffer from model time step if not provided.
    """
    def __init__(self,
                model_run: SCHISM,
                satellite: SatelliteData,
                dist_coast: Optional[xr.Dataset] = None,
                n_nearest: Optional[int] = None,
                search_radius: Optional[float] = None,
                time_buffer: Optional[np.timedelta64] = None,
                weight_power: float = 1.0,
                temporal_interp: bool = False) -> None:
        """
        Parameters
        ----------
        model_run : SCHISM
            Model object containing grid, file paths, and data access
        satellite : SatelliteData
            Satellite data wrapper providing SWH, SLA, etc.
        dist_coast : xarray.Dataset, optional
            Optional dataset containing distance-to-coast info
        n_nearest : int, optional 
            Number of nearest spatial model nodes to use
        search_radius : float, optional
            Radius (in meters) to search for spatial neighbors. 
            If provided, overwrite n_nearest and uses radius-based spatial matching.
        time_buffer : np.timedelta64, optional
            Temporal search buffer; if None, inferred from model timestep
        weight_power : float, default=1.0
            Power exponent for inverse distance weighting
        temporal_interp : bool, default=False
            Whether to perform linear temporal interpolation
        """
        self.model = model_run
        self.sat = satellite
        self.dist_coast = dist_coast["distcoast"] if dist_coast is not None else None
        self.n_nearest = n_nearest
        self.search_radius = search_radius
        self.weight_power = weight_power
        self.temporal_interp = temporal_interp

        if search_radius is not None and n_nearest is not None:
            _logger.warning("Both search_radius and n_nearest provided;" \
            "ignoring n_nearest and using radius-based spatial matching.")
        elif search_radius is None and n_nearest is None:
            raise ValueError("Specify either 'n_nearest' or 'search_radius'")

        # Set locator
        _logger.info("Initializing 3D Geocentric (WGS 84) spatial locator.")
        self.locator = GeocentricSpatialLocator(
            self.model.mesh_x, self.model.mesh_y, model_height=None
        )

        # If radius search is on, nullify n_nearest
        if search_radius is not None:
            self.n_nearest = None  # Prevent accidental use

        # Automatically estimate time buffer if not provided
        if time_buffer is None:
            example_file = self.model.files[0]
            times = self.model.load_variable(example_file)["time"].values

            if len(times) < 2:
                raise ValueError("Cannot infer time_buffer: less than two model timesteps.")

            # Calculate timestep and use half of it as buffer
            timestep = times[1] - times[0]  # Assumes constant step
            self.time_buffer = timestep / 2
            _logger.info(f"Inferred time_buffer as half timestep: {self.time_buffer}")
        else:
            self.time_buffer = time_buffer

    def _extract_model_values(self,
                              m_var: xr.DataArray,
                              times_or_inds: Union[np.ndarray,
                                                   Tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]],
                              nodes: np.ndarray) -> Tuple[np.ndarray,
                                                          np.ndarray]:
        """
        Extract model variable values and corresponding depths at given times and nodes.
        (MODIFIED to be model-agnostic)

        Parameters
        ----------
        m_var : xarray.DataArray
            Model variable to extract from (e.g. significant wave height)
        times_or_inds : tuple or list
            Time indices or interpolation args (ib, ia, wts)
        nodes : np.ndarray
            Node indices of nearest spatial neighbors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Extracted model values and node depths
        """
        model_data = m_var.values
        depths = self.model.mesh_depth
        values, dpts = [], []

        # Find the node dimension name (e.g., 'node' or 'nSCHISM_hgrid_node')
        node_dim = None
        for dim in m_var.dims:
            if dim != 'time':
                node_dim = dim
                break

        if node_dim is None:
            raise ValueError("Could not find a spatial node dimension in model variable.")


        if self.temporal_interp:
            ib, ia, wts = times_or_inds
            for i, nd in enumerate(nodes):
                v0 = model_data[ib[i], nd]
                v1 = model_data[ia[i], nd]
                values.append(v0 * (1 - wts[i]) + v1 * wts[i])
                dpts.append(depths[nd])
        else:
            for i, (t_idx, nd) in enumerate(zip(times_or_inds, nodes)):
                t = m_var["time"].values[t_idx]
                values.append(m_var.sel(time=t, **{node_dim: nd}).values)
                dpts.append(depths[nd])

        # Handle the N=0 case
        if not values:
            k = nodes.shape[1] if nodes.ndim == 2 else 0
            return np.empty((0, k)), np.empty((0, k))

        return np.array(values), np.array(dpts)

    def _coast_distance(self,
                        lats: np.ndarray,
                        lons: np.ndarray) -> np.ndarray:
        """
        Get distance to coast for given lat/lon points using optional dataset.

        Parameters
        ----------
        lats : array-like
            Latitudes of satellite observations
        lons : array-like
            Longitudes of satellite observations

        Returns
        -------
        np.ndarray
            Interpolated coastal distances, or NaNs if unavailable
        """
        if self.dist_coast is None:
            return np.full_like(lats, fill_value=np.nan, dtype=float)
        return self.dist_coast.sel(
            latitude=xr.DataArray(lats, dims="points"),
            longitude=xr.DataArray(lons, dims="points"),
            method="nearest",
        ).values

    def _get_sat_height(self, sat_sub: xr.Dataset) -> np.ndarray:
        """
        Extracts satellite height/altitude from the dataset.
        Defaults to 0m with a warning if not found.
        """
        if 'height' in sat_sub:
            return sat_sub["height"].values
        if 'altitude' in sat_sub:
            return sat_sub["altitude"].values

        _logger.warning("No 'height' or 'altitude' in satellite data. "
                       "Defaulting to 0m for geocentric query. "
                       "This may be inaccurate for altimeter data.")
        return np.zeros_like(sat_sub["lon"].values)

    def _collocate_with_radius(self, sat_sub, m_var, time_args):
        """
        Collocate satellite observations with model output using a spatial search radius.
        This is more challendi

        Parameters
        ----------
        - sat_sub (xarray.Dataset): Subset of satellite data.
        - m_var (str): Model variable name (e.g., 'sigWaveHeight').
        - time_args (tuple or list): Time interpolation arguments or time indices.

        Returns
        -------
        - dict: A dictionary containing collocated model variables:
            * model_swh: 2D array [obs, nearest_nodes]
            * model_dpt: 2D array [obs, nearest_nodes]
            * dist_deltas: 2D array [obs, nearest_nodes] (distances)
            * node_ids: 2D array [obs, nearest_nodes]
            * model_swh_weighted: 1D array of weighted model SWH [obs]
            * bias_raw: 1D array of unweighted biases [obs]
            * bias_weighted: 1D array of weighted biases [obs]

        Notes
        -----
        Padding is applied to all per-observation arrays to ensure they can be stacked into
        uniform 2D arrays, even though the number of nearest model nodes may differ per observation.
        This ensures consistent array dimensions and enables construction of an xarray.Dataset later
        dimension mismatches.
        """
        lons = sat_sub["lon"].values
        lats = sat_sub["lat"].values
        heights = self._get_sat_height(sat_sub)

        all_dists, all_nodes = self.locator.query_radius(
            lons, lats, heights, radius_m=self.search_radius
        )

        flat_nodes = []
        flat_ib, flat_ia, flat_wt = [], [], []
        obs_lens = []

        for i, (nodes, dists) in enumerate(zip(all_nodes, all_dists)):
            obs_lens.append(len(nodes))
            if len(nodes) == 0:
                continue  # no nodes found — handled after extraction

            if self.temporal_interp:
                ib, ia, wts = time_args
                flat_ib.extend([ib[i]] * len(nodes))
                flat_ia.extend([ia[i]] * len(nodes))
                flat_wt.extend([wts[i]] * len(nodes))
            else:
                flat_ib.extend([time_args[i]] * len(nodes))  # just time index

            flat_nodes.extend(nodes)

        # Handle case where no nodes were found for any obs
        if not flat_nodes:
            n_obs = len(lons)
            nan_arr = np.full((n_obs, 1), np.nan)
            return {
                "model_swh": nan_arr,
                "model_dpt": nan_arr,
                "dist_deltas": nan_arr,
                "node_ids": nan_arr,
                "model_swh_weighted": np.full(n_obs, np.nan),
                "bias_raw": np.full(n_obs, np.nan),
                "bias_weighted": np.full(n_obs, np.nan),
            }

        # Perform extraction once
        if self.temporal_interp:
            m_vals, m_dpts = self._extract_model_values(
                m_var, (np.array(flat_ib),
                        np.array(flat_ia),
                        np.array(flat_wt)),np.array(flat_nodes)
            )
        else:
            m_vals, m_dpts = self._extract_model_values(
                m_var, np.array(flat_ib), np.array(flat_nodes)
            )

        # Reshape into per-observation lists
        def unflatten(arr, lens):
            return np.split(arr, np.cumsum(lens)[:-1])

        split_vals = unflatten(m_vals, obs_lens)
        split_dpts = unflatten(m_dpts, obs_lens)
        split_dists = unflatten(
            np.concatenate([np.array(d) for d in all_dists if len(d) > 0]), obs_lens
            )
        split_nodes = unflatten(np.array(flat_nodes), obs_lens)

        # Handle obs with no neighbors
        def pad(arrs):
            max_len = max((len(a) for a in arrs), default=1)
            return np.stack([
                np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrs
            ])

        # Generate weights and weighted values
        weights_list = [inverse_distance_weights(d[None, :], self.weight_power)[0]
                        if len(d) > 0 else np.array([np.nan])
                        for d in split_dists]

        weighted_vals = [np.sum(v * w) if len(v) > 0 else np.nan
                        for v, w in zip(split_vals, weights_list)]

        return {
            "model_swh": pad(split_vals),
            "model_dpt": pad(split_dpts),
            "dist_deltas": pad(split_dists),
            "node_ids": pad([a.astype(float) for a in split_nodes]),
            "model_swh_weighted": np.array(weighted_vals),
            "bias_raw": np.array([
                np.nanmean(v) - s if len(v) > 0 else np.nan
                for v, s in zip(split_vals, sat_sub["swh"].values)
            ]),
            "bias_weighted": np.array(weighted_vals) - sat_sub["swh"].values,
        }

    def _collocate_with_nearest(self, sat_sub, m_var, time_args):
        """
        Perform collocation using nearest-neighbor spatial search.

        For each satellite observation, find a fixed number of nearest model nodes,
        extract model values at relevant times (interpolated or nearest),
        compute inverse-distance weights, and calculate weighted averages.

        Parameters
        ----------
        sat_sub : xarray.Dataset
            Subset of satellite observations to collocate.
        m_var : xarray.DataArray
            Model variable data for the current time slice.
        time_args : tuple or np.ndarray
            Temporal indices or interpolation arguments depending on temporal method.

        Returns
        -------
        dict
            Dictionary containing arrays for:
            - model_swh: model values per neighbor and observation
            - model_dpt: node depths
            - dist_deltas: distances to neighbors
            - node_ids: spatial node indices
            - model_swh_weighted: weighted model values per observation
            - bias_raw: difference between mean model and satellite values
            - bias_weighted: difference between weighted model and satellite values
        """
        lons = sat_sub["lon"].values
        lats = sat_sub["lat"].values
        heights = self._get_sat_height(sat_sub)

        dists, nodes = self.locator.query_nearest(
            lons, lats, heights, k=self.n_nearest
        )

        m_vals, m_dpts = self._extract_model_values(m_var, time_args, nodes)
        weights = inverse_distance_weights(dists, self.weight_power)
        weighted = (m_vals * weights).sum(axis=1)

        return {
            "model_swh": m_vals,
            "model_dpt": m_dpts,
            "dist_deltas": dists,
            "node_ids": nodes,
            "model_swh_weighted": weighted,
            "bias_raw": m_vals.mean(axis=1) - sat_sub["swh"].values,
            "bias_weighted": weighted - sat_sub["swh"].values,
        }

    def run(self,
            output_path: Optional[str] = None) -> xr.Dataset:
        """
        Run the full model–satellite collocation process over all model files.

        This function iterates over all model output files, performs temporal and spatial
        collocation of satellite data with model results, calculates weighted averages,
        biases, and optionally writes the collocated results to a NetCDF file.

        Parameters
        ----------
        output_path : str, optional
            If provided, writes collocated output to NetCDF file

        Returns
        -------
        xarray.Dataset
            Dataset containing collocated satellite and model data
        """
        results = {k: [] for k in [
            "time_sat", "lat_sat", "lon_sat", "source_sat",
            "sat_swh", "sat_sla", "model_swh", "model_dpt",
            "dist_deltas", "node_ids", "time_deltas",
            "model_swh_weighted", "bias_raw", "bias_weighted"
        ]}

        include_coast = self.dist_coast is not None
        if include_coast:
            results["dist_coast"] = []

        for path in tqdm(self.model.files, desc="Collocating..."):
            m_var = self.model.load_variable(path)
            m_times = m_var["time"].values

            if self.temporal_interp:
                sat_sub, ib, ia, wts, tdel = temporal_interpolated(self.sat.ds,
                                                                   m_times,
                                                                   self.time_buffer)
                time_args = (ib, ia, wts)
            else:
                sat_sub, idx, tdel = temporal_nearest(self.sat.ds, m_times, self.time_buffer)
                time_args = idx

            if self.search_radius is not None:
                spatial = self._collocate_with_radius(sat_sub, m_var, time_args)
            else:
                spatial = self._collocate_with_nearest(sat_sub, m_var, time_args)

            results["time_sat"].append(sat_sub["time"].values)
            results["lat_sat"].append(sat_sub["lat"].values)
            results["lon_sat"].append(sat_sub["lon"].values)
            results["source_sat"].append(sat_sub["source"].values)
            results["sat_swh"].append(sat_sub["swh"].values)
            results["sat_sla"].append(sat_sub["sla"].values)
            results["time_deltas"].append(tdel)

            for k in ["model_swh",
                      "model_dpt",
                      "dist_deltas",
                      "node_ids",
                      "model_swh_weighted",
                      "bias_raw",
                      "bias_weighted"]:
                results[k].append(spatial[k])

            if include_coast:
                coast_d = self._coast_distance(sat_sub["lat"].values, sat_sub["lon"].values)
                results["dist_coast"].append(coast_d)

        n_neighbors = None if self.search_radius is not None else self.n_nearest
        ds_out = make_collocated_nc(results, n_neighbors)
        if output_path:
            ds_out.to_netcdf(output_path)
        return ds_out
