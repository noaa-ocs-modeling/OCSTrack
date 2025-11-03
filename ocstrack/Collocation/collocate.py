"""Collocate Module"""

from typing import Optional, Tuple, Union
import logging
import numpy as np
import xarray as xr
from tqdm import tqdm

from ocstrack.Model.model import SCHISM
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation.argofloat import ArgoData
from ocstrack.Collocation.temporal import temporal_nearest, temporal_interpolated
from ocstrack.Collocation.spatial import GeocentricSpatialLocator, inverse_distance_weights
from ocstrack.Collocation.output import make_collocated_nc_2d, make_collocated_nc_3d

# Try to import gsw for accurate depth, but fall back if not installed
try:
    import gsw
    _HAS_GSW = True
except ImportError:
    _HAS_GSW = False

_logger = logging.getLogger(__name__)


class Collocate:
    """
    Model–observation collocation engine (2D and 3D).

    This is the main class. It handles the spatial and temporal
    collocation of observation data (Satellite or Argo) with
    unstructured model outputs (e.g., SCHISM).
    
    It dispatches to either a 2D/surface ('2D', '3D_Surface') or
    3D/profile ('3D_Profile') collocation strategy based on the
    model_run's 'var_type'.

    Methods
    -------
    run(output_path=None)
        Run the full collocation and return a combined xarray.Dataset.
    """
    def __init__(self,
                 model_run: SCHISM,
                 observation: Union[SatelliteData, ArgoData],
                 dist_coast: Optional[xr.Dataset] = None,
                 n_nearest: Optional[int] = 4,
                 search_radius: Optional[float] = None,
                 time_buffer: Optional[np.timedelta64] = None,
                 weight_power: float = 1.0,
                 temporal_interp: bool = False) -> None:
        """
        Initialize the Collocate object.

        Parameters
        ----------
        model_run : SCHISM
            Initialized Model object (e.g., SCHISM) containing grid and file info.
        observation : Union[SatelliteData, ArgoData]
            Initialized Observation object (e.g., SatelliteData or ArgoData).
        dist_coast : xarray.Dataset, optional
            Optional dataset containing distance-to-coast info.
        n_nearest : int, optional 
            Number of nearest spatial model nodes to use (default=4).
        search_radius : float, optional
            Radius (in meters) to search for spatial neighbors. 
            If provided, overwrites n_nearest.
        time_buffer : np.timedelta64, optional
            Temporal search buffer; if None, inferred from model timestep.
        weight_power : float, default=1.0
            Power exponent for inverse distance weighting.
        temporal_interp : bool, default=False
            Whether to perform linear temporal interpolation (default=False).
        
        Raises
        ------
        ValueError
            If both or neither of 'n_nearest'/'search_radius' are provided,
            or if time buffer cannot be inferred.
        """
        self.model = model_run
        self.obs = observation
        self.dist_coast = dist_coast["distcoast"] if dist_coast is not None else None
        self.n_nearest = n_nearest
        self.search_radius = search_radius
        self.weight_power = weight_power
        self.temporal_interp = temporal_interp

        if isinstance(self.obs, SatelliteData):
            self.obs_time_coord = 'time'
        elif isinstance(self.obs, ArgoData):
            self.obs_time_coord = 'JULD'
        else:
            raise TypeError("Observation type must be SatelliteData or ArgoData")
        
        # Store the collocation type based on the model dict
        self.collocation_type = self.model.model_dict.get('var_type', '2D')

        if search_radius is not None and n_nearest is not None:
            _logger.warning("Both search_radius and n_nearest provided; "
                            "ignoring n_nearest and using radius-based spatial matching.")
            self.n_nearest = None # Radius search overrides
        elif search_radius is None and n_nearest is None:
            raise ValueError("Specify either 'n_nearest' or 'search_radius'")

        _logger.info("Initializing 3D Geocentric (WGS 84) spatial locator.")
        self.locator = GeocentricSpatialLocator(
            self.model.mesh_x, self.model.mesh_y, model_height=None
        )
        
        if not _HAS_GSW and self.collocation_type == '3D_Profile':
             _logger.warning("`gsw` library not found. `pip install gsw` for accurate depth conversion."
                             " Falling back to simple approximation (dbar * -1.0197).")

        # Time buffer logic based on collocation type
        if time_buffer is None:
            _logger.info("Inferring time_buffer...")
            if not self.model.files:
                raise ValueError("Cannot infer time_buffer: Model has no files.")
            
            # This logic now works for ALL types
            example_file = self.model.files[0]
            
            if self.collocation_type in ['2D', '3D_Surface']:
                times = self.model.load_variable(example_file)["time"].values
            
            elif self.collocation_type == '3D_Profile':
                _logger.info("Loading one 3D file pair for time buffer inference...")
                # Load just one file pair to get time info
                with self.model.load_3d_file_pair(example_file) as m_data:
                    times = m_data.time.values
            else:
                raise ValueError(f"Unknown var_type: {self.collocation_type}")

            if len(times) < 2:
                raise ValueError("Cannot infer time_buffer: less than two model timesteps.")
            
            timestep = np.diff(times).mean()
            self.time_buffer = timestep / 2
            _logger.info(f"Inferred time_buffer as half mean timestep: {self.time_buffer}")
        
        else:
            self.time_buffer = time_buffer

    def run(self, output_path: Optional[str] = None) -> xr.Dataset:
        """
        Run the full model–observation collocation process.

        Dispatches to the correct method (surface or profile)
        based on the 'var_type' provided during initialization.

        Parameters
        ----------
        output_path : str, optional
            If provided, writes collocated output to this NetCDF file path.

        Returns
        -------
        xarray.Dataset
            Dataset containing collocated observation and model data.
        
        Raises
        ------
        TypeError
            If the observation object type does not match the collocation type
            (e.g., using ArgoData for '2D' collocation).
        NotImplementedError
            If the 'var_type' is not recognized.
        """
        if self.collocation_type in ['2D', '3D_Surface']:
            if not isinstance(self.obs, SatelliteData):
                raise TypeError("2D/3D_Surface collocation requires SatelliteData observation type.")
            _logger.info("Starting 2D/Surface collocation...")
            return self._run_surface_collocation(output_path)
        
        elif self.collocation_type == '3D_Profile':
            if not isinstance(self.obs, ArgoData):
                raise TypeError("3D_Profile collocation requires ArgoData observation type.")
            
            if self.search_radius is not None:
                _logger.warning("Radius search is not yet supported for 3D_Profile, "
                                "using n_nearest=4 (default) instead.")
                self.n_nearest = 4 # Default for 3D
            
            _logger.info("Starting 3D Profile collocation...")
            return self._run_profile_collocation(output_path)
        
        else:
            raise NotImplementedError(f"Collocation type {self.collocation_type} not supported.")

    def _run_surface_collocation(self, output_path: Optional[str] = None) -> xr.Dataset:
        """
        Run the collocation process for 2D/surface variables.

        This method iterates over each model file, finds all satellite
        observations within the time window, and performs 2D spatial
        collocation (horizontal only).

        Parameters
        ----------
        output_path : str, optional
            If provided, writes collocated output to this NetCDF file path.

        Returns
        -------
        xarray.Dataset
            Dataset containing collocated 2D satellite and model data.
        """
        # Make variable names generic
        model_var_name = self.model.model_dict['var']
        # Map model var to obs var (assuming SatelliteData)
        obs_var_map = {'sigWaveHeight': 'swh'} 
        obs_var_name = obs_var_map.get(model_var_name, 'swh') # Default to 'swh'
        
        # Generic results dictionary
        results = {k: [] for k in [
            "time_obs", "lat_obs", "lon_obs", "source_obs",
            f"obs_{obs_var_name}", "obs_sla", f"model_{model_var_name}", "model_dpt",
            "dist_deltas", "node_idx", "time_deltas",
            f"model_{model_var_name}_weighted", "bias_raw", "bias_weighted"
        ]}
        
        # Remove keys for optional obs variables if not present
        if 'swh' not in self.obs.ds:
             results.pop(f"obs_{obs_var_name}", None)
        if 'sla' not in self.obs.ds:
             results.pop("obs_sla", None)
        if 'source' not in self.obs.ds:
             results.pop("source_obs", None)

        include_coast = self.dist_coast is not None
        if include_coast:
            results["dist_coast"] = []

        for path in tqdm(self.model.files, desc="Collocating Surface..."):
            m_var = self.model.load_variable(path)
            m_times = m_var["time"].values

            if self.temporal_interp:
                obs_sub, ib, ia, wts, tdel = temporal_interpolated(self.obs.ds,
                                                                   m_times,
                                                                   self.time_buffer,
                                                                   self.obs_time_coord)
                time_args = (ib, ia, wts)
            else:
                obs_sub, idx, tdel = temporal_nearest(self.obs.ds,
                                                      m_times,
                                                      self.time_buffer,
                                                      self.obs_time_coord)
                time_args = idx

            if obs_sub[self.obs_time_coord].size == 0:
                _logger.debug(f"No satellite data for file {path}, skipping.")
                continue

            if self.search_radius is not None:
                spatial = self._collocate_with_radius(obs_sub, m_var, time_args)
            else:
                spatial = self._collocate_with_nearest(obs_sub, m_var, time_args)

            # Append generic results
            results["time_obs"].append(obs_sub[self.obs_time_coord].values)
            results["lat_obs"].append(obs_sub["lat"].values)
            results["lon_obs"].append(obs_sub["lon"].values)
            results["time_deltas"].append(tdel)

            if "source_obs" in results:
                results["source_obs"].append(obs_sub["source"].values)
            if f"obs_{obs_var_name}" in results:
                results[f"obs_{obs_var_name}"].append(obs_sub[obs_var_name].values)
            if "obs_sla" in results:
                results["obs_sla"].append(obs_sub["sla"].values)
            
            results[f"model_{model_var_name}"].append(spatial["model_var"])
            results["model_dpt"].append(spatial["model_dpt"])
            results["dist_deltas"].append(spatial["dist_deltas"])
            results["node_idx"].append(spatial["node_idx"])
            results[f"model_{model_var_name}_weighted"].append(spatial["model_var_weighted"])
            results["bias_raw"].append(spatial["bias_raw"])
            results["bias_weighted"].append(spatial["bias_weighted"])

            if include_coast:
                coast_d = self._coast_distance(obs_sub["lat"].values, obs_sub["lon"].values)
                results["dist_coast"].append(coast_d)

        n_neighbors = None if self.search_radius is not None else self.n_nearest
        ds_out = make_collocated_nc_2d(results, n_neighbors, model_var_name, obs_var_name)
        
        if output_path:
            ds_out.to_netcdf(output_path)
        return ds_out

    def _run_profile_collocation(self, output_path: Optional[str] = None) -> xr.Dataset:
        """
        Run the full 3D profile collocation.

        This method operates on the single, pre-loaded 3D model dataset.
        It performs temporal, spatial (horizontal), and vertical
        collocation for each Argo profile.

        Parameters
        ----------
        output_path : str, optional
            If provided, writes collocated output to this NetCDF file path.

        Returns
        -------
        xarray.Dataset
            Dataset containing collocated 3D profile data.
        """

        _logger.info("Starting 3D Profile collocation (file-by-file)...")
        
        # Get variable names from model_dict
        main_var = self.model.model_dict['var']
        zcor_var = self.model.model_dict['zcor_var']
        
        # Map model var name to argo var name
        obs_var_map = {'temperature': 'temp', 'salinity': 'psal'}
        obs_var = obs_var_map.get(main_var)
        if obs_var is None:
            raise ValueError(f"No Argo variable mapping for model var '{main_var}'")

        max_levels = self.obs.ds.sizes['N_LEVELS']

        # Create lists to store results from each loop
        results_list = {
            "time": [],
            "lat": [],
            "lon": [],
            "time_deltas": [],
            "dist_deltas": [],
            "node_idx": [],
            "argo_depth": [],
            f"argo_{obs_var}": [],
            f"model_{main_var}": [],
        }

        # Loop through model files
        for f_main_path in tqdm(self.model.files, desc="Collocating 3D Profiles"):
            try:
                # Load just this one file's data (temp + zcor)
                m_data = self.model.load_3d_file_pair(f_main_path)
            except Exception as e:
                _logger.warning(f"Skipping file {f_main_path} due to load error: {e}")
                continue

            m_times = m_data["time"].values
            
            # 1. Temporal Collocation (find Argo profiles for *this* file)
            if self.temporal_interp:
                argo_sub, ib, ia, wts, tdel = temporal_interpolated(self.obs.ds,
                                                                   m_times,
                                                                   self.time_buffer,
                                                                   self.obs_time_coord)
                time_args = (ib, ia, wts)
            else:
                argo_sub, idx, tdel = temporal_nearest(self.obs.ds,
                                                       m_times,
                                                       self.time_buffer,
                                                       self.obs_time_coord)
                time_args = idx

            if argo_sub[self.obs_time_coord].size == 0:
                m_data.close()
                continue # No Argo profiles match this model file

            # 2. Spatial Collocation
            lons = argo_sub["LONGITUDE"].values
            lats = argo_sub["LATITUDE"].values
            heights = np.zeros_like(lons)

            dists, nodes = self.locator.query_nearest(
                lons, lats, heights, k=self.n_nearest
            )
            
            # 3. Vertical Collocation
            v_data = self._extract_model_profiles_3d(
                argo_sub=argo_sub,
                time_args=time_args,
                nodes=nodes,
                dists=dists,
                model_var_name=main_var,
                model_zcor_name=zcor_var,
                obs_var_name=obs_var,
                max_levels=max_levels,
                model_data=m_data
            )
            
            # 4. Append results to list
            results_list["time"].append(argo_sub[self.obs_time_coord].values)
            results_list["lat"].append(lats)
            results_list["lon"].append(lons)
            results_list["time_deltas"].append(tdel)
            results_list["dist_deltas"].append(dists)
            results_list["node_idx"].append(nodes)
            results_list["argo_depth"].append(v_data["obs_depth"])
            results_list[f"argo_{obs_var}"].append(v_data["obs_var"])
            results_list[f"model_{main_var}"].append(v_data["model_var_interp"])

            m_data.close() # Close the file before loading the next

        # --- NEW: Concatenate all results at the end ---
        _logger.info("Assembling final dataset from all file chunks...")
        
        # Check if any results were found
        if not results_list["time"]:
            _logger.warning("No collocated 3D data found. Returning empty dataset.")
            return xr.Dataset()

        final_results = {}
        for key, value in results_list.items():
            if key in ["dist_deltas", "node_idx", "argo_depth",
                       f"argo_{obs_var}", f"model_{main_var}"]:
                final_results[key] = np.vstack(value)
            else:
                final_results[key] = np.concatenate(value)

        ds_out = make_collocated_nc_3d(final_results, max_levels)
        
        if output_path:
            ds_out.to_netcdf(output_path)
        return ds_out

    def _extract_model_profiles_3d(self,
                                   argo_sub: xr.Dataset,
                                   time_args: tuple,
                                   nodes: np.ndarray,
                                   dists: np.ndarray,
                                   model_var_name: str,
                                   model_zcor_name: str,
                                   obs_var_name: str,
                                   max_levels: int,
                                   model_data: xr.Dataset
                                   ) -> dict:
        """
        Extracts and collocates 3D model profiles onto Argo vertical levels.

        This performs the "vertical-then-horizontal" interpolation:
        1. Temporal interpolation of model data (if enabled) at each nearest node.
        2. Vertical interpolation at *each* nearest node onto the Argo depth levels.
        3. Spatial inverse-distance-weighting of the vertically collocated profiles.

        Parameters
        ----------
        argo_sub : xr.Dataset
            The subset of Argo profiles to collocate.
        time_args : tuple or np.ndarray
            Temporal indices or interpolation arguments.
        nodes : np.ndarray
            Array of nearest node indices, shape (n_profiles, k_nearest).
        dists : np.ndarray
            Array of distances to nearest nodes, shape (n_profiles, k_nearest).
        model_var_name : str
            The name of the main model variable (e.g., 'temperature').
        model_zcor_name : str
            The name of the model z-coordinate variable (e.g., 'zCoordinates').
        obs_var_name : str
            The name of the main observation variable (e.g., 'temp').
        max_levels : int
            The maximum number of vertical levels for padding the output arrays.
        model_data : xr.Dataset
            Single file.

        Returns
        -------
        dict
            A dictionary containing the padded, collocated profiles:
            - "obs_depth": (n_profiles, max_levels)
            - "obs_var": (n_profiles, max_levels)
            - "model_var_interp": (n_profiles, max_levels)
        """
        
        # Get spatial weights (shape: [n_profiles, k_nearest])
        spatial_weights = inverse_distance_weights(dists, self.weight_power)
        n_profiles = argo_sub[self.obs_time_coord].size
        
        out_obs_depth = np.full((n_profiles, max_levels), np.nan)
        out_obs_var = np.full((n_profiles, max_levels), np.nan)
        out_model_var = np.full((n_profiles, max_levels), np.nan)

        # Get data from the argo_sub (the subset)
        argo_var_name_adj = f"{obs_var_name.upper()}_ADJUSTED"
        argo_var_name_raw = f"{obs_var_name.upper()}"
        argo_all_pres = argo_sub.get('PRES_ADJUSTED', argo_sub['PRES']).values
        argo_all_var = argo_sub.get(argo_var_name_adj, argo_sub[argo_var_name_raw]).values
        argo_all_lats = argo_sub['LATITUDE'].values

        # Use the passed model_data
        model_all_var = model_data[model_var_name]
        model_all_zcor = model_data[model_zcor_name]
        
        for i in tqdm(range(n_profiles), desc="Vertical Collocation"):
            # profile *in the subset*.
            argo_pres_i = argo_all_pres[i, :]
            argo_lat_i = argo_all_lats[i]
            
            if _HAS_GSW:
                argo_depth = gsw.z_from_p(argo_pres_i, argo_lat_i)
            else:
                argo_depth = argo_pres_i * -1.0197
            
            argo_var_i = argo_all_var[i, :]
            
            valid_argo = ~np.isnan(argo_depth) & ~np.isnan(argo_var_i)
            if not np.any(valid_argo):
                continue
                
            argo_depth_valid = argo_depth[valid_argo]
            argo_var_valid = argo_var_i[valid_argo]
            
            sort_idx_argo = np.argsort(argo_depth_valid)
            argo_depth_sorted = argo_depth_valid[sort_idx_argo]
            argo_var_sorted = argo_var_valid[sort_idx_argo]
            n_valid_levels = len(argo_depth_sorted)
            
            out_obs_depth[i, :n_valid_levels] = argo_depth_sorted
            out_obs_var[i, :n_valid_levels] = argo_var_sorted

            # 2. Get Model Data (Temporal Interp)
            node_indices = nodes[i, :] 
            
            if self.temporal_interp:
                ib, ia, wts = time_args
                t_idx_b, t_idx_a, t_wt = ib[i], ia[i], wts[i]
                
                zcor_b = model_all_zcor.isel(time=t_idx_b, nSCHISM_hgrid_node=node_indices).values
                zcor_a = model_all_zcor.isel(time=t_idx_a, nSCHISM_hgrid_node=node_indices).values
                var_b = model_all_var.isel(time=t_idx_b, nSCHISM_hgrid_node=node_indices).values
                var_a = model_all_var.isel(time=t_idx_a, nSCHISM_hgrid_node=node_indices).values
                
                model_zcor_at_nodes = zcor_b * (1 - t_wt) + zcor_a * t_wt
                model_var_at_nodes = var_b * (1 - t_wt) + var_a * t_wt
            
            else: # Temporal nearest
                t_idx = time_args[i]
                model_zcor_at_nodes = model_all_zcor.isel(time=t_idx, nSCHISM_hgrid_node=node_indices).values
                model_var_at_nodes = model_all_var.isel(time=t_idx, nSCHISM_hgrid_node=node_indices).values
            
            # 3. Vertical-then-Horizontal Interpolation
            model_profiles_at_argo_depths = np.full((self.n_nearest, n_valid_levels), np.nan)
            
            for k in range(self.n_nearest):
                model_zcor_k = model_zcor_at_nodes[k, :]
                model_var_k = model_var_at_nodes[k, :]

                valid_model = ~np.isnan(model_zcor_k) & ~np.isnan(model_var_k)
                if not np.any(valid_model):
                    continue 

                model_zcor_k_valid = model_zcor_k[valid_model]
                model_var_k_valid = model_var_k[valid_model]

                sort_idx_model = np.argsort(model_zcor_k_valid)
                model_zcor_k_sorted = model_zcor_k_valid[sort_idx_model]
                model_var_k_sorted = model_var_k_valid[sort_idx_model]
                
                model_profile_k_interp = np.interp(
                    argo_depth_sorted,
                    model_zcor_k_sorted,
                    model_var_k_sorted,
                    left=np.nan, right=np.nan
                )
                
                model_profiles_at_argo_depths[k, :] = model_profile_k_interp

            # 4. Spatial IDW
            weights_i = spatial_weights[i, :]
            
            with np.errstate(invalid='ignore'):
                profiles_T = model_profiles_at_argo_depths.T
                final_model_profile = np.nansum(profiles_T * weights_i, axis=1)
                norm_weights = np.nansum(
                    (~np.isnan(profiles_T)) * weights_i, axis=1
                )
                norm_weights[norm_weights == 0] = np.nan
                final_model_profile = final_model_profile / norm_weights
            
            out_model_var[i, :n_valid_levels] = final_model_profile

        return {
            "obs_depth": out_obs_depth,
            "obs_var": out_obs_var,
            "model_var_interp": out_model_var
        }

    def _get_obs_height(self, obs_sub: xr.Dataset) -> np.ndarray:
        """
        Extracts observation height/altitude from the dataset.

        Defaults to 0m (sea level) if not found, which is
        appropriate for Argo floats or satellites without height data.

        Parameters
        ----------
        obs_sub : xr.Dataset
            The subset of observation data.

        Returns
        -------
        np.ndarray
            An array of observation heights.
        """
        # For Satellite
        if 'height' in obs_sub:
            return obs_sub["height"].values
        if 'altitude' in obs_sub:
            return obs_sub["altitude"].values

        # For ArgoData
        if isinstance(self.obs, ArgoData):
            return np.zeros_like(obs_sub["LONGITUDE"].values)

        _logger.warning("No 'height' or 'altitude' in obs data. "
                        "Defaulting to 0m for geocentric query.")
        # Fallback for SatelliteData without height
        try:
            return np.zeros_like(obs_sub["lon"].values)
        except KeyError: # Fallback for ArgoData if it got here
             return np.zeros_like(obs_sub["LONGITUDE"].values)


    def _collocate_with_radius(self, obs_sub, m_var, time_args):
        """
        Perform 2D collocation using a spatial search radius.

        Parameters
        ----------
        obs_sub : xarray.Dataset
            Subset of satellite observations to collocate.
        m_var : xarray.DataArray
            Model variable data for the current time slice.
        time_args : tuple or np.ndarray
            Temporal indices or interpolation arguments.

        Returns
        -------
        dict
            Dictionary containing 2D collocated arrays (e.g., "model_var", "dist_deltas").
        """
        
        obs_var_map = {'sigWaveHeight': 'swh'}
        model_var_name = self.model.model_dict['var']
        obs_var_name = obs_var_map.get(model_var_name, 'swh')
        
        lons = obs_sub["lon"].values
        lats = obs_sub["lat"].values
        heights = self._get_obs_height(obs_sub)

        all_dists, all_nodes = self.locator.query_radius(
            lons, lats, heights, radius_m=self.search_radius
        )
        
        flat_nodes = []
        flat_ib, flat_ia, flat_wt = [], [], []
        obs_lens = []

        for i, (nodes, dists) in enumerate(zip(all_nodes, all_dists)):
            obs_lens.append(len(nodes))
            if len(nodes) == 0:
                continue 

            if self.temporal_interp:
                ib, ia, wts = time_args
                flat_ib.extend([ib[i]] * len(nodes))
                flat_ia.extend([ia[i]] * len(nodes))
                flat_wt.extend([wts[i]] * len(nodes))
            else:
                flat_ib.extend([time_args[i]] * len(nodes)) 

            flat_nodes.extend(nodes)

        # Handle case where no nodes were found for any obs
        if not flat_nodes:
            n_obs = len(lons)
            nan_arr = np.full((n_obs, 1), np.nan)
            return {
                "model_var": nan_arr,
                "model_dpt": nan_arr,
                "dist_deltas": nan_arr,
                "node_idx": nan_arr,
                "model_var_weighted": np.full(n_obs, np.nan),
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

        weights_list = [inverse_distance_weights(d[None, :], self.weight_power)[0]
                        if len(d) > 0 else np.array([np.nan])
                        for d in split_dists]

        weighted_vals = [np.nansum(v * w) if len(v) > 0 else np.nan
                         for v, w in zip(split_vals, weights_list)]

        return {
            "model_var": pad(split_vals),
            "model_dpt": pad(split_dpts),
            "dist_deltas": pad(split_dists),
            "node_idx": pad([a.astype(float) for a in split_nodes]),
            "model_var_weighted": np.array(weighted_vals),
            "bias_raw": np.array([
                np.nanmean(v) - s if len(v) > 0 else np.nan
                for v, s in zip(split_vals, obs_sub[obs_var_name].values)
            ]),
            "bias_weighted": np.array(weighted_vals) - obs_sub[obs_var_name].values,
        }

    def _collocate_with_nearest(self, obs_sub, m_var, time_args):
        """
        Perform 2D collocation using k-nearest spatial search.

        Parameters
        ----------
        obs_sub : xarray.Dataset
            Subset of satellite observations to collocate.
        m_var : xarray.DataArray
            Model variable data for the current time slice.
        time_args : tuple or np.ndarray
            Temporal indices or interpolation arguments.

        Returns
        -------
        dict
            Dictionary containing 2D collocated arrays (e.g., "model_var", "dist_deltas").
        """

        obs_var_map = {'sigWaveHeight': 'swh'}
        model_var_name = self.model.model_dict['var']
        obs_var_name = obs_var_map.get(model_var_name, 'swh')
        
        lons = obs_sub["lon"].values
        lats = obs_sub["lat"].values
        heights = self._get_obs_height(obs_sub)

        dists, nodes = self.locator.query_nearest(
            lons, lats, heights, k=self.n_nearest
        )

        m_vals, m_dpts = self._extract_model_values(m_var, time_args, nodes)
        weights = inverse_distance_weights(dists, self.weight_power)
        weighted = (m_vals * weights).sum(axis=1)

        return {
            "model_var": m_vals,
            "model_dpt": m_dpts,
            "dist_deltas": dists,
            "node_idx": nodes,
            "model_var_weighted": weighted,
            "bias_raw": m_vals.mean(axis=1) - obs_sub[obs_var_name].values,
            "bias_weighted": weighted - obs_sub[obs_var_name].values,
        }

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
        (Used by 2D/Surface collocation)

        Parameters
        ----------
        m_var : xarray.DataArray
            Model variable to extract from (e.g. significant wave height)
        times_or_inds : tuple or list
            Time indices or interpolation args (ib, ia, wts)
        nodes : np.ndarray
            Node indices of nearest spatial neighbors (can be 1D or 2D)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Extracted model values and node depths
        """
        model_data = m_var.values
        depths = self.model.mesh_depth
        values, dpts = [], []

        node_dim = None
        for dim in m_var.dims:
            if dim != 'time':
                node_dim = dim
                break
        if node_dim is None:
            raise ValueError("Could not find a spatial node dimension in model variable.")

        # Logic for radius search (nodes is 1D)
        if self.search_radius is not None:
             if self.temporal_interp:
                ib, ia, wts = times_or_inds
                for i, nd in enumerate(nodes): # nodes is flat 1D array
                    v0 = model_data[ib[i], nd]
                    v1 = model_data[ia[i], nd]
                    values.append(v0 * (1 - wts[i]) + v1 * wts[i])
                    dpts.append(depths[nd])
             else:
                t_idx = times_or_inds
                for i, nd in enumerate(nodes): # nodes is flat 1D array
                    values.append(model_data[t_idx[i], nd])
                    dpts.append(depths[nd])

        # Logic for k-nearest (nodes is 2D)
        else:
            if self.temporal_interp:
                ib, ia, wts = times_or_inds
                # This handles nodes being shape (n_obs, k_nearest)
                for i in range(len(ib)):
                    nd = nodes[i]
                    v0 = model_data[ib[i], nd]
                    v1 = model_data[ia[i], nd]
                    values.append(v0 * (1 - wts[i]) + v1 * wts[i])
                    dpts.append(depths[nd])
            else:
                t_idx = times_or_inds
                # This handles nodes being shape (n_obs, k_nearest)
                for i, t_idx_i in enumerate(t_idx):
                    nd = nodes[i]
                    t = m_var["time"].values[t_idx_i]
                    values.append(m_var.sel(time=t, **{node_dim: nd}).values)
                    dpts.append(depths[nd])

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
