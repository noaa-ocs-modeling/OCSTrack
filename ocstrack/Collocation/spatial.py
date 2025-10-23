"""Functions for spatial collocation"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple


def lat_lon_to_cartesian_vec(latitude: np.ndarray,
                             longitude: np.ndarray,
                             height: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts geodetic coordinates (latitude, longitude, height) to
    geocentric Cartesian coordinates (X, Y, Z) using numpy.

    This function uses the WGS 84 ellipsoid model.
    Assumes latitude and longitude are in degrees, height in meters.
    Returns (X, Y, Z) in meters.
    """
    # WGS 84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e_sq = 2 * f - f**2  # Eccentricity squared (e^2)

    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    # Calculate N (radius of curvature in the prime vertical)
    n = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)

    # Calculate Cartesian coordinates
    x = (n + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - e_sq) + height) * np.sin(lat_rad)

    return x, y, z

def inverse_distance_weights(distances: np.ndarray,
                             power: float = 1.0) -> np.ndarray:
    """
    Compute inverse distance weights (IDW) with configurable exponent.

    Parameters
    ----------
    distances : np.ndarray
        Distance array to nearest neighbors, shape (N, k)
    power : float, optional
        Power exponent for distance weighting (default is 1.0).
        Use 1.0 for linear, 2.0 for quadratic, etc.

    Returns
    -------
    np.ndarray
        Normalized inverse distance weights of shape (N, k)

    Notes
    -----
    A small epsilon (1e-6) is used to avoid division by zero.
    """
    safe_distances = np.maximum(distances, 1e-6) #to avoid division by zero
    weights = 1.0 / np.power(safe_distances, power)
    return weights / weights.sum(axis=1, keepdims=True)

class GeocentricSpatialLocator:
    """
    KDTree-based spatial query engine using 3D Geocentric (WGS 84) coordinates.

    Handles both nearest-neighbor and radius-based lookups between satellite
    points and model grid nodes using a fast 3D KDTree built on
    ECEF (Earth-Centered, Earth-Fixed) coordinates.
    """

    def __init__(self,
                 model_lon: np.ndarray,
                 model_lat: np.ndarray,
                 model_height: np.ndarray = None) -> None:
        """
        Parameters
        ----------
        model_lon : np.ndarray
            Longitudes of model mesh nodes (degrees)
        model_lat : np.ndarray
            Latitudes of model mesh nodes (degrees)
        model_height : np.ndarray, optional
            Heights of model mesh nodes above ellipsoid (meters).
            If None, defaults to 0 (on the ellipsoid surface).
        """
        if model_height is None:
            model_height = np.zeros_like(model_lon)

        x, y, z = lat_lon_to_cartesian_vec(model_lat, model_lon, model_height)
        self.model_xyz = np.column_stack((x, y, z))
        self.tree = KDTree(self.model_xyz)

    def _get_query_points(self,
                          sat_lon: np.ndarray,
                          sat_lat: np.ndarray,
                          sat_height: np.ndarray) -> np.ndarray:
        """Helper to convert satellite coordinates to 3D Cartesian."""
        x_q, y_q, z_q = lat_lon_to_cartesian_vec(sat_lat, sat_lon, sat_height)
        return np.column_stack((x_q, y_q, z_q))

    def query_nearest(self,
                      sat_lon: np.ndarray,
                      sat_lat: np.ndarray,
                      sat_height: np.ndarray,
                      k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query for the 'k' nearest model nodes to each satellite point.

        Parameters
        ----------
        sat_lon : np.ndarray
            Longitudes of satellite observations (degrees)
        sat_lat : np.ndarray
            Latitudes of satellite observations (degrees)
        sat_height : np.ndarray
            Heights/Altitudes of satellite observations (meters above ellipsoid)
        k : int, optional
            Number of nearest model neighbors (default is 3)

        Returns
        -------
        tuple of np.ndarray
            Distances (meters) and indices of nearest model nodes, shape (N, k)
        """
        query_points = self._get_query_points(sat_lon, sat_lat, sat_height)
        distances, indices = self.tree.query(query_points, k=k)
        return distances, indices

    def query_radius(self,
                     sat_lon: np.ndarray,
                     sat_lat: np.ndarray,
                     sat_height: np.ndarray,
                     radius_m: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Query for all model nodes within the 3D search radius.

        Parameters
        ----------
        sat_lon : np.ndarray
            Longitudes of satellite observations (degrees)
        sat_lat : np.ndarray
            Latitudes of satellite observations (degrees)
        sat_height : np.ndarray
            Heights/Altitudes of satellite observations (meters above ellipsoid)
        radius_m : float
            Search radius in meters

        Returns
        -------
        tuple of list[np.ndarray], list[np.ndarray]
            - Distances (in meters) to all matched model nodes per point
            - Corresponding indices of model nodes per point
        """
        query_points = self._get_query_points(sat_lon, sat_lat, sat_height)

        # Find indices of points within the radius
        indices_list = self.tree.query_ball_point(query_points, r=radius_m)

        distances_list = []

        # Now calculate true distances for those indices
        for i, node_inds in enumerate(indices_list):
            if not node_inds:
                distances_list.append(np.array([]))
                indices_list[i] = np.array([]) # Ensure consistent type
                continue

            query_point = query_points[i]
            node_points = self.model_xyz[node_inds]

            # Calculate true 3D Euclidean distances
            dists = np.linalg.norm(node_points - query_point, axis=1)

            distances_list.append(dists)
            indices_list[i] = np.array(node_inds) # Ensure consistent type

        return distances_list, indices_list
