import numpy as np
from scipy.spatial import KDTree
import math


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

class NearestSpatialLocator:
    """KDTree-based spatial query engine

    Handles nearest-neighbor lookups between satellite points and
    model grid nodes using a fast cKDTree.

    Methods
    -------
    query(lon, lat, k=3) -> Tuple[np.ndarray, np.ndarray]
        Query for the `k` nearest model nodes to each satellite point.

    Notes
    -----
    Coordinates are assumed to be in the same projected or geodetic
    system (e.g., lon/lat or UTM).
    """

    def __init__(self,
                 x_coords: np.ndarray,
                 y_coords: np.ndarray) -> None:
        """
        Parameters
        ----------
        x_coords : np.ndarray
            X-coordinates (e.g., longitude) of model mesh nodes
        y_coords : np.ndarray
            Y-coordinates (e.g., latitude) of model mesh nodes
        """
        self.tree = KDTree(np.column_stack((x_coords, y_coords)))

    def query(self,
              lon: np.ndarray,
              lat: np.ndarray,
              k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        lon : np.ndarray
            Longitudes of satellite observations
        lat : np.ndarray
            Latitudes of satellite observations
        k : int, optional
            Number of nearest model neighbors to return (default is 3)

        Returns
        -------
        tuple of np.ndarray
            Distances and indices of nearest model nodes, both of shape (N, k)
        """
        points = np.column_stack((lon, lat))
        distances, indices = self.tree.query(points, k=k)
        return distances, indices


class RadiusSpatialLocator:
    """Radius-based spatial query engine using projected KDTree

    Performs radius-based neighbor searches from satellite points
    to unstructured model grid nodes. Converts lat/lon coordinates
    into a local Cartesian system (meters) for distance-based filtering.

    Methods
    -------
    query(lon, lat) -> Tuple[list[np.ndarray], list[np.ndarray]]
        Query for all model nodes within the specified radius for each
        satellite observation.

    Notes
    -----
    Coordinates must be provided in geographic degrees (lat/lon).
    Internally uses an equirectangular projection centered on the
    model domain for fast KDTree searches in meters.
    """

    def __init__(self,
                 x_coords: np.ndarray,
                 y_coords: np.ndarray,
                 radius_m: float,
                 origin_lat: float = None,
                 origin_lon: float = None) -> None:
        """
        Parameters
        ----------
        x_coords : np.ndarray
            Longitudes of model mesh nodes
        y_coords : np.ndarray
            Latitudes of model mesh nodes
        radius_m : float
            Search radius in meters
        origin_lat : float, optional
            Optional latitude to use as projection center; defaults to mean of y_coords
        origin_lon : float, optional
            Optional longitude to use as projection center; defaults to mean of x_coords
        """
        self.radius_m = radius_m
        self.lons = np.asarray(x_coords)
        self.lats = np.asarray(y_coords)

        if origin_lat is None:
            origin_lat = np.mean(self.lats)
        if origin_lon is None:
            origin_lon = np.mean(self.lons)

        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.R = 6371000  # Earth radius in meters

        self.xy_coords = np.array([self._latlon_to_xy(lat, lon)
                                   for lat, lon in zip(self.lats, self.lons)])
        self.tree = KDTree(self.xy_coords)

    def _latlon_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert latitude and longitude to local Cartesian x/y in meters.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        tuple[float, float]
            X and Y coordinates in meters relative to origin
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        origin_lat_rad = math.radians(self.origin_lat)
        origin_lon_rad = math.radians(self.origin_lon)

        x = self.R * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
        y = self.R * (lat_rad - origin_lat_rad)
        return (x, y)

    def query(self,
              lon: np.ndarray,
              lat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Query for all model nodes within the search radius for each satellite point.

        Parameters
        ----------
        lon : np.ndarray
            Longitudes of satellite observations
        lat : np.ndarray
            Latitudes of satellite observations

        Returns
        -------
        tuple of list[np.ndarray], list[np.ndarray]
            - Distances (in meters) to all matched model nodes per point
            - Corresponding indices of model nodes per point

        Notes
        -----
        Output lists may have different lengths per satellite point.
        """
        distances_all = []
        indices_all = []

        for la, lo in zip(lat, lon):
            xq, yq = self._latlon_to_xy(la, lo)
            node_inds = self.tree.query_ball_point([xq, yq], r=self.radius_m)

            if not node_inds:
                distances_all.append(np.array([]))
                indices_all.append(np.array([]))
                continue

            # I found this efficient way for calculating euclidean distances in meters:
            dists = np.linalg.norm(self.xy_coords[node_inds] - np.array([xq, yq]), axis=1)
            distances_all.append(np.array(dists))
            indices_all.append(np.array(node_inds))

        return distances_all, indices_all
