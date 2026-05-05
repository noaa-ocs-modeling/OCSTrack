
import numpy as np
from ocstrack.Collocation.spatial import inverse_distance_weights

def test_inverse_distance_weights():
    """
    Test the inverse distance weighting function.
    """
    # Distances to four points
    distances = np.array([[10, 20, 30, 40]]) # Shape (1, 4) for one observation
    power = 1.0
    
    weights = inverse_distance_weights(distances, power)
    
    # Expected weights are proportional to 1/distance and sum to 1.
    inv_d = 1 / distances
    expected_weights = inv_d / np.sum(inv_d)
    
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)

def test_inverse_distance_weights_zero_distance():
    """
    Test that a zero distance results in a weight of 1 for that point
    and 0 for all others.
    """
    distances = np.array([[10, 0, 30, 40]])
    power = 1.0
    
    weights = inverse_distance_weights(distances, power)
    
    expected_weights = np.array([[0, 1, 0, 0]])
    
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)

def test_inverse_distance_weights_with_power():
    """
    Test the weighting with a power greater than 1.
    """
    distances = np.array([[10, 20]]) # Shape (1, 2)
    power = 2.0
    
    weights = inverse_distance_weights(distances, power)
    
    inv_d2 = 1 / (distances**2)
    expected_weights = inv_d2 / np.sum(inv_d2)
    
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)
