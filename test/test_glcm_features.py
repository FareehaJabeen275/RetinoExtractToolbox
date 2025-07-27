import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.glcm_features import extract_glcm_features

def test_extract_glcm_features():
    # Create a dummy grayscale image (simple gradient)
    dummy_image = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (64, 1))
    
    # Call the feature extraction function
    features = extract_glcm_features(dummy_image)

    # Check output type
    assert isinstance(features, dict), "Output should be a dictionary"

    # Check some expected keys presence (at least a subset)
    expected_keys = [
        'homogeneity ', 'contrast ', 'energy ', 'dissimilarity ',
        'angular_second_moment', 'correlation', 'sum_average',
        'mean_x', 'mean_y', 'variance', 'standard_deviation',
        'maximal_prob', 'idm', 'idm_normalized',
        'sum_of_variance', 'difference_of_variance',
        'sum_of_squares', 'entropy', 'sum_entropy',
        'difference_entropy', 'renyi_entropy', 'yager_entropy',
        'kapur_entropy', 'information_measure_of_correlation1',
        'information_measure_of_correlation2', 'cluster_shade',
        'cluster_prominence'
    ]

    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"

    # Check values are numeric (float or np.float64)
    for key, value in features.items():
        # Some features from graycoprops return arrays; convert to float or check array dtype
        if hasattr(value, 'dtype') and hasattr(value, 'shape'):
            # Convert to float if possible
            val = np.asarray(value)
            assert np.issubdtype(val.dtype, np.floating), f"Feature {key} should be float type"
        else:
            assert isinstance(value, (float, np.floating, int)), f"Feature {key} should be a number"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
