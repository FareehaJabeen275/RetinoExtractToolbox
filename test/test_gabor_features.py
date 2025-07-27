import numpy as np
import sys
import os
import pytest

# Ensure the features folder is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.gabor_features import extract_gabor_features

def test_gabor_feature_extraction():
    # Create a dummy grayscale image (100x100 gradient)
    dummy_image = np.tile(np.linspace(0, 1, 100), (100,1))

    # Call the feature extractor
    features = extract_gabor_features(dummy_image)

    # Check output is a dictionary
    assert isinstance(features, dict), "Output should be a dictionary"

    # Check keys exist and have expected format 'Feature_0', 'Feature_1', etc.
    assert all(key.startswith('Feature_') for key in features.keys()), "Feature keys format incorrect"

    # Check values inside each feature dictionary
    expected_keys = [
        'Mean', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis',
        'Energy', 'Entropy', 'Contrast', 'Homogeneity'
    ]

    for feature_dict in features.values():
        # Check each feature dict is a dict
        assert isinstance(feature_dict, dict), "Each feature must be a dictionary"

        # Check all expected stats keys are present
        assert set(expected_keys).issubset(feature_dict.keys()), "Missing keys in feature stats"

        # Check all values are floats
        for key in expected_keys:
            assert isinstance(feature_dict[key], float), f"{key} should be a float"

if __name__ == "__main__":
    pytest.main([__file__])
