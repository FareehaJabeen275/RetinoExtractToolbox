import numpy as np
import pytest
from features import LBP_features  # assuming your code is in features/LBP_features.py

def test_lbp_feature_extraction():
    # Dummy grayscale image (simple 100x100 gradient)
    dummy_image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))

    # Call the LBP feature extractor
    features = LBP_features.extract_lbp_features(dummy_image)

    # Check: output must be a dictionary
    assert isinstance(features, dict), "Output must be a dictionary"

    # Required feature keys
    expected_keys = {'Mean', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Entropy'}
    
    # Check all expected keys are present
    assert expected_keys.issubset(features.keys()), "Some keys are missing in the features"

    # Check all values are floats
    for key in expected_keys:
        assert isinstance(features[key], float), f"{key} is not a float"
