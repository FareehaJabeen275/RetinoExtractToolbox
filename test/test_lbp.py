import numpy as np
import pytest
import sys
import os

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.LBP_features import extract_lbp_features  # ✅ Correct function import


def test_lbp_feature_extraction():
    # Dummy grayscale image (simple 100x100 gradient)
    dummy_image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))

    # Call the LBP feature extractor
    features = extract_lbp_features(dummy_image)  # ✅ Call directly

    # Check: output must be a dictionary
    assert isinstance(features, dict), "Output must be a dictionary"

    # Required feature keys
    expected_keys = {'Mean', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis', 'Entropy'}
    
    # Check all expected keys are present
    assert expected_keys.issubset(features.keys()), "Some keys are missing in the features"

    # Check all values are floats
    for key in expected_keys:
        assert isinstance(features[key], float), f"{key} is not a float"
