import numpy as np
import pytest
import sys
import os

# Add the root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.first_order_statistics import extract_first_order_statistics

def test_first_order_statistics_extraction():
    # Create a dummy grayscale image (gradient)
    dummy_image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

    # Extract statistics
    features = extract_first_order_statistics(dummy_image)

    # Check: output must be a dictionary
    assert isinstance(features, dict), "Output must be a dictionary"

    # Expected feature keys
    expected_keys = {
        'mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis',
        'entropy', 'energy', 'maximum', 'minimum', 'range', 'median'
    }

    # Check if all expected keys are present
    assert expected_keys.issubset(features.keys()), "Some keys are missing in the features"

    # Check that all values are numeric (float/int/numpy types)
    for key in expected_keys:
        assert isinstance(features[key], (int, float, np.integer, np.floating)), f"{key} is not numeric"

    # Optional: sanity check for specific values
    assert features['minimum'] == 0, "Minimum value should be 0"
    assert features['maximum'] == 255, "Maximum value should be 255"
