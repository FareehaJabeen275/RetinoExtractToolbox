import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.chip_histogram_features import extract_chip_histogram_features

def test_chip_histogram_features_extraction():
    # Dummy grayscale image (100x100 gradient)
    dummy_image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))

    # Define a chip region (e.g., 20x20 starting at x=10, y=10)
    chip_coords = (10, 10, 20, 20)

    # Call the chip histogram feature extractor
    features = extract_chip_histogram_features(dummy_image, chip_coords)

    # Check: output must be a dictionary
    assert isinstance(features, dict), "Output must be a dictionary"

    # Required feature keys
    expected_keys = {
        'Mean', 'Standard Deviation', 'Variance',
        'Skewness', 'Kurtosis', 'Entropy', 'Energy'
    }

    # Check if all expected keys are present
    assert expected_keys.issubset(features.keys()), "Some keys are missing in the features"

    # Accept Python and NumPy numeric types
    numeric_types = (int, float, np.integer, np.floating)

    for key in expected_keys:
        assert isinstance(features[key], numeric_types), f"{key} is not a numeric type"
