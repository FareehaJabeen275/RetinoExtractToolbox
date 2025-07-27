import numpy as np
import pytest
import sys
import os

# Add the root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.HOG_features import extract_hog_features  

def test_extract_hog_features_basic():
    # Create a simple synthetic grayscale image (e.g., gradient)
    test_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    
    # Call without visualization
    stats = extract_hog_features(test_img, visualize=False)
    
    # Check stats is a dict
    assert isinstance(stats, dict)
    
    # Check keys present
    expected_keys = ['feature_length', 'mean', 'variance', 'standard_deviation', 'kurtosis', 'energy']
    for key in expected_keys:
        assert key in stats
    
    # Check numeric values are finite
    for key in expected_keys:
        val = stats[key]
        assert np.isfinite(val), f"{key} is not finite"
    
    # Check feature length is positive
    assert stats['feature_length'] > 0

def test_extract_hog_features_visualize():
    test_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    
    stats, hog_img = extract_hog_features(test_img, visualize=True)
    
    # Stats checks (same as above)
    assert isinstance(stats, dict)
    for key in stats:
        assert np.isfinite(stats[key])
    
    # hog_img should be a numpy array and have 2 dimensions
    assert isinstance(hog_img, np.ndarray)
    assert hog_img.ndim == 2
    
    # Optional: check hog_img shape is similar to test_img shape (HOG visualization can differ)
    # Here we allow some tolerance in size difference
    assert abs(hog_img.shape[0] - test_img.shape[0]) < 20
    assert abs(hog_img.shape[1] - test_img.shape[1]) < 20
