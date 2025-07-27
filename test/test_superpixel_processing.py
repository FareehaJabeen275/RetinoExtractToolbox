import numpy as np
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend for tests
import matplotlib.pyplot as plt
import pytest
import sys
import os
from unittest.mock import patch

# Add the parent directory where your features folder is located to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.super_pixels_features import extract_superpixels, visualize_superpixels

def test_extract_superpixels_shape_and_type():
    # Create a dummy RGB image (64x64x3)
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    segments = extract_superpixels(test_img, n_segments=50, compactness=20)
    
    # Check output type and shape
    assert isinstance(segments, np.ndarray)
    assert segments.shape == (64, 64)
    
    # Check segment labels are integers and within expected range
    assert segments.dtype.kind in {'i', 'u'}  # integer type
    assert segments.min() >= 0
    assert segments.max() < 50

@patch('matplotlib.pyplot.show')
def test_visualize_superpixels_runs_without_error(mock_show):
    # Create dummy image and segments
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    segments = extract_superpixels(test_img, n_segments=30)
    
    # Call visualize_superpixels, plt.show is mocked to prevent GUI call
    try:
        visualize_superpixels(test_img, segments)
    except Exception as e:
        pytest.fail(f"visualize_superpixels raised an exception: {e}")

    # Assert plt.show() was called once
    mock_show.assert_called_once()
