import sys
import os
import numpy as np
import pytest

# Add the root directory of your project to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.moments_all import compute_moments  # Adjust 'moments_all' to your actual module filename

def test_compute_moments_grayscale_image():
    img = np.tile(np.arange(10, dtype=np.uint8), (10, 1))
    moments = compute_moments(img)
    expected_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                     'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
    for key in expected_keys:
        assert key in moments
        val = moments[key]
        assert isinstance(val, (int, float, np.floating, np.integer))
        assert np.isfinite(val)

def test_compute_moments_color_image():
    img_color = np.stack([np.tile(np.arange(10, dtype=np.uint8), (10, 1))]*3, axis=2)
    moments = compute_moments(img_color)
    expected_keys = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                     'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
    for key in expected_keys:
        assert key in moments
        val = moments[key]
        assert isinstance(val, (int, float, np.floating, np.integer))
        assert np.isfinite(val)
