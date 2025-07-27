import numpy as np
import pytest
import sys
import os

# Add parent directory to sys.path to import your zernike module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.zernike_momts import extract_zernike_moments  # Adjust import path as needed

def test_extract_zernike_moments_output_type_and_length():
    # Create a dummy grayscale image (64x64) with random values
    test_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    moments = extract_zernike_moments(test_img, radius=21)

    # Output should be a numpy array
    assert isinstance(moments, np.ndarray)

    # Zernike moments length depends on radius, but mahotas returns fixed length for radius=21 (usually 25 moments)
    assert moments.ndim == 1
    assert len(moments) > 0
    # Typical length for radius=21 is 25, but allow flexible length
    assert len(moments) <= 30

def test_extract_zernike_moments_on_constant_image():
    # Constant image should return moments array (no error)
    test_img = np.ones((64, 64), dtype=np.uint8) * 128
    moments = extract_zernike_moments(test_img, radius=21)
    assert isinstance(moments, np.ndarray)
    assert len(moments) > 0
    # All moments should be finite numbers
    assert np.all(np.isfinite(moments))

def test_extract_zernike_moments_with_color_image():
    # Create a dummy color image (64x64x3)
    test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    moments = extract_zernike_moments(test_img, radius=21)
    assert isinstance(moments, np.ndarray)
    assert len(moments) > 0
