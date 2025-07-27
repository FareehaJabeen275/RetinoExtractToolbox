import numpy as np
import pytest
import sys
import os

# Add root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.fractal_dimension import extract_fractal_dimension

def test_extract_fractal_dimension():
    # Create a simple binary pattern (checkerboard-style)
    dummy_image = np.zeros((256, 256), dtype=np.uint8)
    dummy_image[::2, ::2] = 255
    dummy_image[1::2, 1::2] = 255

    # Extract feature
    features = extract_fractal_dimension(dummy_image)

    # Check: output must be a dictionary
    assert isinstance(features, dict), "Output must be a dictionary"

    # Check required key
    assert 'fractal_dimension' in features, "Key 'fractal_dimension' missing"

    # Check value type
    assert isinstance(features['fractal_dimension'], (float, np.floating)), "Fractal dimension is not a float"

    # Optional sanity check
    assert 0 < features['fractal_dimension'] < 3, "Fractal dimension is out of expected range (0-3)"
