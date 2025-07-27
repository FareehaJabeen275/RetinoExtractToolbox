import numpy as np
import cv2
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.hu_momts import extract_hu_moments  # Adjust path as needed

def test_extract_hu_moments_output_shape_and_type():
    # Create a simple binary test image with a white square on black background
    test_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(test_img, (30, 30), (70, 70), 255, -1)

    hu_features = extract_hu_moments(test_img)

    # Check output type and shape
    assert isinstance(hu_features, np.ndarray), "Output should be a numpy array"
    assert hu_features.shape == (7,), "Hu moments output should have length 7"

    # Check no NaN or infinite values
    assert not np.isnan(hu_features).any(), "Output contains NaN values"
    assert not np.isinf(hu_features).any(), "Output contains infinite values"

def test_extract_hu_moments_color_image():
    # Create a color image by stacking grayscale image into 3 channels
    gray_img = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(gray_img, (25, 25), 10, 255, -1)
    color_img = np.stack([gray_img]*3, axis=-1)

    hu_features = extract_hu_moments(color_img)

    # Check output shape and type again
    assert isinstance(hu_features, np.ndarray)
    assert hu_features.shape == (7,)
    assert not np.isnan(hu_features).any()
    assert not np.isinf(hu_features).any()
