import sys
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.Haralick_features1 import extract_haralick_features

def test_extract_haralick_features():
    # Create a simple test image
    test_img = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))
    
    features = extract_haralick_features(test_img, distances=[1], angles=[0])
    
    # Assert the output is a numpy array with expected length
    assert isinstance(features, np.ndarray)
    assert features.size == 6  # 6 features for 1 distance and 1 angle
    
    # Assert no NaNs or Infs
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
