import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.glrlm_features import extract_glrlm_features

def test_extract_glrlm_features():
    # Create a dummy grayscale image (normalized float between 0 and 1)
    dummy_image = np.tile(np.linspace(0, 1, 64), (64, 1)).astype(np.float32)

    features = extract_glrlm_features(dummy_image, num_levels=64)

    # Check output type
    assert isinstance(features, dict), "Output should be a dictionary"

    expected_keys = ['SRE', 'LRE', 'GLN', 'RLN', 'LGRE', 'HGRE']

    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
        assert isinstance(features[key], (float, np.floating, int)), f"{key} should be numeric"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
