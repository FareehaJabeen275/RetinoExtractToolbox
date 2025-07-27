import numpy as np
import pytest
import sys
import os

# Add parent directory to sys.path to import your wavelet features module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.wavelet_features import extract_wavelet_features

def test_extract_wavelet_features_output_keys_and_types():
    # Create a dummy grayscale image (64x64) with random values
    test_img = np.random.randint(0, 256, (64, 64)).astype(np.float32)
    
    features = extract_wavelet_features(test_img, wavelet='db1', level=2)
    
    # The output should be a dict
    assert isinstance(features, dict)
    
    # There should be keys starting with 'coeff_'
    assert any(key.startswith('coeff_') for key in features.keys())
    
    # All feature values should be finite numbers
    for key, val in features.items():
        assert np.isfinite(val), f"Feature {key} is not finite: {val}"

def test_extract_wavelet_features_values_reasonable():
    test_img = np.ones((64, 64), dtype=np.float32) * 128  # Constant image
    
    features = extract_wavelet_features(test_img)
    
    # In a constant image, energy should be positive and entropy should be close to zero
    energy_keys = [k for k in features if 'energy' in k]
    entropy_keys = [k for k in features if 'entropy' in k]
    
    for ek in energy_keys:
        assert features[ek] >= 0
    
    for entk in entropy_keys:
        # Entropy of constant image should be close to zero
        assert features[entk] < 0.1

@pytest.mark.parametrize("wavelet", ['db1', 'haar', 'sym2'])
def test_extract_wavelet_features_different_wavelets(wavelet):
    test_img = np.random.rand(64, 64).astype(np.float32)
    features = extract_wavelet_features(test_img, wavelet=wavelet, level=1)
    assert isinstance(features, dict)
    assert len(features) > 0

