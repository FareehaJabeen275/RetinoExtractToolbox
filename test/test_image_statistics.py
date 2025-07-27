import sys
import os
import numpy as np
import pytest

# Add project root directory to sys.path so that "features" module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.image_statistics import extract_image_statistics  # now import will work

def test_extract_image_statistics_realistic_image():
    # Create a 2D gradient image (64x64)
    test_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))

    stats = extract_image_statistics(test_img)

    # Check the returned dict keys
    expected_keys = ['mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis',
                     'entropy', 'energy', 'maximum', 'minimum', 'range', 'median']
    for key in expected_keys:
        assert key in stats
        val = stats[key]
        assert val is not None
        assert np.isfinite(val), f"{key} should be finite"

    # Additional sanity checks
    assert stats['maximum'] == 63
    assert stats['minimum'] == 0
    assert stats['range'] == 63
    assert stats['median'] == 31.5  # since 0 to 63 evenly spread

if __name__ == "__main__":
    pytest.main()
