import sys
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.histogram_based_statistics import extract_histogram_features


def test_extract_histogram_features():
    # Simple gradient image for test
    test_img = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

    features = extract_histogram_features(test_img)

    expected_keys = ['mean', 'variance', 'skewness', 'kurtosis', 'entropy', 'energy',
                     'contrast', 'peaks', 'spread', 'mode', 'median']
    for key in expected_keys:
        assert key in features

    for key in features:
        val = features[key]
        if isinstance(val, np.ndarray):
            assert not np.isnan(val).any()
            assert not np.isinf(val).any()
        else:
            assert not np.isnan(val)
            assert not np.isinf(val)
