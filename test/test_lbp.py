# test/test_lbp.py

import numpy as np
import cv2
import pytest
from features import LBP_features  # assuming your LBP function is inside features/LBP_features.py

def test_lbp_feature_extraction():
    # 1. Dummy grayscale image (100x100 pixels with gradient)
    dummy_image = np.tile(np.arange(0, 100, dtype=np.uint8), (100, 1))

    # 2. Apply LBP feature extractor
    lbp_feats = LBP_features.extract_lbp_features(dummy_image)

    # 3. Check type and length
    assert isinstance(lbp_feats, (list, np.ndarray)), "Output should be list or numpy array"
    assert len(lbp_feats) > 0, "LBP features should not be empty"
