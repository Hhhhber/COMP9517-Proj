import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def feat_hog(bgr, resize=(96, 96)):
    """HOG feature,return 1D vetex"""
    gray = _to_gray(bgr)
    if resize:
        gray = cv2.resize(gray, resize)
    feat, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )
    return feat.astype(np.float32)

def feat_lbp(bgr, resize=(128, 128), P=8, R=1.0):
    """LBP feature"""
    gray = _to_gray(bgr)
    if resize:
        gray = cv2.resize(gray, resize)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return hist.astype(np.float32)

def extract_feature(bgr, kind="hog"):
    if kind == "hog":
        return feat_hog(bgr)
    elif kind == "lbp":
        return feat_lbp(bgr)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")
