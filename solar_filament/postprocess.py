"""Hysteresis threshold, 8-connected labelling and minimum area on an 8-bit map.

The rule is (T_low, T_high, A). Thresholds act on the 8-bit values: p >= T is value >= ceil(255*T).
A component of {value >= cut(T_low)} is kept when it contains a pixel with value >= cut(T_high) and
its area is at least A pixels. Components are numbered in scipy's labelling order and each becomes
one predicted filament.
"""
import math

import numpy as np
from scipy import ndimage

STRUCTURE = np.ones((3, 3), np.uint8)   # 8-connectivity
MISSING_PACKAGE = ("MISSING_PACKAGE:pycocotools is not installed; it writes the RLE of each filament. "
                   "Install the packages of requirements.txt (pip install -r requirements.txt).")


def mask_module():
    """pycocotools.mask, imported only where an RLE is written, so that the rest of the package and
    reproduce.py --help work without it. A missing package is a refusal with a one-line message."""
    try:
        from pycocotools import mask as mask_util
    except ImportError:
        raise ValueError(MISSING_PACKAGE) from None
    return mask_util


def cut(threshold):
    """8-bit cut for a probability threshold: the smallest 8-bit value v with v >= 255 * T.

    It is computed as ceil(255 * T - 1e-9). The 1e-9 is subtracted so that a product that should be a
    whole number but lands a hair above it in binary floating point gives that whole number, not the
    next one. For the thresholds used here it changes nothing: 255 * 0.15 = 38.25 -> 39,
    255 * 0.5 = 127.5 -> 128, 255 * 0.8 = 204.0 -> 204.
    """
    return int(math.ceil(255 * threshold - 1e-9))


def label8(mask):
    """Labels and count of the 8-connected components of mask > 0."""
    return ndimage.label(mask > 0, structure=STRUCTURE)


def hysteresis_state(values, t_low, t_high):
    """Labels of {values >= cut(t_low)} and each component's area, or -1 if it has no seed.

    A seed is a pixel with values >= cut(t_high).
    """
    labels, count = label8(values >= cut(t_low))
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    seeded = np.zeros(count + 1, bool)
    seeded[np.unique(labels[values >= cut(t_high)])] = True
    seeded[0] = False
    areas[~seeded[1:]] = -1
    return labels, areas


def instance_rows(image_id, values, rule):
    """(filament_id, RLE) rows for one image: hysteresis, then every component with area >= A.

    rule = (t_low, t_high, minimum_area). The id suffix is 1-based and zero-padded to 4 digits.
    The RLE is the compressed COCO string for a 2048 x 2048 mask in column-major order.
    """
    t_low, t_high, minimum_area = rule
    if not (0 < t_low <= t_high <= 1 and isinstance(minimum_area, int) and minimum_area > 0):
        raise ValueError("RULE")
    if values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError("MAP_DTYPE")
    labels, areas = hysteresis_state(values, t_low, t_high)
    mask_util = mask_module()
    rows = []
    kept = 0
    for label in range(1, len(areas) + 1):
        if areas[label - 1] >= minimum_area:
            kept += 1
            encoded = mask_util.encode(np.asfortranarray(labels == label, dtype=np.uint8))
            rows.append((image_id + "_" + str(kept).zfill(4), encoded["counts"].decode("ascii")))
    return rows
