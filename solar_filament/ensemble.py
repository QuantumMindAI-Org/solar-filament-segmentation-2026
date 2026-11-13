"""Ensemble of 8-bit probability maps."""
import numpy as np

FORMULA = "E = (2*S + n) // (2*n)"
FORMULA_NAME = "integer_mean_half_up"


def ensemble_maps(stacks):
    """Integer mean of uint8 maps, rounded half up: E = (2*S + n) // (2*n).

    S is the sum of the n maps. A mean of k + 0.5 becomes k + 1. For n = 1 the map is unchanged.
    """
    if len(stacks) < 1:
        raise ValueError("NO_MAPS")
    shape = stacks[0].shape
    if not all(arr.shape == shape and arr.dtype == np.uint8 for arr in stacks):
        raise ValueError("SHAPE")
    total = np.zeros(shape, dtype=np.int64)
    for arr in stacks:
        total += arr
    n = len(stacks)
    return ((2 * total + n) // (2 * n)).astype(np.uint8)
