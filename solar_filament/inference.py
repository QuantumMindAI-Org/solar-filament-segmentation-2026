"""Loading verified weights and turning an image into an 8-bit probability map."""
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from .model import STRIDE, TILE, UNet, tiled_probability

SHA256_LENGTH = 64


def require(ok, code):
    """Raise ValueError(code) unless ok."""
    if not ok:
        raise ValueError(code)


def is_sha256(value):
    """True for a 64-character lowercase hexadecimal string."""
    return (isinstance(value, str) and len(value) == SHA256_LENGTH
            and all(char in "0123456789abcdef" for char in value))


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def read_verified(path, expected_sha256):
    """Bytes of a weights file, refused unless its sha256 equals expected_sha256."""
    require(is_sha256(expected_sha256), "WEIGHTS_SHA_FORMAT")
    file = Path(path)
    require(file.is_file() and not file.is_symlink(), "WEIGHTS_PATH")
    raw = file.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, "WEIGHTS_SHA_MISMATCH")
    return raw


def load_model(path, expected_sha256, device="cpu"):
    """A U-Net in eval mode with the weights of path (hash checked before torch reads them)."""
    raw = read_verified(path, expected_sha256)
    model = UNet()
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    return model.to(device).eval()


def probability_map(model, image, device="cpu", tile=TILE, stride=STRIDE):
    """Float32 probability map of a uint8 image: 512 tiles, stride 384, overlaps averaged.

    tile and stride keep the values the weights were trained and validated with; only the tests
    pass smaller ones, so that a synthetic image can be small.
    """
    return tiled_probability(model, image, device=device, tile=tile, stride=stride)


def quantise(probability):
    """8-bit map, round(255 * p). The kernels stored maps this way (numpy.rint, clipped)."""
    arr = np.asarray(probability, np.float64)
    require(arr.ndim == 2 and np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1, "PROBABILITY")
    return np.clip(np.rint(255.0 * arr), 0, 255).astype(np.uint8)
