"""Synthetic inputs for the tests: small images, a tiny random-weights U-Net, a throwaway candidate.

No competition image or mask is used or needed.
"""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).absolute().parent.parent


def _have(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


HAVE_PYCOCOTOOLS = _have("pycocotools")
needs_pycocotools = unittest.skipUnless(HAVE_PYCOCOTOOLS, "needs pycocotools, the RLE encoder (requirements.txt)")
# Code run first in a child process to make "import pycocotools" fail, as on a machine without it.
BLOCK_PYCOCOTOOLS = (
    "import sys\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'pycocotools' or name.startswith('pycocotools.'):\n"
    "            raise ModuleNotFoundError(\"No module named 'pycocotools'\", name='pycocotools')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
    "for _name in [n for n in sys.modules if n == 'pycocotools' or n.startswith('pycocotools.')]:\n"
    "    del sys.modules[_name]\n")


def symlink_or_skip(test, link, target):
    """Create a symbolic link, or skip the test where the system does not allow it (Windows without the privilege)."""
    try:
        Path(link).symlink_to(target)
    except (OSError, NotImplementedError) as error:
        test.skipTest("cannot create a symbolic link here (%s); this test needs one" % error)
IMAGE_A = "20150125172714Mh"
IMAGE_B = "20150125172814Mh"


def synthetic_image(size=(64, 64), seed=0):
    """A gradient with two bright bars, as a uint8 array."""
    rng = np.random.RandomState(seed)
    rows, cols = size
    image = (np.linspace(0, 120, cols)[None, :] + np.zeros((rows, 1))).astype(np.uint8)
    image[rows // 4: rows // 4 + 3, 5: cols - 5] = 230
    image[3 * rows // 4: 3 * rows // 4 + 3, 8: cols // 2] = 200
    image = image + rng.randint(0, 5, size=size).astype(np.uint8)
    return image


def write_jpeg(path, image):
    Image.fromarray(image, mode="L").save(path, format="JPEG", quality=95)
    return Path(path)


def random_weights(path, seed=1):
    """Save the state dict of a randomly initialised package U-Net; return its sha256."""
    from solar_filament.model import UNet
    torch.manual_seed(seed)
    model = UNet()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_candidate(folder, models, rule=(0.15, 0.8, 4), expected=None, status="REHEARSAL_TEST"):
    """candidate.json in folder for models = [(name, relative file, sha256)]."""
    payload = {"schema": "solar-filament-candidate/v1", "status": status,
               "models": [{"name": n, "file": f, "sha256": h} for n, f, h in models],
               "ensemble": "integer_mean_half_up",
               "rule": {"T_low": rule[0], "T_high": rule[1], "A": rule[2]},
               "expected_csv_sha256": expected}
    path = Path(folder) / "candidate.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))
    return path
