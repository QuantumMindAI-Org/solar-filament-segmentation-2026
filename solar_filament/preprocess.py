"""Reading H-alpha images. This is the only preprocessing the model sees.

Training and inference both feed the network single-channel images scaled to [0, 1] by dividing the
8-bit values by 255. The division is done inside model.tiled_probability (and in the training data
loader), so this module's job is to read the file, check that it is the expected kind of image, and
return the raw uint8 array. There is no normalisation, no resizing and no cropping beyond the
512 x 512 tiles that model.tiled_probability cuts.
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image

HEIGHT = WIDTH = 2048
IMAGE_ID = re.compile(r"[0-9]{14}[A-Za-z]{2}\Z")
OS_METADATA_FILES = {".ds_store", "thumbs.db", "desktop.ini"}   # compared in lower case


def require(ok, code):
    """Raise ValueError(code) unless ok. One-line checks keep every refusal testable."""
    if not ok:
        raise ValueError(code)


def list_images(images_dir):
    """Sorted (image_id, path) pairs for the .jpeg files of a directory.

    Every entry must be a regular .jpeg file named <14 digits><2 letters>.jpeg. Anything else
    (a symlink, a sub-folder, another suffix) is refused rather than skipped, so a half-empty
    or mixed directory cannot silently produce a short submission. The only entries ignored are
    well-known operating-system metadata: .DS_Store, Thumbs.db, desktop.ini (any case) and a
    __MACOSX folder. Every refusal names the offending entry.
    """
    root = Path(images_dir)
    require(root.is_dir() and not root.is_symlink(), "IMAGES_DIR:" + str(images_dir))
    chosen = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name.lower() in OS_METADATA_FILES or path.name == "__MACOSX":
            continue
        ok = (not path.is_symlink() and path.is_file() and path.suffix == ".jpeg"
              and IMAGE_ID.fullmatch(path.stem) is not None)
        require(ok, "STRAY_FILE:" + path.name)
        chosen.append((path.stem, path))
    require(chosen, "NO_IMAGES:" + str(images_dir))
    return chosen


def load_halpha(path, size=(HEIGHT, WIDTH)):
    """One H-alpha image as a uint8 array of shape size (grayscale JPEG, 2048 x 2048)."""
    with Image.open(path) as photo:
        require(photo.format == "JPEG" and photo.mode == "L" and photo.size == (size[1], size[0]),
                "IMAGE_FORMAT:" + Path(path).name)
        image = np.array(photo, dtype=np.uint8, copy=True)
    return image
