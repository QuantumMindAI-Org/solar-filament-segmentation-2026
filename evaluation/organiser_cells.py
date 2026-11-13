"""The organisers' own overlap and panoptic-quality code, loaded unchanged, for self-evaluation.

organiser_cell3.py.txt and organiser_cell4.py.txt are two cells of the organisers' public
"Self Evaluation Notebook" (Apache-2.0, see NOTICE). They are kept as text files so their bytes
stay the organisers' bytes; this loader checks their sha256 and executes them in a namespace that
holds the names the cells expect (torch, np, pd, mask_util).

The cells expect:
  ground truth    a CSV with filament_id = "<annotator_id>-<image_id>_<n>" and segmentation_rle
  predictions     the submission CSV: filament_id = "<image_id>_<n>" and segmentation_rle
and 2048 x 2048 masks. Only images present in the ground truth are scored.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).absolute().parent
CELLS = {
    "organiser_cell3.py.txt": "e5f5d9037f00413778ced81b2b05997beb5c27234f288d2131142a1810cef545",
    "organiser_cell4.py.txt": "c9dd26fc9a5d863ed59776d76da88b8d417d00ec610208664b69a4fda19360d9",
}


def load_cells(folder=HERE):
    """Namespace with get_overlap_df and get_pq_score. A cell whose hash differs is refused."""
    import numpy as np
    import pandas as pd
    import torch
    from pycocotools import mask as mask_util
    namespace = {"torch": torch, "np": np, "pd": pd, "mask_util": mask_util}
    for name, expected in CELLS.items():
        raw = (Path(folder) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("ORGANISER_CELL_HASH:" + name)
        # exec runs only after the sha256 check above: the bytes are the organisers' two cells, unchanged.
        exec(compile(raw, name, "exec"), namespace)
    return namespace


def evaluate(predictions_csv, labels_csv, folder=HERE):
    """Panoptic quality of a submission CSV against a labelled CSV, by the organisers' code."""
    import pandas as pd
    cells = load_cells(folder)
    truth = pd.read_csv(labels_csv)
    predicted = pd.read_csv(predictions_csv)
    image_ids = set(truth["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1])
    predicted = predicted[predicted["filament_id"].str.split("_", n=1).str[0].isin(image_ids)]
    overlap = cells["get_overlap_df"](truth, predicted)
    return {"pq": float(cells["get_pq_score"](overlap)), "annotator_images": int(len(overlap)),
            "predicted_filaments": int(len(predicted))}
