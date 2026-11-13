#!/usr/bin/env python3
"""Write reproduce.ipynb (nbformat 4.4 JSON) without Jupyter, and check its structure.

The notebook is thin: every code cell calls a function of reproduce.py, stage by stage, with a short
markdown cell before it. Parameters come from environment variables so the same notebook runs
interactively (defaults) and unattended (run_notebook_cells.py sets them):

  SOLAR_IMAGES_DIR  folder of test images             (default: test_images)
  SOLAR_OUT         CSV to write                      (default: submission.csv)
  SOLAR_DEVICE      cpu | cuda | auto                 (default: cpu)
  SOLAR_THREADS     CPU threads                       (default: 4, the count of our rehearsals)
  SOLAR_LIMIT       first N images only (smoke tests) (default: all)
  SOLAR_SUBMITTED_CSV  the submitted CSV, to report differences against (default: the one in candidate.json, if any)
  SOLAR_LABELS      ground-truth CSV for the optional last cell (default: none)

  python -B make_notebook.py [--out PATH]        write the notebook
  python -B make_notebook.py --check PATH        validate an existing notebook's structure
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
# In the release tooling folder the notebook source lives in repo_src/; inside the published repository this
# file is tools/make_notebook.py and the notebook is at the repository root, one folder up.
DEFAULT_OUT = (HERE / "repo_src" / "reproduce.ipynb") if (HERE / "repo_src").is_dir() else (HERE.parent / "reproduce.ipynb")

OVERVIEW = """# Reproduce the submitted predictions

This notebook runs the whole pipeline of our entry to the Solar Filament Segmentation Challenge 2026 on a
folder of H-alpha test images, using the weights stored in this repository. Nothing is trained or downloaded.
Every cell calls a function of `reproduce.py`; the cells are the stages of the pipeline:

```
H-alpha JPEG (2048 x 2048, grayscale)
   |  1. candidate.json        weights files + sha256, ensemble formula, rule (T_low, T_high, minimum area)
   |  2. load_models           each weights file is hash-checked, then loaded into the U-Net
   |  3. predict_maps          512 x 512 tiles, stride 384, sigmoid, averaged overlaps -> 8-bit map per model
   |  4. ensemble              integer mean of the models' maps, rounded half up
   |  5. find_instances        hysteresis threshold (T_low, T_high), 8-connected, minimum area A
   |  6. write_submission      CSV: filament_id, segmentation_rle (COCO RLE counts)
   v  7. check_expected        compare the CSV's sha256 with candidate.json
```

Run it from the repository root with `requirements-notebook.txt` installed. Set `SOLAR_IMAGES_DIR` to the folder
of test images (or edit the next cell); the next cell stops with a plain message if that folder does not exist.
Optional variables: `SOLAR_OUT` (CSV to write, default `submission.csv`), `SOLAR_DEVICE` (cpu, cuda or auto),
`SOLAR_THREADS` (CPU threads, default 4 as in our rehearsals),
`SOLAR_LIMIT` (first N images only), `SOLAR_SUBMITTED_CSV` (the submitted CSV, to report differences against),
`SOLAR_LABELS` (a labelled CSV for the optional last cell)."""

CELLS = [
    ("markdown", OVERVIEW),
    ("code", """import os
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "reproduce.py").is_file():
    raise FileNotFoundError("Start Jupyter in the repository root (the folder with reproduce.py); "
                            "the current folder is " + str(REPO))
sys.path.insert(0, str(REPO))
import reproduce

IMAGES_DIR = os.environ.get("SOLAR_IMAGES_DIR", "test_images")
if not Path(IMAGES_DIR).is_dir():
    raise FileNotFoundError("The images folder " + repr(IMAGES_DIR) + " does not exist. Set the environment variable "
                            "SOLAR_IMAGES_DIR to the folder of test images before starting Jupyter, or edit IMAGES_DIR in this cell.")
OUT = os.environ.get("SOLAR_OUT", "submission.csv")
DEVICE = reproduce.resolve_device(os.environ.get("SOLAR_DEVICE", "cpu"))
THREADS = reproduce.set_threads(int(os.environ.get("SOLAR_THREADS") or reproduce.DEFAULT_THREADS))
LIMIT = int(os.environ["SOLAR_LIMIT"]) if os.environ.get("SOLAR_LIMIT") else None
print("images:", IMAGES_DIR, "| output:", OUT, "| device:", DEVICE, "| threads:", THREADS, "| limit:", LIMIT)"""),
    ("markdown", """## 1. The candidate

`candidate.json` is the single description of what was submitted: which weights files (with sha256), how their
maps are combined, and the post-processing rule."""),
    ("code", """candidate = reproduce.read_candidate(REPO / "candidate.json")
rule = reproduce.rule_of(candidate)
print("status:", candidate["status"], "| models:", [m["name"] for m in candidate["models"]])
print("ensemble:", candidate["ensemble"], "| rule (T_low, T_high, minimum area):", rule)"""),
    ("markdown", """## 2. The input images

The folder must hold only `<14 digits><2 letters>.jpeg` files; anything else is refused, so a short or
mixed folder cannot silently give a short submission."""),
    ("code", """listing = reproduce.list_inputs(IMAGES_DIR, LIMIT)
print(len(listing), "images, first:", listing[0][0])"""),
    ("markdown", """## 3. Load the models

Each weights file's sha256 is compared with `candidate.json` before PyTorch reads it."""),
    ("code", """models = reproduce.load_models(candidate, REPO, DEVICE)
print("loaded:", [name for name, _ in models])"""),
    ("markdown", """## 4. Probability maps

Per image: read it, run the tiled U-Net of every model and quantise each map to 8 bits, `round(255 p)`.
The only preprocessing is the division by 255 inside the tiling function. This is the slow cell (about 13 s per
image per model on four CPU threads). Images are read one at a time; the maps are kept."""),
    ("code", """from solar_filament import preprocess

maps = {}
for image_id, path in listing:
    image = preprocess.load_halpha(path)
    maps[image_id] = reproduce.predict_maps(image, models, DEVICE)
first = maps[listing[0][0]][0]
print(len(maps), "images;", "first map:", first.shape, first.dtype, "min", int(first.min()), "max", int(first.max()))"""),
    ("markdown", """## 5. Ensemble

The integer mean of the models' 8-bit maps, rounded half up: `E = (2*S + n) // (2*n)`. With one model the
map is unchanged."""),
    ("code", """ensembled = {image_id: reproduce.ensemble(per_model) for image_id, per_model in maps.items()}
print("ensembled maps:", len(ensembled))"""),
    ("markdown", """## 6. Instances

Hysteresis with the candidate's rule: connected components (8-neighbours) of {map >= cut(T_low)} that contain a
pixel >= cut(T_high) and have at least A pixels; each becomes one filament, encoded as COCO RLE."""),
    ("code", """rows = []
for image_id, _ in listing:
    rows.extend(reproduce.find_instances(image_id, ensembled[image_id], rule))
print(len(rows), "filaments in", len(listing), "images")"""),
    ("markdown", """## 7. Write the submission and compare

The CSV has the organisers' two columns. Its sha256 is compared with `expected_csv_sha256` of `candidate.json`
(a run limited to the first N images cannot match and says so). If the hashes differ the cell says so plainly and,
when the submitted CSV is available (named by `submitted_csv` in `candidate.json`, which only a final build has, or
given in `SOLAR_SUBMITTED_CSV`), reports how many
images and instances differ and the organisers' PQ of our CSV against the submitted one. A different CPU or GPU can
change an 8-bit probability value by one unit in a few pixels (see README); nothing here forces equality."""),
    ("code", """digest = reproduce.write_submission(rows, OUT)
status = reproduce.check_expected(digest, candidate, partial=LIMIT is not None)
print("csv:", OUT, "| sha256:", digest)
print("comparison with candidate.json:", status)
SUBMITTED = os.environ.get("SOLAR_SUBMITTED_CSV") or reproduce.submitted_path(candidate, REPO / "candidate.json")
if status == "MISMATCH":
    if SUBMITTED:
        difference = reproduce.compare_instances(OUT, SUBMITTED)
        print("\\n".join(reproduce.describe_difference(difference)))
    else:
        print("THE CSV DIFFERS FROM THE SUBMITTED ONE (sha256 mismatch). No submitted CSV is available to compare instances.")
elif status == "MATCH":
    print("Identical to the submitted CSV (same sha256).")
elif status == "NO_EXPECTED_VALUE":
    print("candidate.json has no expected CSV hash (a rehearsal build): nothing to compare against.")"""),
    ("markdown", """## Optional: score a labelled set

If you have ground truth (a CSV with `filament_id` = `<annotator>-<image id>_<n>` and `segmentation_rle`), set
`SOLAR_LABELS` to its path (or edit `LABELS`) and run this cell: it applies the organisers' own overlap and
panoptic-quality code (`evaluation/`, Apache-2.0, see `NOTICE`) to the CSV written above, on the images present
in that file. It needs `pandas`. Left unset, the cell does nothing."""),
    ("code", """LABELS = os.environ.get("SOLAR_LABELS")
if LABELS:
    sys.path.insert(0, str(REPO / "evaluation"))
    import organiser_cells
    print(organiser_cells.evaluate(OUT, LABELS, REPO / "evaluation"))
else:
    print("no labelled set given; skipped")"""),
]


def source_lines(text):
    """nbformat stores a cell's source as a list of lines, each but the last ending in a line feed."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build_notebook():
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source_lines(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def validate_notebook(notebook):
    """The nbformat 4 essentials. Returns a list of problems; empty means the structure is valid."""
    problems = []
    if not isinstance(notebook, dict):
        return ["notebook is not an object"]
    for key in ("cells", "metadata", "nbformat", "nbformat_minor"):
        if key not in notebook:
            problems.append("missing top-level key " + key)
    if notebook.get("nbformat") != 4:
        problems.append("nbformat is not 4")
    if not isinstance(notebook.get("nbformat_minor"), int) or notebook.get("nbformat_minor", 0) > 4:
        problems.append("nbformat_minor must be an int <= 4 (4.5 needs cell ids)")
    metadata = notebook.get("metadata")
    kernel = metadata.get("kernelspec") if isinstance(metadata, dict) else None
    if not (isinstance(kernel, dict) and {"display_name", "language", "name"} <= set(kernel)):
        problems.append("metadata.kernelspec needs display_name, language, name")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        problems.append("cells must be a non-empty list")
        return problems
    for index, cell in enumerate(cells):
        where = "cell %d: " % index
        if not isinstance(cell, dict):
            problems.append(where + "not an object")
            continue
        kind = cell.get("cell_type")
        if kind not in ("code", "markdown", "raw"):
            problems.append(where + "bad cell_type")
        if not isinstance(cell.get("metadata"), dict):
            problems.append(where + "metadata must be an object")
        source = cell.get("source")
        if not (isinstance(source, str) or (isinstance(source, list) and all(isinstance(s, str) for s in source))):
            problems.append(where + "source must be a string or a list of strings")
        if kind == "code":
            if "execution_count" not in cell or not (cell["execution_count"] is None
                                                      or isinstance(cell["execution_count"], int)):
                problems.append(where + "code cell needs execution_count (null or int)")
            if not isinstance(cell.get("outputs"), list):
                problems.append(where + "code cell needs outputs list")
        elif "outputs" in cell or "execution_count" in cell:
            problems.append(where + "only code cells carry outputs and execution_count")
        if "id" in cell:
            problems.append(where + "ids belong to nbformat 4.5")
    return problems


def write_notebook(path):
    notebook = build_notebook()
    problems = validate_notebook(notebook)
    if problems:
        raise ValueError("NOTEBOOK_INVALID:" + "; ".join(problems))
    # bytes, so that the file has LF line ends and UTF-8 on every platform
    Path(path).write_bytes((json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check")
    args = parser.parse_args(argv)
    if args.check:
        problems = validate_notebook(json.loads(Path(args.check).read_text(encoding="utf-8")))
        print(json.dumps({"problems": problems}))
        return 1 if problems else 0
    write_notebook(args.out)
    print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
