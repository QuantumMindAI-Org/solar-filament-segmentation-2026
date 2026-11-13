#!/usr/bin/env python3
"""Reproduce the submitted predictions from H-alpha images, the stored weights and candidate.json.

Command line (run from the repository root):

    python reproduce.py --images-dir PATH_TO_TEST_IMAGES --out submission.csv --threads 4

One function per stage, called in this order (reproduce.ipynb calls the same functions):

    read_candidate -> list_inputs -> load_models -> predict_maps (per image, per model)
      -> ensemble -> find_instances -> write_submission -> check_expected

Device: CPU unless --device cuda (or auto, which takes CUDA when present). Our rehearsals ran on CPU.
On a GPU, or with other library versions, the 8-bit probability maps can differ from the CPU maps by
one unit in a few pixels, see README.md. Nothing is downloaded and nothing is trained.

The comparison with candidate.json and the exit code:
  MATCH              the CSV's sha256 equals expected_csv_sha256                      exit 0
  MISMATCH           it differs; a plain message says so (and how, when the submitted
                     CSV is available: candidate.json "submitted_csv" or --compare-with)  exit 4
  NO_EXPECTED_VALUE  candidate.json has no expected hash (expected_csv_sha256 null,
                     as in a rehearsal build): nothing can be compared                exit 0
  SKIPPED_PARTIAL    --limit was given: a partial CSV cannot match                    exit 0
A refusal (bad input, wrong weights hash, ...) prints REFUSED <code> and exits 2. So does a missing
pycocotools (REFUSED MISSING_PACKAGE:...), which only the instance stage needs; --help works without it.
candidate.json "status" must be FINAL (then expected_csv_sha256 must be set) or REHEARSAL_<name>; anything
else is refused. The JSON line reports the thread count used ("threads"); see README on threads.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from solar_filament import ensemble as ensemble_module  # noqa: E402
from solar_filament import inference, postprocess, preprocess, submission  # noqa: E402

DEFAULT_CANDIDATE = HERE / "candidate.json"
SCHEMA = "solar-filament-candidate/v1"
FINAL_STATUS = "FINAL"                              # the submitted candidate: needs expected_csv_sha256
REHEARSAL_STATUS = re.compile(r"REHEARSAL_[A-Z0-9_]+\Z")   # a rehearsal candidate, e.g. REHEARSAL_K2
DEFAULT_THREADS = 4                                 # the thread count of our rehearsals (README)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(ok, code):
    """Raise ValueError(code) unless ok."""
    if not ok:
        raise ValueError(code)


# ---------------------------------------------------------------- stage 0: the candidate
def read_candidate(path=DEFAULT_CANDIDATE):
    """candidate.json as a validated dict: models with sha256, ensemble formula, rule, expected CSV hash."""
    payload = json.loads(Path(path).read_bytes())
    require(isinstance(payload, dict) and payload.get("schema") == SCHEMA, "CANDIDATE_SCHEMA")
    models = payload.get("models")
    require(isinstance(models, list) and 1 <= len(models) <= 3, "CANDIDATE_MODELS")
    names = set()
    for item in models:
        require(isinstance(item, dict) and isinstance(item.get("name"), str)
                and isinstance(item.get("file"), str) and inference.is_sha256(item.get("sha256")),
                "CANDIDATE_MODELS")
        require(item["name"] not in names, "CANDIDATE_DUPLICATE_MODEL")
        names.add(item["name"])
        relative = PurePosixPath(item["file"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parts,
                "CANDIDATE_WEIGHTS_PATH")
    require(payload.get("ensemble") == ensemble_module.FORMULA_NAME, "CANDIDATE_ENSEMBLE")
    rule = payload.get("rule")
    require(isinstance(rule, dict) and set(rule) == {"T_low", "T_high", "A"}, "CANDIDATE_RULE")
    low, high, area = rule["T_low"], rule["T_high"], rule["A"]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high))
            and isinstance(area, int) and not isinstance(area, bool)
            and 0 < low <= high <= 1 and area > 0, "CANDIDATE_RULE")
    expected = payload.get("expected_csv_sha256")
    require(expected is None or inference.is_sha256(expected), "CANDIDATE_EXPECTED_SHA")
    status = payload.get("status")
    require(status == FINAL_STATUS or (isinstance(status, str) and REHEARSAL_STATUS.fullmatch(status) is not None),
            "CANDIDATE_STATUS")
    require(status != FINAL_STATUS or expected is not None, "CANDIDATE_FINAL_WITHOUT_EXPECTED_SHA")
    submitted = payload.get("submitted_csv")
    if submitted is not None:
        relative = PurePosixPath(submitted) if isinstance(submitted, str) else None
        require(relative is not None and not relative.is_absolute() and ".." not in relative.parts
                and relative.parts and expected is not None, "CANDIDATE_SUBMITTED_CSV")
    return payload


def rule_of(candidate):
    """(T_low, T_high, A) of a candidate."""
    rule = candidate["rule"]
    return float(rule["T_low"]), float(rule["T_high"]), int(rule["A"])


# ---------------------------------------------------------------- stage 1 and 2: inputs and models
def resolve_device(name="cpu"):
    """'cpu' (default), 'cuda', or 'auto' (CUDA when available, else CPU)."""
    import torch
    require(name in ("cpu", "cuda", "auto"), "DEVICE")
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    require(name == "cpu" or torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    return name


def list_inputs(images_dir, limit=None):
    """Sorted (image_id, path) pairs of the .jpeg files in images_dir; limit keeps the first N (tests only)."""
    listing = preprocess.list_images(images_dir)
    return listing if limit is None else listing[:limit]


def load_models(candidate, base_dir=HERE, device="cpu"):
    """[(name, model)] for the candidate's weights files; each file's sha256 is checked before it is read by torch."""
    models = []
    for item in candidate["models"]:
        path = Path(base_dir) / item["file"]
        models.append((item["name"], inference.load_model(path, item["sha256"], device)))
    return models


# ---------------------------------------------------------------- stage 3 to 5: per image
def predict_maps(image, models, device="cpu"):
    """One 8-bit probability map per model (tiled inference, then round(255 * p))."""
    return [inference.quantise(inference.probability_map(model, image, device)) for _name, model in models]


def ensemble(maps):
    """The integer mean of the models' 8-bit maps, rounded half up."""
    return ensemble_module.ensemble_maps(list(maps))


def find_instances(image_id, ensembled_map, rule):
    """Rows (filament_id, RLE) of the instances kept by the hysteresis rule and minimum area."""
    return postprocess.instance_rows(image_id, ensembled_map, rule)


# ---------------------------------------------------------------- stage 6 and 7: the file
def write_submission(rows, out):
    """Write the CSV; return its sha256."""
    return submission.write_csv(out, rows)


def submitted_path(candidate, candidate_path):
    """The submitted CSV shipped in the repository (candidate.json "submitted_csv"), hash-checked, or None."""
    name = candidate.get("submitted_csv")
    if name is None:
        return None
    path = Path(candidate_path).absolute().parent / name
    require(path.is_file() and inference.sha256_bytes(path.read_bytes()) == candidate["expected_csv_sha256"],
            "SUBMITTED_CSV_HASH")
    return path


def check_expected(csv_sha256, candidate, partial=False):
    """'MATCH', 'MISMATCH', 'NO_EXPECTED_VALUE' or 'SKIPPED_PARTIAL' (a limited run cannot match)."""
    expected = candidate.get("expected_csv_sha256")
    if partial:
        return "SKIPPED_PARTIAL"
    if expected is None:
        return "NO_EXPECTED_VALUE"
    return "MATCH" if csv_sha256 == expected else "MISMATCH"


# ---------------------------------------------------------------- comparison with the submitted CSV
def read_instances(csv_path):
    """{image_id: set of RLE strings} of a submission CSV. Instances are compared by their RLE, not by running number."""
    import csv
    found = {}
    with open(csv_path, newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == list(submission.COLUMNS), "CSV_HEADER")
        for filament_id, rle in reader:
            found.setdefault(filament_id.split("_", 1)[0], set()).add(rle)
    return found


def organiser_pq(ours_csv, submitted_csv):
    """PQ of our CSV scored by the organisers' own code (evaluation/) with the submitted CSV as the reference.

    1.0 means the same instances. Returns None with a reason when pandas or torch is not installed.
    """
    import tempfile
    try:
        sys.path.insert(0, str(HERE / "evaluation"))
        import organiser_cells
        import pandas as pd
    except ImportError as error:
        return {"pq": None, "reason": "not computed: " + str(error)}
    finally:
        sys.path.pop(0)
    frame = pd.read_csv(submitted_csv)
    if frame.empty:
        return {"pq": None, "reason": "not computed: the submitted CSV has no filaments"}
    frame["filament_id"] = "submitted-" + frame["filament_id"]          # the cells expect <annotator>-<image id>_<n>
    with tempfile.TemporaryDirectory() as temp:
        reference = Path(temp) / "reference.csv"
        frame.to_csv(reference, index=False, lineterminator="\n")
        return organiser_cells.evaluate(ours_csv, reference, HERE / "evaluation")


def compare_instances(ours_csv, submitted_csv, with_pq=True):
    """How our CSV differs from the submitted one: per image and overall, by instance (RLE), and in the organisers' PQ."""
    ours, theirs = read_instances(ours_csv), read_instances(submitted_csv)
    images = sorted(set(ours) | set(theirs))
    differing = {}
    for image_id in images:
        a, b = ours.get(image_id, set()), theirs.get(image_id, set())
        if a != b:
            differing[image_id] = {"only_in_ours": len(a - b), "only_in_submitted": len(b - a)}
    count = lambda table: sum(len(v) for v in table.values())
    report = {"images": len(images), "images_identical": len(images) - len(differing),
              "instances_ours": count(ours), "instances_submitted": count(theirs),
              "instances_only_in_ours": sum(v["only_in_ours"] for v in differing.values()),
              "instances_only_in_submitted": sum(v["only_in_submitted"] for v in differing.values()),
              "images_that_differ": differing}
    report["organiser_pq_ours_against_submitted"] = organiser_pq(ours_csv, submitted_csv) if with_pq else None
    return report


def describe_difference(report):
    """The plain-language lines the command and the notebook print for a CSV that differs."""
    lines = ["THE CSV DIFFERS FROM THE SUBMITTED ONE (its sha256 is not the expected value).",
             "%d of %d images have identical instances; %d instances occur only in our CSV, %d only in the submitted one."
             % (report["images_identical"], report["images"], report["instances_only_in_ours"],
                report["instances_only_in_submitted"])]
    pq = report.get("organiser_pq_ours_against_submitted")
    if pq and pq.get("pq") is not None:
        lines.append("The organisers' PQ of our CSV against the submitted CSV is %.6f (1.0 would mean the same instances)." % pq["pq"])
    return lines


def save_ensembled_map(directory, image_id, ensembled_map):
    """Optional: keep the ensembled 8-bit map as <id>.npz (key p), the kernels' map layout."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(directory) / (image_id + ".npz"), p=ensembled_map)


def check_threads(threads):
    require(isinstance(threads, int) and not isinstance(threads, bool) and 1 <= threads <= 64, "THREADS")


def set_threads(threads):
    """Use threads CPU threads (environment variables and torch); return the count torch reports.

    One thread against several can change an 8-bit value by one unit in a rare pixel (README), so the
    documented command passes --threads 4, the count of our rehearsals. None leaves the default.
    """
    import torch
    if threads is not None:
        check_threads(threads)
        for name in THREAD_VARIABLES:
            os.environ[name] = str(threads)
        torch.set_num_threads(threads)
    return torch.get_num_threads()


def reproduce(images_dir, out, candidate_path=DEFAULT_CANDIDATE, device="cpu", threads=None,
              save_maps=None, limit=None, compare_with=None):
    """Run every stage over images_dir and write the CSV to out; return a summary dict."""
    if threads is not None:
        check_threads(threads)
        for name in THREAD_VARIABLES:
            os.environ[name] = str(threads)
    candidate = read_candidate(candidate_path)
    postprocess.mask_module()                       # fail before the slow stages if pycocotools is missing
    device = resolve_device(device)
    listing = list_inputs(images_dir, limit)
    models = load_models(candidate, Path(candidate_path).absolute().parent, device)
    used_threads = set_threads(threads)
    rule = rule_of(candidate)
    rows = []
    for image_id, path in listing:
        image = preprocess.load_halpha(path)
        ensembled = ensemble(predict_maps(image, models, device))
        if save_maps is not None:
            save_ensembled_map(save_maps, image_id, ensembled)
        rows.extend(find_instances(image_id, ensembled, rule))
    digest = write_submission(rows, out)
    status = check_expected(digest, candidate, partial=limit is not None)
    summary = {"images": len(listing), "rows": len(rows), "device": device, "threads": used_threads, "csv": str(out),
               "csv_sha256": digest, "expected_csv_sha256": candidate.get("expected_csv_sha256"),
               "comparison": status, "candidate_status": candidate.get("status")}
    reference = compare_with if compare_with is not None else submitted_path(candidate, candidate_path)
    if reference is not None and status != "SKIPPED_PARTIAL" and (status == "MISMATCH" or compare_with is not None):
        summary["difference"] = compare_instances(out, reference)
        found = summary["difference"]
        if status == "MISMATCH" or found["images_identical"] != found["images"]:
            summary["message"] = " ".join(describe_difference(found))
    elif status == "MISMATCH":
        summary["message"] = ("THE CSV DIFFERS FROM THE SUBMITTED ONE (sha256 mismatch). "
                              "No submitted CSV was available to compare instances; pass --compare-with.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, help="folder of <id>.jpeg H-alpha images")
    parser.add_argument("--out", required=True, help="CSV file to write")
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--threads", type=int, help="CPU threads (the documented command uses 4, as our rehearsals)")
    parser.add_argument("--save-maps", help="optional folder for the ensembled 8-bit maps")
    parser.add_argument("--compare-with", help="the submitted CSV: report how our CSV differs from it (instances and PQ)")
    parser.add_argument("--limit", type=int, help="first N images only (smoke tests; the CSV cannot match)")
    args = parser.parse_args(argv)
    summary = reproduce(args.images_dir, args.out, args.candidate, args.device, args.threads,
                        args.save_maps, args.limit, args.compare_with)
    print(json.dumps(summary, sort_keys=True))
    if summary.get("message"):
        print(summary["message"], file=sys.stderr)
    return 4 if summary["comparison"] == "MISMATCH" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print("REFUSED " + str(error), file=sys.stderr)
        raise SystemExit(2)
