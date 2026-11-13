# Solar Filament Segmentation Challenge 2026: QMAI entry

Code, weights and a reproduction notebook for our entry to the Solar Filament Segmentation Challenge 2026
(instance segmentation of filaments in H-alpha images, scored by panoptic quality at IoU > 0.5).
The method is a small from-scratch U-Net, tiled full-resolution inference, an optional ensemble of up to
three models, and a hysteresis threshold with a minimum area that turns probability maps into filament
instances. The technical report describes the method and the evaluation; this repository lets the committee
reproduce the submitted predictions without retraining.

This is the final build (revision 7, which adds E3; see the next section). It holds the submitted candidate (`candidate.json`, `status` FINAL): one model, run 2
(`weights/run2.pt`), with the hysteresis rule T_low 0.15, T_high 0.8, minimum area 1024. It also holds the two
CSV files submitted to the competition, byte for byte (`submitted/e1.csv`, `submitted/e2.csv`), the expected sha256
of each, and the result of our pre-registered holdout (`results/HOLDOUT_RESULT.json`). The technical report is
filed with the organisers; it is not part of this repository. Repository: https://github.com/QuantumMindAI-Org/solar-filament-segmentation-2026

## Reproduce the predictions in three commands

Python 3.12 (Linux x86_64 or macOS):

```
pip install -r requirements.txt
# put the competition's test images (<14 digits><2 letters>.jpeg, 2048 x 2048, grayscale) in one folder
python reproduce.py --images-dir PATH/TO/TEST_IMAGES --out submission.csv --threads 4
```

`--threads 4` is the CPU thread count of our rehearsals; the JSON line reports the count used (field `threads`).
If `pycocotools` (the RLE encoder) cannot be installed, `--help` and the map stages still work, but writing the CSV
needs it, and the command stops with one line, `REFUSED MISSING_PACKAGE:pycocotools ...`, naming `requirements.txt`.

`requirements.txt` is the default and the one to use for reproducing the predictions: it pins the exact
versions of the CPU environment in which `reproduce.py` was rehearsed (Python 3.12.15, torch 2.2.2, numpy
1.26.4; see `REHEARSAL.md`), and other versions can change an 8-bit probability value by one unit in a few
pixels (below). `requirements-cpu-tested.txt` keeps the same set as the record of that rehearsal. For the
notebook install `requirements-notebook.txt` instead (the same pins plus Jupyter).

`reproduce.py` prints one JSON line with the CSV's sha256 and the field `comparison`:

| `comparison` | meaning | exit code |
| --- | --- | --- |
| MATCH | the CSV's sha256 equals `expected_csv_sha256` in `candidate.json` | 0 |
| MISMATCH | it differs; a plain message says so and, when the submitted CSV is available (`submitted_csv` in `candidate.json`, or `--compare-with FILE`), how many images and instances differ and the organisers' PQ of the new CSV against the submitted one | 4 |
| NO_EXPECTED_VALUE | `candidate.json` has no expected hash: nothing can be compared. This is what a rehearsal build prints | 0 |
| SKIPPED_PARTIAL | `--limit N` was given (first N images only): a partial CSV cannot match | 0 |

A refusal (a stray file in the images folder, an image that is not a 2048 x 2048 grayscale JPEG, a weights
file whose sha256 differs) prints `REFUSED <code>` and exits 2; the image-folder refusals also name the offending
file (`REFUSED STRAY_FILE:notes.txt`). The images folder may
hold only the test images; the operating-system files `.DS_Store`, `Thumbs.db`, `desktop.ini` and a
`__MACOSX` folder are ignored.

The same steps, one cell each, are in `reproduce.ipynb`. Start Jupyter in the repository root and set the
environment variable `SOLAR_IMAGES_DIR` to the folder of test images (or edit the first code cell; it stops
with a plain message if the folder does not exist). Optional variables: `SOLAR_OUT` (CSV to write, default
`submission.csv`), `SOLAR_DEVICE` (cpu, cuda or auto), `SOLAR_THREADS` (default 4), `SOLAR_LIMIT` (first N images), `SOLAR_SUBMITTED_CSV`
(the submitted CSV to compare against) and `SOLAR_LABELS` (a labelled CSV for the optional last cell).
Nothing else is needed: the weights are in `weights/`, the ensemble formula and the post-processing rule are
in `candidate.json` and the code.

## A later entry, E3, our final leaderboard entry

After E1 and E2 were submitted and the holdout result was known, a separately trained model with a pretrained
encoder was registered, trained and submitted as E3 (public score 0.35; E1 scored 0.17). The competition counts one
entry per team and takes the best public score, so E3 is our final entry, selected on the leaderboard. E3 has no
holdout result and is not the tested candidate described below. Its code, records, registration and weights are in
`e3/` and `weights/e3_best.pt`; its entry file is `submitted/e3.csv`. See `e3/README.md`.

## The two submitted entries

| entry | candidate file | rule | submitted file | sha256 of the submitted file |
| --- | --- | --- | --- | --- |
| E1 | `candidate.json` | hysteresis: T_low 0.15, T_high 0.8, minimum area 1024 | `submitted/e1.csv` (782 rows, 180 images) | da9de8064832d9be7f7bd81729c6a981bc8a308b0c94758b0797f7d754e55da5 |
| E2 | `candidate_e2.json` | plain: threshold 0.5, minimum area 512 | `submitted/e2.csv` (1223 rows, 180 images) | 13cb7bafc5958acecf152812eaf601456011543cd7476f104301442f965ce066 |

Both entries use the same model and the same 8-bit probability maps; only the rule differs. The command above
reproduces E1. For E2 add `--candidate candidate_e2.json` (and choose another `--out`):

```
python reproduce.py --images-dir PATH/TO/TEST_IMAGES --out submission_e2.csv --threads 4 --candidate candidate_e2.json
```

Each run compares its CSV with the expected sha256 and, if it differs, with the submitted file shipped here.
Before release we applied this repository's rule and CSV code (`solar_filament/postprocess.py`, `submission.py`)
to the 8-bit test maps stored when the submitted files were made: the result was byte-identical to both submitted
files (same sha256). The map stage itself was checked on validation images (`REHEARSAL.md`). A full CPU run over
the 180 test images takes about 30 to 40 minutes for one model (about 10 to 13 seconds per image).

## What is in the repository

| path | content |
| --- | --- |
| `candidate.json`, `candidate_e2.json` | the configuration of E1 and of E2: weights file with sha256, ensemble formula, rule (T_low, T_high, minimum area), expected sha256 of the CSV, the submitted CSV |
| `submitted/` | `e1.csv`, `e2.csv` and `e3.csv`, the three files submitted to the competition, unchanged |
| `e3/` | the later entry E3: its Kaggle script, sources, run records and registration (see `e3/README.md`) |
| `records/SELECTION_CANDIDATE.json` | the selection record that chose the candidate on validation (K2: run 2 alone; validation PQ 0.2142, TP 445, FP 554, FN 1217; the other options' scores) |
| `results/` | `HOLDOUT_RESULT.json`, the result of the single holdout opening, and `GATES.json`, the checks on validation that had to pass before it (below) |
| `weights/` | the trained weights: `run2.pt` (E1, PyTorch state dict, about 7.9 MB, hash-checked before it is loaded) and `e3_best.pt` (E3, 97.9 MB). Non-commercial terms: `WEIGHTS-TERMS.md` |
| `solar_filament/` | the package: `model.py` (U-Net, tiled inference), `preprocess.py`, `inference.py`, `ensemble.py`, `postprocess.py`, `submission.py` |
| `reproduce.py`, `reproduce.ipynb` | the inference pipeline as a command and as a notebook; `tools/` holds the script that writes the notebook and a cell runner used by the tests |
| `REHEARSAL.md` | the CPU rehearsal on validation images, and the measured CPU-GPU difference |
| `training/`, `TRAINING.md` | the training sources as they ran, their plan files, our split list (`training/split.json`) and a standalone training script (`training/train_standalone.py`) |
| `evaluation/` | the organisers' overlap and PQ code (two notebook cells, unchanged) and a loader for it; see "Optional: score a labelled set" |
| `tests/` | unit tests with synthetic images and a tiny random-weights model |
| `requirements.txt`, `requirements-notebook.txt`, `requirements-cpu-tested.txt` | packages and versions (above) |
| `LICENSE`, `LICENSE-APACHE-2.0`, `NOTICE`, `WEIGHTS-TERMS.md` | the code licence (MIT), the Apache licence of the two organiser cells, the notice for them, and the terms of the weights |
| `REPO_MANIFEST.sha256` | sha256 of every file |

## Run names

| name | what it is |
| --- | --- |
| run 2, also K2 | the first model: one seed (20261108), 2000 steps, constant learning rate; `weights/run2.pt` |
| A1, A2, A3 | arm A (longer schedule) with seeds 20261111, 20261112, 20261113, in that order |
| B1, B2, B3 | arm B (arm A plus a clDice term) with the same three seeds, in that order |
| R1, R2, R3 | arm R (run 2's recipe) with the same three seeds, in that order |

`TRAINING.md` gives each recipe; `candidate.json` names the runs the candidate uses.

## The pipeline

1. Read each H-alpha JPEG as a 2048 x 2048 grayscale array (`preprocess.py`). The only preprocessing is the
   division by 255 inside the model's tiling function: no normalisation, resizing or colour handling.
2. Tiled inference (`model.tiled_probability`): 512 x 512 tiles, stride 384, sigmoid probabilities averaged in the
   overlaps. The map is quantised to 8 bits, `round(255 p)`, as the training kernels stored them.
3. Ensemble (`ensemble.py`): the integer mean of the models' 8-bit maps, rounded half up, `E = (2*S + n) // (2*n)`.
   A single model is left unchanged.
4. Rule (`postprocess.py`): components of {map >= cut(T_low)} (8-connected) that contain a pixel >= cut(T_high) and
   have at least A pixels become filaments; the cut of a threshold T is ceil(255 T) on the 8-bit values.
5. Write the CSV (`submission.py`): `filament_id` (image id, underscore, 4-digit running number) and
   `segmentation_rle` (COCO compressed counts, 2048 x 2048). An image with no kept component writes no row.

## Expected output

One CSV for the whole test set in the organisers' submission format. In the final build its sha256 is
`expected_csv_sha256` in `candidate.json`. Per image, nothing is random: there is no sampling and no test-time
augmentation; two runs with the same thread count gave the same maps (see below for other thread counts).

## Environment and run time

Training ran on Kaggle (Tesla T4, CUDA 13.0, Python 3.13.16, torch 2.11.0, numpy 2.1.3; the runs recorded
no other package versions). Inference was rehearsed on a CPU with the versions in `requirements.txt`. One model
on one 2048 x 2048 image took about 12.6 seconds on four CPU threads of an older Intel Mac (measured over the
141-image rehearsal, `REHEARSAL.md`), so budget about 13 seconds x images x models. `reproduce.py` needs about
0.5 GB of memory (peak 523 MB for one model); the notebook keeps every image's 8-bit maps, about 4 MB per image
per model.

CPU is the default; `--device cuda` (or `auto`) uses a GPU. CPU, GPU and other library versions add
floating-point numbers in a different order, so an 8-bit probability value can differ by one unit in a few
pixels: on one of our models and 141 validation images, a Kaggle T4 and our CPU differed in 107 of 591,396,864
pixels, which changed one instance in one image under the hysteresis rule and left TP, FP, FN and PQ unchanged.
The CSV hash is therefore the strict check and a difference is possible on other hardware; the numbers are in
`REHEARSAL.md`, section "CPU against a Kaggle GPU".

The thread count can matter too. On our Mac, 1 thread against 4 gave identical maps on the 3 images we tried
(maximum difference 0). On a Windows machine, an independent read of the repository found one pixel of one
made-up 2048 x 2048 image at 120 with 1 thread and 119 with 2, 4, 8 and 12 threads (one unit; no instance changed),
and identical maps at every thread count on three other images. The documented command therefore passes
`--threads 4`, the count of our rehearsals; `REHEARSAL.md` has the details.

An independent reviewer later read the repository on Windows with its test suite (not the full pipeline: pycocotools
was absent there): the 1024 x 1024 end-to-end test (`tests/test_numerics.py`, which allows at most 1 unit on at most 16
of 1,048,576 pixels) differed from the stored map in 0 pixels at 1, 2 and 4 threads, and on six made-up 2048 x 2048
images 1 thread differed from 4 in exactly one pixel by one unit, not at a cut of the rule, with no instance changed.
We ran the pipeline ourselves only on macOS (CPU) and in the competition's Linux notebooks on Kaggle (training and the
stored validation maps); Windows was read by that reviewer with the suite, not run by us.

## Limits

- The model is small and trained from scratch on a few hundred images; our own validation scores are in the report.
- The rule and the choice of candidate were selected on our validation split. Neither says how the method behaves
  on solar images from other years or instruments.
- `training/train_standalone.py` states the training recipe as runnable code, but it is not the script that
  produced the shipped weights and we have not run it at full length; retraining will not give bit-identical
  weights. See `TRAINING.md`.

## Evaluation split and the holdout

We split the organisers' training set, by group of near-duplicate frames, into 443 training images, 141
validation images (34 groups) and 123 holdout images (25 groups); the image ids are in `training/split.json`.
The holdout was set aside before any model was trained and was not used for training or selection. It was opened
once, on 11 November 2026, for the final candidate, under a rule registered in advance (registration of 10 November
2026 and three addenda, all before any holdout image was read): nothing about the candidate changed after it was
bound, and the result is reported as it fell. The result file is `results/HOLDOUT_RESULT.json` (sha256
0b076c5e0bdc5e581cc6f95a1edfb635f7874ead81310527b6ffdf11148540fa); the report gives the details.

- The candidate C (run 2) scores PQ 0.2091 on the holdout (TP 398, FP 485, FN 1110), against 0.2142 on validation
  (difference -0.0051).
- A frozen classical reference G (no learning) scores PQ 0.0253. The registered hypothesis H1, C beats G, holds:
  PQ(C) minus PQ(G) is 0.1839, with a 95% cluster-bootstrap interval of 0.1634 to 0.1998 (25 groups, 2000 draws).
- H2 (C beats run 2) is void, because the chosen candidate is run 2 itself.
- The agreement between annotators on the same images is PQ 0.3342.

Before the opening, the evaluator had to reproduce the known validation counts exactly; `results/GATES.json` records
those three checks (all passed). The registration documents themselves and the evaluator are not in this
repository: they are working documents that name our internal tooling. Their sha256 values are: registration
7618eaec3cd9c12eb63733504b951a78bebe303218e543b0a25b87eeb3265ede, addendum 1
d59f29e626325515e1bb7996bbf802c8538cdfdd57749600999245e33e7b22bb, addendum 2 6c42dd33445cebba259fb87025d312c61298f3ecf189058cecd6c0434edad486, addendum 3
50e6207e3f0c93fc9f7bd1ae6a6bb1d23273b437ce9952c5035b6d3269722491. The holdout images and the holdout predictions are
not in this repository. The test images were not used for training or selection.

## Optional: score a labelled set

`evaluation/` holds two cells of the organisers' public evaluation notebook, unchanged (`organiser_cell3.py.txt`,
`organiser_cell4.py.txt`: the overlap matrices and the panoptic-quality score), and `organiser_cells.py`, which
checks their sha256 and runs them. The last cell of `reproduce.ipynb` uses them to compute the organisers' PQ of
the CSV against a labelled set, if you set `SOLAR_LABELS` to a ground-truth CSV (`filament_id` =
`<annotator>-<image id>_<n>`, `segmentation_rle`); it scores only the images present in that file.
`evaluation/` computes PQ only. It does not compute Dice scores, IoU distributions or the one-to-many and
many-to-one counts; the distribution figures in the technical report come from a separate script that is not in
this repository.

## Data and licence

The images and annotations are the organisers' (CC BY-NC 4.0, non-commercial) and are not included here; get them
from the competition page (https://kaggle.com/competitions/filament-segmentation-2026). The dataset is described in
Ahmadzadeh et al., "A dataset of manually annotated filaments from H-alpha observations", Scientific Data 11, 1031
(2024). The code is under the MIT licence in `LICENSE`. The two organiser cells in `evaluation/` are included
unchanged under the Apache License 2.0, whose text is `LICENSE-APACHE-2.0` (see `NOTICE`). The weights are not
under the MIT licence: they were trained on the non-commercial data and are provided for non-commercial
reproduction and study only, as stated in `WEIGHTS-TERMS.md` (the organisers' rules exempt input data and
pretrained models with an incompatible licence from the open-source grant).

## AI assistance

The pipeline, code and documentation were produced by AI agents directed and reviewed by the human author, an
independent researcher, who attests to the best of the author's ability that each component can be explained and
reproduced, as the competition rules require when AI tools are used.
