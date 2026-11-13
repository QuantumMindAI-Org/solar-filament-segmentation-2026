# Training

The weights in `weights/` were trained on free Kaggle notebook GPUs (one Tesla T4, CUDA 13.0, torch 2.11.0,
numpy 2.1.3, Python 3.13.16 as recorded by the runs). Inference does not need any of this: `reproduce.py`
loads the stored weights.

This folder has two kinds of training code:

1. `training/train_standalone.py`: one script that trains the package U-Net (`solar_filament.model.UNet`) on the
   competition's `train/` folder and our split list, with no other file. **Same recipe; not the script that
   produced the shipped weights; not run at full length by us.** It is tested for a few steps on CPU on tiny
   synthetic images (`tests/test_train_standalone.py`), which also checks that its losses and its learning-rate
   schedule equal the as-run functions. It will not give bit-identical weights (see its docstring).
2. `training/neural_baseline_r1/` and `training/neural_train_r3/`: the sources exactly as they ran, unchanged, which
   produced the shipped weights through a launcher that is not published.

## The standalone script

```
python -B training/train_standalone.py --train-dir PATH/TO/train --recipe run2 --out weights.pt
python -B training/train_standalone.py --train-dir PATH/TO/train --recipe A --seed 20261111 --out a1.pt
```

`--train-dir` is the organisers' `train/` folder (the `train_images/` folder and the annotation JSON). The
script reads `training/split.json` and opens only the images of its `train` part. `--recipe` is run2, R, A or B
(below); `--steps` sets the step count (2000 for run2 and R, 6000 for A and B by default; the as-run arms A and B
chose it from the measured speed of their first 200 steps, see below). It runs on CPU unless `--device cuda`.
It writes the state dict that `reproduce.py` loads; to use it, put the file in `weights/` and its sha256 in a
`candidate.json`.

## The as-run sources

| file | role |
| --- | --- |
| `training/neural_baseline_r1/model.py` | the U-Net, the loss (BCE + soft Dice), the paired rotation/reflection augmentation, the training crop. Byte-identical to `solar_filament/model.py`. |
| `training/neural_baseline_r1/data.py` | loads training pairs and validation images from the competition files, with a hash check of every input |
| `training/neural_baseline_r1/run.py` | the training run of the first model ("run 2"), with its time budget and its output files |
| `training/neural_baseline_r1/PLAN.json` | run 2's plan as frozen before it ran: model, sampling, optimiser, steps, seed, inference settings |
| `training/neural_train_r3/train_r3.py` | the later training runs: arms A and B (longer schedule, optional clDice term) and arm R (run 2's recipe on three further seeds) |
| `training/neural_train_r3/cldice.py` | soft clDice loss used by arm B |
| `training/neural_train_r3/PLAN_R3.json` | the plan of arms A, B and R: loss weights per arm, seeds, run order, learning-rate schedule, step rule |
| `training/split.json` | our split: the image ids of the train (443), validation (141) and holdout (123) parts, nothing else |

These files import each other by their original package names (`neural_baseline_r1`, `neural_train_r3`),
so the `training/` folder must be on `PYTHONPATH`. They cannot be run from this repository alone: `data.py`
checks every input against pinned hashes of the competition files and of our evaluation-split files, and `run.py`
and `train_r3.py` expect a frozen source manifest and a platform record from the launcher. Those side files, and
the launcher itself (it carries account details), are not published. They are included so that the loss, the
sampling, the optimiser settings and the schedules can be read exactly as they were run; the standalone script
above is the runnable form of the same recipe.

Names in their comments and strings that point at files which are not in this repository: `REVIEW_NOTES.md`,
`PREREG.md`, `PREREG_ADDENDUM_R.md` ("addendum R"), the revision tags "r2.4" and "r2.5", `bootstrap_r3`,
`controller_r3`, and the frozen-manifest and launcher files (`FREEZE.json`, `FREEZE_R3.json`, `QUALIFICATION.json`,
the schema name `qmai.solar-neural-executor-qualification/v1`) belong to our unpublished review notes and launcher
harness; the hash-pinned inputs of `data.py` (`SPLIT_PLAN.json`, `PIXEL_CENSUS.json`, `DATA_HASHES.json`,
`score_reference.py`, `official_scorer.py`, `SCORER_FINDINGS.json`) are our split, data-hash and scorer files. Nothing
is missing from the repository by accident; `training/split.json` carries the image ids of the split file.
`model.py` also carries training-time helpers that `reproduce.py` never calls (`instances`, a plain 0.5 threshold used
for validation during training, and the `check` hook of `tiled_probability`, which the launcher used for its time
budget). The list of folder names that `data.py` refuses to read (line 37) is a safety list from the training host,
which refused any input path under a credential or tool folder.

## Common to every run

- Data: the organisers' training images and annotations (MAGFiLO, Kaggle 2026), split by us into train (443
  images), validation (141) and a holdout (123), in groups so that near-duplicate frames stay on one side.
- Model: grayscale U-Net, five encoder levels with 16, 32, 64, 128 and 256 channels, GroupNorm(8) and ReLU in
  every block, bilinear upsampling, a 1 x 1 convolution head. About 2 million parameters, trained from scratch
  (no pretrained weights).
- Samples: 512 x 512 crops, batch size 4. With probability one half the crop is centred on a random annotated
  pixel, otherwise it is uniform. Each crop gets a random right-angle rotation and an independent horizontal flip.
  The annotator of each sample is drawn uniformly among the image's annotators.
- Targets: the union of one annotator's filament polygons, rasterised to a binary mask.
- Optimiser: AdamW, weight decay 1e-4.
- Loss: binary cross-entropy on the logits plus soft Dice with smoothing 1 (equal weights), except arm B.
- No validation-driven checkpoint selection: the last weights are the ones saved.
- Inference (all runs): 512 tiles, stride 384, sigmoid probabilities averaged where tiles overlap, never
  stitching thresholded masks. See `solar_filament/model.py`.

## Per-run recipes

The run names are defined in the README ("Run names").

- **Run 2 ("K2")**: 2000 steps, constant learning rate 1e-3, seed 20261108, about 35 minutes of training time
  budget on the T4.
- **Arm A** (A1, A2, A3: seeds 20261111, 20261112, 20261113): up to 6000 steps, 200 steps of linear warm-up to 1e-3,
  then cosine decay to 1e-5. The step count is set by a fixed rule from the wall time of the first 200 steps
  (multiples of 500 that fit a 9900 s budget); the loss is run 2's.
- **Arm B** (B1, B2, B3: same seeds): as arm A with the loss 1.0 x BCE + 0.5 x soft Dice + 0.5 x soft clDice; the
  clDice skeletons come from 8 iterations of min/max pooling (`cldice.py`).
- **Arm R** (R1, R2, R3: same seeds): exactly run 2's recipe: 2000 steps, constant learning rate 1e-3, weight decay
  1e-4, run 2's loss.

Which of these runs the candidate uses is stated in `candidate.json` (model names and weights files) and in the
technical report.

## How the runs were launched

Each run was one private Kaggle notebook (kernel), launched through Kaggle's kernel interface with the
competition data attached, on a free T4 allocation of at most three hours (1.5 hours for arm R,
one hour for run 2). The kernel unpacked the sources above, trained one arm with one seed, validated on the 141
validation images, and wrote `weights.pt`, the 8-bit validation probability maps and a score file. One kernel
ran at a time. A run that failed before it started training was retried; a run that diverged was not.
