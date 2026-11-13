# Rehearsal on validation images

This file describes the rehearsal of the one-model candidate K2 (run 2, `weights/run2.pt`). The final candidate is K2: the
same weights, the same rule and the same code, so this is the rehearsal of the final candidate.

For the final build one more check was made on the test set: this repository's rule and CSV code were applied to the 8-bit
maps of the 180 test images that were stored when the submitted files were made. Both CSV files came out byte-identical to the
submitted ones (E1: 782 rows, sha256 da9de8064832d9be7f7bd81729c6a981bc8a308b0c94758b0797f7d754e55da5; E2: 1223 rows, sha256
13cb7bafc5958acecf152812eaf601456011543cd7476f104301442f965ce066).

Before release we ran `reproduce.py` on a CPU over the 141 validation images of our own evaluation split (never the
test images or the holdout) with the weights of `weights/run2.pt` and the rule T_low 0.15, T_high 0.8, minimum area 1024.

- Environment: Python 3.12.15, torch 2.2.2, numpy 1.26.4, CPU, 4 threads; about 12.6 seconds per image (measured while other jobs shared the machine).
- Against the 8-bit maps stored from an earlier run of the same code on the same CPU: 1.000000 of all pixels equal,
  maximum absolute difference 0 (0 of 141 images differ at all). This compares CPU with CPU; the CPU-GPU difference is measured
  on another model in the last section.
- The instances (RLE rows) were identical to those of the code the package was taken from on the stored maps in 141 of 141 images.
- Pooled counts with the pinned scorer: TP 445, FP 554, FN 1217, the same as the counts recorded for these weights and this rule (445, 554, 1217).
- Thread-count probe on our Mac: the same first 3 images with 1 thread instead of 4: 1.000000 of pixels equal, maximum
  absolute difference 0 (the only settings we tried). On a Windows machine (an independent read of the repository, on
  made-up images, CPU, torch 2.6.0) one pixel of one image was 120 with 1 thread and 119 with 2, 4, 8 and 12 threads: one
  unit, not at a cut of the rule, so no instance changed; three other images were identical at every count. So the
  thread count can move a rare pixel by one unit; the documented command fixes it at 4, the count used here.
  An independent reviewer later read the repository on Windows with its test suite (not the full pipeline: pycocotools
  was absent there): the 1024 x 1024 end-to-end test (`tests/test_numerics.py`, which allows at most 1 unit on at most 16
  of 1,048,576 pixels) differed from the stored map in 0 pixels at 1, 2 and 4 threads, and on six made-up 2048 x 2048
  images 1 thread differed from 4 in exactly one pixel by one unit, not at a cut of the rule, with no instance changed.
  We ran the pipeline ourselves only on macOS (CPU) and in the competition's Linux notebooks on Kaggle (training and the
  stored validation maps); Windows was read by that reviewer with the suite, not run by us.

## CPU against a Kaggle GPU

We measured the CPU-GPU difference on one of our trained models (run A1) over the 141 validation images: the 8-bit maps that a Kaggle Tesla T4 (torch 2.11, CUDA 13) stored against the maps `reproduce.py` makes on a CPU (torch 2.2.2, numpy 1.26.4, 4 threads). 591,396,757 of 591,396,864 pixels are equal (99.99998%); 107 pixels differ, every one by exactly 1 unit, in 72 of 141 images. Only 3 of those pixels change side of a threshold cut used by the rules (cut 39 for T_low 0.15: 2; cut 128 for 0.5: 1; cut 204 for T_high 0.8: 0). Under the plain rule (0.5, area 512) the instances are identical in 141 of 141 images, the CSV bytes are identical and TP/FP/FN are 364/1141/1298 on both. Under the hysteresis rule (0.15, 0.8, area 1024) 140 of 141 images are identical; in 1 image one instance occurs only with the CPU maps and one only with the GPU maps, so the two CSV files have different sha256 values, while TP/FP/FN (358/638/1304) and PQ (0.17196) are the same. So a reader should expect the same predictions in almost every image, but a bit-identical CSV cannot be promised on other hardware or library versions: the CSV hash is the strict check and a difference is possible. When the hash differs, `reproduce.py` and the notebook say so in plain words (`reproduce.py` exits with code 4); when the submitted CSV is available (a final build ships it and names it in `candidate.json` as `submitted_csv`; otherwise pass `--compare-with FILE`), they also report how many images and instances differ and the organisers' PQ of the new CSV against the submitted one. Nothing is adjusted to force equality.
