# E3: the later, exploratory entry (our final leaderboard entry)

E3 is a separately trained model, registered and trained AFTER the public scores of E1 and E2 and the holdout
result were known (registration: `registration/PREREG_ADDENDUM_4_TRACK2_E3.md`, sha256
`ea6c3ebcb4398d2559f54140ea6e75fa21bc39895f21452255552efca0e5ccc1`, and its note 1, sha256
`8913af9f8fad7a79b0154d27916f8bec752ef4776fa4f0f1ca96877fac6454ce`). It is not the tested candidate: E3 has no
holdout result, and every tested claim of the technical report concerns E1.

| | |
| --- | --- |
| Model | U-Net with an ImageNet-pretrained ResNet34 encoder (torchvision `resnet34-b627a593.pth`), 3 input channels, 1024 x 1024 working size |
| Targets, loss | mean of the annotators' masks; batch Dice plus binary cross-entropy |
| Training | 30 epochs x 110 steps, batch 16, two T4 GPUs on Kaggle, 86 minutes, training and validation images of the fixed split only |
| Instance rule | hysteresis T_low 0.5, T_high 0.85, minimum area 256, limb mask at 0.9 of the disk radius; chosen from 74 settings on validation |
| Validation PQ | 0.4232 (selected-on: checkpoint and rule chosen on these 141 images; all five cross-validation folds chose the same setting) |
| Public leaderboard | 0.35 (submitted 2026-11-13, ref 58322829); the competition counts one entry per team and takes the best public score, so E3 is our final entry, selected on the leaderboard |
| Entry file | `../submitted/e3.csv`, 1,477 rows, 180 images, sha256 `45eb4079bf0383ef315bb5d5274b9672afbd5d32195978ffc565215f3fefd3bf` |
| Weights | `../weights/e3_best.pt` (PyTorch state dict, fp32, 97.9 MB), sha256 `492d96619b865d1ff4c29a4dee99e33feea670f4012f456c2666b0f55e3e51ba`; non-commercial terms as in `../WEIGHTS-TERMS.md` |

## Files

- `code/kernel/track2_kernel.py` (sha256 `bec85fe24fbe491591212424cff8cc8480a14ee8c2454f7f725f1e38e4a22b5e`): the exact Kaggle script that trained E3 and wrote its entry
  file. It is self-contained: it carries every project file it uses (bundle sha256
  `da6a06483712ce944ad0abec88e0ea61a1c8a263fe912ae556702bdeb1c0e9de`) and the run checked each file's hash.
  `code/kernel/kernel-metadata.json` is its Kaggle settings (two T4 GPUs; internet on only to fetch the hash-checked
  encoder weights).
- `code/track2.py`, `code/score_worker.py`, `code/pack.py`, `code/test_track2.py`: the readable sources of that
  bundle, its packer and its offline test suite (38 tests). The comments name our internal review folders; the files
  are kept byte for byte as they ran.
- `records/`: the run's own outputs: `RUN_LOG.txt` (every step, timed), `STOP_RULE.json` (the stop rule at
  validation PQ 0.27 and its verdict CONTINUE), `FROZEN_RULE.json` (the rule, written before any test image was
  read), `RULE_TABLE.csv`/`.json` (all 74 settings), `SCORE.json`, `TEST_PATH_SELF_CHECK.json`,
  `SUBMISSION_VALIDATION.json`, `SCORING_PREFLIGHT.json`, `PROVENANCE.json` (hashes of every module, the data,
  the split and the pretrained weights), `STATUS.json`.

## Reproduce

Upload `code/kernel/track2_kernel.py` as a Kaggle script with the settings of `kernel-metadata.json`, attached to
the competition's data, and run it (about 1.5 hours on two T4 GPUs). It retrains from the pretrained encoder and
writes `track2_out/submission.csv`. GPU training is not bit-for-bit deterministic, so a rerun gives close but not
identical weights; to reproduce the submitted file exactly, the run's own record is `records/` together with
`weights/e3_best.pt` and the frozen rule. The test path at batch size 1 is required to equal the validation path at
the same batch size exactly, and to stay within 4 grey levels on at most 10 % of the pixels of the batch-4 maps that
were scored (GPU rounding noise measured on a smoke run before the full run).
