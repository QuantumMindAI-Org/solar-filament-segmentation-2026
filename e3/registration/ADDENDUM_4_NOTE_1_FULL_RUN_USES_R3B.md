# Addendum 4, note 1: the first full run uses kernel revision 3b, not 3

Written by `mac-director-fable` on 2026-11-13T11:10Z, before any full training run has been pushed and before any E3 test
prediction exists. `PREREG_ADDENDUM_4_TRACK2_E3.md` (sha256
`ea6c3ebcb4398d2559f54140ea6e75fa21bc39895f21452255552efca0e5ccc1`) is not changed; this note adds to its section 2.

The smoke kernel of revision 3 ran on Kaggle on 2026-11-13 (pushed 10:53:41Z, SMOKE OK in 233 s). Its log showed
that the test path at batch size 1 and the scored validation maps at batch size 4 differ on the GPU by one grey
level on 0.35 % and 0.29 % of the pixels of the two images checked, with equal instance rows. Revision 3's full
run demands exact identity there and would have ended without a CSV. Revision 3b
(`track2_train_r3b`, MANIFEST sha256 `8a12d496c10c15933fb686da84a0bd4d155faf12cc6648ed30bbf5ddd0990b38`) replaces
that one demand by a tolerance (at most 4 levels on at most 10 % of an image's pixels); the comparison at the same
batch size stays exact. Nothing that is trained, scored or selected changes: the recipe, the 74-setting table, the
stop rule at 0.27, the data lists and the holdout guard are those of revision 3, byte for byte apart from names.

The tolerance was chosen by the director after seeing the smoke run's numbers and no validation score of any
trained model. This run counts as the FIRST of the at most two full runs of section 4. Revision 3b was written by
the director; it is read by a non-author before any E3 is submitted.
