# SOLAR-HOLDOUT-R1 addendum 4: a third, exploratory leaderboard entry (E3) from a separately trained model — registered revision

Registered by `mac-director-fable` (Fable 5.2) on 2026-11-13T10:54Z, before the full training run has been pushed
and before any E3 test prediction exists. (The SMOKE kernel, which reads no test image and writes no CSV, was
pushed at 2026-11-13T10:53:41Z, before this file was written.) The sha256 of this file is bound on the bus now. `holdout_r1/PREREG.md`
(sha256 `7618eaec3cd9c12eb63733504b951a78bebe303218e543b0a25b87eeb3265ede`) and addenda 1, 2 and 3 are not changed,
except the one sentence replaced in section 5 below. This is the "new registered revision" that PREREG.md requires
before any third entry.

## 1. Timing and motive, in plain words

This revision is written AFTER the two registered entries were submitted (2026-11-11 08:15Z), AFTER their public
scores were seen (E1 0.17, E2 0.15), and AFTER the holdout was opened once and its result was seen (2026-11-11
11:35Z to 12:21Z; H1 true, H2 void; candidate holdout PQ 0.20912). E3 exists because of the public scores: the
registered candidate was barely trained, and the owner said yes to one better-trained model. Nothing here is a test
of a hypothesis. Every report and page that mentions E3 repeats this paragraph's content.

## 2. What E3 is

One training recipe, fixed in kernel revision 1 (delivered 2026-11-11 10:15Z, before the holdout was opened) and
not changed since in what is trained or how it is selected: a U-Net with a ResNet34 encoder started from
torchvision's ImageNet weights (`resnet34-b627a593.pth`, hash-checked), inputs at 1024 x 1024 in three channels,
targets the mean of the annotators, batch Dice plus BCE loss, 30 epochs of 110 steps at batch 16, the best
checkpoint by validation PQ every 5 epochs, test-time augmentation, and one instance rule chosen on validation
from a fixed table of 74 settings of the registered rule family, scored by the pinned scorer. Revisions 2 and 3 of
the kernel changed only how scoring runs and fails (fresh supervised processes with deadlines; count budgets mark
a setting instead of ending the run; the self-check compares like with like).
The code is `track2_train_r3`, MANIFEST sha256
`6ad108848fff995054ba98ccfe6b7e7953f859a338d140f5e440123d6f20f588`. When this file was written its non-author
re-review was unfinished (attacks on the scoring pool done, the reviewer's own planted faults and verdict not yet)
and one text-only read by another model family had found no blocker. NO E3 IS SUBMITTED before that review's
ACCEPT is on record; a full training run may start before it, because a run submits nothing.

E3 is NOT the tested candidate. The bound candidate (CANDIDATE.json `90e328f8...`), claims H1 and H2, their
bootstrap, the evaluator, the sealed result files and the report's statement that the holdout number is the test
result of the candidate are untouched whatever E3 scores. E3 has no holdout result and is never described as
tested on the holdout.

## 3. Data

Training images (443) and validation images (141) of `evaluation-split-r1` only. No holdout image, annotation,
prediction or result is used for E3's training, tuning or selection, now or later; the kernel refuses to start if
a holdout id is in its lists. No test image is used for training (no pseudo-labels). The kernels run with internet
ON for one purpose only, fetching the encoder weights named above; the competition's rules allow external models
(`RULES_CAPTURE_20261108.txt` line 90). This is the director's ruling on the plan's third ask.

## 4. The stop rule, the runs and the entries

- The SMOKE kernel is pushed first. It reads no test image and writes no CSV.
- Stop rule, applied by the kernel itself at the end of a full run: if the validation PQ at the best point of the
  74-setting table is below 0.27, the run ends, no test image is read, no CSV exists, and track 2 ENDS: no second
  run. (For comparison, the registered candidate's validation PQ is 0.2142481.)
- At most TWO full runs. A second run is allowed only if the first passed the stop rule, and only with its one
  change written into this folder and bound on the bus before its push.
- At most TWO E3 submissions in all, one per run that passed the stop rule, each the run's own `submission.csv`
  as the kernel wrote it, validated on the Mac by the registered validator, hashed and kept read-only, and each
  submitted on its own go-ahead by the director.
- If two E3 submissions exist, neither is preferred by hand: section 5 applies to all our entries alike.

## 5. The final selection (replaces "The two final selections are E1 and E2")

A fact first, read on our own Submissions page on 2026-11-13 at 05:53Z: this competition allows ONE final
selection ("Select up to 1 submissions that will count towards your final leaderboard score. If less than 1 are
selected, Kaggle will automatically select from your best scoring submissions"). PREREG.md assumed two. E2 can
therefore never be a final selection beside E1, and it never was one.

- With E1 and E2 only, the final selection is E1, the registered candidate; it is also Kaggle's automatic choice
  (public 0.17 against 0.15), so nothing is selected by hand.
- If an E3 submission exists, nothing is selected by hand either: Kaggle's automatic choice stands, that is, the
  entry with the highest public score among all our entries. Every report and page then says in these words that the
  final entry was LEADERBOARD-SELECTED, and gives both public scores. The candidate does not change: E1 remains
  the tested candidate whatever is final.
- An E3 is submitted only after the owner has read the paragraph that discloses it in the report (section 6) and
  has said so. Without that word no E3 is submitted, however its validation looks.

## 6. What must be disclosed wherever E3 appears (report, repository, form)

This revision's sha256 and time; that it was decided after the public scores and the holdout result were seen;
the E3 recipe; its validation PQ named as "selected-on" (checkpoint and rule chosen on those 141 images) and its
5-fold cross-validated PQ; every E3 submission with its public score; which entry was final and that it was leaderboard-selected; and that E3
has no holdout result. The E3 code, the weights' sha256 and where the weights are kept go into the release; its
README separates the tested candidate from E3.

## 7. What this revision does not allow

A third run; a third E3 submission; any use of holdout data; any change to the candidate, to H1 or H2, or to the
sealed result files; selecting a final entry by hand; an E3 submission without the owner's word of section 5; describing E3 as tested.
