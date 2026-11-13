"""Tests of training/train_standalone.py on tiny synthetic images (CPU, a few steps). Run from the repository root:

    python -B -m unittest discover -s tests -t . -v

The images are 96 x 96 synthetic JPEGs and the crop is 64 x 64, so a step takes well under a second. No
competition file is used. The validation and holdout ids of the synthetic split point at files that are not
images: training must never open them.
"""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.dont_write_bytecode = True

import train_standalone as ts  # noqa: E402
from solar_filament.model import UNet, loss  # noqa: E402
from tests import helpers  # noqa: E402

TRAIN = ["20150125172714Mh", "20150125172814Mh", "20150125172914Mh"]
VALIDATION = ["20150126000000Ah"]
HOLDOUT = ["20150127000000Bh"]


def synthetic_train_dir(folder, rgb_image=None):
    """train/ with train_images/ and the annotation JSON; returns (train_dir, split path)."""
    folder = Path(folder)
    images = folder / "train" / "train_images"
    images.mkdir(parents=True)
    rows, annotations = [], []
    for index, image_id in enumerate(TRAIN):
        helpers.write_jpeg(images / (image_id + ".jpeg"), helpers.synthetic_image((96, 96), seed=index))
        annotators = ["a1", "a2"] if index == 0 else ["a1"]
        for annotator in annotators:
            group = annotator + "-" + image_id
            rows.append({"id": group, "file_name": image_id + ".jpeg", "height": 96, "width": 96})
            annotations.append({"id": group + "_1", "image_id": group, "iscrowd": 0,
                                "segmentation": [[10, 20, 80, 20, 80, 27, 10, 27]]})
    for image_id in VALIDATION + HOLDOUT:          # listed in the JSON, but their files are not images
        (images / (image_id + ".jpeg")).write_bytes(b"not an image: must never be opened")
        rows.append({"id": "a1-" + image_id, "file_name": image_id + ".jpeg", "height": 96, "width": 96})
    if rgb_image:
        from PIL import Image
        Image.fromarray(np.zeros((96, 96, 3), np.uint8)).save(images / (rgb_image + ".jpeg"), format="JPEG")
    (folder / "train" / ts.ANNOTATIONS_NAME).write_bytes(json.dumps({"images": rows, "annotations": annotations}).encode())
    split = folder / "split.json"
    split.write_text(json.dumps({"schema": "solar-filament-split/v1", "train": TRAIN, "validation": VALIDATION,
                                 "holdout": HOLDOUT}))
    return folder / "train", split


@helpers.needs_pycocotools
class StandaloneTraining(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.train_dir, self.split = synthetic_train_dir(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def run_steps(self, **kwargs):
        lines = []
        kwargs.setdefault("steps", 3)
        summary = ts.train(self.train_dir, self.folder / "out" / "w.pt", split=self.split, tile=64, batch=2,
                           log=lines.append, **kwargs)
        return summary, lines

    def test_a_few_steps_give_finite_losses_and_a_loadable_weights_file(self):
        summary, lines = self.run_steps()
        self.assertTrue(all(math.isfinite(record["loss"]) for record in summary["loss_samples"]))
        self.assertEqual([record["step"] for record in summary["loss_samples"]], [1, 3])
        weights = self.folder / "out" / "w.pt"
        self.assertTrue(weights.is_file())
        self.assertFalse((self.folder / "out" / "w.pt.partial").exists())
        model = UNet()
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.assertEqual(json.loads(lines[-1])["weights_sha256"], summary["weights_sha256"])
        self.assertFalse(torch.are_deterministic_algorithms_enabled())    # the global setting is restored

    def test_recipe_b_runs_with_the_cldice_term(self):
        summary, _ = self.run_steps(recipe="B", seed=20261111, steps=2)
        self.assertTrue(all(math.isfinite(record["loss"]) for record in summary["loss_samples"]))

    def test_same_seed_same_weights(self):
        first, _ = self.run_steps(steps=2)
        second, _ = self.run_steps(steps=2)
        self.assertEqual(first["weights_sha256"], second["weights_sha256"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "RECIPE"):
            self.run_steps(recipe="Z")
        with self.assertRaisesRegex(ValueError, "SEED"):
            self.run_steps(recipe="A")                       # arms A, B, R need an explicit seed
        with self.assertRaisesRegex(ValueError, "STEPS"):
            self.run_steps(steps=0)
        bad = self.folder / "bad_split.json"
        bad.write_text(json.dumps({"train": TRAIN, "validation": TRAIN[:1], "holdout": []}))
        with self.assertRaisesRegex(ValueError, "SPLIT_SCHEMA"):
            ts.train(self.train_dir, self.folder / "w.pt", split=bad, steps=1, tile=64, batch=1, log=lambda line: None)
        bad.write_text(json.dumps({"train": TRAIN}))
        with self.assertRaisesRegex(ValueError, "SPLIT_SCHEMA"):
            ts.train(self.train_dir, self.folder / "w.pt", split=bad, steps=1, tile=64, batch=1, log=lambda line: None)
        bad.write_text(json.dumps({"train": TRAIN + ["20150128000000Ch"], "validation": [], "holdout": []}))
        with self.assertRaisesRegex(ValueError, "NO_ANNOTATOR:20150128000000Ch"):
            ts.train(self.train_dir, self.folder / "w.pt", split=bad, steps=1, tile=64, batch=1, log=lambda line: None)
        with mock.patch.object(ts, "combined_loss", lambda *a: torch.tensor(float("nan"), requires_grad=True)):
            with self.assertRaisesRegex(ValueError, "NONFINITE_LOSS"):
                self.run_steps(steps=1)
        self.assertFalse((self.folder / "out" / "w.pt").exists())

    def test_crowd_annotation_and_annotation_schema_refused(self):
        path = self.train_dir / ts.ANNOTATIONS_NAME
        payload = json.loads(path.read_text())
        payload["annotations"][0]["iscrowd"] = 1
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "CROWD_UNSUPPORTED"):
            self.run_steps(steps=1)
        path.write_text(json.dumps({"images": []}))
        with self.assertRaisesRegex(ValueError, "ANNOTATION_SCHEMA"):
            self.run_steps(steps=1)

    def test_image_format_refusal_names_the_file(self):
        data = ts.TrainingData(self.train_dir, TRAIN)
        with self.assertRaisesRegex(ValueError, "IMAGE_FORMAT:" + TRAIN[0]):
            data.image(TRAIN[0], (64, 64))                    # the file is 96 x 96

    def test_command_line(self):
        out = self.folder / "cli.pt"
        command = [sys.executable, "-B", str(ROOT / "training" / "train_standalone.py"), "--train-dir", str(self.train_dir),
                   "--split", str(self.split), "--out", str(out), "--steps", "1", "--tile", "64", "--batch", "1",
                   "--threads", "1"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(out.is_file())
        result = subprocess.run(command + ["--recipe", "A"], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 2)
        self.assertIn("REFUSED SEED", result.stderr)


class RecipePieces(unittest.TestCase):
    def test_schedules(self):
        self.assertEqual(ts.lr_at(0, 2000, "constant"), 1e-3)
        self.assertEqual(ts.lr_at(1999, 2000, "constant"), 1e-3)
        self.assertAlmostEqual(ts.lr_at(0, 6000, "warmup_cosine"), 1e-3 / 200)
        self.assertAlmostEqual(ts.lr_at(199, 6000, "warmup_cosine"), 1e-3)
        self.assertAlmostEqual(ts.lr_at(5999, 6000, "warmup_cosine"), 1e-5)

    @unittest.skipUnless(importlib.util.find_spec("resource"), "the as-run train_r3 imports the POSIX module resource")
    def test_losses_equal_the_as_run_expressions(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 32, 32)
        target = (torch.rand(2, 1, 32, 32) > 0.7).float()
        self.assertTrue(torch.equal(ts.combined_loss(logits, target, 1.0, 0.0), loss(logits, target)))
        from neural_train_r3 import cldice
        probability = logits.sigmoid()
        dice = (2 * (probability * target).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        by_hand = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target) + 0.5 * (1 - dice).mean()
                   + 0.5 * cldice.soft_cldice(probability, target).mean())
        self.assertTrue(torch.allclose(ts.combined_loss(logits, target, 0.5, 0.5), by_hand, atol=0, rtol=0))
        from neural_train_r3 import train_r3                   # the as-run module (importing it reads no file)
        for dice_weight, cl_weight in ((1.0, 0.0), (0.5, 0.5)):
            self.assertTrue(torch.equal(ts.combined_loss(logits, target, dice_weight, cl_weight),
                                        train_r3.combined_loss(logits, target, dice_weight, cl_weight)))
        for step in (0, 100, 199, 200, 3000, 5999):
            self.assertEqual(ts.lr_at(step, 6000, "warmup_cosine"), train_r3.lr_at(step, 6000))

    def test_shipped_split_list_has_the_published_counts(self):
        payload = json.loads((ROOT / "training" / "split.json").read_text())
        self.assertEqual({k: len(payload[k]) for k in ("train", "validation", "holdout")},
                         {"train": 443, "validation": 141, "holdout": 123})
        ids = payload["train"] + payload["validation"] + payload["holdout"]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(payload) - {"train", "validation", "holdout"}, {"schema", "note"})
        self.assertEqual(ts.read_split(ROOT / "training" / "split.json"), payload["train"])


if __name__ == "__main__":
    unittest.main()
