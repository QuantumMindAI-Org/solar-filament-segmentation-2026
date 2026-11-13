#!/usr/bin/env python3
"""Standalone training of the package U-Net on the organisers' train/ folder and our split list.

SAME RECIPE; NOT THE SCRIPT THAT PRODUCED THE SHIPPED WEIGHTS; NOT RUN AT FULL LENGTH BY US.
The shipped weights were trained by the as-run sources in training/neural_baseline_r1 and
training/neural_train_r3 through a launcher that is not published (see TRAINING.md). This file
restates their recipe with no launcher files: it uses the same model class, crop, augmentation,
sampler, losses and learning-rate schedules, imported from this repository wherever they exist
there. It has been tested only for a few steps on CPU on tiny synthetic images
(tests/test_train_standalone.py). It does not promise bit-identical weights: the as-run code also
checked pinned hashes of every input, ran on a Kaggle T4 with deterministic CUDA settings, and set
arms A and B's step count from the measured speed of the first 200 steps (here --steps gives it).

Recipes (TRAINING.md has them in words):
  run2  2000 steps, AdamW lr 1e-3 constant, weight decay 1e-4, loss BCE + soft Dice, seed 20261108
  R     run2's recipe with another seed (we used 20261111, 20261112, 20261113)
  A     6000 steps by default, linear warm-up over 200 steps to 1e-3, cosine to 1e-5 at the last step,
        loss BCE + soft Dice
  B     as A, loss 1.0 x BCE + 0.5 x soft Dice + 0.5 x soft clDice (neural_train_r3/cldice.py)
Every step draws 4 samples. A sample: a uniform training image, then a uniform annotator of that
image, the union of that annotator's polygons as the target, then the training crop of
solar_filament.model.training_crop (512 x 512; half the time centred on a random annotated pixel),
with a random right-angle rotation and horizontal flip. Only the image ids of the "train" part of
the split list are ever opened; validation and holdout images are not read.

  python -B training/train_standalone.py --train-dir PATH/TO/train --recipe run2 --out weights.pt

--train-dir is the competition's train/ folder: train_images/<image id>.jpeg and the annotation JSON
(MAGFiLO_1.0_Annotations_kaggle2026_train.json, or give --annotations). Prints one JSON line at step 1,
every 100 steps and at the end (with the sha256 of the weights file). Exit 2 on a refusal.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
for folder in (HERE.parent, HERE):            # the repository root (solar_filament) and training/ (neural_train_r3)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402

DEFAULT_SPLIT = HERE / "split.json"
ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RECIPES = {
    "run2": {"steps": 2000, "schedule": "constant", "dice_weight": 1.0, "cl_weight": 0.0, "seed": 20261108},
    "R": {"steps": 2000, "schedule": "constant", "dice_weight": 1.0, "cl_weight": 0.0, "seed": None},
    "A": {"steps": 6000, "schedule": "warmup_cosine", "dice_weight": 1.0, "cl_weight": 0.0, "seed": None},
    "B": {"steps": 6000, "schedule": "warmup_cosine", "dice_weight": 0.5, "cl_weight": 0.5, "seed": None},
}
PEAK_LR, MIN_LR, WARMUP, WEIGHT_DECAY, BATCH = 1e-3, 1e-5, 200, 1e-4, 4


def require(ok, code):
    """Raise ValueError(code) unless ok."""
    if not ok:
        raise ValueError(code)


def lr_at(step, total, schedule):
    """Learning rate of 0-based step: constant 1e-3, or warm-up then cosine (as neural_train_r3.train_r3.lr_at)."""
    if schedule == "constant":
        return PEAK_LR
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    span = max(1, total - 1 - WARMUP)
    progress = min(1.0, (step - WARMUP) / span)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def combined_loss(logits, target, dice_weight, cl_weight):
    """BCE + dice_weight x soft Dice + cl_weight x soft clDice; with (1, 0) this is solar_filament.model.loss."""
    from torch.nn import functional as F
    from solar_filament.model import loss
    if dice_weight == 1.0 and cl_weight == 0.0:
        return loss(logits, target)
    from neural_train_r3 import cldice
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (probability * target).sum(axes) + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    total = F.binary_cross_entropy_with_logits(logits, target) + dice_weight * (1 - dice).mean()
    return total + cl_weight * cldice.soft_cldice(probability, target).mean()


def read_split(path):
    """The train image ids of the split list (the other parts are checked for shape only, never opened)."""
    payload = json.loads(Path(path).read_text())
    require(isinstance(payload, dict) and all(isinstance(payload.get(k), list) for k in ("train", "validation", "holdout")),
            "SPLIT_SCHEMA")
    ids = payload["train"] + payload["validation"] + payload["holdout"]
    require(payload["train"] and len(set(ids)) == len(ids), "SPLIT_SCHEMA")
    return list(payload["train"])


class TrainingData:
    """Train images and the per-annotator polygons of the organisers' annotation JSON."""

    def __init__(self, train_dir, train_ids, annotations=None):
        self.images_dir = Path(train_dir) / "train_images"
        payload = json.loads(Path(annotations or Path(train_dir) / ANNOTATIONS_NAME).read_text())
        require(isinstance(payload, dict) and isinstance(payload.get("images"), list)
                and isinstance(payload.get("annotations"), list), "ANNOTATION_SCHEMA")
        wanted = {image_id + ".jpeg": image_id for image_id in train_ids}
        self.groups = {}                                  # image id -> [annotator image ids]
        self.size = {}                                    # annotator image id -> (height, width)
        for row in payload["images"]:
            image_id = wanted.get(row.get("file_name"))
            if image_id is not None:
                self.groups.setdefault(image_id, []).append(row["id"])
                self.size[row["id"]] = (int(row["height"]), int(row["width"]))
        self.polygons = {}                                # annotator image id -> [segmentation]
        for item in payload["annotations"]:
            if item.get("image_id") in self.size:
                require(item.get("iscrowd", 0) == 0, "CROWD_UNSUPPORTED")
                self.polygons.setdefault(item["image_id"], []).append(item["segmentation"])
        self.train_ids = list(train_ids)
        require(all(self.groups.get(i) for i in self.train_ids), "NO_ANNOTATOR:" + ",".join(
            i for i in self.train_ids if not self.groups.get(i))[:200])

    def image(self, image_id, size):
        from PIL import Image
        path = self.images_dir / (image_id + ".jpeg")
        with Image.open(path) as photo:
            require(photo.format == "JPEG" and photo.mode == "L" and photo.size == (size[1], size[0]),
                    "IMAGE_FORMAT:" + path.name)
            return np.array(photo, dtype=np.uint8, copy=True)

    def target(self, group, size):
        """Union of one annotator's polygons (each annotation rasterised by pycocotools, then merged)."""
        from pycocotools import mask as mask_util
        height, width = size
        encoded = [mask_util.merge(mask_util.frPyObjects(polygons, height, width))
                   for polygons in self.polygons.get(group, [])]
        if not encoded:
            return np.zeros(size, np.uint8)
        return mask_util.decode(mask_util.merge(encoded)).astype(np.uint8)

    def sample(self, rng, tile):
        """One training sample: uniform image, uniform annotator, union target, crop and augmentation."""
        from solar_filament.model import training_crop
        image_id = rng.choice(self.train_ids)
        group = rng.choice(self.groups[image_id])
        size = self.size[group]
        return training_crop(self.image(image_id, size), self.target(group, size), rng, tile=tile)


def train(train_dir, out, recipe="run2", seed=None, steps=None, split=DEFAULT_SPLIT, annotations=None,
          device="cpu", tile=512, batch=BATCH, log=print):
    """Train one model and write its state dict to out. Returns a summary dict."""
    import torch
    from solar_filament.model import UNet
    require(recipe in RECIPES, "RECIPE")
    settings = RECIPES[recipe]
    seed = settings["seed"] if seed is None else seed
    require(isinstance(seed, int), "SEED")
    total = settings["steps"] if steps is None else steps
    require(isinstance(total, int) and total >= 1, "STEPS")
    data = TrainingData(train_dir, read_split(split), annotations)
    if device != "cpu":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        return _loop(data, rng, recipe, settings, seed, total, out, device, tile, batch, log)
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def _loop(data, rng, recipe, settings, seed, total, out, device, tile, batch, log):
    import torch
    from solar_filament.model import UNet
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    model.train()
    history = []
    for step in range(total):
        images, targets = [], []
        for _ in range(batch):
            image, target = data.sample(rng, tile)
            images.append(image.astype(np.float32) / 255.0)
            targets.append(target.astype(np.float32))
        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(targets)[:, None]).to(device)
        lr = lr_at(step, total, settings["schedule"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        objective = combined_loss(model(x), y, settings["dice_weight"], settings["cl_weight"])
        require(bool(torch.isfinite(objective).item()), "NONFINITE_LOSS")
        objective.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == total:
            record = {"step": step + 1, "loss": float(objective.detach().cpu()), "lr": lr}
            history.append(record)
            log(json.dumps(record))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, partial)
    os.replace(partial, out)
    summary = {"recipe": recipe, "seed": seed, "steps": total, "weights": str(out),
               "weights_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "loss_samples": history}
    log(json.dumps({k: summary[k] for k in ("recipe", "seed", "steps", "weights", "weights_sha256")}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True, help="the competition's train/ folder")
    parser.add_argument("--out", required=True, help="weights file to write")
    parser.add_argument("--recipe", default="run2", choices=sorted(RECIPES))
    parser.add_argument("--seed", type=int, help="default: 20261108 for run2; required for A, B, R")
    parser.add_argument("--steps", type=int, help="default: 2000 for run2 and R, 6000 for A and B")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--annotations", help="annotation JSON (default: in --train-dir)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tile", type=int, default=512, help="crop size (512 as trained; small values for tests)")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    train(args.train_dir, args.out, args.recipe, args.seed, args.steps, args.split, args.annotations,
          args.device, args.tile, args.batch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print("REFUSED " + str(error), file=sys.stderr)
        raise SystemExit(2)
