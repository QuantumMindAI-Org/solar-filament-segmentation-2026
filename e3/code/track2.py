"""SOLAR TRACK2-R3: pretrained ResNet34 U-Net at 1024 px, trained and scored on Kaggle GPUs (revision 3).

Implements seats/solar-track2-plan-20261111/PLAN.md (sha256 8d274c43...): whole disk downsampled by 2 to 1024,
ImageNet-pretrained ResNet34 encoder, all training images in RAM, soft targets = mean over annotators of each
annotator's union mask, batch-level Dice + BCE, limb handled (flattened limb darkening as an input channel, radial
channel, optional limb mask in the rule family), flip TTA, validation PQ through the SAME pinned scorer and the SAME
hysteresis/area rule code as the registered path (pp_hysteresis.settings_table -> pp_morph.rebuild_frame ->
pp_minarea.score_counts -> official_scorer.score_rle_frames), CSV through submission_r1/predict_test.py.

r2 (review seats/solar-track2-review-20261111): every scorer call runs in a FRESH interpreter, never the training
process, which receives only the component's GT frame, its uint8 maps and disk geometry; a scoring preflight runs before
training; SMOKE mode for the first push; STOP verdict = no test stage at all; the CSV gets its final name only after
validation and re-derivation; an in-kernel self-check proves the test path equals the validation path.
r3 (review seats/solar-track2-review-r2-20261111 + coordinator rulings): the fresh interpreters are a run-wide
ScoringPool of persistent subprocesses running the shipped module file score_worker.py, started at the very start of
the run while this process is small (Linux keeps ru_maxrss across execve), supervised under hard start / per-task /
death deadlines with named errors (no multiprocessing; nothing waits without a deadline); count budgets are soft only
in periodic and smoke validations (validation recorded "not scoreable" with its numbers, no checkpoint from it) and
hard in the final validation; every other refusal is hard everywhere; the self-check failure names the largest
difference and the number of differing pixels.

This module never submits, never opens a holdout image or converts a holdout annotation, and never reads a test
image before FROZEN_RULE.json is written (and never when the stop rule says STOP, and never in smoke mode).
"""
import hashlib
import io
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCHEMA = 'qmai.solar-track2-r3/v1'
STOP_THRESHOLD = 0.27        # PLAN.md section 4: < 0.27 stop track 2
CONTINUE_THRESHOLD = 0.30    # PLAN.md section 4: >= 0.30 continue
# r3b (director's ruling from the smoke run of r3 on 2 x T4, 2026-11-13: batch 1 against the scored batch-4 maps
# differed by 1 uint8 level on 0.35 % and 0.29 % of the pixels, rows equal): the full run tolerates this GPU
# rounding noise and still refuses a gross difference. The same-batch comparison (a) stays exact.
BATCH_TOL_MAX_LEVELS = 4     # largest allowed |batch-1 map - scored batch-4 map| in uint8 levels
BATCH_TOL_SHARE = 0.10       # largest allowed share of an image's pixels that differ at all
# Real pins (R files, verified 2026-11-11 against neural_baseline_r1/data.py PINS and SPLIT_PLAN provenance).
PINS = {
    'split': '27e3f1056120c6f96ec38d141b6add66a98fdb5daa014c5f67fd558133b515cc',
    'census': '32b05e2ba7fc255dfd3fa8d43323eed51432e646b7396a077aeb557c209af5c4',
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'data_hashes': 'a9b7e9859ce208db60a6231b06eba089890654219c7ff0347a23962774308ffe',
}
REAL_COUNTS = {'train': 443, 'validation': 141, 'holdout': 123}
REAL_VAL_COMPONENTS = 34
REAL_VAL_GROUPS = 234
REAL_TEST_IMAGES = 180
COMPETITION = 'filament-segmentation-2026'
ANNOTATION_REL = 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
# torchvision ResNet34_Weights.IMAGENET1K_V1; the file name carries the first 8 hex of its sha256.
WEIGHTS_URL = 'https://download.pytorch.org/models/resnet34-b627a593.pth'
WEIGHTS_FILE = 'resnet34-b627a593.pth'
WEIGHTS_PREFIX = 'b627a593'
WEIGHTS_DEADLINE_S = 110     # named refusal well inside two minutes
CHILD_RSS_WARN = 1.5 * 2 ** 30
H = W = 2048
# real array shapes (for the real-size rehearsal ballast): X float16, XV float16, Y uint8
REAL_SHAPES = {'X': ((443, 3, 1024, 1024), 'float16'), 'XV': ((141, 3, 1024, 1024), 'float16'),
               'Y': ((443, 1024, 1024), 'uint8')}

# Rule family: hysteresis (T_low, T_high) on the uint8 map, 8-connected components, area >= A, plus an
# optional limb mask (map zeroed where r / R_disk > limb). limb None = no mask = the registered family.
QUICK_SETTINGS = [(0.5, 0.5, 512, None), (0.35, 0.7, 256, None), (0.25, 0.7, 512, None), (0.15, 0.8, 1024, None)]


def full_settings():
    pairs = [(tl, th) for tl in (0.25, 0.35, 0.5) for th in (0.5, 0.7, 0.85) if th >= tl]
    out = []
    for limb in (None, 0.9):
        for tl, th in pairs:
            for a in (128, 256, 512, 1024):
                out.append((tl, th, a, limb))
        out.append((0.15, 0.8, 1024, limb))   # the registered candidate's rule, for comparability
    return out


REAL_CFG = dict(mode='full', dry=False, work=1024, crop=512, batch=16, epochs=30, crops_per_image=4, lr=3e-4,
                wd=1e-4, warmup_epochs=1, eval_every=5, workers=3, preload_procs=4, score_procs=4,
                score_mode='subprocess', fg_crop_p=2 / 3, seed=20261111, tta=True, max_steps=None,
                train_seconds_max=1.75 * 3600, final_eval_seconds_max=40 * 60, quick_eval_seconds_max=8 * 60,
                log_every=25, threads=4, pretrained=True, model='unet', head_bias=None, ballast=False,
                self_check_images=2, smoke_subset_components=3)
# SMOKE: the real data and memory, a few dozen GPU steps, one real validation, no test image, no CSV.
SMOKE_CFG = dict(REAL_CFG, mode='smoke', epochs=1, max_steps=40, eval_every=1, log_every=5,
                 train_seconds_max=15 * 60, quick_eval_seconds_max=6 * 60, final_eval_seconds_max=6 * 60)
# DRY (Mac CPU, synthetic). branch 'stop': the real U-Net, untrained -> STOP. branch 'continue': a 2-parameter
# darkness model (dark = filament) that genuinely scores high on the synthetic set -> CONTINUE -> test stage.
DRY_CFG = dict(REAL_CFG, dry=True, work=256, crop=64, batch=2, epochs=1, crops_per_image=1, eval_every=1,
               workers=0, preload_procs=0, score_procs=1, max_steps=2, train_seconds_max=600,
               final_eval_seconds_max=300, quick_eval_seconds_max=120, log_every=1, threads=2, pretrained=False,
               smoke_subset_components=1)


class Refusal(RuntimeError):
    pass


def require(cond, code):
    if not cond:
        raise Refusal(code)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================ project imports (from src_root only)
class Project:
    """Imports the pinned R modules from src_root. dependencies() must run before any torch parallel work."""

    def __init__(self, src_root):
        self.root = Path(src_root).absolute()
        for sub in ('', 'postproc_minarea_r1', 'postproc_morph_r2', 'postproc_hysteresis_r3', 'submission_r1'):
            p = str(self.root / sub) if sub else str(self.root)
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib.util
        spec = importlib.util.spec_from_file_location('score_reference', self.root / 'score_reference.py')
        sr = importlib.util.module_from_spec(spec)
        sys.modules['score_reference'] = sr
        spec.loader.exec_module(sr)
        self.sr = sr
        require(Path(sr.__file__).absolute().parent == self.root, 'SCORE_REFERENCE_ROOT')
        # pinned scorer + findings, torch thread limits (set once, before training)
        self.np, self.pd, self.Image, self.mask_util, self.scorer = sr.dependencies()
        from neural_baseline_r1 import data as nbd
        import pp_minarea as pp
        import pp_morph as pm
        import pp_hysteresis as ph
        import predict_test as pt
        import validate_submission as vs
        self.nbd, self.pp, self.pm, self.ph, self.pt, self.vs = nbd, pp, pm, ph, pt, vs
        for mod in (nbd, pp, pm, ph, pt, vs, self.scorer):
            require(Path(mod.__file__).absolute().is_relative_to(self.root), 'FOREIGN_MODULE:' + mod.__name__)


_PROJECTS = {}


def get_project(src_root):
    key = str(Path(src_root).absolute())
    if key not in _PROJECTS:
        _PROJECTS[key] = Project(src_root)
    return _PROJECTS[key]


# ============================================================ holdout guard and lists
def guard_lists(train_ids, val_ids, holdout_ids):
    """Refuse to start if any holdout id is in the training or validation list (counts only, no ids printed)."""
    train_ids, val_ids, holdout_ids = set(train_ids), set(val_ids), set(holdout_ids)
    require(holdout_ids, 'HOLDOUT_LIST_EMPTY')
    leak = (train_ids | val_ids) & holdout_ids
    require(not leak, 'HOLDOUT_IN_TRAIN_OR_VALIDATION_LIST:%d' % len(leak))
    require(not (train_ids & val_ids), 'TRAIN_VALIDATION_OVERLAP')
    return True


def holdout_ids_from_plan(split):
    out = set()
    for g in split['groups']:
        if g['partition'] == 'holdout':
            out.update(g['image_ids'])
    return out


def build_lists(nbd, split_raw, census_raw, split_pin, census_pin, expected_counts):
    """Split plan -> train rows, validation rows, validation components. Refuses on any pin mismatch or leak.
    Holdout ids are taken from the plan as names only; nothing about a holdout image is opened."""
    require(sha256(split_raw) == split_pin, 'SPLIT_PLAN_HASH_MISMATCH')
    require(sha256(census_raw) == census_pin, 'PIXEL_CENSUS_HASH_MISMATCH')
    split = json.loads(split_raw)
    census = json.loads(census_raw)
    require(split.get('pixel_census_sha256') == census_pin, 'SPLIT_CENSUS_JOIN')
    holdout = holdout_ids_from_plan(split)
    # our own guard first, on the raw plan, before the pinned partition logic
    plan_train = {i for g in split['groups'] if g['partition'] == 'train' for i in g['image_ids']}
    plan_val = {i for g in split['groups'] if g['partition'] == 'validation' for i in g['image_ids']}
    guard_lists(plan_train, plan_val, holdout)
    partitions, components = nbd.partition_index(split, census, expected=expected_counts)
    train = list(partitions['train'])
    val = list(partitions['validation'])
    guard_lists([r['image_id'] for r in train], [r['image_id'] for r in val], holdout)
    return train, val, components, holdout


# ============================================================ annotations and targets
def allowed_annotations(annotations, allowed_ids):
    """{annotator_group: [annotation]} for allowed physical images only. The image id is checked BEFORE the
    annotation's segmentation is touched; holdout polygons are never converted, counted or read."""
    by_group = {}
    for item in annotations['annotations']:
        group = item.get('image_id')
        if type(group) is not str or group.split('-', 1)[-1] not in allowed_ids:
            continue
        by_group.setdefault(group, []).append(item)
    return by_group


def soft_target(sr, mask_util, row, by_group):
    """Mean over the image's annotators of each annotator's union mask, float32 2048x2048 in [0, 1]."""
    acc = np.zeros((H, W), np.float32)
    groups = row['annotator_image_ids']
    for group in groups:
        encoded = []
        for ann in by_group.get(group, []):
            require(ann.get('iscrowd', 0) == 0, 'CROWD_UNSUPPORTED')
            encoded.append({'size': [H, W], 'counts': sr.polygon_rle(ann['segmentation'], H, W).encode('ascii')})
        if encoded:
            acc += mask_util.decode(mask_util.merge(encoded)).astype(np.float32)
    return acc / max(len(groups), 1)


# ============================================================ disk geometry and input channels
def disk_geometry(im):
    """Same rule as the track-2 diagnosis (diag.py): threshold, fill holes, centroid, R = sqrt(area/pi)."""
    from scipy import ndimage
    thr = max(20, int(np.percentile(im, 30)) // 2)
    disk = ndimage.binary_fill_holes(im > thr)
    area = int(disk.sum())
    if area < 0.05 * im.size:
        return {'cy': im.shape[0] / 2, 'cx': im.shape[1] / 2, 'R': 0.45 * im.shape[0], 'fallback': True}
    ys, xs = np.nonzero(disk)
    return {'cy': float(ys.mean()), 'cx': float(xs.mean()), 'R': float(np.sqrt(area / np.pi)), 'fallback': False}


def radius_map(disk, size, full=H):
    """r / R at the centres of a size x size grid laid over the full-resolution image."""
    f = full / size
    c = (np.arange(size, dtype=np.float32) + 0.5) * f - 0.5
    return np.hypot(c[:, None] - disk['cy'], c[None, :] - disk['cx']).astype(np.float32) / np.float32(disk['R'])


def downsample(a, size):
    f = a.shape[0] // size
    require(a.shape[0] == a.shape[1] == size * f, 'DOWNSAMPLE_SHAPE')
    return a.reshape(size, f, size, f).mean(axis=(1, 3), dtype=np.float32)


def input_channels(im, disk, size):
    """3 x size x size float16: [limb-flattened z-score, raw z-score, radial distance]. Limb handling part 1."""
    img = downsample(im.astype(np.float32), size)
    r = radius_map(disk, size, im.shape[0])
    inside = r <= 1.0
    if inside.sum() < 16:
        inside = np.ones_like(inside)
    bins = np.clip((r * 100).astype(np.int64), 0, 150)
    sums = np.bincount(bins[inside], img[inside], minlength=151)
    cnts = np.bincount(bins[inside], minlength=151)
    prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    valid = np.flatnonzero(cnts > 0)
    prof = np.interp(np.arange(151), valid, prof[valid])           # fill empty bins from neighbours
    prof = np.convolve(np.pad(prof, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    flat = img / np.maximum(prof[bins], 1.0)
    fi = flat[inside]
    c0 = np.where(inside, (flat - np.median(fi)) / (fi.std() + 1e-3), 0.0)
    ri = img[inside]
    c1 = (img - ri.mean()) / (ri.std() + 1e-3)
    c2 = (np.minimum(r, 1.5) - 0.5) * 2.0
    x = np.stack([np.clip(c0, -6, 6), np.clip(c1, -6, 6), c2]).astype(np.float16)
    return x


def limb_mask_uint8(arr, disk, limb):
    """Limb handling part 2 (rule family): zero the 8-bit map where r / R > limb. limb None = unchanged."""
    if limb is None:
        return arr
    out = arr.copy()
    out[radius_map(disk, arr.shape[0], arr.shape[0]) > limb] = 0
    return out


# ============================================================ image access (allowed ids only)
def read_train_image(nbd, Image, dataset_root, row, allowed_ids):
    """Train/validation JPEG, byte and pixel hashes checked against the pinned census row."""
    require(row['image_id'] in allowed_ids, 'HOLDOUT_OR_FOREIGN_IMAGE')
    raw = nbd.read(Path(dataset_root) / 'train/train_images' / row['file_name'], 16 * 1024 ** 2, row['file_sha256'])
    with Image.open(io.BytesIO(raw)) as photo:
        require(photo.format == 'JPEG' and photo.mode == 'L' and photo.size == (W, H), 'PHOTO_FORMAT')
        im = np.array(photo, dtype=np.uint8, copy=True)
    require(sha256(im.tobytes()) == row['pixel_sha256'], 'PHOTO_PIXELS')
    return im


_PRELOAD = {}


def _preload_one(args):
    row, want_target = args
    g = _PRELOAD
    im = read_train_image(g['nbd'], g['Image'], g['dataset_root'], row, g['allowed'])
    disk = disk_geometry(im)
    x = input_channels(im, disk, g['work'])
    out = {'image_id': row['image_id'], 'x': x, 'disk': disk}
    if want_target:
        t = soft_target(g['sr'], g['mask_util'], row, g['by_group'])
        y = np.rint(downsample(t, g['work']) * 255).astype(np.uint8)
        ys, xs = np.nonzero(y >= 64)
        fg = np.stack([ys, xs], 1).astype(np.int32)
        if len(fg) > 4096:
            fg = fg[np.random.default_rng(int(row['pixel_sha256'][:8], 16)).choice(len(fg), 4096, replace=False)]
        out.update(y=y, fg=fg, annotators=len(row['annotator_image_ids']))
    return out


def preload(proj, cfg, rows, by_group, allowed, want_target, log):
    _PRELOAD.update(nbd=proj.nbd, Image=proj.Image, dataset_root=cfg['dataset_root'], allowed=allowed,
                    work=cfg['work'], sr=proj.sr, mask_util=proj.mask_util, by_group=by_group)
    jobs = [(r, want_target) for r in rows]
    t0 = time.time()
    if cfg['preload_procs'] and cfg['preload_procs'] > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(cfg['preload_procs']) as pool:
            items = list(pool.imap(_preload_one, jobs, chunksize=4))
    else:
        items = [_preload_one(j) for j in jobs]
    log('preloaded %d images (targets=%s) in %.1f s' % (len(items), want_target, time.time() - t0))
    return items


# ============================================================ dataset and augmentation
class CropDataset:
    """Index g -> one augmented crop; the draw depends only on (seed, g), not on the worker."""

    def __init__(self, X, Y, FG, crop, seed, fg_p):
        self.X, self.Y, self.FG, self.crop, self.seed, self.fg_p = X, Y, FG, crop, seed, fg_p
        self.size = X.shape[-1]

    def __len__(self):
        return 1 << 62

    def sample(self, g):
        rng = np.random.default_rng([self.seed, int(g)])
        k = int(rng.integers(len(self.X)))
        c, s = self.crop, self.size
        fg = self.FG[k]
        if len(fg) and rng.random() < self.fg_p:
            cy, cx = fg[int(rng.integers(len(fg)))]
            y0 = int(cy) - c // 2 + int(rng.integers(-c // 4, c // 4 + 1))
            x0 = int(cx) - c // 2 + int(rng.integers(-c // 4, c // 4 + 1))
        else:
            y0, x0 = int(rng.integers(0, s - c + 1)), int(rng.integers(0, s - c + 1))
        y0, x0 = min(max(y0, 0), s - c), min(max(x0, 0), s - c)
        x = self.X[k, :, y0:y0 + c, x0:x0 + c].astype(np.float32)
        y = self.Y[k, y0:y0 + c, x0:x0 + c].astype(np.float32) / 255.0
        return augment(x, y, rng)

    def __getitem__(self, g):
        import torch
        x, y = self.sample(g)
        return torch.from_numpy(x), torch.from_numpy(y[None])


def augment(x, y, rng):
    rot = int(rng.integers(4))
    x, y = np.rot90(x, rot, axes=(1, 2)), np.rot90(y, rot)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    for ch in (0, 1):   # intensity jitter on the two image channels, not on the radial channel
        x[ch] = x[ch] * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)
    if rng.random() < 0.3:
        x[:2] += rng.normal(0, rng.uniform(0.02, 0.1), size=x[:2].shape).astype(np.float32)
    return x, y


class EpochSampler:
    def __init__(self, epoch, n):
        self.epoch, self.n = epoch, n

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))

    def __len__(self):
        return self.n


# ============================================================ model (pure torch; torchvision state_dict keys)
def build_model():
    import torch
    from torch import nn

    class BasicBlock(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.downsample = None
            if stride != 1 or cin != cout:
                self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            idt = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(out)) + idt)

    class ResNet34Encoder(nn.Module):
        """Same module names as torchvision.models.resnet34, so its ImageNet state_dict loads by key."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)

            def layer(cin, cout, n, stride):
                return nn.Sequential(BasicBlock(cin, cout, stride), *[BasicBlock(cout, cout, 1) for _ in range(n - 1)])
            self.layer1 = layer(64, 64, 3, 1)
            self.layer2 = layer(64, 128, 4, 2)
            self.layer3 = layer(128, 256, 6, 2)
            self.layer4 = layer(256, 512, 3, 2)

        def forward(self, x):
            x0 = self.relu(self.bn1(self.conv1(x)))
            x1 = self.layer1(self.maxpool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x0, x1, x2, x3, x4

    def cbr(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Up(nn.Module):
        def __init__(self, cin, cskip, cout):
            super().__init__()
            self.block = nn.Sequential(cbr(cin + cskip, cout), cbr(cout, cout))

        def forward(self, x, skip=None):
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            if skip is not None:
                x = torch.cat([x, skip], 1)
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ResNet34Encoder()
            self.d1, self.d2 = Up(512, 256, 256), Up(256, 128, 128)
            self.d3, self.d4, self.d5 = Up(128, 64, 64), Up(64, 64, 32), Up(32, 0, 16)
            self.head = nn.Conv2d(16, 1, 3, 1, 1)

        def forward(self, x):
            x0, x1, x2, x3, x4 = self.encoder(x)
            d = self.d1(x4, x3)
            d = self.d2(d, x2)
            d = self.d3(d, x1)
            d = self.d4(d, x0)
            return self.head(self.d5(d))

    return UNet()


def build_darkness_model():
    """DRY ONLY (branch 'continue'): logit = -a * flattened_channel + b. Two parameters, trainable, geometry-exact
    (it never moves pixels), so the dry run reaches a real CONTINUE verdict and the test path is exercised."""
    import torch
    from torch import nn

    class Darkness(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(3.0))
            self.b = nn.Parameter(torch.tensor(-6.0))

        def forward(self, x):
            return -self.a * x[:, :1] + self.b

    return Darkness()


# ============================================================ pretrained weights (named refusal within ~2 min)
def fetch_weights(dest, log, deadline_s=WEIGHTS_DEADLINE_S, url=WEIGHTS_URL, prefix=WEIGHTS_PREFIX):
    import urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.partial')
    t0 = time.time()
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=20) as resp, open(tmp, 'wb') as f:
            while True:
                if time.time() - t0 > deadline_s:
                    raise Refusal('NO_PRETRAINED_WEIGHTS_DOWNLOAD_TIMEOUT after %.0f s' % (time.time() - t0))
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
    except Refusal:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001  internet off, DNS, HTTP error, socket timeout
        tmp.unlink(missing_ok=True)
        raise Refusal('NO_PRETRAINED_WEIGHTS_DOWNLOAD: %s: %s' % (type(e).__name__, str(e)[:200]))
    full = h.hexdigest()
    if not full.startswith(prefix):
        tmp.unlink(missing_ok=True)
        raise Refusal('PRETRAINED_HASH_MISMATCH: sha256 %s does not start with %s' % (full, prefix))
    os.replace(tmp, dest)
    log('weights downloaded in %.1f s, sha256 %s' % (time.time() - t0, full))
    return dest


def load_pretrained(encoder, cfg, log):
    """Order: an attached file named resnet34-b627a593.pth (cfg weights_path or /kaggle/input/*/), else a download
    with a hard deadline. No silent from-scratch fallback."""
    import torch
    cands = []
    if cfg.get('weights_path'):
        cands.append(Path(cfg['weights_path']))
    if not cfg['dry']:
        base = Path('/kaggle/input')
        if base.is_dir():
            cands += sorted(base.glob('*/' + WEIGHTS_FILE)) + sorted(base.glob('*/*/' + WEIGHTS_FILE))
    path, source = None, None
    for c in cands:
        if c.is_file():
            path, source = c, 'attached:' + str(c)
            break
    if path is None:
        require(not cfg['dry'] or cfg.get('allow_download_in_dry'), 'NO_PRETRAINED_WEIGHTS_FILE')
        path = fetch_weights(Path('/tmp/qmai_weights') / WEIGHTS_FILE if not cfg['dry'] else Path(cfg['out']) / 'w' /
                             WEIGHTS_FILE, log, cfg.get('weights_deadline_s', WEIGHTS_DEADLINE_S),
                             cfg.get('weights_url', WEIGHTS_URL))
        source = 'download:' + cfg.get('weights_url', WEIGHTS_URL)
    full = sha256_file(path)
    require(full.startswith(WEIGHTS_PREFIX) or cfg['dry'], 'PRETRAINED_HASH_MISMATCH')
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith('fc.')}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    require(not missing and not unexpected, 'PRETRAINED_KEYS missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    log('pretrained encoder loaded from %s sha256 %s (%d tensors)' % (source, full, len(state)))
    return {'source': source, 'sha256': full, 'tensors': len(state), 'url': WEIGHTS_URL}


# ============================================================ loss
def loss_fn(logits, target, smooth=1.0):
    """BCE (soft targets) + 1 - soft Dice computed over the WHOLE BATCH (one ratio, not one per crop)."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = (2 * inter + smooth) / (p.sum() + target.sum() + smooth)
    return bce + (1 - dice), {'bce': float(bce.detach()), 'dice': float(dice.detach())}


def old_per_crop_loss(logits, target):
    """The run-2 style loss (BCE + per-crop soft Dice with +1 smoothing), kept only for the test comparison."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.dim()))
    dice = (2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice).mean()


# ============================================================ inference (one function for validation AND test)
def infer_maps(model, X, cfg, device, tta, ids, out_size=H):
    """Yields (image_id, uint8 2048 map): bilinear (align_corners=False) to 2048, then rint(255 p)."""
    import torch
    model.eval()
    bs = 4 if not cfg['dry'] else 2
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device)
            views = [(False, False)] + ([(True, False), (False, True), (True, True)] if tta else [])
            acc = None
            for fh, fv in views:
                v = xb
                if fh:
                    v = torch.flip(v, [3])
                if fv:
                    v = torch.flip(v, [2])
                with autocast_ctx(device):
                    p = torch.sigmoid(model(v).float())
                if fv:
                    p = torch.flip(p, [2])
                if fh:
                    p = torch.flip(p, [3])
                acc = p if acc is None else acc + p
            p = (acc / len(views)).float()
            p = torch.nn.functional.interpolate(p, size=(out_size, out_size), mode='bilinear', align_corners=False)
            p = p.clamp_(0, 1).cpu().numpy()[:, 0]
            for j in range(len(p)):
                yield ids[i + j], np.clip(np.rint(255.0 * p[j].astype(np.float64)), 0, 255).astype(np.uint8)


def autocast_ctx(device):
    import contextlib
    import torch
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def predict_image_rows(proj, model, cfg, device, image_id, im, rule):
    """THE test path for one image (also used by the in-kernel self-check on validation images):
    2048 JPEG -> disk -> channels -> infer_maps -> limb mask of the frozen rule -> predict_test.rows_from_uint8."""
    disk = disk_geometry(im)
    x = input_channels(im, disk, cfg['work'])[None]
    (_, arr), = list(infer_maps(model, x, cfg, device, cfg['tta'], [image_id]))
    arr = limb_mask_uint8(arr, disk, rule['limb'])
    rows = proj.pt.rows_from_uint8(image_id, arr, (rule['T_low'], rule['T_high'], rule['A']))
    return arr, rows


def rss_peak_bytes():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform == 'darwin' else v * 1024)


# ============================================================ scoring: supervised fresh subprocesses (r3)
# Every scorer call runs in a separate interpreter started as `python -B <src_root>/track2_train_r3b/score_worker.py`
# (a small module FILE shipped in the bundle: no multiprocessing, no __main__ re-import, independent of how the
# kernel itself was launched). The run starts score_procs such workers ONCE, at its beginning (ScoringPool), and
# hands them jobs (pickled components) through inbox files; each worker writes a JSON-lines progress file, which the
# parent supervises with hard deadlines:
#   start   : `ready` within SCORE_START_S, and a worker that exits before `ready` is reported at once
#             (SCORING_CHILD_START_FAILED, with its exit code and stderr tail); no ready -> SCORING_CHILD_START_TIMEOUT
#   per task: each task must finish within task_timeout(n_settings, n_images) after task_start -> SCORING_TASK_TIMEOUT
#   pick-up : a job not started within SCORE_START_S -> SCORING_JOB_NOT_PICKED_UP
#   death   : a worker that exits during a job (killed, segfault, OOM) -> SCORING_CHILD_DIED, within about 0.05 s
# Any of these kills every worker (process group) and stops the run with the name. Nothing waits without a deadline.
# Measured cost: about 0.18 s per (setting x image) on this Mac for run-2-like maps (review comparability check:
# 2 settings x 141 images in 36 s on 2 processes, including start-up); child start 2.5 to 4 s here (torch import).
SCORE_START_S = 180                 # >= 40x the measured start; Kaggle disks and imports are slower
SCORE_TASK_BASE_S = 60
SCORE_TASK_PER_SETTING_IMAGE_S = 1.5   # about 8x the measured 0.18 s
COUNT_CODES = ('MASK_COUNT_BUDGET', 'PAIR_COUNT_BUDGET')   # M1: soft (setting not scored), all other refusals hard
FAULT_ENV = 'QMAI_TRACK2_TEST_FAULT'


def task_timeout(cfg, n_settings, n_images):
    return cfg.get('score_task_base_s', SCORE_TASK_BASE_S) + cfg.get(
        'score_task_per_setting_image_s', SCORE_TASK_PER_SETTING_IMAGE_S) * n_settings * n_images


def count_predictions(proj, gt, ids, maps, disks, setting):
    """For a count-budget refusal: the numbers behind it (predictions per image, GT x prediction pairs)."""
    tl, th, a, limb = setting
    n_pred = {}
    for i in ids:
        _labels, areas = proj.ph.hysteresis_state(limb_mask_uint8(maps[i], disks[i], limb), tl, th)
        n_pred[i] = int((areas >= a).sum())
    groups = gt['filament_id'].str.split('_', n=1).str[0]
    pairs = sum(int(n) * n_pred.get(g.split('-', 1)[1], 0) for g, n in groups.value_counts().items())
    return {'max_predictions_per_image': max(n_pred.values()) if n_pred else 0, 'pairs_component': pairs,
            'limits': {'MASK_COUNT_BUDGET': 512, 'PAIR_COUNT_BUDGET': 100000}}


def score_component(proj, gt, ids, maps, disks, groups, soft=COUNT_CODES):
    """Registered path, unmodified: pp_hysteresis.settings_table -> pp_morph.rebuild_frame -> pp_minarea.score_counts
    -> official_scorer.score_rle_frames. A count-budget refusal listed in `soft` marks only that setting (with its
    numbers); anything else raises. Returns ([(limb, settings3, rows)], {setting_key: {code, numbers}})."""
    res, refused = [], {}

    def table_for(fn, s3):
        return proj.ph.settings_table(proj.scorer, proj.mask_util, proj.pd, [(gt, None)], [list(ids)], fn, list(s3))[:, 0, :]

    def code_of(e):
        return str(e) if isinstance(e, ValueError) and str(e) in soft else None
    for limb, s3 in groups:
        def fn(i, limb=limb):
            return limb_mask_uint8(maps[i], disks[i], limb)
        try:
            t = table_for(fn, s3)
        except Exception as e:  # noqa: BLE001
            if code_of(e) is None:
                raise
            t = np.full((len(s3), 4), np.nan)
            for k, s in enumerate(s3):
                try:
                    t[k] = table_for(fn, [s])[0]
                except Exception as e2:  # noqa: BLE001
                    if code_of(e2) is None:
                        raise
                    full = (s[0], s[1], s[2], limb)
                    refused[setting_key(full)] = dict(code=code_of(e2), **count_predictions(proj, gt, ids, maps, disks, full))
        res.append((limb, [tuple(s) for s in s3], t))
    return res, refused


def setting_key(s):
    return '%s|%s|%s|%s' % (s[0], s[1], s[2], s[3])


def _kill_group(p):
    import signal
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.kill()
        except OSError:
            pass
    try:
        p.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def _tail(path, n=1500):
    try:
        return Path(path).read_text(errors='replace')[-n:].replace('\n', ' | ')
    except OSError:
        return ''


class ScoringPool:
    """Persistent scoring workers (score_worker.py run by path), started ONCE while the parent is small and
    supervised with hard deadlines. Every failure is a named Refusal; nothing waits without a deadline:
      start   : a worker that exits before `ready` -> SCORING_CHILD_START_FAILED (exit code, stderr) within ~0.2 s;
                no `ready` within score_start_s -> SCORING_CHILD_START_TIMEOUT
      per task: no `task_done` within task_timeout after `task_start` -> SCORING_TASK_TIMEOUT
      death   : a worker that exits during a job -> SCORING_CHILD_DIED (exit code, stderr) within ~0.2 s
      refusal : a scorer refusal not in `soft` -> SCORER_REFUSED ... <NAME>
    Any failure kills every worker's process group; the pool is then unusable (the run stops)."""

    def __init__(self, proj, cfg, n, log):
        import tempfile
        self.proj, self.cfg, self.log = proj, cfg, log
        self.tmp = Path(tempfile.mkdtemp(prefix='qmai_scorers_', dir=cfg.get('score_tmp')))
        self.dead = None
        worker = Path(proj.root) / 'track2_train_r3b' / 'score_worker.py'
        require(worker.is_file(), 'SCORE_WORKER_MISSING')
        env = dict(os.environ)
        env.pop(FAULT_ENV, None)
        if cfg.get('worker_fault'):
            require(cfg['dry'], 'TEST_FAULT_DRY_ONLY')
            env[FAULT_ENV] = cfg['worker_fault']
        self.parent_rss_at_start = rss_peak_bytes()
        self.w = []
        for k in range(max(1, n)):
            inbox = self.tmp / ('inbox_%d' % k)
            inbox.mkdir()
            prog = self.tmp / ('progress_%d.jsonl' % k)
            prog.touch()
            err = self.tmp / ('stderr_%d.txt' % k)
            with open(err, 'w') as ef:
                p = subprocess.Popen([sys.executable, '-B', str(worker), str(proj.root), str(inbox), str(prog)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=ef, env=env,
                                     cwd=str(self.tmp), start_new_session=True)
            self.w.append({'p': p, 'inbox': inbox, 'progress': prog, 'stderr': err, 'launched': time.time(),
                           'ready': None, 'offset': 0, 'buf': '', 'seq': 0, 'job': None, 'current': None,
                           'since': None})
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- supervision
    def _events(self, c):
        with open(c['progress'], 'rb') as f:
            f.seek(c['offset'])
            data = f.read()
        c['offset'] += len(data)
        c['buf'] += data.decode('ascii')
        evs = []
        while '\n' in c['buf']:
            line, c['buf'] = c['buf'].split('\n', 1)
            evs.append(json.loads(line))
        return evs

    def _check_alive(self, c):
        rc = c['p'].poll()
        if rc is None:
            return
        with open(c['progress'], 'rb') as f:        # unread lines first
            f.seek(c['offset'])
            if f.read():
                return
        if c['ready'] is None:
            self._fail('SCORING_CHILD_START_FAILED: exit code %s after %.1f s; stderr: %s' % (
                rc, time.time() - c['launched'], _tail(c['stderr'])))
        self._fail('SCORING_CHILD_DIED: exit code %s during component %s; stderr: %s' % (
            rc, c['current'], _tail(c['stderr'])))

    def _fail(self, msg):
        self.dead = msg
        self.close()
        raise Refusal(msg)

    def _wait_ready(self):
        start_s = self.cfg.get('score_start_s', SCORE_START_S)
        while any(c['ready'] is None for c in self.w):
            for c in self.w:
                if c['ready'] is not None:
                    continue
                for ev in self._events(c):
                    if ev['event'] == 'ready':
                        c['ready'] = ev
                        if ev.get('fork_sentinel') is not None:
                            self._fail('SCORING_CHILD_NOT_FRESH: the worker sees the parent\'s memory')
                if c['ready'] is None:
                    self._check_alive(c)
                    if time.time() - c['launched'] > start_s:
                        self._fail('SCORING_CHILD_START_TIMEOUT: no ready after %.0f s; stderr: %s' % (
                            time.time() - c['launched'], _tail(c['stderr'])))
            time.sleep(0.05)
        self.start_info = [{'pid': c['ready']['pid'], 'import_s': c['ready']['import_s'],
                            'start_maxrss_gib': c['ready']['start_maxrss'] / 2 ** 30,
                            'ready_maxrss_gib': c['ready']['ready_maxrss'] / 2 ** 30} for c in self.w]
        self.log('SCORING WORKERS READY: %d fresh interpreters (score_worker.py by path), start %s s, ru_maxrss at exec '
                 '%s GiB, after imports %s GiB; parent peak when started %.2f GiB' % (
                     len(self.w), [round(x['import_s'], 1) for x in self.start_info],
                     [round(x['start_maxrss_gib'], 3) for x in self.start_info],
                     [round(x['ready_maxrss_gib'], 3) for x in self.start_info], self.parent_rss_at_start / 2 ** 30))

    def run(self, chunks, soft, seconds_max, n_units, on_done):
        """chunks[k]: list of tasks for worker k. n_units[ci] = settings x images (for the task deadline).
        on_done(ci, res, refused, peak, seconds). Returns True if complete, False if the overall deadline cancelled."""
        import pickle
        require(self.dead is None, 'SCORING_POOL_DEAD: ' + str(self.dead))
        deadline = time.time() + seconds_max
        for c, tasks in zip(self.w, chunks):
            c['job'] = None
            if not tasks:
                continue
            seq = c['seq']
            tmp = c['inbox'] / ('job_%06d.pkl.partial' % seq)
            with open(tmp, 'wb') as f:
                pickle.dump({'tasks': tasks, 'soft': list(soft)}, f, protocol=4)
            os.replace(tmp, c['inbox'] / ('job_%06d.pkl' % seq))
            c['job'], c['current'], c['since'] = seq, None, time.time()
        complete = True
        cancelled = False
        while any(c['job'] is not None for c in self.w):
            now = time.time()
            if now > deadline and not cancelled:
                cancelled, complete = True, False
                for c in self.w:
                    if c['job'] is not None:
                        (c['inbox'] / ('cancel_%06d' % c['job'])).touch()
                self.log('SCORING DEADLINE (%.0f s) reached: remaining components cancelled (NOT_SCORED_DEADLINE); '
                         'running tasks still finish under their own deadline' % seconds_max)
            for c in self.w:
                if c['job'] is None:
                    continue
                for ev in self._events(c):
                    e = ev['event']
                    if ev.get('seq') != c['job']:
                        continue
                    if e == 'task_start':
                        c['current'], c['since'] = ev['ci'], time.time()
                    elif e == 'task_done':
                        on_done(ev['ci'], [(limb, [tuple(s) for s in s3], np.asarray(rows, float))
                                           for limb, s3, rows in ev['res']], ev['refused'], ev['peak'], ev['seconds'])
                        c['current'], c['since'] = None, time.time()
                    elif e == 'error':
                        self._fail('SCORER_REFUSED in a fresh scoring process (component %s): %s' % (ev['ci'], ev['error']))
                    elif e == 'job_done':
                        c['job'], c['current'] = None, None
                        c['seq'] += 1
                if c['job'] is None:
                    continue
                self._check_alive(c)
                if c['current'] is not None:
                    lim = task_timeout(self.cfg, 1, n_units[c['current']])
                    if now - c['since'] > lim:
                        self._fail('SCORING_TASK_TIMEOUT: component %s exceeded %.0f s' % (c['current'], lim))
                elif now - c['since'] > self.cfg.get('score_start_s', SCORE_START_S):
                    self._fail('SCORING_JOB_NOT_PICKED_UP: worker idle for %.0f s with a job waiting' % (now - c['since']))
            time.sleep(0.05)
        return complete

    def close(self):
        import shutil
        for c in getattr(self, 'w', []):
            try:
                (c['inbox'] / 'STOP').touch()
            except OSError:
                pass
        end = time.time() + 5
        for c in getattr(self, 'w', []):
            while c['p'].poll() is None and time.time() < end:
                time.sleep(0.05)
            if c['p'].poll() is None:
                _kill_group(c['p'])
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self.dead is None:
            self.dead = 'CLOSED'


def score_settings(proj, cfg, ctx, maps, disks, settings, seconds_max, log, components=None, soft=COUNT_CODES,
                   pool=None):
    """table (S, len(components), 4) of tp, fp, fn, matched_iou_sum over the selected validation components, plus
    complete flag and stats. score_mode 'subprocess' (the only mode the real kernel allows) uses the run's
    ScoringPool (ctx['pool'] or `pool`; a temporary pool for small callers such as the tests); 'inprocess' exists
    for the forced-fault test only."""
    comps = list(range(len(ctx['comps']))) if components is None else list(components)
    pos = {ci: k for k, ci in enumerate(comps)}
    limbs = sorted({s[3] for s in settings}, key=lambda v: -1 if v is None else v)
    groups = [(limb, [s[:3] for s in settings if s[3] == limb]) for limb in limbs]
    index = {s: i for i, s in enumerate(settings)}
    table = np.full((len(settings), len(comps), 4), np.nan)
    stats = {'mode': cfg['score_mode'], 'tasks': len(comps), 'done': 0, 'child_peak_rss_bytes': [],
             'refused': {}, 'task_seconds_max': 0.0, 'soft': list(soft)}
    t_start = time.time()

    def put(ci, res, refused, peak=None, seconds=0.0):
        for limb, s3, rows in res:
            for k, s in enumerate(s3):
                table[index[(s[0], s[1], s[2], limb)], pos[ci]] = rows[k]
        stats['refused'].update(refused)
        stats['done'] += 1
        if peak is not None:
            stats['child_peak_rss_bytes'].append(peak)
        stats['task_seconds_max'] = max(stats['task_seconds_max'], seconds)

    def task_of(ci):
        ids = ctx['image_ids'][ci]
        return (ci, ctx['comps'][ci][0], list(ids), {i: maps[i] for i in ids}, {i: disks[i] for i in ids}, groups)

    complete = True
    if cfg['score_mode'] == 'inprocess':
        for ci in comps:
            _, gt, ids, m, d, g = task_of(ci)
            try:
                put(ci, *score_component(proj, gt, ids, m, d, g, soft))
            except Exception as e:  # noqa: BLE001
                raise Refusal('SCORER_REFUSED (in-process): %s: %s' % (type(e).__name__, e))
    else:
        require(cfg['score_mode'] == 'subprocess', 'SCORE_MODE')
        own = None
        pool = pool or ctx.get('pool')
        if pool is None:
            pool = own = ScoringPool(proj, cfg, cfg['score_procs'], log)
        try:
            n = len(pool.w)
            chunks = [[] for _ in range(n)]
            load = [0] * n
            for ci in sorted(comps, key=lambda c: -len(ctx['image_ids'][c])):   # balance by images
                k = load.index(min(load))
                chunks[k].append(task_of(ci))
                load[k] += len(ctx['image_ids'][ci])
            units = {ci: len(settings) * len(ctx['image_ids'][ci]) for ci in comps}
            complete = pool.run(chunks, soft, seconds_max, units, put)
            stats['child_start'] = pool.start_info
        finally:
            if own is not None:
                own.close()
    peaks = stats['child_peak_rss_bytes']
    stats['child_peak_rss_max_gib'] = (max(peaks) / 2 ** 30) if peaks else None
    stats['parent_peak_rss_gib'] = rss_peak_bytes() / 2 ** 30
    stats['seconds'] = time.time() - t_start
    stats['child_peak_rss_bytes'] = max(peaks) if peaks else None
    if peaks and max(peaks) > CHILD_RSS_WARN:
        log('WARNING: a scoring process peaked at %.2f GiB (scorer refuses above 2 GiB)' % (max(peaks) / 2 ** 30))
    if stats['refused']:
        log('COUNT-BUDGET REFUSALS (soft here): %s' % json.dumps(stats['refused'], sort_keys=True))
    return table, complete, stats


def pooled_row(pp, rows):
    if np.isnan(rows).any():
        return None
    p = pp.pooled(rows)
    p['sq'] = p['matched_iou_sum'] / p['tp'] if p['tp'] else 0.0
    den = p['tp'] + 0.5 * p['fp'] + 0.5 * p['fn']
    p['rq'] = p['tp'] / den if den else 0.0
    return p


def tie_key(s):
    tl, th, a, limb = s
    return (-tl, th, a, 0 if limb is None else 1, -(limb or 0))   # higher T_low, lower T_high, smaller A, no mask


def best_setting(pp, table, settings, members):
    best, best_pq = None, None
    for si in sorted(range(len(settings)), key=lambda i: tie_key(settings[i])):
        rows = table[si][members]
        if np.isnan(rows).any():
            continue
        q = pp.pooled(rows)['pq']
        if best is None or q > best_pq:
            best, best_pq = si, q
    return best


def cross_validate(pp, table, settings):
    """Component index mod 5, as pp_minarea / pp_hysteresis (the honest, selection-free number)."""
    n = table.shape[1]
    idx = np.arange(n)
    folds = idx % pp.N_FOLDS
    held = np.zeros((n, 4))
    chosen = {}
    for f in range(pp.N_FOLDS):
        test, train = idx[folds == f], idx[folds != f]
        if not len(test) or not len(train):
            continue
        si = best_setting(pp, table, settings, train)
        if si is None:
            return None
        chosen[f] = list(settings[si])
        held[test] = table[si][test]
    return {'chosen_per_fold': chosen, 'cv': pp.pooled(held)}


def rule_table_rows(pp, table, settings, refused=None):
    rows = []
    refused = refused or {}
    for si, s in enumerate(settings):
        p = pooled_row(pp, table[si])
        if p:
            st = {'status': 'SCORED', **p}
        elif setting_key(s) in refused:
            r = refused[setting_key(s)]
            st = {'status': 'REFUSED_COUNT:' + r['code'], 'refusal_numbers': r}
        else:
            st = {'status': 'NOT_SCORED_DEADLINE'}
        rows.append({'T_low': s[0], 'T_high': s[1], 'A': s[2], 'limb': s[3], **st})
    return rows


def periodic_decision(rows, refused, best_pq):
    """Coordinator ruling for PERIODIC (and smoke) validations: any count-budget refusal makes the whole validation
    "not scoreable" (recorded with its numbers); no checkpoint is taken from it; training goes on. (The final
    validation calls the scorer with soft=(), so the same refusal stops the run there.)
    Returns (scoreable, take_checkpoint, best_quick_pq_of_this_validation)."""
    scored = [r for r in rows if r['status'] == 'SCORED']
    top = max((r['pq'] for r in scored), default=-1.0)
    scoreable = not refused and bool(scored)
    return scoreable, bool(scoreable and top > best_pq), top


def verdict_of(pq):
    return 'STOP' if pq < STOP_THRESHOLD else ('CONTINUE' if pq >= CONTINUE_THRESHOLD else 'MARGINAL_0.27_TO_0.30')


# ============================================================ output helpers
class Out:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.log_path = self.root / 'RUN_LOG.txt'
        self.t0 = time.time()

    def log(self, msg):
        line = '[%s +%7.1fs] %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), time.time() - self.t0, msg)
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def banner(self, msg):
        for _ in range(3):
            self.log('#' * 100)
        self.log('#### ' + msg)
        for _ in range(3):
            self.log('#' * 100)

    def json(self, name, obj):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.partial')
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable) + '\n')
        os.replace(tmp, p)
        return p

    def progress(self, obj):
        with open(self.root / 'PROGRESS.jsonl', 'a') as f:
            f.write(json.dumps(obj, sort_keys=True, default=_jsonable) + '\n')


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)


def gpu_status():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().replace('\n', ' | ') if r.returncode == 0 else 'nvidia-smi rc=%d' % r.returncode
    except Exception as e:  # noqa: BLE001
        return 'nvidia-smi unavailable (%s)' % type(e).__name__


def hardware(torch):
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'numpy': np.__version__}
    try:
        info['mem_total_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        info['cuda'] = torch.version.cuda
        info['gpus'] = [{'name': torch.cuda.get_device_name(i),
                         'total_gb': round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1)}
                        for i in range(torch.cuda.device_count())]
    return info


def save_maps_npz(out_dir, maps, schema, extra):
    """INDEX.json + <id>.npz (key p, uint8 2048x2048): the format pp_hysteresis.load_map / select_r4.load_run read."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    images = {}
    for image_id in sorted(maps):
        buf = io.BytesIO()
        np.savez_compressed(buf, p=maps[image_id])
        raw = buf.getvalue()
        (out_dir / (image_id + '.npz')).write_bytes(raw)
        images[image_id] = {'file': image_id + '.npz', 'sha256': sha256(raw), 'shape': list(maps[image_id].shape)}
    index = dict(extra, schema=schema, images=images)
    (out_dir / 'INDEX.json').write_text(json.dumps(index, sort_keys=True, indent=1, default=_jsonable) + '\n')
    return index


def discover_dataset(input_root=Path('/kaggle/input')):
    """Two competition-scoped mount names x two layouts (as bootstrap_r3.discover_dataset). No enumeration."""
    found = []
    for base in (input_root / COMPETITION, input_root / 'competitions' / COMPETITION):
        for root in (base, base / 'MAGFiLO_1.0_Kaggle_2026'):
            if (root / ANNOTATION_REL).is_file() and (root / 'train/train_images').is_dir():
                found.append(root)
    require(len(found) == 1, 'OFFICIAL_MOUNT_MISSING_OR_AMBIGUOUS:%d' % len(found))
    return found[0]


def list_test_inputs(proj, cfg):
    """Test images in name order; each id refused by NAME if the split plan puts it in train, validation or holdout,
    and each file refused unless its bytes match DATA_HASHES. Nothing is read before those checks."""
    pt = proj.pt
    forbidden = pt.load_forbidden(cfg['split_path'], cfg['pins']['split'])
    require(sha256_file(cfg['data_hashes_path']) == cfg['pins']['data_hashes'], 'DATA_HASHES_PIN')
    hash_index = pt.load_hash_index(cfg['data_hashes_path'])
    test_dir = Path(cfg['dataset_root']) / 'test/test_images'
    return test_dir, pt.iter_test_images(test_dir, forbidden, cfg['test_images'], hash_index)


def load_frozen_rule(path):
    fr = json.loads(Path(path).read_text())
    r = fr['rule']
    require(set(r) == {'T_low', 'T_high', 'A', 'limb'}, 'FROZEN_RULE_SCHEMA')
    return fr, {'T_low': float(r['T_low']), 'T_high': float(r['T_high']), 'A': int(r['A']),
                'limb': None if r['limb'] is None else float(r['limb'])}


def finalize_csv(pt, out_root, rows, final_name, validate_fn, rederive_fn):
    """M1: write under a non-submittable temporary name, validate, re-derive from the stored maps, check LF-only
    bytes, and only then rename. Any failure removes the temporary file: no file with a submittable name remains."""
    out_root = Path(out_root)
    tmp = out_root / 'UNVALIDATED_DO_NOT_SUBMIT.tmp'
    final = out_root / final_name
    require(not final.exists(), 'FINAL_CSV_ALREADY_EXISTS')
    try:
        pt.write_csv(tmp, rows)
        raw = tmp.read_bytes()
        require(raw.startswith(b'filament_id,segmentation_rle\n') and b'\r' not in raw and raw.endswith(b'\n'),
                'CSV_BYTES_NOT_LF_ONLY')
        summary = validate_fn(tmp)
        require(isinstance(summary, dict) and summary.get('ok') is True, 'CSV_VALIDATOR_NOT_OK')
        require(rederive_fn() == rows, 'TEST_MAPS_DO_NOT_REPRODUCE_CSV')
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + '.partial').unlink(missing_ok=True)
        raise
    return final, summary


# ============================================================ run
def run(cfg):
    out = Out(cfg['out'])
    status = {'schema': SCHEMA, 'mode': cfg['mode'], 'dry': cfg['dry'], 'status': 'STARTED', 'submitted': False,
              'holdout_opened': False, 'test_images_read': 0}
    out.json('STATUS.json', status)
    try:
        _run(cfg, out, status)
    except Exception as e:  # noqa: BLE001
        status.update(status='FAILED', error=repr(e)[:2000], traceback=traceback.format_exc()[-6000:],
                      parent_peak_rss_gib=rss_peak_bytes() / 2 ** 30)
        out.json('STATUS.json', status)
        out.banner('RUN FAILED: %r -- DO NOT SUBMIT FROM THIS RUN' % (e,))
        raise
    return status


def _run(cfg, out, status):
    T0 = time.time()
    proj = get_project(cfg['src_root'])      # scorer pin + thread limits happen here, before torch work
    import torch
    torch.set_num_threads(cfg['threads'])
    if not cfg['dry']:
        require(cfg['score_mode'] == 'subprocess' and not cfg.get('worker_fault') and FAULT_ENV not in os.environ,
                'REAL_CONFIG_LOCKED')
    # Scoring workers start NOW, while this process is small (Linux keeps ru_maxrss across execve: a worker started
    # later would inherit the training process's 4-5 GiB peak and the scorer's 2 GiB guard would refuse in it).
    pool = ScoringPool(proj, cfg, cfg['score_procs'], out.log) if cfg['score_mode'] == 'subprocess' else None
    status['scoring_workers'] = pool.start_info if pool else None
    out.json('STATUS.json', status)
    try:
        _run_inner(cfg, out, status, proj, pool, T0)
    finally:
        if pool is not None:
            pool.close()


def _run_inner(cfg, out, status, proj, pool, T0):
    import torch
    log = out.log
    hw = hardware(torch)
    log('mode %s%s; hardware: %s' % (cfg['mode'], ' (dry)' if cfg['dry'] else '', json.dumps(hw, sort_keys=True)))
    if not cfg['dry']:
        require(torch.cuda.is_available(), 'NO_CUDA: refusing to train on CPU')
        require(cfg['score_mode'] == 'subprocess' and cfg['model'] == 'unet' and cfg['pretrained'] and not cfg['ballast']
                and not cfg.get('worker_fault') and FAULT_ENV not in os.environ,
                'REAL_CONFIG_LOCKED')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed = cfg['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ---- lists and holdout guard (refuses before any image or annotation is opened)
    pins = cfg['pins']
    split_raw = Path(cfg['split_path']).read_bytes()
    census_raw = Path(cfg['census_path']).read_bytes()
    train_rows, val_rows, components, holdout = build_lists(proj.nbd, split_raw, census_raw, pins['split'],
                                                            pins['census'], cfg['counts'])
    require(len(components) == cfg['val_components'], 'VAL_COMPONENTS')
    require(sum(len(r['annotator_image_ids']) for r in val_rows) == cfg['val_groups'], 'VAL_GROUPS')
    allowed = {r['image_id'] for r in train_rows} | {r['image_id'] for r in val_rows}
    require(not (allowed & holdout), 'HOLDOUT_IN_ALLOWED')
    log('lists: train %d, validation %d images (%d components, %d annotator groups); holdout ids excluded: %d '
        '(names from the plan only, never opened)' % (len(train_rows), len(val_rows), len(components),
                                                     cfg['val_groups'], len(holdout)))
    if cfg.get('dataset_root') is None:
        cfg['dataset_root'] = discover_dataset()

    prov = {'schema': SCHEMA, 'config': {k: v for k, v in cfg.items() if k not in ('bundle',)},
            'bundle': cfg.get('bundle'), 'split_plan_sha256': sha256(split_raw), 'pixel_census_sha256': sha256(census_raw),
            'pins': pins, 'hardware': hw, 'seed': seed, 'plan': 'seats/solar-track2-plan-20261111/PLAN.md sha256 '
            '8d274c4321e05078cdcf27998a4a6dc3c324776243fdadc4b14cb9f908be6e19',
            'module_sha256': {name: sha256_file(Path(m.__file__)) for name, m in (
                ('track2.py', sys.modules[__name__]), ('official_scorer.py', proj.scorer),
                ('pp_hysteresis.py', proj.ph), ('pp_morph.py', proj.pm), ('pp_minarea.py', proj.pp),
                ('predict_test.py', proj.pt), ('validate_submission.py', proj.vs),
                ('neural_baseline_r1/data.py', proj.nbd), ('score_reference.py', proj.sr))}}
    out.json('PROVENANCE.json', prov)

    # ---- model and pretrained weights FIRST: a missing download fails in the first two minutes
    model = build_darkness_model() if cfg['model'] == 'darkness' else build_model()
    require(cfg['model'] in ('unet', 'darkness') and (cfg['model'] == 'unet' or cfg['dry']), 'MODEL')
    if cfg['pretrained']:
        prov['pretrained'] = load_pretrained(model.encoder, cfg, log)
    else:
        prov['pretrained'] = {'source': 'NONE (dry mode)'}
    if cfg.get('head_bias') is not None:
        torch.nn.init.constant_(model.head.bias, cfg['head_bias'])
    out.json('PROVENANCE.json', prov)

    # ---- real-size rehearsal ballast (dry only): the parent becomes as big as the real one
    ballast = []
    if cfg['ballast']:
        require(cfg['dry'], 'BALLAST_DRY_ONLY')
        rng = np.random.default_rng(0)
        for name, (shape, dt) in REAL_SHAPES.items():
            a = np.empty(shape, dtype=dt)
            flat = a.reshape(-1).view(np.uint8)
            for k in range(0, flat.size, 1 << 28):      # random bytes: every page written and incompressible
                flat[k:k + (1 << 28)] = rng.integers(0, 256, min(1 << 28, flat.size - k), dtype=np.uint8)
            ballast.append(a)
        log('BALLAST: real-size arrays allocated and touched: %s = %.2f GiB; parent peak RSS now %.2f GiB' % (
            {k: v[0] for k, v in REAL_SHAPES.items()}, sum(b.nbytes for b in ballast) / 2 ** 30, rss_peak_bytes() / 2 ** 30))

    # ---- annotations (pinned), filtered by image id before any polygon is touched
    ann_raw = proj.nbd.read(Path(cfg['dataset_root']) / ANNOTATION_REL, 64 * 1024 ** 2, pins['annotations'])
    annotations = json.loads(ann_raw)
    del ann_raw
    by_group = allowed_annotations(annotations, allowed)
    for r in train_rows + val_rows:
        for g in r['annotator_image_ids']:
            require(len(by_group.get(g, [])) >= 1, 'REQUIRED_ANNOTATION_GROUP')

    # ---- preload (train with soft targets; validation inputs only)
    tr = preload(proj, cfg, train_rows, by_group, allowed, True, log)
    va = preload(proj, cfg, val_rows, by_group, allowed, False, log)
    X = np.stack([t['x'] for t in tr])
    Y = np.stack([t['y'] for t in tr])
    FG = [t['fg'] for t in tr]
    XV = np.stack([v['x'] for v in va])
    val_ids = [v['image_id'] for v in va]
    val_disks = {v['image_id']: v['disk'] for v in va}
    val_row_by_id = {r['image_id']: r for r in val_rows}
    hist = {}
    for t in tr:
        hist[t['annotators']] = hist.get(t['annotators'], 0) + 1
    log('train tensors X %s (%.2f GB) Y %s; foreground share %.4f; fg-crop centres per image min %d; annotators per '
        'train image %s; disk fallbacks train %d val %d; parent peak RSS %.2f GiB' % (
            X.shape, X.nbytes / 2 ** 30, Y.shape, float((Y / 255.0).mean()), min(len(f) for f in FG),
            dict(sorted(hist.items())), sum(t['disk']['fallback'] for t in tr), sum(v['disk']['fallback'] for v in va),
            rss_peak_bytes() / 2 ** 30))
    del tr, va

    # ---- validation truth through the registered loader (score_reference.annotation_frame)
    comps, image_ids = [], []
    for comp in components:
        rows = comp['images']
        sel = [{**r, 'annotation_counts': {g: len(by_group[g]) for g in r['annotator_image_ids']}} for r in rows]
        comps.append((proj.sr.annotation_frame(annotations, sel), None))
        image_ids.append([r['image_id'] for r in rows])
    ctx = {'comps': comps, 'image_ids': image_ids, 'pool': pool}
    del annotations
    log('validation truth frames built: %d components, %d GT rows' % (len(comps), sum(len(c[0]) for c in comps)))

    # ---- B1 fail-fast: scoring preflight in fresh processes, with the parent at its real size
    pre = scoring_preflight(proj, cfg, ctx, log)
    out.json('SCORING_PREFLIGHT.json', pre)

    # ---- model to device
    nparams = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    ngpu = torch.cuda.device_count() if device.type == 'cuda' else 0
    net = model
    if ngpu > 1:
        class _Amp(torch.nn.Module):     # autocast must be entered inside each DataParallel replica thread
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                with torch.autocast('cuda', dtype=torch.float16):
                    return self.m(x)
        net = torch.nn.DataParallel(_Amp(model))
    log('model %s %.2f M params; device %s; GPUs used %d (%s)' % (
        cfg['model'], nparams / 1e6, device, ngpu, 'DataParallel' if ngpu > 1 else 'single'))

    best, evals, step, epochs_done, stopped_early, train_info = train(
        cfg, out, model, net, device, ngpu, X, Y, FG, XV, val_ids, val_disks, ctx, proj, T0)
    status.update(status='TRAINED', epochs_done=epochs_done, steps=step, stopped_early=stopped_early)
    out.json('STATUS.json', status)
    ckdir = out.root / 'checkpoints'
    if best['sha256'] is None and cfg['mode'] == 'smoke':
        # SMOKE only: its single validation was not scoreable (count budgets on a 40-step map). The smoke still has to
        # exercise save + reload, so the current weights are saved, labelled; they are never a result.
        torch.save(model.state_dict(), ckdir / 'best.pt')
        best = {'pq': None, 'epoch': epochs_done, 'step': step, 'sha256': sha256_file(ckdir / 'best.pt'),
                'note': 'SMOKE_ONLY: validation not scoreable; current weights saved to exercise reload'}
        log('SMOKE: validation not scoreable, current weights saved as best.pt only to test save/reload')
    require(best['sha256'] is not None, 'NO_CHECKPOINT')
    model.load_state_dict(torch.load(str(ckdir / 'best.pt'), map_location='cpu', weights_only=True))
    require(sha256_file(ckdir / 'best.pt') == best['sha256'], 'BEST_CHECKPOINT_CHANGED')

    args = dict(cfg=cfg, out=out, proj=proj, model=model, device=device, XV=XV, val_ids=val_ids, val_disks=val_disks,
                val_row_by_id=val_row_by_id, allowed=allowed, ctx=ctx, best=best, evals=evals, status=status,
                pre=pre, train_info=train_info, T0=T0, epochs_done=epochs_done, stopped_early=stopped_early)
    if cfg['mode'] == 'smoke':
        smoke_stage(**args)
    else:
        final_stage(**args)
    del ballast


def scoring_preflight(proj, cfg, ctx, log):
    """Score the largest validation component twice through the real path: all-zero maps (PQ 0, all FN) and the
    union of its GT masks (an oracle map, PQ well above 0.3). A refusal (e.g. SCORER_RSS_BUDGET) stops the run here,
    minutes in, instead of after training."""
    t0 = time.time()
    ci = max(range(len(ctx['comps'])), key=lambda i: len(ctx['comps'][i][0]))
    gt = ctx['comps'][ci][0]
    ids = ctx['image_ids'][ci]
    img_of = gt['filament_id'].str.split('_', n=1).str[0].str.split('-', n=1).str[1]
    perfect, zero = {}, {}
    for i in ids:
        m = np.zeros((H, W), np.uint8)
        for c in gt.loc[img_of == i, 'segmentation_rle']:
            m |= proj.mask_util.decode({'size': [H, W], 'counts': c.encode('ascii')}).astype(np.uint8)
        perfect[i] = (m * 255).astype(np.uint8)
        zero[i] = np.zeros((H, W), np.uint8)
    disks = {i: {'cy': 1024.0, 'cx': 1024.0, 'R': 1000.0, 'fallback': True} for i in ids}
    mini = {'comps': [ctx['comps'][ci]], 'image_ids': [ids]}
    s = [(0.5, 0.5, 0, None)]
    log('SCORING PREFLIGHT: component %d (%d images, %d GT rows), parent peak RSS %.2f GiB, mode %s' % (
        ci, len(ids), len(gt), rss_peak_bytes() / 2 ** 30, cfg['score_mode']))
    tz, cz, sz = score_settings(proj, cfg, mini, zero, disks, s, 300, log, soft=(), pool=ctx.get('pool'))  # all hard
    tp, cp, sp = score_settings(proj, cfg, mini, perfect, disks, s, 300, log, soft=(), pool=ctx.get('pool'))
    require(cz and cp, 'SCORING_PREFLIGHT_INCOMPLETE')
    rz, rp = proj.pp.pooled(tz[0]), proj.pp.pooled(tp[0])
    require(rz['tp'] == 0 and rz['fn'] == len(gt), 'SCORING_PREFLIGHT_ZERO_MAP')
    require(rp['pq'] > 0.3, 'SCORING_PREFLIGHT_ORACLE_PQ %.4f' % rp['pq'])
    res = {'component': ci, 'images': len(ids), 'gt_rows': len(gt), 'zero_map': rz, 'oracle_union_map': rp,
           'parent_peak_rss_gib': rss_peak_bytes() / 2 ** 30,
           'child_peak_rss_max_gib': max(x for x in (sz['child_peak_rss_max_gib'], sp['child_peak_rss_max_gib'])
                                         if x is not None) if cfg['score_mode'] == 'subprocess' else None,
           'child_start': sp.get('child_start'), 'mode': cfg['score_mode'], 'seconds': time.time() - t0}
    cs = res['child_start'] or []
    res['child_start_seconds'] = [x['import_s'] for x in cs]
    log('SCORING PREFLIGHT OK: zero-map PQ %.4f (FN %d), oracle-map PQ %.4f; parent peak %.2f GiB, scoring child '
        'peak %s GiB, scoring child start %s s (limit %d s), child ru_maxrss at exec %s GiB, %.0f s' % (
            rz['pq'], rz['fn'], rp['pq'], res['parent_peak_rss_gib'],
            None if res['child_peak_rss_max_gib'] is None else round(res['child_peak_rss_max_gib'], 3),
            [round(x['import_s'], 1) for x in cs], cfg.get('score_start_s', SCORE_START_S),
            [round(x['start_maxrss_gib'], 3) for x in cs], res['seconds']))
    return res


def train(cfg, out, model, net, device, ngpu, X, Y, FG, XV, val_ids, val_disks, ctx, proj, T0):
    import torch
    log = out.log
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    GS = getattr(torch.amp, 'GradScaler', None)
    scaler = (GS('cuda') if GS else torch.cuda.amp.GradScaler()) if device.type == 'cuda' else None
    steps_per_epoch = max(1, (cfg['crops_per_image'] * len(X)) // cfg['batch'])
    total_steps = steps_per_epoch * cfg['epochs']
    if cfg['max_steps']:
        total_steps = min(total_steps, cfg['max_steps'])
    warm = max(1, min(int(cfg['warmup_epochs'] * steps_per_epoch), total_steps // 2 or 1))

    def lr_at(step):
        if step < warm:
            return cfg['lr'] * (step + 1) / warm
        q = (step - warm) / max(1, total_steps - warm)
        return cfg['lr'] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(q, 1.0))))
    ds = CropDataset(X, Y, FG, cfg['crop'], cfg['seed'], cfg['fg_crop_p'])
    ckdir = out.root / 'checkpoints'
    ckdir.mkdir()
    best = {'pq': -1.0, 'epoch': None, 'sha256': None}
    evals = []
    step = 0
    nonfinite = 0
    stopped_early = None
    rates = []
    log('training: %d epochs x %d steps (batch %d, crop %d at work %d) = %d steps; eval every %d epochs' % (
        cfg['epochs'], steps_per_epoch, cfg['batch'], cfg['crop'], cfg['work'], total_steps, cfg['eval_every']))
    log('gpu at start: %s' % gpu_status())
    train_t0 = time.time()
    epoch = 0
    for epoch in range(cfg['epochs']):
        if step >= total_steps:
            break
        net.train()
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg['batch'], sampler=EpochSampler(epoch, steps_per_epoch * cfg['batch']),
            num_workers=cfg['workers'], pin_memory=device.type == 'cuda', drop_last=True,
            multiprocessing_context='fork' if cfg['workers'] else None)
        e0 = time.time()
        wait = 0.0
        losses = []
        seen = 0
        tw = time.time()
        for xb, yb in loader:
            if step >= total_steps:
                break
            wait += time.time() - tw
            for gparam in opt.param_groups:
                gparam['lr'] = lr_at(step)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if ngpu > 1:
                logits = net(xb)
            else:
                with autocast_ctx(device):
                    logits = net(xb)
            loss, parts = loss_fn(logits.float(), yb)
            if not torch.isfinite(loss):
                nonfinite += 1
                log('NON-FINITE LOSS at step %d (%d so far)' % (step, nonfinite))
                require(nonfinite < 20, 'NONFINITE_LOSS_REPEATED')
                opt.zero_grad(set_to_none=True)
                tw = time.time()
                continue
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.detach()))
            seen += xb.shape[0]
            step += 1
            if step % cfg['log_every'] == 0 or step <= 3:
                dt = time.time() - e0
                log('step %d/%d epoch %d loss %.4f (bce %.4f dice %.3f) lr %.2e  %.1f crops/s  data-wait %.0f%%  gpu: %s'
                    % (step, total_steps, epoch + 1, losses[-1], parts['bce'], parts['dice'], lr_at(step),
                       seen / max(dt, 1e-6), 100 * wait / max(dt, 1e-6),
                       gpu_status() if step % 100 == 0 or step <= 3 or cfg['mode'] == 'smoke' else '-'))
            tw = time.time()
        dt = time.time() - e0
        rec = {'kind': 'epoch', 'epoch': epoch + 1, 'step': step, 'loss_mean': float(np.mean(losses)) if losses else None,
               'crops_per_s': seen / max(dt, 1e-6), 'full_image_equiv_per_s': seen / max(dt, 1e-6) * (cfg['crop'] / cfg['work']) ** 2,
               'data_wait_fraction': wait / max(dt, 1e-6), 'seconds': dt, 'elapsed': time.time() - T0,
               'max_mem_gb': (torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else None}
        rates.append(rec)
        out.progress(rec)
        log('EPOCH %d done: loss %.4f, %.1f crops/s (%.1f full work-size images/s), data-wait %.0f%%, %.0f s, peak GPU mem %s GB'
            % (epoch + 1, rec['loss_mean'] if rec['loss_mean'] is not None else float('nan'), rec['crops_per_s'],
               rec['full_image_equiv_per_s'], 100 * rec['data_wait_fraction'], dt,
               None if rec['max_mem_gb'] is None else round(rec['max_mem_gb'], 2)))
        last_epoch = (epoch + 1 == cfg['epochs']) or step >= total_steps
        over = time.time() - train_t0 > cfg['train_seconds_max']
        if (epoch + 1) % cfg['eval_every'] == 0 or last_epoch or over:
            te = time.time()
            maps = dict(infer_maps(model, XV, cfg, device, False, val_ids))
            table, complete, st = score_settings(proj, cfg, ctx, maps, val_disks, QUICK_SETTINGS,
                                                 cfg['quick_eval_seconds_max'], log)
            rows = rule_table_rows(proj.pp, table, QUICK_SETTINGS, st['refused'])
            scoreable, take, top = periodic_decision(rows, st['refused'], best['pq'])
            if not scoreable:
                log('VALIDATION epoch %d NOT SCOREABLE: %s; no checkpoint from this evaluation, training goes on' % (
                    epoch + 1, json.dumps(st['refused'], sort_keys=True)))
            ev = {'kind': 'eval', 'epoch': epoch + 1, 'step': step, 'quick_rows': rows, 'best_quick_pq': top,
                  'scoreable': scoreable, 'not_scoreable': None if scoreable else st['refused'],
                  'complete': complete, 'scoring': st, 'seconds': time.time() - te, 'elapsed': time.time() - T0}
            evals.append(ev)
            out.progress(ev)
            log('VALIDATION epoch %d (no TTA, quick grid): ' % (epoch + 1) + '; '.join(
                '(%s,%s,%s) PQ %s' % (r['T_low'], r['T_high'], r['A'], ('%.4f' % r['pq']) if r['status'] == 'SCORED' else 'n/a')
                for r in rows) + '  -> best %.4f  [registered baseline 0.2142; stop rule %.2f]; parent peak %.2f GiB, '
                'scoring child peak %.3f GiB, %.0f s' % (
                    top, STOP_THRESHOLD, st['parent_peak_rss_gib'], st['child_peak_rss_max_gib'] or -1, ev['seconds']))
            if take:
                path = ckdir / 'best.pt'
                torch.save(model.state_dict(), path)
                best = {'pq': top, 'epoch': epoch + 1, 'step': step, 'sha256': sha256_file(path)}
                log('new best checkpoint epoch %d quick PQ %.4f sha256 %s' % (epoch + 1, top, best['sha256']))
            del maps
        if over:
            stopped_early = 'TRAIN_TIME_CAP after epoch %d' % (epoch + 1)
            log('TRAIN TIME CAP reached (%.0f s): stopping training after epoch %d' % (cfg['train_seconds_max'], epoch + 1))
            break
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
               ckdir / 'last_fp16.pt')   # fp16 to keep the output small; best.pt stays fp32
    log('training finished: %d epochs, %d steps, %.0f s; best epoch %s quick PQ %.4f' % (
        epoch + 1, step, time.time() - train_t0, best['epoch'], best['pq']))
    info = {'crops_per_s_last_epoch': rates[-1]['crops_per_s'] if rates else None,
            'data_wait_last_epoch': rates[-1]['data_wait_fraction'] if rates else None,
            'train_seconds': time.time() - train_t0, 'max_gpu_mem_gb': rates[-1]['max_mem_gb'] if rates else None}
    return best, evals, step, epoch + 1, stopped_early, info


def self_check_test_path(proj, cfg, model, device, frozen_path, chosen, val_maps_by_id, val_disks, val_row_by_id,
                         allowed, ids, log, xv_of, strict_batch=True):
    """M3 in the kernel: the exact test-path function (predict_image_rows, rule read back from FROZEN_RULE.json, batch 1)
    run on validation images, compared with
      (a) the validation path at the SAME batch size (validation inputs XV, batch 1, the chosen setting, the validation
          disk geometry): must be identical, always; any difference is a test-path fault;
      (b) the batch-4 validation maps that were actually scored: in the full run at most BATCH_TOL_MAX_LEVELS levels
          on at most BATCH_TOL_SHARE of the pixels (r3b, ruled from the r3 smoke run); the smoke run only reports (b).
    Every failure names the largest absolute difference and the number of differing pixels."""
    _fr, rule = load_frozen_rule(frozen_path)
    out = []
    for image_id in ids:
        im = read_train_image(proj.nbd, proj.Image, cfg['dataset_root'], val_row_by_id[image_id], allowed)
        arr_t, rows_t = predict_image_rows(proj, model, cfg, device, image_id, im, rule)
        (_, ref1), = list(infer_maps(model, xv_of(image_id)[None], cfg, device, cfg['tta'], [image_id]))
        arr_1 = limb_mask_uint8(ref1, val_disks[image_id], chosen[3])
        rows_1 = proj.pt.rows_from_uint8(image_id, arr_1, tuple(chosen[:3]))
        arr_4 = limb_mask_uint8(val_maps_by_id[image_id], val_disks[image_id], chosen[3])
        rows_4 = proj.pt.rows_from_uint8(image_id, arr_4, tuple(chosen[:3]))
        d1 = np.abs(arr_t.astype(np.int16) - arr_1.astype(np.int16))
        d4 = np.abs(arr_t.astype(np.int16) - arr_4.astype(np.int16))
        rec = {'image_id': image_id, 'rows_test_path': len(rows_t),
               'same_batch': {'rows': len(rows_1), 'rows_equal': rows_t == rows_1, 'max_abs_diff': int(d1.max()),
                              'pixels_differing': int((d1 > 0).sum())},
               'scored_batch4': {'rows': len(rows_4), 'rows_equal': rows_t == rows_4, 'max_abs_diff': int(d4.max()),
                                 'pixels_differing': int((d4 > 0).sum())}}
        out.append(rec)
        require(int(d1.max()) == 0 and rows_t == rows_1,
                'TEST_PATH_SELF_CHECK_FAILED on %s (same batch size): max abs diff %d, %d pixels differ, rows %d vs %d'
                % (image_id, int(d1.max()), int((d1 > 0).sum()), len(rows_t), len(rows_1)))
        if strict_batch:
            # r3b: tolerance instead of identity (see BATCH_TOL_*); rows against the batch-4 maps are recorded in
            # `rec`, not required to be equal, because a 1-level difference can move an instance's edge by a pixel.
            require(int(d4.max()) <= BATCH_TOL_MAX_LEVELS and int((d4 > 0).sum()) <= BATCH_TOL_SHARE * d4.size,
                    'TEST_PATH_SELF_CHECK_FAILED on %s (batch 1 vs the scored batch-4 maps): max abs diff %d, %d pixels '
                    'differ, rows %d vs %d' % (image_id, int(d4.max()), int((d4 > 0).sum()), len(rows_t), len(rows_4)))
    log('TEST-PATH SELF-CHECK on %d validation images (strict batch identity: %s): %s' % (
        len(ids), strict_batch, [(o['rows_test_path'], o['same_batch']['max_abs_diff'], o['scored_batch4']['max_abs_diff'],
                                  o['scored_batch4']['pixels_differing']) for o in out]))
    return out


def pick_check_ids(maps, chosen, k):
    cut = int(math.ceil(255 * chosen[1] - 1e-9))
    return sorted(maps, key=lambda i: (-int((maps[i] >= cut).sum()), i))[:k]


def final_stage(cfg, out, proj, model, device, XV, val_ids, val_disks, val_row_by_id, allowed, ctx, best, evals,
                status, pre, train_info, T0, epochs_done, stopped_early):
    log = out.log
    te = time.time()
    maps = dict(infer_maps(model, XV, cfg, device, cfg['tta'], val_ids))
    save_maps_npz(out.root / 'val_maps', maps, 'qmai.solar-track2-r3.val-maps/v1',
                  {'weights_sha256': best['sha256'], 'tta': 'flip4' if cfg['tta'] else 'none', 'limb_mask_applied': False,
                   'quantisation': 'rint(255*p), uint8, 2048x2048, bilinear upsample from the work resolution',
                   'disk_geometry': {k: val_disks[k] for k in sorted(val_disks)}})
    log('validation maps written (%d, TTA %s) in %.0f s' % (len(maps), cfg['tta'], time.time() - te))
    settings = cfg.get('full_settings') or full_settings()
    # FINAL validation feeds the frozen rule and the CSV: every refusal is hard here (coordinator ruling)
    table, complete, st = score_settings(proj, cfg, ctx, maps, val_disks, settings, cfg['final_eval_seconds_max'], log,
                                         soft=())
    rows = rule_table_rows(proj.pp, table, settings, st['refused'])
    out.json('RULE_TABLE.json', {'settings_order': '(T_low, T_high, A, limb)', 'checkpoint_sha256': best['sha256'],
                                 'tta': cfg['tta'], 'complete': complete, 'scoring': st, 'rows': rows,
                                 'note': 'counts per annotator-image group (pinned v6 scorer, IoU > 0.5), pooled over '
                                         'the validation components; same unit as the registered 0.2142'})
    with open(out.root / 'RULE_TABLE.csv', 'w') as f:
        f.write('T_low,T_high,A,limb,status,tp,fp,fn,matched_iou_sum,pq,sq,rq\n')
        for r in rows:
            f.write(','.join(str(r.get(k, '')) for k in ('T_low', 'T_high', 'A', 'limb', 'status', 'tp', 'fp', 'fn',
                                                         'matched_iou_sum', 'pq', 'sq', 'rq')) + '\n')
    bi = best_setting(proj.pp, table, settings, np.arange(table.shape[1]))
    require(bi is not None, 'NO_SCORED_SETTING')
    chosen = settings[bi]
    best_row = rows[bi]
    cv = cross_validate(proj.pp, table, settings)
    plain = rows[settings.index((0.5, 0.5, 512, None))] if (0.5, 0.5, 512, None) in settings else None
    registered = rows[settings.index((0.15, 0.8, 1024, None))] if (0.15, 0.8, 1024, None) in settings else None
    d1 = best_row['pq']
    verdict = verdict_of(d1)
    stop = {'rule': 'PLAN.md section 4 gate D1: validation PQ (best grid point) >= 0.30 continue; 0.27-0.30 '
                    'continue only with time; < 0.27 stop track 2',
            'checkpoint': 'end of the planned %d-epoch run (best periodic checkpoint)' % cfg['epochs'],
            'epochs_done': epochs_done, 'stopped_early': stopped_early, 'validation_pq_best_grid': d1,
            'validation_cv_pq': cv['cv']['pq'] if cv else None, 'threshold': STOP_THRESHOLD, 'verdict': verdict,
            'submit_allowed_by_stop_rule': verdict != 'STOP', 'chosen_rule': list(chosen),
            'test_stage_runs': verdict != 'STOP', 'registered_baseline_validation_pq': 0.2142481}
    out.json('STOP_RULE.json', stop)
    score = {'schema': SCHEMA + '.score', 'status': 'COMPLETE_VALIDATION_UNACCEPTED', 'holdout_scored': False,
             'test_inference': False, 'weights': {'sha256': best['sha256'], 'file': 'checkpoints/best.pt'},
             'weights_sha256': best['sha256'], 'best_epoch': best['epoch'],
             'pooled_area512': ({k: plain[k] for k in ('tp', 'fp', 'fn', 'pq')} if plain and plain['status'] == 'SCORED' else None),
             'registered_rule_0.15_0.8_1024': registered, 'chosen': best_row, 'cv': cv, 'scoring_preflight': pre,
             'train': train_info, 'val_maps_index_sha256': sha256_file(out.root / 'val_maps/INDEX.json'),
             'periodic_evals': evals}
    out.json('SCORE.json', score)
    frozen_path = out.json('FROZEN_RULE.json', {
        'schema': SCHEMA + '.frozen-rule', 'frozen_at_epoch': time.time(),
        'rule': {'T_low': chosen[0], 'T_high': chosen[1], 'A': chosen[2], 'limb': chosen[3]},
        'weights_sha256': best['sha256'], 'tta': cfg['tta'], 'work': cfg['work'], 'verdict': verdict,
        'selection': 'highest pooled validation PQ over RULE_TABLE (ties: higher T_low, lower T_high, smaller A, '
                     'no limb mask); 5-fold CV reported, not used',
        'note': 'written before any test image is read; with verdict STOP no test image is read at all'})
    if verdict == 'STOP':
        out.banner('STOP RULE: validation PQ %.4f < %.2f at the planned checkpoint. TRACK 2 STOPS. NO TEST IMAGE IS READ '
                   'AND NO CSV IS WRITTEN.' % (d1, STOP_THRESHOLD))
        status.update(status='COMPLETE_STOPPED', verdict=verdict, validation_pq=d1, seconds_total=time.time() - T0)
        out.json('STATUS.json', status)
        log('DONE status COMPLETE_STOPPED in %.0f s. This kernel never submits.' % (time.time() - T0))
        return
    out.banner('STOP RULE PASSED: validation PQ %.4f (CV %.4f) >= %.2f, verdict %s. Rule %s. Submitting still needs '
               'review, the registered revision and the owner\'s go-ahead.' % (
                   d1, cv['cv']['pq'] if cv else float('nan'), STOP_THRESHOLD, verdict, list(chosen)))
    status.update(status='VALIDATED', verdict=verdict, validation_pq=d1)
    out.json('STATUS.json', status)
    try:
        check_ids = pick_check_ids(maps, chosen, cfg['self_check_images'])
        pos = {i: k for k, i in enumerate(val_ids)}
        sc = self_check_test_path(proj, cfg, model, device, frozen_path, chosen, maps, val_disks, val_row_by_id,
                                  allowed, check_ids, log, lambda i: XV[pos[i]])
        out.json('TEST_PATH_SELF_CHECK.json', sc)
        del maps
        test_stage(proj, cfg, out, model, device, frozen_path, best['sha256'], status, log)
        status.update(status='COMPLETE')
    except Exception as e:  # noqa: BLE001  keep the validation outputs; no submittable CSV can exist (finalize_csv)
        status.update(status='COMPLETE_VALIDATION_ONLY_TEST_FAILED', test_error=repr(e)[:1000])
        log('TEST STAGE FAILED: %r' % (e,))
        log(traceback.format_exc()[-3000:])
    status['seconds_total'] = time.time() - T0
    out.json('STATUS.json', status)
    log('DONE status %s in %.0f s. This kernel never submits.' % (status['status'], time.time() - T0))


def test_stage(proj, cfg, out, model, device, frozen_path, weights_sha, status, log):
    pt, vs = proj.pt, proj.vs
    fr, rule = load_frozen_rule(frozen_path)
    require(fr['verdict'] != 'STOP' and fr['weights_sha256'] == weights_sha, 'TEST_STAGE_NOT_ALLOWED')
    test_dir, chosen_imgs = list_test_inputs(proj, cfg)
    rule3 = (rule['T_low'], rule['T_high'], rule['A'])
    maps, rows = {}, []
    t0 = time.time()
    for image_id, path in chosen_imgs:
        im = pt.load_jpeg(path)
        status['test_images_read'] += 1
        arr, r = predict_image_rows(proj, model, cfg, device, image_id, im, rule)
        maps[image_id] = arr
        rows.extend(r)
    log('test inference: %d images, %d rows in %.0f s' % (len(maps), len(rows), time.time() - t0))
    index = save_maps_npz(out.root / 'test_maps', maps, 'qmai.solar-track2-r3.test-maps/v1',
                          {'weights_sha256': [weights_sha], 'rule': dict(rule, limb_applied_to_maps=rule['limb']),
                           'quantisation': 'rint(255*p), uint8, 2048x2048', 'tta': 'flip4' if cfg['tta'] else 'none'})

    def rederive():
        back = []
        for image_id in maps:
            raw = (out.root / 'test_maps' / index['images'][image_id]['file']).read_bytes()
            require(sha256(raw) == index['images'][image_id]['sha256'], 'TEST_MAP_SHA')
            back.extend(pt.rows_from_uint8(image_id, np.load(io.BytesIO(raw))['p'], rule3))
        return back

    if not cfg['dry']:
        def validate(path):
            return vs.validate_file(path, test_dir, cfg['split_path'], expect_images=cfg['test_images'])
    else:
        forbidden = pt.load_forbidden(cfg['split_path'], cfg['pins']['split'])

        def validate(path):
            import csv as _csv
            with open(path, newline='') as f:
                back_rows = list(_csv.DictReader(f))
            return vs.validate_rows(back_rows, official_ids=set(maps), forbidden=forbidden)
    final, summary = finalize_csv(pt, out.root, rows, 'submission.csv', validate, rederive)
    out.json('SUBMISSION_VALIDATION.json', {'file': final.name, 'sha256': sha256_file(final), 'validator': summary,
                                            'rule': rule, 'weights_sha256': weights_sha, 'submitted': False,
                                            'test_maps_index_sha256': sha256_file(out.root / 'test_maps/INDEX.json'),
                                            'order': 'written as UNVALIDATED_DO_NOT_SUBMIT.tmp, LF-checked, validated, '
                                                     're-derived from test_maps, then renamed'})
    log('submission written: %s (%d rows) sha256 %s; validator %s' % (final.name, len(rows), sha256_file(final),
                                                                     json.dumps(summary, sort_keys=True)))


def smoke_stage(cfg, out, proj, model, device, XV, val_ids, val_disks, val_row_by_id, allowed, ctx, best, evals,
                status, pre, train_info, T0, epochs_done, stopped_early):
    """SMOKE: after the real preload, preflight, a few dozen real GPU steps and one real validation: checkpoint
    reload, a rule table on a subset with TTA, the stop-rule logic and the test-path self-check. No test image, no CSV."""
    import torch
    log = out.log
    ckdir = out.root / 'checkpoints'
    # checkpoint save / reload: a fresh model from best.pt gives the same maps
    fresh = build_darkness_model() if cfg['model'] == 'darkness' else build_model()
    fresh.load_state_dict(torch.load(str(ckdir / 'best.pt'), map_location='cpu', weights_only=True))
    fresh = fresh.to(device)
    a = dict(infer_maps(model, XV[:2], cfg, device, False, val_ids[:2]))
    b = dict(infer_maps(fresh, XV[:2], cfg, device, False, val_ids[:2]))
    reload_ok = all(np.array_equal(a[i], b[i]) for i in a)
    require(reload_ok, 'CHECKPOINT_RELOAD_MISMATCH')
    log('checkpoint reload OK (maps identical on 2 validation images)')
    del fresh
    # rule table on a subset of components, with TTA, through the real scoring path
    sub = list(range(min(cfg['smoke_subset_components'], len(ctx['comps']))))
    sub_ids = [i for ci in sub for i in ctx['image_ids'][ci]]
    pos = {i: k for k, i in enumerate(val_ids)}
    sub_maps = dict(infer_maps(model, XV[[pos[i] for i in sub_ids]], cfg, device, cfg['tta'], sub_ids))
    settings = cfg.get('full_settings') or full_settings()
    table, complete, st = score_settings(proj, cfg, ctx, sub_maps, val_disks, settings, cfg['final_eval_seconds_max'],
                                         log, components=sub)
    rows = rule_table_rows(proj.pp, table, settings, st['refused'])
    bi = best_setting(proj.pp, table, settings, np.arange(table.shape[1]))
    require(bi is not None, 'NO_SCORED_SETTING')
    chosen = settings[bi]
    d1 = rows[bi]['pq']
    logic = {str(v): verdict_of(v) for v in (0.0, 0.2699, 0.27, 0.2999, 0.30, 0.5)}
    require(logic == {'0.0': 'STOP', '0.2699': 'STOP', '0.27': 'MARGINAL_0.27_TO_0.30', '0.2999': 'MARGINAL_0.27_TO_0.30',
                      '0.3': 'CONTINUE', '0.5': 'CONTINUE'}, 'STOP_RULE_LOGIC')
    out.json('SMOKE_RULE_TABLE.json', {'components': sub, 'images': len(sub_ids), 'rows': rows, 'complete': complete,
                                       'scoring': st, 'note': 'SMOKE subset only; not a result'})
    frozen_path = out.json('SMOKE_FROZEN_RULE.json', {'rule': {'T_low': chosen[0], 'T_high': chosen[1], 'A': chosen[2],
                                                               'limb': chosen[3]}, 'weights_sha256': best['sha256'],
                                                      'verdict': 'SMOKE', 'note': 'smoke only; never used for a test image'})
    check_ids = pick_check_ids(sub_maps, chosen, cfg['self_check_images'])
    sc = self_check_test_path(proj, cfg, model, device, frozen_path, chosen, sub_maps, val_disks, val_row_by_id,
                              allowed, check_ids, log, lambda i: XV[pos[i]], strict_batch=False)
    batch = {'max_abs_diff': max(o['scored_batch4']['max_abs_diff'] for o in sc),
             'pixels_differing': sum(o['scored_batch4']['pixels_differing'] for o in sc),
             'rows_equal_all': all(o['scored_batch4']['rows_equal'] for o in sc),
             'identical': all(o['scored_batch4']['max_abs_diff'] == 0 for o in sc)}
    log('SMOKE SELF-CHECK batch-1 vs batch-4: max abs diff %d, %d pixels differ, rows equal %s%s' % (
        batch['max_abs_diff'], batch['pixels_differing'], batch['rows_equal_all'],
        '' if batch['identical'] else '  <-- the full run requires identity here: report these numbers for a ruling'))
    csvs = [p.name for p in out.root.rglob('*.csv')]
    require(status['test_images_read'] == 0 and not csvs and not (out.root / 'test_maps').exists(), 'SMOKE_TOUCHED_TEST')
    res = {'status': 'SMOKE_OK', 'seconds_total': time.time() - T0, 'scoring_preflight': pre,
           'train': train_info, 'steps': evals[-1]['step'] if evals else None,
           'validation_quick_rows_all_141_or_dry': evals[-1]['quick_rows'] if evals else None,
           'validation_scoreable': evals[-1].get('scoreable') if evals else None,
           'validation_not_scoreable': evals[-1].get('not_scoreable') if evals else None,
           'validation_best_quick_pq': best['pq'], 'checkpoint_sha256': best['sha256'], 'checkpoint_reload_ok': reload_ok,
           'subset_rule_table_best': {'setting': list(chosen), 'pq': d1, 'verdict_of_subset_pq': verdict_of(d1)},
           'stop_rule_logic': logic, 'self_check': sc, 'self_check_batch1_vs_batch4': batch, 'scoring_last': st,
           'scoring_workers': ctx['pool'].start_info if ctx.get('pool') else None,
           'parent_peak_rss_gib': rss_peak_bytes() / 2 ** 30, 'gpu_status': gpu_status(),
           'test_images_read': 0, 'csv_written': False,
           'note': 'SMOKE ONLY. Numbers after a few dozen steps mean nothing about the model; the full run is a separate push.'}
    out.json('SMOKE_OK.json', res)
    status.update(status='SMOKE_OK', seconds_total=time.time() - T0)
    out.json('STATUS.json', status)
    out.banner('SMOKE OK in %.0f s: preflight, %s steps at %.1f crops/s, one real validation (best quick PQ %.4f), '
               'reload, subset rule table, stop-rule logic, test-path self-check. No test image read, no CSV.' % (
                   time.time() - T0, res['steps'], train_info['crops_per_s_last_epoch'] or -1,
                   best['pq'] if best['pq'] is not None else -1))


# ============================================================ synthetic data for the dry kernel and tests
def make_synthetic(root, n_train=6, n_val=3, n_hold=2, n_test=2):
    """Tiny 2048x2048 dataset in the official layout + split plan, census, data hashes, annotations.
    Every image: limb-darkened disk (R 900), three dark filaments in separate bands (all annotated), one dark
    UNannotated blob at r/R ~ 0.95 (a limb false positive for any dark-detector). Holdout ids exist in the plan and in
    the annotation JSON but have NO image file: opening one would fail."""
    from PIL import Image
    root = Path(root)
    ds = root / 'dataset'
    (ds / 'train/train_images').mkdir(parents=True)
    (ds / 'test/test_images').mkdir(parents=True)
    rng = np.random.default_rng(7)
    ids = {'train': [], 'validation': [], 'holdout': [], 'test': []}
    k = 0
    for part, n in (('train', n_train), ('validation', n_val), ('holdout', n_hold), ('test', n_test)):
        for _ in range(n):
            k += 1
            ids[part].append('2020%02d%02d120000%s' % (1 + k // 28, 1 + k % 28, ['Bh', 'Ch', 'Lh'][k % 3]))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - 1024, xx - 1024) / 900.0
    base = np.where(r <= 1, 180 * (1 - 0.35 * r ** 2), 5).astype(np.float32)
    census_records, ann_images, ann_list, hashes = [], [], [], []
    aid = 0
    for part in ('train', 'validation', 'holdout', 'test'):
        for image_id in ids[part]:
            boxes = []
            for band in (650, 1000, 1350):
                x0 = int(rng.integers(600, 1100))
                y0 = band + int(rng.integers(-40, 40))
                boxes.append((x0, y0, x0 + int(rng.integers(150, 300)), y0 + int(rng.integers(40, 70))))
            img = base.copy()
            for x0, y0, x1, y1 in boxes:
                img[y0:y1, x0:x1] -= 70
            img[994:1054, 1849:1909] -= 70          # unannotated dark blob, centre r/R = 0.95
            img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
            n_groups = 1 if part != 'train' else 1 + (len(census_records) % 2)
            groups = ['%06d-%s' % (30101 + j, image_id) for j in range(n_groups)]
            if part != 'holdout':
                folder = ds / ('test/test_images' if part == 'test' else 'train/train_images')
                p = folder / (image_id + '.jpeg')
                Image.fromarray(img, 'L').save(p, quality=95)
                raw = p.read_bytes()
                with Image.open(p) as ph:
                    pix = np.array(ph, np.uint8)
                rel = 'data/official/MAGFiLO_1.0_Kaggle_2026/%s/%s.jpeg' % (
                    'test/test_images' if part == 'test' else 'train/train_images', image_id)
                hashes.append({'path': rel, 'bytes': len(raw), 'sha256': sha256(raw)})
                file_sha, pix_sha, nbytes = sha256(raw), sha256(pix.tobytes()), len(raw)
            else:
                rel = 'data/official/MAGFiLO_1.0_Kaggle_2026/train/train_images/%s.jpeg' % image_id
                file_sha, pix_sha, nbytes = '0' * 64, '0' * 64, 1
            if part == 'test':
                continue
            census_records.append({'image_id': image_id, 'file_name': image_id + '.jpeg', 'annotator_image_ids': groups,
                                   'path': rel, 'file_sha256': file_sha, 'bytes': nbytes, 'pixel_sha256': pix_sha})
            for gi, g in enumerate(groups):
                ann_images.append({'id': g, 'file_name': image_id + '.jpeg', 'height': H, 'width': W})
                for bi, (x0, y0, x1, y1) in enumerate(boxes):
                    if gi == 1 and bi == 1:
                        continue      # second annotator misses one filament -> soft target 0.5 there
                    aid += 1
                    ann_list.append({'id': 'a%05d' % aid, 'image_id': g, 'iscrowd': 0,
                                     'segmentation': [[x0, y0, x1, y0, x1, y1, x0, y1]]})
    census = {'records': census_records}
    census_raw = json.dumps(census, sort_keys=True).encode()
    groups = []
    for part in ('train', 'validation', 'holdout'):
        members = ids[part]
        for c in [members[i:i + 2] for i in range(0, len(members), 2)]:
            groups.append({'group_id': 'group-%s-%d' % (part, len(groups)), 'partition': part, 'image_ids': c})
    split = {'groups': groups, 'input_provenance': {'diagnostic_image_ids': []}, 'pixel_census_sha256': sha256(census_raw)}
    split_raw = json.dumps(split, sort_keys=True).encode()
    contract = root / 'contract'
    (contract / 'evaluation-split-r1').mkdir(parents=True)
    (contract / 'evaluation-split-r1/SPLIT_PLAN.json').write_bytes(split_raw)
    (contract / 'evaluation-split-r1/PIXEL_CENSUS.json').write_bytes(census_raw)
    dh = json.dumps({'files': hashes}, sort_keys=True).encode()
    (contract / 'DATA_HASHES.json').write_bytes(dh)
    ann_raw = json.dumps({'images': ann_images, 'annotations': ann_list}).encode()
    (ds / ANNOTATION_REL).write_bytes(ann_raw)
    return {'dataset_root': ds, 'split_path': contract / 'evaluation-split-r1/SPLIT_PLAN.json',
            'census_path': contract / 'evaluation-split-r1/PIXEL_CENSUS.json', 'data_hashes_path': contract / 'DATA_HASHES.json',
            'pins': {'split': sha256(split_raw), 'census': sha256(census_raw), 'annotations': sha256(ann_raw),
                     'data_hashes': sha256(dh)},
            'counts': {'train': n_train, 'validation': n_val, 'holdout': n_hold},
            'val_components': sum(1 for g in groups if g['partition'] == 'validation'),
            'val_groups': n_val, 'test_images': n_test, 'ids': ids}


DRY_SETTINGS = [(0.5, 0.5, 512, None), (0.25, 0.5, 128, None), (0.15, 0.8, 1024, None),
                (0.5, 0.5, 512, 0.9), (0.15, 0.8, 1024, 0.9)]


def real_config(src_root, out, bundle, smoke=False):
    src = Path(src_root)
    cfg = dict(SMOKE_CFG if smoke else REAL_CFG)
    cfg.update(src_root=src, dataset_root=None, out=Path(out), split_path=src / 'evaluation-split-r1/SPLIT_PLAN.json',
               census_path=src / 'evaluation-split-r1/PIXEL_CENSUS.json', data_hashes_path=src / 'DATA_HASHES.json',
               pins=dict(PINS), counts=dict(REAL_COUNTS), val_components=REAL_VAL_COMPONENTS, val_groups=REAL_VAL_GROUPS,
               test_images=REAL_TEST_IMAGES, bundle=bundle)
    return cfg


def dry_config(src_root, work_dir, bundle, branch='stop', smoke=False, ballast=False, score_inprocess=False, syn=None,
               worker_fault=None):
    work = Path(work_dir)
    syn = syn or make_synthetic(work / 'synthetic')
    cfg = dict(DRY_CFG)
    require(branch in ('stop', 'continue'), 'DRY_BRANCH')
    cfg.update(src_root=Path(src_root), dataset_root=syn['dataset_root'], out=work / 'out', split_path=syn['split_path'],
               census_path=syn['census_path'], data_hashes_path=syn['data_hashes_path'], pins=syn['pins'],
               counts=syn['counts'], val_components=syn['val_components'], val_groups=syn['val_groups'],
               test_images=syn['test_images'], bundle=bundle, full_settings=list(DRY_SETTINGS),
               model='darkness' if branch == 'continue' else 'unet', ballast=ballast,
               score_mode='inprocess' if score_inprocess else 'subprocess', worker_fault=worker_fault)
    if smoke:
        cfg.update(mode='smoke', max_steps=2, eval_every=1)
    return cfg


def main(argv, src_root, bundle):
    import argparse
    ap = argparse.ArgumentParser(description='QMAI solar track2 r3 kernel (never submits)')
    ap.add_argument('--smoke', action='store_true', help='SMOKE mode (the mandatory first push)')
    ap.add_argument('--dry', action='store_true', help='synthetic data, CPU, every output file')
    ap.add_argument('--work', help='dry mode: empty work folder')
    ap.add_argument('--branch', default='stop', choices=('stop', 'continue'), help='dry: which verdict to reach')
    ap.add_argument('--ballast', action='store_true', help='dry: make the parent as big as the real one (~3.9 GiB)')
    ap.add_argument('--score-inprocess', action='store_true', help='dry: FORCE the faulty in-process scoring (test)')
    ap.add_argument('--worker-fault', choices=('import', 'kill', 'sleep'), help='dry: inject a scoring-child fault (test)')
    a, _unknown = ap.parse_known_args(argv)   # a notebook-style runner may append its own arguments
    if a.dry:
        require(a.work, 'DRY_NEEDS_WORK')
        cfg = dry_config(src_root, a.work, bundle, a.branch, a.smoke, a.ballast, a.score_inprocess,
                         worker_fault=a.worker_fault)
    else:
        require(not (a.ballast or a.score_inprocess or a.worker_fault), 'TEST_OPTIONS_ARE_DRY_ONLY')
        require(Path('/kaggle/working').is_dir(), 'KAGGLE_RUNTIME_ONLY (use --dry elsewhere)')
        cfg = real_config(src_root, '/kaggle/working/track2_out', bundle, smoke=a.smoke)
    st = run(cfg)
    return 0 if st['status'] in ('COMPLETE', 'COMPLETE_STOPPED', 'SMOKE_OK') else 1
