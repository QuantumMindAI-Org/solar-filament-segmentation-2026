"""SOLAR-TRAIN-R3 training source: run 2's package with a longer schedule and an optional clDice term.

Imported unmodified from run 2 (neural_baseline_r1): model (UNet, tiled_probability, instances, training_crop,
loss), data (Data and its holdout guard), run (Budget, Artifacts, pool, verify_source, canonical, THREADS).
Marked COPY blocks below are the only places where run 2 code could not take a parameter.
Differences from run 2 are listed in REVIEW_NOTES.md. Run only through the reviewed Kaggle payload.
"""
import argparse
import calendar
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np

from neural_baseline_r1 import run as r2
from neural_baseline_r1.data import PINS, Data, digest, read, require
from . import cldice

BASELINE_FREEZE = '3539b77590fff270c985bcc479cb8c008b27b4a94defa773c8be05ff95be37d5'
ARMS = {'A': {'dice_weight': 1.0, 'cl_weight': 0.0}, 'B': {'dice_weight': 0.5, 'cl_weight': 0.5},
        'R': {'dice_weight': 1.0, 'cl_weight': 0.0}}   # r2.5 arm R = run 2's recipe (addendum R); its loss is run 2's own `model.loss`, see train_loop_r
SEEDS = (20261111, 20261112, 20261113)
PEAK_LR, MIN_LR, WARMUP = 1e-3, 1e-5, 200
DEFAULT_STEPS = 6000
STEP_QUANTUM = 500
TRAIN_BUDGET_SECONDS = 2.75 * 3600
ALLOCATION_SECONDS = 3 * 3600
# r2.5 (addendum R): arm R is run 2's recipe: 2000 steps, AdamW at a CONSTANT lr 1e-3, weight decay 1e-4, run 2's loss (BCE + soft Dice),
# at most 1.5 h per run. No warm-up, no decay, no step rule.
ARM_R_STEPS = 2000
ARM_R_LR = 0.001
ARM_R_WEIGHT_DECAY = 0.0001
ALLOCATION_R_SECONDS = 5400
# r2.4: the three-hour budget starts at KERNEL ENTRY (the queue is not billed): deadline <= entry + ALLOCATION - BOOT_MARGIN_SECONDS.
BOOT_MARGIN_SECONDS = 300
# r2.4: age of the HOST's quota/paid-mode observation AT KERNEL ENTRY (4 h of queue); checked once at entry and carried forward in the qualification.
QUEUE_MAX_SECONDS = 14400
# r2.4: free GPU hours expire 2026-11-14T00:00:00Z; same number as bootstrap_r3.WEEK_END_EPOCH and controller_r3.WEEK_END_EPOCH.
WEEK_END_EPOCH = calendar.timegm((2026, 11, 14, 0, 0, 0))
VALIDATION_RESERVE_SECONDS = 600
AREA_FILTER = 512
MAX_OUTPUT = r2.MAX_OUTPUT


class NonFiniteLoss(ValueError):
    def __init__(self, step):
        super().__init__('NONFINITE_LOSS')
        self.step = step


# ------------------------------------------------------------------ schedule and step rule
def lr_at(step, total):
    """0-based step. Linear warm-up to PEAK_LR over WARMUP steps, then cosine to MIN_LR at the last step."""
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    span = max(1, total - 1 - WARMUP)
    progress = min(1.0, (step - WARMUP) / span)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def choose_steps(seconds_for_200, pinned=None):
    """Pre-registered rule. seconds_for_200: wall seconds of steps 1..200. Returns the total step count."""
    require(seconds_for_200 > 0 and math.isfinite(seconds_for_200), 'STEP_TIMING')
    if pinned is not None:
        require(type(pinned) is int and pinned % STEP_QUANTUM == 0 and STEP_QUANTUM <= pinned <= DEFAULT_STEPS, 'PINNED_STEPS')
        return pinned
    rate = seconds_for_200 / WARMUP
    if rate * DEFAULT_STEPS <= TRAIN_BUDGET_SECONDS:
        return DEFAULT_STEPS
    fit = int(TRAIN_BUDGET_SECONDS / rate // STEP_QUANTUM) * STEP_QUANTUM
    require(fit >= STEP_QUANTUM, 'STEP_RULE_BELOW_ONE_QUANTUM')
    return fit


# ------------------------------------------------------------------ loss
def combined_loss(logits, target, dice_weight=1.0, cl_weight=0.0):
    """BCE + dice_weight * soft Dice loss + cl_weight * soft clDice loss.

    COPY of neural_baseline_r1.model.loss's expressions (same ops, same order) so that
    dice_weight=1, cl_weight=0 is bit-identical to run 2's loss; verified by test.
    """
    import torch
    from torch.nn import functional as F
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (probability * target).sum(axes) + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    term = (1 - dice).mean()
    total = F.binary_cross_entropy_with_logits(logits, target) + (term if dice_weight == 1.0 else dice_weight * term)
    if cl_weight:
        total = total + cl_weight * cldice.soft_cldice(probability, target).mean()
    return total


def arm_loss(arm):
    cfg = ARMS[arm]
    return lambda logits, target: combined_loss(logits, target, cfg['dice_weight'], cfg['cl_weight'])


# ------------------------------------------------------------------ training loop
def train_loop(model, sample, arm, seed, device, check=lambda: None, pinned_steps=None, clock=time.monotonic,
               optimizer_factory=None, log=lambda record: None, total_cap=DEFAULT_STEPS):
    """sample(rng) -> (image uint8 HxW, target HxW) already cropped/augmented. Returns a summary dict."""
    import torch
    rng = random.Random(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)
    objective_fn = arm_loss(arm)
    optimizer = (optimizer_factory or (lambda p: torch.optim.AdamW(p, lr=PEAK_LR, weight_decay=0.0001)))(model.parameters())
    total = pinned_steps if pinned_steps is not None else None
    if pinned_steps is not None:
        choose_steps(1.0, pinned_steps)   # validates
    history = []; step = 0; started = clock(); seconds_200 = None
    model.train()
    while total is None or step < total:
        check()
        images = []; targets = []
        for _ in range(4):
            image, target = sample(rng)
            images.append(image.astype(np.float32) / 255.0); targets.append(target.astype(np.float32))
        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(targets)[:, None]).to(device)
        lr = lr_at(step, total if total is not None else DEFAULT_STEPS)   # step < WARMUP when total is None
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        objective = objective_fn(model(x), y)
        if not torch.isfinite(objective).item():
            raise NonFiniteLoss(step + 1)
        objective.backward(); optimizer.step()
        if device != 'cpu':
            torch.cuda.synchronize()
        check(); step += 1
        if step == 1 or step % 100 == 0:
            record = {'step': step, 'loss': float(objective.detach().cpu()), 'lr': lr}
            history.append(record); log(record)
        if step == WARMUP:
            seconds_200 = clock() - started
            chosen = choose_steps(seconds_200, pinned_steps)
            chosen = min(chosen, total_cap)
            total = chosen
            log({'event': 'STEP_RULE', 'seconds_first_200_steps': seconds_200, 'seconds_per_step': seconds_200 / WARMUP,
                 'chosen_steps': chosen, 'pinned': pinned_steps is not None})
        del x, y, objective, images, targets
    return {'steps': step, 'chosen_steps': total, 'seconds_first_200_steps': seconds_200,
            'seconds_per_step_first_200': (seconds_200 / WARMUP) if seconds_200 else None,
            'training_seconds': clock() - started, 'loss_samples': history, 'arm': arm, 'seed': seed,
            'pinned_steps': pinned_steps}


def seed_run2(seed):
    """Run 2's seeding, statement for statement and in its order (neural_baseline_r1/run.py: rng, np, torch, cuda), done once after the data
    object exists and before the model is built. Returns the sampling rng. Arm R does not reseed afterwards (arms A/B reseed inside train_loop)."""
    import torch
    rng = random.Random(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return rng


def train_loop_r(model, sample, rng, device, check=lambda: None, clock=time.monotonic, log=lambda record: None, steps=ARM_R_STEPS):
    """Arm R: run 2's loop (neural_baseline_r1/run.py, `while steps < 2000`): 4 samples per step from `sample(rng)`, AdamW(lr=0.001,
    weight_decay=0.0001) with NO per-step lr change, `optimizer.zero_grad(set_to_none=True)`, run 2's own `loss` imported from
    neural_baseline_r1.model (not copied), backward, step. Differences from run 2, all outside the arithmetic: the loop is bounded by the step count only
    (run 2 also stopped on a wall-clock phase end, here `check()` raises TIME_BUDGET instead), a non-finite loss raises NonFiniteLoss (the harness status),
    and the first-200-step time is measured for the saved STEP_RULE.json (it is not used)."""
    import torch
    from neural_baseline_r1.model import loss as run2_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=ARM_R_LR, weight_decay=ARM_R_WEIGHT_DECAY)
    history = []; step = 0; started = clock(); seconds_200 = None
    model.train()
    while step < steps:
        check()
        images = []; targets = []
        for _ in range(4):
            image, target = sample(rng)
            images.append(image.astype(np.float32) / 255.0); targets.append(target.astype(np.float32))
        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(targets)[:, None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        objective = run2_loss(model(x), y)
        if not torch.isfinite(objective).item():
            raise NonFiniteLoss(step + 1)
        objective.backward(); optimizer.step()
        if device != 'cpu':
            torch.cuda.synchronize()
        check(); step += 1
        if step == 1 or step % 100 == 0:
            record = {'step': step, 'loss': float(objective.detach().cpu()), 'lr': ARM_R_LR}
            history.append(record); log(record)
        if step == WARMUP:
            seconds_200 = clock() - started
        del x, y, objective, images, targets
    return {'steps': step, 'chosen_steps': steps, 'seconds_first_200_steps': seconds_200,
            'seconds_per_step_first_200': (seconds_200 / WARMUP) if seconds_200 else None,
            'training_seconds': clock() - started, 'loss_samples': history, 'arm': 'R', 'seed': None,
            'pinned_steps': steps}


# ------------------------------------------------------------------ probability maps
def to_uint8(p):
    return np.clip(np.rint(255.0 * np.asarray(p, np.float64)), 0, 255).astype(np.uint8)


def npz_bytes(arr):
    buffer = io.BytesIO(); np.savez_compressed(buffer, p=arr); return buffer.getvalue()


def read_map(raw):
    return np.load(io.BytesIO(raw))['p']


def area_filter_labels(labels, number, minimum=AREA_FILTER):
    """Drop instances below `minimum` pixels and renumber 1..k (same order). Returns (labels, k)."""
    counts = np.bincount(labels.ravel(), minlength=number + 1)
    keep = np.flatnonzero(counts[1:] >= minimum) + 1
    mapping = np.zeros(number + 1, labels.dtype)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=labels.dtype)
    return mapping[labels], int(len(keep))


# ------------------------------------------------------------------ artifacts (COPY with wider name list)
class ArtifactsR3(r2.Artifacts):
    """COPY of neural_baseline_r1.run.Artifacts.write: only the allowed-name test and the attempt cap differ
    (probability maps add 141 files + INDEX.json). All atomic-publication logic is unchanged."""
    NAMES = {'QUALIFICATION.json', 'RUN.json', 'FAILURE.json', 'SCORE.json', 'weights.pt', 'INDEX.json', 'STEP_RULE.json'}
    MAX_ATTEMPTS = 260

    @staticmethod
    def allowed(name):
        import re
        return (name in ArtifactsR3.NAMES or (name.startswith('component-') and name.endswith('.csv') and name[10:-4].isdigit())
                or re.fullmatch(r'[0-9]{14}[A-Za-z]{2}\.npz', name) is not None)

    def write(self, name, value, *, before_publish=lambda: None):
        require(self.terminal is None, 'TERMINAL_ALREADY_PUBLISHED')
        require(self.allowed(name), 'OUTPUT_NAME')
        raw = value if type(value) is bytes else r2.canonical(value) + b'\n'
        reserve = 0 if name == 'FAILURE.json' else 1024**2
        require(len(raw) <= 32 * 1024**2 and self.bytes + len(raw) <= MAX_OUTPUT - reserve and
                len(self.attempts) < (self.MAX_ATTEMPTS + 1 if name == 'FAILURE.json' else self.MAX_ATTEMPTS), 'ARTIFACT_LIMIT')
        result = {'name': name, 'bytes': len(raw), 'sha256': digest(raw)}
        temporary = self.path / ('.pending-' + uuid.uuid4().hex)
        attempt = {**result, 'temporary': temporary.name, 'state': 'RESERVED'}
        with r2.alarm_blocked():
            self.bytes += len(raw); self.attempts.append(attempt)
        fd = None
        try:
            with r2.alarm_blocked():
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                self.pending.append(temporary.name)
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                require(count > 0, 'SHORT_ARTIFACT_WRITE')
                view = view[count:]
            os.fsync(fd)
            with r2.alarm_blocked():
                os.close(fd); fd = None
                before_publish()
                os.link(temporary, self.path / name, follow_symlinks=False)
                self.files.append(result); attempt['state'] = 'PUBLISHED'
                if name in {'RUN.json', 'FAILURE.json'}:
                    self.terminal = name
                temporary.unlink(); self.pending.remove(temporary.name)
                directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return result
        except BaseException:
            with r2.alarm_blocked():
                if attempt['state'] != 'PUBLISHED':
                    attempt['state'] = 'FAILED_RESERVED'
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    temporary.unlink(missing_ok=True)
                    if temporary.name in self.pending:
                        self.pending.remove(temporary.name)
                except OSError:
                    pass
            raise


# ------------------------------------------------------------------ qualification and source (COPY with 3 h bounds)
def qualify_r3(value, freeze_sha, output_name, *, now=None, budget_seconds=ALLOCATION_SECONDS):
    """COPY of run.qualify: allocation cap 3 h (not 1 h) and free quota >= 10800 s (observed ~1 quota-second per
    wall-second in run 2). Same closed field set, same binding checks.
    r2: the stop is the in-kernel deadline (bootstrap_r3.child supervisor). The 120 s server-stop receipt is carried
    only as a reference and may not be claimed as a 10800 s stop (QUALIFICATION_OVERCLAIM).
    r2.4: the budget is anchored at kernel entry: in_kernel_deadline_epoch <= kernel_entry_epoch + 10800 - 300 and now < deadline;
    the host-evidence age is the age AT ENTRY (carried in the qualification, not re-measured now); the run must end before the week end.
    r2.5: budget_seconds is 10800 for arms A/B and 5400 for arm R (the caller passes it); everything that was ALLOCATION_SECONDS is that budget."""
    now = time.time() if now is None else now
    fields = {'schema', 'run_id', 'freeze_sha256', 'output_name', 'platform', 'notebook_ref',
              'private', 'internet', 'accelerator_name', 'allocated_gpus', 'free_gpu_seconds_remaining',
              'observed_at_epoch', 'kernel_entry_epoch', 'host_evidence_age_at_entry_seconds',
              'budget_is_anchored_at_kernel_entry', 'in_kernel_deadline_epoch',
              'in_kernel_deadline_enforced', 'server_timeout_requested_seconds', 'server_timeout_verified_seconds',
              'server_stop_verified_at_requested_length', 'server_stop_receipt_sha256', 'environment_receipt_sha256',
              'competition_attachment_verified', 'owner_authorization_sha256'}
    require(type(value) is dict and set(value) == fields, 'QUALIFICATION_FIELDS')
    require(value['schema'] == 'qmai.solar-neural-executor-qualification/v1' and
            value['platform'] == 'kaggle' and value['freeze_sha256'] == freeze_sha and
            str(uuid.UUID(value['run_id'])) == value['run_id'] and value['output_name'] == output_name and
            output_name == 'solar-unet-' + value['run_id'], 'QUALIFICATION_BINDING')
    for key in ('freeze_sha256', 'server_stop_receipt_sha256', 'environment_receipt_sha256', 'owner_authorization_sha256'):
        require(type(value[key]) is str and len(value[key]) == 64 and all(c in '0123456789abcdef' for c in value[key]), 'QUALIFICATION_HASH')
    for key in ('private', 'in_kernel_deadline_enforced', 'competition_attachment_verified', 'budget_is_anchored_at_kernel_entry'):
        require(value[key] is True, 'QUALIFICATION_REQUIRED')
    require(budget_seconds in (ALLOCATION_SECONDS, ALLOCATION_R_SECONDS) and value['server_timeout_requested_seconds'] == budget_seconds and
            value['server_timeout_verified_seconds'] == 120 and
            value['server_stop_verified_at_requested_length'] is False, 'QUALIFICATION_OVERCLAIM')
    require(value['internet'] is False, 'OFFLINE_REQUIRED')
    for key in ('notebook_ref', 'accelerator_name'):
        require(type(value[key]) is str and 0 < len(value[key]) <= 128 and value[key].isprintable(), 'QUALIFICATION_TEXT')
    require(type(value['allocated_gpus']) is int and 1 <= value['allocated_gpus'] <= 2, 'ALLOCATION_GPU_COUNT')
    for key in ('free_gpu_seconds_remaining', 'observed_at_epoch', 'kernel_entry_epoch', 'host_evidence_age_at_entry_seconds',
                'in_kernel_deadline_epoch'):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0, 'QUALIFICATION_TIME')
    entry = value['kernel_entry_epoch']
    require(0 <= entry - value['observed_at_epoch'] <= QUEUE_MAX_SECONDS and  # observed_at_epoch = the host time; the age is the age AT ENTRY
            abs(value['host_evidence_age_at_entry_seconds'] - (entry - value['observed_at_epoch'])) < 1e-3, 'QUALIFICATION_QUEUE_AGE')
    require(entry + budget_seconds <= WEEK_END_EPOCH, 'KERNEL_WEEK_END')
    require(entry <= now < value['in_kernel_deadline_epoch'] <= entry + budget_seconds - BOOT_MARGIN_SECONDS,
            'QUALIFICATION_EXPIRED_OR_EXTENDED')
    require(value['free_gpu_seconds_remaining'] >= budget_seconds, 'INSUFFICIENT_FREE_QUOTA')
    return value['in_kernel_deadline_epoch']


PACKAGE = Path(__file__).absolute().parent


def verify_source_r3(expected_freeze_sha, package=PACKAGE):
    """Frozen run 2 closure via run.verify_source, then this package's own pinned file list."""
    r2.verify_source(BASELINE_FREEZE)
    freeze = json.loads(read(package / 'FREEZE_R3.json', 32768, expected_freeze_sha))
    require(set(freeze['files']) == {'__init__.py', 'cldice.py', 'train_r3.py', 'PLAN_R3.json'}, 'R3_SOURCE_CLOSURE')
    for name, pin in freeze['files'].items():
        read(package / name, 262144, pin)
    plan = json.loads(read(package / 'PLAN_R3.json', 32768, freeze['files']['PLAN_R3.json']))
    require(plan['baseline_freeze_sha256'] == BASELINE_FREEZE and plan['seeds'] == list(SEEDS)
            and plan['arms'] == ARMS and plan['allocation_seconds'] == ALLOCATION_SECONDS
            and plan['arm_r'] == {'steps': ARM_R_STEPS, 'lr': ARM_R_LR, 'weight_decay': ARM_R_WEIGHT_DECAY, 'allocation_seconds': ALLOCATION_R_SECONDS,
                                  'warmup': False, 'decay': False, 'step_rule': False, 'loss': 'neural_baseline_r1.model.loss',
                                  'order': ['R1', 'R2', 'R3']}
            and plan['addendum_r_sha256'] == 'ffda73d2f571ea80351127d15539110c161c97765875db9bb750857386dfccd2', 'R3_PLAN_CONTRACT')
    return plan


# ------------------------------------------------------------------ validation (COPY of run.validate + maps + area filter)
def validate_r3(model, data, artifacts, budget, device='cuda:0', expected_counts=(34, 141, 234)):
    """COPY of neural_baseline_r1.run.validate. Added: store uint8 probability map per image; score the raw
    (>=0.5, 8-connected) instances AND the same instances with area >= 512 removed. Raw path and CSVs unchanged."""
    from neural_baseline_r1.model import tiled_probability, instances
    raw_metrics = []; filt_metrics = []; completed = []; pairs = 0; scorer_seconds = 0.0
    index = {}
    for ci, component in enumerate(data.components):
        rows = []; rows512 = []
        for item in component['images']:
            budget.check()
            probability = tiled_probability(model, data.image(item), device=device, check=budget.check)
            amap = to_uint8(probability)
            blob = npz_bytes(amap)
            name = item['image_id'] + '.npz'
            entry = artifacts.write(name, blob)
            index[item['image_id']] = {'file': name, 'shape': list(amap.shape), 'sha256': entry['sha256'], 'bytes': entry['bytes']}
            labels, number = instances(probability)
            rows.extend(data.prediction_rows(item['image_id'], labels, number))
            labels2, number2 = area_filter_labels(labels, number)
            rows512.extend(data.prediction_rows(item['image_id'], labels2, number2))
            completed.append(item['image_id'])
            del probability, labels, labels2
        frames = {}
        for key, selected_rows in (('raw', rows), ('area512', rows512)):
            import csv
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'], lineterminator='\n')
            writer.writeheader(); writer.writerows(selected_rows)
            frames[key] = stream.getvalue().encode()
        truth, selected = data.truth(component['images'])
        for key, metrics in (('raw', raw_metrics), ('area512', filt_metrics)):
            frame, summary = data.helper.prediction_frame(frames[key], component['images'], check=budget.check)
            count = sum(n * summary[row['image_id']]['instance_count'] for row in selected for n in row['annotation_counts'].values())
            require(count <= 100000 and pairs + count <= 2000000 and scorer_seconds < 1200, 'SCORER_BUDGET')
            started = time.monotonic()
            metric = data.scorer.score_rle_frames(truth, frame, height=2048, width=2048, max_groups=64,
                max_masks_per_side=512, max_pairs=100000, max_seconds=min(120, 1200 - scorer_seconds), max_rle_bytes=64 * 1024**2)
            scorer_seconds += time.monotonic() - started; pairs += count
            expected = {g for row in selected for g in row['annotator_image_ids']}
            require(metric['pair_count'] == count and metric['prediction_only_rows_ignored'] == 0 and
                    len(metric['groups']) == len(expected) and {g['annotator_image'] for g in metric['groups']} == expected,
                    'SCORER_GROUP_JOIN')
            metrics.append(metric)
        artifacts.write('component-' + str(ci).zfill(3) + '.csv', frames['raw'])
        budget.check()
    require(len(raw_metrics) == len(filt_metrics) == expected_counts[0] and
            len(completed) == len(set(completed)) == expected_counts[1] and
            sum(len(m['groups']) for m in raw_metrics) == expected_counts[2], 'INCOMPLETE_VALIDATION')
    return {'pooled': r2.pool(raw_metrics), 'pooled_area512': r2.pool(filt_metrics), 'components': raw_metrics,
            'components_area512': filt_metrics, 'images': completed, 'scorer_seconds': scorer_seconds,
            'holdout_scored': False, 'test_inference': False, 'index': index}


# ------------------------------------------------------------------ run
def run(args):
    require(sys.dont_write_bytecode, 'PYTHON_B_REQUIRED')
    require(args.arm in ARMS and args.seed in SEEDS, 'ARM_OR_SEED')
    arm_r = args.arm == 'R'  # r2.5
    require(not arm_r or args.steps == ARM_R_STEPS, 'ARM_R_STEPS')
    plan = verify_source_r3(args.freeze_sha256)
    qualification = json.loads(read(args.qualification, 16384))
    deadline = qualify_r3(qualification, args.freeze_sha256, Path(args.output).name,
                          budget_seconds=ALLOCATION_R_SECONDS if arm_r else ALLOCATION_SECONDS)  # r2.5
    for name in r2.THREADS:
        os.environ[name] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    artifacts = ArtifactsR3(args.output)
    artifacts.write('QUALIFICATION.json', qualification)
    steps = 0; outcome = None; summary = None
    try:
        with r2.Budget(deadline) as budget:
            import torch
            from neural_baseline_r1.model import UNet, training_crop
            require(torch.cuda.is_available() and torch.cuda.device_count() == qualification['allocated_gpus'], 'CUDA_NOT_QUALIFIED')
            require(all(torch.cuda.get_device_name(i) == qualification['accelerator_name']
                        for i in range(torch.cuda.device_count())), 'ACCELERATOR_IDENTITY')
            require(torch.cuda.mem_get_info(0)[0] >= 12 * 1024**3, 'GPU_FREE_MEMORY')
            data = Data(args.contract_root, args.dataset_root, check=budget.check)
            if arm_r: rng_r = seed_run2(args.seed)  # r2.5: run 2's seeding order, once, before the model
            torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
            torch.manual_seed(args.seed)
            model = UNet().to('cuda:0')
            parameters = sum(v.numel() for v in model.parameters())
            require(parameters < 3000000, 'PARAMETER_BOUND')
            torch.cuda.reset_peak_memory_stats(0); budget.cuda = torch.cuda
            started = time.monotonic()
            budget.phase_end = min(budget.end - VALIDATION_RESERVE_SECONDS, started + TRAIN_BUDGET_SECONDS + 120)
            require(budget.phase_end - started >= 600, 'NO_TRAINING_BUDGET')

            def sample(rng):
                image, target, _, _ = data.training_pair(rng)
                return training_crop(image, target, rng)

            if arm_r:  # r2.5: run 2's recipe, no step rule, no schedule
                summary = train_loop_r(model, sample, rng_r, 'cuda:0', check=budget.check, log=lambda rec: print(json.dumps(rec), flush=True))
                summary['seed'] = args.seed
            else:
                summary = train_loop(model, sample, args.arm, args.seed, 'cuda:0', check=budget.check,
                                     pinned_steps=args.steps, log=lambda rec: print(json.dumps(rec), flush=True))
            steps = summary['steps']
            artifacts.write('STEP_RULE.json', {k: summary[k] for k in ('chosen_steps', 'seconds_first_200_steps',
                            'seconds_per_step_first_200', 'pinned_steps', 'arm', 'seed')})
            budget.phase_end = min(budget.end - 15, time.monotonic() + 1500)
            checkpoint = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            buffer = io.BytesIO(); torch.save(checkpoint, buffer)
            weights = artifacts.write('weights.pt', buffer.getvalue()); del checkpoint, buffer
            outcome = validate_r3(model, data, artifacts, budget)
            budget.check(); verify_source_r3(args.freeze_sha256)
            index = outcome.pop('index')
            artifacts.write('INDEX.json', {'schema': 'solar-pp-r3-probs/v1', 'weights_sha256': weights['sha256'],
                                           'encoding': 'uint8 rint(255*p), key p', 'images': index})
            score = artifacts.write('SCORE.json', {'status': 'COMPLETE_VALIDATION_UNACCEPTED', 'arm': args.arm, 'seed': args.seed,
                **outcome, 'freeze_sha256': args.freeze_sha256, 'input_pins': PINS, 'weights': weights,
                'training': {k: summary[k] for k in ('steps', 'chosen_steps', 'training_seconds', 'seconds_per_step_first_200',
                                                      'pinned_steps', 'loss_samples')},
                'loss_config': ARMS[args.arm], 'parameters': parameters, 'model_selection': 'final checkpoint only',
                'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
                                'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)},
                'peak_device_allocated_bytes': torch.cuda.max_memory_allocated(0),
                'task_accepted': False, 'generalization_proven': False})
            budget.phase_end = budget.end
            del model, data
            torch.cuda.empty_cache(); budget.check()
            terminal = {'status': 'COMPLETED_PENDING_EXTERNAL_RELEASE', 'score': score, 'arm': args.arm, 'seed': args.seed,
                        'chosen_steps': summary['chosen_steps'], 'files': list(artifacts.files),
                        'platform_release_verified': False, 'task_accepted': False,
                        'kernel_entry_epoch': qualification['kernel_entry_epoch'],  # r2.5: r2.4 read a field it had removed (KeyError at the very end)
                        'ended_at_epoch': time.time(), 'no_retry': True}
            budget.check()
        artifacts.write('RUN.json', terminal, before_publish=budget.check)
        return 0
    except BaseException as error:
        artifacts.failure(failure_record(error, args.arm, args.seed, steps, outcome is not None, artifacts.files))
        return 2


def failure_record(error, arm, seed, steps, validation_complete, files):
    nonfinite = isinstance(error, NonFiniteLoss)
    return {'status': 'FAILED_NONFINITE_LOSS' if nonfinite else 'FAILED_NO_RETRY',
            'step': getattr(error, 'step', None), 'arm': arm, 'seed': seed,
            'error_type': type(error).__name__, 'code': str(error)[:160], 'training_steps': steps,
            'validation_complete': validation_complete, 'files': list(files),
            'platform_release_verified': False, 'task_accepted': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze-sha256', 'qualification', 'contract-root', 'dataset-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--arm', required=True, choices=sorted(ARMS))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--steps', type=int, default=None, help='pinned step count (multiple of 500); omit for the first run')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
