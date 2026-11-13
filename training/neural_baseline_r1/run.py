"""Single bounded candidate, no platform client, downloads, retry or submission.

Platform facts are trusted executor attestations, not independently established
by this Python process. A separate actual allocation stop/release path is required.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
import uuid
from .data import PINS, Data, digest, read, require

PACKAGE = Path(__file__).absolute().parent
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
MAX_OUTPUT = 512*1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def verify_source(expected):
    raw = read(PACKAGE/'FREEZE.json', 32768, expected)
    freeze = json.loads(raw)
    required = {'__init__.py', 'data.py', 'model.py', 'run.py', 'test_neural.py', 'PLAN.json', 'README.md'}
    require(set(freeze['files']) == required, 'SOURCE_CLOSURE')
    for name, pin in freeze['files'].items():
        read(PACKAGE/name, 32768, pin)
    plan = json.loads(read(PACKAGE/'PLAN.json', 32768, freeze['files']['PLAN.json']))
    require(plan['input_pins'] == PINS and plan['seed'] == 20261108 and
            plan['training_steps'] == 2000 and plan['training_seconds'] == 2100 and
            plan['allocation_seconds'] == 3600 and plan['output_bytes'] == MAX_OUTPUT, 'PLAN_CONTRACT')
    return plan


def qualify(value, freeze_sha, output_name, *, now=None):
    """Validate explicit executor facts; no synthetic/default qualification."""
    now = time.time() if now is None else now
    fields = {'schema', 'run_id', 'freeze_sha256', 'output_name', 'platform', 'notebook_ref',
              'private', 'internet', 'accelerator_name', 'allocated_gpus', 'free_gpu_seconds_remaining',
              'observed_at_epoch', 'allocation_started_at_epoch', 'external_stop_deadline_epoch',
              'external_stop_verified', 'external_stop_receipt_sha256', 'environment_receipt_sha256',
              'competition_attachment_verified', 'owner_authorization_sha256'}
    require(type(value) is dict and set(value) == fields, 'QUALIFICATION_FIELDS')
    require(value['schema'] == 'qmai.solar-neural-executor-qualification/v1' and
            value['platform'] == 'kaggle' and value['freeze_sha256'] == freeze_sha and
            str(uuid.UUID(value['run_id'])) == value['run_id'] and value['output_name'] == output_name and
            output_name == 'solar-unet-' + value['run_id'], 'QUALIFICATION_BINDING')
    for key in ('freeze_sha256', 'external_stop_receipt_sha256', 'environment_receipt_sha256', 'owner_authorization_sha256'):
        require(type(value[key]) is str and len(value[key]) == 64 and
                all(c in '0123456789abcdef' for c in value[key]), 'QUALIFICATION_HASH')
    for key in ('private', 'external_stop_verified', 'competition_attachment_verified'):
        require(value[key] is True, 'QUALIFICATION_REQUIRED')
    require(value['internet'] is False, 'OFFLINE_REQUIRED')
    for key in ('notebook_ref', 'accelerator_name'):
        require(type(value[key]) is str and 0 < len(value[key]) <= 128 and value[key].isprintable(), 'QUALIFICATION_TEXT')
    require(type(value['allocated_gpus']) is int and 1 <= value['allocated_gpus'] <= 2, 'ALLOCATION_GPU_COUNT')
    for key in ('free_gpu_seconds_remaining', 'observed_at_epoch', 'allocation_started_at_epoch', 'external_stop_deadline_epoch'):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0, 'QUALIFICATION_TIME')
    require(0 <= now-value['observed_at_epoch'] <= 120 and
            value['allocation_started_at_epoch'] <= value['observed_at_epoch'] and
            now < value['external_stop_deadline_epoch'] <= value['allocation_started_at_epoch']+3600,
            'QUALIFICATION_EXPIRED_OR_EXTENDED')
    require(value['free_gpu_seconds_remaining'] >= 3600*value['allocated_gpus'], 'INSUFFICIENT_FREE_QUOTA')
    return value['external_stop_deadline_epoch']


@contextmanager
def alarm_blocked():
    """Keep publication/accounting indivisible with respect to our timer."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class Artifacts:
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(self.path.resolve() == self.path and not self.path.exists(), 'OUTPUT_EXISTS_OR_SYMLINK')
        self.path.mkdir(mode=0o700)
        self.bytes = 0; self.files = []; self.attempts = []; self.pending = []
        self.terminal = None

    def write(self, name, value, *, before_publish=lambda: None):
        require(self.terminal is None, 'TERMINAL_ALREADY_PUBLISHED')
        require(name in {'QUALIFICATION.json', 'RUN.json', 'FAILURE.json', 'SCORE.json', 'weights.pt'} or
                name.startswith('component-') and name.endswith('.csv') and name[10:-4].isdigit(), 'OUTPUT_NAME')
        raw = value if type(value) is bytes else canonical(value)+b'\n'
        reserve = 0 if name == 'FAILURE.json' else 1024**2
        require(len(raw) <= 32*1024**2 and self.bytes+len(raw) <= MAX_OUTPUT-reserve and
                len(self.attempts) < (45 if name == 'FAILURE.json' else 44), 'ARTIFACT_LIMIT')
        result = {'name': name, 'bytes': len(raw), 'sha256': digest(raw)}
        # Charge the maximum bytes before touching disk. Failed attempts retain
        # their reservation even when cleanup succeeds; no repeated-write escape.
        temporary = self.path/('.pending-'+uuid.uuid4().hex)
        attempt = {**result, 'temporary': temporary.name, 'state': 'RESERVED'}
        with alarm_blocked():
            self.bytes += len(raw); self.attempts.append(attempt)
        fd = None
        try:
            with alarm_blocked():
                fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
                self.pending.append(temporary.name)
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                require(count > 0, 'SHORT_ARTIFACT_WRITE')
                view = view[count:]
            os.fsync(fd)
            with alarm_blocked():
                os.close(fd); fd = None
                before_publish()
                # Same-directory hard link is atomic and refuses an existing
                # destination. A final name can never expose a partial write.
                os.link(temporary, self.path/name, follow_symlinks=False)
                self.files.append(result); attempt['state'] = 'PUBLISHED'
                if name in {'RUN.json', 'FAILURE.json'}:
                    self.terminal = name
                temporary.unlink(); self.pending.remove(temporary.name)
                directory = os.open(self.path, os.O_RDONLY|os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return result
        except BaseException:
            with alarm_blocked():
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
                    pass  # Retained reservation accounts for any residual temp.
            raise

    def failure(self, value):
        """Never contradict an already atomically published terminal record."""
        if self.terminal is not None:
            return False
        self.write('FAILURE.json', {**value, 'charged_bytes_before_failure': self.bytes,
            'attempts': list(self.attempts), 'pending': list(self.pending)})
        return True


class Budget:
    def __init__(self, wall_deadline):
        self.created_wall = time.time(); self.created_mono = time.monotonic()
        self.wall_end = wall_deadline
        self.end = self.created_mono + max(0, wall_deadline-self.created_wall)
        self.phase_end = self.end; self.cuda = None

    def check(self, *_):
        require(self.created_wall <= time.time() < self.wall_end and
                time.monotonic() < min(self.end, self.phase_end), 'TIME_BUDGET')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss*1024
        require(rss <= 2*1024**3, 'HOST_RSS_BUDGET')
        if self.cuda is not None:
            require(self.cuda.max_memory_allocated(0) <= 12*1024**3, 'GPU_ALLOCATION_BUDGET')

    def __enter__(self):
        require(signal.getitimer(signal.ITIMER_REAL) == (0, 0), 'EXISTING_TIMER')
        self.old = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.check); signal.setitimer(signal.ITIMER_REAL, .5, .5)
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, self.old)


def pool(metrics):
    require(len(metrics) > 0, 'EMPTY_VALIDATION')
    result = {key: sum(m[key] for m in metrics) for key in ('tp', 'fp', 'fn', 'matched_iou_sum', 'pair_count')}
    result['denominator'] = result['tp']+.5*result['fp']+.5*result['fn']
    result['pq'] = result['matched_iou_sum']/result['denominator'] if result['denominator'] else 0
    return result


def validate(model, data, artifacts, budget):
    from .model import tiled_probability, instances
    metrics = []; completed = []; pairs = 0; scorer_seconds = 0.0
    for index, component in enumerate(data.components):
        rows = []
        for item in component['images']:
            budget.check()
            probability = tiled_probability(model, data.image(item), device='cuda:0', check=budget.check)
            labels, number = instances(probability)
            rows.extend(data.prediction_rows(item['image_id'], labels, number))
            completed.append(item['image_id'])
            del probability, labels
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'], lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        raw = stream.getvalue().encode()
        frame, summary = data.helper.prediction_frame(raw, component['images'], check=budget.check)
        truth, selected = data.truth(component['images'])
        count = sum(n*summary[row['image_id']]['instance_count'] for row in selected for n in row['annotation_counts'].values())
        require(count <= 100000 and pairs+count <= 1000000 and scorer_seconds < 600, 'SCORER_BUDGET')
        started = time.monotonic()
        metric = data.scorer.score_rle_frames(truth, frame, height=2048, width=2048, max_groups=64,
            max_masks_per_side=512, max_pairs=100000, max_seconds=min(120, 600-scorer_seconds), max_rle_bytes=64*1024**2)
        scorer_seconds += time.monotonic()-started; pairs += count
        expected = {g for row in selected for g in row['annotator_image_ids']}
        require(metric['pair_count'] == count and metric['prediction_only_rows_ignored'] == 0 and
                len(metric['groups']) == len(expected) and {g['annotator_image'] for g in metric['groups']} == expected,
                'SCORER_GROUP_JOIN')
        require(scorer_seconds <= 600, 'SCORER_TIME')
        artifacts.write('component-'+str(index).zfill(3)+'.csv', raw)
        metrics.append(metric); budget.check()
    require(len(metrics) == 34 and len(completed) == len(set(completed)) == 141 and
            sum(len(m['groups']) for m in metrics) == 234, 'INCOMPLETE_VALIDATION')
    return {'pooled': pool(metrics), 'components': metrics, 'images': completed,
            'scorer_seconds': scorer_seconds, 'holdout_scored': False, 'test_inference': False}


def run(args):
    require(sys.dont_write_bytecode, 'PYTHON_B_REQUIRED')
    plan = verify_source(args.freeze_sha256)
    qualification = json.loads(read(args.qualification, 16384))
    deadline = qualify(qualification, args.freeze_sha256, Path(args.output).name)
    for name in THREADS:
        os.environ[name] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    artifacts = Artifacts(args.output)
    artifacts.write('QUALIFICATION.json', qualification)
    steps = 0; trained = False; outcome = None
    try:
        with Budget(deadline) as budget:
            import numpy as np
            import torch
            from .model import UNet, loss, training_crop, SEED
            require(torch.cuda.is_available() and torch.cuda.device_count() == qualification['allocated_gpus'], 'CUDA_NOT_QUALIFIED')
            require(all(torch.cuda.get_device_name(i) == qualification['accelerator_name']
                        for i in range(torch.cuda.device_count())), 'ACCELERATOR_IDENTITY')
            require(torch.cuda.mem_get_info(0)[0] >= 12*1024**3, 'GPU_FREE_MEMORY')
            data = Data(args.contract_root, args.dataset_root, check=budget.check)
            rng = random.Random(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
            model = UNet().to('cuda:0')
            parameters = sum(value.numel() for value in model.parameters())
            require(parameters < 3000000, 'PARAMETER_BOUND')
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            torch.cuda.reset_peak_memory_stats(0); budget.cuda = torch.cuda
            started = time.monotonic()
            budget.phase_end = min(budget.end-1200, started+2100)
            require(budget.phase_end-started >= 60, 'NO_TRAINING_AND_VALIDATION_BUDGET')
            history = []
            model.train()
            while steps < 2000 and budget.phase_end-time.monotonic() > 30:
                budget.check(); images = []; targets = []
                for _ in range(4):
                    image, target, _, _ = data.training_pair(rng)
                    image, target = training_crop(image, target, rng)
                    images.append(image.astype(np.float32)/255.0); targets.append(target.astype(np.float32))
                x = torch.from_numpy(np.stack(images)[:, None]).to('cuda:0')
                y = torch.from_numpy(np.stack(targets)[:, None]).to('cuda:0')
                optimizer.zero_grad(set_to_none=True)
                objective = loss(model(x), y)
                require(torch.isfinite(objective).item(), 'NONFINITE_LOSS')
                objective.backward(); optimizer.step(); torch.cuda.synchronize()
                budget.check(); steps += 1
                if steps == 1 or steps % 100 == 0:
                    history.append({'step': steps, 'loss': float(objective.detach().cpu())})
                del x, y, objective, images, targets, image, target
            require(steps > 0, 'NO_TRAINING_STEPS')
            training_seconds = time.monotonic()-started; trained = True
            budget.phase_end = min(budget.end-15, time.monotonic()+1200)
            del optimizer
            checkpoint = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            buffer = io.BytesIO(); torch.save(checkpoint, buffer)
            weights = artifacts.write('weights.pt', buffer.getvalue()); del checkpoint, buffer
            outcome = validate(model, data, artifacts, budget)
            budget.check(); verify_source(args.freeze_sha256)
            score = artifacts.write('SCORE.json', {'status': 'COMPLETE_VALIDATION_UNACCEPTED', **outcome,
                'freeze_sha256': args.freeze_sha256, 'input_pins': PINS, 'weights': weights,
                'training_steps': steps, 'training_seconds': training_seconds, 'parameters': parameters,
                'loss_samples': history, 'seed': SEED, 'model_selection': 'final checkpoint only',
                'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
                    'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)},
                'peak_device_allocated_bytes': torch.cuda.max_memory_allocated(0),
                'task_accepted': False, 'generalization_proven': False})
            budget.phase_end = budget.end
            del model, data
            torch.cuda.empty_cache(); budget.check()
            terminal = {'status': 'COMPLETED_PENDING_EXTERNAL_RELEASE', 'score': score,
                'files': list(artifacts.files), 'platform_release_verified': False, 'task_accepted': False,
                'allocation_started_at_epoch': qualification['allocation_started_at_epoch'],
                'ended_at_epoch': time.time(), 'no_retry': True}
            budget.check()
        # Timer teardown and the last ordinary check precede terminal commit.
        # Recheck after the temp write/fsync, immediately before publication.
        artifacts.write('RUN.json', terminal, before_publish=budget.check)
        return 0
    except BaseException as error:
        artifacts.failure({'status': 'FAILED_NO_RETRY', 'error_type': type(error).__name__,
            'code': str(error)[:160], 'training_steps': steps, 'training_loop_completed': trained,
            'validation_complete': outcome is not None, 'files': list(artifacts.files),
            'platform_release_verified': False, 'task_accepted': False})
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze-sha256', 'qualification', 'contract-root', 'dataset-root', 'output'):
        parser.add_argument('--'+name, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
