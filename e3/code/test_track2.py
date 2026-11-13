"""TRACK2-R3 offline tests: Mac CPU, synthetic data, no real training, no network.

Run from R:  .venv-solar/bin/python -B -m unittest -v track2_train_r3b/test_track2.py
Real-size rehearsal (about 4 GiB parent; take seats/solar-memfix-20261111/HEAVY.lock first):
             TRACK2_REAL_SIZE=1 .venv-solar/bin/python -B -m unittest -v track2_train_r3b.test_track2.TestRealSize
The tests import the project exactly as the kernel does: the rendered kernel's bundle is extracted into a temp
folder and every module comes from there. Real files read: SPLIT_PLAN.json and PIXEL_CENSUS.json (names and hashes
only, for the real-pin guard test). No holdout or test image of the real data is opened.
"""
import csv
import faulthandler
import importlib.util
import signal
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
KERNEL = HERE / 'kernel' / 'track2_kernel.py'
KERNEL_SMOKE = HERE / 'kernel_smoke' / 'track2_kernel.py'
TMP = SRC = T2 = PROJ = SYN = None
DRY = {}
TEST_SECONDS = 300          # hard per-test limit (SIGALRM); the whole suite also has a hard cap below
SUITE_SECONDS = 1200


def launch(kernel, how, work, extra):
    """Start a dry kernel the three ways Kaggle may run a script: by path, through runpy, or exec of its text
    (a notebook cell: no __file__, ipykernel-style argv)."""
    args = ['--dry', '--work', str(work)] + extra
    if how == 'path':
        cmd = [sys.executable, '-B', str(kernel)] + args
    elif how == 'runpy':
        cmd = [sys.executable, '-B', '-c', 'import runpy, sys; sys.argv = [%r] + %r; runpy.run_path(%r, run_name="__main__")'
               % (str(kernel), args, str(kernel))]
    else:
        cmd = [sys.executable, '-B', '-c', 'import sys; sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/k.json"] + %r; '
               'g = {"__name__": "__main__"}; exec(compile(open(%r).read(), "<kaggle-cell>", "exec"), g)' % (args, str(kernel))]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(work),
                            start_new_session=True)


def setUpModule():
    global TMP, SRC, T2, PROJ, SYN
    faulthandler.dump_traceback_later(SUITE_SECONDS, exit=True)     # the suite itself cannot hang
    TMP = Path(tempfile.mkdtemp(prefix='track2r3-test-')).resolve()   # resolve: /var is a link on macOS
    # three dry kernels, each launched a different way, run in parallel while the unit tests run
    for name, kernel, how, extra in (('continue', KERNEL, 'path', ['--branch', 'continue']),
                                     ('stop', KERNEL, 'runpy', ['--branch', 'stop']),
                                     ('smoke', KERNEL_SMOKE, 'exec', [])):
        work = TMP / ('dry_' + name)
        work.mkdir()
        DRY[name] = (work, launch(kernel, how, work, extra), time.time(), how)
    spec = importlib.util.spec_from_file_location('_track2_kernel_boot', KERNEL)
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)                                    # defines extract(); main() is guarded
    SRC = TMP / 'qmai_src'
    boot.extract(SRC)
    sys.path.insert(0, str(SRC / 'track2_train_r3b'))
    import track2
    T2 = track2
    PROJ = track2.get_project(SRC)    # pinned scorer; torch thread limits set before any torch work
    SYN = track2.make_synthetic(TMP / 'syn')


def tearDownModule():
    for work, proc, _, _ in DRY.values():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    faulthandler.cancel_dump_traceback_later()
    shutil.rmtree(TMP, ignore_errors=True)


class Timed(unittest.TestCase):
    """Every test gets a hard SIGALRM limit, so no single test can hang."""

    def setUp(self):
        def boom(signum, frame):
            raise TimeoutError('TEST_HARD_TIMEOUT %d s' % TEST_SECONDS)
        signal.signal(signal.SIGALRM, boom)
        signal.alarm(getattr(self, 'LIMIT', TEST_SECONDS))

    def tearDown(self):
        signal.alarm(0)


SCORING_FORBIDDEN = (r'\bmultiprocessing\b', r'(?<![A-Za-z_])Pool\(', r'ProcessPoolExecutor', r'concurrent\.futures',
                     r'\bos\.fork\b', r'get_context\(')


def assert_no_multiprocessing_in_scoring(tc):
    """The scoring path (preflight, score_settings, ScoringPool) must start its children as fresh interpreters
    (subprocess of score_worker.py by path), never through a multiprocessing pool or a bare fork."""
    import inspect
    import re
    for fn in (T2.scoring_preflight, T2.score_settings, T2.ScoringPool):
        src = inspect.getsource(fn)
        for pat in SCORING_FORBIDDEN:
            tc.assertIsNone(re.search(pat, src), '%s in %s' % (pat, fn.__name__))


def synthetic_split():
    return Path(SYN['split_path']).read_bytes(), Path(SYN['census_path']).read_bytes()


def spawn_cfg(**kw):
    return dict({'score_mode': 'subprocess', 'score_procs': 1, 'dry': True}, **kw)


def census_rows(ids):
    census = json.loads(Path(SYN['census_path']).read_text())
    return [r for r in census['records'] if r['image_id'] in ids]


class TestPackage(Timed):
    def test_kernels_are_current_and_bundle_pins(self):
        r = subprocess.run([sys.executable, '-B', str(HERE / 'pack.py'), '--check'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for folder, kid, mode in (('kernel', 'quantummindaillc/qmai-solar-track2-r3b', 'full'),
                                  ('kernel_smoke', 'quantummindaillc/qmai-solar-track2-r3b-smoke', 'smoke')):
            meta = json.loads((HERE / folder / 'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'])
            self.assertEqual((meta['id'], meta['competition_sources']), (kid, ['filament-segmentation-2026']))
            self.assertIn("MODE = %r" % mode, (HERE / folder / 'track2_kernel.py').read_text()[:2000])
        for name, pin in (('evaluation-split-r1/SPLIT_PLAN.json', T2.PINS['split']),
                          ('evaluation-split-r1/PIXEL_CENSUS.json', T2.PINS['census']),
                          ('DATA_HASHES.json', T2.PINS['data_hashes'])):
            self.assertEqual(T2.sha256_file(SRC / name), pin)
        src = (SRC / 'track2_train_r3b/track2.py').read_text()
        for word in ('competitions submit', 'submission_submit', 'competition_submit'):
            self.assertNotIn(word, src)
        assert_no_multiprocessing_in_scoring(self)   # supervised subprocesses only (the preload fork pool never scores)
        self.assertEqual(T2.REAL_CFG['score_mode'], 'subprocess')
        self.assertEqual(T2.SMOKE_CFG['score_mode'], 'subprocess')
        self.assertTrue((SRC / 'track2_train_r3b/score_worker.py').is_file())


class TestHoldoutGuard(Timed):
    def test_guard_lists_refuses_a_leak(self):
        with self.assertRaises(T2.Refusal) as e:
            T2.guard_lists(['a', 'h1'], ['b'], ['h1', 'h2'])
        self.assertEqual(str(e.exception), 'HOLDOUT_IN_TRAIN_OR_VALIDATION_LIST:1')
        with self.assertRaises(T2.Refusal):
            T2.guard_lists(['a'], ['h2'], ['h1', 'h2'])
        self.assertTrue(T2.guard_lists(['a'], ['b'], ['h1']))

    def test_split_hash_mismatch_refused(self):
        split_raw, census_raw = synthetic_split()
        with self.assertRaises(T2.Refusal) as e:
            T2.build_lists(PROJ.nbd, split_raw + b' ', census_raw, SYN['pins']['split'], SYN['pins']['census'], SYN['counts'])
        self.assertEqual(str(e.exception), 'SPLIT_PLAN_HASH_MISMATCH')

    def test_pinned_plan_with_holdout_in_train_refused(self):
        split_raw, census_raw = synthetic_split()
        split = json.loads(split_raw)
        hold = next(g for g in split['groups'] if g['partition'] == 'holdout')['image_ids'][0]
        next(g for g in split['groups'] if g['partition'] == 'train')['image_ids'].append(hold)
        bad = json.dumps(split, sort_keys=True).encode()
        with self.assertRaises(T2.Refusal) as e:
            T2.build_lists(PROJ.nbd, bad, census_raw, T2.sha256(bad), SYN['pins']['census'], SYN['counts'])
        self.assertTrue(str(e.exception).startswith('HOLDOUT_IN_TRAIN_OR_VALIDATION_LIST'))

    def test_run_refuses_before_opening_anything(self):
        split_raw, census_raw = synthetic_split()
        split = json.loads(split_raw)
        hold = next(g for g in split['groups'] if g['partition'] == 'holdout')['image_ids'][0]
        next(g for g in split['groups'] if g['partition'] == 'validation')['image_ids'].append(hold)
        bad_path = TMP / 'leaky_SPLIT_PLAN.json'
        bad_path.write_bytes(json.dumps(split, sort_keys=True).encode())
        cfg = T2.dry_config(SRC, TMP / 'leaky_run', None, syn=SYN)
        cfg.update(split_path=bad_path, pins=dict(cfg['pins'], split=T2.sha256_file(bad_path)))
        reads = []
        orig_read, orig_img = PROJ.nbd.read, T2.read_train_image
        PROJ.nbd.read = lambda *a, **k: reads.append(a[0]) or orig_read(*a, **k)
        T2.read_train_image = lambda *a, **k: reads.append('image') or orig_img(*a, **k)
        try:
            with self.assertRaises(T2.Refusal):
                T2.run(cfg)
        finally:
            PROJ.nbd.read, T2.read_train_image = orig_read, orig_img
        self.assertEqual(reads, [], 'nothing may be opened before the guard')
        st = json.loads((TMP / 'leaky_run/out/STATUS.json').read_text())
        self.assertEqual(st['status'], 'FAILED')
        self.assertIn('HOLDOUT_IN_TRAIN_OR_VALIDATION_LIST', st['error'])

    def test_real_plan_with_real_pins(self):
        split_raw = (R / 'evaluation-split-r1/SPLIT_PLAN.json').read_bytes()
        census_raw = (R / 'evaluation-split-r1/PIXEL_CENSUS.json').read_bytes()
        train, val, comps, hold = T2.build_lists(PROJ.nbd, split_raw, census_raw, T2.PINS['split'], T2.PINS['census'],
                                                 T2.REAL_COUNTS)
        self.assertEqual((len(train), len(val), len(comps), len(hold)), (443, 141, 34, 123))
        self.assertEqual(sum(len(r['annotator_image_ids']) for r in val), 234)
        ids = {r['image_id'] for r in train} | {r['image_id'] for r in val}
        self.assertFalse(ids & hold)
        with self.assertRaises(T2.Refusal):
            T2.build_lists(PROJ.nbd, split_raw.replace(b'"validation"', b'"validatioN"', 1), census_raw,
                           T2.PINS['split'], T2.PINS['census'], T2.REAL_COUNTS)

    def test_image_reader_refuses_holdout_row(self):
        row = census_rows(set(SYN['ids']['holdout']))[0]
        with self.assertRaises(T2.Refusal) as e:
            T2.read_train_image(PROJ.nbd, PROJ.Image, SYN['dataset_root'], row, set(SYN['ids']['train']))
        self.assertEqual(str(e.exception), 'HOLDOUT_OR_FOREIGN_IMAGE')

    def test_test_folder_split_id_refused_by_name(self):          # reviewer plant P05
        ds = TMP / 'p05_dataset'
        shutil.copytree(SYN['dataset_root'], ds)
        hold = SYN['ids']['holdout'][0]
        shutil.copy(next((ds / 'test/test_images').iterdir()), ds / 'test/test_images' / (hold + '.jpeg'))
        cfg = {'split_path': SYN['split_path'], 'pins': SYN['pins'], 'data_hashes_path': SYN['data_hashes_path'],
               'dataset_root': ds, 'test_images': SYN['test_images'] + 1}
        with self.assertRaises(ValueError) as e:
            T2.list_test_inputs(PROJ, cfg)
        self.assertEqual(str(e.exception), 'SPLIT_IMAGE_REFUSED')


class Poison(list):
    """A segmentation that fails if anything touches it."""
    def __iter__(self):
        raise AssertionError('holdout polygon touched')

    def __len__(self):
        raise AssertionError('holdout polygon touched')

    def __getitem__(self, i):
        raise AssertionError('holdout polygon touched')


def annotations(poison=True):
    data = json.loads((Path(SYN['dataset_root']) / T2.ANNOTATION_REL).read_text())
    hold = set(SYN['ids']['holdout'])
    if poison:
        for a in data['annotations']:
            if a['image_id'].split('-', 1)[1] in hold:
                a['segmentation'] = Poison()
    return data


def preload_syn(part, want_target, work=256):
    allowed = set(SYN['ids']['train']) | set(SYN['ids']['validation'])
    by_group = T2.allowed_annotations(annotations(poison=False), allowed)
    cfg = {'dataset_root': SYN['dataset_root'], 'work': work, 'preload_procs': 0}
    return T2.preload(PROJ, cfg, census_rows(set(SYN['ids'][part])), by_group, allowed, want_target, lambda s: None)


class TestLabels(Timed):
    def test_holdout_polygons_never_touched(self):
        data = annotations()
        allowed = set(SYN['ids']['train']) | set(SYN['ids']['validation'])
        by_group = T2.allowed_annotations(data, allowed)
        self.assertFalse({g.split('-', 1)[1] for g in by_group} & set(SYN['ids']['holdout']))
        rows = census_rows(set(SYN['ids']['validation']))
        sel = [{**r, 'annotation_counts': {g: len(by_group[g]) for g in r['annotator_image_ids']}} for r in rows]
        gt = PROJ.sr.annotation_frame(data, sel)
        self.assertEqual(len(gt), 3 * len(rows))

    def test_soft_target_is_mean_of_annotators(self):
        data = annotations(poison=False)
        row = next(r for r in census_rows(set(SYN['ids']['train'])) if len(r['annotator_image_ids']) == 2)
        by_group = T2.allowed_annotations(data, {row['image_id']})
        t = T2.soft_target(PROJ.sr, PROJ.mask_util, row, by_group)
        vals = sorted(np.unique(t).tolist())
        print('\n[label choice] soft target values on a 2-annotator image (one annotator missed one filament): %s;'
              ' pixels at 1.0: %d, at 0.5: %d' % (vals, int((t == 1).sum()), int((t == 0.5).sum())))
        self.assertEqual(vals, [0.0, 0.5, 1.0])
        self.assertGreater(int((t == 0.5).sum()), 1000)

    def test_foreground_centred_crops_exist(self):                 # reviewer plant P07
        tr = preload_syn('train', True)
        self.assertTrue(all(len(t['fg']) > 0 for t in tr))
        X = np.stack([t['x'] for t in tr])
        Y = np.stack([t['y'] for t in tr])
        ds = T2.CropDataset(X, Y, [t['fg'] for t in tr], 64, 3, 2 / 3)
        share = np.mean([float(ds[g][1].max()) > 0.4 for g in range(60)])
        print('\n[crops] share of 64-px crops (at 256) that contain a filament: %.2f (uniform crops alone: ~0.2)' % share)
        self.assertGreater(share, 0.5)


class TestLimbAndInputs(Timed):
    def disk_image(self):
        yy, xx = np.mgrid[0:2048, 0:2048]
        r = np.hypot(yy - 1000, xx - 1050) / 880.0
        return np.where(r <= 1, 200 * (1 - 0.5 * r ** 2), 4).astype(np.uint8)

    def test_disk_geometry_and_limb_mask(self):
        d = T2.disk_geometry(self.disk_image())
        self.assertAlmostEqual(d['cy'], 1000, delta=3)
        self.assertAlmostEqual(d['cx'], 1050, delta=3)
        self.assertAlmostEqual(d['R'], 880, delta=0.02 * 880)
        arr = np.full((2048, 2048), 200, np.uint8)
        m = T2.limb_mask_uint8(arr, d, 0.9)
        rr = T2.radius_map(d, 2048)
        self.assertTrue((m[rr > 0.9] == 0).all() and (m[rr <= 0.9] == 200).all())
        self.assertIs(T2.limb_mask_uint8(arr, d, None), arr)

    def test_flattened_channel_removes_limb_darkening(self):
        im = self.disk_image()
        d = T2.disk_geometry(im)
        x = T2.input_channels(im, d, 128).astype(np.float32)
        r = T2.radius_map(d, 128)
        inner, outer = (r > 0.2) & (r < 0.4), (r > 0.75) & (r < 0.85)
        flat_gap = abs(x[0][inner].mean() - x[0][outer].mean())
        raw_gap = abs(x[1][inner].mean() - x[1][outer].mean())
        print('\n[limb] inner-vs-outer mean gap: flattened channel %.3f, raw channel %.3f' % (flat_gap, raw_gap))
        self.assertEqual(x.shape, (3, 128, 128))
        self.assertLess(flat_gap, 0.25 * raw_gap)


def identity_model():
    import torch

    class Ident(torch.nn.Module):
        def forward(self, x):
            return torch.logit(x[:, :1].clamp(1e-4, 1 - 1e-4))
    return Ident()


class TestPipeline(Timed):
    def test_crop_dataset_shapes_determinism_and_joint_augmentation(self):
        rng = np.random.default_rng(0)
        Y = (rng.random((3, 64, 64)) > 0.9).astype(np.uint8) * 255
        X = np.stack([np.stack([Y[i] / 255.0, Y[i] / 255.0, np.zeros((64, 64))]) for i in range(3)]).astype(np.float16)
        FG = [np.argwhere(Y[i] > 128).astype(np.int32) for i in range(3)]
        ds = T2.CropDataset(X, Y, FG, 32, 5, 2 / 3)
        for g in range(20):
            x, y = ds[g]
            self.assertEqual((tuple(x.shape), tuple(y.shape), str(x.dtype)), ((3, 32, 32), (1, 32, 32), 'torch.float32'))
            self.assertTrue(np.array_equal(x[2].numpy(), np.zeros((32, 32), np.float32)))
            fg = y[0].numpy() > 0.5
            if fg.any() and (~fg).any():
                self.assertGreater(x[0].numpy()[fg].mean() - x[0].numpy()[~fg].mean(), 0.5)
        a, b = ds[7], ds[7]
        self.assertTrue(np.array_equal(a[0].numpy(), b[0].numpy()))
        self.assertEqual(list(T2.EpochSampler(2, 10)), list(range(20, 30)))

    def test_upsampling_geometry_and_tta(self):                    # reviewer plants P08, P09 and TTA flips
        import torch
        rng = np.random.default_rng(1)
        pat = rng.uniform(0.05, 0.95, (2, 64, 64)).astype(np.float32)
        pat[:, 5:9, 40:60] = 0.99                                   # asymmetric feature
        X = np.stack([np.stack([p, p, p]) for p in pat])
        ref = torch.nn.functional.interpolate(torch.from_numpy(pat[:, None]), size=(2048, 2048), mode='bilinear',
                                              align_corners=False).numpy()[:, 0]
        ref = np.clip(np.rint(255.0 * ref.astype(np.float64)), 0, 255).astype(np.uint8)
        cfg = {'dry': True}
        dev = torch.device('cpu')
        for tta in (False, True):
            got = dict(T2.infer_maps(identity_model(), X, cfg, dev, tta, ['a', 'b']))
            for k, i in (('a', 0), ('b', 1)):
                d = np.abs(got[k].astype(int) - ref[i].astype(int))
                self.assertLessEqual(int(d.max()), 1, 'tta=%s' % tta)   # float averaging of 4 identical views
        blk = got['a'].reshape(64, 32, 64, 32).mean(axis=(1, 3)) / 255.0
        self.assertGreater(float(blk[5:9, 40:60].mean()), 0.85)     # orientation: not transposed, not flipped

    def test_model_keys_and_shapes(self):
        import torch
        m = T2.build_model()
        keys = list(m.encoder.state_dict())
        self.assertEqual(len(keys) + 2, 218)
        for k in ('conv1.weight', 'bn1.running_var', 'layer2.0.downsample.0.weight', 'layer4.2.bn2.num_batches_tracked'):
            self.assertIn(k, keys)
        with torch.no_grad():
            m.eval()
            self.assertEqual(tuple(m(torch.zeros(1, 3, 64, 64)).shape), (1, 1, 64, 64))
        sd = {k: v.clone() for k, v in m.encoder.state_dict().items()}
        sd['fc.weight'] = torch.zeros(1000, 512)
        sd['fc.bias'] = torch.zeros(1000)
        p = TMP / T2.WEIGHTS_FILE
        torch.save(sd, p)
        cfg = {'dry': True, 'weights_path': str(p), 'out': TMP}
        self.assertEqual(T2.load_pretrained(T2.build_model().encoder, cfg, lambda s: None)['tensors'], 216)
        sd['layer9.weight'] = torch.zeros(1)
        torch.save(sd, p)
        with self.assertRaises(T2.Refusal):
            T2.load_pretrained(T2.build_model().encoder, cfg, lambda s: None)
        with self.assertRaises(T2.Refusal):
            T2.load_pretrained(T2.build_model().encoder, {'dry': True, 'out': TMP}, lambda s: None)

    def test_weights_download_refuses_fast_by_name(self):
        t0 = time.time()
        with self.assertRaises(T2.Refusal) as e:      # unreachable address: no network is used
            T2.fetch_weights(TMP / 'w1' / T2.WEIGHTS_FILE, lambda s: None, 10, 'http://127.0.0.1:9/resnet34-b627a593.pth')
        self.assertTrue(str(e.exception).startswith('NO_PRETRAINED_WEIGHTS_DOWNLOAD'))
        src = TMP / 'not_the_weights.bin'
        src.write_bytes(b'x' * 1000)
        with self.assertRaises(T2.Refusal) as e:      # file:// stand-in for a wrong file: hash prefix refused
            T2.fetch_weights(TMP / 'w2' / T2.WEIGHTS_FILE, lambda s: None, 10, src.as_uri())
        self.assertTrue(str(e.exception).startswith('PRETRAINED_HASH_MISMATCH'))
        self.assertFalse((TMP / 'w2' / T2.WEIGHTS_FILE).exists())
        self.assertLess(time.time() - t0, 30)


class TestLoss(Timed):
    def test_loss_does_not_prefer_empty(self):
        import torch
        t = torch.zeros(1, 1, 64, 64)
        t[..., 30:34, 5:60] = 1.0
        e = torch.zeros(1, 1, 64, 64)
        L = lambda p: torch.logit(torch.clamp(p, 1e-4, 1 - 1e-4))
        empty, perfect, weak = L(torch.full_like(t, 1e-3)), L(t * 0.999 + 0.0005), L(t * 0.3 + 1e-3)
        new = {k: float(T2.loss_fn(v, t)[0]) for k, v in (('empty', empty), ('weak_0.3', weak), ('perfect', perfect))}
        tb = torch.cat([t, e])
        right = torch.cat([perfect, L(torch.full_like(e, 1e-3))])
        allempty = torch.cat([empty, L(torch.full_like(e, 1e-3))])
        new_b = {k: float(T2.loss_fn(v, tb)[0]) for k, v in (('correct', right), ('all_empty', allempty))}
        old_b = {k: float(T2.old_per_crop_loss(v, tb)) for k, v in (('correct', right), ('all_empty', allempty))}
        print('\n[loss] filament crop: %s; batch [filament, empty] new %s old %s' % (
            {k: round(v, 4) for k, v in new.items()}, {k: round(v, 4) for k, v in new_b.items()},
            {k: round(v, 4) for k, v in old_b.items()}))
        self.assertGreater(new['empty'], new['weak_0.3'])
        self.assertGreater(new['weak_0.3'], new['perfect'])
        self.assertLess(new_b['correct'], new_b['all_empty'])
        self.assertGreater(old_b['correct'], 0.3)
        self.assertLess(new_b['correct'], 0.05)


def rect_ann(gid, aid, x0, y0, x1, y1):
    return {'id': aid, 'image_id': gid, 'iscrowd': 0, 'segmentation': [[x0, y0, x1, y0, x1, y1, x0, y1]]}


class TestScoringAndCsv(Timed):
    image_id = '20200301120000Ch'

    def toy(self):
        gid = '030101-' + self.image_id
        anns = [rect_ann(gid, 'g1', 300, 900, 600, 960), rect_ann(gid, 'g2', 900, 1100, 1300, 1150),
                rect_ann(gid, 'g3', 1000, 500, 1200, 560)]
        data = {'images': [{'id': gid, 'file_name': self.image_id + '.jpeg', 'height': 2048, 'width': 2048}],
                'annotations': anns}
        row = {'image_id': self.image_id, 'file_name': self.image_id + '.jpeg', 'annotator_image_ids': [gid],
               'annotation_counts': {gid: 3}}
        gt = PROJ.sr.annotation_frame(data, [row])
        mu = PROJ.mask_util
        dec = lambda a: mu.decode({'size': [2048, 2048], 'counts': PROJ.sr.polygon_rle(a['segmentation'], 2048, 2048).encode()})
        arr = np.zeros((2048, 2048), np.uint8)
        arr[dec(anns[0]) > 0] = 230
        arr[dec(anns[1]) > 0] = 230
        arr[40:80, 1000:1100] = 230                    # false positive at the top limb (r > 0.9), area 4000
        return gt, arr

    def score(self, arr, settings, mode):
        gt, _ = self.toy()
        disk = {'cy': 1024.0, 'cx': 1024.0, 'R': 1000.0, 'fallback': False}
        ctx = {'comps': [(gt, None)], 'image_ids': [[self.image_id]]}
        return T2.score_settings(PROJ, spawn_cfg(score_mode=mode), ctx, {self.image_id: arr}, {self.image_id: disk},
                                 settings, 60, print)

    def test_rule_and_scorer_known_pq_in_a_fresh_process(self):
        _, arr = self.toy()
        settings = [(0.5, 0.5, 512, None), (0.5, 0.5, 512, 0.9), (0.5, 0.5, 8192, None)]
        table, complete, st = self.score(arr, settings, 'subprocess')
        rows = T2.rule_table_rows(PROJ.pp, table, settings)
        print('\n[scorer] toy PQ by setting (spawned scorer, child peak %.3f GiB): %s' % (
            st['child_peak_rss_max_gib'], [(r['A'], r['limb'], r['tp'], r['fp'], r['fn'], round(r['pq'], 4)) for r in rows]))
        self.assertTrue(complete)
        self.assertEqual((rows[0]['tp'], rows[0]['fp'], rows[0]['fn']), (2, 1, 1))
        self.assertAlmostEqual(rows[0]['pq'], 2 / 3, places=6)
        self.assertEqual((rows[1]['tp'], rows[1]['fp'], rows[1]['fn']), (2, 0, 1))
        self.assertAlmostEqual(rows[1]['pq'], 0.8, places=6)
        self.assertEqual((rows[2]['tp'], rows[2]['fp'], rows[2]['fn']), (2, 0, 1))
        self.assertTrue(st['child_start'] and all(c['pid'] != os.getpid() for c in st['child_start']))   # another process
        t2, _, _ = self.score(arr, settings, 'inprocess')
        self.assertTrue(np.array_equal(table, t2))

    def test_count_budget_soft_in_grids_hard_in_preflight(self):    # review M1
        arr = np.zeros((2048, 2048), np.uint8)
        arr[100:1900:60, 100:1900:60] = 255                          # 900 isolated pixels -> > 512 predictions
        settings = [(0.5, 0.5, 0, None), (0.5, 0.5, 512, None)]
        table, complete, st = self.score(arr, settings, 'subprocess')
        rows = T2.rule_table_rows(PROJ.pp, table, settings, st['refused'])
        self.assertEqual(rows[0]['status'], 'REFUSED_COUNT:MASK_COUNT_BUDGET')
        self.assertEqual(rows[0]['refusal_numbers']['max_predictions_per_image'], 900)     # the numbers are recorded
        self.assertEqual(rows[1]['status'], 'SCORED')                 # the other setting is still scored
        gt, _ = self.toy()
        ctx = {'comps': [(gt, None)], 'image_ids': [[self.image_id]]}
        disk = {'cy': 1024.0, 'cx': 1024.0, 'R': 1000.0, 'fallback': False}
        with self.assertRaises(T2.Refusal) as e:                      # soft=() as in the preflight: hard, by name
            T2.score_settings(PROJ, spawn_cfg(), ctx, {self.image_id: arr}, {self.image_id: disk}, settings[:1], 60,
                              print, soft=())
        self.assertIn('MASK_COUNT_BUDGET', str(e.exception))
        self.assertIn('fresh scoring process', str(e.exception))

    def test_periodic_not_scoreable_and_final_hard(self):        # coordinator ruling (1)
        import inspect
        ok = [{'status': 'SCORED', 'pq': 0.31}, {'status': 'SCORED', 'pq': 0.29}]
        self.assertEqual(T2.periodic_decision(ok, {}, 0.25), (True, True, 0.31))
        self.assertEqual(T2.periodic_decision(ok, {}, 0.40), (True, False, 0.31))
        refused = {'0.25|0.5|128|None': {'code': 'MASK_COUNT_BUDGET', 'max_predictions_per_image': 700}}
        part = ok + [{'status': 'REFUSED_COUNT:MASK_COUNT_BUDGET'}]
        self.assertEqual(T2.periodic_decision(part, refused, 0.0)[:2], (False, False))   # no checkpoint from it
        self.assertEqual(T2.periodic_decision([{'status': 'NOT_SCORED_DEADLINE'}], {}, -1.0)[:2], (False, False))
        src = inspect.getsource(T2.final_stage)
        self.assertRegex(src, r"score_settings\([^)]*soft=\(\)")                  # final validation: hard
        self.assertRegex(inspect.getsource(T2.scoring_preflight), r"soft=\(\)")     # preflight: hard

    def test_csv_writer_validator_and_lf_bytes(self):              # reviewer plant P14
        _, arr = self.toy()
        pt, vs = PROJ.pt, PROJ.vs
        rows = []
        for i in SYN['ids']['test']:
            rows.extend(pt.rows_from_uint8(i, arr, (0.5, 0.5, 512)))
        self.assertEqual(len(rows), 3 * len(SYN['ids']['test']))
        p = TMP / 'toy_submission.csv'
        pt.write_csv(p, rows)
        raw = p.read_bytes()
        self.assertTrue(raw.startswith(b'filament_id,segmentation_rle\n') and raw.endswith(b'\n'))
        self.assertNotIn(b'\r', raw)
        summary = vs.validate_file(p, Path(SYN['dataset_root']) / 'test/test_images', None, expect_images=len(SYN['ids']['test']))
        self.assertTrue(summary['ok'] and summary['overlap'] == 0 and summary['images_without_rows'] == 0)
        forbidden = pt.load_forbidden(SYN['split_path'], SYN['pins']['split'])
        with open(p, newline='') as f:
            back = list(csv.DictReader(f))
        with self.assertRaises(ValueError) as e:
            vs.validate_rows([dict(back[0], filament_id=SYN['ids']['holdout'][0] + '_0001')], forbidden=forbidden)
        self.assertEqual(str(e.exception), 'SPLIT_IMAGE_REFUSED')

    def test_finalize_csv_leaves_no_submittable_file_on_failure(self):   # M1
        _, arr = self.toy()
        rows = PROJ.pt.rows_from_uint8(SYN['ids']['test'][0], arr, (0.5, 0.5, 512))
        ok = lambda p: {'ok': True}

        def bad_validator(p):
            raise ValueError('EMPTY_FRACTION')
        for k, (val, red) in enumerate(((bad_validator, lambda: rows), (ok, lambda: rows[:1]), (lambda p: {'ok': False}, lambda: rows))):
            d = TMP / ('m1_%d' % k)
            d.mkdir()
            with self.assertRaises((ValueError, T2.Refusal)):
                T2.finalize_csv(PROJ.pt, d, rows, 'submission.csv', val, red)
            self.assertEqual(sorted(x.name for x in d.iterdir()), [], 'case %d' % k)
        d = TMP / 'm1_ok'
        d.mkdir()
        final, _ = T2.finalize_csv(PROJ.pt, d, rows, 'submission.csv', ok, lambda: rows)
        self.assertEqual(sorted(x.name for x in d.iterdir()), ['submission.csv'])


class TestTestPath(Timed):
    """M3: the test path (predict_image_rows + FROZEN_RULE.json read back) against the validation path."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.dev = torch.device('cpu')
        cls.cfg = dict(T2.DRY_CFG, dataset_root=SYN['dataset_root'])
        cls.model = T2.build_darkness_model()
        va = preload_syn('validation', False)
        cls.ids = [v['image_id'] for v in va]
        cls.disks = {v['image_id']: v['disk'] for v in va}
        xv = np.stack([v['x'] for v in va])
        cls.maps = dict(T2.infer_maps(cls.model, xv, cls.cfg, cls.dev, True, cls.ids))
        ids = cls.ids
        cls.xv = staticmethod(lambda i: xv[ids.index(i)])
        cls.rows_by_id = {r['image_id']: r for r in census_rows(set(cls.ids))}
        cls.allowed = set(SYN['ids']['train']) | set(SYN['ids']['validation'])

    def frozen(self, rule, name):
        p = TMP / name
        p.write_text(json.dumps({'rule': rule, 'verdict': 'CONTINUE', 'weights_sha256': 'x'}))
        return p

    def test_test_path_equals_validation_path(self):               # reviewer plants P10, P13, P15
        chosen = (0.5, 0.5, 512, 0.9)
        p = self.frozen({'T_low': 0.5, 'T_high': 0.5, 'A': 512, 'limb': 0.9}, 'fr1.json')
        res = T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, p, chosen, self.maps, self.disks,
                                      self.rows_by_id, self.allowed, self.ids, print, self.xv)
        self.assertEqual([r['rows_test_path'] for r in res], [3] * len(self.ids))   # three filaments, limb blob masked
        # the limb mask is visible: without it every image gains the blob at r/R 0.95
        nolimb = [len(PROJ.pt.rows_from_uint8(i, self.maps[i], (0.5, 0.5, 512))) for i in self.ids]
        self.assertEqual(nolimb, [4] * len(self.ids))
        with self.assertRaises(T2.Refusal):     # a frozen rule that disagrees with the scored setting is caught
            T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, self.frozen(
                {'T_low': 0.5, 'T_high': 0.5, 'A': 512, 'limb': None}, 'fr2.json'), chosen, self.maps, self.disks,
                self.rows_by_id, self.allowed, self.ids[:1], print, self.xv)

    def test_batch_identity_failure_names_its_numbers(self):       # coordinator point 3
        chosen = (0.5, 0.5, 512, 0.9)
        p = self.frozen({'T_low': 0.5, 'T_high': 0.5, 'A': 512, 'limb': 0.9}, 'fr3.json')
        i = self.ids[0]
        shifted = dict(self.maps)
        m = self.maps[i].astype(np.int16)
        m[m > 0] -= 1                                      # a one-level difference on the non-zero pixels
        shifted[i] = m.clip(0, 255).astype(np.uint8)
        with self.assertRaises(T2.Refusal) as e:          # full run: strict
            T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, p, chosen, shifted, self.disks,
                                    self.rows_by_id, self.allowed, [i], print, self.xv)
        msg = str(e.exception)
        print('\n[self-check failure message] %s' % msg)
        self.assertIn('max abs diff 1', msg)
        self.assertRegex(msg, r'\d+ pixels differ')
        res = T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, p, chosen, shifted, self.disks,
                                      self.rows_by_id, self.allowed, [i], print, self.xv, strict_batch=False)
        self.assertEqual(res[0]['scored_batch4']['max_abs_diff'], 1)           # smoke: reported, not fatal
        self.assertGreater(res[0]['scored_batch4']['pixels_differing'], 0)
        self.assertEqual(res[0]['same_batch']['max_abs_diff'], 0)

    def test_r3b_batch_tolerance_passes_noise_and_refuses_gross(self):          # r3b, ruled from the r3 smoke run
        chosen = (0.5, 0.5, 512, 0.9)
        p = self.frozen({'T_low': 0.5, 'T_high': 0.5, 'A': 512, 'limb': 0.9}, 'fr4.json')
        i = self.ids[0]
        base = self.maps[i]
        noisy = dict(self.maps)                             # one level on about 2 % of the pixels: GPU rounding noise
        m = base.astype(np.int16).ravel().copy()
        sel = np.flatnonzero(m > 0)[::50]                   # spread over the image, about 2 % of its non-zero pixels
        m[sel] -= 1
        noisy[i] = m.reshape(base.shape).clip(0, 255).astype(np.uint8)
        res = T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, p, chosen, noisy, self.disks,
                                      self.rows_by_id, self.allowed, [i], print, self.xv)        # full run: passes
        self.assertLessEqual(res[0]['scored_batch4']['max_abs_diff'], T2.BATCH_TOL_MAX_LEVELS)
        self.assertLessEqual(res[0]['scored_batch4']['pixels_differing'], T2.BATCH_TOL_SHARE * base.size)
        self.assertGreater(res[0]['scored_batch4']['pixels_differing'], 0)      # the noise case is not empty
        self.assertEqual(res[0]['same_batch']['max_abs_diff'], 0)               # (a) stays exact
        gross = dict(self.maps)                             # five levels: more than the tolerance, refused by name
        g = base.astype(np.int16)
        g[g > 5] -= 5
        gross[i] = g.clip(0, 255).astype(np.uint8)
        with self.assertRaises(T2.Refusal) as e:
            T2.self_check_test_path(PROJ, self.cfg, self.model, self.dev, p, chosen, gross, self.disks,
                                    self.rows_by_id, self.allowed, [i], print, self.xv)
        self.assertIn('max abs diff 5', str(e.exception))
        self.assertEqual((T2.BATCH_TOL_MAX_LEVELS, T2.BATCH_TOL_SHARE), (4, 0.10))

    def test_frozen_thresholds_are_used(self):                     # reviewer plant P11
        import torch
        logit = torch.full((1, 1, 256, 256), -8.0)
        logit[..., 60:70, 40:120] = 2.2          # p 0.90: seeded at T_high 0.7
        logit[..., 150:160, 40:120] = 0.0        # p 0.50: above T_low 0.3, never reaches T_high 0.7

        class Fixed(torch.nn.Module):
            def forward(self, x):
                return logit.expand(x.shape[0], 1, 256, 256)
        im = np.full((2048, 2048), 150, np.uint8)
        cfg = dict(self.cfg, tta=False)
        rule = {'T_low': 0.3, 'T_high': 0.7, 'A': 64, 'limb': None}
        arr, rows = T2.predict_image_rows(PROJ, Fixed(), cfg, self.dev, SYN['ids']['test'][0], im, rule)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows, PROJ.pt.rows_from_uint8(SYN['ids']['test'][0], arr, (0.3, 0.7, 64)))
        self.assertEqual(len(PROJ.pt.rows_from_uint8(SYN['ids']['test'][0], arr, (0.3, 0.3, 64))), 2)


class TestScoringFaults(Timed):
    """B2: a scoring child that cannot start, dies mid-task or hangs must end in a NAMED error within its deadline."""

    def ctx(self):
        i = '20200301120000Ch'
        g = '030101-' + i
        gt = PROJ.sr.annotation_frame({'images': [{'id': g, 'file_name': i + '.jpeg', 'height': 2048, 'width': 2048}],
                                       'annotations': [rect_ann(g, 'g1', 300, 900, 600, 960)]},
                                      [{'image_id': i, 'file_name': i + '.jpeg', 'annotator_image_ids': [g],
                                        'annotation_counts': {g: 1}}])
        return ({'comps': [(gt, None)], 'image_ids': [[i]]}, {i: np.zeros((2048, 2048), np.uint8)},
                {i: {'cy': 1024.0, 'cx': 1024.0, 'R': 1000.0, 'fallback': False}})

    def run_fault(self, proj, **cfg):
        ctx, maps, disks = self.ctx()
        t0 = time.time()
        with self.assertRaises(T2.Refusal) as e:
            T2.score_settings(proj, spawn_cfg(**cfg), ctx, maps, disks, [(0.5, 0.5, 0, None)], 600, lambda s: None)
        return str(e.exception), time.time() - t0

    def test_child_import_fails(self):
        broken = TMP / 'broken_src'                     # the reviewer's demo: score_reference.py missing in the root
        if not broken.exists():
            shutil.copytree(SRC, broken)
            (broken / 'score_reference.py').unlink()
        proj = T2.Project.__new__(T2.Project)
        proj.root = broken
        msg, sec = self.run_fault(proj)
        print('\n[fault import, broken bundle] %.1f s: %s' % (sec, msg[:110]))
        self.assertTrue(msg.startswith('SCORING_CHILD_START_FAILED'))
        self.assertIn('score_reference.py', msg)
        self.assertLess(sec, 30)
        msg, sec = self.run_fault(PROJ, worker_fault='import')
        print('[fault import, injected] %.1f s: %s' % (sec, msg[:80]))
        self.assertTrue(msg.startswith('SCORING_CHILD_START_FAILED') and 'QMAI_TEST_FAULT_IMPORT' in msg)
        self.assertLess(sec, 30)

    def test_child_killed_mid_task(self):
        msg, sec = self.run_fault(PROJ, worker_fault='kill')
        print('\n[fault kill] %.1f s: %s' % (sec, msg[:80]))
        self.assertTrue(msg.startswith('SCORING_CHILD_DIED: exit code -9'))
        self.assertLess(sec, 30)

    def test_child_sleeps_past_task_deadline(self):
        msg, sec = self.run_fault(PROJ, worker_fault='sleep', score_task_base_s=4, score_task_per_setting_image_s=0)
        print('\n[fault sleep, task deadline 4 s] %.1f s: %s' % (sec, msg[:80]))
        self.assertTrue(msg.startswith('SCORING_TASK_TIMEOUT'))
        self.assertLess(sec, 4 + 30)                   # start-up (torch import) + the 4 s task deadline

    def test_child_start_deadline(self):
        msg, sec = self.run_fault(PROJ, score_start_s=0.5)
        print('\n[fault start deadline 0.5 s] %.1f s: %s' % (sec, msg[:80]))
        self.assertTrue(msg.startswith('SCORING_CHILD_START_TIMEOUT'))
        self.assertLess(sec, 15)

    def test_scoring_children_are_fresh_interpreters_not_fork(self):   # coordinator point 2
        assert_no_multiprocessing_in_scoring(self)
        T2._FORK_SENTINEL = 'parent'           # a forked child would see this; a fresh interpreter cannot
        try:
            pool = T2.ScoringPool(PROJ, spawn_cfg(), 2, lambda s: None)
            try:
                for c in pool.w:
                    self.assertEqual(c['p'].args[0], sys.executable)
                    self.assertTrue(c['p'].args[2].endswith('track2_train_r3b/score_worker.py'))
                    self.assertIsNone(c['ready']['fork_sentinel'])
                    self.assertEqual(c['ready']['argv0'], 'score_worker.py')
                    self.assertNotEqual(c['ready']['pid'], os.getpid())
            finally:
                pool.close()
        finally:
            del T2._FORK_SENTINEL

    def test_worker_entry_is_an_importable_module(self):            # coordinator point 2
        path = SRC / 'track2_train_r3b' / 'score_worker.py'
        spec = importlib.util.spec_from_file_location('score_worker', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)               # importing must not start anything
        self.assertTrue(callable(mod.main))
        r = subprocess.run([sys.executable, '-B', '-c', 'import sys; sys.path.insert(0, %r); import score_worker; '
                            'print(score_worker.main.__name__)' % str(path.parent)], capture_output=True, text=True,
                           timeout=60)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, 'main'), r.stderr[-500:])

    def test_fault_injection_is_dry_only(self):
        ctx, maps, disks = self.ctx()
        with self.assertRaises(T2.Refusal) as e:
            T2.score_settings(PROJ, dict(spawn_cfg(worker_fault='kill'), dry=False), ctx, maps, disks,
                              [(0.5, 0.5, 0, None)], 60, lambda s: None)
        self.assertEqual(str(e.exception), 'TEST_FAULT_DRY_ONLY')


class TestDryKernel(Timed):
    """The three dry kernels started in setUpModule (in parallel)."""

    def result(self, name):
        work, proc, t0, how = DRY[name]
        out, err = proc.communicate(timeout=TEST_SECONDS - 20)
        print('\n[dry %s, launched by %s] %.1f s; last log line: %s' % (
            name, how, time.time() - t0, out.strip().splitlines()[-1] if out.strip() else ''))
        return work / 'out', proc.returncode, out, err

    def common(self, out):
        prov = json.loads((out / 'PROVENANCE.json').read_text())
        self.assertEqual(len(prov['bundle']['bundle_sha256']), 64)
        pre = json.loads((out / 'SCORING_PREFLIGHT.json').read_text())
        self.assertEqual(pre['mode'], 'subprocess')
        self.assertTrue(pre['child_start_seconds'] and max(pre['child_start_seconds']) < 60)
        self.assertGreater(pre['oracle_union_map']['pq'], 0.9)
        self.assertLess(pre['child_peak_rss_max_gib'], 1.0)
        log = (out / 'RUN_LOG.txt').read_text()
        for word in ('hardware:', 'crops/s', 'VALIDATION epoch 1', 'EPOCH 1 done', 'SCORING PREFLIGHT OK'):
            self.assertIn(word, log)

    def test_continue_branch_end_to_end(self):
        out, rc, so, se = self.result('continue')
        self.assertEqual(rc, 0, so[-3000:] + se[-3000:])
        self.common(out)
        st = json.loads((out / 'STATUS.json').read_text())
        self.assertEqual((st['status'], st['submitted'], st['test_images_read']), ('COMPLETE', False, len(SYN['ids']['test'])))
        stop = json.loads((out / 'STOP_RULE.json').read_text())
        self.assertEqual(stop['verdict'], 'CONTINUE')
        self.assertEqual(stop['chosen_rule'], [0.5, 0.5, 512, 0.9])
        self.assertTrue((out / 'submission.csv').is_file())
        self.assertFalse(any(p.name.startswith('UNVALIDATED') or 'BLOCKED' in p.name for p in out.iterdir()))
        raw = (out / 'submission.csv').read_bytes()
        self.assertNotIn(b'\r', raw)
        self.assertEqual(raw.count(b'\n'), 1 + 3 * len(SYN['ids']['test']))      # three filaments per test image
        sv = json.loads((out / 'SUBMISSION_VALIDATION.json').read_text())
        self.assertTrue(sv['validator']['ok'])
        self.assertEqual(sv['sha256'], T2.sha256_file(out / 'submission.csv'))
        sc = json.loads((out / 'TEST_PATH_SELF_CHECK.json').read_text())
        self.assertTrue(all(r['rows_test_path'] == 3 and r['same_batch']['rows_equal'] and r['scored_batch4']['rows_equal']
                            and r['same_batch']['max_abs_diff'] == 0 and r['scored_batch4']['max_abs_diff'] == 0
                            for r in sc))
        frozen = json.loads((out / 'FROZEN_RULE.json').read_text())
        self.assertLess(frozen['frozen_at_epoch'], (out / 'test_maps/INDEX.json').stat().st_mtime)
        self.assertEqual(len(list((out / 'test_maps').glob('*.npz'))), len(SYN['ids']['test']))
        index = PROJ.ph.load_index(out / 'val_maps')
        self.assertEqual(sorted(index['images']), sorted(SYN['ids']['validation']))
        m = PROJ.ph.load_map(out / 'val_maps', index, SYN['ids']['validation'][0], set(SYN['ids']['validation']))
        self.assertEqual((m.dtype, m.shape), (np.dtype('uint8'), (2048, 2048)))
        score = json.loads((out / 'SCORE.json').read_text())
        self.assertEqual(score['weights_sha256'], index['weights_sha256'])
        self.assertTrue(all(k in score['pooled_area512'] for k in ('tp', 'fp', 'fn')))
        self.assertEqual(T2.sha256_file(out / 'checkpoints/best.pt'), score['weights_sha256'])

    def test_stop_branch_reads_no_test_image_and_writes_no_csv(self):   # M2
        out, rc, so, se = self.result('stop')
        self.assertEqual(rc, 0, so[-3000:] + se[-3000:])
        self.common(out)
        st = json.loads((out / 'STATUS.json').read_text())
        self.assertEqual((st['status'], st['test_images_read']), ('COMPLETE_STOPPED', 0))
        self.assertEqual(json.loads((out / 'STOP_RULE.json').read_text())['verdict'], 'STOP')
        self.assertEqual([p.name for p in out.rglob('*.csv')], ['RULE_TABLE.csv'])
        for name in ('test_maps', 'SUBMISSION_VALIDATION.json', 'TEST_PATH_SELF_CHECK.json'):
            self.assertFalse((out / name).exists(), name)
        self.assertIn('NO TEST IMAGE IS READ', (out / 'RUN_LOG.txt').read_text())

    def test_smoke_mode(self):
        out, rc, so, se = self.result('smoke')
        self.assertEqual(rc, 0, so[-3000:] + se[-3000:])
        self.common(out)
        sm = json.loads((out / 'SMOKE_OK.json').read_text())
        self.assertEqual((sm['status'], sm['test_images_read'], sm['csv_written'], sm['checkpoint_reload_ok']),
                         ('SMOKE_OK', 0, False, True))
        self.assertEqual(list(out.rglob('*.csv')), [])
        self.assertFalse((out / 'test_maps').exists())
        self.assertEqual(json.loads((out / 'STATUS.json').read_text())['status'], 'SMOKE_OK')


@unittest.skipUnless(os.environ.get('TRACK2_REAL_SIZE') == '1', 'real-size rehearsal: set TRACK2_REAL_SIZE=1 and hold HEAVY.lock')
class TestRealSize(Timed):
    LIMIT = 600
    """B1 at real size: the parent holds the real arrays (~3.86 GiB, every page written). Scoring in fresh spawned
    processes must pass; the same run with scoring forced into the parent must fail with SCORER_RSS_BUDGET."""

    def run_dry(self, name, extra):
        work = TMP / ('realsize_' + name)
        work.mkdir()
        t0 = time.time()
        r = subprocess.run([sys.executable, '-B', str(KERNEL), '--dry', '--work', str(work), '--ballast'] + extra,
                           capture_output=True, text=True, timeout=280)
        return work / 'out', r, time.time() - t0

    def test_real_size_spawn_passes_and_inprocess_fails(self):
        out, r, sec = self.run_dry('spawn', [])
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        pre = json.loads((out / 'SCORING_PREFLIGHT.json').read_text())
        st = json.loads((out / 'STATUS.json').read_text())
        print('\n[real size, spawn] %.0f s, status %s; parent peak %.2f GiB; scoring child peak %.3f GiB' % (
            sec, st['status'], pre['parent_peak_rss_gib'], pre['child_peak_rss_max_gib']))
        # 3.86 GiB of random bytes are written; how much macOS keeps resident depends on its memory pressure. The
        # property that matters: the parent is above the scorer's 2 GiB guard while scoring passes in the child.
        self.assertGreater(pre['parent_peak_rss_gib'], 2.0)
        self.assertTrue(pre['child_start_seconds'])
        self.assertLess(pre['child_peak_rss_max_gib'], 1.0)
        self.assertEqual(st['status'], 'COMPLETE_STOPPED')
        out2, r2, sec2 = self.run_dry('inprocess', ['--score-inprocess'])
        st2 = json.loads((out2 / 'STATUS.json').read_text())
        print('[real size, forced in-process] %.0f s, rc %d, status %s, parent peak %.2f GiB, error %s' % (
            sec2, r2.returncode, st2['status'], st2['parent_peak_rss_gib'], st2['error'][:160]))
        self.assertNotEqual(r2.returncode, 0)
        self.assertEqual(st2['status'], 'FAILED')
        self.assertIn('SCORER_RSS_BUDGET', st2['error'])
        self.assertFalse((out2 / 'checkpoints').exists())          # refused at the preflight, before any training


if __name__ == '__main__':
    unittest.main(verbosity=2)
