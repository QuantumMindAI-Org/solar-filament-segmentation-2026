"""Render the TRACK2-R3 Kaggle push folders, offline only:
  kernel_smoke/  (quantummindaillc/qmai-solar-track2-r3b-smoke)  SMOKE mode, the mandatory first push
  kernel/        (quantummindaillc/qmai-solar-track2-r3b)        the full run
Both embed the same bundle; they differ only in MODE and in the kernel id/title.

The official push uploads only the declared code_file, so (as neural_train_r3/pack_r3.py did) every project file
the kernel needs is embedded in one script as a zlib+base64 JSON bundle with a sha256 per file and one for the
bundle. Contract files are checked against their pins before they are packed.

  python3 -B pack.py            writes kernel/ and kernel_smoke/, prints the hashes
  python3 -B pack.py --check    renders in memory and compares (exit 1 on drift)
"""
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
FILES = (
    'official_scorer.py', 'SCORER_FINDINGS.json', 'score_reference.py', 'DATA_HASHES.json',
    'evaluation-split-r1/SPLIT_PLAN.json', 'evaluation-split-r1/PIXEL_CENSUS.json',
    'neural_baseline_r1/__init__.py', 'neural_baseline_r1/data.py',
    'postproc_minarea_r1/pp_minarea.py', 'postproc_morph_r2/pp_morph.py', 'postproc_hysteresis_r3/pp_hysteresis.py',
    'submission_r1/predict_test.py', 'submission_r1/validate_submission.py',
    'track2_train_r3b/track2.py', 'track2_train_r3b/score_worker.py',
)
CONTRACT_PINS = {
    'evaluation-split-r1/SPLIT_PLAN.json': '27e3f1056120c6f96ec38d141b6add66a98fdb5daa014c5f67fd558133b515cc',
    'evaluation-split-r1/PIXEL_CENSUS.json': '32b05e2ba7fc255dfd3fa8d43323eed51432e646b7396a077aeb557c209af5c4',
    'DATA_HASHES.json': 'a9b7e9859ce208db60a6231b06eba089890654219c7ff0347a23962774308ffe',
    'score_reference.py': 'c6cf51893ab69045d9b37f479a325275c9c752229eea4800c1ff200f242bccd8',
    'official_scorer.py': 'ab1cc26f86f76293d6760cb3f076b4800d607f4a2c39772e2c8e38d6b7b43a0a',
    'SCORER_FINDINGS.json': 'b36137a7b4902bd1aa0f5b3c8791adfd3b4e581df20875cfc9aa14fd71ad3d04',
}
KERNELS = {'kernel': ('full', 'quantummindaillc/qmai-solar-track2-r3b', 'QMAI Solar Track2 R3b'),
           'kernel_smoke': ('smoke', 'quantummindaillc/qmai-solar-track2-r3b-smoke', 'QMAI Solar Track2 R3b Smoke')}
METADATA = {
    'id': None, 'title': None, 'code_file': 'track2_kernel.py', 'language': 'python',
    'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
    'enable_internet': True,   # only to fetch the ImageNet ResNet34 weights; see OPEN_POINTS.md item 1
    'machine_shape': 'NvidiaTeslaT4', 'competition_sources': ['filament-segmentation-2026'],
    'dataset_sources': [], 'model_sources': [], 'kernel_sources': [], 'docker_image_pinning_type': 'original',
}

BOOT = '''# QMAI solar track2 r3 Kaggle kernel (MODE below). Generated offline by track2_train_r3b/pack.py. It never submits.
# The project files are embedded below (zlib + base64 JSON, sha256 per file and for the bundle).
import base64, hashlib, json, sys, zlib
from pathlib import Path
MODE = %(mode)r
BUNDLE_SHA256 = %(pin)r
BUNDLE = %(bundle)r


def extract(dest):
    raw = zlib.decompress(base64.b64decode(BUNDLE, validate=True))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise SystemExit('BUNDLE_HASH_MISMATCH')
    files = json.loads(raw)['files']
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    for name, item in sorted(files.items()):
        data = base64.b64decode(item['base64'], validate=True)
        if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
            raise SystemExit('BUNDLE_FILE_HASH_MISMATCH: ' + name)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'xb') as stream:
            stream.write(data)
    return {name: item['sha256'] for name, item in files.items()}


def main(argv):
    argv = list(argv)
    if MODE == 'smoke' and '--smoke' not in argv:
        argv.insert(0, '--smoke')
    if '--dry' in argv:
        dest = Path(argv[argv.index('--work') + 1]).absolute() / 'qmai_src'
    else:
        dest = Path('/kaggle/working/qmai_src')
    manifest = extract(dest)
    try:
        own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:  # __file__ may be undefined under a notebook-style runner
        own = None
    sys.path.insert(0, str(dest / 'track2_train_r3b'))
    import track2
    return track2.main(argv, dest, {'bundle_sha256': BUNDLE_SHA256, 'files_sha256': manifest, 'mode': MODE,
                                    'kernel_script_sha256': own})


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def collect():
    files = {}
    for name in FILES:
        p = R / name
        if p.is_symlink() or not p.is_file():
            raise ValueError('MISSING_OR_LINK: ' + name)
        raw = p.read_bytes()
        if name in CONTRACT_PINS and sha(raw) != CONTRACT_PINS[name]:
            raise ValueError('CONTRACT_PIN: ' + name)
        files[name] = {'bytes': len(raw), 'sha256': sha(raw), 'base64': base64.b64encode(raw).decode('ascii')}
    raw = json.dumps({'schema': 'qmai.solar-track2-bundle/v1', 'files': files}, sort_keys=True,
                     separators=(',', ':')).encode()
    return raw, files


def render():
    raw, files = collect()
    bundle = base64.b64encode(zlib.compress(raw, 9)).decode('ascii')
    out = {}
    for folder, (mode, kid, title) in KERNELS.items():
        script = (BOOT % {'pin': sha(raw), 'bundle': bundle, 'mode': mode}).encode()
        if len(script) > 4 * 1024 ** 2:
            raise ValueError('SCRIPT_SIZE')
        meta = (json.dumps(dict(METADATA, id=kid, title=title), sort_keys=True, indent=2) + '\n').encode()
        out[folder + '/track2_kernel.py'] = script
        out[folder + '/kernel-metadata.json'] = meta
    return out, sha(raw), {n: e['sha256'] for n, e in files.items()}


def main(argv):
    rendered, pin, files = render()
    out = HERE
    if '--check' in argv:
        drift = [n for n, v in rendered.items() if not (out / n).is_file() or (out / n).read_bytes() != v]
        print(json.dumps({'bundle_sha256': pin, 'drift': drift}, sort_keys=True))
        return 1 if drift else 0
    for name, value in rendered.items():
        (out / name).parent.mkdir(exist_ok=True)
        (out / name).write_bytes(value)
    print(json.dumps({'bundle_sha256': pin, 'kernel_files': {n: sha(v) for n, v in rendered.items()},
                      'bundled': files, 'uploaded': False}, sort_keys=True, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
