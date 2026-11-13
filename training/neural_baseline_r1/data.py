"""Lazy, pinned metadata and selected-only image/mask access. No import-time I/O."""
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from types import ModuleType

PINS = {
    'evaluation-split-r1/SPLIT_PLAN.json': '27e3f1056120c6f96ec38d141b6add66a98fdb5daa014c5f67fd558133b515cc',
    'evaluation-split-r1/PIXEL_CENSUS.json': '32b05e2ba7fc255dfd3fa8d43323eed51432e646b7396a077aeb557c209af5c4',
    'DATA_HASHES.json': 'a9b7e9859ce208db60a6231b06eba089890654219c7ff0347a23962774308ffe',
    'score_reference.py': 'c6cf51893ab69045d9b37f479a325275c9c752229eea4800c1ff200f242bccd8',
    'official_scorer.py': 'ab1cc26f86f76293d6760cb3f076b4800d607f4a2c39772e2c8e38d6b7b43a0a',
    'SCORER_FINDINGS.json': 'b36137a7b4902bd1aa0f5b3c8791adfd3b4e581df20875cfc9aa14fd71ad3d04',
}
ANNOTATION_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
IMAGE_ID = re.compile(r'[0-9]{14}[A-Za-z]{2}\Z')
COUNTS = {'train': 443, 'validation': 141, 'holdout': 123}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum, expected=None):
    """Read one regular, bounded file through non-symlink parent descriptors."""
    path = Path(path).absolute()
    require(not set(path.parts) & {'.codex', '.claude', '.grok', '.grok-accounts',
                                  '.ssh', '.aws', '.config', 'credentials', 'sessions'}, 'PRIVATE_PATH')
    with ExitStack() as stack:
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        for part in path.parts[1:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        stack.callback(os.close, fd)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, 'INPUT_BOUND')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(fd)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(len(raw) == before.st_size and signature(before) == signature(after), 'INPUT_CHANGED')
    require(expected is None or digest(raw) == expected, 'INPUT_HASH')
    return raw


def partition_index(split, census, *, expected=COUNTS):
    records = census['records']
    by_id = {row['image_id']: row for row in records}
    require(len(by_id) == len(records), 'DUPLICATE_IMAGE')
    assigned = set(); partitions = {name: [] for name in expected}; components = []
    diagnostic = set(split['input_provenance']['diagnostic_image_ids'])
    group_ids = set(); annotation_ids = set()
    for group in split['groups']:
        name = group['partition']; ids = group['image_ids']
        require(name in partitions and ids and len(set(ids)) == len(ids) and
                set(ids) <= by_id.keys() and not assigned.intersection(ids) and
                group['group_id'] not in group_ids, 'PARTITION_MEMBERSHIP')
        assigned.update(ids); group_ids.add(group['group_id'])
        require(not diagnostic.intersection(ids) or name == 'train', 'DIAGNOSTIC_LEAK')
        rows = []
        for identity in ids:
            row = by_id[identity]
            require(IMAGE_ID.fullmatch(identity) and row['file_name'] == identity + '.jpeg', 'IMAGE_ID')
            annotations = row['annotator_image_ids']
            require(annotations and len(set(annotations)) == len(annotations) and
                    not annotation_ids.intersection(annotations), 'ANNOTATOR_IDS')
            require(all(a.split('-', 1)[-1] == identity for a in annotations), 'ANNOTATOR_IMAGE_JOIN')
            annotation_ids.update(annotations); rows.append(row)
        partitions[name].extend(rows)
        if name == 'validation':
            components.append({'group_id': group['group_id'], 'images': rows})
    require(assigned == by_id.keys() and {k: len(v) for k, v in partitions.items()} == expected, 'PARTITION_COUNTS')
    return partitions, components


class Data:
    def __init__(self, contract_root, dataset_root, *, check=lambda: None):
        self.root = Path(contract_root).absolute()
        self.dataset = Path(dataset_root).absolute()
        self.check = check
        raw = {name: read(self.root/name, 2*1024**2, pin) for name, pin in PINS.items()}
        split = json.loads(raw['evaluation-split-r1/SPLIT_PLAN.json'])
        census = json.loads(raw['evaluation-split-r1/PIXEL_CENSUS.json'])
        require(split['pixel_census_sha256'] == PINS['evaluation-split-r1/PIXEL_CENSUS.json'] and
                split['input_provenance']['official_metadata_sha256'] == ANNOTATION_SHA, 'SPLIT_JOIN')
        self.partitions, self.components = partition_index(split, census)
        require(len(self.components) == 34 and sum(len(r['annotator_image_ids'])
                for r in self.partitions['validation']) == 234, 'FULL_VALIDATION')
        manifest = json.loads(raw['DATA_HASHES.json'])
        require(manifest['extracted'] is True and manifest['archive_sha256'] == split['input_provenance']['archive_sha256']
                and split['input_provenance']['dataset_manifest_sha256'] == PINS['DATA_HASHES.json'], 'MANIFEST_JOIN')
        entries = {row['path']: row for row in manifest['files']}
        require(len(entries) == len(manifest['files']), 'DUPLICATE_MANIFEST')
        for rows in self.partitions.values():
            for row in rows:
                expected_path = 'data/official/MAGFiLO_1.0_Kaggle_2026/train/train_images/' + row['file_name']
                require(row['path'] == expected_path and entries[expected_path]['sha256'] == row['file_sha256']
                        and entries[expected_path]['bytes'] == row['bytes'], 'PHOTO_MANIFEST_JOIN')
        self.allowed = {r['image_id']: r for name in ('train', 'validation') for r in self.partitions[name]}
        self.helper = ModuleType('_neural_pinned_scorer_helper')
        self.helper.__file__ = str(self.root/'score_reference.py')
        exec(compile(raw['score_reference.py'], self.helper.__file__, 'exec'), self.helper.__dict__)
        # Set inherited scorer thread limits before any training starts.
        self.np, self.pd, self.Image, self.masks, self.scorer = self.helper.dependencies()
        self.check()
        payload = read(self.dataset/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json', 64*1024**2, ANNOTATION_SHA)
        self.annotations = json.loads(payload); del payload
        require(type(self.annotations) is dict and type(self.annotations.get('annotations')) is list
                and type(self.annotations.get('images')) is list, 'ANNOTATION_SCHEMA')
        image_groups = {row['id']: row for row in self.annotations['images']}
        require(len(image_groups) == len(self.annotations['images']), 'DUPLICATE_ANNOTATOR_IMAGE')
        for row in self.allowed.values():
            for group in row['annotator_image_ids']:
                item = image_groups.get(group, {})
                require(item.get('file_name') == row['file_name'] and item.get('height') == 2048
                        and item.get('width') == 2048, 'ANNOTATION_IMAGE_JOIN')
        self.by_group = {}
        annotation_ids = set()
        for item in self.annotations['annotations']:
            self.check()
            group = item.get('image_id')
            # The shared JSON is parsed, but holdout polygons are never converted.
            if type(group) is str and group.split('-', 1)[-1] in self.allowed:
                identity = item.get('id')
                require(type(identity) is str and identity not in annotation_ids, 'DUPLICATE_ANNOTATION')
                annotation_ids.add(identity)
                self.by_group.setdefault(group, []).append(item)
        for row in self.allowed.values():
            require(all(1 <= len(self.by_group.get(g, [])) <= 512 for g in row['annotator_image_ids']), 'REQUIRED_ANNOTATION_GROUP')

    def image(self, row):
        require(self.allowed.get(row['image_id']) == row, 'HOLDOUT_OR_FOREIGN_IMAGE')
        raw = read(self.dataset/'train/train_images'/row['file_name'], 16*1024**2, row['file_sha256'])
        require(len(raw) == row['bytes'], 'PHOTO_SIZE')
        with self.Image.open(io.BytesIO(raw)) as photo:
            require(photo.format == 'JPEG' and photo.mode == 'L' and photo.size == (2048, 2048), 'PHOTO_FORMAT')
            image = self.np.array(photo, dtype=self.np.uint8, copy=True)
        require(digest(image.tobytes()) == row['pixel_sha256'], 'PHOTO_PIXELS')
        self.check(); return image

    def training_pair(self, rng):
        # Separate uniform draws: extra annotators do not give an image extra weight.
        row = rng.choice(self.partitions['train'])
        group = rng.choice(row['annotator_image_ids'])
        image = self.image(row)
        encoded = []
        for annotation in self.by_group[group]:
            self.check()
            require(annotation.get('iscrowd', 0) == 0, 'CROWD_UNSUPPORTED')
            value = self.helper.polygon_rle(annotation['segmentation'], 2048, 2048)
            encoded.append({'size': [2048, 2048], 'counts': value.encode('ascii')})
        target = self.masks.decode(self.masks.merge(encoded)).astype(self.np.uint8)
        return image, target, row['image_id'], group

    def truth(self, rows):
        selected = [{**row, 'annotation_counts': {g: len(self.by_group[g]) for g in row['annotator_image_ids']}}
                    for row in rows]
        return self.helper.annotation_frame(self.annotations, selected, check=self.check), selected

    def prediction_rows(self, image_id, labels, count):
        require(image_id in self.allowed and 0 <= count <= 512, 'PREDICTION_SCOPE')
        rows = []
        for index in range(1, count+1):
            self.check()
            mask = self.np.asfortranarray(labels == index, dtype=self.np.uint8)
            rows.append({'filament_id': image_id + '_' + str(index).zfill(4),
                         'segmentation_rle': self.masks.encode(mask)['counts'].decode('ascii')})
        return rows
