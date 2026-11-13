"""Tests of the package modules. Run from the repository root:

    python -B -m unittest discover -s tests -t . -v

Each refusal has a test; removing the check makes that test fail (checked by the release tooling).
"""
import csv
import io
import re
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from solar_filament import ensemble, inference, postprocess, preprocess, submission  # noqa: E402
from tests import helpers  # noqa: E402


def two_blob_map():
    """10 x 12 map. Hand-made:
    A: (1,1)=210 seed, (1,2) (1,3) (2,1) (2,2) =100, and (3,3)=100 touching (2,2) only diagonally -> 6 px.
    B: (4,6) (4,7) (4,8) (4,9) (5,6) =100 and (5,7)=255 seed -> 6 px.
    C: (0,9)=255, one pixel, seeded but below the minimum area.
    D: rows 7-9, columns 0-2 =100, 9 px, no seed.
    """
    values = np.zeros((10, 12), np.uint8)
    for row, col, value in [(1, 1, 210), (1, 2, 100), (1, 3, 100), (2, 1, 100), (2, 2, 100), (3, 3, 100),
                            (4, 6, 100), (4, 7, 100), (4, 8, 100), (4, 9, 100), (5, 6, 100), (5, 7, 255),
                            (0, 9, 255)]:
        values[row, col] = value
    values[7:10, 0:3] = 100
    return values


class EnsembleTests(unittest.TestCase):
    def test_half_up_and_mean(self):
        a = np.array([[1, 10]], np.uint8)
        b = np.array([[2, 11]], np.uint8)
        # (1,2): S=3, n=2 -> (6+2)//4 = 2 (1.5 rounds up); (10,11): S=21 -> (42+2)//4 = 11 (10.5 rounds up)
        self.assertEqual(ensemble.ensemble_maps([a, b]).tolist(), [[2, 11]])
        c = np.array([[0, 255]], np.uint8)
        d = np.array([[1, 255]], np.uint8)
        e = np.array([[1, 254]], np.uint8)
        # (0,1,1): S=2, n=3 -> (4+3)//6 = 1 ; (255,255,254): S=764 -> (1528+3)//6 = 255
        self.assertEqual(ensemble.ensemble_maps([c, d, e]).tolist(), [[1, 255]])

    def test_single_map_unchanged_and_dtype(self):
        a = np.arange(6, dtype=np.uint8).reshape(2, 3)
        out = ensemble.ensemble_maps([a])
        self.assertTrue(np.array_equal(out, a))
        self.assertEqual(out.dtype, np.uint8)

    def test_refuses_empty_list(self):
        with self.assertRaisesRegex(ValueError, "NO_MAPS"):
            ensemble.ensemble_maps([])

    def test_refuses_shape_or_dtype_mismatch(self):
        with self.assertRaisesRegex(ValueError, "SHAPE"):
            ensemble.ensemble_maps([np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8)])
        with self.assertRaisesRegex(ValueError, "SHAPE"):
            ensemble.ensemble_maps([np.zeros((2, 2), np.float32)])


class CutsAndLabels(unittest.TestCase):
    """The rule's cuts and components; numpy and scipy only (r4: run without pycocotools)."""

    def test_cuts(self):
        self.assertEqual([postprocess.cut(t) for t in (0.15, 0.5, 0.8, 1.0)], [39, 128, 204, 255])

    def test_hysteresis_state_by_hand(self):
        labels, areas = postprocess.hysteresis_state(two_blob_map(), 0.15, 0.8)
        # scipy numbers components in row-major order of their first pixel: C=1, A=2, B=3, D=4
        self.assertEqual(areas.tolist(), [1, 6, 6, -1])
        self.assertEqual(int(labels[3, 3]), 2)       # the diagonal pixel joined A: 8-connectivity


def block_with_seed(block_value, seed_value, size=20):
    """r5: a size x size block of one value with one pixel of another in its middle, on a zero 30 x 30 map."""
    values = np.zeros((30, 30), np.uint8)
    values[5:5 + size, 5:5 + size] = block_value
    values[5 + size // 2, 5 + size // 2] = seed_value
    return values


class StubEncoder:
    """Stands in for pycocotools.mask in the area tests: the 'RLE' is the pixel count, so no package is needed."""

    @staticmethod
    def encode(mask):
        return {"counts": str(int(mask.sum())).encode("ascii")}


class RuleBoundaries(unittest.TestCase):
    """r5: each boundary of the rule with its smallest input (README: components of {map >= cut(T_low)} that contain a
    pixel >= cut(T_high) and have at least A pixels). Cuts for (0.15, 0.8): 39 and 204. numpy and scipy only."""

    def test_pixels_equal_to_the_low_cut_join_the_seeded_component(self):
        labels, areas = postprocess.hysteresis_state(block_with_seed(39, 204), 0.15, 0.8)
        self.assertEqual(areas.tolist(), [400])                  # the review's input: one component of area 400

    def test_pixels_one_unit_below_the_low_cut_do_not_join(self):
        labels, areas = postprocess.hysteresis_state(block_with_seed(38, 204), 0.15, 0.8)
        self.assertEqual(areas.tolist(), [1])                    # only the seed itself is above the low cut

    def test_a_pixel_equal_to_the_high_cut_seeds_a_component(self):
        labels, areas = postprocess.hysteresis_state(block_with_seed(100, 204), 0.15, 0.8)
        self.assertEqual(areas.tolist(), [400])

    def test_a_pixel_one_unit_below_the_high_cut_seeds_nothing(self):
        labels, areas = postprocess.hysteresis_state(block_with_seed(100, 203), 0.15, 0.8)
        self.assertEqual(areas.tolist(), [-1])                   # a component, but unseeded

    def test_a_component_of_exactly_the_minimum_area_is_kept(self):
        values = block_with_seed(39, 204)                         # one seeded component of 400 pixels
        with unittest.mock.patch.object(postprocess, "mask_module", lambda: StubEncoder):
            kept = postprocess.instance_rows(helpers.IMAGE_A, values, (0.15, 0.8, 400))
            dropped = postprocess.instance_rows(helpers.IMAGE_A, values, (0.15, 0.8, 401))
        self.assertEqual(kept, [(helpers.IMAGE_A + "_0001", "400")])
        self.assertEqual(dropped, [])

    def test_one_pixel_less_than_the_minimum_area_is_dropped(self):
        values = block_with_seed(39, 204)
        values[5, 5] = 0                                          # 399 pixels, still one seeded component
        with unittest.mock.patch.object(postprocess, "mask_module", lambda: StubEncoder):
            self.assertEqual(postprocess.instance_rows(helpers.IMAGE_A, values, (0.15, 0.8, 400)), [])
            self.assertEqual(postprocess.instance_rows(helpers.IMAGE_A, values, (0.15, 0.8, 399)),
                             [(helpers.IMAGE_A + "_0001", "399")])


@helpers.needs_pycocotools
class PostprocessTests(unittest.TestCase):

    def test_two_blob_rows_match_hand_counts(self):
        rows = postprocess.instance_rows("20150125172714Mh", two_blob_map(), (0.15, 0.8, 4))
        # COCO counts are column-major: index = column * 10 + row. Worked out by hand:
        #   A pixels at indices 11 12 21 22 31 33   -> zeros 11, ones 2, zeros 8, ones 2, zeros 8, ones 1, zeros 1, ones 1, zeros 86
        #   B pixels at indices 64 65 74 75 84 94   -> zeros 64, ones 2, zeros 8, ones 2, zeros 8, ones 1, zeros 9, ones 1, zeros 25
        counts_a = [11, 2, 8, 2, 8, 1, 1, 1, 86]
        counts_b = [64, 2, 8, 2, 8, 1, 9, 1, 25]
        self.assertEqual(sum(counts_a), 120)
        self.assertEqual(sum(counts_b), 120)
        expected = []
        for index, counts in ((1, counts_a), (2, counts_b)):
            from pycocotools import mask as mask_util
            rle = mask_util.frPyObjects({"size": [10, 12], "counts": counts}, 10, 12)
            expected.append(("20150125172714Mh_%04d" % index, rle["counts"].decode("ascii")))
        self.assertEqual(rows, expected)

    def test_two_blob_csv_text(self):
        rows = postprocess.instance_rows("20150125172714Mh", two_blob_map(), (0.15, 0.8, 4))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "sub.csv"
            submission.write_csv(out, rows)
            text = out.read_text()
        lines = text.split("\n")
        self.assertEqual(lines[0], "filament_id,segmentation_rle")
        self.assertEqual(lines[1].split(",")[0], "20150125172714Mh_0001")
        self.assertEqual(lines[2].split(",")[0], "20150125172714Mh_0002")
        self.assertEqual(lines[3], "")
        self.assertNotIn("\r", text)
        self.assertNotIn('"', text)

    def test_rle_decodes_to_hand_pixels(self):
        rows = postprocess.instance_rows("20150125172714Mh", two_blob_map(), (0.15, 0.8, 4))
        from pycocotools import mask as mask_util
        mask = mask_util.decode({"size": [10, 12], "counts": rows[0][1].encode("ascii")})
        self.assertEqual(sorted(zip(*np.nonzero(mask))), [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 3)])

    def test_minimum_area_and_seed_filters(self):
        values = two_blob_map()
        self.assertEqual(len(postprocess.instance_rows("20150125172714Mh", values, (0.15, 0.8, 1))), 3)   # C joins
        self.assertEqual(len(postprocess.instance_rows("20150125172714Mh", values, (0.15, 0.8, 7))), 0)
        # at T_high 0.99 the cut is 253: A's seed (210) is lost, B's (255) stays; C is below the area.
        rows = postprocess.instance_rows("20150125172714Mh", values, (0.15, 0.99, 4))
        self.assertEqual(len(rows), 1)

    def test_equal_cuts_are_allowed_for_the_plain_rule(self):
        # threshold 0.5 for both cuts, as in the plain-threshold entry: every pixel >= 128 is a seed
        rows = postprocess.instance_rows("20150125172714Mh", two_blob_map(), (0.5, 0.5, 1))
        # only the pixels >= 128 remain: (1,1)=210, (5,7)=255 and (0,9)=255, three single-pixel components
        self.assertEqual(len(rows), 3)

    def test_empty_map_gives_no_rows(self):
        self.assertEqual(postprocess.instance_rows("20150125172714Mh", np.zeros((10, 12), np.uint8), (0.15, 0.8, 4)), [])

    def test_rule_refused(self):
        values = two_blob_map()
        for rule in [(0.8, 0.15, 4), (0.0, 0.8, 4), (0.15, 1.5, 4), (0.15, 0.8, 0), (0.15, 0.8, 4.0)]:
            with self.assertRaisesRegex(ValueError, "RULE"):
                postprocess.instance_rows("20150125172714Mh", values, rule)

    def test_map_dtype_refused(self):
        with self.assertRaisesRegex(ValueError, "MAP_DTYPE"):
            postprocess.instance_rows("20150125172714Mh", two_blob_map().astype(np.float32), (0.15, 0.8, 4))
        with self.assertRaisesRegex(ValueError, "MAP_DTYPE"):
            postprocess.instance_rows("20150125172714Mh", np.zeros((2, 3, 4), np.uint8), (0.15, 0.8, 4))


class SubmissionTests(unittest.TestCase):
    def test_write_returns_file_hash_and_is_atomic(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "sub" / "x.csv"
            digest = submission.write_csv(out, [("20150125172714Mh_0001", "abc")])
            self.assertEqual(digest, hashlib.sha256(out.read_bytes()).hexdigest())
            self.assertEqual(sorted(p.name for p in out.parent.iterdir()), ["x.csv"])
            with out.open(newline="") as stream:
                self.assertEqual(list(csv.reader(stream)), [["filament_id", "segmentation_rle"],
                                                            ["20150125172714Mh_0001", "abc"]])

    def test_refuses_bad_id_and_bad_rle(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "FILAMENT_ID"):
                submission.write_csv(Path(folder) / "a.csv", [("bad_id_1", "abc")])
            with self.assertRaisesRegex(ValueError, "RLE"):
                submission.write_csv(Path(folder) / "a.csv", [("20150125172714Mh_0001", "")])
            with self.assertRaisesRegex(ValueError, "RLE"):
                submission.write_csv(Path(folder) / "a.csv", [("20150125172714Mh_0001", "é")])
            self.assertEqual(list(Path(folder).iterdir()), [])


class QuantiseTests(unittest.TestCase):
    def test_round_half_to_even_and_range(self):
        out = inference.quantise(np.array([[0.0, 0.5, 1.0, 0.002]], np.float32))
        self.assertEqual(out.tolist(), [[0, 128, 255, 1]])     # 127.5 -> 128 (even), 0.51 -> 1
        self.assertEqual(out.dtype, np.uint8)

    def test_refuses_nan_range_and_ndim(self):
        for bad in (np.array([[np.nan]]), np.array([[1.5]]), np.array([[-0.1]]), np.zeros(3)):
            with self.assertRaisesRegex(ValueError, "PROBABILITY"):
                inference.quantise(bad)


class PreprocessTests(unittest.TestCase):
    def test_load_roundtrip_and_format_refusals(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            good = helpers.write_jpeg(Path(folder) / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image())
            image = preprocess.load_halpha(good, size=(64, 64))
            self.assertEqual((image.shape, image.dtype), ((64, 64), np.uint8))
            with self.assertRaisesRegex(ValueError, "IMAGE_FORMAT"):
                preprocess.load_halpha(good)                    # default size is 2048 x 2048
            rgb = Path(folder) / "rgb.jpeg"
            Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(rgb, format="JPEG")
            with self.assertRaisesRegex(ValueError, "IMAGE_FORMAT:rgb.jpeg"):
                preprocess.load_halpha(rgb, size=(64, 64))
            png = Path(folder) / "x.png"
            Image.fromarray(helpers.synthetic_image()).save(png, format="PNG")
            with self.assertRaisesRegex(ValueError, "IMAGE_FORMAT"):
                preprocess.load_halpha(png, size=(64, 64))

    def test_list_images_sorted_and_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helpers.write_jpeg(root / (helpers.IMAGE_B + ".jpeg"), helpers.synthetic_image())
            helpers.write_jpeg(root / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image())
            (root / ".DS_Store").write_bytes(b"x")
            self.assertEqual([i for i, _ in preprocess.list_images(root)], [helpers.IMAGE_A, helpers.IMAGE_B])
            (root / "notes.txt").write_text("x")
            with self.assertRaisesRegex(ValueError, "STRAY_FILE:notes.txt"):
                preprocess.list_images(root)

    def test_os_metadata_is_ignored_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helpers.write_jpeg(root / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image())
            for name in (".DS_Store", "Thumbs.db", "thumbs.db", "desktop.ini", "Desktop.ini"):
                (root / name).write_bytes(b"x")
            (root / "__MACOSX").mkdir()
            (root / "__MACOSX" / ("._" + helpers.IMAGE_A + ".jpeg")).write_bytes(b"x")
            self.assertEqual([i for i, _ in preprocess.list_images(root)], [helpers.IMAGE_A])
            for stray in ("README", "x.JPEG", ".hidden", "Thumbs.db.bak"):
                (root / stray).write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "STRAY_FILE:" + stray.replace(".", "\\.") + r"\Z"):
                    preprocess.list_images(root)
                (root / stray).unlink()
            (root / "sub").mkdir()
            with self.assertRaisesRegex(ValueError, "STRAY_FILE:sub"):
                preprocess.list_images(root)

    def test_list_images_refusals_name_the_folder_or_file(self):
        # r5: no symbolic link needed, so this runs everywhere. The folder name holds regex metacharacters (as a
        # Windows temp path holds backslashes), so the expected text must go through re.escape to match.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "images (a+b) [1]"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "NO_IMAGES:" + re.escape(str(root)) + r"\Z"):
                preprocess.list_images(root)
            helpers.write_jpeg(root / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image())
            helpers.write_jpeg(root / "badname.jpeg", helpers.synthetic_image())
            with self.assertRaisesRegex(ValueError, r"STRAY_FILE:badname\.jpeg\Z"):
                preprocess.list_images(root)
            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "IMAGES_DIR:" + re.escape(str(missing)) + r"\Z"):
                preprocess.list_images(missing)

    def test_list_images_refuses_a_symlinked_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            real = helpers.write_jpeg(root / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image())
            helpers.symlink_or_skip(self, root / (helpers.IMAGE_B + ".jpeg"), real)
            with self.assertRaisesRegex(ValueError, "STRAY_FILE:" + helpers.IMAGE_B):
                preprocess.list_images(root)


class WeightsTests(unittest.TestCase):
    def test_load_model_roundtrip(self):
        from solar_filament.model import UNet
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "w.pt"
            digest = helpers.random_weights(path, seed=3)
            model = inference.load_model(path, digest)
            torch.manual_seed(3)
            expected = UNet().eval()
            probe = torch.rand(1, 1, 32, 32)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(probe), expected(probe)))
            self.assertFalse(model.training)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "w.pt"
            digest = helpers.random_weights(path)
            with self.assertRaisesRegex(ValueError, "WEIGHTS_SHA_MISMATCH"):
                inference.read_verified(path, "0" * 64)
            for bad in ("", "abc", digest.upper(), digest[:-1], None, 5):
                with self.assertRaisesRegex(ValueError, "WEIGHTS_SHA_FORMAT"):
                    inference.read_verified(path, bad)
            with self.assertRaisesRegex(ValueError, "WEIGHTS_PATH"):
                inference.read_verified(Path(folder) / "missing.pt", digest)
            link = Path(folder) / "link.pt"
            helpers.symlink_or_skip(self, link, path)
            with self.assertRaisesRegex(ValueError, "WEIGHTS_PATH"):
                inference.read_verified(link, digest)

    def test_probability_map_small_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "w.pt"
            digest = helpers.random_weights(path)
            model = inference.load_model(path, digest)
            image = helpers.synthetic_image((64, 64))
            probability = inference.probability_map(model, image, tile=32, stride=24)
            self.assertEqual((probability.shape, probability.dtype), ((64, 64), np.float32))
            self.assertTrue(0 <= probability.min() <= probability.max() <= 1)
            again = inference.probability_map(model, image, tile=32, stride=24)
            self.assertTrue(np.array_equal(probability, again))


class LineEnds(unittest.TestCase):
    """r4: every text file the repository code writes has LF line ends on every platform (Windows included)."""

    SOURCES = ["reproduce.py", "solar_filament", "tools", "training/train_standalone.py", "evaluation/organiser_cells.py",
               "tests/numeric_fixtures.py"]

    def calls(self):
        import ast
        for name in self.SOURCES:
            path = ROOT / name
            for file in (sorted(path.glob("*.py")) if path.is_dir() else [path]):
                for node in ast.walk(ast.parse(file.read_bytes())):
                    if isinstance(node, ast.Call):
                        yield file.relative_to(ROOT).as_posix(), node

    def test_no_text_mode_write_without_an_explicit_newline(self):
        import ast
        problems = []
        for where, node in self.calls():
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            keywords = {k.arg for k in node.keywords}
            if name == "write_text" and "newline" not in keywords:
                problems.append((where, node.lineno, "write_text without newline="))
            if name == "open" and "newline" not in keywords:
                modes = [a.value for a in node.args[:2] if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and re.fullmatch(r"[rwax][bt+]*", a.value)] + [
                    k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                if any(isinstance(m, str) and "w" in m and "b" not in m for m in modes):
                    problems.append((where, node.lineno, "text-mode open without newline="))
            if name == "to_csv" and "lineterminator" not in keywords:
                problems.append((where, node.lineno, "to_csv without lineterminator="))
        self.assertEqual(problems, [])

    def test_the_csv_writer_writes_line_feeds_only(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "x.csv"
            submission.write_csv(out, [("20150125172714Mh_0001", "abc"), ("20150125172714Mh_0002", "de")])
            self.assertEqual(out.read_bytes(),
                             b"filament_id,segmentation_rle\n20150125172714Mh_0001,abc\n20150125172714Mh_0002,de\n")


if __name__ == "__main__":
    unittest.main()
