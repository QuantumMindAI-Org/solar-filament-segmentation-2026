"""Numeric pins of the inference path, with numpy and torch only (no optional package, no competition data).

r4, after an executed read on a second machine showed that the earlier tests pinned no numerics: each test below
fixes one stage exactly or within a stated tolerance, so that a one-line change of a constant or an operation
fails a test. The release tooling plants such changes one at a time (mutation_check.py, numeric section) and
requires each to turn a test red with pycocotools blocked.

Tolerances: the stage tests on fixed small arrays are exact. The pinned forward pass of the network with weights
built from a formula is compared with a stored reference within 1e-4 x the largest absolute value (other CPUs and
library versions sum floating-point numbers in another order). The end-to-end 8-bit map of the shipped weights on a
synthetic 1024 x 1024 image may differ from the stored map by at most 1 unit in at most 16 of its 1,048,576 pixels
(an independent read measured one pixel by one unit between thread counts on a 2048 x 2048 image).
"""
import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from solar_filament import ensemble, inference, model  # noqa: E402
from tests import numeric_fixtures as fixtures  # noqa: E402

END_TO_END_MAX_UNITS = 1
END_TO_END_MAX_PIXELS = 16


class TilingConstants(unittest.TestCase):
    def test_tile_and_stride(self):
        self.assertEqual((model.TILE, model.STRIDE), (512, 384))
        for function in (model.tiled_probability, inference.probability_map):
            parameters = inspect.signature(function).parameters
            self.assertEqual((parameters["tile"].default, parameters["stride"].default), (512, 384), function.__name__)

    def test_tile_positions(self):
        self.assertEqual(model.positions(2048), [0, 384, 768, 1152, 1536])
        self.assertEqual(model.positions(1024), [0, 384, 512])
        self.assertEqual(model.positions(1000), [0, 384, 488])       # the last tile is moved to the border
        self.assertEqual(model.positions(512), [0])
        self.assertEqual(model.positions(40, 16, 8), [0, 8, 16, 24])


class Offset(torch.nn.Module):
    """Stand-in network: logits = 3 * input + (a different constant for every tile), so overlaps disagree."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return 3.0 * value + (0.37 * self.calls - 1.1)


def reference_average(image, tile, stride):
    """The tiled average written out independently: float32 crops / 255, float64 sums and counts."""
    starts = model.positions(image.shape[0], tile, stride)
    total = np.zeros(image.shape, np.float64)
    count = np.zeros(image.shape, np.float64)
    calls = 0
    for y in starts:
        for x in starts:
            calls += 1
            crop = image[y:y + tile, x:x + tile].astype(np.float32) / np.float32(255.0)
            logits = torch.from_numpy(3.0 * crop) + np.float32(0.37 * calls - 1.1)
            total[y:y + tile, x:x + tile] += logits.float().sigmoid().numpy()
            count[y:y + tile, x:x + tile] += 1
    return (total / count).astype(np.float32)


class TiledAverage(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        self.image = rng.randint(0, 256, size=(40, 40)).astype(np.uint8)

    def test_division_by_255_and_overlap_average_exactly(self):
        got = model.tiled_probability(Offset(), self.image, device="cpu", tile=16, stride=8)
        want = reference_average(self.image, 16, 8)
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.array_equal(got, want), "max difference %g" % np.abs(got - want).max())

    def test_a_single_tile_is_the_sigmoid_of_the_scaled_crop(self):
        image = np.full((16, 16), 51, np.uint8)               # 51 / 255 = 0.2
        got = model.tiled_probability(Offset(), image, device="cpu", tile=16, stride=16)
        want = torch.tensor(3.0 * np.float32(51) / np.float32(255.0) + np.float32(0.37 - 1.1)).float().sigmoid().item()
        self.assertTrue(np.all(got == np.float32(want)))


class Quantiser(unittest.TestCase):
    def test_exact_halves_round_to_even(self):
        # (k + 0.5) / 255 times 255 is exactly k + 0.5 in float64 for these k; numpy.rint rounds half to even
        ks = np.array([0, 1, 2, 3, 126, 127, 253, 254])
        got = inference.quantise(((ks + 0.5) / 255.0).reshape(1, -1))
        self.assertEqual(got.tolist(), [[0, 2, 2, 4, 126, 128, 254, 254]])

    def test_float64_arithmetic_on_float32_input(self):
        # float32 probabilities whose 255 * p lands on the other side of a half in float32 arithmetic
        bits = np.array([1053148614, 1058412054, 1049990550, 1060385844, 1063543908, 1064201838], np.uint32)
        got = inference.quantise(bits.view(np.float32).reshape(1, -1))
        self.assertEqual(got.tolist(), [[99, 149, 75, 179, 227, 237]])

    def test_ends_and_type(self):
        got = inference.quantise(np.array([[0.0, 1.0, 0.0019, 0.9981]]))
        self.assertEqual(got.tolist(), [[0, 255, 0, 255]])
        self.assertEqual(got.dtype, np.uint8)


class Ensemble(unittest.TestCase):
    def test_integer_mean_half_up_for_one_two_three_maps(self):
        a = np.array([[0, 1, 10, 255, 7]], np.uint8)
        b = np.array([[1, 2, 11, 254, 7]], np.uint8)
        c = np.array([[1, 2, 11, 254, 9]], np.uint8)
        self.assertEqual(ensemble.ensemble_maps([a]).tolist(), a.tolist())
        # n = 2: means 0.5, 1.5, 10.5, 254.5, 7 -> 1, 2, 11, 255, 7 (half up)
        self.assertEqual(ensemble.ensemble_maps([a, b]).tolist(), [[1, 2, 11, 255, 7]])
        # n = 3: means 2/3, 5/3, 32/3, 763/3, 23/3 -> 1, 2, 11, 254, 8 (nearest)
        self.assertEqual(ensemble.ensemble_maps([a, b, c]).tolist(), [[1, 2, 11, 254, 8]])
        d = np.array([[0, 0, 0, 0, 0]], np.uint8)
        e = np.array([[0, 1, 0, 0, 0]], np.uint8)
        f = np.array([[2, 1, 0, 0, 0]], np.uint8)
        # n = 3, means 2/3 and 2/3 -> 1; a mean of exactly k + 1/2 cannot occur for n = 3
        self.assertEqual(ensemble.ensemble_maps([d, e, f]).tolist(), [[1, 1, 0, 0, 0]])


class WeightsHash(unittest.TestCase):
    def test_the_shipped_weights_are_hash_checked(self):
        candidate = json.loads((ROOT / "candidate.json").read_bytes())
        item = candidate["models"][0]
        self.assertEqual(len(inference.read_verified(ROOT / item["file"], item["sha256"])), (ROOT / item["file"]).stat().st_size)
        with self.assertRaisesRegex(ValueError, "WEIGHTS_SHA_MISMATCH"):
            inference.read_verified(ROOT / item["file"], "0" * 64)


class ForwardPass(unittest.TestCase):
    def test_fixed_weights_on_a_fixed_input_match_the_reference(self):
        reference = json.loads(fixtures.FORWARD_REFERENCE.read_bytes())
        want = np.array(reference["values"], np.float64).reshape(reference["shape"])
        got = fixtures.forward_output()
        self.assertEqual(got.shape, want.shape)
        tolerance = 1e-4 * np.abs(want).max()
        self.assertLess(np.abs(got - want).max(), tolerance)
        self.assertGreater(np.abs(want).max(), 0.5)              # the reference is not a flat output
        self.assertGreater(want.std(), 0.05)


class EndToEndMap(unittest.TestCase):
    def test_shipped_weights_on_a_synthetic_image(self):
        want = np.load(fixtures.MAP_REFERENCE)["p"]
        self.assertEqual((want.shape, want.dtype), ((1024, 1024), np.uint8))
        self.assertGreater(int((want >= 39).sum()), 1000)           # the map has content at the rule's cuts
        got = fixtures.end_to_end_map()
        difference = np.abs(got.astype(np.int16) - want.astype(np.int16))
        self.assertLessEqual(int(difference.max()), END_TO_END_MAX_UNITS)
        self.assertLessEqual(int((difference > 0).sum()), END_TO_END_MAX_PIXELS)


if __name__ == "__main__":
    unittest.main()
