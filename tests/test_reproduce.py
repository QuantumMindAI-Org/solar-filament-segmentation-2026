"""Tests of reproduce.py: candidate validation, stages, the command line, determinism.

The end-to-end test runs the real tiled inference (512 tiles) with a randomly initialised U-Net of the
package's own model class on one synthetic 2048 x 2048 image. The command-line exit-code tests replace
the model call with a stub so they stay fast; they test the control flow, not the network.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import reproduce  # noqa: E402
from solar_filament import inference  # noqa: E402
from tests import helpers  # noqa: E402

SHA = "a" * 64


def make_candidate(folder, **changes):
    """candidate.json with one throwaway weights file in folder/weights; returns (path, payload)."""
    folder = Path(folder)
    digest = helpers.random_weights(folder / "weights" / "t.pt")
    path = helpers.write_candidate(folder, [("t", "weights/t.pt", digest)])
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload))
    return path, payload


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path, self.payload = make_candidate(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, payload):
        self.path.write_text(json.dumps(payload))
        return self.path

    def test_valid_candidate_and_rule(self):
        candidate = reproduce.read_candidate(self.path)
        self.assertEqual(reproduce.rule_of(candidate), (0.15, 0.8, 4))

    def test_the_shipped_candidate_is_valid(self):
        candidate = reproduce.read_candidate(ROOT / "candidate.json")
        self.assertTrue(1 <= len(candidate["models"]) <= 3)

    def refused(self, code, **changes):
        payload = dict(self.payload, **changes)
        with self.assertRaisesRegex(ValueError, code):
            reproduce.read_candidate(self.write(payload))

    def test_refusals(self):
        self.refused("CANDIDATE_SCHEMA", schema="other")
        self.refused("CANDIDATE_MODELS", models=[])
        self.refused("CANDIDATE_MODELS", models=[{"name": "a", "file": "w", "sha256": SHA}] * 4)
        self.refused("CANDIDATE_MODELS", models=[{"name": "a", "file": "w", "sha256": "short"}])
        self.refused("CANDIDATE_DUPLICATE_MODEL", models=[{"name": "a", "file": "w1", "sha256": SHA},
                                                        {"name": "a", "file": "w2", "sha256": SHA}])
        self.refused("CANDIDATE_WEIGHTS_PATH", models=[{"name": "a", "file": "../w.pt", "sha256": SHA}])
        self.refused("CANDIDATE_WEIGHTS_PATH", models=[{"name": "a", "file": "/etc/w.pt", "sha256": SHA}])
        self.refused("CANDIDATE_ENSEMBLE", ensemble="median")
        self.refused("CANDIDATE_RULE", rule={"T_low": 0.8, "T_high": 0.15, "A": 4})
        self.refused("CANDIDATE_RULE", rule={"T_low": 0.15, "T_high": 0.8, "A": 0})
        self.refused("CANDIDATE_RULE", rule={"T_low": 0.15, "T_high": 0.8, "A": True})
        self.refused("CANDIDATE_RULE", rule={"T_low": 0.15, "T_high": 0.8})
        self.refused("CANDIDATE_EXPECTED_SHA", expected_csv_sha256="nothex")

    def test_status_is_read(self):
        # r4: FINAL needs an expected hash; an unknown status is refused; a rehearsal status runs as documented
        self.refused("CANDIDATE_FINAL_WITHOUT_EXPECTED_SHA", status="FINAL", expected_csv_sha256=None)
        for bad in ("WHATEVER", "final", "REHEARSAL_", "TEST", None, 5):
            self.refused("CANDIDATE_STATUS", status=bad)
        payload = dict(self.payload, status="FINAL", expected_csv_sha256=SHA)
        self.assertEqual(reproduce.read_candidate(self.write(payload))["status"], "FINAL")
        for good in ("REHEARSAL_K2", "REHEARSAL_TEST"):
            candidate = reproduce.read_candidate(self.write(dict(self.payload, status=good, expected_csv_sha256=None)))
            self.assertEqual(reproduce.check_expected(SHA, candidate), "NO_EXPECTED_VALUE")
            candidate = reproduce.read_candidate(self.write(dict(self.payload, status=good, expected_csv_sha256=SHA)))
            self.assertEqual(reproduce.check_expected("b" * 64, candidate), "MISMATCH")
        shipped = reproduce.read_candidate(ROOT / "candidate.json")              # the final build ships the FINAL candidate
        self.assertEqual(shipped["status"], "FINAL")
        self.assertEqual(hashlib.sha256((ROOT / shipped["submitted_csv"]).read_bytes()).hexdigest(), shipped["expected_csv_sha256"])

    def test_set_threads_reports_the_count(self):
        import torch
        before = torch.get_num_threads()
        try:
            self.assertEqual(reproduce.set_threads(3), 3)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(reproduce.set_threads(None), 3)
            with self.assertRaisesRegex(ValueError, "THREADS"):
                reproduce.set_threads(0)
        finally:
            torch.set_num_threads(before)

    def test_check_expected_states(self):
        candidate = reproduce.read_candidate(self.path)
        self.assertEqual(reproduce.check_expected(SHA, candidate), "NO_EXPECTED_VALUE")
        candidate["expected_csv_sha256"] = SHA
        self.assertEqual(reproduce.check_expected(SHA, candidate), "MATCH")
        self.assertEqual(reproduce.check_expected("b" * 64, candidate), "MISMATCH")
        self.assertEqual(reproduce.check_expected(SHA, candidate, partial=True), "SKIPPED_PARTIAL")

    def test_device_refusals(self):
        with self.assertRaisesRegex(ValueError, "DEVICE"):
            reproduce.resolve_device("tpu")
        self.assertEqual(reproduce.resolve_device("cpu"), "cpu")
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA_UNAVAILABLE"):
                reproduce.resolve_device("cuda")
            self.assertEqual(reproduce.resolve_device("auto"), "cpu")

    def test_threads_refused(self):
        for bad in (0, 65, "4", True):
            with self.assertRaisesRegex(ValueError, "THREADS"):
                reproduce.reproduce(self.folder, self.folder / "o.csv", self.path, threads=bad)

    def test_wrong_weights_hash_stops_before_inference(self):
        payload = dict(self.payload)
        payload["models"] = [dict(payload["models"][0], sha256="0" * 64)]
        with self.assertRaisesRegex(ValueError, "WEIGHTS_SHA_MISMATCH"):
            reproduce.load_models(reproduce.read_candidate(self.write(payload)), self.folder)


class StageTests(unittest.TestCase):
    def test_ensemble_stage_equals_package_ensemble(self):
        a = np.array([[1, 10]], np.uint8)
        b = np.array([[2, 11]], np.uint8)
        self.assertEqual(reproduce.ensemble([a, b]).tolist(), [[2, 11]])

    @helpers.needs_pycocotools
    def test_find_instances_stage(self):
        values = np.zeros((10, 12), np.uint8)
        values[2:5, 2:6] = 255
        rows = reproduce.find_instances(helpers.IMAGE_A, values, (0.15, 0.8, 4))
        self.assertEqual([r[0] for r in rows], [helpers.IMAGE_A + "_0001"])


@helpers.needs_pycocotools
class CommandLineTests(unittest.TestCase):
    """Control flow of main(): exit 0, 4 on a hash mismatch, 2 on a refusal. The network call is stubbed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path, self.payload = make_candidate(self.folder)
        self.images = self.folder / "images"
        self.images.mkdir()
        for name in (helpers.IMAGE_A, helpers.IMAGE_B):
            helpers.write_jpeg(self.images / (name + ".jpeg"), helpers.synthetic_image((2048, 2048)))
        self.stub = mock.patch.object(inference, "probability_map", side_effect=self.fake_map)
        self.stub.start()

    def tearDown(self):
        self.stub.stop()
        self.temp.cleanup()

    @staticmethod
    def fake_map(model, image, device="cpu", **kwargs):
        return (image.astype(np.float32) / 255.0)

    def run_main(self, *extra):
        out = self.folder / "out.csv"
        argv = ["--images-dir", str(self.images), "--out", str(out), "--candidate", str(self.path)] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = reproduce.main(argv)
        return code, json.loads(captured.getvalue()), out

    def test_threads_are_reported_in_the_json_line(self):
        import torch
        before = torch.get_num_threads()
        try:
            _code, summary, _out = self.run_main("--threads", "2")
            self.assertEqual(summary["threads"], 2)
        finally:
            torch.set_num_threads(before)

    def test_csv_bytes_end_lines_with_a_line_feed(self):
        _code, _summary, out = self.run_main()
        raw = out.read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"\n"))

    def test_exit_zero_without_expected_value(self):
        code, summary, out = self.run_main()
        self.assertEqual((code, summary["comparison"], summary["images"]), (0, "NO_EXPECTED_VALUE", 2))
        self.assertEqual(summary["csv_sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(out.read_text().split("\n")[0], "filament_id,segmentation_rle")

    def test_match_then_mismatch_exit_codes(self):
        _code, summary, _out = self.run_main()
        self.payload["expected_csv_sha256"] = summary["csv_sha256"]
        self.path.write_text(json.dumps(self.payload))
        code, summary2, _ = self.run_main()
        self.assertEqual((code, summary2["comparison"]), (0, "MATCH"))
        self.payload["expected_csv_sha256"] = "0" * 64
        self.path.write_text(json.dumps(self.payload))
        code, summary3, out = self.run_main()
        self.assertEqual((code, summary3["comparison"]), (4, "MISMATCH"))
        self.assertTrue(out.exists())        # the file is still written; the exit code reports the mismatch

    def test_limit_skips_the_comparison_and_save_maps_writes_npz(self):
        maps = self.folder / "maps"
        code, summary, _ = self.run_main("--limit", "1", "--save-maps", str(maps))
        self.assertEqual((code, summary["images"], summary["comparison"]), (0, 1, "SKIPPED_PARTIAL"))
        self.assertEqual([p.name for p in maps.iterdir()], [helpers.IMAGE_A + ".npz"])
        stored = np.load(maps / (helpers.IMAGE_A + ".npz"))["p"]
        self.assertEqual((stored.shape, stored.dtype), ((2048, 2048), np.uint8))

    def test_refusal_exit_code_through_module_entry(self):
        (self.images / "stray.txt").write_text("x")
        with self.assertRaisesRegex(ValueError, "STRAY_FILE"):
            self.run_main()


def rectangle_rle(top, bottom, left, right):
    from pycocotools import mask as mask_util
    mask = np.zeros((2048, 2048), np.uint8)
    mask[top:bottom, left:right] = 1
    return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def write_rows(path, rows):
    Path(path).write_bytes(("filament_id,segmentation_rle\n" + "".join("%s,%s\n" % row for row in rows)).encode("ascii"))
    return Path(path)


@helpers.needs_pycocotools
class SubmittedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.a = rectangle_rle(10, 110, 10, 60)
        self.b = rectangle_rle(500, 600, 500, 700)
        self.c = rectangle_rle(900, 950, 900, 990)
        self.submitted = write_rows(self.folder / "sub.csv", [(helpers.IMAGE_A + "_0001", self.a),
                                                              (helpers.IMAGE_A + "_0002", self.b),
                                                              (helpers.IMAGE_B + "_0001", self.c)])

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_csv_has_no_difference_and_pq_one(self):
        ours = write_rows(self.folder / "ours.csv", [(helpers.IMAGE_A + "_0001", self.a), (helpers.IMAGE_A + "_0002", self.b),
                                                     (helpers.IMAGE_B + "_0001", self.c)])
        report = reproduce.compare_instances(ours, self.submitted)
        self.assertEqual((report["images"], report["images_identical"], report["instances_only_in_ours"],
                          report["instances_only_in_submitted"]), (2, 2, 0, 0))
        self.assertAlmostEqual(report["organiser_pq_ours_against_submitted"]["pq"], 1.0, places=9)

    def test_renumbered_but_identical_instances_are_not_a_difference(self):
        ours = write_rows(self.folder / "ours.csv", [(helpers.IMAGE_A + "_0001", self.b), (helpers.IMAGE_A + "_0002", self.a),
                                                     (helpers.IMAGE_B + "_0001", self.c)])
        self.assertEqual(reproduce.compare_instances(ours, self.submitted, with_pq=False)["images_identical"], 2)

    def test_differences_counted_by_hand(self):
        moved = rectangle_rle(500, 600, 900, 1100)          # B moved: no overlap with the submitted B
        ours = write_rows(self.folder / "ours.csv", [(helpers.IMAGE_A + "_0001", self.a), (helpers.IMAGE_A + "_0002", moved)])
        report = reproduce.compare_instances(ours, self.submitted)
        self.assertEqual(report["images_identical"], 0)
        self.assertEqual(report["images_that_differ"], {helpers.IMAGE_A: {"only_in_ours": 1, "only_in_submitted": 1},
                                                        helpers.IMAGE_B: {"only_in_ours": 0, "only_in_submitted": 1}})
        self.assertEqual((report["instances_ours"], report["instances_submitted"]), (2, 3))
        # submitted as reference: A matches (IoU 1), B and C are missed (FN 2), the moved prediction is a FP: 1 / (1 + 0.5 + 1)
        self.assertAlmostEqual(report["organiser_pq_ours_against_submitted"]["pq"], 0.4, places=9)
        lines = reproduce.describe_difference(report)
        self.assertIn("THE CSV DIFFERS", lines[0])
        self.assertIn("0 of 2 images have identical instances; 1 instances occur only in our CSV, 2 only in the submitted one.", lines[1])
        self.assertIn("0.400000", lines[2])

    def test_pq_is_reported_as_not_computed_without_pandas(self):
        with mock.patch.dict(sys.modules, {"pandas": None}):
            result = reproduce.organiser_pq(self.submitted, self.submitted)
        self.assertIsNone(result["pq"])
        self.assertIn("not computed", result["reason"])

    def test_pq_is_not_computed_for_an_empty_reference(self):
        empty = write_rows(self.folder / "empty.csv", [])
        result = reproduce.organiser_pq(self.submitted, empty)
        self.assertIsNone(result["pq"])
        self.assertIn("no filaments", result["reason"])

    def test_header_refused(self):
        bad = self.folder / "bad.csv"
        bad.write_text("id,rle\n")
        with self.assertRaisesRegex(ValueError, "CSV_HEADER"):
            reproduce.read_instances(bad)

    def test_submitted_csv_field_and_hash_check(self):
        path, payload = make_candidate(self.folder / "c")
        for bad in ("/abs.csv", "../x.csv", 5):
            changed = dict(payload, submitted_csv=bad, expected_csv_sha256=SHA)
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "CANDIDATE_SUBMITTED_CSV"):
                reproduce.read_candidate(path)
        path.write_text(json.dumps(dict(payload, submitted_csv="sub.csv", expected_csv_sha256=None)))
        with self.assertRaisesRegex(ValueError, "CANDIDATE_SUBMITTED_CSV"):
            reproduce.read_candidate(path)
        right = hashlib.sha256(self.submitted.read_bytes()).hexdigest()
        shutil.copy(self.submitted, path.parent / "sub.csv")
        path.write_text(json.dumps(dict(payload, submitted_csv="sub.csv", expected_csv_sha256=right)))
        candidate = reproduce.read_candidate(path)
        self.assertEqual(reproduce.submitted_path(candidate, path), path.parent / "sub.csv")
        candidate["expected_csv_sha256"] = SHA
        with self.assertRaisesRegex(ValueError, "SUBMITTED_CSV_HASH"):
            reproduce.submitted_path(candidate, path)
        self.assertIsNone(reproduce.submitted_path(dict(payload), path))


@helpers.needs_pycocotools
class MismatchReportTests(unittest.TestCase):
    """The command-line behaviour when the CSV hash differs: say so, report the instance difference, exit 4."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path, self.payload = make_candidate(self.folder)
        self.images = self.folder / "images"
        self.images.mkdir()
        helpers.write_jpeg(self.images / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image((2048, 2048)))
        self.stub = mock.patch.object(inference, "probability_map",
                                      side_effect=lambda model, image, device="cpu", **k: image.astype(np.float32) / 255.0)
        self.stub.start()

    def tearDown(self):
        self.stub.stop()
        self.temp.cleanup()

    def run_main(self, *extra):
        out = self.folder / "out.csv"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()) as captured, contextlib.redirect_stderr(err):
            code = reproduce.main(["--images-dir", str(self.images), "--out", str(out), "--candidate", str(self.path)] + list(extra))
        return code, json.loads(captured.getvalue()), err.getvalue()

    def test_mismatch_without_a_submitted_csv_says_so_plainly(self):
        self.payload["expected_csv_sha256"] = "0" * 64
        self.path.write_text(json.dumps(self.payload))
        code, summary, err = self.run_main()
        self.assertEqual(code, 4)
        self.assertIn("THE CSV DIFFERS FROM THE SUBMITTED ONE", summary["message"])
        self.assertIn("THE CSV DIFFERS", err)
        self.assertIn("--compare-with", err)
        self.assertNotIn("difference", summary)

    def test_mismatch_with_the_shipped_submitted_csv_reports_instances(self):
        _code, first, _ = self.run_main()
        shipped = self.folder / "submitted.csv"
        lines = (self.folder / "out.csv").read_text().split("\n")
        shipped.write_text("\n".join(lines[:-2]) + "\n")           # drop the last instance: the "submitted" file differs
        self.payload.update(submitted_csv="submitted.csv", expected_csv_sha256=hashlib.sha256(shipped.read_bytes()).hexdigest())
        self.path.write_text(json.dumps(self.payload))
        code, summary, err = self.run_main()
        self.assertEqual(code, 4)
        self.assertEqual(summary["comparison"], "MISMATCH")
        self.assertEqual(summary["difference"]["instances_only_in_ours"], 1)
        self.assertEqual(summary["difference"]["instances_only_in_submitted"], 0)
        self.assertIn("1 instances occur only in our CSV", err)

    def test_compare_with_flag_reports_even_without_expected_value(self):
        _code, first, _ = self.run_main()
        same = self.folder / "same.csv"
        shutil.copy(self.folder / "out.csv", same)
        code, summary, err = self.run_main("--compare-with", str(same))
        self.assertEqual((code, summary["comparison"]), (0, "NO_EXPECTED_VALUE"))
        self.assertEqual(summary["difference"]["images_identical"], summary["difference"]["images"])
        self.assertEqual(err, "")                                       # identical: nothing to warn about
        other = write_rows(self.folder / "other.csv", [(helpers.IMAGE_A + "_0001", rectangle_rle(0, 5, 0, 5))])
        code, summary, err = self.run_main("--compare-with", str(other))
        self.assertEqual(code, 0)
        self.assertIn("THE CSV DIFFERS", err)

    def test_limited_run_does_not_compare(self):
        other = write_rows(self.folder / "other.csv", [(helpers.IMAGE_A + "_0001", rectangle_rle(0, 5, 0, 5))])
        _code, summary, err = self.run_main("--limit", "1", "--compare-with", str(other))
        self.assertNotIn("difference", summary)
        self.assertEqual(err, "")


class ScriptEntryTests(unittest.TestCase):
    @helpers.needs_pycocotools
    def test_refusal_through_the_script_exits_2_with_a_message(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty"
            empty.mkdir()
            result = subprocess.run([sys.executable, "-B", str(ROOT / "reproduce.py"), "--images-dir", str(empty),
                                     "--out", str(Path(temp) / "o.csv")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("REFUSED NO_IMAGES", result.stderr)
            self.assertFalse((Path(temp) / "o.csv").exists())

    def test_help_runs_and_documents_the_options(self):
        import subprocess
        result = subprocess.run([sys.executable, "-B", str(ROOT / "reproduce.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for option in ("--images-dir", "--out", "--candidate", "--device", "--save-maps", "--limit"):
            self.assertIn(option, result.stdout)


class WithoutPycocotools(unittest.TestCase):
    """r4: the optional RLE encoder is imported only where an RLE is written (child processes with it blocked)."""

    def run_blocked(self, args, cwd=None):
        import subprocess
        code = helpers.BLOCK_PYCOCOTOOLS + "import runpy\nsys.argv = %r\nrunpy.run_path(%r, run_name='__main__')\n" % (
            ["reproduce.py"] + list(args), str(ROOT / "reproduce.py"))
        return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300, cwd=cwd)

    def test_help_and_argument_errors_work_without_it(self):
        result = self.run_blocked(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--threads", result.stdout)
        self.assertNotIn("Traceback", result.stderr)
        result = self.run_blocked(["--images-dir", "x", "--out", "y", "--threads", "four"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid int value", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_run_stops_with_one_line_naming_the_package_and_the_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path, _payload = make_candidate(folder)
            images = folder / "images"
            images.mkdir()
            helpers.write_jpeg(images / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image((64, 64)))
            result = self.run_blocked(["--images-dir", str(images), "--out", str(folder / "o.csv"),
                                       "--candidate", str(path), "--threads", "1"])
            self.assertEqual(result.returncode, 2)
            lines = result.stderr.strip().splitlines()
            self.assertEqual(len(lines), 1, result.stderr)
            self.assertTrue(lines[0].startswith("REFUSED MISSING_PACKAGE:pycocotools"), lines[0])
            self.assertIn("requirements.txt", lines[0])
            self.assertFalse((folder / "o.csv").exists())

    def test_the_package_imports_and_the_map_stages_run_without_it(self):
        import subprocess
        code = helpers.BLOCK_PYCOCOTOOLS + (
            "sys.path.insert(0, %r)\n"
            "import numpy as np\n"
            "import reproduce\n"
            "from solar_filament import postprocess, inference, ensemble\n"
            "assert postprocess.cut(0.15) == 39\n"
            "try:\n"
            "    postprocess.instance_rows('20150125172714Mh', np.zeros((8, 8), np.uint8), (0.15, 0.8, 4))\n"
            "except ValueError as error:\n"
            "    print(str(error))\n") % str(ROOT)
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("MISSING_PACKAGE:pycocotools"), result.stdout)


@helpers.needs_pycocotools
class EndToEndTests(unittest.TestCase):
    def test_real_tiled_inference_is_deterministic_and_well_formed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path, _payload = make_candidate(folder)
            images = folder / "images"
            images.mkdir()
            helpers.write_jpeg(images / (helpers.IMAGE_A + ".jpeg"), helpers.synthetic_image((2048, 2048)))
            first = reproduce.reproduce(images, folder / "a.csv", path, threads=2)
            second = reproduce.reproduce(images, folder / "b.csv", path, threads=2)
            self.assertEqual(first["csv_sha256"], second["csv_sha256"])
            lines = (folder / "a.csv").read_text().split("\n")
            self.assertEqual(lines[0], "filament_id,segmentation_rle")
            self.assertEqual(first["rows"], len(lines) - 2)
            self.assertEqual(first["comparison"], "NO_EXPECTED_VALUE")
            self.assertEqual(first["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
