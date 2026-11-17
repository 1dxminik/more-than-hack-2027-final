"""Small synthetic contracts: no real match files, labels or tuning."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import phase1
from scripts.make_synthetic import FILES, GROUND_TRUTH, TRACKING, generate, sha256, verify_bundle


class SubmissionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "fixture.csv"

    def write(self, text):
        self.csv.write_text(text, encoding="utf-8")

    def test_valid_labels_including_abstention(self):
        self.write("frame,label\n0,H\n1,A\n2,D\n3,X\n")
        self.assertEqual(phase1.validate_csv(self.csv, expected=4), 4)

    def test_invalid_contracts_are_rejected(self):
        cases = {
            "short": "frame,label\n0,H\n",
            "one_based": "frame,label\n1,H\n2,A\n",
            "duplicate": "frame,label\n0,H\n0,A\n",
            "reordered": "frame,label\n1,H\n0,A\n",
            "blank": "frame,label\n0,H\n1,\n",
            "unknown_label": "frame,label\n0,H\n1,Contested\n",
            "extra_index": "frame,label,index\n0,H,0\n1,A,1\n",
            "wrong_header": "time,label\n0,H\n1,A\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self.write(content)
                with self.assertRaises(ValueError):
                    phase1.validate_csv(self.csv, expected=2)

    def test_default_contract_requires_full_length(self):
        self.write("frame,label\n0,H\n1,A\n")
        with self.assertRaisesRegex(ValueError, "199319"):
            phase1.validate_csv(self.csv)


class SyntheticTests(unittest.TestCase):
    def test_generation_is_deterministic_and_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            one = generate(first, 3000)
            two = generate(second, 3000)
            self.assertEqual(one, two)
            self.assertEqual(generate(first, 3000), one)
            for relative in FILES:
                self.assertEqual(sha256(first / relative), sha256(second / relative))
            lines = (first / TRACKING).read_text().splitlines()
            self.assertEqual(len(lines), 3000)
            self.assertEqual([int(line.split(":", 1)[0]) for line in lines], list(range(1, 3001)))
            self.assertEqual(phase1.validate_csv(first / GROUND_TRUTH, expected=3000), 3000)
            self.assertTrue(all(one["labels"][label] > 0 for label in "HADX"))
            self.assertGreater(one["missing_ball_on_live_frames"], 0)

    def test_unrecognized_or_modified_input_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / GROUND_TRUTH).write_text("unrecognized")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                generate(root, 3000)
            self.assertEqual((root / GROUND_TRUTH).read_text(), "unrecognized")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate(root, 3000)
            (root / GROUND_TRUTH).write_text("modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_bundle(root)

    def test_existing_data_environment_is_overridden(self):
        with patch.dict(os.environ, {"MTAH_DATA_DIR": "DO_NOT_OPEN_THIS_DIRECTORY"}):
            env = phase1.synthetic_environment()
        self.assertEqual(Path(env["MTAH_DATA_DIR"]), phase1.ROOT / "data/synthetic")

    def test_path_cannot_escape_synthetic_root(self):
        from scripts.make_synthetic import local_path
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                local_path(Path(temporary), "../outside.csv")


class MetricSemanticsTests(unittest.TestCase):
    def test_abstention_preserves_spell_but_costs_frame_score(self):
        pm = phase1.metrics()
        gt = pm.label_codes(np.array(list("HHHHDDDDAAAA")))
        prediction = pm.label_codes(np.array(list("HXXHDDDDAAAA")))
        result = pm.evaluate(gt, prediction)
        self.assertEqual(result["spell_f1_mean"], 1.0)
        self.assertLess(result["macro_f1"], 1.0)

    def test_dead_ball_separates_same_team_possessions(self):
        pm = phase1.metrics()
        gt = pm.label_codes(np.array(list("HHDDHH")))
        result = pm.evaluate(gt, gt.copy())
        self.assertEqual(result["n_gt_spells"], 2)
        self.assertEqual(result["spell_f1_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
