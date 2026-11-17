"""Behavioral checks on invented trajectories with the actual frozen model assets."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.install_assets import install
from scripts.predict_c5 import export, validate_csv
from scripts.synthetic import generate, observations
from solution import assets
from solution.runtime import Runtime
from solution.tracking import load_tracking
from solution.n14.trace_support import SINK


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime()

    def test_finite_inference_with_missing_ball_fragmented_tracks_and_outage(self):
        players, ball, teams = observations(2200)
        labels, details = self.runtime.predict_block(players, ball, teams, diagnostics=True)
        self.assertEqual(labels.shape, (2200,))
        self.assertTrue(np.isin(labels, list("HADX")).all())
        self.assertTrue(np.isfinite(details["probability"]).all())
        np.testing.assert_allclose(details["probability"].sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(details["meta_probability"].sum(1), 1, atol=1e-6)
        self.assertIsNone(SINK.get())

    def test_repeated_calls_do_not_keep_previous_half_state(self):
        p, b, t = observations(320)
        first = self.runtime.predict_block(p, b, t)
        self.runtime.predict_block(np.full_like(p, np.nan), np.full_like(b, np.nan), t)
        second = self.runtime.predict_block(p, b, t)
        np.testing.assert_array_equal(first, second)

    def test_periods_reset_and_outside_frames_remain_X(self):
        p, b, t = observations(660)
        labels = self.runtime.predict_match(p, b, t, [(10, 310), (350, 650)])
        np.testing.assert_array_equal(labels[10:310], self.runtime.predict_block(p[10:310], b[10:310], t))
        np.testing.assert_array_equal(labels[350:650], self.runtime.predict_block(p[350:650], b[350:650], t))
        for section in (labels[:10], labels[310:350], labels[650:]):
            self.assertTrue((section == "X").all())

    def test_invalid_periods_and_infinity_are_rejected(self):
        p, b, t = observations(320)
        for periods in ([], [(0, 200), (199, 320)], [(0, 321)], [(0, 1)]):
            with self.assertRaises(ValueError):
                self.runtime.predict_match(p, b, t, periods)
        b[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.runtime.predict_block(p, b, t)


class AssetTests(unittest.TestCase):
    def test_changed_frozen_decoding_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = json.loads(assets.CONFIG.read_text(encoding="utf-8"))
            config["variant"]["min_D_len"] += 1
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(assets, "CONFIG", path), self.assertRaises(ValueError):
                assets.configuration()

    def test_missing_additional_and_changed_asset_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights"
            shutil.copytree(assets.DEFAULT_WEIGHTS, target)
            assets.verify(target)
            extra = target / "unexpected.npz"
            extra.write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                assets.verify(target)
            extra.unlink()
            victim = target / "meta.npz"
            original = victim.read_bytes()
            victim.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ValueError):
                assets.verify(target)
            victim.unlink()
            with self.assertRaises(ValueError):
                assets.verify(target)

    def test_local_asset_installer_uses_only_manifest_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "package"
            for row in assets.manifest()["files"]:
                destination = package / row["package_member"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(assets.DEFAULT_WEIGHTS / row["file"], destination)
            (package / "unrelated.csv").write_text("invented,not,a,model\n", encoding="utf-8")
            target = root / "installed"
            install(package, target)
            self.assertEqual(set(assets.verify(target)), {r["file"] for r in assets.manifest()["files"]})
            self.assertFalse((target / "unrelated.csv").exists())


class ExportTests(unittest.TestCase):
    def test_explicit_ATD_export_and_official_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate(root / "input", 320)
            xml = root / "input/tactical/synthetic_atd.xml"
            raw = root / "input/tactical/synthetic_atd.txt"
            p, b, t = load_tracking(xml, raw, 320)
            out = root / "prediction.csv"
            export(xml, raw, out, frames=320, periods=[(0, 320)])
            validate_csv(out, 320)
            rows = out.read_bytes().splitlines()
            self.assertEqual(rows[0], b"frame,label")
            self.assertEqual(len(rows), 321)
            self.assertFalse(out.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r", out.read_bytes())
            original = out.read_bytes()
            with self.assertRaises(ValueError):
                export(xml, raw, out, frames=320, periods=[(0, 320)])
            self.assertEqual(out.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
