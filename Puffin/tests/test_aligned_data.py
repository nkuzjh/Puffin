from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from csgo_seen10 import dataset as module
from csgo_seen10.dataset import CsgoSeen10Dataset, collate_seen10
from csgo_seen10.paths import resolve_data_root, resolve_eval_python
from csgo_seen10.prompts import build_pose_prompt, check_prompt_lengths


class AlignedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (32, 32)).save(self.root / "radar.png")
        Image.new("RGB", (32, 32)).save(self.root / "target.jpg")
        self.row = dict(sample_id="cs_agency/file_num1_frame_0001", map_name="cs_agency", file_frame="file_num1_frame_0001",
                        image_path=str(self.root / "target.jpg"), radar_path=str(self.root / "radar.png"), pose=[0] * 5,
                        pose_raw=dict(x=128.0, y=256.0, z=100.125, pitch=math.pi / 2, yaw=math.pi),
                        z_calibration=dict(z_min=70, z_max=395))
        self.protocol = SimpleNamespace(BenchmarkData=lambda _: SimpleNamespace(rows=lambda *args, **kwargs: [self.row]))

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return CsgoSeen10Dataset(data_root=str(self.root), radar_size=224, target_size=448, pose_mode="text", **kwargs)

    def test_physical_text_and_sizes(self):
        with patch.object(module, "_load_protocol", return_value=self.protocol):
            sample = self.dataset()[0]
        self.assertEqual(tuple(sample["cam_values"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["pixel_values"].shape), (3, 448, 448))
        self.assertNotIn("pose_values", sample)
        self.assertIn("x=128.0, y=256.0, z=100.125, pitch=90.0, yaw=180.0", sample["text"])
        self.assertNotIn("pose_values", collate_seen10([sample])["data"]["cam2image"])

    def test_inference_never_decodes_target(self):
        self.row["image_path"] = str(self.root / "MISSING.jpg")
        with patch.object(module, "_load_protocol", return_value=self.protocol):
            for split in ("seen_discrete_test", "seen_continuous"):
                sample = self.dataset(split=split, include_target=False)[0]
                self.assertNotIn("pixel_values", sample)

    def test_reject_mixed_pose_modes(self):
        with patch.object(module, "_load_protocol", return_value=self.protocol):
            text = self.dataset()[0]
            numeric = CsgoSeen10Dataset(data_root=str(self.root))[0]
        with self.assertRaisesRegex(ValueError, "numeric and text"):
            collate_seen10([text, numeric])

    def test_pose_validation_and_truncation(self):
        self.assertIn("z range [70.00, 395.00]", build_pose_prompt(self.row))
        tokenizer = SimpleNamespace(encode=lambda *a, **kw: list(range(257)))
        with self.assertRaisesRegex(ValueError, "truncated"):
            check_prompt_lengths(tokenizer, [self.row])
        self.row["pose_raw"]["x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            build_pose_prompt(self.row)

    def test_explicit_bad_path_never_falls_back(self):
        with patch.dict(os.environ, {"CSGO_DATA_ROOT": str(self.root)}):
            with self.assertRaises(FileNotFoundError):
                resolve_data_root(self.root / "invalid")
        with self.assertRaises(FileNotFoundError):
            resolve_eval_python(self.root / "invalid-python")


if __name__ == "__main__":
    unittest.main()
