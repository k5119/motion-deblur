"""배치 결과에서 참조 영상 및 방향 후보의 부재를 검증한다."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from motion_blur.batch import evaluate_case, find_inputs


class BatchExperimentTest(unittest.TestCase):
    def test_captured_blur_omits_reference_metrics(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        options = Namespace(estimated_length=None, real_length=15, k=0.005)
        rows, images = evaluate_case(
            {"image": "captured.png"}, image, None, None, options, lambda observed: observed, "cpu", None,
        )
        self.assertEqual([row["status"] for row in rows], [
            "ok", "skipped_no_ground_truth", "skipped_no_direction", "ok",
        ])
        self.assertEqual(set(images), {"input", "nafnet"})
        for row in rows:
            self.assertNotIn("psnr_db", row)
            self.assertNotIn("ssim", row)
        self.assertIsNone(rows[2]["estimated_angle_deg"])
        self.assertEqual(rows[2]["estimated_length_px"], 15)

    def test_quick_selects_first_supported_file_per_type(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            public = root / "public"
            public.mkdir()
            for filename in ("b.jpg", "a.PNG", "0.csv"):
                (public / filename).touch()
            (root / "legacy.jpg").touch()
            self.assertEqual(find_inputs(root, quick=True), [("public", public / "a.PNG")])
            self.assertEqual(len(find_inputs(root, quick=False)), 2)
