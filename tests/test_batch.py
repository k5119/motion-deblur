"""배치 결과에서 참조 영상 및 방향 후보의 부재를 검증한다."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from motion_blur.batch import BatchOptions, evaluate_case, find_inputs


class BatchExperimentTest(unittest.TestCase):
    def test_captured_blur_omits_reference_metrics(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        options = BatchOptions(Path("input"), Path("output"))
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

    def test_batch_size_options(self) -> None:
        self.assertEqual(BatchOptions(Path("input"), Path("output")).max_size, 0)
        self.assertEqual(BatchOptions(Path("input"), Path("output"), max_size=0).max_size, 0)
        with self.assertRaises(ValueError):
            BatchOptions(Path("input"), Path("output"), max_size=-1)

    @unittest.skipUnless(find_spec("matplotlib"), "batch chart dependency is optional")
    def test_quality_groups_require_matching_conditions_and_references(self) -> None:
        from motion_blur.batch_report import METHOD_LABELS, paired_quality

        base = dict(image="frame.png", data_type="public", processed_width=100,
                    processed_height=80, actual_angle_deg=0, actual_length_px=9,
                    status="ok", psnr_db=30.0, ssim=0.8)
        rows = [dict(base, method=method) for method in METHOD_LABELS]
        rows.extend(dict(base, method=method, actual_angle_deg=45, actual_length_px=15)
                    for method in METHOD_LABELS)
        rows.extend(dict(base, method=method, data_type="captured_blurred")
                    for method in METHOD_LABELS)
        rows.extend(dict(base, method=method, actual_length_px=21, processed_width=200)
                    for method in METHOD_LABELS if method != "nafnet")
        rows.append(dict(base, method="nafnet", actual_length_px=21, processed_width=300))
        rows.extend(dict(base, method=method, actual_angle_deg=90,
                         status="failed" if method == "nafnet" else "ok")
                    for method in METHOD_LABELS)
        rows[0]["psnr_db"] = float("inf")
        self.assertEqual({key: len(group) for key, group in paired_quality(rows, "ssim").items()},
                         {(0, 9): 1, (45, 15): 1})
        self.assertEqual(paired_quality(rows, "psnr_db")[(0, 9)][0]["input"], float("inf"))

    def test_manual_psf_runs_when_direction_cannot_be_estimated(self) -> None:
        source = np.full((80, 520, 3), 128, dtype=np.uint8)
        options = BatchOptions(Path("input"), Path("output"), real_angle=195, real_length=9)
        rows, images = evaluate_case(
            {"image": "captured.png"}, source, None, None, options, lambda observed: observed, "cpu", None,
        )
        estimated, manual = rows[2:4]
        self.assertEqual(estimated["status"], "skipped_no_direction")
        self.assertEqual(manual["status"], "ok")
        self.assertEqual(manual["applied_angle_deg"], 15)
        self.assertEqual(manual["applied_length_px"], 9)
        self.assertEqual(manual["analysis_width"], 512)
        self.assertEqual(images["wiener_manual"].shape, source.shape)
        self.assertNotIn("psnr_db", manual)
