"""임계값 실험의 지표와 분할 검사를 확인한다."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from motion_blur.image_io import save_image


@unittest.skipUnless(find_spec("matplotlib"), "report dependency is optional")
class ThresholdValidationTest(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        from motion_blur.validate_thresholds import laplacian_scores

        scores = laplacian_scores([
            (True, 5, None), (False, 6, None), (True, 12, None), (False, 15, None),
        ], 10)
        self.assertEqual([scores[name] for name in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual([scores[name] for name in ("accuracy", "precision", "recall", "f1")], [0.5] * 4)

    def test_duplicate_file_across_splits_is_rejected(self) -> None:
        from motion_blur.validate_thresholds import measure_manifest

        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = save_image(root / "sample.png", np.zeros((64, 64, 3), dtype=np.uint8))
            (root / "copy.png").write_bytes(image.read_bytes())
            manifest = root / "data.csv"
            manifest.write_text("path,label,split\nsample.png,sharp,validation\ncopy.png,sharp,test\n")
            with self.assertRaisesRegex(ValueError, "중복"):
                measure_manifest(manifest)
