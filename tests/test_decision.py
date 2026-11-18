"""합성 대조군으로 복원 판단 규칙을 검증한다."""

import unittest

import cv2
import numpy as np

from motion_blur.analysis import analyze_image
from motion_blur.decision import CorrectionDecision, decide_correction
from tests.validate_motion_direction import linear_psf, textures


class CorrectionDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.images = textures()

    def test_sharp_and_directionally_blurred_texture(self) -> None:
        image = self.images["multiscale"]
        blurred = cv2.filter2D(image, -1, linear_psf(15, 45), borderType=cv2.BORDER_REFLECT_101)
        for sample, expected in [(image, CorrectionDecision.SKIP), (blurred, CorrectionDecision.APPLY)]:
            with self.subTest(expected=expected):
                assessment = decide_correction(analyze_image(sample))
                self.assertIs(assessment.decision, expected)
                self.assertIn("Laplacian", assessment.reason)

    def test_flat_extreme_exposure_and_directional_edges_defer(self) -> None:
        for label, image, reason in [
            ("flat", self.images["flat"], "명암 변화"),
            ("dark", np.full_like(self.images["flat"], 12), "노출 분포"),
            ("bright", np.full_like(self.images["flat"], 240), "노출 분포"),
            ("sharp lines", self.images["lines"], "FFT 방향성"),
        ]:
            with self.subTest(label=label):
                assessment = decide_correction(analyze_image(image))
                self.assertIs(assessment.decision, CorrectionDecision.UNDETERMINED)
                self.assertIn(reason, assessment.reason)

    def test_low_sharpness_without_motion_axis_defers(self) -> None:
        blurred = cv2.GaussianBlur(self.images["multiscale"], (21, 21), 4)
        analysis = analyze_image(blurred)
        self.assertIsNone(analysis.motion_direction.angle_deg)
        assessment = decide_correction(analysis)
        self.assertIs(assessment.decision, CorrectionDecision.UNDETERMINED)
        self.assertIn("FFT 방향 추정 불가", assessment.reason)
