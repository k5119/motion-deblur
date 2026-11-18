"""선형 PSF, Wiener 복원, 화질 지표의 회귀 테스트."""

import math
import unittest

import numpy as np

from motion_blur.classical import apply_motion_blur, compare_quality, linear_psf, wiener_restore
from tests.validate_motion_direction import textures


class ClassicalRestorationTest(unittest.TestCase):
    def test_psf_is_normalized_centered_and_axial(self) -> None:
        for length in (1, 8, 15):
            for angle in (0, 45, 90, 135):
                with self.subTest(length=length, angle=angle):
                    kernel = linear_psf(length, angle)
                    self.assertEqual(kernel.shape[0] % 2, 1)
                    self.assertTrue((kernel >= 0).all())
                    self.assertAlmostEqual(float(kernel.sum()), 1)
                    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-14)
                    np.testing.assert_allclose(kernel, linear_psf(length, angle + 180))

    def test_convolution_preserves_channel_and_axis(self) -> None:
        impulse = np.zeros((65, 65, 3), dtype=np.uint8)
        impulse[32, 32, 2] = 255
        result = apply_motion_blur(impulse, linear_psf(9, 0))
        y, x = np.nonzero(result[:, :, 2])
        self.assertTrue((y == 32).all())
        self.assertEqual(int(x.min() + x.max()), 64)
        self.assertEqual(int(result[:, :, :2].sum()), 0)

    def test_matching_psf_improves_controlled_blur(self) -> None:
        reference = textures()["multiscale"]
        kernel = linear_psf(15, 45)
        blurred = apply_motion_blur(reference, kernel)
        restored = wiener_restore(blurred, kernel)
        self.assertEqual(restored.shape, reference.shape)
        self.assertEqual(restored.dtype, np.uint8)
        base_psnr, base_ssim = compare_quality(reference, blurred)
        psnr, ssim = compare_quality(reference, restored)
        self.assertGreater(psnr, base_psnr)
        self.assertGreater(ssim, base_ssim)

    def test_metrics_and_regularization_validation(self) -> None:
        sample = np.full((5, 7, 3), 128, dtype=np.uint8)
        psnr, ssim = compare_quality(sample, sample)
        self.assertTrue(math.isinf(psnr))
        self.assertAlmostEqual(ssim, 1)
        psnr, ssim = compare_quality(sample, sample + 10)
        self.assertAlmostEqual(psnr, 20 * math.log10(255 / 10))
        c1 = (0.01 * 255) ** 2
        self.assertAlmostEqual(ssim, (2 * 128 * 138 + c1) / (128**2 + 138**2 + c1))
        with self.assertRaises(ValueError):
            compare_quality(sample, sample[:-1])
        with self.assertRaises(ValueError):
            wiener_restore(sample, linear_psf(9, 30), k=0)
        restored = wiener_restore(sample, linear_psf(9, 30), k=0.01)
        self.assertTrue((restored == 127).all())
