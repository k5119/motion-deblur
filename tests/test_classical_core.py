"""OTF 중심 정렬과 CPU 고전 복원의 수치 경로를 검증한다."""

import numpy as np
import pytest

from motion_blur.classical import (
    BOUNDARIES, apply_motion_blur, compare_quality, linear_psf, psf2otf,
    richardson_lucy_restore, synthesize_motion_blur, wiener_restore,
)
from tests.validate_motion_direction import textures


def circular_convolution(image, psf):
    """작은 검증용 영상에서 정의대로 커널 오프셋을 순환 이동하고 합산한다."""
    center_y, center_x = (size // 2 for size in psf.shape)
    return sum(
        psf[y, x] * np.roll(image, (y - center_y, x - center_x), axis=(0, 1))
        for y, x in np.ndindex(psf.shape)
    )


def test_psf2otf_matches_direct_circular_convolution():
    # 비대칭 PSF는 중심 또는 위상 부호를 뒤집는 실수를 드러낸다.
    psf = np.array([[0, 0, 0], [0.2, 0.5, 0], [0, 0.3, 0]])
    image = np.zeros((9, 11))
    image[0, 10] = 1
    otf = psf2otf(psf, image.shape)
    assert otf[0, 0] == pytest.approx(1 + 0j)
    actual = np.fft.irfft2(np.fft.rfft2(image) * otf, s=image.shape, axes=(0, 1))
    np.testing.assert_allclose(actual, circular_convolution(image, psf), atol=1e-12)


@pytest.mark.parametrize("angle", [0, 45, 90])
def test_psf_direction(angle):
    psf = linear_psf(15, angle)
    y, x = np.indices(psf.shape) - psf.shape[0] // 2
    # 계수의 2차 모멘트로 영상 좌표계의 주축을 측정한다.
    moment_xx = np.sum(psf * x**2)
    moment_yy = np.sum(psf * y**2)
    moment_xy = np.sum(psf * x * y)
    measured = np.degrees(np.arctan2(2 * moment_xy, moment_xx - moment_yy)) / 2 % 180
    assert abs((measured - angle + 90) % 180 - 90) < 1e-10


def test_rl_first_iteration_matches_direct_update_and_adjoint():
    psf = np.array([[0, 0, 0], [0.2, 0.5, 0], [0, 0.3, 0]])
    image = np.random.default_rng(7).integers(0, 256, (9, 11, 3), dtype=np.uint8)
    expected = np.empty(image.shape, dtype=np.float64)
    for channel in range(3):
        observed = image[:, :, channel] / 255
        initial = np.full(observed.shape, 0.5)
        predicted = circular_convolution(initial, psf)
        update = initial * circular_convolution(observed / predicted, psf[::-1, ::-1])
        expected[:, :, channel] = update * 255
    actual = richardson_lucy_restore(image, psf, 1, boundary="wrap")
    rounded = np.rint(expected)
    # float32 FFT의 반올림 동점만 ±1 LSB를 허용한다.
    ties = np.isclose(expected % 1, 0.5, atol=1e-5)
    np.testing.assert_array_equal(actual[~ties], rounded[~ties])
    np.testing.assert_allclose(actual[ties], rounded[ties], atol=1, rtol=0)


def test_rl_improves_fixed_fixture_without_assuming_universal_gain():
    reference = textures()["multiscale"][:120, :140]
    psf = linear_psf(15, 45)
    blurred = apply_motion_blur(reference, psf)
    first = richardson_lucy_restore(blurred, psf, 1)
    restored = richardson_lucy_restore(blurred, psf, 20)
    assert compare_quality(reference, restored)[0] > compare_quality(reference, blurred)[0]
    assert not np.array_equal(restored, first)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_boundary_preserves_color_shape_range_and_zero_stability(boundary):
    image = np.zeros((19, 23, 3), dtype=np.uint8)
    image[:, :, 2] = 120
    psf = linear_psf(9, 45)
    for restore in (wiener_restore, richardson_lucy_restore):
        result = restore(image, psf, boundary=boundary)
        assert result.shape == image.shape and result.dtype == np.uint8
        assert np.all(result[:, :, :2] == 0)
        assert np.all(result[:, :, 2] >= 119)


def test_noise_seed_and_no_noise_compatibility():
    image = np.full((41, 35, 3), 128, dtype=np.uint8)
    psf = linear_psf(15, 45)
    np.testing.assert_array_equal(synthesize_motion_blur(image, psf), apply_motion_blur(image, psf))
    noisy = synthesize_motion_blur(image, psf, noise_sigma=0.01, seed=702)
    np.testing.assert_array_equal(noisy, synthesize_motion_blur(image, psf, noise_sigma=0.01, seed=702))
    assert not np.array_equal(noisy, synthesize_motion_blur(image, psf, noise_sigma=0.01, seed=703))
    assert np.std(noisy / 255) == pytest.approx(0.01, abs=0.001)


def test_invalid_core_inputs():
    image = np.zeros((9, 11, 3), dtype=np.uint8)
    psf = linear_psf(5, 0)
    with pytest.raises(ValueError):
        psf2otf(psf, (3, 4))
    with pytest.raises(ValueError):
        psf2otf(psf * 2, (9, 11))
    with pytest.raises(ValueError):
        richardson_lucy_restore(image, psf, 0)
    with pytest.raises(ValueError):
        wiener_restore(image, psf, boundary="invalid")
    with pytest.raises(ValueError):
        synthesize_motion_blur(image, psf, noise_sigma=float("nan"))
