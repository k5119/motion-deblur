"""선형 PSF를 이용한 합성 모션 블러와 Wiener 복원."""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .image_io import Image

DEFAULT_WIENER_K = 0.005
DEFAULT_NOISE_SEED = 0
DEFAULT_RL_ITERATIONS = 20
BOUNDARIES = ("wrap", "reflect", "taper")
Boundary = Literal["wrap", "reflect", "taper"]


def linear_psf(length: int, angle: float) -> np.ndarray:
    """길이와 시계 방향 각도로 정규화된 아화소 선형 PSF를 만든다."""
    if length < 1 or not np.isfinite(angle):
        raise ValueError("PSF 길이는 양의 정수, 각도는 유한한 값이어야 합니다.")
    # 짝수 길이에도 중심 픽셀을 기준으로 대칭인 홀수 크기 배열을 사용한다.
    size = length + 2 + (length % 2 == 0)
    center = size // 2
    radians = np.deg2rad(angle % 180)
    distance = np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 32)
    x = center + distance * np.cos(radians)
    y = center + distance * np.sin(radians)
    left = np.floor(x).astype(int)
    top = np.floor(y).astype(int)
    fx, fy = x - left, y - top
    kernel = np.zeros((size, size))
    for dy, wy in enumerate((1 - fy, fy)):
        for dx, wx in enumerate((1 - fx, fx)):
            np.add.at(kernel, (top + dy, left + dx), wy * wx)
    return kernel / kernel.sum()


def apply_motion_blur(image: Image, psf: NDArray[np.float64]) -> Image:
    """uint8 BGR과 정규화된 선형 PSF를 받아 반사 경계로 2차원 컨볼루션한다."""
    # filter2D는 상관 연산이므로 커널을 뒤집어 컨볼루션을 수행한다.
    return cv2.filter2D(image, -1, psf[::-1, ::-1], borderType=cv2.BORDER_REFLECT_101)


def _validate_image(image: Image) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("비어 있지 않은 uint8 BGR 이미지가 필요합니다.")


def _validate_psf(psf: np.ndarray) -> None:
    if (psf.ndim != 2 or any(size % 2 != 1 for size in psf.shape)
            or not np.isfinite(psf).all() or np.any(psf < 0) or not np.isclose(psf.sum(), 1)):
        raise ValueError("유한한 비음수 계수와 합이 1인 홀수 크기 2차원 PSF가 필요합니다.")


def _to_uint8(values: np.ndarray, peak: float = 1.0) -> Image:
    if not np.isfinite(values).all():
        raise ValueError("복원 결과에 NaN 또는 Inf가 있습니다.")
    return np.clip(np.rint(values * (255 / peak)), 0, 255).astype(np.uint8)


def synthesize_motion_blur(
    image: Image, psf: np.ndarray, *, noise_sigma: float = 0.0, seed: int = DEFAULT_NOISE_SEED,
) -> Image:
    """블러 후 [0, 1] 단위의 가우시안 잡음을 더하고 uint8 BGR로 변환한다."""
    _validate_image(image)
    _validate_psf(psf)
    if not np.isfinite(noise_sigma) or not 0 <= noise_sigma <= 1 or seed < 0:
        raise ValueError("잡음 표준편차는 0~1, 시드는 음이 아닌 정수여야 합니다.")
    if noise_sigma == 0:
        return apply_motion_blur(image, psf)

    normalized = image.astype(np.float32) / 255
    blurred = cv2.filter2D(normalized, -1, psf[::-1, ::-1], borderType=cv2.BORDER_REFLECT_101)
    rng = np.random.default_rng(seed)
    blurred += rng.standard_normal(blurred.shape, dtype=np.float32) * np.float32(noise_sigma)
    return _to_uint8(blurred)


def psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """홀수 크기 PSF 중심을 (0, 0)으로 옮기고 실수 FFT의 반쪽 스펙트럼을 반환한다."""
    _validate_psf(psf)
    kh, kw = psf.shape
    if len(shape) != 2 or shape[0] < kh or shape[1] < kw:
        raise ValueError("FFT 영상 크기는 PSF보다 작을 수 없습니다.")
    kernel = np.zeros(shape, dtype=psf.dtype)
    kernel[:kh, :kw] = psf
    centered = np.roll(kernel, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.rfft2(centered)


def _prepare_boundary(image: Image, psf: np.ndarray, boundary: Boundary) -> tuple[Image, tuple[slice, slice], NDArray[np.float64] | None]:
    """경계 옵션에 따라 관측 영상과 원본 영역의 슬라이스, 감쇠 가중치를 반환한다."""
    if boundary not in BOUNDARIES:
        raise ValueError(f"경계 처리는 {BOUNDARIES} 중 하나여야 합니다.")
    height, width = image.shape[:2]
    if boundary == "wrap":
        return image, (slice(0, height), slice(0, width)), None
    pad_y, pad_x = (size - 1 for size in psf.shape)
    padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    crop = (slice(pad_y, pad_y + height), slice(pad_x, pad_x + width))
    window = None
    if boundary == "taper":
        # 원본 영역은 그대로 두고 바깥 패딩만 채널 평균으로 감쇠한다.
        y, x = (np.arange(size) for size in padded.shape[:2])
        wy = np.minimum(np.minimum(y, y[::-1]) / pad_y, 1) if pad_y else np.ones(y.size)
        wx = np.minimum(np.minimum(x, x[::-1]) / pad_x, 1) if pad_x else np.ones(x.size)
        window = wy[:, None] * wx
    return padded, crop, window


def wiener_restore(
    image: Image, psf: NDArray[np.float64], k: float = DEFAULT_WIENER_K, *, boundary: Boundary = "taper",
) -> Image:
    """PSF의 켤레와 K를 사용하는 주파수 역필터로 uint8 BGR 영상을 복원한다."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("Wiener K는 0보다 큰 유한한 값이어야 합니다.")
    _validate_image(image)
    _validate_psf(psf)
    padded, crop, window = _prepare_boundary(image, psf, boundary)
    shape = padded.shape[:2]
    transfer = psf2otf(psf, shape)
    gain = transfer.conj() / (np.abs(transfer) ** 2 + k)

    restored = np.empty_like(image)
    for channel in range(image.shape[2]):
        # 기본 Wiener의 기존 0~255 float64 연산 및 반올림을 유지한다.
        values = padded[:, :, channel].astype(np.float64)
        if window is not None:
            mean = float(image[:, :, channel].mean())
            values = mean + (values - mean) * window
        spectrum = np.fft.rfft2(values) * gain
        spatial = np.fft.irfft2(spectrum, s=shape, axes=(0, 1))
        restored[:, :, channel] = _to_uint8(spatial[crop], peak=255)
    return restored


def richardson_lucy_restore(
    image: Image, psf: np.ndarray, iterations: int = DEFAULT_RL_ITERATIONS, *, boundary: Boundary = "taper",
) -> Image:
    """양의 초기값에서 PSF와 그 수반 연산으로 RL을 반복하고 최종 BGR을 반환한다."""
    if iterations < 1:
        raise ValueError("RL 반복 횟수는 1 이상이어야 합니다.")
    _validate_image(image)
    _validate_psf(psf)
    padded, crop, window = _prepare_boundary(image, psf, boundary)
    shape = padded.shape[:2]
    transfer = psf2otf(psf.astype(np.float32), shape)
    adjoint = transfer.conj()
    epsilon = np.float32(1e-6)
    if window is not None:
        window = window.astype(np.float32)

    restored = np.empty_like(image)
    for channel in range(image.shape[2]):
        observed = padded[:, :, channel].astype(np.float32) / 255
        if window is not None:
            mean = observed[crop].mean()
            observed = mean + (observed - mean) * window
        estimate = np.full(shape, 0.5, dtype=np.float32)
        for _ in range(iterations):
            prediction = np.fft.irfft2(np.fft.rfft2(estimate) * transfer, s=shape, axes=(0, 1))
            ratio = observed / np.maximum(prediction, epsilon)
            correction = np.fft.irfft2(np.fft.rfft2(ratio) * adjoint, s=shape, axes=(0, 1))
            # 음수 반올림 오차는 제거한다. 순환 연산에서 단위합 PSF의 Aᵀ1은 1이다.
            estimate *= np.maximum(correction, 0)
            if not np.isfinite(estimate).all():
                raise ValueError("RL 반복 중 수치 발산이 발생했습니다.")
        restored[:, :, channel] = _to_uint8(estimate[crop])
    return restored


def compare_quality(reference: Image, image: Image) -> tuple[float, float]:
    """같은 해상도의 BGR 이미지에서 전체 픽셀 PSNR(dB)과 평균 SSIM을 반환한다."""
    if reference.shape != image.shape:
        raise ValueError("PSNR·SSIM에는 동일 해상도와 채널 수의 영상이 필요합니다.")

    original = reference.astype(np.float64)
    candidate = image.astype(np.float64)
    mse = float(np.mean((original - candidate) ** 2))
    psnr = float(10 * np.log10(255**2 / mse)) if mse else float("inf")

    window_size = min(11, *original.shape[:2])
    if window_size % 2 == 0:
        window_size -= 1

    def local_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return cv2.GaussianBlur(values, (window_size, window_size), 1.5)

    mu_x, mu_y = local_mean(original), local_mean(candidate)
    var_x = local_mean(original * original) - mu_x**2
    var_y = local_mean(candidate * candidate) - mu_y**2
    covariance = local_mean(original * candidate) - mu_x * mu_y
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    margin = window_size // 2
    if margin:
        ssim = ssim[margin:-margin, margin:-margin]
    return psnr, float(ssim.mean())
