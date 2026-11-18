"""선형 PSF를 이용한 합성 모션 블러와 Wiener 복원."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .image_io import Image

DEFAULT_WIENER_K = 0.005


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


def wiener_restore(image: Image, psf: NDArray[np.float64], k: float = DEFAULT_WIENER_K) -> Image:
    """중앙이 원점인 홀수 크기 PSF와 양의 K로 BGR 채널을 각각 복원한다."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("Wiener K는 0보다 큰 유한한 값이어야 합니다.")

    height, width = image.shape[:2]
    kh, kw = psf.shape
    pad_y, pad_x = kh - 1, kw - 1
    padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    shape = padded.shape[:2]
    kernel = np.zeros(shape)
    kernel[:kh, :kw] = psf
    kernel = np.roll(kernel, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer = np.fft.rfft2(kernel)
    gain = transfer.conj() / (np.abs(transfer) ** 2 + k)

    # 원본 영역의 가중치는 1이다. 패딩 바깥쪽만 채널 평균으로 부드럽게 감쇠한다.
    y = np.minimum(np.arange(shape[0]), np.arange(shape[0])[::-1])
    x = np.minimum(np.arange(shape[1]), np.arange(shape[1])[::-1])
    window = np.minimum(y / max(pad_y, 1), 1)[:, None] * np.minimum(x / max(pad_x, 1), 1)
    restored = np.empty_like(image)
    for channel in range(image.shape[2]):
        values = padded[:, :, channel].astype(np.float64)
        mean = float(image[:, :, channel].mean())
        values = mean + (values - mean) * window
        spectrum = np.fft.rfft2(values) * gain
        spatial = np.fft.irfft2(spectrum, s=shape, axes=(0, 1))
        cropped = spatial[pad_y:pad_y + height, pad_x:pad_x + width]
        restored[:, :, channel] = np.clip(np.rint(cropped), 0, 255).astype(np.uint8)
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
