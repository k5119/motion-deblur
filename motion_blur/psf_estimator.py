"""선형 블러 방향과 켑스트럼 기반 길이 후보를 계산한다."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .image_io import Image, limit_image_size
from .motion_direction import MotionDirection, estimate_motion_direction

ANALYSIS_MAX_SIZE = 512
GRADIENT_MIN_CONFIDENCE = 0.20
MIN_LENGTH_PROMINENCE = 3.0
DirectionMethod = Literal["fft", "gradient"]


@dataclass(frozen=True)
class PSFEstimate:
    """실제 분석 크기, 두 방향 후보, 길이와 추정 불가 사유를 담는다."""

    original_size: tuple[int, int]
    analysis_size: tuple[int, int]
    direction_method: DirectionMethod
    fft: MotionDirection
    gradient: MotionDirection
    angle_deg: float | None
    length_analysis_px: int | None
    length_original_px: float | None
    length_prominence: float | None
    length_scores: tuple[tuple[int, float], ...]
    reason: str | None


def gradient_direction(gray: np.ndarray) -> MotionDirection:
    """Sobel 구조 텐서에서 에너지가 가장 낮은 방향과 이방성을 구한다."""
    if min(gray.shape) < 64:
        return MotionDirection(None, 0.0)
    smoothed = cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), 1.0)
    dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0)[2:-2, 2:-2]
    dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1)[2:-2, 2:-2]
    xx, yy, xy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    total = xx + yy
    if total <= 1e-10:
        return MotionDirection(None, 0.0)
    confidence = min(1.0, float(np.hypot(xx - yy, 2 * xy) / total))
    axis = (np.degrees(np.arctan2(2 * xy, xx - yy)) / 2 + 90) % 180
    return MotionDirection(float(axis) if confidence >= GRADIENT_MIN_CONFIDENCE else None, confidence)


def to_original_axis(
    angle_deg: float, length_px: float,
    analysis_size: tuple[int, int], original_size: tuple[int, int],
) -> tuple[float, float]:
    """각 축의 실제 리사이즈 비율을 반영해 PSF 끝점 간 거리 L−1과 방향을 환산한다."""
    radians = np.deg2rad(angle_deg)
    dx = np.cos(radians) * original_size[0] / analysis_size[0]
    dy = np.sin(radians) * original_size[1] / analysis_size[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 180), float(1 + (length_px - 1) * np.hypot(dx, dy))


def cepstral_length(gray: np.ndarray, angle_deg: float) -> tuple[int | None, float, tuple[tuple[int, float], ...]]:
    """방향 주변의 켑스트럼 음의 골을 탐색하고 강도가 낮으면 보류한다."""
    height, width = gray.shape
    max_lag = min(40, min(height, width) // 4)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    windowed = (gray.astype(np.float32) - float(gray.mean())) * window
    magnitude = np.abs(np.fft.rfft2(windowed))
    floor = max(float(np.percentile(magnitude, 10)), 1e-6)
    cepstrum = np.fft.fftshift(np.fft.irfft2(np.log(np.maximum(magnitude, floor)), s=gray.shape, axes=(0, 1)))
    lags = np.arange(4, max_lag + 1, dtype=np.float32)[:, None]
    angles = np.deg2rad(np.array([angle_deg + offset for offset in (-5, 0, 5, 85, 90, 95)], dtype=np.float32))
    x = width // 2 + lags * np.cos(angles)
    y = height // 2 + lags * np.sin(angles)
    profile = cv2.remap(cepstrum.astype(np.float32), x, y, cv2.INTER_LINEAR)
    samples = profile[:, :3].mean(axis=1) - profile[:, 3:].mean(axis=1)
    # 인접 지연의 평균으로 완만한 바닥을 제거해 국소 골만 비교한다.
    contrast = (samples[:-4] + samples[4:]) / 2 - samples[2:-2]
    median = np.median(contrast)
    scale = max(float(1.4826 * np.median(np.abs(contrast - median))), 1e-6)
    scores = (contrast - median) / scale
    candidates = tuple((int(lag) + 1, float(score)) for lag, score in zip(lags[2:-2, 0], scores))
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    return candidates[index][0] if confidence >= MIN_LENGTH_PROMINENCE else None, confidence, candidates


def estimate_psf(
    image: Image, *, method: DirectionMethod = "fft", angle_override: float | None = None,
) -> PSFEstimate:
    """최대 512px에서 추정하고 길이와 방향을 입력 영상의 단위로 환산한다."""
    if method not in {"fft", "gradient"}:
        raise ValueError("방향 추정 방식은 fft 또는 gradient여야 합니다.")
    if angle_override is not None and not np.isfinite(angle_override):
        raise ValueError("수동 각도는 유한한 값이어야 합니다.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("비어 있지 않은 uint8 BGR 이미지가 필요합니다.")
    scaled = limit_image_size(image, ANALYSIS_MAX_SIZE)
    original_size = (image.shape[1], image.shape[0])
    analysis_size = (scaled.shape[1], scaled.shape[0])
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    fft = estimate_motion_direction(gray)
    gradient = gradient_direction(gray)
    direction = fft if method == "fft" else gradient
    angle = direction.angle_deg
    if angle_override is not None:
        angle, _ = to_original_axis(angle_override, 1, original_size, analysis_size)

    length = prominence = original_length = original_angle = None
    scores = ()
    reason = "no_direction"
    if angle is not None:
        original_angle, _ = to_original_axis(angle, 1, analysis_size, original_size)
        reason = "insufficient_analysis_size"
        if min(gray.shape) >= 64:
            length, prominence, scores = cepstral_length(gray, angle)
            reason = "weak_length_candidate"
            if length is not None:
                _, original_length = to_original_axis(angle, length, analysis_size, original_size)
                reason = None

    return PSFEstimate(original_size, analysis_size, method, fft, gradient, original_angle,
                       length, original_length, prominence, scores, reason)
