"""OpenCV와 NumPy를 사용해 원본 이미지의 품질 지표를 계산한다."""

from dataclasses import dataclass

import cv2
import numpy as np

from .image_io import Image
from .motion_direction import MotionDirection, estimate_motion_direction

DARK_PIXEL_MAX = 50
BRIGHT_PIXEL_MIN = 205


@dataclass(frozen=True)
class ImageQualityAnalysis:
    """밝기·노출 분포, Laplacian 분산과 FFT 방향 후보를 담는다."""

    mean_brightness: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    grayscale_histogram: tuple[int, ...]
    laplacian_variance: float
    motion_direction: MotionDirection


def analyze_image(image: Image) -> ImageQualityAnalysis:
    """(H, W, 3) uint8 BGR 배열을 받아 이미지 품질 지표를 반환한다.

    OpenCV로 8비트 그레이스케일로 변환한다. 평균 밝기는 명암값의
    산술평균(0~255)이며, 히스토그램의 각 구간은 해당 명암값의 픽셀 수다.
    어두운 비율은 명암값 0~50, 밝은 비율은 205~255인 픽셀 수를 전체
    픽셀 수로 나눈 값이다. 밝기 구간은 노출 분포를 기술하는 데 사용한다.
    선명도 지표는 부호를 유지한 64비트 Laplacian 응답(ksize=1)의
    모집단 분산이다. 강한 경계가 많으면 이 수치가 커질 수 있다.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("분석 입력은 비어 있지 않은 (H, W, 3) uint8 BGR 배열이어야 합니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    return ImageQualityAnalysis(
        mean_brightness=float(gray.mean()),
        dark_pixel_ratio=float(histogram[: DARK_PIXEL_MAX + 1].sum() / gray.size),
        bright_pixel_ratio=float(histogram[BRIGHT_PIXEL_MIN:].sum() / gray.size),
        grayscale_histogram=tuple(histogram.tolist()),
        laplacian_variance=float(laplacian.var()),
        motion_direction=estimate_motion_direction(gray),
    )
