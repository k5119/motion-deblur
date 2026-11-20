"""FFT 스펙트럼에서 선형 방향성의 축을 추정한다."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

MIN_CONFIDENCE = 0.20
HARMONIC_SCALE = 0.65


@dataclass(frozen=True)
class MotionDirection:
    """후보 축은 오른쪽 수평 0°, 시계 방향 [0°, 180°)이다.

    angle_deg가 None이면 추정 불가다. confidence는 0~1의 방향성 점수다.
    """

    angle_deg: float | None
    confidence: float


def estimate_motion_direction(gray: NDArray[np.uint8], *, min_confidence: float = MIN_CONFIDENCE) -> MotionDirection:
    """8비트 그레이스케일에서 선형 모션과 양립하는 방향성의 축을 반환한다.

    DC 및 프레임 경계의 영향을 줄인 로그 스펙트럼을 극좌표에서 표본화한다.
    주파수 대역별 2차 각도 성분으로 180° 주기의 강한 스펙트럼 축을 찾는다.
    선형 PSF는 이동 방향으로 주파수를 감쇠시키므로, 강한 축에 수직인
    방향이 공간 영역의 이동 축 후보가 된다.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("FFT 신뢰도 기준은 0~1이어야 합니다.")
    height, width = gray.shape
    if min(height, width) < 64 or gray.min() == gray.max():
        return MotionDirection(None, 0.0)

    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    centered = gray.astype(np.float32) - float(gray.mean())
    fft = cv2.dft(centered * window, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude = cv2.magnitude(fft[..., 0], fft[..., 1])
    # 극좌표 표본화 전에 0주파수를 중앙으로 이동한다.
    magnitude = np.fft.fftshift(magnitude)
    np.divide(magnitude, float(np.sqrt(gray.size)), out=magnitude)
    np.log1p(magnitude, out=magnitude)

    angles = np.deg2rad(np.arange(180, dtype=np.float32))
    radii = np.linspace(0.06, 0.32, 72, dtype=np.float32)[:, None]
    # 주파수는 cycles/pixel 단위다. 직사각형 FFT는 축마다 눈금이 다르다.
    map_x = width // 2 + width * radii * np.cos(angles)
    map_y = height // 2 + height * radii * np.sin(angles)
    # 양선형 표본화. 각 좌표는 DC에서 최대 0.32 cycles/pixel 떨어져 있다.
    left, top = map_x.astype(np.intp), map_y.astype(np.intp)
    fx, fy = map_x - left, map_y - top
    upper = (1 - fx) * magnitude[top, left] + fx * magnitude[top, left + 1]
    lower = (1 - fx) * magnitude[top + 1, left] + fx * magnitude[top + 1, left + 1]
    polar = (1 - fy) * upper + fy * lower
    angular_profiles = polar.reshape(3, 24, angles.size).mean(axis=1)
    angular_basis = np.exp(2j * angles)
    harmonics = (angular_profiles - angular_profiles.mean(axis=1, keepdims=True)) @ angular_basis
    harmonics /= angles.size

    mean_harmonic = harmonics.mean()
    mean_strength = float(np.abs(harmonics).mean())
    if mean_strength == 0:
        return MotionDirection(None, 0.0)
    # 대역별 축이 다르면 복소평균의 크기가 줄어 점수가 낮아진다.
    agreement = min(1.0, abs(mean_harmonic) / mean_strength)
    confidence = float(min(1.0, abs(mean_harmonic) / HARMONIC_SCALE) * agreement)
    if confidence < min_confidence:
        return MotionDirection(None, confidence)

    spectrum_axis = np.degrees(np.angle(mean_harmonic)) / 2
    return MotionDirection(float((spectrum_axis + 90) % 180), confidence)
