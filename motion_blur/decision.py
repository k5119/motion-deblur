"""이미지 품질 지표로 복원 권장 여부를 잠정 평가한다."""

from dataclasses import dataclass
from enum import Enum
from math import sqrt

from .analysis import ImageQualityAnalysis

# 합성 PSF와 대조군에서 정한 잠정 기준. Laplacian은 아래 긴 변 기준으로 비교한다.
ASSESSMENT_MAX_SIZE = 512
MEAN_BRIGHTNESS_RANGE = (50.0, 205.0)
MAX_EXPOSURE_TAIL_RATIO = 0.50
MIN_GRAY_STDDEV = 8.0
LOW_LAPLACIAN_VARIANCE = 300.0
HIGH_LAPLACIAN_VARIANCE = 1000.0
MIN_SUPPORTING_DIRECTION_SCORE = 0.30


class CorrectionDecision(str, Enum):
    """복원 권장 여부와 판단 보류를 구분한다."""

    APPLY = "apply"
    SKIP = "skip"
    UNDETERMINED = "undetermined"

    @property
    def label(self) -> str:
        """터미널에 표시할 한국어 판단 결과를 반환한다."""
        return {
            CorrectionDecision.APPLY: "복원 권장",
            CorrectionDecision.SKIP: "복원 불필요",
            CorrectionDecision.UNDETERMINED: "판단 보류",
        }[self]


@dataclass(frozen=True)
class CorrectionAssessment:
    """권장 여부와 해당 판단을 설명하는 로그용 문장을 담는다."""

    decision: CorrectionDecision
    reason: str


def decide_correction(
    analysis: ImageQualityAnalysis, *, low_laplacian: float = LOW_LAPLACIAN_VARIANCE,
    high_laplacian: float = HIGH_LAPLACIAN_VARIANCE,
) -> CorrectionAssessment:
    """긴 변을 최대 512px로 제한한 이미지의 지표에서 판단과 근거를 반환한다.

    노출이나 명암 대비가 판단을 어렵게 하면 보류한다. 낮은 선명도에는
    FFT 방향성의 뒷받침을, 높은 선명도에는 상충하는 방향 후보가 없는
    상태를 요구한다. FFT 점수는 주파수 방향 특징의 강도를 나타낸다.
    """
    brightness = analysis.mean_brightness
    dark, bright = analysis.dark_pixel_ratio, analysis.bright_pixel_ratio
    low_brightness, high_brightness = MEAN_BRIGHTNESS_RANGE
    if not low_brightness <= brightness <= high_brightness or max(dark, bright) >= MAX_EXPOSURE_TAIL_RATIO:
        return CorrectionAssessment(
            CorrectionDecision.UNDETERMINED,
            f"노출 분포가 치우침 (평균 {brightness:.1f}, 어두운 {dark:.1%}, 밝은 {bright:.1%}).",
        )

    histogram = analysis.grayscale_histogram
    gray_stddev = sqrt(
        sum(count * (level - brightness) ** 2 for level, count in enumerate(histogram))
        / sum(histogram)
    )
    if gray_stddev < MIN_GRAY_STDDEV:
        return CorrectionAssessment(
            CorrectionDecision.UNDETERMINED,
            f"명암 변화가 작아 경계 선명도를 판단하기 어려움 (표준편차 {gray_stddev:.1f}).",
        )

    laplacian = analysis.laplacian_variance
    direction = analysis.motion_direction
    if laplacian <= low_laplacian and (
        direction.angle_deg is not None and direction.confidence >= MIN_SUPPORTING_DIRECTION_SCORE
    ):
        return CorrectionAssessment(
            CorrectionDecision.APPLY,
            f"낮은 선명도 (Laplacian {laplacian:.1f})와 뚜렷한 FFT 방향성 ({direction.confidence:.3f}).",
        )

    if laplacian >= high_laplacian and direction.angle_deg is None:
        return CorrectionAssessment(
            CorrectionDecision.SKIP,
            f"충분한 명암 대비와 높은 선명도 (Laplacian {laplacian:.1f}), 상충하는 FFT 방향 후보 없음.",
        )

    fft_evidence = (
        f"FFT 방향성 {direction.confidence:.3f}"
        if direction.angle_deg is not None else f"FFT 방향 추정 불가 ({direction.confidence:.3f})"
    )
    return CorrectionAssessment(
        CorrectionDecision.UNDETERMINED,
        f"선명도와 방향성 지표로 확정하기 어려움 (Laplacian {laplacian:.1f}, {fft_evidence}).",
    )
