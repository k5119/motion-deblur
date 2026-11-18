"""분석 결과에 따른 보정 여부 판단 인터페이스."""

from enum import Enum

from .analysis import ImageQualityAnalysis


class CorrectionDecision(str, Enum):
    """보정 필요, 보정 불필요, 판단 보류를 구분한다."""

    APPLY = "apply"
    SKIP = "skip"
    UNDETERMINED = "undetermined"


def decide_correction(analysis: ImageQualityAnalysis) -> CorrectionDecision:
    """ImageQualityAnalysis를 받아 보정 여부를 반환한다."""
    return CorrectionDecision.UNDETERMINED
