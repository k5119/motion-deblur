"""단일 이미지의 모션 블러 분석 인터페이스."""

from dataclasses import dataclass
from enum import Enum

from .image_io import Image


class AnalysisStatus(str, Enum):
    """분석의 구현 및 완료 상태."""

    NOT_IMPLEMENTED = "not_implemented"
    COMPLETE = "complete"


@dataclass(frozen=True)
class BlurAnalysis:
    """분석 상태와 선택적 측정값. 미측정 값은 None이다.

    length_px는 추정 이동 길이이며 angle_deg는 이미지 오른쪽 방향을
    기준으로 시계 방향으로 측정한 블러 축 각도다. 각도 범위는 [0, 180)이다.
    """

    status: AnalysisStatus
    length_px: float | None = None
    angle_deg: float | None = None


def analyze_image(image: Image) -> BlurAnalysis:
    """(H, W, 3) uint8 BGR 배열을 받아 분석 상태와 측정값을 반환한다."""
    return BlurAnalysis(status=AnalysisStatus.NOT_IMPLEMENTED)
