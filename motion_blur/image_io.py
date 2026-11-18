"""OpenCV 기반 이미지 입력 및 PNG 저장."""

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]


def load_image(path: Path) -> Image:
    """파일을 읽어 (H, W, 3) uint8 BGR 배열을 반환한다.

    파일 접근 오류는 OSError, 디코딩 오류는 ValueError로 전달한다.
    """
    # Python이 경로를 열고 OpenCV는 바이트를 받아 Windows 한글 경로를 처리한다.
    encoded = path.read_bytes()
    if not encoded:
        raise ValueError(f"입력 파일이 비어 있습니다. ({path})")

    try:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"이미지를 디코딩할 수 없습니다. ({path})") from exc

    if image is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다. ({path})")
    return image


def save_image(path: Path, image: Image) -> Path:
    """(H, W, 3) uint8 BGR 배열을 PNG로 저장하고 출력 경로를 반환한다."""
    if path.suffix.lower() != ".png":
        raise ValueError("출력 파일 확장자는 .png여야 합니다.")
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError("저장할 이미지는 비어 있지 않은 uint8 BGR 배열이어야 합니다.")

    try:
        success, encoded = cv2.imencode(".png", image)
    except cv2.error as exc:
        raise ValueError(f"PNG 인코딩에 실패했습니다. ({path})") from exc

    if not success:
        raise ValueError(f"PNG 인코딩에 실패했습니다. ({path})")

    path.parent.mkdir(parents=True, exist_ok=True)
    # 배타적으로 열어 기존 파일을 보호한다.
    with path.open("xb") as output:
        output.write(encoded.tobytes())
    return path
