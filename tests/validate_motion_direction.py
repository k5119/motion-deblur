"""합성 선형 PSF 방향 검증. 프로젝트 루트에서 python -m tests.validate_motion_direction."""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

from motion_blur.analysis import analyze_image
from motion_blur.image_io import load_image, save_image

ANGLES = (0, 30, 45, 60, 90, 135)
LENGTHS = (9, 15, 21)


def linear_psf(length: int, angle: float) -> np.ndarray:
    """길이와 시계 방향 각도로 정규화된 아화소 선형 PSF를 만든다."""
    size = length + 2
    center = size // 2
    radians = np.deg2rad(angle)
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


def axial_error(expected: float, estimated: float) -> float:
    """180° 주기의 최소 각도 차이를 반환한다."""
    return abs((estimated - expected + 90) % 180 - 90)


def textures() -> dict[str, np.ndarray]:
    """서로 다른 주파수 분포의 시드 고정 검증 이미지를 만든다."""
    rng = np.random.default_rng(702)
    shape = (384, 512)
    noise = rng.normal(0, 1, shape).astype(np.float32)
    coarse = cv2.resize(rng.random((24, 32), dtype=np.float32), shape[::-1], interpolation=cv2.INTER_CUBIC)
    dots = np.full(shape, 30, dtype=np.uint8)
    for _ in range(700):
        x, y = (int(value) for value in rng.integers((0, 0), shape[::-1]))
        cv2.circle(dots, (x, y), int(rng.integers(1, 8)), int(rng.integers(75, 245)), -1)
    lines = np.full(shape, 128, dtype=np.uint8)
    line_rng = np.random.default_rng(3)
    for _ in range(560):
        x, y = (int(value) for value in line_rng.integers((0, 0), shape[::-1]))
        cv2.line(lines, (x, y), (x + int(line_rng.integers(8, 60)), y), int(line_rng.integers(0, 256)), 1)
    gray = {
        "grain": np.clip(128 + 45 * noise, 0, 255).astype(np.uint8),
        "multiscale": np.clip(35 + 180 * coarse + 17 * noise, 0, 255).astype(np.uint8),
        "dots": dots,
        "flat": np.full(shape, 128, dtype=np.uint8),
        "stripes": np.broadcast_to(((np.arange(shape[1]) // 12) % 2 * 220).astype(np.uint8), shape),
        "lines": lines,
    }
    return {name: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for name, image in gray.items()}


def main() -> None:
    """전체 비교를 CSV로 출력하고 요청 시 사용한 PNG를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, help="검증 이미지 저장 폴더")
    parser.add_argument("--image", type=Path, action="append", default=[], help="추가 원본 이미지")
    args = parser.parse_args()
    images = textures()
    images.update({f"input-{i}-{path.stem}": load_image(path) for i, path in enumerate(args.image, 1)})
    writer = csv.writer(sys.stdout)
    writer.writerow(("texture", "length_px", "applied_deg", "estimated_deg", "axial_error_deg", "confidence"))
    for name, image in images.items():
        applied = [(length, angle) for length in LENGTHS for angle in ANGLES]
        if name in {"flat", "lines"}:
            applied = []
        elif name == "stripes":
            # 줄무늬와 평행한 움직임은 화소를 바꾸지 않는 실패 사례다.
            applied = [(15, 90)]
        for length, angle in [(0, None), *applied]:
            case = image if angle is None else cv2.filter2D(
                image, -1, linear_psf(length, angle), borderType=cv2.BORDER_REFLECT_101,
            )
            result = analyze_image(case).motion_direction
            estimated = result.angle_deg
            error = axial_error(angle, estimated) if angle is not None and estimated is not None else None
            writer.writerow((name, length, angle, estimated, error, result.confidence))
            if args.output_dir is not None:
                label = "original" if angle is None else f"l{length}-a{angle}"
                save_image(args.output_dir / f"{name}-{label}.png", case)


if __name__ == "__main__":
    main()
