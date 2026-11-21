"""다운로드 없이 GoPro·RealBlur 소규모 합성 로더 검증 자료를 만든다."""

import argparse
from pathlib import Path

import numpy as np

from motion_blur.classical import apply_motion_blur, linear_psf
from motion_blur.image_io import save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 데이터셋 로더 검증 자료 생성")
    parser.add_argument("output", type=Path, help="새 출력 폴더")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(702)
    reference = rng.integers(20, 236, (112, 144, 3), dtype=np.uint8)
    kernel = linear_psf(15, 30)
    ordinary = apply_motion_blur(reference, kernel)
    # 로더 조건 구분을 위한 별도 합성 이미지다.
    encoded = np.rint((reference / 255) ** 2.2 * 255).astype(np.uint8)
    gamma = np.rint((apply_motion_blur(encoded, kernel) / 255) ** (1 / 2.2) * 255).astype(np.uint8)
    gopro = Path("gopro/GOPRO_Large/test/fixture")
    realblur = Path("realblur/fixture")
    for parent in (gopro, realblur):
        save_image(args.output / parent / "sharp/000001.png", reference)
        save_image(args.output / parent / "blur/000001.png", ordinary)
    save_image(args.output / gopro / "blur_gamma/000001.png", gamma)
    with (args.output / "pairs.csv").open("x", encoding="utf-8") as file:
        file.write("dataset,id,blur,sharp,condition\n")
        for condition in ("blur", "blur_gamma"):
            file.write(f"gopro,fixture/000001.png,{gopro.as_posix()}/{condition}/000001.png,"
                       f"{gopro.as_posix()}/sharp/000001.png,{condition}\n")
        file.write(f"realblur-j,fixture/000001.png,{realblur.as_posix()}/blur/000001.png,"
                   f"{realblur.as_posix()}/sharp/000001.png,\n")


if __name__ == "__main__":
    main()
