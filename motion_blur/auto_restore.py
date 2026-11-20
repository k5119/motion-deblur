"""추정 또는 수동 PSF 하나로 Wiener와 RL을 실행한다."""

import argparse
from dataclasses import asdict
import logging
from pathlib import Path

import numpy as np

from .classical import BOUNDARIES, DEFAULT_RL_ITERATIONS, DEFAULT_WIENER_K, linear_psf, richardson_lucy_restore, wiener_restore
from .experiment import save_json
from .image_io import load_image, save_image
from .psf_estimator import estimate_psf

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """자동 후보가 부족하면 이유를 기록하고 필요한 수동 입력을 요청한다."""
    parser = argparse.ArgumentParser(description="자동 PSF 후보와 수동 조정으로 Wiener·RL 복원")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="새 결과 디렉터리")
    parser.add_argument("--direction", choices=("fft", "gradient"), default="fft")
    parser.add_argument("--angle", type=float, help="복원 영상 좌표계의 수동 각도")
    parser.add_argument("--length", type=int, help="복원 영상 픽셀 단위의 수동 길이")
    parser.add_argument("--k", type=float, default=DEFAULT_WIENER_K)
    parser.add_argument("--iterations", type=int, default=DEFAULT_RL_ITERATIONS)
    parser.add_argument("--boundary", choices=BOUNDARIES, default="taper")
    args = parser.parse_args(argv)
    if args.length is not None and args.length < 1:
        parser.error("수동 길이는 1 이상이어야 합니다.")
    if not np.isfinite(args.k) or args.k <= 0 or args.iterations < 1:
        parser.error("K는 유한한 양수이고 RL 반복 횟수는 양의 정수여야 합니다.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.output.exists():
            raise FileExistsError(f"새 결과 디렉터리를 지정하세요. ({args.output})")
        image = load_image(args.input)
        estimate = estimate_psf(image, method=args.direction, angle_override=args.angle)
        angle = args.angle % 180 if args.angle is not None else estimate.angle_deg
        length = args.length
        if length is None and estimate.length_original_px is not None:
            length = round(estimate.length_original_px)
        LOGGER.info("복원 크기 %s / 분석 크기 %s", estimate.original_size, estimate.analysis_size)
        LOGGER.info("FFT 후보 %s (%.3f) / 그래디언트 후보 %s (%.3f)",
                    estimate.fft.angle_deg, estimate.fft.confidence,
                    estimate.gradient.angle_deg, estimate.gradient.confidence)
        report = {
            "estimate": asdict(estimate),
            "applied": {"angle_deg": angle, "length_px": length,
                        "manual_angle": args.angle is not None, "manual_length": args.length is not None,
                        "k": args.k, "iterations": args.iterations, "boundary": args.boundary},
            "status": "skipped",
        }
        args.output.mkdir(parents=True, exist_ok=False)
        if angle is None or length is None:
            LOGGER.warning("자동 후보 부족 (%s). --angle 또는 --length를 지정하세요.", estimate.reason)
            save_json(args.output / "psf_estimate.json", report)
            return 1

        # 두 복원에서 같은 PSF 배열을 사용한다.
        psf = linear_psf(length, angle)
        LOGGER.info("적용 PSF 후보 %.2f° / %dpx", angle, length)
        save_image(args.output / "wiener.png", wiener_restore(image, psf, args.k, boundary=args.boundary))
        save_image(args.output / "rl.png", richardson_lucy_restore(image, psf, args.iterations, boundary=args.boundary))
        report["status"] = "ok"
        save_json(args.output / "psf_estimate.json", report)
        return 0
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        LOGGER.error("처리 실패 — %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
