"""알려진 합성 PSF의 추정 오차·복원 수치를 CSV로 비교한다."""

import argparse
import csv
from pathlib import Path
import sys

from motion_blur.classical import compare_quality, linear_psf, richardson_lucy_restore, synthesize_motion_blur, wiener_restore
from motion_blur.image_io import load_image
from motion_blur.psf_estimator import estimate_psf
from tests.validate_motion_direction import axial_error


def main() -> None:
    parser = argparse.ArgumentParser(description="공개 영상의 PSF 추정 비교")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(("image", "noise_sigma", "actual_angle", "actual_length", "method", "analysis_width",
                     "estimated_angle", "angle_error", "direction_score", "estimated_length", "length_error",
                     "length_prominence", "status", "wiener_psnr", "wiener_ssim", "rl_psnr", "rl_ssim",
                     "blurred_psnr", "blurred_ssim"))
    for path in args.images:
        source = load_image(path)
        for angle, length in ((0, 9), (30, 15), (45, 21), (90, 21), (135, 35)):
            true_psf = linear_psf(length, angle)
            for sigma in (0, 0.01):
                observed = synthesize_motion_blur(source, true_psf, noise_sigma=sigma, seed=702)
                baseline = compare_quality(source, observed)
                for method in ("known", "fft", "gradient"):
                    if method == "known":
                        candidate_angle, candidate_length = angle, length
                        direction_score = prominence = None
                        width = status = None
                    else:
                        result = estimate_psf(observed, method=method)
                        candidate_angle = result.angle_deg
                        candidate_length = (round(result.length_original_px)
                                            if result.length_original_px is not None else None)
                        direction_score = getattr(result, method).confidence
                        prominence = result.length_prominence
                        width, status = result.analysis_size[0], result.reason
                    metrics = (None,) * 4
                    if candidate_angle is not None and candidate_length is not None:
                        psf = linear_psf(candidate_length, candidate_angle)
                        metrics = (*compare_quality(source, wiener_restore(observed, psf)),
                                   *compare_quality(source, richardson_lucy_restore(observed, psf)))
                    writer.writerow((
                        path.name, sigma, angle, length, method, width,
                        candidate_angle, axial_error(angle, candidate_angle) if candidate_angle is not None else None,
                        direction_score, candidate_length,
                        abs(length - candidate_length) if candidate_length is not None else None,
                        prominence, status or "ok", *metrics, *baseline,
                    ))


if __name__ == "__main__":
    main()
