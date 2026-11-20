"""정답·추정·오차 PSF로 Wiener K값을 비교한다."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
from matplotlib.figure import Figure

from .analysis import analyze_image
from .batch_report import save_figure
from .classical import apply_motion_blur, compare_quality, linear_psf, wiener_restore
from .decision import ASSESSMENT_MAX_SIZE
from .image_io import Image, limit_image_size, load_image, save_image

K_VALUES = (0.001, 0.005, 0.01, 0.02, 0.05)


def run_sweep(reference: Image, output: Path, *, angle: float, length: int) -> None:
    """단일 합성 사례의 복원 결과, 절대 차 영상, 수치와 K 그래프를 저장한다."""
    true_psf = linear_psf(length, angle)
    observed = apply_motion_blur(reference, true_psf)
    analysis = analyze_image(limit_image_size(observed, ASSESSMENT_MAX_SIZE))
    direction = analysis.motion_direction.angle_deg
    scenarios = (
        ("known", angle, length),
        ("estimated", direction, length),
        ("angle_error", angle + 15, length),
        ("length_error", angle, length + 4),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_image(output / "reference.png", reference)
    save_image(output / "blurred.png", observed)
    rows = []
    for method, psf_angle, psf_length in scenarios:
        psf = None if psf_angle is None else linear_psf(psf_length, psf_angle)
        for k in K_VALUES:
            row = dict(method=method, actual_angle_deg=angle % 180, actual_length_px=length,
                       applied_angle_deg=None if psf_angle is None else psf_angle % 180,
                       applied_length_px=psf_length, k=k, psnr_db=None, ssim=None, elapsed_ms=None,
                       status="skipped_no_direction")
            if psf is not None:
                start = perf_counter()
                restored = wiener_restore(observed, psf, k)
                elapsed = (perf_counter() - start) * 1000
                psnr, ssim = compare_quality(reference, restored)
                row.update(psnr_db=psnr, ssim=ssim, elapsed_ms=elapsed, status="ok")
                save_image(output / f"{method}-{k:g}.png", restored)
                save_image(output / f"{method}-{k:g}-absdiff.png", cv2.absdiff(reference, restored))
            rows.append(row)

    with (output / "wiener_k.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = Figure(figsize=(12, 4), layout="constrained")
    for ax, metric in zip(fig.subplots(1, 3), ("psnr_db", "ssim", "elapsed_ms")):
        for method, _, _ in scenarios:
            points = [r for r in rows if r["method"] == method and r["status"] == "ok"]
            ax.plot([r["k"] for r in points], [r[metric] for r in points], "o-", label=f"{method} (n={len(points)})")
        ax.set(xscale="log", xlabel="K", title=metric)
    fig.axes[0].legend()
    fig.suptitle(f"{reference.shape[1]}×{reference.shape[0]} | {angle % 180:g}° | {length}px")
    save_figure(fig, output / "wiener_k.png", dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wiener K 및 PSF 오차 비교")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--angle", type=float, default=45)
    parser.add_argument("--length", type=int, default=15)
    args = parser.parse_args()
    run_sweep(load_image(args.input), args.output, angle=args.angle, length=args.length)


if __name__ == "__main__":
    main()
