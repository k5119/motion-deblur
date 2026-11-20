"""같은 관측 입력으로 NAFNet 원본 크기와 축소 실행을 비교한다."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import torch

from . import DEFAULT_WEIGHTS
from .classical import apply_motion_blur, compare_quality, linear_psf
from .image_io import limit_image_size, load_image, save_image
from .restoration import load_model, restore_image, select_device


def main() -> int:
    parser = argparse.ArgumentParser(description="NAFNet 크기별 복원 및 CUDA 메모리 점검")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--real", action="store_true", help="참조가 없는 흐린 촬영 영상")
    parser.add_argument("--max-size", type=int, default=384, help="축소 비교 실행의 긴 변")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    if args.max_size < 1:
        parser.error("축소 비교 크기는 양수여야 합니다.")

    source = load_image(args.input)
    height, width = source.shape[:2]
    reference = None if args.real else source
    observed = source if args.real else apply_motion_blur(source, linear_psf(15, 45))
    device = select_device(args.device)
    model = load_model(args.weights, device)
    args.output.mkdir(parents=True, exist_ok=False)
    save_image(args.output / "observed.png", observed)
    if reference is not None:
        save_image(args.output / "reference.png", reference)

    rows = []
    for name, sample in (("original", observed), ("limited", limit_image_size(observed, args.max_size))):
        row = dict(mode=name, device=str(device), original_width=width, original_height=height,
                   inference_width=sample.shape[1], inference_height=sample.shape[0],
                   elapsed_ms=None, peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None,
                   psnr_db=None, ssim=None, status="ok", detail=None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        try:
            restored = restore_image(sample, model)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row["elapsed_ms"] = (perf_counter() - started) * 1000
        except torch.OutOfMemoryError as exc:
            row.update(status="out_of_memory", detail=str(exc))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            save_image(args.output / f"{name}.png", restored)
            if restored.shape != observed.shape:
                restored = cv2.resize(restored, (width, height), interpolation=cv2.INTER_CUBIC)
                save_image(args.output / f"{name}_aligned.png", restored)
            if reference is not None:
                row["psnr_db"], row["ssim"] = compare_quality(reference, restored)
        if device.type == "cuda":
            row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            row["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        rows.append(row)
    with (args.output / "nafnet.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
