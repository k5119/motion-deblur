"""합성 모션 블러, Wiener 복원과 NAFNet 비교를 실행하는 CLI."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Literal

import cv2
import numpy as np

from . import DEFAULT_WEIGHTS
from .classical import (
    BOUNDARIES, DEFAULT_NOISE_SEED, DEFAULT_RL_ITERATIONS, DEFAULT_WIENER_K,
    Boundary, compare_quality, linear_psf, richardson_lucy_restore, synthesize_motion_blur, wiener_restore,
)
from .image_io import Image, limit_image_size, load_image, save_image
from .motion_direction import MotionDirection, estimate_motion_direction

LOGGER = logging.getLogger(__name__)
PSF_MODEL = "linear_bilinear_reflect101_v1"


def save_json(path: Path, data: dict) -> None:
    """JSON을 새 파일로 저장한다. 기존 파일은 배타적 생성으로 보호한다."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_psf(path: Path, image: Image) -> tuple[int, float]:
    """합성 시 저장한 JSON에서 이 이미지 크기에 맞는 PSF 파라미터를 읽는다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("model") != PSF_MODEL:
        raise ValueError("지원하는 선형 PSF 메타데이터 형식이 아닙니다.")
    if (data.get("height"), data.get("width")) != image.shape[:2]:
        raise ValueError("PSF 메타데이터와 입력 영상의 해상도가 다릅니다.")
    length, angle = data.get("length_px"), data.get("angle_deg")
    if type(length) is not int or type(angle) not in (int, float):
        raise ValueError("PSF 메타데이터에 유효한 길이와 각도가 필요합니다.")
    return length, float(angle)


def estimated_wiener(
    image: Image, length: int, k: float, *, direction: MotionDirection | None = None,
    boundary: Boundary = "taper",
) -> tuple[Image | None, dict]:
    """FFT 방향 후보와 사용자가 지정한 길이로 Wiener 복원을 시도한다."""
    if direction is None:
        direction = estimate_motion_direction(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    info = {"length_px": length, "angle_deg": direction.angle_deg, "direction_score": direction.confidence}
    if direction.angle_deg is None:
        LOGGER.warning("추정 PSF 복원 생략 — FFT 방향 추정 불가 (점수 %.3f).", direction.confidence)
        return None, info
    LOGGER.info("추정 PSF — 각도 %.2f°, 지정 길이 %dpx, 점수 %.3f", direction.angle_deg, length, direction.confidence)
    return wiener_restore(image, linear_psf(length, direction.angle_deg), k, boundary=boundary), info


def nafnet_restore(image: Image, weights: Path, device_choice: Literal["auto", "cpu", "cuda"]) -> Image:
    """기존 NAFNet 추론을 호출해 입력과 동일한 크기의 이미지를 반환한다."""
    from .restoration import load_model, restore_image, select_device

    device = select_device(device_choice)
    LOGGER.info("NAFNet 비교 장치 — %s", device)
    return restore_image(image, load_model(weights, device))


def run_synthetic(args: argparse.Namespace, reference: Image) -> None:
    """합성 블러를 저장하고 비교 모드에서는 동일 크기로 모든 지표를 계산한다."""
    if args.output_dir.exists():
        raise FileExistsError(f"새 출력 폴더를 지정하세요. ({args.output_dir})")
    if args.max_size:
        reference = limit_image_size(reference, args.max_size)

    psf = linear_psf(args.length, args.angle)
    blurred = synthesize_motion_blur(reference, psf, noise_sigma=args.noise_sigma, seed=args.seed)
    height, width = reference.shape[:2]
    metadata = {
        "model": PSF_MODEL,
        "length_px": args.length,
        "angle_deg": args.angle % 180,
        "height": height,
        "width": width,
        "noise_sigma": args.noise_sigma,
        "seed": args.seed,
    }

    images = {"blurred": blurred}
    if args.command == "compare":
        images["reference"] = reference
        LOGGER.info("정답 PSF — 각도 %.2f°, 길이 %dpx, K=%g", metadata["angle_deg"], args.length, args.k)
        images["wiener_known"] = wiener_restore(blurred, psf, args.k)
        estimated, estimation = estimated_wiener(blurred, args.estimated_length, args.k)
        if estimated is not None:
            images["wiener_estimated"] = estimated
        images["nafnet"] = nafnet_restore(blurred, args.weights, args.device)

        report = {
            "width": width, "height": height, "wiener_k": args.k,
            "estimated_psf": {**estimation, "status": "ok" if estimated is not None else "skipped_no_direction"},
            "metrics": {},
        }
        for name, image in images.items():
            if name == "reference":
                continue
            psnr, ssim = compare_quality(reference, image)
            LOGGER.info("%s — PSNR %.3f dB, SSIM %.4f", name, psnr, ssim)
            report["metrics"][name] = {"psnr_db": psnr if np.isfinite(psnr) else "inf", "ssim": ssim}

    if args.command == "core-compare":
        images["reference"] = reference
        report = {"psf": metadata, "wiener_k": args.k, "metrics": {}}
        for boundary in BOUNDARIES:
            for count in (None, *dict.fromkeys(args.iterations)):
                started = perf_counter()
                if count is None:
                    method = "wiener"
                    image = wiener_restore(blurred, psf, args.k, boundary=boundary)
                else:
                    method = f"rl_{count}"
                    image = richardson_lucy_restore(blurred, psf, count, boundary=boundary)
                elapsed_ms = (perf_counter() - started) * 1000
                name = f"{method}_{boundary}"
                images[name] = image
                psnr, ssim = compare_quality(reference, image)
                report["metrics"][name] = {
                    "psnr_db": psnr if np.isfinite(psnr) else "inf", "ssim": ssim, "elapsed_ms": elapsed_ms,
                }
        psnr, ssim = compare_quality(reference, blurred)
        report["metrics"]["blurred"] = {"psnr_db": psnr if np.isfinite(psnr) else "inf", "ssim": ssim}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, image in images.items():
        save_image(args.output_dir / f"{name}.png", image)
    save_json(args.output_dir / "psf.json", metadata)
    if args.command in {"compare", "core-compare"}:
        save_json(args.output_dir / "metrics.json", report)
    LOGGER.info("저장 완료 — %s (%d × %d)", args.output_dir, width, height)


def main(argv: list[str] | None = None) -> int:
    """실험 CLI 인자를 처리하고 성공 0, 처리 실패 또는 추정 불가 1을 반환한다."""
    parser = argparse.ArgumentParser(description="합성 모션 블러와 복원 비교 실험")
    commands = parser.add_subparsers(dest="command", required=True)
    synthesize = commands.add_parser("synthesize", help="선명한 원본에서 블러와 PSF JSON 생성")
    compare = commands.add_parser("compare", help="선명한 원본에서 합성 후 Wiener·NAFNet 비교")
    core_compare = commands.add_parser("core-compare", help="CPU에서 경계와 RL 반복 횟수 비교")
    core_compare.add_argument("--iterations", type=int, nargs="+", default=[10, 20, 40])
    for command in (synthesize, compare, core_compare):
        command.add_argument("reference", type=Path, help="선명한 원본 이미지")
        command.add_argument("output_dir", type=Path, help="새 출력 폴더")
        command.add_argument("--angle", type=float, required=True, help="오른쪽 수평 0°, 시계 방향")
        command.add_argument("--length", type=int, required=True, help="합성 PSF 길이(px)")
        command.add_argument("--max-size", type=int, help="합성 전에 적용할 긴 변 최대 크기")
        command.add_argument("--noise-sigma", type=float, default=0.0, help="0~1 픽셀 범위 기준 잡음 표준편차")
        command.add_argument("--seed", type=int, default=DEFAULT_NOISE_SEED)
    compare.add_argument("--estimated-length", type=int, required=True, help="추정 PSF에 사용할 길이(px)")
    compare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    compare.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)

    wiener = commands.add_parser("wiener", help="블러 이미지를 CPU에서 Wiener 복원")
    rl = commands.add_parser("rl", help="블러 이미지를 CPU에서 Richardson–Lucy 복원")
    rl.add_argument("--iterations", type=int, default=DEFAULT_RL_ITERATIONS)
    for command in (wiener, rl):
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path, help="새 출력 PNG")
        command.add_argument("--boundary", choices=BOUNDARIES, default="taper")
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--psf", type=Path, help="합성 시 저장한 정답 PSF JSON")
        source.add_argument("--angle", type=float, help="수동 PSF 각도")
        if command is wiener:
            source.add_argument("--estimate", action="store_true", help="FFT 방향 후보 사용")
        command.add_argument("--length", type=int, help="수동 또는 추정 PSF 길이(px)")
    for command in (compare, core_compare, wiener):
        command.add_argument("--k", type=float, default=DEFAULT_WIENER_K, help="양의 Wiener 정규화 상수")
    args = parser.parse_args(argv)
    if args.command in {"wiener", "rl"} and (args.psf is None) != (args.length is not None):
        parser.error("--length는 --estimate 또는 --angle과 함께 지정해야 합니다.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command in {"compare", "core-compare", "wiener"} and (not np.isfinite(args.k) or args.k <= 0):
            raise ValueError("Wiener K는 0보다 큰 유한한 값이어야 합니다.")
        if args.command == "compare" and args.estimated_length < 1:
            raise ValueError("추정 PSF 길이는 1 이상의 정수여야 합니다.")
        if args.command == "core-compare" and min(args.iterations) < 1:
            raise ValueError("RL 반복 횟수는 1 이상이어야 합니다.")
        if args.command not in {"wiener", "rl"}:
            run_synthetic(args, load_image(args.reference))
            return 0

        if args.output.suffix.lower() != ".png":
            raise ValueError("출력 파일 확장자는 .png여야 합니다.")
        if args.output.exists():
            raise FileExistsError(f"새 출력 파일을 지정하세요. ({args.output})")
        image = load_image(args.input)
        if getattr(args, "estimate", False):
            if args.length < 1:
                raise ValueError("추정 PSF 길이는 1 이상의 정수여야 합니다.")
            restored, _ = estimated_wiener(image, args.length, args.k, boundary=args.boundary)
            if restored is None:
                return 1
        else:
            length, angle = read_psf(args.psf, image) if args.psf else (args.length, args.angle)
            LOGGER.info("%s PSF — 각도 %.2f°, 길이 %dpx", "정답" if args.psf else "수동", angle, length)
            psf = linear_psf(length, angle)
            if args.command == "rl":
                restored = richardson_lucy_restore(image, psf, args.iterations, boundary=args.boundary)
            else:
                restored = wiener_restore(image, psf, args.k, boundary=args.boundary)
        LOGGER.info("%s 저장 완료 — %s (%s)", args.command, save_image(args.output, restored), args.boundary)
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError, cv2.error) as exc:
        LOGGER.error("처리 실패 — %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
